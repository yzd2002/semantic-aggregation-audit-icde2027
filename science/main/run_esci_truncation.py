#!/usr/bin/env python3
"""Run fixed short, long and replay pointwise judgments with GPU ownership checks."""
import argparse
import collections
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import time
from pathlib import Path

import psutil
import run_experiment as base

PROTOCOL='PROTOCOL_esci_truncation_sensitivity.md'
SUFFIX='\nOutput one character: Y or N.'
BATCH_SIZE=256


def sha(text):return hashlib.sha256(text.encode('utf8')).hexdigest()


def gpu_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return {int(x.strip()) for x in text.splitlines() if x.strip().isdigit()}


def assert_own_gpu():
    proc=psutil.Process(os.getpid());ours={proc.pid}|{x.pid for x in proc.children(recursive=True)}
    if gpu_processes()-ours:raise RuntimeError('Another GPU process appeared; stop our run without affecting it')


def parse(text):return text if text in ('Y','N') else None


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--preflight',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.model.name!='Qwen3-4B-Instruct-2507' or os.environ.get('VLLM_USE_V1')!='1':
        raise ValueError('Expected recorded Qwen model and vLLM V1')
    packages=('vllm','torch','transformers','numpy','xgrammar','outlines-core','setuptools')
    versions={x:metadata.version(x) for x in packages}
    expected={'vllm':'0.10.2','torch':'2.8.0+cu128','transformers':'4.57.1','numpy':'2.2.6',
              'xgrammar':'0.1.23','outlines-core':'0.2.11','setuptools':'79.0.1'}
    if versions!=expected:raise RuntimeError('Dependency overlay differs from protocol: '+str(versions))
    p=a.selection/'inputs.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
    selection=json.loads((a.selection/'selection.json').read_text());pre=json.loads(a.preflight.read_text())
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if (selection['protocol']!=PROTOCOL or selection['inputs_sha256']!=digest
        or pre['selection_inputs_sha256']!=digest or not pre['all_within_limit']
        or pre['model']!=a.model.name or len(rows)!=250 or sum(len(r['items']) for r in rows)!=5160):
        raise ValueError('Selection or preflight mismatch')
    if gpu_processes():raise RuntimeError('GPU already in use; no model loaded')
    stat=os.statvfs(a.out.parent);free=stat.f_bavail*stat.f_frsize
    if free<200*1024*1024:raise RuntimeError('Insufficient free disk space for the trace and logs')
    a.out.mkdir(parents=True,exist_ok=False)
    summary={'protocol':PROTOCOL,'model':a.model.name,'selection_inputs_sha256':digest,'versions':versions,
             'runtime':{'engine':'V1','dtype':'bfloat16','max_model_len':4096,'max_num_seqs':16,
                        'gpu_memory_utilization':.70,'enforce_eager':True,'prefix_caching':True,
                        'temperature':0,'request_batch_size':BATCH_SIZE},
             'system':base.SYSTEM,'retry_suffix':SUFFIX,'mode_order':['short','long','replay'],'modes':{}}
    dump(a.out/'summary.json',summary)
    import torch
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm=LLM(model=str(a.model),tokenizer=str(a.model),trust_remote_code=False,dtype='bfloat16',
            max_model_len=4096,gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,
            enable_prefix_caching=True,disable_log_stats=True)
    tok=llm.get_tokenizer();summary['runtime']['gpu']=torch.cuda.get_device_name(0);assert_own_gpu()
    def wrap(system,text,tokenize=False):
        return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],
                                       tokenize=tokenize,add_generation_prompt=True)
    params=SamplingParams(temperature=0,max_tokens=4,guided_decoding=GuidedDecodingParams(regex='[YN]'))
    control=wrap('Follow the user instruction exactly.','Reply with only the word BANANA.')
    controls={}
    for mode,setting in [('ordinary',SamplingParams(temperature=0,max_tokens=128)),('guided',params)]:
        assert_own_gpu();r=llm.generate([control],setting,use_tqdm=False)[0];o=r.outputs[0]
        controls[mode]={'text':o.text,'finish_reason':o.finish_reason,'input_tokens':len(r.prompt_token_ids),
                        'output_tokens':len(o.token_ids)}
    controls['checks']={'ordinary_invalid':parse(controls['ordinary']['text']) is None,
                        'guided_valid':parse(controls['guided']['text']) is not None}
    dump(a.out/'guidance_sanity.json',controls)
    if not all(controls['checks'].values()):raise RuntimeError('Guidance negative control failed; no candidate inference')
    for mode,field in [('short','short'),('long','long'),('replay','short')]:
        req=[]
        for row in rows:
            for item in row['items']:
                prompt=base.p_point(row,{'text':item[field]})
                req.append({'query_id':row['query_id'],'item_id':item['id'],'product_id':item['product_id'],
                            'text_changed':item['text_changed'],'prompt':prompt})
        records=[];start_time=time.perf_counter()
        with (a.out/(mode+'_responses.jsonl')).open('x',encoding='utf8') as output:
            for offset in range(0,len(req),BATCH_SIZE):
                batch=req[offset:offset+BATCH_SIZE];assert_own_gpu()
                first=llm.generate([wrap(base.SYSTEM,x['prompt']) for x in batch],params,use_tqdm=False)
                bad=[i for i,r in enumerate(first) if parse(r.outputs[0].text) is None]
                assert_own_gpu()
                retry=llm.generate([wrap(base.SYSTEM,batch[i]['prompt']+SUFFIX) for i in bad],params,use_tqdm=False) if bad else []
                retries=dict(zip(bad,retry))
                for i,(request,response) in enumerate(zip(batch,first)):
                    o=response.outputs[0];retried=retries.get(i);last=retried.outputs[0] if retried else o
                    rec={k:request[k] for k in ('query_id','item_id','product_id','text_changed')}
                    rec.update({'prompt_sha256':sha(request['prompt']),
                        'wrapped_sha256':sha(wrap(base.SYSTEM,request['prompt'])),
                        'first':o.text,'final':last.text,'first_finish_reason':o.finish_reason,
                        'retry_finish_reason':retried.outputs[0].finish_reason if retried else None,
                        'first_input_tokens':len(response.prompt_token_ids),'first_output_tokens':len(o.token_ids),
                        'retry_input_tokens':len(retried.prompt_token_ids) if retried else 0,
                        'retry_output_tokens':len(retried.outputs[0].token_ids) if retried else 0})
                    records.append(rec);output.write(json.dumps(rec,ensure_ascii=False)+'\n')
                output.flush()
                if offset%(BATCH_SIZE*4)==0:print(mode,'completed',min(offset+BATCH_SIZE,len(req)),'/',len(req),flush=True)
        failed_pools=set(r['query_id'] for r in records if parse(r['final']) is None)
        stats={'initial_calls':len(records),'retry_calls':sum(r['retry_finish_reason'] is not None for r in records),
               'valid_first':sum(parse(r['first']) is not None for r in records),
               'valid_final':sum(parse(r['final']) is not None for r in records),
               'complete_pools':len(rows)-len(failed_pools),
               'first_finish_reasons':dict(collections.Counter(r['first_finish_reason'] for r in records)),
               'logical_tokens':{k:sum(r['first_'+k+'_tokens']+r['retry_'+k+'_tokens'] for r in records)
                                 for k in ('input','output')},
               'max_retry_prompt_plus_cap':pre['max_retry_prompt_plus_cap'][field],
               'offline_generation_seconds':time.perf_counter()-start_time}
        summary['modes'][mode]=stats;dump(a.out/'summary.json',summary)
        print(mode,json.dumps(stats),flush=True)
    print('Completed ESCI truncation sensitivity execution.',flush=True)


if __name__=='__main__':main()
