#!/usr/bin/env python3
"""Post-restart ACORD paired keyed plans; persist every completed mode."""
import argparse,collections,hashlib,importlib.metadata as md,json,os,subprocess,time
from pathlib import Path
import psutil
from batch_prompts import SYSTEM,RETRY_SUFFIX,prompt,regex,parse

EXPECTED_POOLS=114
MODES=(('eight_original',8,False),('four_original',4,False),('four_stable_reverse',4,True),('four_replay',4,False),('eight_replay',8,False))
WANTED={'vllm':'0.10.2','torch':'2.8.0+cu128','transformers':'4.57.1','numpy':'2.2.6',
        'xgrammar':'0.1.25','outlines-core':'0.1.26','setuptools':'80.9.0','tokenizers':'0.22.0'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def memory_guard(initial=False):
    stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    available=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
    available+=stat.get('active_file',0)+stat.get('inactive_file',0)
    if available<(9 if initial else 1.5)*1024**3:raise RuntimeError('Insufficient memory reserve; stop this run')
    if initial:
        for name,minimum in (('/dev/shm',10*1024**3),('/',32*1024**2)):
            x=os.statvfs(name)
            if x.f_bavail*x.f_frsize<minimum:raise RuntimeError('Insufficient free space on '+name)

def gpu_processes():
    x=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return {int(s.strip()) for s in x.splitlines() if s.strip().isdigit()}

def assert_own_gpu():
    p=psutil.Process(os.getpid());ours={p.pid}|{x.pid for x in p.children(recursive=True)}
    if gpu_processes()-ours:raise RuntimeError('Another GPU process appeared; stop this run without stopping it')

def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    for n in ('model','selection','baseline','preflight','protocol','weights_manifest','manifest','out'):
        p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('VLLM_USE_V1')!='1' or a.model.name!='Qwen3-4B-Instruct-2507':raise ValueError('Unexpected model or engine mode')
    versions={n:md.version(n) for n in WANTED}
    if versions!=WANTED:raise RuntimeError('Environment differs from the frozen recovery protocol')
    manifest=json.loads(a.manifest.read_text());meta=json.loads((a.selection/'selection.json').read_text())
    pre=json.loads((a.preflight/'summary.json').read_text());base=json.loads((a.baseline/'summary.json').read_text());data=a.selection/'inputs.jsonl'
    if (sha(a.protocol)!=manifest['protocol_sha256'] or sha(data)!=manifest['selection_sha256'] or sha(data)!=meta['inputs_sha256']
        or sha(a.baseline/'summary.json')!=manifest['baseline_sha256'] or not base.get('completed') or base['selection_inputs_sha256']!=sha(data)
        or sha(a.preflight/'summary.json')!=manifest['preflight_sha256'] or pre['selection_sha256']!=sha(data)
        or pre['protocol_sha256']!=sha(a.protocol) or pre['prompt_records_sha256']!=sha(a.preflight/'prompts.jsonl')
        or pre['prompt_code_sha256']!=sha(Path(__file__).with_name('batch_prompts.py')) or pre['base_prompt_code_sha256']!=sha(Path(__file__).with_name('acord_prompts.py'))
        or pre['overflow_pool_count'] or pre['pool_count']!=EXPECTED_POOLS or pre['prompt_count']!=EXPECTED_POOLS*24
        or sha(Path(__file__))!=manifest['runner_sha256'] or sha(a.weights_manifest)!=manifest['weights_manifest_sha256']):
        raise ValueError('Frozen code, input, preflight or baseline changed')
    for name,expected in pre['tokenizer_assets_sha256'].items():
        if sha(a.model/name)!=expected:raise ValueError('Tokenizer assets differ')
    rows=[json.loads(x) for x in data.read_text(encoding='utf8').splitlines() if x.strip()]
    if len(rows)!=EXPECTED_POOLS or any(len(r['items'])!=32 for r in rows):raise ValueError('Unexpected pools')
    weights={}
    for line in a.weights_manifest.read_text().splitlines():
        h,path=line.split(None,1);name=Path(path.strip()).name
        if name in weights:raise ValueError('Duplicate weight')
        weights[name]=h
    if weights!=base['weights_sha256'] or set(weights)!={f'model-0000{i}-of-00003.safetensors' for i in (1,2,3)}:raise ValueError('Prior weight manifest mismatch')
    memory_guard(initial=True)
    if gpu_processes():raise RuntimeError('GPU is in use; no model loaded')
    for name,expected in weights.items():
        memory_guard()
        if sha(a.model/name)!=expected:raise ValueError('Model weight hash mismatch')
    a.out.mkdir(parents=True,exist_ok=False)
    summary={'scope':'Post-hoc recovery after earlier exposed result and lost traces; same new unsupported environment for both sizes',
             'selection_sha256':sha(data),'baseline_sha256':sha(a.baseline/'summary.json'),'protocol_sha256':sha(a.protocol),
             'preflight_sha256':sha(a.preflight/'summary.json'),'manifest_sha256':sha(a.manifest),'versions':versions,'weights_sha256':weights,
             'runtime':{'dtype':'bfloat16','max_model_len':4096,'max_num_seqs':16,'gpu_memory_utilization':.7,'enforce_eager':True,'prefix_caching':True},
             'mode_order':[x[0] for x in MODES],'modes':{}}
    dump(a.out/'summary.json',summary)
    import torch
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm=LLM(model=str(a.model),tokenizer=str(a.model),trust_remote_code=False,dtype='bfloat16',max_model_len=4096,
            gpu_memory_utilization=.7,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=True,disable_log_stats=True)
    tok=llm.get_tokenizer();summary['runtime']['gpu']=torch.cuda.get_device_name(0);assert_own_gpu()
    def wrap(text,system=SYSTEM,tokenize=False):
        return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],tokenize=tokenize,add_generation_prompt=True)
    settings={n:SamplingParams(temperature=0,max_tokens=128,guided_decoding=GuidedDecodingParams(regex=regex(n))) for n in (4,8)}
    control=wrap('Reply with only the word BANANA.',system='Follow the user instruction exactly.');tests={}
    for name,params in (('ordinary',SamplingParams(temperature=0,max_tokens=128)),('four',settings[4]),('eight',settings[8])):
        assert_own_gpu();response=llm.generate([control],params,use_tqdm=False)[0];o=response.outputs[0]
        tests[name]={'text':o.text,'finish_reason':o.finish_reason,'input_tokens':len(response.prompt_token_ids),'output_tokens':len(o.token_ids)}
    tests['checks']={'ordinary_invalid_for_4':parse(tests['ordinary']['text'],4) is None,'ordinary_invalid_for_8':parse(tests['ordinary']['text'],8) is None,
                     'guided_4_valid':parse(tests['four']['text'],4) is not None,'guided_8_valid':parse(tests['eight']['text'],8) is not None}
    dump(a.out/'guidance_sanity.json',tests)
    if not all(tests['checks'].values()):raise RuntimeError('Guidance negative control failed; no candidate inference occurred')
    for mode,size,reverse in MODES:
        requests=[]
        for row in rows:
            for start in range(0,32,size):
                text,ids=prompt(row,start,size,reverse);requests.append({'pool_id':row['pool_id'],'start':start,'size':size,'presented_ids':ids,'prompt':text})
        worst=max(len(wrap(x['prompt']+RETRY_SUFFIX,tokenize=True))+128 for x in requests)
        if worst>4096:raise RuntimeError('Prompt exceeds context')
        memory_guard();assert_own_gpu();start_time=time.perf_counter()
        first=llm.generate([wrap(x['prompt']) for x in requests],settings[size],use_tqdm=False)
        if len(first)!=len(requests):raise RuntimeError('Unexpected number of responses')
        bad=[i for i,x in enumerate(first) if parse(x.outputs[0].text,size) is None]
        assert_own_gpu();retry=llm.generate([wrap(requests[i]['prompt']+RETRY_SUFFIX) for i in bad],settings[size],use_tqdm=False) if bad else []
        if len(retry)!=len(bad):raise RuntimeError('Unexpected number of retries')
        retry_map=dict(zip(bad,retry));records=[]
        for i,(req,response) in enumerate(zip(requests,first)):
            out=response.outputs[0];rr=retry_map.get(i);last=rr.outputs[0] if rr else out
            records.append({'pool_id':req['pool_id'],'start':req['start'],'size':size,'presented_ids':req['presented_ids'],
                'prompt_sha256':hashlib.sha256(req['prompt'].encode()).hexdigest(),'first':out.text,'final':last.text,
                'first_finish_reason':out.finish_reason,'retry_finish_reason':rr.outputs[0].finish_reason if rr else None,
                'first_input_tokens':len(response.prompt_token_ids),'first_output_tokens':len(out.token_ids),
                'retry_input_tokens':len(rr.prompt_token_ids) if rr else 0,'retry_output_tokens':len(rr.outputs[0].token_ids) if rr else 0})
        dest=a.out/(mode+'_responses.jsonl')
        with dest.open('x',encoding='utf8',newline='\n') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())
        stats={'initial_calls':len(requests),'retry_calls':len(bad),'valid_first':sum(parse(r['first'],size) is not None for r in records),
               'valid_final':sum(parse(r['final'],size) is not None for r in records),
               'first_finish_reasons':dict(collections.Counter(r['first_finish_reason'] for r in records)),
               'logical_tokens':{k:sum(r['first_'+k+'_tokens']+r['retry_'+k+'_tokens'] for r in records) for k in ('input','output')},
               'max_prompt_with_retry_plus_cap':worst,'offline_generation_seconds':time.perf_counter()-start_time,'responses_sha256':sha(dest)}
        summary['modes'][mode]=stats;dump(a.out/'summary.json',summary);print(mode,json.dumps(stats),flush=True)
    summary['completed']=True;dump(a.out/'summary.json',summary);print('Completed all recovery modes.',flush=True)

if __name__=='__main__':main()
