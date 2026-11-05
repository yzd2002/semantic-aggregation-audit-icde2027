#!/usr/bin/env python3
"""One-model owned-GPU runner; no retries and no raw-string trimming."""
import argparse,collections,json,os,subprocess,time,traceback,sys,platform,importlib.metadata as package_metadata
from common import *

def gpu_processes():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    return [{'pid':int(line.split(',')[0]),'record':line} for line in value.splitlines() if line.strip()]
def check_owned():
    import psutil
    owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)};running=gpu_processes()
    if any(p['pid'] not in owned for p in running):raise RuntimeError('Unowned GPU process detected; refusing further requests: '+str(running))
    return running

def main(model):
    amendment=json.loads((ROOT/'RUNNER_V2_FROZEN_SHA256.json').read_text())
    for rel,want in amendment['files'].items():
        if sha(ROOT/rel)!=want:raise ValueError('Runner amendment file differs: '+rel)
    out=ROOT/f'run_{model}';out.mkdir(exist_ok=False)
    try:
        frozen=json.loads((ROOT/'prepared/EXECUTION_FROZEN_SHA256.json').read_text())
        for rel,want in frozen['files'].items():
            if sha(ROOT/rel)!=want:raise ValueError('Frozen file differs: '+rel)
        cfg=json.loads((ROOT/'prepared/runtime_configuration.json').read_text())
        current={p:package_metadata.version(p) for p in cfg['packages']}
        if current!=cfg['packages'] or platform.python_version()!=cfg['python'] or sys.executable!=cfg['python_executable']:
            raise RuntimeError('Runtime differs from frozen preparation')
        for path,want in json.loads((ROOT/'prepared/MODEL_FILE_SHA256.json').read_text())[model].items():
            if sha(path)!=want:raise ValueError('Model file changed: '+path)
        if gpu_processes():raise RuntimeError('GPU is not idle before loading model')
        jobs=read(ROOT/'prepared'/f'{model}_requests.jsonl')
        write_json(out/'START.json',{'pid':os.getpid(),'started_unix':time.time(),'model':str(MODELS[model]),'planned_jobs':len(jobs),
            'frozen_sha256':sha(ROOT/'prepared/EXECUTION_FROZEN_SHA256.json'),'request_sha256':sha(ROOT/'prepared'/f'{model}_requests.jsonl')})
        import torch,vllm
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        llm=LLM(model=str(MODELS[model]),tokenizer=str(MODELS[model]),trust_remote_code=False,dtype='bfloat16',
            max_model_len=4096,gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,
            enable_prefix_caching=False,disable_log_stats=True)
        tok=llm.get_tokenizer()
        def wrap(system,user):
            value=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
            return value,{'prompt_token_ids':tok.encode(value,add_special_tokens=False)}
        def params(rule):return SamplingParams(temperature=0,max_tokens=128,
            guided_decoding=GuidedDecodingParams(regex=rule,backend='xgrammar',disable_fallback=True))
        # Validate parser's handling of numeric order, duplicate keys and raw whitespace.
        assert parse('{"9":"Y","10":"N"}',[9,10])=={9:'Y',10:'N'}
        assert parse('{"10":"N","9":"Y"}',[9,10]) is None
        assert parse('{"10":"N","9":"Y"}',[9,10],True)=={10:'N',9:'Y'}
        assert parse(' {"11":"Y"}',[11]) is None and parse('{"11":"Y","11":"N"}',[11]) is None
        text,control=wrap('Follow the user instruction exactly.','Reply with only the word BANANA.')
        ordinary=llm.generate([control],SamplingParams(temperature=0,max_tokens=128),use_tqdm=False)[0]
        controls=[{'ids':[2,9,10,11],'descending':False},{'ids':[2,9,10,11],'descending':True},
            {'ids':[11],'descending':False},{'ids':[32],'descending':True}]
        output=llm.generate([control]*len(controls),[params(grammar(x['ids'],x['descending'])) for x in controls],use_tqdm=False)
        evidence={'ordinary_raw':ordinary.outputs[0].text,'wrapped_prompt':text,'checks':[]}
        for spec,response in zip(controls,output):
            raw=response.outputs[0].text;valid=parse(raw,spec['ids'],spec['descending']) is not None
            evidence['checks'].append({**spec,'raw':raw,'grammar':grammar(spec['ids'],spec['descending']),'valid':valid,
                'finish_reason':response.outputs[0].finish_reason,'input_tokens':len(response.prompt_token_ids),'output_tokens':len(response.outputs[0].token_ids)})
        evidence['valid']=all(x['valid'] for x in evidence['checks']) and all(parse(evidence['ordinary_raw'],x['ids'],x['descending']) is None for x in controls)
        write_json(out/'negative_controls.json',evidence)
        if not evidence['valid']:raise RuntimeError('Guidance negative controls failed; data inference not started')
        synthetic={'query':'white ceramic cup','items':[{'id':9,'text':'Name: white ceramic cup'},{'id':10,'text':'Name: black steel bucket'}]}
        requests=[];metadata=[];parameters=[]
        for cond in CONDITIONS:
            items=synthetic['items'] if cond[0]=='A' else list(reversed(synthetic['items']))
            _,request=wrap(SYSTEMS['wands'],prompt(synthetic,items));requests.append(request)
            metadata.append({'condition':cond,'ids':[9,10]});parameters.append(params(grammar([9,10],cond[-1]=='D')))
        for i in synthetic['items']:
            _,request=wrap(SYSTEMS['wands'],prompt(synthetic,[i]));requests.append(request)
            metadata.append({'condition':'S','ids':[i['id']]});parameters.append(params(grammar([i['id']])))
        smoke=llm.generate(requests,parameters,use_tqdm=False);smokerecord=[]
        for meta,response in zip(metadata,smoke):
            raw=response.outputs[0].text;smokerecord.append({**meta,'raw':raw,'valid':parse(raw,meta['ids'],meta['condition'].endswith('D')) is not None})
        write_json(out/'synthetic_smoke.json',{'records':smokerecord,'valid':all(x['valid'] for x in smokerecord),'semantic_labels_not_used_as_gate':True})
        if not all(x['valid'] for x in smokerecord):raise RuntimeError('Synthetic smoke failed; data inference not started')
        write_json(out/'DATA_START.json',{'time_unix':time.time(),'owned_gpu_processes':check_owned(),'request_count':len(jobs),
            'negative_control_sha256':sha(out/'negative_controls.json'),'smoke_sha256':sha(out/'synthetic_smoke.json')})
        count=valid=0;start=time.perf_counter();totals=collections.Counter()
        with (out/'responses.jsonl').open('x',encoding='utf8') as records,(out/'blocks.jsonl').open('x',encoding='utf8') as blocks:
            for offset in range(0,len(jobs),64):
                owned=check_owned();batch=jobs[offset:offset+64];t=time.perf_counter()
                responses=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in batch],[params(j['grammar']) for j in batch],use_tqdm=False)
                seconds=time.perf_counter()-t
                if len(responses)!=len(batch):raise RuntimeError('Response count changed')
                for job,response in zip(batch,responses):
                    answer=response.outputs[0];raw=answer.text;parsed=parse(raw,job['ids'],job['condition'].endswith('D'))
                    if list(response.prompt_token_ids)!=job['prompt_token_ids']:raise RuntimeError('Actual input tokens differ from preflight')
                    meta={k:job[k] for k in ('request_id','ordinal','block','dataset','query_id','start','size','condition','repeat','ids','chunk_ordinal','prompt_sha256','grammar_sha256','id_map_sha256')}
                    rec={**meta,'engine_request_id':response.request_id,'raw':raw,'parsed':parsed,'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,
                        'actual_input_tokens':len(response.prompt_token_ids),'actual_output_tokens':len(answer.token_ids),'output_token_ids':list(answer.token_ids),
                        'input_token_ids_sha256':digest(json.dumps(list(response.prompt_token_ids),separators=(',',':')))}
                    records.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n');count+=1;valid+=parsed is not None
                    totals['input_tokens']+=len(response.prompt_token_ids);totals['output_tokens']+=len(answer.token_ids)
                records.flush();block={'block':offset//64,'requests':len(batch),'first_ordinal':offset,'last_ordinal':offset+len(batch)-1,'seconds':seconds,'gpu_processes':owned,'completed_unix':time.time()}
                blocks.write(json.dumps(block)+'\n');blocks.flush()
                print(model,'block',offset//64,'jobs',count,'/',len(jobs),'valid',valid,'elapsed',round(time.perf_counter()-start,2),flush=True)
        write_json(out/'summary.json',{'model':model,'selected_pools':180,'jobs':count,'valid_jobs':valid,'invalid_jobs':count-valid,
            'actual_tokens':dict(totals),'mixed_workload_seconds':time.perf_counter()-start,'completed_unix':time.time(),
            'response_sha256':sha(out/'responses.jsonl'),'block_sha256':sha(out/'blocks.jsonl'),'status':'complete'})
    except BaseException as e:
        write_json(out/'FAILURE.json',{'time_unix':time.time(),'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);main(p.parse_args().model)
