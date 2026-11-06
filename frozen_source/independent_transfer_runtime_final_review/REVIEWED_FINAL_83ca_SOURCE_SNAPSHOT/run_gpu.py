#!/usr/bin/env python3
"""New immutable execution; selected GPU only, controls before science, no retries."""
import argparse,collections,importlib.metadata as metadata,os,platform,subprocess,sys,time,traceback
from common import *
from gpu_gate import verify
from return_journal import generate_preserving_returns
from runtime_observer import RuntimeObserver,LAUNCH_ENV_KEYS,launch_environment

def append(path,r):
    with path.open('a',encoding='utf8') as f:f.write(compact(r)+'\n');f.flush()

def processes(gpu):
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    return [{'pid':int(line.split(',')[0]),'record':line} for line in raw.splitlines() if line.strip()]

def owned(gpu):
    import psutil
    allowed={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)};running=processes(gpu)
    if any(r['pid'] not in allowed for r in running):raise RuntimeError('Selected GPU has another task; stop submissions: '+str(running))
    return running

def collect_record(j,response,diagnostic=False):
    if list(response.prompt_token_ids)!=j['prompt_token_ids']:raise RuntimeError('Actual prompt differs from preflight: '+j['request_id'])
    if len(response.outputs)!=1:raise RuntimeError('Unexpected number of engine output sequences: '+j['request_id'])
    answer=response.outputs[0];parsed=parse(answer.text,j['output_ids']) if j['output_ids'] else None
    meta={k:v for k,v in j.items() if k not in ('system','unwrapped_user','wrapped_prompt','prompt_token_ids','grammar','input_ids_in_order','id_map','prompt_tokens_preflight','canonical_output_token_lengths_allY_allN')}
    return {**meta,'engine_request_id':response.request_id,'raw':answer.text,'strict_valid':parsed is not None,'labels':parsed,
        'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'actual_input_tokens':len(response.prompt_token_ids),
        'actual_output_tokens':len(answer.token_ids),'output_token_ids':list(answer.token_ids),
        'input_token_ids_sha256':digest(compact(list(response.prompt_token_ids))),'diagnostic':diagnostic}

def main(model,gpu):
    # Both direct invocation and launcher invocation require the exact release.
    release=verify(model,gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError('Selected GPU visibility differs from release')
    out=ROOT/f'run_{model}';out.mkdir(exist_ok=False)
    observer=None
    try:
        import single_rank_nccl_compat as compat
        assert compat.ACTIVE and Path(compat.__file__).resolve()==ROOT/'runtime_override/single_rank_nccl_compat.py'
        config=json.loads((ROOT/'prepared/runtime_configuration.json').read_text())
        versions={p:metadata.version(p) for p in config['packages']}
        assert versions==config['packages'] and platform.python_version()==config['python'] and sys.executable==config['python_executable']
        contract_path=ROOT/'prepared/runtime_observation_contract.json'
        contract=json.loads(contract_path.read_text())
        assert sha(ROOT/'runtime_observer.py')==contract['runtime_observer_sha256']
        for key,want in launch_environment(ROOT,model).items():assert os.environ.get(key)==want,(key,os.environ.get(key),want)
        installed=Path(__import__('importlib.util',fromlist=['find_spec']).find_spec('vllm').origin).parent
        for name,want in contract['vllm_source_sha256'].items():assert sha(installed/name)==want,name
        loc=json.loads((ROOT/'model_locations.json').read_text())[model]
        manifest=json.loads((ROOT/'prepared/MODEL_FILE_SHA256.json').read_text())[model]
        for name,want in manifest['files'].items():assert sha(Path(loc)/name)==want
        if processes(gpu):raise RuntimeError('Selected GPU not idle before loading')
        jobs=read_lines(ROOT/'prepared'/f'{model}_requests.jsonl');controljobs=read_lines(ROOT/'prepared'/f'{model}_controls.jsonl')
        write_json(out/'START.json',{'started_unix':time.time(),'pid':os.getpid(),'model':model,'gpu':gpu,
            'new_real_replication':True,'prior_aggregate_results_known':True,'planned_scientific_calls':len(jobs),
            'freeze_sha256':sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json'),'request_sha256':sha(ROOT/'prepared'/f'{model}_requests.jsonl'),
            'GPU_release':release,'runtime_versions':versions,'runtime_cache_root':str(ROOT/'runtime_cache'/model),
            'runtime_observation_contract_sha256':sha(contract_path),
            'launch_environment':{k:os.environ.get(k) for k in LAUNCH_ENV_KEYS}})
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        llm=LLM(model=loc,tokenizer=loc,trust_remote_code=False,dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=4096,gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=False,
            disable_log_stats=True,guided_decoding_backend='xgrammar',guided_decoding_disable_fallback=True)
        def params(j):return SamplingParams(temperature=0,max_tokens=128,guided_decoding=None if j.get('guided') is False else GuidedDecodingParams(regex=j['grammar'],backend='xgrammar',disable_fallback=True))
        observer=RuntimeObserver(llm,out,model,contract)
        diagnostics=[]
        for name in ('ordinary_banana','guided_banana','synthetic_format_smoke'):
            owned(gpu);batch=[j for j in controljobs if j['name']==name]
            append(out/'diagnostic_submissions.jsonl',{'name':name,'request_ids':[j['request_id'] for j in batch],'submitted_unix':time.time()})
            observer.begin(batch,'diagnostic/'+name)
            values=generate_preserving_returns(llm,batch,[params(j) for j in batch],out/'engine_returns.jsonl','diagnostic/'+name)
            observer.bind_returns(batch,values,out/'engine_returns.jsonl')
            for j,v in zip(batch,values):
                r=collect_record(j,v,True);r.update(wrapped_prompt=j['wrapped_prompt'],grammar=j.get('grammar'));append(out/'diagnostics.jsonl',r);diagnostics.append(r)
        ordinary=diagnostics[0];good='BANANA' in ordinary['raw'] and not ordinary['strict_valid'] and all(r['strict_valid'] for r in diagnostics[1:])
        write_json(out/'CONTROL_RESULT.json',{'status':'PASS' if good else 'FAIL','diagnostic_calls':len(diagnostics),
            'diagnostic_sha256':sha(out/'diagnostics.jsonl'),'semantic_labels_not_a_gate':True,'parser_checks':parser_checks()})
        if not good:raise RuntimeError('Structured controls failed; no scientific calls submitted')
        write_json(out/'DATA_START.json',{'time_unix':time.time(),'model':model,'control_sha256':sha(out/'CONTROL_RESULT.json'),'owned_gpu_processes':owned(gpu)})
        count=valid=0;totals=collections.Counter();started=time.perf_counter()
        with (out/'responses.jsonl').open('x',encoding='utf8') as f:
            for offset in range(0,len(jobs),64):
                active=owned(gpu);batch=jobs[offset:offset+64];t=time.perf_counter()
                append(out/'block_submissions.jsonl',{'block':offset//64,'first_ordinal':offset,'last_ordinal':offset+len(batch)-1,
                    'request_ids':[j['request_id'] for j in batch],'submitted_unix':time.time()})
                observer.begin(batch,'science',offset//64)
                values=generate_preserving_returns(llm,batch,[params(j) for j in batch],out/'engine_returns.jsonl','science',offset//64)
                observer.bind_returns(batch,values,out/'engine_returns.jsonl')
                for j,v in zip(batch,values):
                    row=collect_record(j,v);f.write(compact(row)+'\n');count+=1;valid+=row['strict_valid']
                    totals['input_tokens']+=row['actual_input_tokens'];totals['output_tokens']+=row['actual_output_tokens']
                f.flush();append(out/'blocks.jsonl',{'block':offset//64,'requests':len(batch),'seconds':time.perf_counter()-t,'completed_unix':time.time(),'owned_gpu_processes':active})
                print(model,'block',offset//64,'returned',count,'/',len(jobs),'valid',valid,flush=True)
        control_tokens={key:sum(r['actual_'+key] for r in diagnostics) for key in ('input_tokens','output_tokens')}
        write_json(out/'summary.json',{'status':'complete','model':model,'scientific_calls':count,'valid_scientific_calls':valid,
            'invalid_scientific_calls':count-valid,'diagnostic_calls':20,'all_calls':count+20,
            'scientific_observed_tokens':dict(totals),'diagnostic_observed_tokens':control_tokens,
            'all_observed_tokens':{key:totals[key]+control_tokens[key] for key in control_tokens},
            'interleaved_seconds':time.perf_counter()-started,'completed_unix':time.time(),'response_sha256':sha(out/'responses.jsonl'),
            'engine_return_journal_sha256':sha(out/'engine_returns.jsonl'),
            'effective_runtime_observation_receipts':observer.receipts(),
            'blocks_sha256':sha(out/'blocks.jsonl'),'GPU_release':release})
        from verify_effective_runtime import verify as verify_effective
        write_json(out/'EFFECTIVE_RUNTIME_VERIFICATION.json',verify_effective(ROOT,model))
        # This is a fresh raw execution. Analysis may also be replayed independently on CPU.
        from analyze_transfer import analyze
        analyze(model,ROOT/f'analysis_{model}')
    except BaseException as error:
        write_json(out/'FAILURE.json',{'model':model,'gpu':gpu,'time_unix':time.time(),'type':type(error).__name__,
            'message':str(error),'traceback':traceback.format_exc(),'automatic_retry':False,
            'engine_return_journal_sha256':sha(out/'engine_returns.jsonl') if (out/'engine_returns.jsonl').exists() else None,
            'all_actual_returns_before_validation_file':'engine_returns.jsonl' if (out/'engine_returns.jsonl').exists() else None,
            'effective_runtime_observation_receipts':observer.receipts() if observer else None})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--gpu',required=True);a=p.parse_args();main(a.model,a.gpu)
