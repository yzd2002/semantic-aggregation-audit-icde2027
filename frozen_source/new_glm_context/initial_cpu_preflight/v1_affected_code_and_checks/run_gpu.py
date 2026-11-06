#!/usr/bin/env python3
"""Frozen native-GLM runner. Every scientific attempt/raw response is retained."""
import collections
import argparse
import importlib.metadata as metadata
import importlib.util
import os
import platform
import subprocess
import sys
import time
import traceback
from common import *

DEVICE=None


def append(path,row):
    with path.open('a',encoding='utf8') as stream:
        stream.write(compact(row)+'\n');stream.flush()


def gpu_processes():
    value=subprocess.check_output(['nvidia-smi','--id='+str(DEVICE),'--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
        '--format=csv,noheader,nounits'],text=True)
    return [{'pid':int(line.split(',')[0]),'record':line} for line in value.splitlines() if line.strip()]


def check_owned():
    import psutil
    owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
    rows=gpu_processes()
    if any(r['pid'] not in owned for r in rows):
        raise RuntimeError('A foreign process is using the assigned GPU; no further request is submitted: '+str(rows))
    return rows


def worker_identity(worker):
    import sys
    model=worker.model_runner.model
    params=list(model.parameters())
    return {'worker_pid':os.getpid(),'model_class':type(model).__name__,'model_module':type(model).__module__,
        'parameter_dtypes':sorted({str(p.dtype) for p in params}),'parameters':sum(p.numel() for p in params),
        'remote_model_modules_loaded':[name for name in sys.modules if 'modeling_chatglm' in name or 'configuration_chatglm' in name]}


def main():
    global DEVICE
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--device',type=int,required=True);args=parser.parse_args();model=args.model;DEVICE=args.device
    out=ROOT/f'run_{model}'
    out.mkdir(exist_ok=False)
    llm=None
    try:
        authorization_path=ROOT/f'GPU_GO_{model}.json'
        authorization=json.loads(authorization_path.read_text())
        if authorization.get('authorized_by')!='root' or authorization.get('device')!=DEVICE or authorization.get('model')!=model or authorization.get('freeze_sha256')!=sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json'):
            raise RuntimeError('Missing explicit root authorization for this exact frozen run')
        frozen=json.loads((ROOT/'PREINFERENCE_FROZEN_SHA256.json').read_text())
        for relative,want in frozen['files'].items():
            if sha(ROOT/relative)!=want:raise RuntimeError('Frozen scientific input/code changed: '+relative)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(DEVICE):raise RuntimeError('GPU visibility differs from root authorization')
        cfg=json.loads((ROOT/'prepared'/f'runtime_configuration_{model}.json').read_text())
        versions={p:metadata.version(p) for p in cfg['packages']}
        if versions!=cfg['packages'] or platform.python_version()!=cfg['python'] or sys.executable!=cfg['python_executable']:
            raise RuntimeError('Runtime differs from current CPU preparation')
        installed_vllm=Path(importlib.util.find_spec('vllm').origin).parent
        for relative,want in cfg['vllm_source_sha256'].items():
            if sha(installed_vllm/relative)!=want:raise RuntimeError('Inspected vLLM source changed: '+relative)
        model_entry=json.loads((ROOT/'prepared/MODEL_FILE_SHA256.json').read_text())[model]
        model_path=Path(model_entry['path'])
        for relative,want in model_entry['files'].items():
            if sha(model_path/relative)!=want:raise RuntimeError('Verified fixed-revision model file changed: '+relative)
        import single_rank_nccl_compat as adapter
        if not adapter.ACTIVE or Path(adapter.__file__).resolve()!=ROOT/'runtime_override/single_rank_nccl_compat.py':
            raise RuntimeError('Reviewed process-local adapter is not active')
        if gpu_processes():raise RuntimeError('Assigned GPU is occupied; refusing to load')
        envelope_path=ROOT/'prepared'/f'REQUEST_ENVELOPE_{model}.json'
        envelope=json.loads(envelope_path.read_text())
        write_json(out/'START.json',{'pid':os.getpid(),'started_unix':time.time(),'model':model,'model_path':str(model_path),
            'model_revision':model_entry.get('revision'),'frozen_manifest_sha256':sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json'),
            'authorization_sha256':sha(authorization_path),'request_envelope_sha256':sha(envelope_path),
            'planned_unique_calls':envelope['unique_all_calls'],'runtime_versions':versions,
            'launch_environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TMPDIR','XDG_CACHE_HOME','HF_HOME',
                'HF_MODULES_CACHE','TORCH_HOME','VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT','NUMBA_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR',
                'TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','FLASHINFER_WORKSPACE_BASE','PYTHONPATH',
                'INTENTSAT_SINGLE_RANK_NCCL_COMPAT','VLLM_USE_V1','VLLM_NO_USAGE_STATS','DO_NOT_TRACK','PYTHONDONTWRITEBYTECODE')}})
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        llm=LLM(**cfg['llm_kwargs'])
        def parameters(rule=None):
            return SamplingParams(**cfg['requested_sampling'],guided_decoding=None if rule is None else
                GuidedDecodingParams(regex=rule,backend='xgrammar',disable_fallback=True))
        native=llm.collective_rpc(worker_identity)
        expected=cfg['expected_native']
        if len(native)!=1 or native[0]['model_class']!=expected['class'] or native[0]['model_module']!=expected['module'] or native[0]['remote_model_modules_loaded'] or native[0]['parameter_dtypes']!=['torch.bfloat16']:
            raise RuntimeError('Actual loaded worker is not the required native BF16 ChatGLMForCausalLM: '+str(native))
        engine=llm.llm_engine
        config=engine.vllm_config
        model_config=engine.model_config
        tokenizer=llm.get_tokenizer()
        remote_modules=[name for name in sys.modules if 'modeling_chatglm' in name or 'configuration_chatglm' in name]
        if remote_modules:raise RuntimeError('Remote model/configuration module was loaded: '+str(remote_modules))
        actual={'created_unix':time.time(),'worker_identity':native,
            'engine_class':type(engine).__module__+'.'+type(engine).__name__,
            'native_hf_config_class':type(model_config.hf_config).__module__+'.'+type(model_config.hf_config).__name__,
            'model_impl':getattr(model_config.model_impl,'value',model_config.model_impl),'dtype':str(model_config.dtype),
            'max_model_len':model_config.max_model_len,'max_num_seqs':config.scheduler_config.max_num_seqs,
            'tensor_parallel_size':config.parallel_config.tensor_parallel_size,'enforce_eager':model_config.enforce_eager,
            'prefix_caching':config.cache_config.enable_prefix_caching,'gpu_memory_utilization':config.cache_config.gpu_memory_utilization,
            'tokenizer_class':type(tokenizer).__name__,'primary_eos_token_id':tokenizer.eos_token_id,
            'generation_config_fields':engine.processor.generation_config_fields,
            'guided_decoding_backend':config.decoding_config.backend,
            'guided_decoding_disable_fallback':config.decoding_config.disable_fallback,
            'remote_model_modules_loaded':remote_modules}
        assert actual['model_impl']=='vllm' and actual['max_model_len']==4096 and actual['max_num_seqs']==16
        assert actual['tensor_parallel_size']==1 and actual['enforce_eager'] and not actual['prefix_caching']
        assert actual['primary_eos_token_id']==cfg['primary_eos_token_id'] and actual['guided_decoding_backend']=='xgrammar' and actual['guided_decoding_disable_fallback']
        write_json(out/'EFFECTIVE_NATIVE_RUNTIME.json',actual)
        original_process=engine.processor.process_inputs
        def audited_process(*args,**kwargs):
            result=original_process(*args,**kwargs)
            request=result[1];p=request.sampling_params
            if p is not None:
                controls={k:getattr(p,k) for k in ('n','temperature','top_k','top_p','min_p','max_tokens','ignore_eos',
                    'repetition_penalty','presence_penalty','frequency_penalty','seed','skip_special_tokens','spaces_between_special_tokens')}
                controls.update(stop_token_ids=sorted(p.stop_token_ids),all_stop_token_ids=sorted(p.all_stop_token_ids),
                    primary_eos_token_id=request.eos_token_id,sampling_type=p.sampling_type.name,
                    guided_backend=p.guided_decoding.backend if p.guided_decoding else None,
                    guided_disable_fallback=p.guided_decoding.disable_fallback if p.guided_decoding else None,
                    grammar_sha256=digest(p.guided_decoding.regex) if p.guided_decoding else None)
                append(out/'effective_requests.jsonl',{'engine_request_id':request.request_id,'time_unix':time.time(),
                    'prompt_token_ids_sha256':digest(compact(request.prompt_token_ids)),'actual_processed_controls':controls})
                assert (p.temperature,p.top_p,p.top_k,p.max_tokens,p.n)==(0,1,0,128,1)
                assert sorted(p.all_stop_token_ids)==cfg['stop_token_ids'] and request.eos_token_id==cfg['primary_eos_token_id']
                assert p.sampling_type.name=='GREEDY' and not p.ignore_eos
                if p.guided_decoding:
                    assert p.guided_decoding.backend=='xgrammar' and p.guided_decoding.disable_fallback
            return result
        engine.processor.process_inputs=audited_process
        def collect(job,response,diagnostic=False):
            if list(response.prompt_token_ids)!=job['prompt_token_ids']:raise RuntimeError('Engine prompt tokens changed: '+job['request_id'])
            answer=response.outputs[0];raw=answer.text
            parsed=parse(raw,job['output_ids']) if job.get('output_ids') else None
            keys=('request_id','ordinal','block','task','target_id','selected_ordinal','scope','context','repeat','requested_ids',
                  'output_ids','prompt_sha256','grammar_sha256','id_map_sha256','execution_mode','source_main_request_id','source_historical_request_id')
            return {**{k:job[k] for k in keys if k in job},'engine_request_id':response.request_id,'raw':raw,'raw_sha256':digest(raw),
                'strict_valid':parsed is not None,'target_label':int(parsed[1]=='Y') if parsed is not None and 1 in parsed else None,
                'parsed':parsed,'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,
                'actual_input_tokens':len(response.prompt_token_ids),'actual_output_tokens':len(answer.token_ids),
                'output_token_ids':list(answer.token_ids),'input_token_ids_sha256':digest(compact(list(response.prompt_token_ids))),
                'diagnostic':diagnostic}
        diagnostic_jobs=read_lines(ROOT/'prepared'/f'{model}_diagnostic_requests.jsonl')
        diagnostics=[]
        for name in ('ordinary_banana','guided_banana','synthetic_format_smoke'):
            batch=[j for j in diagnostic_jobs if j['diagnostic_group']==name]
            check_owned();append(out/'diagnostic_submissions.jsonl',{'name':name,'request_ids':[j['request_id'] for j in batch],
                'submitted_unix':time.time()})
            responses=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in batch],
                [parameters(j.get('grammar') if j['guided'] else None) for j in batch],use_tqdm=False)
            if len(responses)!=len(batch):raise RuntimeError('Diagnostic count mismatch')
            current=[]
            for job,response in zip(batch,responses):
                row={**collect(job,response,True),'wrapped_prompt':job['wrapped_prompt'],'grammar':job.get('grammar')}
                append(out/'diagnostics.jsonl',row);diagnostics.append(row);current.append(row)
            if name=='guided_banana':
                ordinary=diagnostics[0]
                okay='BANANA' in ordinary['raw'] and all(r['strict_valid'] for r in current) and all(parse(ordinary['raw'],j['output_ids']) is None for j in batch)
                write_json(out/'negative_controls.json',{'valid':okay,'ordinary':ordinary,'guided':current,'parser_checks':parser_checks()})
                if not okay:raise RuntimeError('Incompatible-instruction grammar controls failed before scientific inference')
            if name=='synthetic_format_smoke':
                okay=all(r['strict_valid'] for r in current)
                write_json(out/'synthetic_smoke.json',{'valid':okay,'records':current,'semantic_labels_not_used_as_gate':True})
                if not okay:raise RuntimeError('Synthetic format checks failed before scientific inference')
        write_json(out/'DATA_START.json',{'time_unix':time.time(),'diagnostic_calls':len(diagnostics),
            'physical_device':DEVICE,'owned_gpu_processes':check_owned()})
        summaries={}
        destinations={'mixed':out,'historical-extra-mixed':out/'historical_extra','serial':out/'serial',
            'historical-extra-serial':out/'historical_extra_serial'}
        for mode in envelope['execution_order']:
            destination=destinations[mode]
            if destination!=out:destination.mkdir(exist_ok=False)
            entry=envelope['modes'][mode]
            jobs=read_lines(ROOT/entry['file'])
            write_json(destination/('MODE_START.json' if mode=='mixed' else 'START.json'),{'time_unix':time.time(),'mode':mode,
                'planned_calls':len(jobs),'same_loaded_engine':True,'execution_order':envelope['execution_order'],
                'request_sha256':sha(ROOT/entry['file']),'prior_mode_summary_sha256':{m:sha(destinations[m]/'summary.json') for m in summaries}})
            count=valid=0;totals=collections.Counter();start=time.perf_counter()
            width=entry['requests_per_submission']
            with (destination/'responses.jsonl').open('x',encoding='utf8') as sink:
                for offset in range(0,len(jobs),width):
                    batch=jobs[offset:offset+width];owned=check_owned();tick=time.perf_counter()
                    assert len({j['block'] for j in batch})==1
                    append(destination/'block_submissions.jsonl',{'block':batch[0]['block'],'first_ordinal':batch[0]['ordinal'],
                        'last_ordinal':batch[-1]['ordinal'],'request_ids':[j['request_id'] for j in batch],'submitted_unix':time.time()})
                    values=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in batch],
                        [parameters(j['grammar']) for j in batch],use_tqdm=False)
                    if len(values)!=len(batch):raise RuntimeError('Scientific response count mismatch in '+mode)
                    for job,response in zip(batch,values):
                        row=collect(job,response);sink.write(compact(row)+'\n');count+=1;valid+=row['strict_valid']
                        totals['input_tokens']+=row['actual_input_tokens'];totals['output_tokens']+=row['actual_output_tokens']
                    sink.flush()
                    append(destination/'blocks.jsonl',{'block':batch[0]['block'],'requests':len(batch),'seconds':time.perf_counter()-tick,
                        'completed_unix':time.time(),'owned_gpu_processes':owned})
                    if 'serial' not in mode or count%32==0 or count==len(jobs):
                        print(mode,'returned',count,'/',len(jobs),'valid',valid,'elapsed',round(time.perf_counter()-start,2),flush=True)
            summary={'status':'complete','model':model,'mode':mode,'scientific_calls':count,'valid_scientific_calls':valid,
                'invalid_scientific_calls':count-valid,'observed_tokens':dict(totals),'execution_seconds':time.perf_counter()-start,
                'completed_unix':time.time(),'response_sha256':sha(destination/'responses.jsonl'),'block_sha256':sha(destination/'blocks.jsonl')}
            write_json(destination/'summary.json',summary);summaries[mode]=summary
        all_tokens={k:sum(s['observed_tokens'][k] for s in summaries.values())+sum(r['actual_'+k] for r in diagnostics)
                    for k in ('input_tokens','output_tokens')}
        write_json(out/'ALL_WORK_COMPLETE.json',{'status':'complete','unique_scientific_calls':sum(s['scientific_calls'] for s in summaries.values()),
            'diagnostic_calls':len(diagnostics),'all_calls':sum(s['scientific_calls'] for s in summaries.values())+len(diagnostics),
            'modes':summaries,'all_observed_tokens':all_tokens,'completed_unix':time.time()})
    except BaseException as error:
        write_json(out/'FAILURE.json',{'time_unix':time.time(),'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':main()
