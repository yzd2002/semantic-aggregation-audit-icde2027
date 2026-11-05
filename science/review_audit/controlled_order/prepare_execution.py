#!/usr/bin/env python3
"""Freeze both models, exact prompts, grammars, token IDs and inherited runtime."""
import importlib.metadata as metadata,json,platform,subprocess,time,sys
from common import *

def main():
    frozen=json.loads((ROOT/'selection/PREINFERENCE_INPUT_CODE_SHA256.json').read_text())
    for rel,want in frozen['files'].items():
        if sha(ROOT/rel)!=want:raise ValueError('Input/code changed after selection freeze: '+rel)
    out=ROOT/'prepared';out.mkdir(exist_ok=False)
    checks=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (out/'pip_check.log').write_text(checks.stdout+checks.stderr)
    cfg={'created_unix':time.time(),'environment_status':'Inherited same-machine runtime with known version conflicts; not a fresh-machine reproduction.',
        'python':platform.python_version(),'python_executable':sys.executable,'packages':{p:metadata.version(p) for p in ['torch','vllm','transformers','numpy','xgrammar','outlines_core','setuptools','pandas','pyarrow']},
        'pip_check_exit':checks.returncode,'guided_backend':'xgrammar','disable_guidance_fallback':True,
        'max_model_len':4096,'max_tokens':128,'max_num_seqs':16,'block_requests':64,'dtype':'bfloat16',
        'temperature':0,'enforce_eager':True,'prefix_caching':False,'gpu_memory_utilization':0.70,
        'token_input_rule':'apply_chat_template to string, encode with add_special_tokens=False, pass explicit token IDs to vLLM',
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)}
    write_json(out/'runtime_configuration.json',cfg)
    from transformers import AutoTokenizer
    inputs={d:read(ROOT/'selection'/f'{d}_inputs.jsonl') for d in ('esci','wands')}
    bykey={(d,r['query_id']):r for d,rows in inputs.items() for r in rows};jobs=read(ROOT/'selection/schedule.jsonl')
    models={};preflight={}
    for model,path in MODELS.items():
        print('Hashing model',model,flush=True);models[model]={str(p):sha(p) for p in sorted(path.iterdir()) if p.is_file()}
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        prepared=[];maximum=0
        for job in jobs:
            row=bykey[job['dataset'],job['query_id']];items=[row['items'][i-1] for i in job['ids']]
            if job['condition'].startswith('R'):items=list(reversed(items))
            user=prompt(row,items);messages=[{'role':'system','content':SYSTEMS[job['dataset']]},{'role':'user','content':user}]
            wrapped=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            tokenids=tok.encode(wrapped,add_special_tokens=False);maximum=max(maximum,len(tokenids)+128)
            if len(tokenids)+128>4096:raise RuntimeError(f'{model} context overflow at {job["request_id"]}; no data inference permitted')
            desc=job['condition'].endswith('D');rule=grammar(job['ids'],desc)
            mapping={'presented_ids':[i['id'] for i in items],'output_ids':sorted(job['ids'],reverse=desc),
                'occurrences':{str(i['id']):{'pool_index':i['id']-1,'product_id':i.get('product_id'),'example_id':i.get('example_id')} for i in items}}
            prepared.append({**job,'user_prompt':user,'wrapped_prompt':wrapped,'prompt_token_ids':tokenids,
                'prompt_tokens_preflight':len(tokenids),'prompt_sha256':digest(wrapped),'user_prompt_sha256':digest(user),
                'grammar':rule,'grammar_sha256':digest(rule),'id_map':mapping,
                'id_map_sha256':digest(json.dumps(mapping,sort_keys=True,separators=(',',':')))})
        write_lines(out/f'{model}_requests.jsonl',prepared)
        preflight[model]={'requests':len(prepared),'max_prompt_plus_cap':maximum,'request_file_sha256':sha(out/f'{model}_requests.jsonl')}
        print(model,'preflight',preflight[model],flush=True)
    write_json(out/'MODEL_FILE_SHA256.json',models);write_json(out/'BOTH_MODEL_CONTEXT_PREFLIGHT.json',preflight)
    files=[*ROOT.glob('*.py'),ROOT/'PROTOCOL_controlled_order.md',ROOT/'PRESELECTION_CLARIFICATIONS.md',*out.glob('*'),* (ROOT/'selection').glob('*')]
    write_json(out/'EXECUTION_FROZEN_SHA256.json',{'created_unix':time.time(),'files':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    print('Both models, complete schedule, prompts and runtime frozen before GPU judgments.',flush=True)
if __name__=='__main__':main()
