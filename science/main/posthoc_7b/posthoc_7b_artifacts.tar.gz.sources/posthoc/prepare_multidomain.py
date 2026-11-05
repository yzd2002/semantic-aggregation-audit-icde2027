#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
base=Path('/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility')
out=Path('/dev/shm/intentsat-7b-followup-20261031')
protocol='PROTOCOL_7b_multidomain_posthoc.md'
def substitute(s,old,new):
 if s.count(old)!=1:raise RuntimeError('Expected exactly one occurrence: '+old[:80])
 return s.replace(old,new)
def write_new(name,s):
 p=out/name
 with p.open('x',encoding='utf8') as f:f.write(s)
wrapper='''\n    def guarded_generate(prompts, params, use_tqdm=False):\n        results = []\n        for start in range(0, len(prompts), 64):\n            assert_own_gpu()\n            chosen = params[start:start+64] if isinstance(params, list) else params\n            results.extend(llm.generate(prompts[start:start+64], chosen, use_tqdm=use_tqdm))\n        return results\n'''
s=(base/'run_selective_model_sensitivity.py').read_text()
s=substitute(s,"PROTOCOL = 'PROTOCOL_selective_model_sensitivity.md'",f"PROTOCOL = '{protocol}'")
s=substitute(s,"from select_selective_execution import N, PROTOCOL as SELECTION_PROTOCOL, lines", "from select_selective_execution import N, PROTOCOL as SELECTION_PROTOCOL, lines\nfrom run_offline_wands_replay import assert_own_gpu, gpu_processes")
s=substitute(s,"os.path.basename(a.model) != 'SmolLM2-1.7B-Instruct'","os.path.basename(a.model) != 'Qwen2.5-7B-Instruct'")
s=s.replace('recorded SmolLM2 model','recorded Qwen2.5 model')
s=substitute(s,"    a.out.mkdir(parents=True, exist_ok=False)","    if gpu_processes():\n        raise RuntimeError('GPU in use; no model loaded.')\n    a.out.mkdir(parents=True, exist_ok=False)")
s=s.replace('llm.generate(', 'guarded_generate(')
s=substitute(s,'    tok = llm.get_tokenizer()\n','    tok = llm.get_tokenizer()\n'+wrapper)
s=substitute(s,"'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection'", "'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection', 'batch_size': 64")
write_new('run_wands_7b_posthoc.py',s)
s=(base/'analyze_selective_model_sensitivity.py').read_text()
s=substitute(s,"PROTOCOL = 'PROTOCOL_selective_model_sensitivity.md'",f"PROTOCOL = '{protocol}'")
write_new('analyze_wands_7b_posthoc.py',s)
s=(base/'verify_selective_model_sensitivity.py').read_text()
s=substitute(s,"'PROTOCOL_selective_model_sensitivity.md'",repr(protocol))
s=substitute(s,"'SmolLM2-1.7B-Instruct'", "'Qwen2.5-7B-Instruct'")
write_new('verify_wands_7b_posthoc.py',s)
s=(base/'run_trec_covid_smollm.py').read_text()
s=substitute(s,"PROTOCOL = 'PROTOCOL_trec_covid_smollm.md'",f"PROTOCOL = '{protocol}'")
s=substitute(s,"a.model.name != 'SmolLM2-1.7B-Instruct'", "a.model.name != 'Qwen2.5-7B-Instruct'")
s=s.replace('recorded SmolLM2 model','recorded Qwen2.5 model')
s=s.replace('llm.generate(', 'guarded_generate(')
s=substitute(s,'    tokenizer = llm.get_tokenizer()\n','    tokenizer = llm.get_tokenizer()\n'+wrapper)
s=substitute(s,"'gpu_memory_utilization': .70, 'enforce_eager': True, 'prefix_caching': True", "'gpu_memory_utilization': .70, 'enforce_eager': True, 'prefix_caching': True, 'batch_size': 64")
write_new('run_trec_7b_posthoc.py',s)
files=[out/name for name in [protocol,'prepare_multidomain.py','run_wands_7b_posthoc.py','analyze_wands_7b_posthoc.py','verify_wands_7b_posthoc.py','run_trec_7b_posthoc.py']]
files += [base/name for name in ['wands_selective_execution_selection/inputs.jsonl','wands_selective_execution_selection/selection.json','trec_covid_transfer_selection/inputs.jsonl','trec_covid_transfer_selection/selection.json']]
manifest={}
for p in files:
 name=('posthoc/'+p.name) if p.is_relative_to(out) else ('artifact/'+p.relative_to(base).as_posix())
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 manifest[name]=h.hexdigest()
p=out/'PRE_7B_MULTIDOMAIN_SHA256.json'
with p.open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
print('Prepared and froze',len(manifest),'multidomain files; manifest:',hashlib.sha256(p.read_bytes()).hexdigest())
