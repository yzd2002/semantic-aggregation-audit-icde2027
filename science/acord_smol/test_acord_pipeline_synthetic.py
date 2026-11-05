#!/usr/bin/env python3
"""End-to-end synthetic fixture; no GPU, model, or ACORD judgments are used."""
import hashlib,json,os,sys,types
from pathlib import Path
import run_acord_holdout as runner
import analyze_acord_holdout as analysis
import acord_prompts

HERE=Path(__file__).resolve().parent
root=Path(os.environ.get('INTENTSAT_ACORD_SYNTHETIC_DIR',str(HERE/'SYNTHETIC_PIPELINE_TEST_v2')))
if root.exists():raise FileExistsError('Choose a fresh fixture directory; preserve prior test data')
root.mkdir();model=root/'SmolLM2-1.7B-Instruct';model.mkdir();sel=root/'selection';sel.mkdir();pre=root/'preflight';pre.mkdir()
rows=[]
for scenario in ('uniform','case_control'):
    items=[{'doc_id':'synthetic-'+str(i),'text':'candidate clause '+str(i),'score':3 if i%2==0 else 0,
            'gold':'Y' if i%2==0 else 'N','source_chars':20} for i in range(32)]
    rows.append({'pool_id':scenario+':SYNTHETIC','query_id':'SYNTHETIC','scenario':scenario,'query':'candidate',
                 'query_category':'synthetic','query_type':'synthetic','items':items})
data=sel/'inputs.jsonl';data.write_text(''.join(json.dumps(x)+'\n' for x in rows))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=HERE/'PROTOCOL_acord_holdout.md';amendment=HERE/'AMENDMENT_prefix_search.md';selection_sha=sha(data)
(sel/'selection.json').write_text(json.dumps({'protocol':protocol.name,'protocol_sha256':sha(protocol),
                                              'inputs_sha256':selection_sha,'eligible_queries':1,
                                              'prefix_length':2400,'prefix_amendment_sha256':sha(amendment),
                                              'base_inputs_sha256':'SYNTHETIC_BASE','source_sha256':'SYNTHETIC_SOURCE'}))
for name in ('tokenizer.json','tokenizer_config.json','config.json'):(model/name).write_text('SYNTHETIC '+name)
asset={n:sha(model/n) for n in ('tokenizer.json','tokenizer_config.json','config.json')}
records=[]
for row in rows:
    tasks=[]
    for start in (0,8,16,24):
        for mode,reversed_order in (('keyed_original',False),('keyed_reverse',True)):
            prompt,ids=acord_prompts.keyed_prompt(row,start,reversed_order);tasks.append((mode,start,prompt,128,ids))
    for i in range(32):tasks.append(('pointwise',i,acord_prompts.point_prompt(row,i),4,None))
    for mode,start,prompt,cap,ids in tasks:
        count=lambda x:len((acord_prompts.SYSTEM+'\n'+x).split())
        records.append({'pool_id':row['pool_id'],'mode':mode,'start':start,'presented_ids':ids,
                        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                        'prompt_tokens':count(prompt),'retry_prompt_tokens':count(prompt+acord_prompts.RETRY_SUFFIX),
                        'output_cap':cap,'worst_total':count(prompt+acord_prompts.RETRY_SUFFIX)+cap})
(pre/'prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
(pre/'summary.json').write_text(json.dumps({'selection_sha256':selection_sha,'overflow_pool_count':0,
    'prompt_code_sha256':sha(HERE/'acord_prompts.py'),'prompt_records_sha256':sha(pre/'prompts.jsonl'),
    'tokenizer_assets_sha256':asset}))
runner.EXPECTED_QWEN_PREFLIGHT_SHA=sha(pre/'summary.json')
runner.EXPECTED_SMOL_PREFLIGHT_SHA=sha(pre/'summary.json')
decision=root/'decision.json'
decision.write_text(json.dumps({'selected_cap_unicode_code_points':2400,
    'prefix_search_amendment_sha256':sha(amendment),'base_600_inputs_sha256':'SYNTHETIC_BASE',
    'source_sha256':'SYNTHETIC_SOURCE','variants':{'2400':{
    'selection_sha256':selection_sha,'preflight_summary_sha256':sha(pre/'summary.json'),
    'prompt_code_sha256':sha(HERE/'acord_prompts.py'),'overflow_pool_count':0}}}))
manifest=root/'weights.txt'
with manifest.open('w') as f:
    p=model/'model.safetensors';p.write_text('SYNTHETIC WEIGHT');f.write(sha(p)+'  '+str(p)+'\n')
runner.EXPECTED_SMOL_WEIGHT_SHA=sha(model/'model.safetensors')

class FakeGuidance:
    def __init__(self,regex):self.regex=regex
class FakeSampling:
    def __init__(self,temperature,max_tokens,guided_decoding=None):self.guided_decoding=guided_decoding
class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        text='\n'.join(x['content'] for x in messages)
        return list(range(len(text.split()))) if tokenize else text
class FakeOutput:
    def __init__(self,text):self.text=text;self.finish_reason='stop';self.token_ids=list(range(len(text.split())+1))
class FakeResponse:
    def __init__(self,prompt,text):self.outputs=[FakeOutput(text)];self.prompt_token_ids=list(range(len(prompt.split())))
class FakeLLM:
    def __init__(self,**kwargs):self.failed_once=False
    def get_tokenizer(self):return FakeTokenizer()
    def generate(self,prompts,params,use_tqdm=False):
        out=[]
        for p in prompts:
            if 'Reply with only the word BANANA.' in p:
                answer=('BANANA' if params.guided_decoding is None else
                        'Y' if params.guided_decoding.regex=='[YN]' else
                        '{'+','.join(json.dumps(str(i))+':"Y"' for i in range(1,9))+'}')
            elif 'Clauses in order:\n' in p:
                body=p.split('Clauses in order:\n',1)[1].split('\nReturn only a JSON object',1)[0]
                presented=[json.loads(x)['id'] for x in body.split('\n')]
                labels={idx:('Y' if j%2==0 else 'N') for j,idx in enumerate(presented)}
                answer='{'+','.join(json.dumps(str(i))+':"'+labels[i]+'"' for i in range(1,9))+'}'
                if not self.failed_once and runner.RETRY_SUFFIX not in p:
                    self.failed_once=True;answer='SYNTHETIC_INVALID'
            else:answer='Y'
            out.append(FakeResponse(p,answer))
        return out

fake_torch=types.ModuleType('torch');fake_torch.cuda=types.SimpleNamespace(get_device_name=lambda _: 'SYNTHETIC GPU (not used)')
fake_vllm=types.ModuleType('vllm');fake_vllm.LLM=FakeLLM;fake_vllm.SamplingParams=FakeSampling
fake_sampling=types.ModuleType('vllm.sampling_params');fake_sampling.GuidedDecodingParams=FakeGuidance
sys.modules.update({'torch':fake_torch,'vllm':fake_vllm,'vllm.sampling_params':fake_sampling})
versions={'vllm':'0.10.2','torch':'2.8.0+cu128','transformers':'4.57.1','numpy':'2.2.6',
          'xgrammar':'0.1.23','outlines-core':'0.2.11','setuptools':'79.0.1'}
runner.metadata.version=lambda name:versions[name]
runner.memory_guard=lambda initial=False:None
runner.gpu_processes=lambda:set()
runner.assert_own_gpu=lambda:None
os.environ['VLLM_USE_V1']='1'
out=root/'mock_run';sys.argv=['run_acord_holdout.py','--model',str(model),'--selection',str(sel),
    '--preflight',str(pre),'--protocol',str(protocol),'--amendment',str(amendment),
    '--decision',str(decision),'--followup-protocol',str(HERE/'PROTOCOL_smol_followup.md'),
    '--weights-manifest',str(manifest),'--out',str(out)]
runner.main()
summary=json.loads((out/'summary.json').read_text());assert summary['completed'] and summary['runtime']['gpu'].startswith('SYNTHETIC')
assert summary['modes']['original']['retry_calls']==1
assert summary['modes']['selective_pointwise']['initial_calls']==64
assert summary['selective_pipeline']['initial_calls']==80 and summary['selective_pipeline']['retry_calls']==1
dest=root/'synthetic_analysis.json';analysis.main(sel,out,dest);result=json.loads(dest.read_text())
for scenario in ('uniform','case_control'):
    x=result['scenarios'][scenario];assert x['attempted_pools']==x['common_complete_pools']==1
    assert x['metrics']['original']['ratio_mae']==0 and x['midpoint_ratio_mae']==0
    assert x['metrics']['selective']['ratio_mae']==.5 and abs(x['metrics']['selective']['f1']-2/3)<1e-12
    assert x['metrics']['lexical_fallback']['ratio_mae']==.5 and x['selective_minus']['lexical_fallback']['ratio_mae']==0
    assert x['selective_minus']['original']['query_bootstrap_95']==[.5,.5]
import verify_acord_run
verify_acord_run.main(sel,pre,out,dest)
assert summary['prefix_length_unicode_code_points']==2400 and summary['prefix_decision_sha256']==sha(decision)
tampered=json.loads(decision.read_text());tampered['selected_cap_unicode_code_points']=1200
bad_decision=root/'tampered_decision.json';bad_decision.write_text(json.dumps(tampered))
bad_out=root/'must_not_exist';sys.argv=['run_acord_holdout.py','--model',str(model),'--selection',str(sel),
    '--preflight',str(pre),'--protocol',str(protocol),'--amendment',str(amendment),
    '--decision',str(bad_decision),'--followup-protocol',str(HERE/'PROTOCOL_smol_followup.md'),
    '--weights-manifest',str(manifest),'--out',str(bad_out)]
try:runner.main()
except ValueError:pass
else:raise AssertionError('A tampered decision was accepted')
assert not bad_out.exists()
bad_amendment=root/'tampered_amendment.md';bad_amendment.write_text('SYNTHETIC TAMPERED AMENDMENT')
bad_out2=root/'also_must_not_exist';sys.argv=['run_acord_holdout.py','--model',str(model),'--selection',str(sel),
    '--preflight',str(pre),'--protocol',str(protocol),'--amendment',str(bad_amendment),
    '--decision',str(decision),'--followup-protocol',str(HERE/'PROTOCOL_smol_followup.md'),
    '--weights-manifest',str(manifest),'--out',str(bad_out2)]
try:runner.main()
except ValueError:pass
else:raise AssertionError('A tampered amendment was accepted')
assert not bad_out2.exists()
bad_protocol=root/'tampered_followup.md';bad_protocol.write_text('SYNTHETIC TAMPERED FOLLOWUP')
bad_out3=root/'third_must_not_exist';sys.argv=['run_acord_holdout.py','--model',str(model),'--selection',str(sel),
    '--preflight',str(pre),'--protocol',str(protocol),'--amendment',str(amendment),
    '--decision',str(decision),'--followup-protocol',str(bad_protocol),
    '--weights-manifest',str(manifest),'--out',str(bad_out3)]
try:runner.main()
except ValueError:pass
else:raise AssertionError('A tampered follow-up protocol was accepted')
assert not bad_out3.exists()
print('SYNTHETIC fixture passed, including tampered decision, amendments and follow-up rejection before output creation; no GPU inference occurred.')
