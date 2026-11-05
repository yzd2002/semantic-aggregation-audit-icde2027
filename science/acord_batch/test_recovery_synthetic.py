#!/usr/bin/env python3
"""No-GPU integration fixture for failure, retry, mode order and frozen inputs."""
import hashlib,json,os,sys,types
from pathlib import Path
import run_batch_recovery as run
import analyze_batch as analysis
import verify_batch as verify
from batch_prompts import SYSTEM,RETRY_SUFFIX,prompt

here=Path(__file__).resolve().parent;root=Path(os.environ['INTENTSAT_RECOVERY_TEST_DIR'])
if root.exists():raise FileExistsError('Use a fresh synthetic test directory')
root.mkdir(parents=True);model=root/'Qwen3-4B-Instruct-2507';model.mkdir();sel=root/'selection';sel.mkdir()
pre=root/'preflight';pre.mkdir();base=root/'baseline';base.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for scenario in ('uniform','case_control'):
    rows.append({'pool_id':scenario+':SYNTHETIC','scenario':scenario,'query':'candidate',
                 'items':[{'text':'synthetic clause '+str(i),'gold':'Y' if i%2==0 else 'N'} for i in range(32)]})
data=sel/'inputs.jsonl';data.write_text(''.join(json.dumps(x)+'\n' for x in rows));selection_sha=sha(data)
(sel/'selection.json').write_text(json.dumps({'inputs_sha256':selection_sha,'prefix_length':2400}))
for name in ('tokenizer.json','tokenizer_config.json','config.json'):(model/name).write_text('SYNTHETIC '+name)
assets={n:sha(model/n) for n in ('tokenizer.json','tokenizer_config.json','config.json')}
names=[f'model-0000{i}-of-00003.safetensors' for i in (1,2,3)]
for n in names:(model/n).write_text('SYNTHETIC '+n)
weights={n:sha(model/n) for n in names};weights_file=root/'weights.txt'
weights_file.write_text(''.join(v+'  '+str(model/k)+'\n' for k,v in weights.items()))
(base/'summary.json').write_text(json.dumps({'completed':True,'selection_inputs_sha256':selection_sha,'weights_sha256':weights}))
protocol=here/'PROTOCOL_recovery_batch.md';pre_records=[]
for row in rows:
    for size in (4,8):
        for start in range(0,32,size):
            for mode,reverse in (('original',False),('stable_reverse',True)):
                text,ids=prompt(row,start,size,reverse)
                count=lambda s:len((SYSTEM+'\n'+s).split())
                pre_records.append({'pool_id':row['pool_id'],'size':size,'start':start,'mode':mode,'presented_ids':ids,
                    'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':count(text),'retry_prompt_tokens':count(text+RETRY_SUFFIX)})
(pre/'prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pre_records))
(pre/'summary.json').write_text(json.dumps({'selection_sha256':selection_sha,'protocol_sha256':sha(protocol),
    'prompt_records_sha256':sha(pre/'prompts.jsonl'),'prompt_code_sha256':sha(here/'batch_prompts.py'),
    'base_prompt_code_sha256':sha(here/'acord_prompts.py'),'overflow_pool_count':0,'pool_count':2,'prompt_count':48,'tokenizer_assets_sha256':assets}))
manifest=root/'manifest.json';manifest.write_text(json.dumps({'protocol_sha256':sha(protocol),'selection_sha256':selection_sha,
    'baseline_sha256':sha(base/'summary.json'),'preflight_sha256':sha(pre/'summary.json'),
    'runner_sha256':sha(here/'run_batch_recovery.py'),'weights_manifest_sha256':sha(weights_file)}))
class Guide:
    def __init__(self,regex):self.regex=regex
class Sample:
    def __init__(self,temperature,max_tokens,guided_decoding=None):self.guided_decoding=guided_decoding
class Tokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        t='\n'.join(x['content'] for x in messages);return list(range(len(t.split()))) if tokenize else t
class Output:
    def __init__(self,text):self.text=text;self.finish_reason='stop';self.token_ids=list(range(len(text.split())+1))
class Response:
    def __init__(self,p,text):self.outputs=[Output(text)];self.prompt_token_ids=list(range(len(p.split())))
class Model:
    def __init__(self,**kw):self.failed=False
    def get_tokenizer(self):return Tokenizer()
    def generate(self,prompts,params,use_tqdm=False):
        responses=[]
        for text in prompts:
            size=params.guided_decoding.regex.count('[YN]') if params.guided_decoding else None
            if 'Reply with only the word BANANA.' in text:
                answer='BANANA' if size is None else '{'+','.join(json.dumps(str(i))+':"N"' for i in range(1,size+1))+'}'
            else:
                body=text.split('Clauses in order:\n',1)[1].split('\nReturn only',1)[0];items=[json.loads(s) for s in body.splitlines()]
                labels={item['id']:('Y' if int(item['clause'].split()[-1])%2==0 else 'N') for item in items}
                answer='{'+','.join(json.dumps(str(i))+':"'+labels[i]+'"' for i in range(1,size+1))+'}'
                if not self.failed and RETRY_SUFFIX not in text:self.failed=True;answer='SYNTHETIC_INVALID'
            responses.append(Response(text,answer))
        return responses

torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(get_device_name=lambda _: 'SYNTHETIC GPU (not used)')
vllm=types.ModuleType('vllm');vllm.LLM=Model;vllm.SamplingParams=Sample
sampling=types.ModuleType('vllm.sampling_params');sampling.GuidedDecodingParams=Guide
sys.modules.update({'torch':torch,'vllm':vllm,'vllm.sampling_params':sampling})
run.EXPECTED_POOLS=2;run.md.version=lambda name:run.WANTED[name];run.memory_guard=lambda initial=False:None
run.gpu_processes=lambda:set();run.assert_own_gpu=lambda:None;os.environ['VLLM_USE_V1']='1'
def args(out,freeze):return ['run_batch_recovery.py','--model',str(model),'--selection',str(sel),'--baseline',str(base),
    '--preflight',str(pre),'--protocol',str(protocol),'--weights-manifest',str(weights_file),'--manifest',str(freeze),'--out',str(out)]
out=root/'mock_run';sys.argv=args(out,manifest);run.main();s=json.loads((out/'summary.json').read_text())
assert s['completed'] and list(s['modes'])==[x[0] for x in run.MODES]
assert s['modes']['eight_original']['initial_calls']==8 and s['modes']['eight_original']['retry_calls']==1
assert s['modes']['four_original']['initial_calls']==16 and s['modes']['four_replay']['retry_calls']==0
for name,size in (('original',8),('pointwise',1)):
    records=[]
    for row in rows:
        for start in range(0,32,size):
            value=('{'+','.join(json.dumps(str(i))+':"N"' for i in range(1,9))+'}' if size==8 else row['items'][start]['gold'])
            records.append({'pool_id':row['pool_id'],'start':start,'final':value})
    (base/(name+'_responses.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in records))
dest=root/'analysis.json';analysis.main(sel,out,base,dest);result=json.loads(dest.read_text())
for scenario in ('uniform','case_control'):
    r=result['scenarios'][scenario]
    assert r['attempted_pools']==r['common_complete_pools']==1 and r['four_minus_eight_ratio_mae']==0
    assert r['metrics']['four_original']['f1']==1 and r['metrics']['prior_full_pointwise']['ratio_mae']==0
    assert r['four_stable_reverse_changed_labels']==r['four_replay_changed_labels']==r['eight_replay_changed_labels']==0
    assert r['prior_eight_vs_recovery_changed_labels']==16 and r['query_bootstrap_95']==[0,0]
verify.main(sel,pre,out,base,dest)
bad_analysis=json.loads(dest.read_text());bad_analysis['scenarios']['uniform']['metrics']['four_original']['ratio_mae']=.9
bad_path=root/'tampered_analysis.json';bad_path.write_text(json.dumps(bad_analysis))
try:verify.main(sel,pre,out,base,bad_path)
except AssertionError:pass
else:raise AssertionError('Tampered statistic was accepted')
bad=json.loads(manifest.read_text());bad['protocol_sha256']='tampered';badfile=root/'tampered.json';badfile.write_text(json.dumps(bad))
badout=root/'must_not_exist';sys.argv=args(badout,badfile)
try:run.main()
except ValueError:pass
else:raise AssertionError('Tampered protocol was accepted')
assert not badout.exists()
print('Synthetic recovery runner passed both sizes, retries, strict negative control, mode order and tamper rejection; no GPU inference.')
