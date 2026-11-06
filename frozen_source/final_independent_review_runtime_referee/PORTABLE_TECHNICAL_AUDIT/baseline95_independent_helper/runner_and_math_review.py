#!/usr/bin/env python3
"""Independent synthetic oracles versus actual frozen runner/analyzer. CPU only."""
import os,sys,json,hashlib,contextlib,io,re,types,copy,collections,random,itertools
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit');E=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import run_gpu as runner
import analyze_transfer as analyzer
import decision_metrics as metrics
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def h(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def sha(p):return h(Path(p).read_bytes())
def lines(p):
 if not p.exists():return []
 return [json.loads(t) for t in p.read_text().splitlines() if t.strip()]
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rows(p,x):
 with p.open('x') as f:
  for r in x:f.write(compact(r)+'\n')
def own_gold(pred,gold):
 cnt=collections.Counter({k:0 for k in ('tp','tn','fp','fn')})
 for p,g in zip(pred,gold):
  if type(p) is int and p in (0,1):cnt['tp' if p==g==1 else 'tn' if p==g==0 else 'fp' if p==1 else 'fn']+=1
 full=len(pred)==128 and all(type(v) is int and v in (0,1) for v in pred);n=sum(gold);total=sum(pred) if full else None
 tp,tn,fp,fn=(cnt[k] for k in ('tp','tn','fp','fn'))
 return {'complete':full,'fixed_denominator':128,'valid_outcomes':sum(cnt.values()),'invalid_or_missing_outcomes':128-sum(cnt.values()),'confusion_counts_on_valid_outcomes':dict(cnt),'gold_positive_count':n,'predicted_positive_count':total,'predicted_positive_fraction':total/128 if full else None,'accuracy':(tp+tn)/128 if full else None,'error':(fp+fn)/128 if full else None,'balanced_accuracy':(tp/n+tn/(128-n))/2 if full and 0<n<128 else None,'f1':((2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0) if full else None,'signed_count_bias':total-n if full else None,'absolute_ratio_error':abs(total-n)/128 if full else None,'opposing_gold_error_pairs':min(fp,fn) if full else None,'canceled_gold_error_mass':2*min(fp,fn)/128 if full else None,'constant_prediction':('Y' if total==128 else 'N' if total==0 else None) if full else None,'count_status':'computed_on_all128' if full else 'abstain_incomplete_relation'}
def own_fidelity(a,b):
 full=all(len(z)==128 and all(type(v) is int and v in (0,1) for v in z) for z in (a,b));n=sum(x!=y for x,y in zip(a,b)) if full else None
 return {'complete':full,'numerator':n,'fixed_denominator':128,'value':n/128 if full else None}
def math_checks():
 rng=random.Random(2026110619);checks=0
 for pattern in range(32):
  gold=[rng.randrange(2) for _ in range(128)] if pattern<28 else [int(pattern%2)]*128
  pred=[rng.randrange(2) for _ in range(128)]
  if pattern%6==0:pred[127]=None
  if pattern%6==1:pred[0]=True
  if pattern%6==2:pred[64]='1'
  if pattern%6==3:pred=[int(pattern%2)]*128
  other=[rng.randrange(2) for _ in range(128)]
  expected=own_gold(pred,gold);assert expected==metrics.gold_metrics(pred,gold);checks+=len(expected)
  f=own_fidelity(pred,other);assert f==metrics.disagreement(pred,other)
  for tau in (Fraction(1,20),Fraction(1,10),Fraction(1,5)):
   want=None if not f['complete'] else f['numerator']*tau.denominator<=128*tau.numerator
   assert metrics.empirical_indicator(f,tau)==want;checks+=1
 # Exhaust every valid/invalid two-replay boundary in integer disagreement units.
 gold=[i%2 for i in range(128)];base=[0]*128
 for a,b in ((0,0),(6,7),(12,13),(25,26),(0,128),(128,128),(None,0),(0,None),(None,None),(6,6),(25,25),(26,26)):
  replays={}
  for index,n in enumerate((a,b),1):
   p=[1]*n+[0]*(128-n) if n is not None else [None]*128
   replays[str(index)]={'singleton_fidelity':own_fidelity(p,base),'gold':own_gold(p,gold)}
  for tau in (Fraction(1,20),Fraction(1,10),Fraction(1,5)):
   got=analyzer.relation_summary(replays,tau)
   assert got['within_tolerance_MEAN']==(None if a is None or b is None else (a+b)*tau.denominator<=256*tau.numerator)
   assert got['within_tolerance_BOTH']==(None if a is None or b is None else max(a,b)*tau.denominator<=128*tau.numerator)
   assert got['mean_S_disagreement']==(None if a is None or b is None else (a+b)/256);checks+=3
 # Nonlinear COUNT is computed per relation before averaging: opposite biases do not cancel.
 rels={}
 for name,p in [('a',[1]*128),('b',[0]*128)]:
  g=own_gold(p,gold);rep={str(i):{'singleton_fidelity':own_fidelity(p,base),'gold':g} for i in (1,2)};rels[name]={'summary':analyzer.relation_summary(rep,Fraction(1,5))}
 got=analyzer.aggregate_relations(rels);assert got['public_gold_relation_macro_mean']['absolute_ratio_error']==0.5;checks+=1
 rels['b']['summary']['within_tolerance_MEAN']=None
 got=analyzer.aggregate_relations(rels);assert got['tolerance_MEAN']=={'pass':0,'fail':1,'unknown':1,'fixed_relation_denominator':2,'pass_fraction':None};checks+=1
 # Unknown per-request usage must produce null totals, lower bounds and fixed missing counts.
 jobs=[{'request_id':str(i),'occurrence_ids':[i+1]} for i in range(4)]
 returned={'0':{'actual_input_tokens':12,'actual_output_tokens':3},'1':{'actual_input_tokens':None,'actual_output_tokens':4},'2':{'actual_input_tokens':9,'actual_output_tokens':None}}
 got=analyzer.cost(jobs,returned,{'0','1','2'});assert got['actual_total_tokens'] is None and got['actual_input_tokens'] is None and got['actual_output_tokens'] is None
 assert got['unknown_input_token_call_count']==got['unknown_output_token_call_count']==2 and got['observed_total_token_lower_bound']==28;checks+=6
 cal={'planned_calls':1152,'returned_calls':1152,'planned_requested_items':3328,'actual_input_tokens':400000,'actual_output_tokens':20000,'actual_total_tokens':420000}
 total=analyzer.plus_calibration(got,cal);assert total['planned_calls']==1156 and total['actual_total_tokens'] is None;checks+=2
 for k in (2,4,8):
  saving=128-128//k;equal=(1152+saving-1)//saving;strict=1152//saving+1;be=metrics.arithmetic_break_even(k)
  assert be['equal_or_lower_call_cost_relations']==equal and be['strictly_lower_call_cost_relations']==strict;checks+=2
 assert metrics.arithmetic_break_even(1)['equal_or_lower_call_cost_relations'] is None
 return {'status':'PASS','individual_equalities':checks,'cases':32,'complete_fixed_128_COUNT':True,'127_valid_and_invalid_bool_string_None_abstain':True,'MEAN_and_BOTH_exact_integer_boundaries':True,'nonlinear_relation_macro_COUNT':True,'unknown_usage_null_not_zero':True,'producer_math_used_as_oracle':False}
def runner_checks():
 science=lines(R/'prepared/qwen_requests.jsonl')[:3];schedule=lines(R/'prepared/qwen_schedule.jsonl')[:3];controls=lines(R/'prepared/qwen_controls.jsonl');cases=[]
 for stage,fault in [('science','short'),('science','extra'),('science','prompt_second'),('science','multiple_outputs'),('science','invalid_raw'),('science','unknown_output_tokens'),('science','duplicate_engine_id'),('diagnostic','short'),('diagnostic','extra'),('diagnostic','prompt_second'),('diagnostic','invalid_raw')]:
  fixture=E/'runner_fixtures'/f'{stage}_{fault}';fixture.mkdir(parents=True,exist_ok=False);(fixture/'prepared').mkdir();(fixture/'runtime_override').mkdir()
  write(fixture/'model_locations.json',{'qwen':str(fixture/'CPU_SYNTHETIC_MODEL_NOT_LOADED')});write(fixture/'prepared/MODEL_FILE_SHA256.json',{'qwen':{'files':{}}})
  write(fixture/'prepared/runtime_configuration.json',{'packages':{},'python':__import__('platform').python_version(),'python_executable':sys.executable,'FLASHINFER_WORKSPACE_BASE':str(fixture/'CPU_SYNTHETIC_CACHE')})
  rows(fixture/'prepared/qwen_requests.jsonl',science);rows(fixture/'prepared/qwen_schedule.jsonl',schedule);rows(fixture/'prepared/qwen_controls.jsonl',controls)
  class Sample:
   def __init__(self,**kw):self.__dict__.update(kw)
  observed=[]
  class Engine:
   def __init__(self,**kw):self.calls=0
   def generate(self,prompts,parameters,use_tqdm=False):
    self.calls+=1;out=[];target=self.calls==(4 if stage=='science' else 3)
    for i,(p,param) in enumerate(zip(prompts,parameters)):
     ids=[] if param.guided_decoding is None else re.findall(r'"(\d+)":',param.guided_decoding.regex)
     raw='BANANA' if param.guided_decoding is None else compact({i:'Y' for i in ids})
     if target and fault=='invalid_raw' and i==1:raw='{"1":"Y"}\nCPU_SYNTHETIC_INVALID'
     tokens=list(p['prompt_token_ids'])
     if target and fault=='prompt_second' and i==1:tokens.append(987654321)
     answer=types.SimpleNamespace(text=raw,token_ids=None if target and fault=='unknown_output_tokens' and i==1 else [9000+self.calls,1000+i],finish_reason='stop',stop_reason=None)
     seqs=[answer,copy.deepcopy(answer)] if target and fault=='multiple_outputs' and i==1 else [answer]
     eid=f'CPU_SYNTHETIC_{self.calls}_{i}' if not(target and fault=='duplicate_engine_id' and i==1) else f'CPU_SYNTHETIC_{self.calls}_0'
     out.append(types.SimpleNamespace(prompt_token_ids=tokens,outputs=seqs,request_id=eid))
    if target and fault=='short':out=out[:-1]
    if target and fault=='extra':out.append(copy.deepcopy(out[-1]));out[-1].request_id+='EXTRA'
    observed.append(out);return out
  mod=types.ModuleType('vllm');mod.LLM=Engine;mod.SamplingParams=Sample;sample=types.ModuleType('vllm.sampling_params');sample.GuidedDecodingParams=Sample
  compat=types.ModuleType('single_rank_nccl_compat');compat.ACTIVE=True;compat.__file__=str(fixture/'runtime_override/single_rank_nccl_compat.py')
  actualsha=runner.sha
  def fixture_sha(p):return 'CPU_SYNTHETIC_NO_REAL_FREEZE' if Path(p).name=='PREINFERENCE_FROZEN_SHA256.json' else actualsha(p)
  failure=None;log=io.StringIO()
  with patch.object(runner,'ROOT',fixture),patch.object(runner,'verify',lambda *a:{'CPU_SYNTHETIC_NO_GPU_GO':True}),patch.object(runner,'processes',lambda *a:[]),patch.object(runner,'owned',lambda *a:[]),patch.object(runner,'sha',fixture_sha),patch.object(analyzer,'analyze',lambda *a:None),patch.dict(sys.modules,{'vllm':mod,'vllm.sampling_params':sample,'single_rank_nccl_compat':compat}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'CPU_SYNTHETIC','FLASHINFER_WORKSPACE_BASE':str(fixture/'CPU_SYNTHETIC_CACHE')}),contextlib.redirect_stdout(log):
   try:runner.main('qwen','CPU_SYNTHETIC')
   except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
  (fixture/'runner_stdout.log').write_text(log.getvalue());run=fixture/'run_qwen';native=lines(run/'engine_returns.jsonl');canon=lines(run/'responses.jsonl')
  assert len(native)==len(observed)
  for saved,actual in zip(native,observed):
   assert saved['actual_return_count']==len(actual)
   for a,b in zip(saved['actual_returns'],actual):
    assert a['engine_request_id']==b.request_id and a['actual_prompt_token_ids']==b.prompt_token_ids
    assert [z['raw'] for z in a['outputs']]==[z.text for z in b.outputs]
    assert [z['output_token_ids'] for z in a['outputs']]==[z.token_ids for z in b.outputs]
  should_fail=not(stage=='science' and fault in ('invalid_raw','duplicate_engine_id'))
  assert (failure is not None)==should_fail
  assert (run/'FAILURE.json').exists()==should_fail
  if stage=='diagnostic':assert not canon and not (run/'DATA_START.json').exists()
  if stage=='science' and fault=='invalid_raw':assert len(canon)==3 and canon[1]['labels'] is None and canon[1]['strict_valid'] is False
  if stage=='science' and fault in ('short','extra'):assert canon==[]
  # Actual analyzer must reject ambiguous IDs and bind every canonical raw to the journal.
  analyzed=None;rejected=None
  with patch.object(analyzer,'ROOT',fixture),patch.object(analyzer,'sha',fixture_sha):
   try:analyzed=analyzer.load_trace('qwen')[-1]
   except BaseException as exc:rejected=type(exc).__name__+': '+str(exc)
  if fault=='duplicate_engine_id':assert rejected is not None
  else:
   assert rejected is None
   literal=analyzed['all_native_returns_science_journal' if stage=='science' else 'all_native_returns_control_journal']
   actual=[a for e in native if (e['stage']=='science')==(stage=='science') for a in e['actual_returns']]
   assert literal['observed_input_token_sum_from_actual_returns']==sum(a['actual_input_tokens'] for a in actual if type(a['actual_input_tokens']) is int)
   assert literal['observed_output_token_sum_from_actual_returns']==sum(a['actual_output_tokens'] for a in actual if type(a['actual_output_tokens']) is int)
   if fault in ('short','extra','prompt_second','multiple_outputs','unknown_output_tokens'):
    z=analyzed['science_union' if stage=='science' else 'controls'];assert z['actual_total_tokens'] is None
  cases.append({'stage':stage,'fault':fault,'status':'PASS','runner_exception':failure,'analyzer_rejection':rejected,'all_actual_return_sequences_preserved':sum(sum(len(z.outputs) for z in b) for b in observed),'canonical_scientific_rows':len(canon),'fixture':str(fixture.relative_to(E))})
 return {'status':'PASS','test_count':len(cases),'actual_runner_source_sha256':sha(R/'run_gpu.py'),'actual_journal_source_sha256':sha(R/'return_journal.py'),'actual_analyzer_source_sha256':sha(R/'analyze_transfer.py'),'NO_GPU':True,'cases':cases}
def main():
 a=math_checks();write(E/'07_INDEPENDENT_MATH.json',a);print('independent math',a['individual_equalities'],'PASS',flush=True)
 b=runner_checks();write(E/'08_INDEPENDENT_RUNNER_FAULTS.json',b);print('actual runner faults',b['test_count'],'PASS',flush=True)
if __name__=='__main__':main()
