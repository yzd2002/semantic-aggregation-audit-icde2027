#!/usr/bin/env python3
"""Full frozen schedules through actual CPU-faked runner/analyzer; independent oracle."""
import os,sys,json,hashlib,contextlib,io,re,types,collections,time
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit');E=Path(__file__).resolve().parent
sys.path.insert(0,str(R));sys.path.insert(0,str(E))
import run_gpu as runner,analyze_transfer as analyzer
from runner_and_math_review import own_gold,own_fidelity
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def h(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def sha(p):return h(Path(p).read_bytes())
def load(p):return json.loads(p.read_text())
def lines(p):
 with p.open() as f:return [json.loads(t) for t in f if t.strip()]
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def expected_label(j,i):
 base=int((i+int(j['task']=='boolq'))%3==0);rep=j['repeat'];relindex=int(j['relation_id'].split('/')[1])
 if j['execution_type']=='S':return base
 if j['execution_type']=='P':return base ^ int(i%31<rep)
 salt=int(h(j['configuration_id'])[:4],16)%7
 return base ^ int((i+rep*7+relindex)%19<1+salt)
def deliberately_invalid(j):
 return j['model']=='smol' and j['relation_id']=='boolq/19' and j['configuration_id']=='k8/random/C' and j['repeat']==2 and j['batch_index']==0
def model_test(model):
 f=E/'CPU_SYNTHETIC_FULL_PIPELINE'/model;f.mkdir(parents=True,exist_ok=False);(f/'prepared').mkdir();(f/'runtime_override').mkdir()
 for name in ('POLICY_LOCK.json','PREINFERENCE_FROZEN_SHA256.json','inputs'):(f/name).symlink_to(R/name,target_is_directory=(R/name).is_dir())
 for p in (R/'prepared').iterdir():
  if p.is_file() and p.name not in ('runtime_configuration.json','MODEL_FILE_SHA256.json'):(f/'prepared'/p.name).symlink_to(p)
 write(f/'model_locations.json',{model:str(f/'CPU_SYNTHETIC_NO_MODEL')});write(f/'prepared/MODEL_FILE_SHA256.json',{model:{'files':{}}})
 write(f/'prepared/runtime_configuration.json',{'packages':{},'python':__import__('platform').python_version(),'python_executable':sys.executable,'FLASHINFER_WORKSPACE_BASE':str(f/'CPU_SYNTHETIC_CACHE')})
 jobs=lines(R/'prepared'/f'{model}_schedule.jsonl');controls=lines(R/'prepared'/f'{model}_controls.jsonl');goldrows=lines(R/'inputs/gold.jsonl');gold={}
 for r in goldrows:gold.setdefault(r['relation_id'],[None]*128)[r['occurrence_id']-1]=r['gold']
 vector=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(lambda:[None]*128)));expected_records={};counts={'science':0,'controls':0}
 class Param:
  def __init__(self,**kw):self.__dict__.update(kw)
 class Engine:
  def __init__(self,**kw):self.call=0;self.offset=0
  def generate(self,prompts,parameters,use_tqdm=False):
   self.call+=1;science=self.call>3;out=[];block=jobs[self.offset:self.offset+len(prompts)] if science else [j for j in controls if j['name']==('ordinary_banana','guided_banana','synthetic_format_smoke')[self.call-1]]
   assert len(block)==len(prompts)
   for j,p,param in zip(block,prompts,parameters):
    ids=j['output_ids'] if not science else sorted(j['occurrence_ids'],reverse=j['execution_type']=='O')
    if science:
     values={i:expected_label(j,i) for i in ids};raw=compact({str(i):'Y' if values[i] else 'N' for i in ids})
     if deliberately_invalid(j):raw+='\n'
     for i in j['occurrence_ids']:vector[j['relation_id']][j['configuration_id']][j['repeat']][i-1]=None if deliberately_invalid(j) else values[i]
    else:raw='BANANA' if param.guided_decoding is None else compact({str(i):'Y' for i in ids})
    tids=[100+i for i in range(2+len(ids))];eid='CPU_SYNTHETIC_FULL_'+j['request_id'];counts['science' if science else 'controls']+=1
    expected_records[j['request_id']]={'input_tokens':len(p['prompt_token_ids']),'output_tokens':len(tids),'raw':raw}
    out.append(types.SimpleNamespace(prompt_token_ids=list(p['prompt_token_ids']),request_id=eid,outputs=[types.SimpleNamespace(text=raw,token_ids=tids,finish_reason='stop',stop_reason=None)]))
   if science:self.offset+=len(block)
   return out
 mod=types.ModuleType('vllm');mod.LLM=Engine;mod.SamplingParams=Param;sampling=types.ModuleType('vllm.sampling_params');sampling.GuidedDecodingParams=Param;compat=types.ModuleType('single_rank_nccl_compat');compat.ACTIVE=True;compat.__file__=str(f/'runtime_override/single_rank_nccl_compat.py');stdout=io.StringIO()
 with patch.object(runner,'ROOT',f),patch.object(runner,'verify',lambda *a:{'CPU_SYNTHETIC_NO_GPU_RELEASE':True}),patch.object(runner,'processes',lambda *a:[]),patch.object(runner,'owned',lambda *a:[]),patch.object(analyzer,'ROOT',f),patch.dict(sys.modules,{'vllm':mod,'vllm.sampling_params':sampling,'single_rank_nccl_compat':compat}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'CPU_SYNTHETIC','FLASHINFER_WORKSPACE_BASE':str(f/'CPU_SYNTHETIC_CACHE')}),contextlib.redirect_stdout(stdout):runner.main(model,'CPU_SYNTHETIC')
 (f/'runner_stdout.log').write_text(stdout.getvalue());result=load(f/f'analysis_{model}/analysis.json');equalities=0
 def eq(a,b):
  nonlocal equalities
  assert a==b,(a,b);equalities+=1
 for rel,configs in vector.items():
  for config,replays in configs.items():
   for repeat,pred in replays.items():
    s=vector[rel]['singleton'][repeat];got=result['all_executed_configurations'][rel][config]['replays'][str(repeat)];eq(got['gold'],own_gold(pred,gold[rel]));eq(got['singleton_fidelity'],None if config=='singleton' else own_fidelity(pred,s))
    both=all(all(type(z) is int and z in (0,1) for z in vec) for vec in (pred,s))
    if config!='singleton' and both:
     flips=sum(a!=b for a,b in zip(pred,s));wr=sum(p==g and ref!=g for p,ref,g in zip(pred,s,gold[rel]));rw=sum(p!=g and ref==g for p,ref,g in zip(pred,s,gold[rel]));delta=sum(pred)-sum(s)
     eq(got['gold_and_count_transport_from_reference'],{'reference_wrong_execution_right':wr,'reference_right_execution_wrong':rw,'execution_minus_reference_gold_error':(rw-wr)/128,'signed_count_change_from_reference':delta,'absolute_ratio_change_from_reference':abs(delta)/128,'candidate_flips':flips,'opposing_flip_pairs':(flips-abs(delta))//2})
 policies=[p for p in load(R/'POLICY_LOCK.json')['payload']['policies'] if p['model']==model]
 def execution_cost(sub):
  return {'planned_calls':len(sub),'returned_calls':len(sub),'planned_requested_items':sum(len(j['occurrence_ids']) for j in sub),'actual_input_tokens':sum(expected_records[j['request_id']]['input_tokens'] for j in sub),'actual_output_tokens':sum(expected_records[j['request_id']]['output_tokens'] for j in sub)}
 for p in policies:
  pid='/'.join(p[k] for k in ('model','task','tolerance','selector'));got=result['all_policies'][pid];config='singleton_execution' if p['chosen']['configuration_id']=='singleton' else p['chosen']['configuration_id'];rels=sorted(r for r in vector if r.startswith(p['task']+'/'));tau=Fraction(p['tolerance']);means=[];boths=[];ars=[]
  eq(got['executed_configuration_id'],config)
  for rel in rels:
   ns=[];gs=[]
   for rep in (1,2):
    pred=vector[rel][config][rep];ref=vector[rel]['singleton'][rep];loss=own_fidelity(pred,ref);g=own_gold(pred,gold[rel]);entry=got['relations'][rel]['replays'][str(rep)];eq(entry['gold'],g);eq(entry['singleton_fidelity'],loss);eq(entry['empirical_tolerance_satisfied'],None if not loss['complete'] else loss['numerator']*tau.denominator<=128*tau.numerator);ns.append(loss['numerator']);gs.append(g['absolute_ratio_error'])
   mean_ok=None if any(n is None for n in ns) else sum(ns)*tau.denominator<=256*tau.numerator;both_ok=None if any(n is None for n in ns) else max(ns)*tau.denominator<=128*tau.numerator;means.append(mean_ok);boths.append(both_ok);ars.append(None if any(g is None for g in gs) else sum(gs)/2)
   eq(got['relations'][rel]['summary']['within_tolerance_MEAN'],mean_ok);eq(got['relations'][rel]['summary']['within_tolerance_BOTH'],both_ok)
  for key,vs in [('MEAN',means),('BOTH',boths)]:eq(got['aggregate']['tolerance_'+key],{'pass':sum(v is True for v in vs),'fail':sum(v is False for v in vs),'unknown':sum(v is None for v in vs),'fixed_relation_denominator':len(rels),'pass_fraction':None if any(v is None for v in vs) else sum(v is True for v in vs)/len(rels)})
  eq(got['aggregate']['public_gold_relation_macro_mean']['absolute_ratio_error'],None if any(z is None for z in ars) else sum(ars)/len(ars))
  eq(len(got['all_prefix_costs']),len(rels))
  for m,prefix in enumerate(got['all_prefix_costs'],1):
   for rep in (1,2):
    exejobs=[j for j in jobs if j['relation_id'] in rels[:m] and j['repeat']==rep and j['configuration_id']==config];pjobs=[j for j in jobs if j['relation_id'] in rels[:m] and j['repeat']==rep and j['execution_type']=='P'];exe=execution_cost(exejobs);baseline=execution_cost(pjobs);entry=prefix['replays'][str(rep)]
    for k in exe:eq(entry['execution'][k],exe[k]);eq(entry['independent_P_baseline'][k],baseline[k]);eq(entry['old_calibration_once_plus_execution'][k],p['observed_standalone_calibration_cost'][k]+exe[k])
    total=exe['actual_input_tokens']+exe['actual_output_tokens']+p['observed_standalone_calibration_cost']['actual_total_tokens'];pd=1152+len(exejobs)-len(pjobs);td=total-baseline['actual_input_tokens']-baseline['actual_output_tokens'];eq(entry['planned_calls_difference_vs_P'],pd);eq(entry['observed_tokens_difference_vs_P'],td);eq(entry['calls_amortized'],pd<=0);eq(entry['tokens_amortized'],td<=0)
 for task,refs in result['fixed_zero_calibration_references'].items():
  rels=sorted(r for r in vector if r.startswith(task+'/'))
  for label,config in [('P_baseline','singleton_execution'),('random_k8_C','k8/random/C'),('diversity_k8_C','k8/diversity/C'),('S_reference','singleton')]:
   eq(refs[label]['calibration_calls'],0)
   for m,prefix in enumerate(refs[label]['all_prefix_execution_costs'],1):
    for rep in (1,2):
     exe=execution_cost([j for j in jobs if j['relation_id'] in rels[:m] and j['repeat']==rep and j['configuration_id']==config]);entry=prefix['replays'][str(rep)]
     for key in exe:eq(entry['execution_cost'][key],exe[key])
 summary=load(f/f'run_{model}/summary.json');eq(summary['scientific_calls'],len(jobs));eq(summary['diagnostic_calls'],20);eq(summary['invalid_scientific_calls'],sum(deliberately_invalid(j) for j in jobs))
 for kind in ('input','output'):
  eq(summary['scientific_observed_tokens'][kind+'_tokens'],sum(expected_records[j['request_id']][kind+'_tokens'] for j in jobs));eq(summary['diagnostic_observed_tokens'][kind+'_tokens'],sum(expected_records[j['request_id']][kind+'_tokens'] for j in controls))
 # Corrupting an otherwise consistent canonical response must be rejected against the durable return.
 run=f/f'run_{model}';original=run/'responses.jsonl';tamper=run/'responses_TAMPER_TEST.jsonl';source=original.read_text();first,rest=source.split('\n',1);row=json.loads(first);row['raw']=row['raw'].replace('"Y"','"N"',1) if '"Y"' in row['raw'] else row['raw'].replace('"N"','"Y"',1);row['labels']={k:int(v=='Y') for k,v in json.loads(row['raw']).items()};tamper.write_text(compact(row)+'\n'+rest)
 oldoptional=analyzer.optional
 def injected(p):return lines(tamper) if p==original else oldoptional(p)
 rejected=False
 with patch.object(analyzer,'ROOT',f),patch.object(analyzer,'optional',injected):
  try:analyzer.load_trace(model)
  except AssertionError:rejected=True
 eq(rejected,True)
 return {'status':'PASS','model':model,'independent_equalities':equalities,'full_science_calls':len(jobs),'controls':20,'synthetic_invalid_calls':sum(deliberately_invalid(j) for j in jobs),'policies_checked':len(policies),'all_relation_replay_gold_fidelity_and_transport':True,'all_prefix_calibration_and_fixed_baseline_costs':True,'canonical_tamper_rejected_against_durable_journal':True,'actual_runner_sha256':sha(R/'run_gpu.py'),'actual_analyzer_sha256':sha(R/'analyze_transfer.py'),'fixture_root':str(f.relative_to(E)),'scientific_model_executed':False,'GPU_executed':False}
def main():
 start=time.time();reports=[]
 for model in ('qwen','smol'):
  x=model_test(model);write(E/f'09_FULL_PIPELINE_{model}.json',x);reports.append(x);print(model,x['independent_equalities'],'PASS',flush=True)
 write(E/'09_FULL_PIPELINE_SUMMARY.json',{'status':'PASS','models':reports,'elapsed_seconds':time.time()-start,'all_output_labels_are_CPU_SYNTHETIC':True})
if __name__=='__main__':main()
