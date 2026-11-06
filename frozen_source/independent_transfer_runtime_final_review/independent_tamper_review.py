"""Independent CPU-only adversarial receipt replay; no producer math imported."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,shutil,time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(t) for t in p.read_text().splitlines() if t.strip()]
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def rows(p,x):p.write_text(''.join(json.dumps(t,separators=(',',':'))+'\n' for t in x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 spec=importlib.util.spec_from_file_location('independently_probed_runtime_verifier',a.root/'verify_effective_runtime.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 cases=[('clean',None,False),('raw_stale_hash','raw',False),('raw_rehashed','raw',True),('labels_rehashed','labels',True),('strict_valid_rehashed','strict_valid',True),('tokens_rehashed','output_token_ids',True),('finish_rehashed','finish_reason',True),('stop_rehashed','stop_reason',True),('output_cost_rehashed','actual_output_tokens',True),('input_cost_rehashed','actual_input_tokens',True),('canonical_missing','missing',True),('canonical_duplicate','duplicate',True),('diagnostic_raw_rehashed','diagnostic_raw',True),('summary_journal_hash','summary_journal_hash',False),('summary_response_hash','summary_response_hash',False),('summary_valid_calls','summary_valid_calls',False),('summary_all_calls','summary_all_calls',False),('summary_science_tokens','summary_science_tokens',False),('summary_control_tokens','summary_control_tokens',False),('summary_total_tokens','summary_total_tokens',False),('receipt_bytes','receipt_bytes',False),('native_journal_bytes','native_journal_bytes',False)]
 result=[]
 for name,mutation,rehashed in cases:
  root=a.out/name;shutil.copytree(a.fixture,root,symlinks=True);out=root/'run_qwen';s=load(out/'summary.json');r=lines(out/'responses.jsonl');diag=lines(out/'diagnostics.jsonl')
  if mutation=='raw':r[0]['raw']='CPU_SYNTHETIC_CORRUPTED_RAW'
  elif mutation=='labels':r[0]['labels']={str(k):1-int(val) for k,val in r[0]['labels'].items()}
  elif mutation=='strict_valid':r[0]['strict_valid']=False
  elif mutation=='output_token_ids':r[0]['output_token_ids']=[999];r[0]['actual_output_tokens']=1
  elif mutation=='finish_reason':r[0]['finish_reason']='CPU_SYNTHETIC_BAD_FINISH'
  elif mutation=='stop_reason':r[0]['stop_reason']='CPU_SYNTHETIC_BAD_STOP'
  elif mutation=='actual_output_tokens':r[0]['actual_output_tokens']=999999
  elif mutation=='actual_input_tokens':r[0]['actual_input_tokens']=999999
  elif mutation=='missing':r.pop()
  elif mutation=='duplicate':r.append(copy.deepcopy(r[-1]))
  elif mutation=='diagnostic_raw':diag[0]['raw']='CPU_SYNTHETIC_CORRUPTED_CONTROL';rows(out/'diagnostics.jsonl',diag)
  elif mutation=='summary_journal_hash':s['engine_return_journal_sha256']='0'*64
  elif mutation=='summary_response_hash':s['response_sha256']='0'*64
  elif mutation=='summary_valid_calls':s['valid_scientific_calls']+=1
  elif mutation=='summary_all_calls':s['all_calls']+=1
  elif mutation=='summary_science_tokens':s['scientific_observed_tokens']['output_tokens']+=1
  elif mutation=='summary_control_tokens':s['diagnostic_observed_tokens']['input_tokens']+=1
  elif mutation=='summary_total_tokens':s['all_observed_tokens']['output_tokens']+=1
  elif mutation=='receipt_bytes':
   z=lines(out/'effective_return_bindings.jsonl');z[-1]['bindings'][0]['engine_request_id']='CPU_BAD_EID';rows(out/'effective_return_bindings.jsonl',z)
  elif mutation=='native_journal_bytes':
   z=lines(out/'engine_returns.jsonl');z[-1]['actual_returns'][0]['outputs'][0]['raw']='CPU_BAD_RAW';rows(out/'engine_returns.jsonl',z)
  if mutation in {'raw','labels','strict_valid','output_token_ids','finish_reason','stop_reason','actual_output_tokens','actual_input_tokens','missing','duplicate'}:
   rows(out/'responses.jsonl',r)
  if rehashed:s['response_sha256']=sha(out/'responses.jsonl')
  if mutation is not None:write(out/'summary.json',s)
  if mutation=='diagnostic_raw':
   control=load(out/'CONTROL_RESULT.json');control['diagnostic_sha256']=sha(out/'diagnostics.jsonl');write(out/'CONTROL_RESULT.json',control)
   data=load(out/'DATA_START.json');data['control_sha256']=sha(out/'CONTROL_RESULT.json');write(out/'DATA_START.json',data)
  error=None
  try:v.verify(root,'qwen');accepted=True
  except BaseException as e:accepted=False;error=type(e).__name__+': '+str(e)
  result.append({'case':name,'canonical_summary_response_hash_recomputed':rehashed,'accepted':accepted,'exception':error,'status':'PASS' if accepted==(mutation is None) else 'FAIL'})
 report={'status':'PASS' if all(t['status']=='PASS' for t in result) else 'FAIL','test_count':len(result),'producer_verifier_sha256':sha(a.root/'verify_effective_runtime.py'),'created_unix':time.time(),'all_fixtures_CPU_SYNTHETIC':True,'actual_GPU_model_or_math_module_import':False,'cases':result};write(a.out/'RESULT.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
