#!/usr/bin/env python3
"""Independent CPU reconstruction. Does not import any producer module/math."""
import collections,csv,hashlib,json,math,os,random,re,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit')
E=Path(__file__).resolve().parent
B=R.parent;D=B/'reconstruction/decision_audit';C=B/'recovered/context_audit'
EXPECTED='3349e50072791650512c7708ebf1c54a37ed2012b0a72cf785730666b3e2b38c'
TASKS=('dblp_acm','boolq');MODELS=('qwen','smol');FIELDS=('title','authors','venue','year')
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def h(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def sha(p):
 q=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(8<<20),b''):q.update(z)
 return q.hexdigest()
def lines(p):
 with Path(p).open() as f:return [json.loads(t) for t in f if t.strip()]
def load(p):return json.loads(Path(p).read_text())
def save(n,x):
 p=E/n
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def norm(x):return ' '.join(x.casefold().split())
def fp(r):
 f=r['features'];v=[[norm(f[s][k]) for k in FIELDS] for s in ('left','right')] if r['task']=='dblp_acm' else [norm(f[k]) for k in ('question','passage')]
 return h(compact(v))
def identities(r):
 if r['task']=='dblp_acm':
  f=r['features'];ids={('citation',h(compact([norm(f[s][k]) for k in FIELDS]))) for s in ('left','right')}
  ids.update((k,r[k]) for k in ('left_id','right_id') if k in r)
  return ids
 return {(k,norm(r['features'][k])) for k in ('question','passage')}
def parse(raw,ids):
 if not isinstance(raw,str):return None
 pattern=r'\{'+','.join(re.escape('"'+str(i)+'":')+'"[YN]"' for i in ids)+r'\}'
 if re.fullmatch(pattern,raw) is None:return None
 ps=json.loads(raw,object_pairs_hook=list)
 if [k for k,v in ps]!=list(map(str,ids)):return None
 return {int(k):int(v=='Y') for k,v in ps}
def freeze():
 assert sha(R/'PREINFERENCE_FROZEN_SHA256.json')==EXPECTED
 f=load(R/'PREINFERENCE_FROZEN_SHA256.json');assert len(f['files'])==95
 for name,digest in f['files'].items():assert sha(R/name)==digest,name
 actual=set(p.name for p in R.glob('*.py'))|{'POLICY_LOCK.json','model_locations.json'}
 for d in ('source_reference','sources','inputs','prepared','provenance','docs','verification','runtime_override'):
  actual.update(str(p.relative_to(R)) for p in (R/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('INDEPENDENT_REVIEW.json','GPU_RELEASE.json'))
 assert actual==set(f['files'])
 assert not (R/'GPU_RELEASE.json').exists()
 assert all(not (R/f'run_{m}').exists() for m in MODELS)
 return {'files':len(actual),'sha256':EXPECTED,'GPU_release_absent':True,'run_directories_absent':True}
def policy_reconstruction():
 lock=load(R/'POLICY_LOCK.json');assert sha(R/'POLICY_LOCK.json')==sha(D/'POLICY_LOCK.json')
 policies=lock['payload']['policies'];assert len(policies)==36
 out=[];manifests={}
 for model in MODELS:
  req=lines(D/'prepared'/f'{model}_dev_requests.jsonl');s=lines(D/'prepared/dev_schedule.jsonl')
  assert len(req)==len(s) and all(all(j[k]==v for k,v in base.items()) for j,base in zip(req,s))
  byid={j['request_id']:j for j in req};raw=lines(D/f'run_dev_{model}/responses.jsonl');ret={}
  v=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(lambda:[None]*128)))
  subs=lines(D/f'run_dev_{model}/block_submissions.jsonl')
  assert [x for b in subs for x in b['request_ids']]==[j['request_id'] for j in req]
  for row in raw:
   rid=row['request_id'];assert rid not in ret;j=byid[rid]
   assert all(row[k]==j[k] for k in s[0] if k in row)
   assert row['actual_input_tokens']==len(j['prompt_token_ids'])
   assert row['input_token_ids_sha256']==h(compact(j['prompt_token_ids']))
   assert row['actual_output_tokens']==len(row['output_token_ids'])
   parsed=parse(row['raw'],j['output_ids']);assert row['strict_valid']==(parsed is not None)
   assert parsed==({int(k):v for k,v in row['labels'].items()} if row['labels'] is not None else None)
   for i in j['occurrence_ids']:
    dst=v[j['task']][j['configuration_id']][j['repeat']];assert dst[i-1] is None
    dst[i-1]=parsed[i] if parsed is not None else None
   ret[rid]=row
  assert set(ret)==set(byid)
  manifests[model]={'responses_sha256':sha(D/f'run_dev_{model}/responses.jsonl'),'requests_sha256':sha(D/'prepared'/f'{model}_dev_requests.jsonl'),'calls':len(raw)}
  for p in [p for p in policies if p['model']==model]:
   task=p['task'];selector=p['selector'];tau=Fraction(p['tolerance']);types={'Input':('C','I'),'Output':('C','O'),'Repeat':('C',)}[selector];nr=4 if selector=='Repeat' else 2
   reference=v[task]['singleton'];led=[]
   reference_ok=all(len(reference[r])==128 and all(type(x) is int and x in (0,1) for x in reference[r]) for r in (1,2))
   for k in (2,4,8):
    for group in ('random','diversity'):
     for typ in types:
      cid=f'k{k}/{group}/{typ}';vectors=v[task][cid]
      full=all(len(vectors[r])==128 and all(type(x) is int and x in (0,1) for x in vectors[r]) for r in range(1,nr+1))
      errors=sum(a!=b for r in range(1,nr+1) for a,b in zip(vectors[r],reference[1+(r-1)%2])) if full and reference_ok else None
      led.append({'configuration_id':cid,'k':k,'grouping':group,'execution_type':typ,'observations':nr,'complete_candidate':full,'complete_reference':reference_ok,'disagreements':errors,'fixed_denominator':128*nr,'raw_disagreement':errors/(128*nr) if errors is not None else None,'eligible':errors is not None and errors*tau.denominator<=tau.numerator*128*nr,'one_execution_calls':128//k})
   assert led==p['eligibility_ledger'],(model,task,selector,p['tolerance'])
   eligible=[x for x in led if x['eligible']]
   if eligible:
    b=min(eligible,key=lambda q:(q['one_execution_calls'],Fraction(q['disagreements'],q['fixed_denominator']),('random','diversity').index(q['grouping']),q['execution_type']!='C'))
    chosen={k:b[k] for k in ('configuration_id','k','grouping','execution_type','one_execution_calls')};chosen['reason']='eligible_min_calls_then_exact_loss_then_fixed_tie'
   else:chosen={'configuration_id':'singleton','k':1,'grouping':'none','execution_type':'S','one_execution_calls':128,'reason':'no_eligible_batch' if reference_ok else 'reference_calibration_unavailable'}
   assert chosen==p['chosen']
   cj=[j for j in req if j['task']==task and (j['execution_type']=='S' or (selector=='Repeat' and j['execution_type']=='C') or (selector!='Repeat' and j['repeat']<=2 and j['execution_type'] in types))]
   cost={'planned_calls':len(cj),'returned_calls':len(cj),'planned_requested_items':sum(len(j['occurrence_ids']) for j in cj),'actual_input_tokens':sum(ret[j['request_id']]['actual_input_tokens'] for j in cj),'actual_output_tokens':sum(ret[j['request_id']]['actual_output_tokens'] for j in cj),'request_ids_sha256':h(compact([j['request_id'] for j in cj]))}
   cost['actual_total_tokens']=cost['actual_input_tokens']+cost['actual_output_tokens']
   assert cost==p['observed_standalone_calibration_cost'] and cost['planned_calls']==1152 and cost['planned_requested_items']==3328
   out.append({'policy':'/'.join(p[k] for k in ('model','task','tolerance','selector')),'chosen':chosen,'eligibility_rows':len(led),'calibration_cost':cost})
 return policies,{'status':'PASS','policies':out,'raw_manifests':manifests,'producer_modules_imported':False}
def sources_and_history(tokenizers):
 data={t:[] for t in TASKS};gold={};tables=[]
 for name in ('tableA.csv','tableB.csv'):
  with (R/'sources/dblp_acm'/name).open(newline='') as f:rs=list(csv.DictReader(f))
  tables.append({r['id']:{k:r[k] for k in FIELDS} for r in rs});assert len(tables[-1])==len(rs)
 with (R/'sources/dblp_acm/test.csv').open(newline='') as f:
  for i,r in enumerate(csv.DictReader(f)):
   a,b=r['ltable_id'],r['rtable_id'];sid=f'test:{i:05d}:L{a}:R{b}'
   if a not in tables[0] or b not in tables[1] or not tables[0][a]['title'].strip() or not tables[1][b]['title'].strip():continue
   data['dblp_acm'].append({'task':'dblp_acm','source_split':'test','source_index':i,'source_id':sid,'left_id':a,'right_id':b,'features':{'left':tables[0][a],'right':tables[1][b]}});gold['dblp_acm',sid]=int(r['label'])
 for i,r in enumerate(lines(R/'sources/boolq/validation.jsonl')):
  assert i==r['row_idx'];sid=f'validation:{i:05d}'
  if any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('question','passage')):continue
  data['boolq'].append({'task':'boolq','source_split':'validation','source_index':i,'source_id':sid,'features':{k:r[k] for k in ('question','passage')}});gold['boolq',sid]=int(r['answer'])
 lengths={};flat=[r for t in TASKS for r in data[t]]
 for model,tok in tokenizers.items():
  for start in range(0,len(flat),128):
   rr=flat[start:start+128];batch=tok([compact(r['features']) for r in rr],add_special_tokens=False,truncation=False,padding=False)['input_ids']
   for r,ids in zip(rr,batch):lengths.setdefault((r['task'],r['source_id']),{})[model]=len(ids)
 assert {(r['task'],r['source_id']):r['lengths'] for r in lines(R/'inputs/all_feature_lengths.jsonl')}==lengths
 knownid={t:set() for t in TASKS};knownfp={t:set() for t in TASKS};knownidentity={t:set() for t in TASKS};exposed=set();histmanifest=[]
 def add(r):
  t=r['task'];knownfp[t].add(fp(r));knownidentity[t].update(identities(r));exposed.add((t,r.get('source_id'),fp(r),tuple(sorted(identities(r)))))
  if r.get('source_id'):knownid[t].add(r['source_id'])
 def prompt_records(text):
  for line in text.splitlines():
   if not line.startswith('{'):continue
   try:o=json.loads(line)
   except json.JSONDecodeError:continue
   if not isinstance(o,dict) or not isinstance(o.get('features'),dict):continue
   f=o['features'];t='dblp_acm' if set(f)=={'left','right'} else 'boolq' if set(f)=={'question','passage'} else None
   assert t is not None
   yield {'task':t,'features':f}
 for row in lines(C/'selection/targets.jsonl'):
  add(row['target'])
  for rr in row['contexts'].values():
   for r in rr:add(r)
 for r in lines(D/'inputs/relations.jsonl'):
  if r['task']=='dblp_acm':r={**r,**dict(zip(('left_id','right_id'),re.fullmatch(r'(?:test|train|valid):\d+:L(.+):R(.+)',r['source_id']).groups()))}
  add(r)
 for p in sorted((D/'prepared').glob('*_requests.jsonl'))+sorted((D/'protocol_history').rglob('*_requests.jsonl')):
  n=0
  with p.open() as f:
   for line in f:
    j=json.loads(line)
    for r in prompt_records(j.get('unwrapped_user',j.get('wrapped_prompt',''))):assert fp(r) in knownfp[r['task']];n+=1
  histmanifest.append({'path':str(p),'sha256':sha(p),'record_occurrences':n})
 for p in sorted(C.glob('run_*/diagnostics.jsonl'))+sorted(D.glob('run_*/diagnostics.jsonl')):
  for row in lines(p):
   for r in prompt_records(row.get('wrapped_prompt','')):add(r)
 savedexposed=set((r['task'],r.get('source_id'),fp(r),tuple(tuple(i) for i in r['identities'])) for r in lines(R/'inputs/historical_exposed_records.jsonl'))
 assert exposed==savedexposed
 report={};chosen=[];eligibleall=[]
 for task in TASKS:
  seen=set();eligible=[]
  for r in sorted(data[task],key=lambda r:r['source_id']):
   if max(lengths[task,r['source_id']].values())>512 or r['source_id'] in knownid[task] or fp(r) in knownfp[task] or identities(r)&knownidentity[task] or identities(r)&seen:continue
   eligible.append({**r,'feature_sha256':h(compact(r['features'])),'normalized_feature_sha256':fp(r),'token_lengths':lengths[task,r['source_id']]});seen.update(identities(r))
  assert len(eligible)=={'dblp_acm':420,'boolq':2571}[task]
  eligibleall.extend(eligible);ranked=sorted(eligible,key=lambda r:(h(compact([2026110210,task,r['source_id']])),r['source_id']));nr=min(20,len(eligible)//128)
  for i,r in enumerate(ranked[:nr*128]):chosen.append({**r,'phase':'transfer','relation_id':f'{task}/{i//128:02d}','relation_index':i//128,'occurrence_id':1+i%128,'reserve_ordinal':None})
  report[task]={'source_records':len(data[task]),'historical_source_ids':len(knownid[task]),'historical_identity_count':len(knownidentity[task]),'eligible':len(eligible),'relations':nr,'ranked_source_ids_sha256':h(compact([r['source_id'] for r in ranked]))}
 assert eligibleall==lines(R/'inputs/eligible_records.jsonl');assert chosen==lines(R/'inputs/relations.jsonl')
 goldrows=[{k:r[k] for k in ('task','relation_id','occurrence_id','source_id')}|{'gold':gold[r['task'],r['source_id']]} for r in chosen]
 assert goldrows==lines(R/'inputs/gold.jsonl')
 return chosen,{'status':'PASS','tasks':report,'historical_unique_exposure_tuples':len(exposed),'scientific_prompt_files_independently_read':histmanifest,'complete_fields_and_public_gold_join':True,'selection_uses_no_gold':True}
def tfidf_parts(rows):
 parts=[];audits={};bases={}
 def terms(text):
  w=re.findall(r'(?u)\b\w+\b',text.lower());return collections.Counter(['u:'+x for x in w]+['b:'+a+' '+b for a,b in zip(w,w[1:])])
 for task in TASKS:
  bank=lines(R/'sources'/f'{task}_background_bank.jsonl');counts=[terms(compact(r['features'])) for r in bank];df=collections.Counter()
  for cnt in counts:df.update(cnt.keys())
  vocab=sorted(df);pos={w:i for i,w in enumerate(vocab)};idf=np.array([math.log((len(bank)+1)/(df[w]+1))+1 for w in vocab])
  audits[task]={'documents':len(bank),'vocabulary_terms':len(vocab),'vocabulary_sha256':h(compact(vocab)),'idf_sha256':h(idf.tobytes()),'bank_ids_sha256':h(compact([r['source_id'] for r in bank]))}
  for key,value in audits[task].items():assert load(R/'inputs/LEXICAL_INDEX_AUDIT.json')[task][key]==value
  for rel in sorted({r['relation_id'] for r in rows if r['task']==task}):
   rr=sorted([r for r in rows if r['relation_id']==rel],key=lambda r:r['occurrence_id']);vals=[];cols=[];ip=[0]
   for r in rr:
    terms_here=terms(compact(r['features']));ij=sorted((pos[w],n) for w,n in terms_here.items() if w in pos)
    v=np.array([n*idf[i] for i,n in ij],dtype=np.float64);den=np.linalg.norm(v)
    if den:v=v/den
    vals.extend(v);cols.extend(i for i,n in ij);ip.append(len(vals))
   mat=csr_matrix((vals,cols,ip),shape=(128,len(vocab)));cos=(mat@mat.T).toarray()
   seed=h(compact([2026110207,'random',task,'test']));perm=list(range(1,129));random.Random(int(seed,16)).shuffle(perm)
   bases[rel]={'seed_sha256':seed,'permutation':perm,'salt':'test'}
   ranks={i:h(compact([2026110207,'diversity-anchor',task,'test',i])) for i in range(1,129)}
   for k in (2,4,8):
    for grp in ('random','diversity'):
     if grp=='random':batches=[perm[start:start+k] for start in range(0,128,k)]
     else:
      available=set(range(1,129));batches=[]
      while available:
       i=min(available,key=lambda i:(ranks[i],i));available.remove(i);b=[i]
       while available and len(b)<k:
        i=min(available,key=lambda i:(max(float(cos[i-1,j-1]) for j in b),i));available.remove(i);b.append(i)
       batches.append(b)
     parts.append({'task':task,'relation_id':rel,'k':k,'grouping':grp,'batches':batches,'batch_membership_sha256':h(compact(batches)),'mean_within_batch_cosine':float(np.mean([cos[i-1,j-1] for b in batches for ix,i in enumerate(b) for j in b[ix+1:]]))})
 assert bases==load(R/'inputs/RANDOM_BASE_PERMUTATIONS.json')
 actual=lines(R/'inputs/partitions.jsonl');assert len(parts)==len(actual)==138
 for a,b in zip(parts,actual):
  assert all(a[k]==b[k] for k in a if k!='mean_within_batch_cosine');assert abs(a['mean_within_batch_cosine']-b['mean_within_batch_cosine'])<1e-12
 return parts,{'status':'PASS','partitions':len(parts),'independently_rebuilt_bank_IDF':audits,'groupingseed':2026110207,'literal_phase_salt':'test','membership_exact_match':True,'mean_cosine_absolute_tolerance':1e-12}
def plans_and_requests(rows,parts,policies,tokenizers):
 byrec={(r['relation_id'],r['occurrence_id']):r for r in rows};pb={(p['relation_id'],p['k'],p['grouping']):p for p in parts};allroles={};alljobs={};report={}
 systems={
 'dblp_acm':"Decide whether each pair of bibliographic citations refers to the same publication. Use only that pair's own title, authors, venue, and year fields. Other records are not additional evidence for it. Output Y for the same publication and N for different publications. Judge only the requested record IDs. Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.",
 'boolq':'Answer each yes/no question using only its own passage. Other records are not additional evidence for it. Output Y if the answer is yes and N if the answer is no. Judge only the requested record IDs. Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.'}
 for model in MODELS:
  allroles[model]={};jobs=[]
  for task in TASKS:
   roles=collections.defaultdict(list)
   for p in policies:
    if (p['model'],p['task'])==(model,task):roles['singleton_execution' if p['chosen']['configuration_id']=='singleton' else p['chosen']['configuration_id']].append('policy/'+'/'.join(p[k] for k in ('model','task','tolerance','selector')))
   roles['k8/random/C'].append('fixed/random_k8_C');roles['k8/diversity/C'].append('fixed/diversity_k8_C');roles['singleton_execution'].append('independent_P_baseline');roles['singleton'].append('common_S_reference');allroles[model][task]=dict(roles)
   for rel in sorted({r['relation_id'] for r in rows if r['task']==task}):
    for config,role in sorted(roles.items()):
     if config.startswith('singleton'):k=1;grp='none';typ='S' if config=='singleton' else 'P';batches=[[i] for i in range(1,129)];membership=None
     else:
      ks,grp,typ=config.split('/');k=int(ks[1:]);p=pb[rel,k,grp];batches=p['batches'];membership=p['batch_membership_sha256']
     for rep in (1,2):
      for ix,ids in enumerate(batches):jobs.append({'phase':'transfer','model':model,'task':task,'relation_id':rel,'configuration_id':config,'k':k,'grouping':grp,'execution_type':typ,'repeat':rep,'batch_index':ix,'occurrence_ids':sorted(ids),'roles':role,'batch_membership_sha256':membership,'request_id':f'transfer/{model}/{rel}/{config}/r{rep}/b{ix:03d}'})
  random.Random(2026110211).shuffle(jobs)
  for i,j in enumerate(jobs):j.update(ordinal=i,block=i//64)
  assert jobs==lines(R/'prepared'/f'{model}_schedule.jsonl');alljobs[model]=jobs
  total=maximum=0;singleton={};coverage=collections.defaultdict(list);bad=[]
  with (R/'prepared'/f'{model}_requests.jsonl').open() as f:
   for index,line in enumerate(f):
    j=json.loads(line);base=jobs[index];assert all(j[k]==v for k,v in base.items());rel=j['relation_id'];ids=sorted(j['occurrence_ids']);typ=j['execution_type'];t=j['task']
    ordered=ids[::-1] if typ=='I' else ids;output=ids[::-1] if typ=='O' else ids
    user='Records:\n'+'\n'.join(compact({'id':str(i),'features':byrec[rel,i]['features']}) for i in ordered)+'\nRequested IDs: '+compact(ids)+'\nReturn one Y or N value for each requested ID.'
    grammar=r'\{'+','.join(re.escape('"'+str(i)+'":')+'"[YN]"' for i in output)+r'\}'
    assert j['system']==systems[t] and j['unwrapped_user']==user and j['input_ids_in_order']==ordered and j['requested_ids']==ids and j['output_ids']==output and j['grammar']==grammar
    wrapped=tokenizers[model].apply_chat_template([{'role':'system','content':systems[t]},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    assert wrapped==j['wrapped_prompt'] and h(wrapped)==j['prompt_sha256'] and h(grammar)==j['grammar_sha256']
    tokenids=tokenizers[model].encode(wrapped,add_special_tokens=False);assert tokenids==j['prompt_token_ids'] and len(tokenids)==j['prompt_tokens_preflight'];total+=len(tokenids);maximum=max(maximum,len(tokenids))
    assert len(tokenids)+128<=4096
    assert j['id_map_sha256']==h(compact(j['id_map']))
    for i in ids:
     r=byrec[rel,i];want={k:r[k] for k in ('task','phase','source_id','reserve_ordinal','feature_sha256')};want['relation_id']=rel;assert j['id_map'][str(i)]==want
    coverage[rel,j['configuration_id'],j['repeat']]+=ids
    if typ in ('S','P'):
     sig=(wrapped,grammar,compact(j['id_map']));key=(rel,ids[0]);assert key not in singleton or singleton[key]==sig;singleton[key]=sig
   assert index+1==len(jobs)
  assert all(sorted(v)==list(range(1,129)) for v in coverage.values())
  assert len(jobs)=={'qwen':15552,'smol':13248}[model]
  for rel,i in byrec:
   assert all((rel,c,r) in coverage for c in ('singleton','singleton_execution') for r in (1,2))
  controls=lines(R/'prepared'/f'{model}_controls.jsonl');assert len(controls)==20
  for j in controls:
   assert tokenizers[model].encode(j['wrapped_prompt'],add_special_tokens=False)==j['prompt_token_ids'];assert h(j['wrapped_prompt'])==j['prompt_sha256']
   if j.get('guided'):
    pattern=r'\{'+','.join(re.escape('"'+str(i)+'":')+'"[YN]"' for i in j['output_ids'])+r'\}'
    assert j['grammar']==pattern
    for letter in ('Y','N'):assert len(tokenizers[model].encode(compact({str(i):letter for i in j['output_ids']}),add_special_tokens=False))<=128
  report[model]={'scientific_calls':len(jobs),'requested_science_item_judgments':sum(len(j['occurrence_ids']) for j in jobs),'planned_science_input_tokens':total,'max_prompt_plus_cap':maximum+128,'controls':20,'all_prepared_prompts_retokenized':True,'P1_P2_S1_S2_separate_request_ids_but_identical_prompts':True}
 assert allroles==load(R/'prepared/ROLE_PLAN_MAP.json')
 assert h(compact({'relations':rows,'partitions':parts,'schedules':alljobs}))==load(R/'inputs/FEATURE_ONLY_PLAN_HASH.json')['sha256']
 return {'status':'PASS','models':report,'science_calls':sum(len(v) for v in alljobs.values()),'control_calls':40}
def boundaries_and_overlap(rows,policies):
 doc=load(R/'prepared/POLICY_DESIGN_BOUNDARY.json');arithmetic=[]
 for p in policies:
  nr=3 if p['task']=='dblp_acm' else 20;k=p['chosen']['k'];expected=[{'M':m,'calibration_calls':1152,'execution_calls':128//k*m,'P_baseline_calls':128*m,'calibration_once_plus_execution_minus_P':1152+128//k*m-128*m} for m in range(1,nr+1)]
  entry=next(q for q in doc['all36'] if all(q[x]==p[x] for x in ('model','task','tolerance','selector')))
  assert entry['all_prefixes']==expected and entry['any_prefix_call_amortization']==any(x['calibration_once_plus_execution_minus_P']<=0 for x in expected)
  if p['tolerance'] in ('0.05','0.10'):assert not entry['any_prefix_call_amortization']
  arithmetic.append({'policy':'/'.join(p[k] for k in ('model','task','tolerance','selector')),'min_prefix_call_delta':min(x['calibration_once_plus_execution_minus_P'] for x in expected)})
 pairs=[]
 for model in MODELS:
  for t in TASKS:
   for tau in ('0.05','0.10','0.20'):
    q=[p for p in policies if (p['model'],p['task'],p['tolerance'])==(model,t,tau)];a=next(p for p in q if p['selector']=='Input');b=next(p for p in q if p['selector']=='Output');pairs.append(a['chosen']['configuration_id']==b['chosen']['configuration_id'])
 assert len(pairs)==12 and sum(pairs)==8
 old={(r['task'],r['target_id']):r for r in lines(C/'selection/targets.jsonl')};new=lines(B/'new_glm_context/selection/targets.jsonl');allnew=[];changed=[];groups=0
 for row in new:
  prev=old[row['task'],row['target_id']]['contexts']['random'];now=row['contexts']['random'];delta=[(a['source_id'],fp(a))!=(b['source_id'],fp(b)) for a,b in zip(prev,now)]
  if any(delta):
   groups+=1
   for i,r in enumerate(now):allnew.append((row['target_id'],i,r))
  for i,ok in enumerate(delta):
   if ok:changed.append((row['target_id'],i,now[i]))
 overlap=load(R/'prepared/CROSS_STUDY_FEATURE_OVERLAP.json');assert groups==34 and len(allnew)==102 and len(changed)==100
 for name,background in [('all_slots_in_redrawn_contexts_overlap',allnew),('changed_slots_overlap',changed)]:
  matches=[];types=collections.Counter();touched=set()
  for p in rows:
   for target,slot,r in background:
    if p['task']!=r['task']:continue
    ids=identities(p)&identities(r);sameid=p['source_id']==r['source_id'];samefp=fp(p)==fp(r)
    if ids or sameid or samefp:
     touched.add((p['task'],p['source_id']));types.update(k for k,v in ids);matches.append({'task':p['task'],'panel_relation_id':p['relation_id'],'panel_source_id':p['source_id'],'glm_context_target_id':target,'glm_background_source_id':r['source_id'],'slot':slot,'same_source_id':sameid,'same_complete_normalized_features':samefp,'overlapping_identities':[list(x) for x in sorted(ids)]})
  assert overlap[name]=={'unique_panel_records_with_any_overlap':len(touched),'pair_count_including_reused_background_slots':len(matches),'overlapping_identity_type_counts':dict(types),'pairs':matches}
 return {'status':'PASS','all36_prefix_call_arithmetic':arithmetic,'identical_Input_Output_pairs':8,'total_Input_Output_pairs':12,'primary_05_and_secondary_10_cannot_amortize':True,'redrawn_context_groups':groups,'changed_background_slots':len(changed),'overlap_unique_transfer_records':overlap['all_slots_in_redrawn_contexts_overlap']['unique_panel_records_with_any_overlap'],'overlap_pairs':overlap['all_slots_in_redrawn_contexts_overlap']['pair_count_including_reused_background_slots'],'shared_source_independence_caution':True}
def main():
 start=time.time();report={'schema_version':1,'scope':'Independent CPU-only pre-inference scientific review; no model generated or GPU initialized.'}
 report['freeze_start']=freeze();save('01_FREEZE.json',report['freeze_start']);print('freeze PASS',flush=True)
 policies,report['policies']=policy_reconstruction();save('02_POLICIES.json',report['policies']);print('36 policies rederived from raw PASS',flush=True)
 from transformers import AutoTokenizer
 loc=load(R/'model_locations.json');tok={m:AutoTokenizer.from_pretrained(loc[m],local_files_only=True,trust_remote_code=False) for m in MODELS}
 rows,report['selection']=sources_and_history(tok);save('03_SELECTION.json',report['selection']);print('source/history/selection/gold PASS',flush=True)
 parts,report['partitions']=tfidf_parts(rows);save('04_PARTITIONS.json',report['partitions']);print('138 independent partitions PASS',flush=True)
 report['requests']=plans_and_requests(rows,parts,policies,tok);save('05_REQUESTS.json',report['requests']);print('all schedules and actual tokenizer prompts PASS',flush=True)
 report['boundaries']=boundaries_and_overlap(rows,policies);save('06_BOUNDARIES_OVERLAP.json',report['boundaries']);print('call arithmetic and overlap PASS',flush=True)
 report['freeze_end']=freeze();report['elapsed_seconds']=time.time()-start;report['status']='PASS';save('INDEPENDENT_CORE_RESULTS.json',report)
 print('Independent core PASS',flush=True)
if __name__=='__main__':main()
