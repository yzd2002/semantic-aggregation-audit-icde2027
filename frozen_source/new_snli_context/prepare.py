#!/usr/bin/env python3
"""CPU-only complete-field, three-tokenizer selection and frozen request preparation."""
from common import *
cpu_environment()
import collections,importlib.metadata as metadata,math,platform,random,subprocess,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

def load_features(source):
 data={};exclusions={}
 for split in ('test','train'):
  table=pq.read_table(source/f'{split}.parquet')
  assert table.num_rows==SOURCE_ROWS[split]
  result=[];bad=[]
  for index,row in enumerate(table.to_pylist()):
   source_id=f'{split}:{index:06d}'
   if type(row.get('label')) is not int or row['label'] not in (0,1,2):
    bad.append({'source_id':source_id,'reason':'no_valid_original_label_metadata'});continue
   if any(not isinstance(row.get(k),str) or not row[k].strip() for k in SOURCE_FIELDS):
    bad.append({'source_id':source_id,'reason':'malformed_or_empty_premise_hypothesis'});continue
   result.append({'task':'snli','source_split':split,'source_index':index,'source_id':source_id,
                  'features':{k:row[k] for k in SOURCE_FIELDS}})
  data[split]=result;exclusions[split]=bad
 return data,exclusions

def load_gold(source):
 """Called only after feature-only selection and the executed perturbation check."""
 result={}
 for split in ('test','train'):
  for index,label in enumerate(pq.read_table(source/f'{split}.parquet',columns=['label'])['label'].to_pylist()):
   if label in (0,1,2):result[f'{split}:{index:06d}']={'original_class':label,'binary':int(label==0)}
 return result

def mutate_gold(source,dest):
 dest.mkdir(exist_ok=False);counts={}
 for split in ('test','train'):
  table=pq.read_table(source/f'{split}.parquet')
  labels=table['label'].to_pylist();mutated=[(x+1)%3 if x in (0,1,2) else x for x in labels]
  changed=table.set_column(table.schema.get_field_index('label'),'label',pa.array(mutated,type=table.schema.field('label').type))
  pq.write_table(changed,dest/f'{split}.parquet')
  counts[split]={'all_rows':len(labels),'changed_valid_categories':sum(a!=b for a,b in zip(labels,mutated)),
                 'sha256':sha(dest/f'{split}.parquet')}
 return counts

def add_lengths(data,tokenizers,out):
 with out.open('x',encoding='utf8') as sink:
  for split,records in data.items():
   texts=[feature_string(r) for r in records]
   lengths={m:[] for m in MODELS}
   for model,tok in tokenizers.items():
    start_time=time.perf_counter()
    for start in range(0,len(texts),512):
     batch=tok(texts[start:start+512],add_special_tokens=False,truncation=False,padding=False)['input_ids']
     lengths[model].extend(map(len,batch))
     if start%65536==0 or start+512>=len(texts):
      print('lengths',split,model,min(start+512,len(texts)),len(texts),'seconds',round(time.perf_counter()-start_time,1),flush=True)
   for i,r in enumerate(records):
    r['feature_sha256']=digest(texts[i]);r['token_lengths']={m:lengths[m][i] for m in MODELS}
    sink.write(compact({'source_id':r['source_id'],'source_split':split,'feature_sha256':r['feature_sha256'],'token_lengths':r['token_lengths']})+'\n')

class LexicalIndex:
 """Restored audit TF-IDF contract, raw-count TF, bank-only smoothed IDF, float64."""
 def __init__(self,bank):
  counts=[self.terms(feature_string(r)) for r in bank];df=collections.Counter()
  for row in counts:df.update(row.keys())
  self.vocabulary=sorted(df);self.position={w:i for i,w in enumerate(self.vocabulary)}
  self.idf=np.array([math.log((1+len(bank))/(1+df[w]))+1 for w in self.vocabulary])
  self.matrix=self.encode_counts(counts)
  self.audit={'documents':len(bank),'vocabulary_terms':len(self.vocabulary),'word_regex':r'(?u)\b\w+\b',
   'lowercase':True,'ngram_range':[1,2],'tf':'raw count','idf':'log((1+n)/(1+df))+1','norm':'l2',
   'vocabulary_sha256':digest(compact(self.vocabulary)),'idf_sha256':digest(self.idf.tobytes()),
   'bank_ids_sha256':digest(compact([r['source_id'] for r in bank]))}
 @staticmethod
 def terms(text):
  words=re.findall(r'(?u)\b\w+\b',text.lower())
  return collections.Counter(['u:'+w for w in words]+['b:'+a+' '+b for a,b in zip(words,words[1:])])
 def encode_counts(self,rows):
  values=[];indices=[];indptr=[0]
  for row in rows:
   pairs=sorted((self.position[w],v) for w,v in row.items() if w in self.position)
   weights=np.array([count*self.idf[i] for i,count in pairs],dtype=np.float64)
   norm=float(np.linalg.norm(weights))
   if norm:weights/=norm
   values.extend(weights);indices.extend(i for i,v in pairs);indptr.append(len(values))
  return csr_matrix((values,indices,indptr),shape=(len(rows),len(self.vocabulary)),dtype=np.float64)
 def scores(self,target):return (self.matrix@self.encode_counts([self.terms(feature_string(target))]).T).toarray().ravel()

def length_prompts(target,contexts,tokenizers):
 result={m:{} for m in MODELS}
 for model,tok in tokenizers.items():
  for context,background in contexts.items():
   result[model][context]={}
   for scope in SCOPES:
    system,user,_=prompt('snli',[target]+background,scope)
    wrapped=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    result[model][context][scope]=len(tok.encode(wrapped,add_special_tokens=False,truncation=False))
 return result

def select(data,tokenizers):
 """This function receives no gold category or model predictions."""
 rng=random.Random(f'{SEED}|context');eligible={};length_exclusions=[]
 for split,records in data.items():
  eligible[split]=[]
  for r in records:
   if any(n>MAX_FEATURE_TOKENS for n in r['token_lengths'].values()):
    length_exclusions.append({'source_id':r['source_id'],'reason':'complete_features_over_512_in_at_least_one_tokenizer','token_lengths':r['token_lengths']})
   else:eligible[split].append(r)
 reserve=[];reserve_rejections=[];reserved_texts=set()
 for r in sorted(eligible['test'],key=lambda r:(hash_order('reserve',r['source_id']),r['source_id'])):
  if identities(r)&reserved_texts:
   reserve_rejections.append({'source_id':r['source_id'],'reason':'reserved_complete_text_overlap'});continue
  reserve.append(r);reserved_texts.update(identities(r))
  if len(reserve)==256:break
 if len(reserve)!=256:raise RuntimeError(f'Reserve shortfall: {len(reserve)} of required 256; inference forbidden')
 prebank=[];bank_rejections=[];fingerprints=set()
 for r in sorted(eligible['train'],key=lambda r:r['source_id']):
  fp=normalized_feature_fingerprint(r)
  reason='reserved_complete_text_overlap' if identities(r)&reserved_texts else 'duplicate_normalized_complete_pair' if fp in fingerprints else None
  if reason:bank_rejections.append({'source_id':r['source_id'],'reason':reason});continue
  prebank.append(r);fingerprints.add(fp)
 bank=sorted(prebank,key=lambda r:(hash_order('bank',r['source_id']),r['source_id']))[:10000]
 if len(bank)<6:raise RuntimeError('Insufficient background bank; inference forbidden')
 index=LexicalIndex(bank);bank_ids=[identities(r) for r in bank]
 bank_lengths={m:np.array([r['token_lengths'][m] for r in bank]) for m in MODELS}
 chosen=[];feasibility=[];proposals=[]
 for ordinal,target in enumerate(reserve):
  if len(chosen)==128:
   feasibility.append({'source_id':target['source_id'],'reserve_ordinal':ordinal,'status':'unused_quota_reached'});continue
  scores=index.scores(target);ranked=sorted(range(len(bank)),key=lambda i:(-float(scores[i]),bank[i]['source_id']))
  near=[];used=set();near_ranks=[]
  for rank,i in enumerate(ranked,1):
   if bank_ids[i]&used:continue
   near.append(i);used.update(bank_ids[i]);near_ranks.append({'source_id':bank[i]['source_id'],'rank':rank,'similarity':float(scores[i])})
   if len(near)==3:break
  if len(near)<3:
   feasibility.append({'source_id':target['source_id'],'reserve_ordinal':ordinal,'status':'insufficient_distinct_near'});continue
  near_records=[bank[i] for i in near]
  near_lengths=length_prompts(target,{'near':near_records},tokenizers)
  accepted=None;attempt_total=0;attempts={}
  for width in (16,32,64):
   pools=[]
   for i in near:
    mask=np.ones(len(bank),dtype=bool)
    for model in MODELS:mask&=np.abs(bank_lengths[model]-bank_lengths[model][i])<=width
    pools.append([k for k in np.flatnonzero(mask).tolist() if not bank_ids[k]&used])
   attempts[str(width)]=0
   for attempt in range(1,2001):
    attempt_total+=1;attempts[str(width)]+=1;rand=[];random_used=set();empty=False
    for pool in pools:
     remaining=[k for k in pool if not bank_ids[k]&random_used]
     if not remaining:empty=True;break
     k=rng.choice(remaining);rand.append(k);random_used.update(bank_ids[k])
    proposal={'target_id':target['source_id'],'width':width,'attempt_at_width':attempt,'proposal_ids':[bank[i]['source_id'] for i in rand]}
    if empty:
     proposal['status']='empty_uniform_draw_pool';proposals.append(proposal);break
    other_lengths=length_prompts(target,{'random':[bank[i] for i in rand]},tokenizers)
    lengths={m:{**near_lengths[m],**other_lengths[m]} for m in MODELS}
    matched=all(abs(lengths[m]['near'][s]-lengths[m]['random'][s])<=24 for m in MODELS for s in SCOPES)
    fits=all(lengths[m][c][s]+MAX_OUTPUT_TOKENS<=MAX_MODEL_LEN for m in MODELS for c in CONTEXTS for s in SCOPES)
    proposal.update(prompt_lengths=lengths,status='accepted' if matched and fits else 'length_mismatch' if not matched else 'context_overflow')
    proposals.append(proposal)
    if matched and fits:accepted=(rand,width,lengths);break
   if accepted is not None:break
  if accepted is None:
   feasibility.append({'source_id':target['source_id'],'reserve_ordinal':ordinal,'status':'no_matched_context','attempts_by_width':attempts});continue
  rand,width,lengths=accepted
  singletons={m:prepared_prompt(tok,'snli',[target],'singleton')['prompt_tokens_preflight'] for m,tok in tokenizers.items()}
  if any(n+MAX_OUTPUT_TOKENS>MAX_MODEL_LEN for n in singletons.values()):
   feasibility.append({'source_id':target['source_id'],'reserve_ordinal':ordinal,'status':'singleton_overflow'});continue
  item={'task':'snli','target_id':target['source_id'],'reserve_ordinal':ordinal,'selected_ordinal':len(chosen),
   'target':target,'contexts':{'near':near_records,'random':[bank[i] for i in rand]},'near_ranks':near_ranks,
   'random_similarities':[float(scores[i]) for i in rand],
   'length_audit':{'width':width,'attempts_total':attempt_total,'attempts_by_width':attempts,'prompt_lengths':lengths,
    'singleton_prompt_lengths':singletons,'near_minus_random':{m:{s:lengths[m]['near'][s]-lengths[m]['random'][s] for s in SCOPES} for m in MODELS}}}
  chosen.append(item);feasibility.append({'source_id':target['source_id'],'reserve_ordinal':ordinal,'status':'selected','selected_ordinal':len(chosen)-1})
 if len(chosen)!=128:raise RuntimeError(f'Target shortfall {len(chosen)} of 128; inference forbidden')
 report={'source_metadata_valid_counts':{s:len(r) for s,r in data.items()},'feature_length_eligible_counts':{s:len(r) for s,r in eligible.items()},
  'feature_length_exclusions':length_exclusions,'reserve_count':len(reserve),'reserve_rejections':reserve_rejections,
  'bank_pre_cap_count':len(prebank),'bank_count':len(bank),'bank_cap':10000,'bank_rejections':bank_rejections,
  'bank_cap_hash_domain':f'{SEED}|bank|source_id','lexical_index':index.audit,'selected_count':len(chosen),'shortfall':0,'feasibility':feasibility}
 print('feature_only_select','reserve',len(reserve),'bank_pre_cap',len(prebank),'bank',len(bank),'selected',len(chosen),flush=True)
 return {'targets':chosen,'reserves':[{'reserve_ordinal':i,'record':r} for i,r in enumerate(reserve)],
  'bank':bank,'report':report,'proposals':proposals}

def selection_signature(result):
 prompts=[]
 for r in result['targets']:
  for context in CONTEXTS:
   for scope in SCOPES:prompts.append(prompt('snli',[r['target']]+r['contexts'][context],scope)[:2])
  prompts.append(prompt('snli',[r['target']],'singleton')[:2])
 return {'selection_sha256':digest(compact(result)),'all_unwrapped_prompt_pairs_sha256':digest(compact(prompts))}

def schedules(targets):
 jobs=[]
 for row in targets:
  for scope in (*SCOPES,'singleton'):
   for context in (CONTEXTS if scope!='singleton' else ('none',)):
    for repeat in (1,2):
     jobs.append({'task':'snli','target_id':row['target_id'],'selected_ordinal':row['selected_ordinal'],
      'scope':scope,'context':context,'repeat':repeat,'execution_mode':'mixed',
      'request_id':f'snli/{row["selected_ordinal"]:03d}/{scope}/{context}/{repeat}'})
 random.Random(f'{SEED}|schedule').shuffle(jobs)
 for i,j in enumerate(jobs):j.update(ordinal=i,block=i//64)
 serial=[dict(j) for j in jobs if j['selected_ordinal']<32]
 random.Random(f'{SEED}|serial').shuffle(serial)
 for i,j in enumerate(serial):
  j.update(source_main_request_id=j['request_id'],request_id='serial/'+j['request_id'],ordinal=i,block=i,execution_mode='serial')
 assert len(jobs)==1792 and len(serial)==448
 return jobs,serial

def synthetic_records(panel,context):
 panels=[[
  {'premise':'A woman is running in a park.','hypothesis':'A person is running outdoors.'},
  {'premise':'A boy is eating an apple.','hypothesis':'A child is eating fruit.'},
  {'premise':'A dog is sleeping.','hypothesis':'An animal is awake and jumping.'},
  {'premise':'Two people are talking.','hypothesis':'The two people are siblings.'}],
  [{'premise':'A cat is sleeping on a mat.','hypothesis':'A dog is running.'},
  {'premise':'A man is playing a guitar.','hypothesis':'Someone is making music.'},
  {'premise':'A child is riding a bicycle.','hypothesis':'The child is wearing a red helmet.'},
  {'premise':'The room is empty.','hypothesis':'Several people are sitting in the room.'}]]
 records=panels[panel]
 if context=='random':records=[records[0],*reversed(records[1:])]
 return [{'features':r} for r in records]

def diagnostics(tokenizer):
 messages=[{'role':'system','content':'Follow the user instruction exactly.'},{'role':'user','content':'Reply with only the word BANANA.'}]
 wrapped=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 tokens=tokenizer.encode(wrapped,add_special_tokens=False,truncation=False)
 assert tokens==tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
 base={'wrapped_prompt':wrapped,'prompt_token_ids':tokens,'prompt_tokens_preflight':len(tokens),'prompt_sha256':digest(wrapped),
       'execution_mode':'diagnostic','max_tokens':128}
 result=[{**base,'request_id':'diagnostic/ordinary_banana','output_ids':[],'requested_ids':[],'grammar':None,'guided':False}]
 for i,ids in enumerate(([1],[1,2,3,4],[2,3,4,1],[9,10,11],[11])):
  result.append({**base,'request_id':f'diagnostic/guided_banana/{i}','output_ids':ids,'requested_ids':ids,'grammar':grammar(ids),
                 'grammar_sha256':digest(grammar(ids)),'guided':True})
 for panel in (0,1):
  for scope in (*SCOPES,'singleton'):
   for context in (CONTEXTS if scope!='singleton' else ('none',)):
    records=synthetic_records(panel,context)
    if scope=='singleton':records=records[:1]
    p=prepared_prompt(tokenizer,'snli',records,scope)
    result.append({**p,'request_id':f'diagnostic/smoke/{panel}/{scope}/{context}','task':'snli','scope':scope,
                   'context':context,'synthetic_panel':panel,'guided':True,'execution_mode':'diagnostic','max_tokens':128})
 assert len(result)==20 and sum(len(j['output_ids']) for j in result)==51
 for i,j in enumerate(result):j.update(ordinal=i,block=0 if i==0 else 1 if i<6 else 2)
 return result

def main():
 started=time.time()
 assert (ROOT/'sources/SOURCE_LICENSE_SCHEMA_VERIFIED.json').is_file()
 out=ROOT/'selection';out.mkdir(exist_ok=False)
 prep=ROOT/'prepared';prep.mkdir(exist_ok=False)
 code=[ROOT/x for x in ('common.py','prepare.py','PROTOCOL_snli_context.md','verify_sources.py','snapshot_tokenizers.py')]
 write_json(out/'PRESELECTION_CODE_SHA256.json',{'created_unix':started,'files':{p.name:sha(p) for p in code}})
 tokenizers=load_tokenizers();data,bad=load_features(ROOT/'sources')
 write_json(out/'metadata_field_exclusions.json',bad)
 add_lengths(data,tokenizers,out/'all_feature_lengths.jsonl')
 result=select(data,tokenizers);signature=selection_signature(result)
 write_json(out/'FEATURE_ONLY_SELECTION_SIGNATURE.json',signature)
 mutated=ROOT/'gold_perturbation_sources';mutation_report=mutate_gold(ROOT/'sources',mutated)
 changed,changed_bad=load_features(mutated)
 assert changed_bad==bad
 for split,records in changed.items():
  assert len(records)==len(data[split])
  for r,original in zip(records,data[split]):
   assert r=={k:v for k,v in original.items() if k not in ('feature_sha256','token_lengths')}
   r['feature_sha256']=original['feature_sha256'];r['token_lengths']=original['token_lengths'].copy()
 changed_result=select(changed,tokenizers);changed_signature=selection_signature(changed_result)
 assert changed_signature==signature
 original_main,original_serial=schedules(result['targets'])
 changed_main,changed_serial=schedules(changed_result['targets'])
 assert original_main==changed_main and original_serial==changed_serial
 write_json(out/'GOLD_PERTURBATION_CHECK.json',{'status':'PASS','mutation':'0→1→2→0 for every valid label; -1 unchanged',
  'sources':mutation_report,'selection_reexecuted':True,'fresh_feature_objects_verified_identical':True,
  'original':signature,'perturbed':changed_signature,'main_schedule_identical':True,'serial_schedule_identical':True})
 write_lines(out/'reserves.jsonl',result['reserves']);write_lines(out/'background_bank.jsonl',result['bank'])
 write_lines(out/'random_proposals.jsonl',result['proposals']);write_json(out/'selection_report.json',result['report'])
 # First evaluation attachment after label-blind selection and invariance verification.
 gold=load_gold(ROOT/'sources')
 for r in result['targets']:
  r.update(gold=gold[r['target_id']]['binary'],original_gold_class=gold[r['target_id']]['original_class'])
  r['background_gold']={c:[gold[b['source_id']]['binary'] for b in r['contexts'][c]] for c in CONTEXTS}
  r['background_original_gold_class']={c:[gold[b['source_id']]['original_class'] for b in r['contexts'][c]] for c in CONTEXTS}
 write_lines(out/'targets.jsonl',result['targets'])
 targets={r['target_id']:r for r in result['targets']}
 write_lines(prep/'schedule.jsonl',original_main);write_lines(prep/'serial_schedule.jsonl',original_serial)
 write_json(prep/'block_membership.json',[[j['request_id'] for j in original_main[s:s+64]] for s in range(0,1792,64)])
 write_json(prep/'serial_block_membership.json',[[j['request_id']] for j in original_serial])
 write_json(prep/'PARSER_CHECKS.json',{'status':'PASS','checks':parser_checks()})
 preflight={};ledger=[];costs={}
 for model,tok in tokenizers.items():
  jobs=[]
  for job in original_main:
   r=targets[job['target_id']];records=[r['target']]+(r['contexts'][job['context']] if job['scope']!='singleton' else [])
   p=prepared_prompt(tok,'snli',records,job['scope'])
   assert p['prompt_tokens_preflight']+128<=4096
   mapping={str(i+1):{'source_id':rec['source_id'],'source_split':rec['source_split'],'feature_sha256':rec['feature_sha256']} for i,rec in enumerate(records)}
   jobs.append({**job,**p,'max_tokens':128,'id_map':mapping,'id_map_sha256':digest(compact(mapping))})
  lookup={j['request_id']:j for j in jobs}
  serial=[{**lookup[j['source_main_request_id']],**j} for j in original_serial]
  controls=diagnostics(tok)
  for r in result['targets']:
   prefixes=[];token_prefixes=[]
   for c in CONTEXTS:
    ff=lookup[f'snli/{r["selected_ordinal"]:03d}/full-first/{c}/1'];fl=lookup[f'snli/{r["selected_ordinal"]:03d}/full-last/{c}/1']
    assert ff['wrapped_prompt']==fl['wrapped_prompt'] and ff['prompt_token_ids']==fl['prompt_token_ids']
    for s in SCOPES:
     p=lookup[f'snli/{r["selected_ordinal"]:03d}/{s}/{c}/1']
     prefixes.append(p['target_input_prefix_sha256']);token_prefixes.append(p['target_prefix_token_ids_sha256'])
     assert p['id_map']['1']['source_id']==r['target_id']
   assert len(set(prefixes))==len(set(token_prefixes))==1
  for phase,rows,expected in [('main',jobs,1792),('serial',serial,448),('diagnostic',controls,20)]:
   assert len(rows)==expected
   write_lines(prep/f'{model}_{phase}_requests.jsonl',rows)
   for j in rows:
    ledger.append({'call_id':model+'/'+j['request_id'],'model':model,'phase':phase,'request_id':j['request_id'],
     'planned_input_tokens':len(j['prompt_token_ids']),'max_output_tokens':128,'requested_YN_label_slots':len(j['output_ids']),
     'prompt_sha256':j['prompt_sha256'],'grammar_sha256':j.get('grammar_sha256'),
     'state':'not_submitted','observed_input_tokens':None,'observed_output_tokens':None,'raw_output':None})
  costs[model]={phase:{'calls':len(rows),'planned_input_tokens':sum(len(j['prompt_token_ids']) for j in rows),
      'requested_YN_label_slots':sum(len(j['output_ids']) for j in rows),'maximum_output_tokens':128*len(rows)}
     for phase,rows in [('main',jobs),('serial',serial),('diagnostic',controls)]}
  assert costs[model]['main']['requested_YN_label_slots']==4864
  assert costs[model]['serial']['requested_YN_label_slots']==1216
  preflight[model]={'main_requests':len(jobs),'serial_requests':len(serial),'diagnostic_requests':len(controls),
   'maximum_main_prompt_tokens':max(j['prompt_tokens_preflight'] for j in jobs),
   'max_prompt_plus_cap':max(j['prompt_tokens_preflight']+128 for j in jobs+serial+controls),
   'max_near_random_prompt_difference':max(abs(r['length_audit']['near_minus_random'][model][s]) for r in result['targets'] for s in SCOPES),
   'target_id_position_and_prefix':'PASS','full_first_last_byte_and_token_equality':'PASS','serial_copy_identity':'PASS',
   'main_request_sha256':sha(prep/f'{model}_main_requests.jsonl')}
  print('preflight',model,compact(preflight[model]),flush=True)
 assert len(ledger)==6780 and len({j['call_id'] for j in ledger})==6780
 write_lines(prep/'planned_call_ledger.jsonl',ledger)
 write_json(prep/'PLANNED_COSTS.json',{'models':costs,'totals':{
  'scientific_calls':sum(x[p]['calls'] for x in costs.values() for p in ('main','serial')),
  'diagnostic_calls':sum(x['diagnostic']['calls'] for x in costs.values()),
  'scientific_requested_label_slots':sum(x[p]['requested_YN_label_slots'] for x in costs.values() for p in ('main','serial')),
  'diagnostic_requested_label_slots':sum(x['diagnostic']['requested_YN_label_slots'] for x in costs.values()),
  'planned_input_tokens_all':sum(p['planned_input_tokens'] for x in costs.values() for p in x.values())},
  'observed_inference_tokens':0,'observed_scientific_calls':0,'observed_diagnostic_calls':0})
 write_json(prep/'THREE_MODEL_CONTEXT_PREFLIGHT.json',preflight)
 counts=collections.Counter(r['original_gold_class'] for r in result['targets'])
 write_json(prep/'GOLD_MAPPING_AND_QUALITY_PLAN.json',{'labels_attached_after_feature_selection_and_perturbation':True,
  'class_names':{'0':'entailment','1':'neutral','2':'contradiction'},'binary_mapping':{'0':1,'1':0,'2':0},
  'selected_original_class_counts':dict(counts),'selected_binary_reference_count':sum(r['gold'] for r in result['targets']),
  'metrics':['confusion','error_rate','F1','balanced_accuracy','all_Y','all_N','singleton','wrong_to_right','right_to_wrong'],
  'annotator_vote_quality':'Unavailable in this Parquet distribution; no confidence filtering',
  'COUNT_relations':{'main':{'denominator':128,'target_ordinals':'0..127','backgrounds_included':False},
                     'serial':{'denominator':32,'target_ordinals':'0..31','backgrounds_included':False}},
  'COUNT_missing_policy':'abstain per condition or pair whenever any selected relation target is invalid/missing; denominator stays fixed'})
 write_json(prep/'ALPHA_PLAN.json',{'primary_family':'three new SNLI model full-first replacement-minus-replay contrasts',
  'models':list(MODELS),'confidence':0.9875,'alpha_per_contrast':0.0125,'bonferroni_sum_bound':0.0375,
  'quantiles':[0.00625,0.99375],'positive_primary_evidence':'98.75% percentile interval lower endpoint > 0',
  'old_and_new_tasks_unified_FWER_claim':False,'bootstrap_draws':4000,'bootstrap_seed':SEED,
  'bootstrap_scope':'approximate descriptive intervals on selected targets; no guaranteed population or causal coverage',
  'secondary_confidence':0.95})
 versions={p:metadata.version(p) for p in PACKAGES}
 check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
 (prep/'pip_check.log').write_text(check.stdout+check.stderr)
 write_json(prep/'runtime_configuration.json',{'python':platform.python_version(),'python_executable':sys.executable,
  'packages':versions,'pip_check_exit':check.returncode,'max_model_len':4096,'max_tokens':128,'max_num_seqs':16,
  'block_requests':64,'dtype':'bfloat16','temperature':0,'enforce_eager':True,'tensor_parallel_size':1,
  'prefix_caching':False,'guided_backend':'xgrammar','disable_guidance_fallback':True,'gpu_memory_utilization':0.70,
  'seed':SEED,'cpu_cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
  'runtime_status':'CPU-only preparation; GPU compatibility and explicit authorization still pending'})
 write_json(prep/'PREINFERENCE_STATUS.json',{'status':'NO_GPU_GO','scientific_calls_attempted':0,'diagnostic_calls_attempted':0,
  'models_loaded':0,'cuda_used':False,'targets':128,'reserve':256,'background_bank':len(result['bank']),
  'cpu_preparation_seconds':time.time()-started,'next_gate':'independent CPU review and explicit root inference authorization'})
 print('CPU preparation complete; zero model judgments, NO_GPU_GO',flush=True)

if __name__=='__main__':main()
