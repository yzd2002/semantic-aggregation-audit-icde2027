#!/usr/bin/env python3
"""CPU-only feature selection, gold-perturbation check and exact two-model preflight."""
import argparse,collections,csv,importlib.metadata as metadata,math,os,platform,random,shutil,subprocess,sys,time
import numpy as np
from scipy.sparse import csr_matrix
from common import *

def load_features(source):
 data={t:{} for t in TASKS}; exclusions={}
 tables=[]
 for name in ('tableA.csv','tableB.csv'):
  with (source/'dblp_acm'/name).open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
  assert len({r['id'] for r in rows})==len(rows)
  tables.append({r['id']:{k:r[k] for k in FIELDS} for r in rows})
 for split in ('train','test'):
  result=[];bad=[]
  with (source/'dblp_acm'/f'{split}.csv').open(newline='',encoding='utf8') as f:
   for index,row in enumerate(csv.DictReader(f)):
    a,b=row.get('ltable_id'),row.get('rtable_id')
    reason=None
    if a not in tables[0] or b not in tables[1]:reason='nonjoinable'
    elif not tables[0][a]['title'].strip() or not tables[1][b]['title'].strip():reason='empty_title'
    if reason:bad.append({'source_index':index,'reason':reason});continue
    result.append({'task':'dblp_acm','source_split':split,'source_index':index,
     'source_id':f'{split}:{index:05d}:L{a}:R{b}','left_id':a,'right_id':b,
     'features':{'left':tables[0][a],'right':tables[1][b]}})
  data['dblp_acm'][split]=result;exclusions['dblp_acm/'+split]=bad
 for split in ('train','validation'):
  result=[];bad=[]
  for index,row in enumerate(read_lines(source/'boolq'/f'{split}.jsonl')):
   assert row['row_idx']==index
   if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('question','passage')):
    bad.append({'source_index':index,'reason':'malformed_or_empty_question_passage'});continue
   result.append({'task':'boolq','source_split':split,'source_index':index,'source_id':f'{split}:{index:05d}',
    'features':{k:row[k] for k in ('question','passage')}})
  data['boolq'][split]=result;exclusions['boolq/'+split]=bad
 return data,exclusions

def load_gold(source):
 """Only called after chosen features and prompt-selection signature are saved."""
 lookup={}
 for split in ('train','test'):
  with (source/'dblp_acm'/f'{split}.csv').open(newline='',encoding='utf8') as f:
   for index,row in enumerate(csv.DictReader(f)):
    key=('dblp_acm',f'{split}:{index:05d}:L{row["ltable_id"]}:R{row["rtable_id"]}')
    assert row['label'] in ('0','1');lookup[key]=int(row['label'])
 for split in ('train','validation'):
  for row in read_lines(source/'boolq'/f'{split}.jsonl'):
   assert type(row['answer']) is bool
   lookup['boolq',f'{split}:{row["row_idx"]:05d}']=int(row['answer'])
 return lookup

def mutate_gold(source,dest):
 dest.mkdir(exist_ok=False)
 for task in TASKS:(dest/task).mkdir()
 for name in ('tableA.csv','tableB.csv'):
  shutil.copyfile(source/'dblp_acm'/name,dest/'dblp_acm'/name)
 counts={}
 for split in ('train','test','valid'):
  path=source/'dblp_acm'/f'{split}.csv'
  with path.open(newline='',encoding='utf8') as f:
   reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
  for r in rows:r['label']=str(1-int(r['label']))
  with (dest/'dblp_acm'/f'{split}.csv').open('x',newline='',encoding='utf8') as f:
   writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
  counts['dblp_acm/'+split]=len(rows)
 for split in ('train','validation'):
  rows=read_lines(source/'boolq'/f'{split}.jsonl')
  for r in rows:r['answer']=not r['answer']
  write_lines(dest/'boolq'/f'{split}.jsonl',rows);counts['boolq/'+split]=len(rows)
 return counts

def add_lengths(data,tokenizers):
 result=[]
 for task in TASKS:
  for split,records in data[task].items():
   texts=[feature_string(r) for r in records]
   lengths={m:[] for m in MODELS}
   for m,tok in tokenizers.items():
    for start in range(0,len(texts),256):
     batch=tok(texts[start:start+256],add_special_tokens=False,truncation=False,padding=False)['input_ids']
     lengths[m].extend(map(len,batch))
   for i,r in enumerate(records):
    r['feature_sha256']=digest(texts[i]);r['token_lengths']={m:lengths[m][i] for m in MODELS}
    result.append({'task':task,'source_id':r['source_id'],'feature_sha256':r['feature_sha256'],'token_lengths':r['token_lengths']})
 return result

class LexicalIndex:
 """Raw-count TF times smoothed IDF, Unicode words, unigram/bigram, L2."""
 def __init__(self,bank):
  self.bank=bank;counts=[self.terms(feature_string(r)) for r in bank];df=collections.Counter()
  for c in counts:df.update(c.keys())
  self.vocabulary=sorted(df);self.position={w:i for i,w in enumerate(self.vocabulary)}
  self.idf=np.array([math.log((1+len(bank))/(1+df[w]))+1 for w in self.vocabulary])
  self.matrix=self.encode_counts(counts)
  self.audit={'documents':len(bank),'vocabulary_terms':len(self.vocabulary),
   'word_regex':r'(?u)\b\w+\b','lowercase':True,'ngram_range':[1,2],'tf':'raw count',
   'idf':'log((1+number_of_bank_records)/(1+document_frequency))+1','norm':'l2',
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

def length_prompts(task,target,contexts,tokenizers):
 lengths={}
 for m,tok in tokenizers.items():
  lengths[m]={}
  for context,background in contexts.items():
   lengths[m][context]={}
   for scope in SCOPES:
    system,user,_=prompt(task,[target]+background,scope)
    wrapped=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    lengths[m][context][scope]=len(tok.encode(wrapped,add_special_tokens=False))
 return lengths

def select(data,tokenizers):
 """This function accepts feature-only records; it never reads source labels."""
 target_rng=random.Random(SEEDS['target']);context_rng=random.Random(SEEDS['context'])
 chosen=[];reserves=[];banks={};report={};proposals=[]
 for task in TASKS:
  eligible={};excluded=[]
  for split,rows in data[task].items():
   eligible[split]=[]
   for r in rows:
    if any(n>512 for n in r['token_lengths'].values()):
     excluded.append({'source_id':r['source_id'],'source_split':split,'reason':'feature_over_512_tokens','token_lengths':r['token_lengths']})
    else:eligible[split].append(r)
  reserve=[];reserve_rejects=[];seen=set()
  if task=='dblp_acm':
   byleft=collections.defaultdict(list)
   for r in eligible['test']:byleft[r['left_id']].append(r)
   lefts=sorted(byleft);target_rng.shuffle(lefts)
   for left in lefts:
    candidates=sorted(byleft[left],key=lambda r:r['source_id']);target_rng.shuffle(candidates)
    selected=None
    for r in candidates:
     overlap=identities(r)&seen
     if overlap:reserve_rejects.append({'source_id':r['source_id'],'reason':'reserved_identity_overlap'});continue
     selected=r;break
    if selected is not None:reserve.append(selected);seen.update(identities(selected))
    if len(reserve)==256:break
  else:
   candidates=eligible['validation'][:];target_rng.shuffle(candidates)
   for r in candidates:
    if identities(r)&seen:reserve_rejects.append({'source_id':r['source_id'],'reason':'reserved_identity_overlap'});continue
    reserve.append(r);seen.update(identities(r))
    if len(reserve)==256:break
  bank=[];bank_rejections=[];fingerprints=set()
  for r in sorted(eligible['train'],key=lambda r:r['source_id']):
   fp=normalized_feature_fingerprint(r)
   reason='reserved_identity_overlap' if identities(r)&seen else 'duplicate_complete_features' if fp in fingerprints else None
   if reason:bank_rejections.append({'source_id':r['source_id'],'reason':reason});continue
   bank.append(r);fingerprints.add(fp)
  index=LexicalIndex(bank);banks[task]=bank
  bank_identities=[identities(r) for r in bank];bank_lengths={m:np.array([r['token_lengths'][m] for r in bank]) for m in MODELS}
  selected_count=0;feasibility=[]
  for reserve_ordinal,target in enumerate(reserve):
   reserves.append({'task':task,'reserve_ordinal':reserve_ordinal,'record':target})
   if selected_count>=128:
    feasibility.append({'source_id':target['source_id'],'reserve_ordinal':reserve_ordinal,'status':'unused_quota_reached'});continue
   scores=index.scores(target);ranks=sorted(range(len(bank)),key=lambda i:(-float(scores[i]),bank[i]['source_id']))
   near=[];used=set();rank_records=[]
   for rank,i in enumerate(ranks,1):
    if bank_identities[i]&used:continue
    near.append(i);used.update(bank_identities[i]);rank_records.append({'source_id':bank[i]['source_id'],'rank':rank,'similarity':float(scores[i])})
    if len(near)==3:break
   if len(near)!=3:
    feasibility.append({'source_id':target['source_id'],'reserve_ordinal':reserve_ordinal,'status':'insufficient_distinct_near'});continue
   near_records=[bank[i] for i in near]
   near_lengths=length_prompts(task,target,{'near':near_records},tokenizers)
   accepted=None;attempt_total=0;attempt_width={}
   for width in (16,32,64):
    pools=[]
    for i in near:
     mask=np.ones(len(bank),dtype=bool)
     for m in MODELS:mask&=np.abs(bank_lengths[m]-bank_lengths[m][i])<=width
     pools.append([k for k in np.flatnonzero(mask).tolist() if not bank_identities[k]&used])
    attempt_width[str(width)]=0
    for attempt in range(1,2001):
     attempt_total+=1;attempt_width[str(width)]+=1;rand=[];random_used=set();empty=False
     for pool in pools:
      remaining=[k for k in pool if not bank_identities[k]&random_used]
      if not remaining:empty=True;break
      k=context_rng.choice(remaining);rand.append(k);random_used.update(bank_identities[k])
     prop={'task':task,'target_id':target['source_id'],'width':width,'attempt_at_width':attempt,'proposal_ids':[bank[i]['source_id'] for i in rand]}
     if empty:
      prop['status']='empty_uniform_draw_pool';proposals.append(prop);break
     random_records=[bank[i] for i in rand]
     random_lengths=length_prompts(task,target,{'random':random_records},tokenizers)
     lengths={m:{**near_lengths[m],**random_lengths[m]} for m in MODELS}
     matched=all(abs(lengths[m]['near'][s]-lengths[m]['random'][s])<=24 for m in MODELS for s in SCOPES)
     context_ok=all(lengths[m][c][s]+128<=4096 for m in MODELS for c in CONTEXTS for s in SCOPES)
     prop['prompt_lengths']=lengths;prop['status']='accepted' if matched and context_ok else 'length_mismatch' if not matched else 'context_overflow'
     proposals.append(prop)
     if matched and context_ok:accepted=(rand,width,lengths);break
    if accepted is not None:break
   if accepted is None:
    feasibility.append({'source_id':target['source_id'],'reserve_ordinal':reserve_ordinal,'status':'no_length_matched_context','attempts_by_width':attempt_width});continue
   rand,width,lengths=accepted
   # Singleton cap is independently checked before accepting this target.
   singleton_lengths={}
   for m,tok in tokenizers.items():
    p=prepared_prompt(tok,task,[target],'singleton');singleton_lengths[m]=p['prompt_tokens_preflight']
   if any(v+128>4096 for v in singleton_lengths.values()):
    feasibility.append({'source_id':target['source_id'],'reserve_ordinal':reserve_ordinal,'status':'singleton_overflow'});continue
   item={'task':task,'target_id':target['source_id'],'reserve_ordinal':reserve_ordinal,'selected_ordinal':selected_count,
    'target':target,'contexts':{'near':near_records,'random':[bank[i] for i in rand]},
    'near_ranks':rank_records,'random_similarities':[float(scores[i]) for i in rand],
    'length_audit':{'width':width,'attempts_total':attempt_total,'attempts_by_width':attempt_width,'prompt_lengths':lengths,
     'singleton_prompt_lengths':singleton_lengths,'near_minus_random':{m:{s:lengths[m]['near'][s]-lengths[m]['random'][s] for s in SCOPES} for m in MODELS}}}
   chosen.append(item);selected_count+=1
   feasibility.append({'source_id':target['source_id'],'reserve_ordinal':reserve_ordinal,'status':'selected','selected_ordinal':selected_count-1})
  report[task]={'source_counts':{s:len(r) for s,r in data[task].items()},'eligible_counts':{s:len(r) for s,r in eligible.items()},
   'feature_length_exclusions':excluded,'reserve_count':len(reserve),'reserve_rejections':reserve_rejects,
   'bank_count':len(bank),'bank_rejections':bank_rejections,'lexical_index':index.audit,
   'selected_count':selected_count,'shortfall':128-selected_count,'feasibility':feasibility}
  print(task,'reserve',len(reserve),'bank',len(bank),'selected',selected_count,flush=True)
 return {'targets':chosen,'reserves':reserves,'banks':banks,'report':report,'proposals':proposals}

def selection_signature(result):
 prompts=[]
 for r in result['targets']:
  for c in CONTEXTS:
   for s in SCOPES:prompts.append(prompt(r['task'],[r['target']]+r['contexts'][c],s)[:2])
  prompts.append(prompt(r['task'],[r['target']],'singleton')[:2])
 return {'selection_sha256':digest(compact(result)),'unwrapped_prompts_sha256':digest(compact(prompts))}

def main():
 from transformers import AutoTokenizer
 out=ROOT/'selection';out.mkdir(exist_ok=False);prep=ROOT/'prepared';prep.mkdir(exist_ok=False)
 locations=json.loads((ROOT/'model_locations.json').read_text())
 tokenizers={m:AutoTokenizer.from_pretrained(locations[m],local_files_only=True,trust_remote_code=False) for m in MODELS}
 sources=ROOT/'sources'
 write_json(out/'SOURCE_FILE_SHA256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources.rglob('*')) if p.is_file()})
 write_json(out/'PRESELECTION_CODE_SHA256.json',{'created_unix':time.time(),'files':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'common.py',ROOT/'prepare.py',ROOT/'PROTOCOL_context_audit.md',ROOT/'IMPLEMENTATION_DETAILS.md']}})
 data,malformed=load_features(sources);lengths=add_lengths(data,tokenizers)
 write_lines(out/'all_feature_lengths.jsonl',lengths);write_json(out/'malformed_exclusions.json',malformed)
 print('Selecting features only; gold evaluation fields are not attached.',flush=True)
 result=select(data,tokenizers);sig=selection_signature(result)
 write_json(out/'FEATURE_ONLY_SELECTION_SIGNATURE.json',sig)
 mutation_root=ROOT/'gold_perturbation_sources';mutations=mutate_gold(sources,mutation_root)
 changed,changed_bad=load_features(mutation_root)
 bykey={(r['task'],r['source_id']):r for task in TASKS for rows in data[task].values() for r in rows}
 for task in TASKS:
  for rows in changed[task].values():
   for r in rows:
    original=bykey[r['task'],r['source_id']]
    assert r=={k:v for k,v in original.items() if k not in ('feature_sha256','token_lengths')}
    r['feature_sha256']=original['feature_sha256'];r['token_lengths']=original['token_lengths'].copy()
 assert changed_bad==malformed and changed==data
 changed_result=select(changed,tokenizers);changed_sig=selection_signature(changed_result)
 assert changed_sig==sig
 write_json(out/'GOLD_PERTURBATION_CHECK.json',{'status':'PASS','all_source_gold_labels_inverted':mutations,
  'original':sig,'perturbed':changed_sig,'selection_reexecuted':True,'feature_input_objects_identical':True,
  'mutated_source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(mutation_root.rglob('*')) if p.is_file()}})
 write_lines(out/'reserves.jsonl',result['reserves']);write_lines(out/'random_proposals.jsonl',result['proposals'])
 for task,bank in result['banks'].items():write_lines(out/f'{task}_background_bank.jsonl',bank)
 write_json(out/'selection_report.json',result['report'])
 # First and only scientific gold attachment, after label-blind selection and perturbation check.
 gold=load_gold(sources)
 for r in result['targets']:
  r['gold']=gold[r['task'],r['target_id']]
  r['background_gold']={c:[gold[r['task'],b['source_id']] for b in r['contexts'][c]] for c in CONTEXTS}
 write_lines(out/'targets.jsonl',result['targets'])
 schedule=[]
 for r in result['targets']:
  for scope in (*SCOPES,'singleton'):
   for context in (CONTEXTS if scope!='singleton' else ('none',)):
    for repeat in (1,2):
     schedule.append({'task':r['task'],'target_id':r['target_id'],'selected_ordinal':r['selected_ordinal'],
      'scope':scope,'context':context,'repeat':repeat,
      'request_id':f'{r["task"]}/{r["selected_ordinal"]:03d}/{scope}/{context}/{repeat}'})
 random.Random(SEEDS['schedule']).shuffle(schedule)
 for ordinal,job in enumerate(schedule):job.update(ordinal=ordinal,block=ordinal//64)
 write_lines(prep/'schedule.jsonl',schedule)
 write_json(prep/'block_membership.json',[[j['request_id'] for j in schedule[s:s+64]] for s in range(0,len(schedule),64)])
 bytarget={(r['task'],r['target_id']):r for r in result['targets']};preflight={};model_hashes={}
 write_json(prep/'PARSER_CHECKS.json',{'status':'PASS','checks':parser_checks()})
 for model,tok in tokenizers.items():
  jobs=[]
  for job in schedule:
   r=bytarget[job['task'],job['target_id']]
   records=[r['target']]+(r['contexts'][job['context']] if job['scope']!='singleton' else [])
   p=prepared_prompt(tok,job['task'],records,job['scope'])
   assert p['prompt_tokens_preflight']+128<=4096
   mapping={str(i+1):{'source_id':rec['source_id'],'source_split':rec['source_split'],'feature_sha256':rec['feature_sha256']} for i,rec in enumerate(records)}
   jobs.append({**job,**p,'id_map':mapping,'id_map_sha256':digest(compact(mapping))})
  write_lines(prep/f'{model}_requests.jsonl',jobs)
  lookup={(j['task'],j['target_id'],j['scope'],j['context'],j['repeat']):j for j in jobs}
  for r in result['targets']:
   keys=(r['task'],r['target_id'])
   prefixes=[];tokenprefixes=[]
   for c in CONTEXTS:
    ff=lookup[(*keys,'full-first',c,1)];fl=lookup[(*keys,'full-last',c,1)]
    assert ff['wrapped_prompt']==fl['wrapped_prompt'] and ff['prompt_token_ids']==fl['prompt_token_ids']
    for s in SCOPES:
     x=lookup[(*keys,s,c,1)];prefixes.append(x['target_input_prefix_sha256']);tokenprefixes.append(x['target_prefix_token_ids_sha256'])
   assert len(set(prefixes))==len(set(tokenprefixes))==1
  preflight[model]={'requests':len(jobs),'max_prompt_plus_cap':max(j['prompt_tokens_preflight']+128 for j in jobs),
   'request_sha256':sha(prep/f'{model}_requests.jsonl'),'target_prefix_checks':'PASS','full_first_last_byte_equality':'PASS'}
  print(model,'preflight',preflight[model],flush=True)
  print('Hashing model',model,flush=True)
  model_hashes[model]={'path':locations[model],'files':{p.name:sha(p) for p in sorted(Path(locations[model]).iterdir()) if p.is_file()}}
 write_json(prep/'MODEL_FILE_SHA256.json',model_hashes);write_json(prep/'BOTH_MODEL_CONTEXT_PREFLIGHT.json',preflight)
 checks=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
 (prep/'pip_check.log').write_text(checks.stdout+checks.stderr)
 cfg={'created_unix':time.time(),'environment_status':'Inherited same-machine runtime with known dependency conflicts and process-local singleton NCCL adapter; not clean-machine replication.',
  'python':platform.python_version(),'python_executable':sys.executable,'packages':{p:metadata.version(p) for p in PACKAGES},'pip_check_exit':checks.returncode,
  'max_model_len':4096,'max_tokens':128,'max_num_seqs':16,'block_requests':64,'dtype':'bfloat16','temperature':0,
  'enforce_eager':True,'tensor_parallel_size':1,'prefix_caching':False,'guided_backend':'xgrammar','disable_guidance_fallback':True,
  'gpu_memory_utilization':0.70,'seeds':SEEDS,'bootstrap_draws':4000,
  'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)}
 write_json(prep/'runtime_configuration.json',cfg)
 print('CPU preparation complete; no GPU inference has occurred. Await independent review and explicit root authorization.',flush=True)

if __name__=='__main__':main()
