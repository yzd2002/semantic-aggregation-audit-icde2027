"""Portable contracts; no model imports or gold access in feature helpers."""
from pathlib import Path
import hashlib,json,re

ROOT=Path(__file__).resolve().parent
TASKS=('dblp_acm','boolq')
MODELS=('qwen','smol')
SCOPES=('full-first','target-only','full-last')
CONTEXTS=('near','random')
SEEDS={'target':2026110201,'context':2026110202,'schedule':2026110203,'bootstrap':2026110204}
PACKAGES=('torch','vllm','transformers','numpy','scipy','tokenizers','xgrammar','outlines_core','setuptools','pandas','pyarrow')
FIELDS=('title','authors','venue','year')
SYSTEMS={
 'dblp_acm':('Decide whether each pair of bibliographic citations refers to the same publication. '
  'Use only that pair\'s own title, authors, venue, and year fields. Other records are not additional evidence for it. '
  'Output Y for the same publication and N for different publications. Judge only the requested record IDs. '
  'Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.'),
 'boolq':('Answer each yes/no question using only its own passage. Other records are not additional evidence for it. '
  'Output Y if the answer is yes and N if the answer is no. Judge only the requested record IDs. '
  'Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.')}

def compact(obj):return json.dumps(obj,ensure_ascii=False,separators=(',',':'))
def digest(value):return hashlib.sha256(value if isinstance(value,bytes) else value.encode('utf8')).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def read_lines(path):
 with Path(path).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def write_lines(path,rows):
 with Path(path).open('x',encoding='utf8') as f:
  for r in rows:f.write(compact(r)+'\n')
def normalize(s):return ' '.join(s.casefold().split())
def citation_fingerprint(citation):return digest(compact([normalize(citation[k]) for k in FIELDS]))
def feature_string(record):return compact(record['features'])
def identities(record):
 """Observed source-qualified IDs plus cross-role full feature identities."""
 if record['task']=='dblp_acm':
  return {('left_id',record['left_id']),('right_id',record['right_id']),
   ('citation',citation_fingerprint(record['features']['left'])),('citation',citation_fingerprint(record['features']['right']))}
 return {('question',normalize(record['features']['question'])),('passage',normalize(record['features']['passage']))}
def normalized_feature_fingerprint(record):
 if record['task']=='dblp_acm':
  return digest(compact([[normalize(record['features'][side][f]) for f in FIELDS] for side in ('left','right')]))
 return digest(compact([normalize(record['features'][f]) for f in ('question','passage')]))
def requested_ids(scope):return [1] if scope in ('target-only','singleton') else [1,2,3,4]
def output_ids(scope):return [2,3,4,1] if scope=='full-last' else requested_ids(scope)
def grammar(ids):return r'\{'+','.join(re.escape('"'+str(i)+'":')+'"[YN]"' for i in ids)+r'\}'
def parse(raw,ids):
 if not isinstance(raw,str) or re.fullmatch(grammar(ids),raw) is None:return None
 pairs=json.loads(raw,object_pairs_hook=list)
 if [int(k) for k,v in pairs]!=list(ids) or len({k for k,v in pairs})!=len(ids):return None
 return {int(k):v for k,v in pairs}
def prompt(task,records,scope):
 lines=[compact({'id':str(i+1),'features':r['features']}) for i,r in enumerate(records)]
 user='Records:\n'+'\n'.join(lines)+'\nRequested IDs: '+compact(requested_ids(scope))+'\nReturn one Y or N value for each requested ID.'
 return SYSTEMS[task],user,lines[0]
def prepared_prompt(tokenizer,task,records,scope):
 system,user,target_line=prompt(task,records,scope)
 wrapped=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
 ids=tokenizer.encode(wrapped,add_special_tokens=False)
 user_start=wrapped.index(user);start=user_start+len('Records:\n');end=start+len(target_line)
 enc=tokenizer(wrapped,add_special_tokens=False,return_offsets_mapping=True)
 assert enc['input_ids']==ids
 prefix_n=sum(1 for a,b in enc['offset_mapping'] if b<=end)
 prefix=wrapped[:end]
 return {'system':system,'unwrapped_user':user,'wrapped_prompt':wrapped,'prompt_token_ids':ids,
  'prompt_tokens_preflight':len(ids),'requested_ids':requested_ids(scope),'output_ids':output_ids(scope),
  'grammar':grammar(output_ids(scope)),'prompt_sha256':digest(wrapped),'grammar_sha256':digest(grammar(output_ids(scope))),
  'target_input_span_chars':[start,end],'target_input_span_bytes':[len(wrapped[:start].encode()),len(wrapped[:end].encode())],
  'target_input_line':target_line,'target_input_prefix':prefix,'target_input_prefix_sha256':digest(prefix),
  'target_prefix_token_count':prefix_n,'target_prefix_token_ids_sha256':digest(compact(ids[:prefix_n])),
  'fixed_output_prefix_before_target':'{"1":"' if scope!='full-last' else None}
def parser_checks():
 examples=[('{"1":"Y"}',[1],True),('{"1":"Y","2":"N","3":"Y","4":"N"}',[1,2,3,4],True),
 ('{"2":"Y","3":"N","4":"Y","1":"N"}',[2,3,4,1],True),
 ('{"9":"Y","10":"N","11":"Y"}',[9,10,11],True),('{"11":"N"}',[11],True),
 (' {"1":"Y"}',[1],False),('{"1":"Y"}\n',[1],False),('{"1":"Y","1":"N"}',[1],False),
 ('{"1": "Y"}',[1],False),('{"1":"Y","2":"N"}',[1],False),('{"1":"yes"}',[1],False),
 ('{"2":"N","1":"Y"}',[1,2],False),('{"1":"Y"}',[1,2],False),('{"1":"Y"',[1],False)]
 rows=[{'raw':raw,'ids':ids,'expected_valid':valid,'actual_valid':parse(raw,ids) is not None} for raw,ids,valid in examples]
 assert all(x['expected_valid']==x['actual_valid'] for x in rows)
 return rows
