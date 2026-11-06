"""Actual-relation batching contracts. Pure standard library; gold is separate."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent
TASKS=('dblp_acm','boolq');MODELS=('qwen','smol');PHASES=('dev','test')
KS=(2,4,8);GROUPS=('random','diversity');TYPES=('C','I','O')
SEEDS={'grouping':2026110207,'dev_schedule':2026110208,'test_schedule':2026110209}
SYSTEMS={
 'dblp_acm':('Decide whether each pair of bibliographic citations refers to the same publication. '
  'Use only that pair\'s own title, authors, venue, and year fields. Other records are not additional evidence for it. '
  'Output Y for the same publication and N for different publications. Judge only the requested record IDs. '
  'Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.'),
 'boolq':('Answer each yes/no question using only its own passage. Other records are not additional evidence for it. '
  'Output Y if the answer is yes and N if the answer is no. Judge only the requested record IDs. '
  'Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.')}
PACKAGES=('torch','vllm','transformers','numpy','scipy','tokenizers','xgrammar','outlines_core','setuptools','pandas','pyarrow')

def compact(obj):return json.dumps(obj,ensure_ascii=False,separators=(',',':'))
def digest(value):return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read_lines(path):
 with Path(path).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def write_json(path,obj):
 with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_lines(path,rows):
 with Path(path).open('x',encoding='utf8') as f:
  for row in rows:f.write(compact(row)+'\n')
def feature_string(record):return compact(record['features'])
def config_id(k,group,kind):return f'k{k}/{group}/{kind}'
def repeats(phase,kind):return range(1,5) if phase=='dev' and kind=='C' else range(1,3)
def grammar(ids):
 assert ids and len(ids)==len(set(ids)) and all(type(x) is int and x>0 for x in ids)
 return r'\{'+','.join(re.escape('"'+str(i)+'":')+'"[YN]"' for i in ids)+r'\}'
def parse(raw,ids):
 if not isinstance(raw,str) or re.fullmatch(grammar(ids),raw) is None:return None
 pairs=json.loads(raw,object_pairs_hook=list)
 if [int(k) for k,v in pairs]!=list(ids) or len({k for k,v in pairs})!=len(ids):return None
 return {int(k):int(v=='Y') for k,v in pairs}
def render(task,records,kind):
 ordered=sorted(records,key=lambda r:r['occurrence_id'],reverse=kind=='I')
 requested=sorted(r['occurrence_id'] for r in records);outputs=list(reversed(requested)) if kind=='O' else requested
 user='Records:\n'+'\n'.join(compact({'id':str(r['occurrence_id']),'features':r['features']}) for r in ordered)
 user+='\nRequested IDs: '+compact(requested)+'\nReturn one Y or N value for each requested ID.'
 return {'system':SYSTEMS[task],'unwrapped_user':user,'input_ids_in_order':[r['occurrence_id'] for r in ordered],
  'requested_ids':requested,'output_ids':outputs,'grammar':grammar(outputs)}
def prepare_prompt(tokenizer,task,records,kind):
 p=render(task,records,kind)
 wrapped=tokenizer.apply_chat_template([{'role':'system','content':p['system']},{'role':'user','content':p['unwrapped_user']}],tokenize=False,add_generation_prompt=True)
 ids=tokenizer.encode(wrapped,add_special_tokens=False)
 mapping={str(r['occurrence_id']):{'task':r['task'],'phase':r['phase'],'source_id':r['source_id'],
  'reserve_ordinal':r['reserve_ordinal'],'feature_sha256':r['feature_sha256']} for r in sorted(records,key=lambda r:r['occurrence_id'])}
 return {**p,'wrapped_prompt':wrapped,'prompt_token_ids':ids,'prompt_tokens_preflight':len(ids),'prompt_sha256':digest(wrapped),
  'grammar_sha256':digest(p['grammar']),'id_map':mapping,'id_map_sha256':digest(compact(mapping))}
def parser_checks():
 specs=[([9,10,127,128],'{"9":"Y","10":"N","127":"Y","128":"N"}',True),
 ([128,127,10,9],'{"128":"N","127":"Y","10":"N","9":"Y"}',True),
 ([128],'{"128":"N"}',True),([1],'{"1":"Y","1":"N"}',False),([1],' {"1":"Y"}',False),
 ([1],'{"1":"Y"}\n',False),([1],'{"1": "Y"}',False),([1,2],'{"2":"N","1":"Y"}',False),
 ([1],'{"1":"Y","2":"N"}',False),([1],'{"1":"yes"}',False),([1,2],'{"1":"Y"}',False)]
 result=[{'ids':ids,'raw':raw,'expected_valid':want,'valid':parse(raw,ids) is not None} for ids,raw,want in specs]
 assert all(x['expected_valid']==x['valid'] for x in result)
 return result
