"""Portable contracts and tokenizer-only preparation helpers; no model loading."""
from pathlib import Path
import hashlib,json,os,re,sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
MODELS=('qwen','smol','glm')
TASKS=('snli',)
SCOPES=('full-first','target-only','full-last')
CONTEXTS=('near','random')
SEED=2026110611
SEEDS={k:SEED for k in ('target','context','schedule','bootstrap','serial')}
SNLI_REVISION='cdb5c3d5eed6ead6e5a341c8e56e669bb666725b'
SOURCE_SHA256={'test':'4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c',
               'train':'ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620'}
SOURCE_SIZES={'test':411531,'train':19614612}
SOURCE_ROWS={'test':10000,'train':550152}
SOURCE_FIELDS=('premise','hypothesis')
MAX_FEATURE_TOKENS=512
MAX_MODEL_LEN=4096
MAX_OUTPUT_TOKENS=128
PACKAGES=('torch','vllm','transformers','numpy','scipy','tokenizers','xgrammar','outlines_core','setuptools','pandas','pyarrow','tiktoken')
SYSTEMS={'snli':('Decide whether each premise entails its paired hypothesis. '
 'Use only that record\'s own complete premise and hypothesis. Other records are not additional evidence for it. '
 'Output Y if the hypothesis must be true given the premise, and N if it is neutral or contradicts the premise. '
 'Judge only the requested record IDs. Return an exact JSON object with quoted record IDs and quoted Y or N values, '
 'with no whitespace or explanation.')}

def compact(obj):return json.dumps(obj,ensure_ascii=False,separators=(',',':'))
def digest(value):return hashlib.sha256(value if isinstance(value,bytes) else value.encode('utf8')).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def read_lines(path):
 with Path(path).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def write_json(path,obj):
 with Path(path).open('x',encoding='utf8') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def write_lines(path,rows):
 with Path(path).open('x',encoding='utf8') as f:
  for row in rows:f.write(compact(row)+'\n')
def normalize(value):return ' '.join(value.casefold().split())
def feature_string(record):return compact(record['features'])
def identities(record):return {normalize(record['features'][k]) for k in SOURCE_FIELDS}
def normalized_feature_fingerprint(record):return digest(compact([normalize(record['features'][k]) for k in SOURCE_FIELDS]))
def hash_order(domain,source_id):return digest(f'{SEED}|{domain}|{source_id}')
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

def cpu_environment():
 os.environ['CUDA_VISIBLE_DEVICES']=''
 os.environ['HF_HOME']=str(ROOT/'cpu_cache/hf')
 os.environ['HF_MODULES_CACHE']=str(ROOT/'cpu_cache/modules')
 os.environ['TRANSFORMERS_OFFLINE']='1'
 os.environ['HF_HUB_OFFLINE']='1'
 os.environ['TOKENIZERS_PARALLELISM']='false'
 os.environ['OMP_NUM_THREADS']='2'
 os.environ['OPENBLAS_NUM_THREADS']='2'
 os.environ['MKL_NUM_THREADS']='2'
 os.environ['PYTHONDONTWRITEBYTECODE']='1'

def load_tokenizers():
 cpu_environment()
 from transformers import AutoTokenizer
 return {m:AutoTokenizer.from_pretrained(str(ROOT/'tokenizers'/m),local_files_only=True,
           trust_remote_code=(m=='glm'),use_fast=(m!='glm')) for m in MODELS}

def token_bytes(tokenizer,token_id):
 if token_id in tokenizer.added_tokens_decoder:return tokenizer.added_tokens_decoder[token_id].content.encode('utf8')
 value=tokenizer.decoder[token_id]
 assert isinstance(value,bytes)
 return value

def prepared_prompt(tokenizer,task,records,scope):
 system,user,target_line=prompt(task,records,scope)
 messages=[{'role':'system','content':system},{'role':'user','content':user}]
 wrapped=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 ids=tokenizer.encode(wrapped,add_special_tokens=False,truncation=False)
 assert ids==tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
 start=wrapped.index(user)+len('Records:\n');end=start+len(target_line)
 start_byte,end_byte=len(wrapped[:start].encode()),len(wrapped[:end].encode())
 if tokenizer.is_fast:
  enc=tokenizer(wrapped,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
  assert enc['input_ids']==ids
  offsets=[(len(wrapped[:a].encode()),len(wrapped[:b].encode())) for a,b in enc['offset_mapping']]
  method='Native fast-tokenizer character offsets converted to UTF-8 byte offsets'
 else:
  chunks=[token_bytes(tokenizer,i) for i in ids]
  assert b''.join(chunks)==wrapped.encode('utf8')
  offsets=[];cursor=0
  for chunk in chunks:offsets.append((cursor,cursor+len(chunk)));cursor+=len(chunk)
  method='Exact concatenation of native tiktoken byte pieces and literal added tokens'
 prefix_n=sum(b<=end_byte for a,b in offsets)
 prefix=wrapped[:end]
 return {'system':system,'unwrapped_user':user,'wrapped_prompt':wrapped,'prompt_token_ids':ids,
  'prompt_tokens_preflight':len(ids),'requested_ids':requested_ids(scope),'output_ids':output_ids(scope),
  'grammar':grammar(output_ids(scope)),'prompt_sha256':digest(wrapped),'grammar_sha256':digest(grammar(output_ids(scope))),
  'target_input_span_chars':[start,end],'target_input_span_bytes':[start_byte,end_byte],
  'target_input_line':target_line,'target_input_prefix':prefix,'target_input_prefix_sha256':digest(prefix),
  'target_prefix_token_count':prefix_n,'target_prefix_token_ids_sha256':digest(compact(ids[:prefix_n])),
  'target_token_end_exact_boundary':bool(prefix_n and offsets[prefix_n-1][1]==end_byte),
  'token_offset_method':method,'token_byte_offsets_sha256':digest(compact(offsets)),
  'fixed_output_prefix_before_target':'{"1":"' if scope!='full-last' else None}

def parser_checks():
 examples=[('{"1":"Y"}',[1],True),('{"1":"Y","2":"N","3":"Y","4":"N"}',[1,2,3,4],True),
 ('{"2":"Y","3":"N","4":"Y","1":"N"}',[2,3,4,1],True),
 ('{"9":"Y","10":"N","11":"Y"}',[9,10,11],True),('{"11":"N"}',[11],True),
 (' {"1":"Y"}',[1],False),('{"1":"Y"}\n',[1],False),('{"1":"Y","1":"N"}',[1],False),
 ('{"1": "Y"}',[1],False),('{"1":"Y","2":"N"}',[1],False),('{"1":"yes"}',[1],False),
 ('{"2":"N","1":"Y"}',[1,2],False),('{"1":"Y"}',[1,2],False),('{"1":"Y"',[1],False),
 ('{"1":"Y","2":"N","3":"N","4":"Y","5":"N"}',[1,2,3,4],False)]
 rows=[{'raw':r,'ids':ids,'expected_valid':valid,'actual_valid':parse(r,ids) is not None} for r,ids,valid in examples]
 assert all(x['expected_valid']==x['actual_valid'] for x in rows)
 return rows
