import hashlib,json,re,random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ARTIFACT=Path('/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility')
SOURCE=Path('/home/gpu/intent2s_esci_official_2026')
EXAMPLES_SHA='4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'
PRODUCTS_SHA='25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'
FIELDS=('product_title','product_description','product_bullet_point','product_brand','product_color')
MODELS={'qwen':Path('/root/models/Qwen/Qwen3-4B-Instruct-2507'),'smol':Path('/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct')}
SYSTEMS={'esci':"You judge whether each product can satisfy a shopper's query. Mark Y for an exact match or a functional substitute, and N for a mere accessory, an unrelated product, or a product lacking a stated essential requirement. Use only the supplied information. Do not assume unknown product properties.",
 'wands':"You judge whether a product fully matches a shopper's query. Mark Y only if it matches the requested product type and every stated essential modifier; otherwise mark N. Use only the supplied information. Do not assume unknown product properties."}
CONDITIONS=('AA','RA','AD','RD')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def read(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def write_json(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write_lines(p,rows):
    with Path(p).open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def norm(s):return str(s).strip().casefold()
def text_of(values):return ' '.join(v for v in values if isinstance(v,str) and v)[:600]
def grammar(ids,descending=False):
    ids=sorted(ids,reverse=descending)
    if not ids or len(set(ids))!=len(ids) or any(type(i) is not int or i<1 for i in ids):raise ValueError('Invalid occurrence IDs')
    return r'\{'+','.join(re.escape(json.dumps(str(i)))+r':"[YN]"' for i in ids)+r'\}'
def parse(raw,ids,descending=False):
    if not isinstance(raw,str) or re.fullmatch(grammar(ids,descending),raw) is None:return None
    pairs=json.loads(raw,object_pairs_hook=list)
    if [k for k,v in pairs]!=[str(i) for i in sorted(ids,reverse=descending)]:return None
    return {int(k):v for k,v in pairs}
def prompt(row,items):
    return ('Query: '+row['query']+'\nProducts in order:\n'+
        '\n'.join(json.dumps({'id':x['id'],'text':x['text']},ensure_ascii=False) for x in items)+
        '\nReturn only a JSON object mapping each given product id to Y or N for that product. '
        'Use the given ids, not positions; include every id exactly once. Either key order is allowed. '
        'Do not add spaces or explanations.')
def prior_sources(artifact=ARTIFACT):
    q=set();pid=set();texts=set();sources=[];trains=[]
    for path in sorted(artifact.rglob('inputs.jsonl')):
        rows=read(path);grades={str(i.get('grade')) for r in rows for i in r.get('items',[])}
        kind='esci' if grades and grades<=set('ESCI') else 'wands' if grades and grades<={'Exact','Partial','Irrelevant'} else None
        if not kind:continue
        q.update(norm(r['query']) for r in rows)
        if kind=='esci':
            pid.update(str(i['product_id']) for r in rows for i in r['items'] if 'product_id' in i)
            for r in rows:
                for item in r['items']:
                    visible=[item[k] for k in ('text','short','long') if isinstance(item.get(k),str)]
                    if not visible:raise ValueError('Prior ESCI item lacks a recorded text variant: '+str(path))
                    texts.update(value[:600] for value in visible)
        sources.append({'path':str(path),'sha256':sha(path),'kind':kind,'pools':len(rows),
            'candidate_occurrences_without_product_id':sum('product_id' not in i for r in rows for i in r['items'])})
    for path in sorted(artifact.rglob('train_manifest.jsonl')):
        rows=read(path)
        if not rows or not all({'query','product_id'}<=r.keys() for r in rows):raise ValueError('Unknown training manifest '+str(path))
        q.update(norm(r['query']) for r in rows);pid.update(str(r['product_id']) for r in rows);trains.extend(rows)
        sources.append({'path':str(path),'sha256':sha(path),'kind':'training_manifest','rows':len(rows)})
    return q,pid,texts,trains,sources
def schedule(inputs):
    rng=random.Random(2026110217);chunks=[]
    for dataset,rows in inputs.items():
        for row in rows:
            for start in range(0,len(row['items']),8):chunks.append((dataset,row,start))
    rng.shuffle(chunks);jobs=[]
    for chunk_no,(dataset,row,start) in enumerate(chunks):
        items=row['items'][start:start+8];group=[]
        for cond in CONDITIONS:
            for repeat in (0,1):group.append({'dataset':dataset,'query_id':row['query_id'],'start':start,'size':len(items),
                'condition':cond,'repeat':repeat,'ids':[x['id'] for x in items],'chunk_ordinal':chunk_no})
        for item in items:
            for repeat in (0,1):group.append({'dataset':dataset,'query_id':row['query_id'],'start':item['id']-1,'size':1,
                'condition':'S','repeat':repeat,'ids':[item['id']],'chunk_ordinal':chunk_no})
        rng.shuffle(group)
        for job in group:
            job['ordinal']=len(jobs);job['block']=len(jobs)//64;job['request_id']=f"r{len(jobs):06d}";jobs.append(job)
    return jobs
