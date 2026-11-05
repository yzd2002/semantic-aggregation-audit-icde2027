#!/usr/bin/env python3
"""Select WANDS judged pools and run the existing operator harness, without modifying it."""
import argparse,hashlib,json,random,sys
from pathlib import Path
import pandas as pd
import run_experiment as base
import run_full_pools as harness

HERE=Path(__file__).resolve().parent
SEED=20271110
EXPECTED={'query.csv':'63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95',
          'label.csv':'c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef',
          'product.csv':'d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6'}
SYSTEM=("You judge whether a product fully matches a shopper's query. "
        "Mark Y only if it matches the requested product type and every stated essential modifier; otherwise mark N. "
        "Use only the supplied information. Do not assume unknown product properties.")

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def make_text(product):
    return (f"Name: {product['product_name'][:140]}\n"
            f"Class: {product['product_class'][:60]}\n"
            f"Features: {product['product_features'][:210]}\n"
            f"Description: {product['product_description'][:140]}")[:600]

def sample(wands_dir):
    wands_dir=Path(wands_dir)
    for name,want in EXPECTED.items():
        got=sha256(wands_dir/name)
        if got!=want:raise ValueError(f'Unexpected WANDS source SHA-256 for {name}: {got}')
    q=pd.read_csv(wands_dir/'query.csv',sep='\t',keep_default_na=False,dtype={'query_id':'int64','query':str})
    l=pd.read_csv(wands_dir/'label.csv',sep='\t',keep_default_na=False,dtype={'id':'int64','query_id':'int64','product_id':'int64','label':str})
    p=pd.read_csv(wands_dir/'product.csv',sep='\t',keep_default_na=False,
        usecols=['product_id','product_name','product_class','product_features','product_description'],
        dtype={'product_id':'int64','product_name':str,'product_class':str,'product_features':str,'product_description':str})
    if q.query_id.duplicated().any() or p.product_id.duplicated().any() or l.id.duplicated().any():raise ValueError('Non-unique source key')
    if not l.query_id.isin(q.query_id).all() or not l.product_id.isin(p.product_id).all():raise ValueError('Unmatched foreign key')
    if not l.label.isin(['Exact','Partial','Irrelevant']).all():raise ValueError('Unknown grade')
    group=l.groupby(['query_id','product_id'],sort=False).label.nunique()
    conflicting=group[group>1].index
    unique=l.sort_values(['query_id','product_id','id']).drop_duplicates(['query_id','product_id'],keep='first').set_index(['query_id','product_id'])
    unique=unique.drop(index=conflicting).reset_index()
    prior=set()
    for name in ('run_test_160','run_validation_500','run_full_pools_250'):
        with (HERE/name/'inputs.jsonl').open(encoding='utf8') as f:
            prior.update(json.loads(line)['query'].strip().casefold() for line in f)
    normalized=q['query'].str.strip().str.casefold()
    allowed=q[normalized.ne('') & ~normalized.isin(prior)]
    counts=unique.groupby('query_id').size()
    eligible=sorted(int(v) for v in allowed.query_id if counts.get(v,0)>=32)
    if len(eligible)<160:raise ValueError(f'Only {len(eligible)} eligible query IDs')
    chosen=random.Random(SEED).sample(eligible,160)
    qs=q.set_index('query_id');products=p.set_index('product_id').to_dict('index')
    groups={int(qid):grp.sort_values(['id','product_id']) for qid,grp in unique[unique.query_id.isin(chosen)].groupby('query_id')}
    rows=[]
    for qid in chosen:
        grp=groups[qid];entries=grp.to_dict('records')
        seed=int.from_bytes(hashlib.sha256((str(SEED)+'\0'+str(qid)).encode()).digest()[:8],'big')
        rng=random.Random(seed);chosen_entries=rng.sample(entries,32);rng.shuffle(chosen_entries)
        items=[]
        for i,e in enumerate(chosen_entries,1):
            items.append({'id':i,'text':make_text(products[int(e['product_id'])]),'grade':e['label'],
                          'gold':'Y' if e['label']=='Exact' else 'N',
                          'example_id':str(e['id']),'product_id':str(e['product_id']),'join_missing':False})
        rows.append({'query_id':str(qid),'query':str(qs.at[qid,'query']),'items':items})
    meta={'source':'Wayfair WANDS','revision':'3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5',
          'source_query_count':len(q),'source_label_rows':len(l),'source_product_count':len(p),
          'duplicate_query_product_rows':int(l.duplicated(['query_id','product_id']).sum()),
          'conflicting_query_product_groups_removed':len(conflicting),'queries_excluded_by_prior_text':int(normalized.isin(prior).sum()),
          'eligible_query_ids':len(eligible),'sample_size':len(rows),'candidate_count':len(rows)*32,
          'candidates_per_sampled_pool':32,'label_mapping':'Exact=Y; Partial/Irrelevant=N','system_instruction':SYSTEM,
          'products_with_all_source_text_fields_empty':sum(not any(products[int(it['product_id'])][k] for k in
             ('product_name','product_class','product_features','product_description')) for row in rows for it in row['items'])}
    return rows,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wands-dir',type=Path,default=Path('/home/gpu/intent2s_wands_official_2026'))
    ap.add_argument('--model',default='/root/models/Qwen/Qwen3-4B-Instruct-2507');ap.add_argument('--out',default='run_wands_160')
    args=ap.parse_args();rows,meta=sample(args.wands_dir)
    harness.sample_official=lambda _examples,_products:(rows,meta)
    base.SYSTEM=SYSTEM
    sys.argv=['run_full_pools.py','--model',args.model,'--out',args.out]
    harness.main()
    path=Path(args.out)/'summary.json';summary=json.loads(path.read_text())
    summary['protocol']='PROTOCOL_wands.md';summary['dataset']='WANDS';summary['system_instruction']=SYSTEM
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print('WANDS protocol metadata recorded.',flush=True)
if __name__=='__main__':main()
