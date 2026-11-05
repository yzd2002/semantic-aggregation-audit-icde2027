#!/usr/bin/env python3
"""Whole-query exclusions and frozen schedule for the new controlled audit."""
import collections,hashlib,json,random,time
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
from common import *

def main():
    destination=ROOT/'selection'
    if destination.exists():raise FileExistsError(destination)
    examples=SOURCE/'shopping_queries_dataset_examples.parquet';products=SOURCE/'shopping_queries_dataset_products.parquet'
    if sha(examples)!=EXAMPLES_SHA or sha(products)!=PRODUCTS_SHA:raise ValueError('Pinned source checksum mismatch')
    q,pids,oldtext,train,sources=prior_sources()
    print('Prior sources',len(sources),'training manifest rows',len(train),flush=True)
    ex=pd.read_parquet(examples,columns=['example_id','query','query_id','product_id','product_locale','esci_label','small_version','split'],
        filters=[('product_locale','=','us'),('small_version','=',1),('split','=','test')])
    if ex[['example_id','query_id','query','product_id']].isna().any().any() or ex.example_id.duplicated().any():raise ValueError('Invalid example identity')
    ex['normalized_query']=ex['query'].map(norm)
    train_ids={str(r['product_id']) for r in train};needed=set(ex.product_id.astype(str))|train_ids
    tab=ds.dataset(products,format='parquet').to_table(columns=['product_id','product_locale',*FIELDS],
        filter=(ds.field('product_locale')=='us')&ds.field('product_id').isin(sorted(needed))).to_pandas()
    if tab.duplicated(['product_id','product_locale']).any():raise ValueError('Ambiguous product ID')
    product_text={str(row[0]):text_of(row[2:]) for row in tab.itertuples(index=False,name=None)}
    if not train_ids<=product_text.keys():raise ValueError('Missing training product join')
    train_text={product_text[i] for i in train_ids};oldtext.update(train_text)
    ex['visible_text']=ex.product_id.astype(str).map(product_text)
    ex['old_pid']=ex.product_id.astype(str).isin(pids);ex['old_text']=ex.visible_text.isin(oldtext)
    meta=ex.groupby('query_id',sort=False).agg(query=('normalized_query','first'),norms=('normalized_query','nunique'),
        prior_product=('old_pid','any'),prior_text=('old_text','any'))
    meta['prior_query']=meta['query'].isin(q)
    allowed=meta[(meta.norms==1)&(meta['query']!='')&~meta.prior_query&~meta.prior_product&~meta.prior_text]
    if allowed['query'].duplicated().any():raise ValueError('Eligible normalized query maps to multiple IDs')
    eligible=sorted(map(int,allowed.index));chosen=random.Random(2026110201).sample(eligible,128)
    output=[]
    for qid in chosen:
        part=ex[ex.query_id==qid].sort_values('example_id')
        if part.empty or part.visible_text.isna().any() or (part.visible_text=='').any():raise ValueError('Selected pool missing join/empty text; do not replace')
        if not part.esci_label.isin(list('ESCI')).all():raise ValueError('Unsupported selected grade; do not replace')
        items=[{'example_id':str(r.example_id),'product_id':str(r.product_id),'text':r.visible_text,'grade':r.esci_label,
            'gold':'Y' if r.esci_label in ('E','S') else 'N'} for r in part.itertuples()]
        seed=int.from_bytes(hashlib.sha256(('2026110201\0'+str(qid)).encode()).digest()[:8],'big');random.Random(seed).shuffle(items)
        for i,item in enumerate(items,1):item['id']=i
        output.append({'query_id':str(qid),'query':str(part['query'].iloc[0]),'items':items})
    wpath=ARTIFACT/'wands_selective_execution_selection/inputs.jsonl';wands=read(wpath)
    if len(wands)!=52 or any(len(r['items'])!=32 for r in wands):raise ValueError('Unexpected WANDS sample')
    for r in wands:
        if [x['id'] for x in r['items']]!=list(range(1,33)):raise ValueError('Unexpected WANDS IDs')
    destination.mkdir()
    write_lines(destination/'esci_inputs.jsonl',output);write_lines(destination/'wands_inputs.jsonl',wands)
    write_lines(destination/'schedule.jsonl',schedule({'esci':output,'wands':wands}))
    exclusion={'sources':sources,'normalization':'strip().casefold()','text_rule':'five fields joined by a space, first 600 characters; prior input text[:600]',
        'prior_query_count':len(q),'prior_product_id_count':len(pids),'prior_visible_text_count':len(oldtext),
        'training_unique_product_ids':len(train_ids),'training_unique_visible_texts':len(train_text),
        'whole_query_reasons':{'prior_query':int(meta.prior_query.sum()),'any_prior_product':int(meta.prior_product.sum()),'any_prior_text':int(meta.prior_text.sum())},
        'source_hashes':{'examples':EXAMPLES_SHA,'products':PRODUCTS_SHA,'wands_input':sha(wpath)},'eligible_query_ids':eligible,'selected_query_ids':chosen,
        'excluded_queries':[{'query_id':str(i),'prior_query':bool(r.prior_query),'prior_product':bool(r.prior_product),'prior_text':bool(r.prior_text),'ambiguous_or_empty':bool(r.norms!=1 or not r.query)}
             for i,r in meta.iterrows() if i not in set(eligible)]}
    write_json(destination/'exclusion_manifest.json',exclusion)
    freeze={'created_unix':time.time(),'status':'Inputs and schedule frozen before label summaries and before new GPU judgments',
        'files':{str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('*.py'),ROOT/'PROTOCOL_controlled_order.md',ROOT/'PRESELECTION_CLARIFICATIONS.md',*destination.glob('*')]}}
    write_json(destination/'PREINFERENCE_INPUT_CODE_SHA256.json',freeze)
    # Summarize labels only after the inputs and code have been frozen above.
    info={'eligible_esci_queries':len(eligible),'samples':{name:{'pools':len(rows),'candidates':sum(len(r['items']) for r in rows),
        'grade_counts':dict(collections.Counter(i['grade'] for r in rows for i in r['items'])),'pool_min':min(len(r['items']) for r in rows),'pool_max':max(len(r['items']) for r in rows)}
        for name,rows in [('esci',output),('wands',wands)]},'schedule_jobs':len(read(destination/'schedule.jsonl'))}
    write_json(destination/'selection_summary.json',info);print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':main()
