#!/usr/bin/env python3
"""Independently rebuild all selected product prefixes from the official Parquet."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS=('product_title','product_description','product_bullet_point','product_brand','product_color')
EXPECTED='25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()


def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--products',type=Path,required=True)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True)
    a=ap.parse_args()
    old=read(a.prior/'inputs.jsonl');new=read(a.selection/'inputs.jsonl')
    info=json.loads((a.selection/'selection.json').read_text())
    assert sha(a.products)==info['products_sha256']==EXPECTED
    assert sha(a.prior/'inputs.jsonl')==info['prior_inputs_sha256']
    assert sha(a.selection/'inputs.jsonl')==info['inputs_sha256']
    ids=sorted({it['product_id'] for row in old for it in row['items']})
    import pandas as pd
    products=pd.read_parquet(a.products,columns=['product_id','product_locale',*FIELDS],
        filters=[('product_locale','=','us'),('product_id','in',ids)])
    assert not products.duplicated(['product_id','product_locale']).any()
    d=products.set_index('product_id').to_dict('index');assert set(d)==set(ids)
    rebuilt=[];changed=over=changed_queries=0
    for row in old:
        items=[];qchanged=False
        for it in row['items']:
            p=d[it['product_id']];text=' '.join(p[k] for k in FIELDS if isinstance(p[k],str) and p[k])
            assert not it['join_missing'] and text[:600]==it['text']
            is_changed=text[:600]!=text[:1200];changed+=is_changed;over+=len(text)>1200;qchanged|=is_changed
            items.append({'id':it['id'],'product_id':it['product_id'],'example_id':it['example_id'],
                'grade':it['grade'],'gold':it['gold'],'short':text[:600],'long':text[:1200],
                'source_characters':len(text),'text_changed':is_changed})
        rebuilt.append({'query_id':row['query_id'],'query':row['query'],'items':items});changed_queries+=qchanged
    assert rebuilt==new and len(new)==info['pools']==250
    assert sum(len(r['items']) for r in new)==info['candidates']==5160
    assert len(ids)==info['distinct_products'] and changed==info['changed_candidates']
    assert over==info['source_text_over_1200_characters'] and changed_queries==info['changed_queries']
    assert info['short_cap_characters']==600 and info['long_cap_characters']==1200
    print('Official product joins, field order, both prefixes, labels and all 5,160 input associations independently verified.')
    print('Changed candidates:',changed,'changed queries:',changed_queries,'distinct products:',len(ids))


if __name__=='__main__':main()
