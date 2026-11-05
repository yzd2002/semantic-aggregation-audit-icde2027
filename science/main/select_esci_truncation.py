#!/usr/bin/env python3
"""Reconstruct aligned 600- and 1,200-character ESCI product prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCT_SHA256 = '25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'
PROTOCOL = 'PROTOCOL_esci_truncation_sensitivity.md'
FIELDS = ('product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(x)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--products', type=Path, required=True)
    ap.add_argument('--prior', type=Path, default=HERE/'run_full_pools_250')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if sha(a.products) != PRODUCT_SHA256: raise ValueError('Official product Parquet hash differs')
    rows = [json.loads(x) for x in (a.prior/'inputs.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != 250 or sum(len(r['items']) for r in rows) != 5160:
        raise ValueError('Unexpected archived sample')
    ids = sorted({it['product_id'] for r in rows for it in r['items']})
    import pyarrow.dataset as ds
    data = ds.dataset(a.products, format='parquet')
    table = data.to_table(columns=['product_id','product_locale',*FIELDS],
                          filter=(ds.field('product_locale') == 'us') & ds.field('product_id').isin(ids))
    products = {}
    for item in table.to_pylist():
        pid=item['product_id']
        if pid in products:raise ValueError('Duplicate product/locale key')
        products[pid]=item
    if set(products) != set(ids):raise ValueError('Missing product joins')
    selected=[]; changed=longer=0; changed_queries=0
    for row in rows:
        items=[];query_changed=False
        for item in row['items']:
            p=products[item['product_id']]
            full=' '.join(p[k] for k in FIELDS if isinstance(p[k],str) and p[k])
            short, long=full[:600],full[:1200]
            if short != item['text'] or item.get('join_missing'):
                raise ValueError('Rebuilt short prefix differs from archived input')
            diff=short!=long;changed+=diff;longer+=len(full)>1200;query_changed|=diff
            items.append({'id':item['id'],'product_id':item['product_id'],'example_id':item['example_id'],
                          'grade':item['grade'],'gold':item['gold'],'short':short,'long':long,
                          'source_characters':len(full),'text_changed':diff})
        selected.append({'query_id':row['query_id'],'query':row['query'],'items':items})
        changed_queries+=query_changed
    a.out.mkdir(parents=True,exist_ok=False)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    (a.out/'inputs.jsonl').write_text(payload,encoding='utf8')
    info={'protocol':PROTOCOL,'prior_inputs_sha256':sha(a.prior/'inputs.jsonl'),
          'products_sha256':sha(a.products),'inputs_sha256':sha(a.out/'inputs.jsonl'),
          'pools':len(selected),'candidates':sum(len(r['items']) for r in selected),
          'distinct_products':len(ids),'changed_candidates':changed,'changed_queries':changed_queries,
          'source_text_over_1200_characters':longer,'short_cap_characters':600,'long_cap_characters':1200}
    (a.out/'selection.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Aligned original sample:',info['pools'],'pools,',info['candidates'],'candidates; longer text differs for',changed)


if __name__=='__main__':main()
