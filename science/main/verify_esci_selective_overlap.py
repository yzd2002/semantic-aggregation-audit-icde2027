#!/usr/bin/env python3
"""Post-hoc exact-text and cross-source query-overlap audit; no replacement."""
import argparse
import json
from pathlib import Path

import pyarrow.dataset as ds
from select_esci_selective_holdout import PRODUCTS_SHA, sha

ROOT = Path(__file__).resolve().parent
FIELDS = ('product_title','product_description','product_bullet_point','product_brand','product_color')


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--products',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output path.')
    if sha(a.products)!=PRODUCTS_SHA:
        raise ValueError('Pinned product source hash differs.')
    new=lines(a.selection/'inputs.jsonl')
    new_queries={r['query'].strip().casefold() for r in new}
    new_text={x['text'] for r in new for x in r['items']}
    text_sources={}
    for name in ('run_test_160','run_validation_500','run_full_pools_250'):
        text_sources[name]={x['text'] for r in lines(ROOT/name/'inputs.jsonl') for x in r['items']}
    train=lines(ROOT/'run_supervised_baseline_1000/train_manifest.jsonl')
    train_ids={r['product_id'] for r in train}
    table=ds.dataset(a.products,format='parquet').to_table(columns=['product_id','product_locale',*FIELDS],
        filter=(ds.field('product_locale')=='us') & ds.field('product_id').isin(sorted(train_ids)))
    products=table.to_pylist()
    product_map={r['product_id']:r for r in products}
    if len(product_map)!=len(products) or set(product_map)!=train_ids:
        raise ValueError('Training product joins differ.')
    text_sources['supervised_training']={
        ' '.join(r[k] for k in FIELDS if isinstance(r[k],str) and r[k])[:600] for r in products}
    overlap={name:len(new_text & old) for name,old in text_sources.items()}
    combined=set().union(*text_sources.values())
    affected_rows=sum(x['text'] in combined for r in new for x in r['items'])
    affected_queries=sum(any(x['text'] in combined for x in r['items']) for r in new)
    cross={}
    for name in ('run_wands_160','wands_holdout_selection','wands_fallback_validation_selection',
                 'wands_selective_execution_selection','trec_covid_transfer_selection'):
        old={r['query'].strip().casefold() for r in lines(ROOT/name/'inputs.jsonl')}
        cross[name]=len(new_queries & old)
    result={'protocol':'post-hoc exact-text and cross-source normalized-query check; no samples replaced',
            'distinct_new_texts':len(new_text), 'distinct_exact_texts_shared_by_source':overlap,
            'distinct_exact_texts_shared_with_any_prior':len(new_text & combined),
            'new_candidate_rows_with_any_exact_text_overlap':affected_rows,
            'new_queries_with_any_exact_text_overlap':affected_queries,
            'cross_source_normalized_query_overlaps':cross,
            'training_product_ids_checked':len(train_ids),
            'limitation':'Exact 600-character string equality is not semantic or source-product identity; paraphrases and unknown pretraining exposure remain.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Saved counts-only post-hoc exact-text and cross-source query audit.')


if __name__=='__main__':
    main()
