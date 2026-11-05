#!/usr/bin/env python3
"""Post-hoc sensitivity excluding new queries with any exact old product text."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from analyze_esci_selective_holdout import lines, rebuild
from analyze_order_fallback_validation import paired
from run_compact_prompt_ablation import metrics
from select_esci_selective_holdout import PRODUCTS_SHA, sha
from verify_esci_selective_holdout import close
from verify_esci_selective_secondary import paired_independent

ROOT=Path(__file__).resolve().parent
FIELDS=('product_title','product_description','product_bullet_point','product_brand','product_color')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--products',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--qwen-run',type=Path,required=True)
    p.add_argument('--smollm-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a new output directory.')
    if sha(a.products)!=PRODUCTS_SHA:raise ValueError('Pinned product source differs.')
    rows=lines(a.selection/'inputs.jsonl')
    prior=set()
    for name in ('run_test_160','run_validation_500','run_full_pools_250'):
        prior.update(x['text'] for r in lines(ROOT/name/'inputs.jsonl') for x in r['items'])
    training_ids={r['product_id'] for r in lines(ROOT/'run_supervised_baseline_1000/train_manifest.jsonl')}
    product_rows=ds.dataset(a.products,format='parquet').to_table(columns=['product_id','product_locale',*FIELDS],
        filter=(ds.field('product_locale')=='us') & ds.field('product_id').isin(sorted(training_ids))).to_pylist()
    if len(product_rows)!=len(training_ids) or {r['product_id'] for r in product_rows}!=training_ids:
        raise ValueError('Training product joins differ.')
    prior.update(' '.join(r[k] for k in FIELDS if isinstance(r[k],str) and r[k])[:600] for r in product_rows)
    indices=[i for i,r in enumerate(rows) if not any(x['text'] in prior for x in r['items'])]
    selected_indices=set(indices)
    excluded=[r['query_id'] for i,r in enumerate(rows) if i not in selected_indices]
    overlap=json.loads((ROOT/'ESCI_SELECTIVE_OVERLAP_ANALYSIS.json').read_text())
    if len(excluded)!=overlap['new_queries_with_any_exact_text_overlap'] or len(indices)+len(excluded)!=250:
        raise ValueError('Text-overlap audit differs.')
    ref=lines(a.reference/'predictions.jsonl')
    if [x['query_id'] for x in ref]!=[x['query_id'] for x in rows]:raise ValueError('Reference alignment differs.')
    supervised=[r['pred'] for r in ref]
    result={'protocol':'post-hoc sensitivity after inspecting the main new-pool results',
            'excluded_queries_with_any_exact_text_overlap':excluded,'included_queries':len(indices),
            'included_candidates':sum(len(rows[i]['items']) for i in indices),
            'limitation':'Exact text non-overlap is not semantic independence or proof against pretraining exposure.',
            'models':{}}
    for model,run in (('qwen',a.qwen_run),('smollm',a.smollm_run)):
        events={mode:lines(run/(mode+'_responses.jsonl')) for mode in
                ('original','stable_reverse','selective_pointwise','replay','pointwise')}
        pred,_,_,_=rebuild(rows,events)
        if any(pred[k][i] is None for i in indices for k in ('original','actual_fallback','pointwise')):
            raise ValueError('Sensitivity subset contains an incomplete plan.')
        pred['supervised']=supervised
        pred['always_Y']=[['Y']*len(r['items']) for r in rows]
        draws=np.random.default_rng(20271203).integers(0,len(indices),size=(4000,len(indices)))
        comparison=paired(rows,pred,indices,'original','actual_fallback',draws)
        independent=paired_independent(rows,pred['original'],pred['actual_fallback'],indices,seed=20271203)
        if any(not close(independent[k],comparison[k]) for k in independent):
            raise ValueError('Independent paired reconstruction differs.')
        result['models'][model]={'quality':{k:metrics(rows,pred[k],indices) for k in
                                  ('original','actual_fallback','pointwise','always_Y','supervised')},
                                 'selective_minus_keyed':comparison}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Post-hoc exact-text-exclusion sensitivity independently checked on',len(indices),'queries.')


if __name__=='__main__':main()
