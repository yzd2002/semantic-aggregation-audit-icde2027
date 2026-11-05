#!/usr/bin/env python3
"""Independently recompute actual-selective crossed query–product resampling."""
import argparse
import json
from pathlib import Path

import numpy as np
from verify_order_fallback_validation import keyed, lines, point


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    a=ap.parse_args();rows=lines(a.selection/'inputs.jsonl')
    prior=json.loads((a.run/'paired_analysis.json').read_text())
    saved=json.loads((a.run/'two_way_analysis.json').read_text())
    modes={m:lines(a.run/(m+'_responses.jsonl')) for m in ('original','replay','stable_reverse','pointwise')}
    sparse={(x['query_id'],x['start']):point(x['final']) for x in lines(a.run/'selective_pointwise_responses.jsonl')}
    original=[];fallback=[];gold=[];products=[];queries=[]
    for q,row in enumerate(rows):
        vectors={}
        for mode in ('original','replay','stable_reverse'):
            value=[]
            for rec in modes[mode][q*4:(q+1)*4]:
                x=keyed(rec['final']);value.extend(x if x is not None else [None]*8)
            vectors[mode]=value
        full=[point(x['final']) for x in modes['pointwise'][q*32:(q+1)*32]]
        h=[sparse.get((row['query_id'],i)) for i in range(32)]
        if any(None in v for v in vectors.values()) or None in full:continue
        actual=[x if x==y else z for x,y,z in zip(vectors['original'],vectors['stable_reverse'],h)]
        if None in actual:continue
        original.append([x=='Y' for x in vectors['original']]);fallback.append([x=='Y' for x in actual])
        gold.append([x['gold']=='Y' for x in row['items']]);products.append([int(x['product_id']) for x in row['items']])
        queries.append(row['query_id'])
    assert queries == prior['common_query_ids'] and queries
    distinct=sorted({p for row in products for p in row});index={p:i for i,p in enumerate(distinct)}
    mapping=np.asarray([[index[p] for p in row] for row in products],dtype=int)
    x=np.asarray(original,dtype=float);y=np.asarray(fallback,dtype=float);g=np.asarray(gold,dtype=float)
    rng=np.random.default_rng(20271203);contrasts=[]
    for _ in range(4000):
        qcount=np.bincount(rng.integers(0,len(queries),len(queries)),minlength=len(queries))
        pcount=np.bincount(rng.integers(0,len(distinct),len(distinct)),minlength=len(distinct))
        weights=pcount[mapping];den=weights.sum(axis=1)
        assert not np.any((qcount>0)&(den==0))
        safe=np.where(den==0,1,den)
        ref=(weights*g).sum(axis=1)/safe;orig=(weights*x).sum(axis=1)/safe;fb=(weights*y).sum(axis=1)/safe
        contrasts.append(float(np.dot(qcount,np.abs(fb-ref)-np.abs(orig-ref))/qcount.sum()))
    ci=np.quantile(contrasts,[.025,.975])
    assert saved['protocol']=='PROTOCOL_selective_model_sensitivity.md'
    assert saved['selection_inputs_sha256']==prior['selection_inputs_sha256']
    assert saved['common_complete_pools']==len(queries) and saved['distinct_product_ids']==len(distinct)
    assert saved['repeated_product_ids']==int(sum(v>1 for v in np.bincount(mapping.ravel())))
    assert saved['resamples']==4000 and saved['seed']==20271203
    assert np.allclose(ci,saved['actual_fallback_minus_original_ratio_mae_query_product_ci'],rtol=0,atol=1e-12)
    print('Actual-selective crossed query–product resampling independently verified on',len(queries),'queries and',len(distinct),'products.')


if __name__=='__main__':main()
