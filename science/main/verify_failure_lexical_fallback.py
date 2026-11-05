#!/usr/bin/env python3
"""Independently reconstruct all fallback decisions and paired query statistics."""
import argparse
import collections
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {'esci_qwen': ('run_full_pools_250', 250, 20271041),
        'wands_qwen': ('run_wands_160', 160, 20271042),
        'esci_smollm': ('run_full_pools_smollm_250', 250, 20271043),
        'wands_smollm': ('run_wands_smollm_160', 160, 20271044)}
METHOD = 'two_stage_compact_fallback'


def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''): h.update(x)
    return h.hexdigest()


def near(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(near(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(near(x, y) for x, y in zip(a, b))
    if a is None or b is None: return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):return abs(a-b) < 1e-12
    return a == b


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    info=json.loads((a.run/'summary.json').read_text());details=read(a.run/'details.jsonl')
    assert info['protocol']=='PROTOCOL_failure_lexical_fallback.md' and info['analysis_is_post_hoc'] is True
    recorded={(x['dataset'],x['query_id']):x for x in details};assert len(recorded)==len(details)
    for label,(directory,n,seed) in RUNS.items():
        src=HERE/directory;rows=read(src/'inputs.jsonl');outputs=read(src/'outputs.jsonl')
        prior=json.loads((src/'summary.json').read_text());s=info['datasets'][label]
        assert len(rows)==len(outputs)==n and s['source']==directory
        assert s['source_sha256']=={k:sha(src/k) for k in ('inputs.jsonl','outputs.jsonl','summary.json')}
        bounds_file=src/'failure_bounds_details.jsonl'
        bounds={x['query_id']:x for x in read(bounds_file) if x['method']==METHOD} if bounds_file.exists() else None
        source={'whole':collections.Counter(),'partial':collections.Counter()}
        pairs={k:[] for k in ('pointwise','lexical','whole','partial','all_yes','all_no')};failed=0
        for row,out in zip(rows,outputs):
            tid=row['query_id'];assert tid==out['query_id'] and row['query']==out['query']
            gold=[x['gold'] for x in row['items']];m=len(gold);assert m and set(gold)<={'Y','N'}
            recovered={};visited=set();plan=out['plans'][METHOD]
            for chunk in plan['raw']:
                start,length=chunk['start'],chunk['size']
                assert type(start) is int and type(length) is int and length>0 and start>=0 and start+length<=m
                indices=list(range(start,start+length));assert not visited.intersection(indices);visited.update(indices)
                final=chunk['final']
                if isinstance(final,str) and re.fullmatch('[YN]{'+str(length)+'}',final):
                    recovered.update(zip(indices,final))
            assert visited==set(range(m))
            v=[recovered.get(i) for i in range(m)];is_failed=None in v
            assert (plan['pred'] is None)==is_failed
            if not is_failed:assert plan['pred']==v
            failed+=is_failed
            q=set(re.findall(r'\w+',row['query'].casefold()))
            lex=['Y' if q and not q.difference(re.findall(r'\w+',x['text'].casefold())) else 'N' for x in row['items']]
            whole=lex[:] if is_failed else v[:]
            partial=[v[i] if v[i] is not None else lex[i] for i in range(m)]
            point=out['plans']['pointwise']['pred']
            assert point is not None and len(point)==m and set(point)<={'Y','N'}
            assert point==[x['final'] if x['final'] in ('Y','N') else None for x in out['plans']['pointwise']['raw']]
            assert recorded[(label,tid)]=={'dataset':label,'query_id':tid,'n':m,'failed':is_failed,
                'recovered':v,'lexical':lex,'whole':whole,'partial':partial,'pointwise':point}
            source['whole']['model']+=0 if is_failed else m
            source['whole']['lexical']+=m if is_failed else 0
            source['partial']['model']+=len(recovered)
            source['partial']['lexical']+=m-len(recovered)
            if bounds is not None and is_failed:
                b=bounds[tid]
                assert b['n']==m and b['accepted']==len(recovered)
                assert b['accepted_positive']==sum(x=='Y' for x in recovered.values())
                assert b['accepted_mistakes']==sum(gold[i]!=v[i] for i in recovered)
            choices={'pointwise':point,'lexical':lex,'whole':whole,'partial':partial,
                     'all_yes':['Y']*m,'all_no':['N']*m}
            for k,pred in choices.items():pairs[k].append((gold,pred))
        assert s['attempted_pools']==n and s['failed_pools']==failed and s['compact_complete_pools']==n-failed
        assert n-failed==prior['plans'][METHOD]['complete_pools']
        assert s['source_counts']=={k:dict(v) for k,v in source.items()}
        if bounds is not None:assert len(bounds)==failed
        errors={}
        for name,values in pairs.items():
            counts=collections.Counter((g,p) for gold,pred in values for g,p in zip(gold,pred))
            tp=counts['Y','Y'];tn=counts['N','N'];fp=counts['N','Y'];fn=counts['Y','N']
            e=[abs(Fraction(pred.count('Y')-gold.count('Y'),len(gold))) for gold,pred in values]
            expected={'candidates':tp+tn+fp+fn,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
                'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
                'ratio_mae':float(sum(e)/len(e))}
            assert near(s['metrics'][name],expected),(label,name)
            errors[name]=np.array([float(x) for x in e])
        assert s['bootstrap_seed']==seed and s['bootstrap_draws']==4000
        draws=np.random.default_rng(seed).integers(0,n,size=(4000,n))
        for left,right in (('whole','pointwise'),('partial','pointwise'),('partial','whole')):
            delta=errors[left]-errors[right];values=delta[draws].mean(axis=1)
            expected={'ratio_mae_difference':float(delta.mean()),
                      'query_bootstrap_95':[float(x) for x in np.percentile(values,[2.5,97.5])]}
            assert near(s['comparisons'][left+'_minus_'+right],expected),(label,left,right)
        print(label,'queries',n,'failed',failed,'all per-query fallbacks, metrics and paired intervals verified')
    assert len(recorded)==sum(x[1] for x in RUNS.values())
    print('Independent strict chunk parsing, lexical reconstruction and statistics passed.')


if __name__=='__main__':main()
