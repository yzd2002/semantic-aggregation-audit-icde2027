#!/usr/bin/env python3
"""Independently recompute paired query effects from saved binary judgments."""
import argparse, json, math
from pathlib import Path
import numpy as np
from run_keyed_order_audit import parse_keyed


def load(path):
    with path.open(encoding='utf8') as f:return [json.loads(line) for line in f if line.strip()]


def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)


def vectors(dataset,run):
    records=load(run/'original_responses.jsonl')
    index={(x['pool_id'],x['start']):x for x in records};assert len(index)==len(records)
    selective={x['pool_id']:x['pred'] for x in load(run/'selective_outputs.jsonl')}
    assert len(selective)==len(dataset)
    result={}
    for row in dataset:
        key=row['pool_id'];chunks=[parse_keyed(index[(key,j)]['final'],list(range(1,9))) for j in (0,8,16,24)]
        original=[chunk[k] for chunk in chunks for k in range(1,9)] if all(x is not None for x in chunks) else None
        result[key]=(original,selective[key])
    return result


def main(selection,qwen,smol,comparison):
    dataset=load(selection/'inputs.jsonl');saved=json.loads(comparison.read_text());a=vectors(dataset,qwen);b=vectors(dataset,smol)
    for scenario in ('case_control','uniform'):
        attempted=[r for r in dataset if r['scenario']==scenario]
        qa=[r for r in attempted if all(x is not None for x in a[r['pool_id']])]
        sb=[r for r in attempted if all(x is not None for x in b[r['pool_id']])]
        common=[r for r in attempted if all(x is not None for x in a[r['pool_id']]+b[r['pool_id']])]
        out=saved['scenarios'][scenario]
        assert (len(attempted),len(qa),len(sb),len(common))==(out['attempted'],out['complete_by_model']['qwen'],out['complete_by_model']['smol'],out['shared_complete'])
        if not common:continue
        e={'qwen':{'original':[],'selective':[]},'smol':{'original':[],'selective':[]}}
        for row in common:
            g=sum(item['gold']=='Y' for item in row['items'])
            for model,pair in [('qwen',a[row['pool_id']]),('smol',b[row['pool_id']])]:
                for j,name in enumerate(('original','selective')):
                    e[model][name].append(abs(g-sum(x=='Y' for x in pair[j]))/32)
        n=len(common)
        for model in e:
            for name in e[model]:assert close(sum(e[model][name])/n,out['ratio_mae'][model][name])
        q=[y-x for x,y in zip(e['qwen']['original'],e['qwen']['selective'])]
        s=[y-x for x,y in zip(e['smol']['original'],e['smol']['selective'])]
        diff=[x-y for x,y in zip(s,q)]
        assert close(sum(q)/n,out['qwen_selective_minus_original']) and close(sum(s)/n,out['smol_selective_minus_original'])
        assert close(sum(diff)/n,out['difference_of_differences'])
        rng=np.random.default_rng(out['query_bootstrap_seed']);indices=rng.integers(0,n,size=(4000,n))
        draws=[sum(diff[i] for i in row)/n for row in indices]
        assert all(close(x,y) for x,y in zip(np.quantile(draws,[.025,.975]),out['query_bootstrap_95']))
        print(scenario,'verified',n,'matched complete queries and paired difference intervals')
    print('Independent matched-query verification passed; no inference rerun.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--qwen',type=Path,required=True);p.add_argument('--smol',type=Path,required=True);p.add_argument('--comparison',type=Path,required=True)
    a=p.parse_args();main(a.selection,a.qwen,a.smol,a.comparison)
