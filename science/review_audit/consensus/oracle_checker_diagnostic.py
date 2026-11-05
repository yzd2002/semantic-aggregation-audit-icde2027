#!/usr/bin/env python3
"""Post-hoc oracle checker diagnostic; not an implementable algorithm or new evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_consensus_baselines import SETTINGS,read,keyed_vectors,exact_uniform,mean_contrast
from verify_consensus_audit import dp

def run(base,out):
    if out.exists():raise FileExistsError(out)
    result={'scope':'Post-hoc oracle diagnostic on exposed labels. Replace checked candidates with reference labels to separate routing enrichment from checker failure. Not deployable; no latency or generalization claim.', 'settings':{}}
    for si,(name,selection,folder,nq) in enumerate(SETTINGS):
        rows=read(base/selection/'inputs.jsonl');a=keyed_vectors(rows,read(base/folder/'original_responses.jsonl'));b=keyed_vectors(rows,read(base/folder/'stable_reverse_responses.jsonl'))
        hmap={(r['query_id'],r['start']):r['final']=='Y' for r in read(base/folder/'pointwise_responses.jsonl')}
        per=[];pooled=dict(disagreements=0,keyed_errors_disagree=0,pointwise_errors_disagree=0)
        for row,x,y in zip(rows,a,b):
            g=np.asarray([i['gold']=='Y' for i in row['items']]);h=np.asarray([hmap[row['query_id'],i] for i in range(len(g))]);mask=x!=y;n=len(g);k=int(mask.sum());oracle=np.where(mask,g,x)
            ex=exact_uniform(x,g,g,k);assert ex==dp(x.astype(int).tolist(),g.astype(int).tolist(),g.astype(int).tolist(),k)
            ref=float(np.mean(x!=g))*(1-k/n);route=float(np.mean(oracle!=g));mae=float(abs(int(oracle.sum())-int(g.sum()))/n)
            per.append({'query_id':row['query_id'],'oracle_routed_candidate_error':route,'oracle_uniform_expected_candidate_error':ref,
                'oracle_routed_mae':mae,'oracle_uniform_expected_mae':float(ex)})
            pooled['disagreements']+=k;pooled['keyed_errors_disagree']+=int(np.count_nonzero((x!=g)&mask));pooled['pointwise_errors_disagree']+=int(np.count_nonzero((h!=g)&mask))
        m=lambda k:float(np.mean([x[k] for x in per]))
        record={'pools':nq,'pooled_disagreement_diagnostics':pooled,'means':{k:m(k) for k in per[0] if k!='query_id'},
                'oracle_routed_minus_uniform':{metric:mean_contrast([x['oracle_routed_'+metric]-x['oracle_uniform_expected_'+metric] for x in per],2026110310+si) for metric in ('mae','candidate_error')},'per_pool':per}
        result['settings'][name]=record;print(name,json.dumps({k:v for k,v in record.items() if k!='per_pool'}))
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.artifact,a.out)
