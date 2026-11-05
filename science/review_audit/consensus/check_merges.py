import json, math
from pathlib import Path
import numpy as np

BASE=Path('/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility')
OUT=Path('/dev/shm/intentsat-review-20261102/methods')
SETS=[('WANDS/Qwen','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1'),('WANDS/SmolLM2','wands_selective_execution_selection','run_wands_selective_execution_smollm_v1'),('ESCI/Qwen','esci_selective_holdout_selection','run_esci_selective_holdout_qwen'),('ESCI/SmolLM2','esci_selective_holdout_selection','run_esci_selective_holdout_smollm')]
def read(p):return [json.loads(x) for x in p.read_text().split('\n') if x]
def getvec(rows,recs):
    d={(r['query_id'],r['start']):r for r in recs}; out=[]
    for r in rows:
        v=[]
        for k in range(0,len(r['items']),8):
            obj=json.loads(d[r['query_id'],k]['final'])
            assert list(obj)==list(map(str,range(1,len(obj)+1)))
            v.extend([obj[str(i)]=='Y' for i in range(1,len(obj)+1)])
        assert len(v)==len(r['items']);out.append(np.array(v))
    return out
def metrics(g,p):
    tp=sum(np.count_nonzero(x&y) for x,y in zip(g,p));fp=sum(np.count_nonzero(~x&y) for x,y in zip(g,p));fn=sum(np.count_nonzero(x&~y) for x,y in zip(g,p));tn=sum(np.count_nonzero(~x&~y) for x,y in zip(g,p))
    return dict(f1=2*tp/(2*tp+fp+fn) if tp else 0, balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,candidate_error_macro=float(np.mean([np.mean(x!=y) for x,y in zip(g,p)])),ratio_mae=float(np.mean([abs(x.mean()-y.mean()) for x,y in zip(g,p)])),tp=tp,fp=fp,fn=fn,tn=tn)
def compare(x,y,seed=802):
    d=np.asarray(x)-np.asarray(y);rng=np.random.default_rng(seed);draw=rng.integers(0,len(d),size=(4000,len(d)))
    return dict(mean=float(d.mean()),ci=np.quantile(d[draw].mean(axis=1),[.025,.975]).tolist(),wins=int((d<0).sum()),ties=int((d==0).sum()),losses=int((d>0).sum()))

results={}
budget=json.loads((BASE/'posthoc_routing_budget/routing_budget_result.json').read_text())['settings']
for name,sel,run in SETS:
    rows=read(BASE/sel/'inputs.jsonl');n=len(rows);g=[np.array([i['gold']=='Y' for i in r['items']]) for r in rows]
    a=getvec(rows,read(BASE/run/'original_responses.jsonl'));b=getvec(rows,read(BASE/run/'stable_reverse_responses.jsonl'))
    hmap={(r['query_id'],r['start']):r['final']=='Y' for r in read(BASE/run/'pointwise_responses.jsonl')}
    h=[np.array([hmap[r['query_id'],i] for i in range(len(r['items']))]) for r in rows]
    s=read(BASE/run/'selective_outputs.jsonl');assert [x['query_id'] for x in s]==[r['query_id'] for r in rows]
    plans={'keyed':a,'reverse':b,'pointwise':h,'selective':[np.array([x=='Y' for x in r['pred']]) for r in s], 'cached':[np.where(x==y,x,z) for x,y,z in zip(a,b,h)],'and':[x&y for x,y in zip(a,b)],'or':[x|y for x,y in zip(a,b)]}
    errors={k:np.array([abs(x.mean()-y.mean()) for x,y in zip(g,p)]) for k,p in plans.items()}
    uniform=budget[name]['per_pool'];assert [x['query_id'] for x in uniform]==[r['query_id'] for r in rows]
    errors['uniform']=np.array([x['expected_random_mae'] for x in uniform])
    result={'metrics':{k:metrics(g,p) for k,p in plans.items()},'contrasts':{f'{left}_minus_{right}':compare(errors[left],errors[right]) for left,right in [('selective','and'),('selective','or'),('selective','reverse'),('cached','uniform')]},'strata':{}}
    for typ,mask in [('all_negative',[not x.any() for x in g]),('mixed',[x.any() and not x.all() for x in g]),('all_positive',[x.all() for x in g])]:
        mask=np.array(mask);m=int(mask.sum())
        if m:result['strata'][typ]={'pools':m,'mae':{k:float(v[mask].mean()) for k,v in errors.items()},'cached_minus_uniform':compare(errors['cached'][mask],errors['uniform'][mask]),'selective_minus_keyed':compare(errors['selective'][mask],errors['keyed'][mask])}
    result['pointwise_on_disagreements']={'Y':sum(int(z[x!=y].sum()) for x,y,z in zip(a,b,h)),'total':sum(int((x!=y).sum()) for x,y in zip(a,b))}
    results[name]=result
    print(name,json.dumps(result))
(OUT/'merges_result.json').write_text(json.dumps(results,indent=2)+'\n')
