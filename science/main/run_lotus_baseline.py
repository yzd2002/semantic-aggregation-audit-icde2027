#!/usr/bin/env python3
"""Evaluate actual LOTUS sem_filter against saved original pools; see PROTOCOL_lotus.md."""
import argparse,json,re,time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import lotus
from lotus.models import LM
from importlib.metadata import version

HERE=Path(__file__).resolve().parent
INSTRUCTION=('The {product} is an exact match or functional substitute for the shopper query {query}; '
             'exclude mere accessories, unrelated products, and products lacking a stated essential requirement, '
             'and do not assume unstated properties.')
METHODS=('pointwise','batch_json','two_stage_json','two_stage_compact_fallback')
def read_jsonl(path):
    with path.open(encoding='utf8') as f:return [json.loads(line) for line in f]
def metrics(golds,preds):
    arr=[]
    for g,p in zip(golds,preds):
        assert len(g)==len(p) and len(g)>0
        g=np.array(g,dtype=int);p=np.array(p,dtype=int)
        tp=int(((g==1)&(p==1)).sum());fp=int(((g==0)&(p==1)).sum())
        tn=int(((g==0)&(p==0)).sum());fn=int(((g==1)&(p==0)).sum())
        arr.append((tp,fp,tn,fn,abs(float(g.mean()-p.mean()))))
    return np.array(arr,dtype=float)
def summarize(inds,golds,lotus_preds,qwen_preds):
    records={name:metrics([golds[i] for i in inds],[pr[i] for i in inds])
             for name,pr in [('lotus_sem_filter',lotus_preds),('qwen_pointwise',qwen_preds)]}
    rng=np.random.default_rng(20271109);draw=rng.integers(0,len(inds),size=(4000,len(inds)))
    result={'pools':len(inds),'resamples':4000,'seed':20271109,'methods':{}};boot={}
    for name,a in records.items():
        tp,fp,tn,fn=a[:,:4].sum(axis=0)
        actual={'f1':float(2*tp/(2*tp+fp+fn)),
                'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2),
                'ratio_mae':float(a[:,4].mean())}
        c=a[draw,:4].sum(axis=1);tp,fp,tn,fn=c.T
        f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
        sens=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=tp+fn>0)
        spec=np.divide(tn,tn+fp,out=np.zeros_like(tn),where=tn+fp>0)
        boot[name]={'f1':f1,'balanced_accuracy':(sens+spec)/2,'ratio_mae':a[draw,4].mean(axis=1)}
        for k,v in boot[name].items():actual[k+'_ci']=[float(x) for x in np.quantile(v,[.025,.975])]
        result['methods'][name]=actual
    result['lotus_minus_pointwise_ci']={k:[float(x) for x in np.quantile(boot['lotus_sem_filter'][k]-boot['qwen_pointwise'][k],[.025,.975])]
                                       for k in ('f1','ratio_mae')}
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--endpoint',default='http://127.0.0.1:18483/v1')
    ap.add_argument('--served-model',default='qwen-intentsat');ap.add_argument('--out',type=Path,default=Path('run_lotus_baseline'))
    a=ap.parse_args();u=urlparse(a.endpoint)
    if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost'):raise ValueError('This reproducibility run requires a local loopback endpoint')
    a.out.mkdir(exist_ok=False,parents=True)
    pools=read_jsonl(HERE/'run_full_pools_250/inputs.jsonl');reference=read_jsonl(HERE/'run_full_pools_250/outputs.jsonl')
    records=[{'query_id':p['query_id'],'candidate_id':it['id'],'query':p['query'],'product':it['text']}
             for p in pools for it in p['items']]
    assert len(pools)==250 and len(records)==5160
    df=pd.DataFrame(records)
    lm=LM(model='hosted_vllm/'+a.served_model,api_base=a.endpoint,api_key='local',
          max_ctx_len=4096,max_tokens=16,max_batch_size=16,temperature=0)
    lotus.settings.configure(lm=lm,enable_cache=False)
    start=time.monotonic()
    res=df.sem_filter(INSTRUCTION,return_all=True,return_raw_outputs=True,default=True,suffix='_lotus')
    seconds=time.monotonic()-start
    assert len(res)==len(df) and list(res['query_id'])==list(df['query_id']) and list(res['candidate_id'])==list(df['candidate_id'])
    golds=[];preds=[];qwen=[];counts={'raw_neither_substring':0,'raw_both_substrings':0,'strict_incomplete_pools':0,'raw_empty':0};offset=0
    with (a.out/'predictions.jsonl').open('w',encoding='utf8') as f:
        for pool,ref in zip(pools,reference):
            assert pool['query_id']==ref['query_id'];n=len(pool['items']);part=res.iloc[offset:offset+n];offset+=n
            ys=[int(bool(x)) for x in part['_lotus']];raw=[str(x) if x is not None else '' for x in part['raw_output_lotus']]
            golds.append([int(it['gold']=='Y') for it in pool['items']]);preds.append(ys)
            qwen.append([int(x=='Y') for x in ref['plans']['pointwise']['pred']])
            valid=[]
            for text in raw:
                low=text.lower();t='true' in low;fals='false' in low
                counts['raw_neither_substring']+=int(not t and not fals);counts['raw_both_substrings']+=int(t and fals);counts['raw_empty']+=int(not text.strip())
                words=set(re.findall(r'\b(?:true|false)\b',low));valid.append(len(words)==1)
            counts['strict_incomplete_pools']+=int(not all(valid))
            f.write(json.dumps({'query_id':pool['query_id'],'pred':['Y' if y else 'N' for y in ys],
                                'raw':raw,'strict_valid':valid},ensure_ascii=False)+'\n')
    assert offset==len(records)
    common=[i for i,o in enumerate(reference) if all(o['plans'][m]['pred'] is not None for m in METHODS)]
    assert len(common)==207
    result={'protocol':'PROTOCOL_lotus.md','versions':{x:version(x) for x in ('lotus-ai','litellm','pandas','numpy')},
      'instruction':INSTRUCTION,'candidates':len(records),'query_pools':len(pools),'model':'Qwen3-4B-Instruct-2507',
      'max_tokens':16,'max_concurrency':16,'temperature':0,'sem_filter_default':True,
      'online_client_wall_seconds':seconds,'raw_diagnostics':counts,
      'all_250':summarize(list(range(250)),golds,preds,qwen),'qwen_common_207':summarize(common,golds,preds,qwen)}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
