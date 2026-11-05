#!/usr/bin/env python3
"""Independently check supervised selection, leakage guard, predictions and saved metrics."""
import argparse,json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(line) for line in f]
def metrics(gold,pred):
    tp=fp=tn=fn=0;maes=[]
    for g,p in zip(gold,pred):
        assert len(g)==len(p) and len(g)>0
        maes.append(abs(sum(x=='Y' for x in g)/len(g)-sum(x=='Y' for x in p)/len(p)))
        for x,y in zip(g,p):
            tp+=x=='Y' and y=='Y';fp+=x=='N' and y=='Y';tn+=x=='N' and y=='N';fn+=x=='Y' and y=='N'
    return {'f1':2*tp/(2*tp+fp+fn),'balanced_accuracy':((tp/(tp+fn))+(tn/(tn+fp)))/2,'ratio_mae':sum(maes)/len(maes)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_supervised_baseline')
    ap.add_argument('--examples',type=Path,default=Path('/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_examples.parquet'))
    a=ap.parse_args();manifest=lines(a.run/'train_manifest.jsonl');preds=lines(a.run/'predictions.jsonl')
    summary=json.loads((a.run/'summary.json').read_text());test=lines(HERE/'run_full_pools_250/inputs.jsonl');outputs=lines(HERE/'run_full_pools_250/outputs.jsonl')
    prior=set()
    for name in ('run_test_160','run_validation_500','run_full_pools_250'):
        prior.update(x['query'].strip().casefold() for x in lines(HERE/name/'inputs.jsonl'))
    assert len(manifest)==50000 and not {r['query'].strip().casefold() for r in manifest}&prior
    ex=pd.read_parquet(a.examples,columns=['example_id','query','query_id','product_id','product_locale','esci_label','small_version','split'],
         filters=[('product_locale','=','us'),('small_version','=',1),('split','=','train')])
    ex=ex[ex.esci_label.isin(['E','S','C','I']) & ex['query'].notna()].copy()
    normalized=ex['query'].str.strip().str.casefold();ex=ex[normalized.ne('') & ~normalized.isin(prior)].sort_values('example_id')
    ex=ex.sample(n=50000,random_state=20271106)
    expected=[{'example_id':str(x.example_id),'query_id':str(x.query_id),'query':x.query,
               'product_id':str(x.product_id),'grade':x.esci_label} for x in ex.itertuples(index=False)]
    assert manifest==expected,'Training sample could not be reproduced'
    assert len(preds)==len(test)==len(outputs)==250
    for pool,p in zip(test,preds):
        assert pool['query_id']==p['query_id'] and len(pool['items'])==len(p['pred'])==len(p['positive_probability'])
        assert all(0<=v<=1 and label==('Y' if v>=0.5 else 'N') for label,v in zip(p['pred'],p['positive_probability']))
    plans=('pointwise','batch_json','two_stage_json','two_stage_compact_fallback')
    common=[i for i,o in enumerate(outputs) if all(o['plans'][m]['pred'] is not None for m in plans)]
    for subset,idx in [('all_250',list(range(250))),('qwen_common_207',common)]:
        gold=[[x['gold'] for x in test[i]['items']] for i in idx]
        for method,pr in [('supervised_lexical',[preds[i]['pred'] for i in idx]),('qwen_pointwise',[outputs[i]['plans']['pointwise']['pred'] for i in idx])]:
            actual=metrics(gold,pr);expected=summary[subset]['methods'][method]
            for k,v in actual.items():assert abs(v-expected[k])<1e-10,(subset,method,k,v,expected[k])
    train_ids={r['product_id'] for r in manifest};test_ids={it['product_id'] for r in test for it in r['items']}
    print('Supervised selection, query isolation, predictions and summaries verified.')
    print('Training examples:',len(manifest),'evaluation pools:',len(test),'shared train/test product IDs:',len(train_ids&test_ids))
if __name__=='__main__':main()
