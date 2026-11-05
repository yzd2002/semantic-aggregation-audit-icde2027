#!/usr/bin/env python3
"""Reproduce LOTUS 1.2.4 filter parsing, strict diagnostics and saved metrics."""
import argparse,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(path):
    with path.open(encoding='utf8') as f:return [json.loads(line) for line in f]
def metric(golds,preds):
    tp=fp=tn=fn=0;mae=[]
    for g,p in zip(golds,preds):
        assert len(g)==len(p) and len(g)>0
        mae.append(abs(sum(x=='Y' for x in g)/len(g)-sum(x=='Y' for x in p)/len(p)))
        for x,y in zip(g,p):
            tp+=x=='Y' and y=='Y';fp+=x=='N' and y=='Y';tn+=x=='N' and y=='N';fn+=x=='Y' and y=='N'
    return {'f1':2*tp/(2*tp+fp+fn),'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2,'ratio_mae':sum(mae)/len(mae)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_lotus_baseline');a=ap.parse_args()
    pools=read(HERE/'run_full_pools_250/inputs.jsonl');reference=read(HERE/'run_full_pools_250/outputs.jsonl')
    results=read(a.run/'predictions.jsonl');summary=json.loads((a.run/'summary.json').read_text())
    assert len(pools)==len(reference)==len(results)==250
    counts={'postprocessed_defaulted_candidates':0,'postprocessed_both_tokens':0,
            'strict_incomplete_pools':0,'raw_neither_substring':0,'raw_both_substrings':0,'raw_empty':0}
    for pool,record in zip(pools,results):
        assert pool['query_id']==record['query_id'] and len(pool['items'])==len(record['pred'])==len(record['raw'])==len(record['strict_valid'])
        valid=[]
        for raw,pred,was_valid in zip(record['raw'],record['pred'],record['strict_valid']):
            assert isinstance(raw,str)
            # lotus.sem_ops.postprocessors.cot_postprocessor first extracts text after the first literal Answer:.
            answer=raw.split('Answer:',1)[1].strip() if 'Answer:' in raw else raw.strip()
            low=answer.lower();pos='true' in low;neg='false' in low
            counts['postprocessed_defaulted_candidates']+=int(not pos and not neg)
            counts['postprocessed_both_tokens']+=int(pos and neg)
            expected='Y' if pos or not neg else 'N'
            assert pred==expected
            rawlow=raw.lower();rawpos='true' in rawlow;rawneg='false' in rawlow
            counts['raw_neither_substring']+=int(not rawpos and not rawneg)
            counts['raw_both_substrings']+=int(rawpos and rawneg);counts['raw_empty']+=int(not raw.strip())
            okay=len(set(re.findall(r'\b(?:true|false)\b',rawlow)))==1
            assert okay==was_valid;valid.append(okay)
        counts['strict_incomplete_pools']+=int(not all(valid))
    for k,v in summary['raw_diagnostics'].items():assert counts[k]==v,(k,v,counts[k])
    methods=('pointwise','batch_json','two_stage_json','two_stage_compact_fallback')
    common=[i for i,o in enumerate(reference) if all(o['plans'][m]['pred'] is not None for m in methods)]
    assert len(common)==207
    for subset,idx in [('all_250',list(range(250))),('qwen_common_207',common)]:
        golds=[[it['gold'] for it in pools[i]['items']] for i in idx]
        plans={'lotus_sem_filter':[results[i]['pred'] for i in idx],
               'qwen_pointwise':[reference[i]['plans']['pointwise']['pred'] for i in idx]}
        for name,preds in plans.items():
            for k,v in metric(golds,preds).items():assert abs(v-summary[subset]['methods'][name][k])<1e-10
    (a.run/'parse_audit.json').write_text(json.dumps(counts,indent=2)+'\n')
    print('LOTUS parsing and summaries independently verified:',json.dumps(counts))
if __name__=='__main__':main()
