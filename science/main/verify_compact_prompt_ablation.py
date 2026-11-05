#!/usr/bin/env python3
"""Reparse saved ablation chunks and verify completion and quality summaries."""
import argparse,json
from pathlib import Path
from run_validation import parse_compact
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def metrics(rows,preds,idx):
    tp=fp=tn=fn=0;mae=[]
    for i in idx:
        gold=[it['gold'] for it in rows[i]['items']];p=preds[i];assert len(gold)==len(p)
        mae.append(abs(sum(x=='Y' for x in gold)/len(gold)-sum(x=='Y' for x in p)/len(p)))
        for g,y in zip(gold,p):tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn) and (tn+fp) else None,
            'ratio_mae':sum(mae)/len(mae) if mae else None}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_compact_prompt_ablation_smollm')
    ap.add_argument('--esci-run',type=Path,default=HERE/'run_full_pools_smollm_250');ap.add_argument('--wands-run',type=Path,default=HERE/'run_wands_smollm_160');a=ap.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());assert summary['protocol']=='PROTOCOL_compact_prompt_ablation.md'
    for name,source in [('esci',a.esci_run),('wands',a.wands_run)]:
        rows=lines(source/'inputs.jsonl');original=lines(source/'outputs.jsonl');outputs=lines(a.run/(name+'_outputs.jsonl'))
        assert len(rows)==len(original)==len(outputs);preds=[];first_valid=final_valid=example_new=chunk_count=0
        for row,old,new in zip(rows,original,outputs):
            assert row['query_id']==old['query_id']==new['query_id'];recovered=[None]*len(row['items']);seen=set()
            for c in new['raw']:
                chunk_count+=1
                indices=set(range(c['start'],c['start']+c['size']));assert not seen&indices and max(indices)<len(recovered);seen|=indices
                first_valid+=parse_compact(c['first'],c['size']) is not None;parsed=parse_compact(c['final'],c['size']);final_valid+=parsed is not None
                example_new+=c['final'].strip() in ('YNNY','YNNY.')
                if parsed is not None:recovered[c['start']:c['start']+c['size']]=parsed
            pred=recovered if None not in recovered else None;assert pred==new['pred'];preds.append(pred)
        idx=[i for i,p in enumerate(preds) if p is not None]
        prior=[i for i,o in enumerate(original) if o['plans']['two_stage_compact_fallback']['pred'] is not None]
        stats=summary['datasets'][name];assert len(idx)==stats['corrected_complete_pools'] and len(prior)==stats['original_complete_pools']
        assert len(set(idx)&set(prior))==stats['common_complete_pools']
        example_old=sum(c['final'].strip() in ('YNNY','YNNY.') for o in original for c in o['plans']['two_stage_compact_fallback']['raw'])
        assert chunk_count==stats['chunks'] and first_valid==stats['valid_first_chunks'] and final_valid==stats['valid_final_chunks']
        assert example_new==stats['literal_example_final_corrected'] and example_old==stats['literal_example_final_original']
        if idx:
            measured=metrics(rows,preds,idx);saved=stats['corrected_quality_on_own_complete']
            for k,v in measured.items():
                if v is None:assert saved[k] is None
                else:assert abs(v-saved[k])<1e-10,(name,k,v,saved[k])
        print(name,'original complete',len(prior),'corrected complete',len(idx),'valid final chunks',final_valid,'OK')
    print('Compact-example ablation outputs and summaries independently verified.')
if __name__=='__main__':main()
