#!/usr/bin/env python3
"""Compare strict and delimiter-tolerant parsing on retained model outputs."""
import argparse,json
from pathlib import Path
from run_validation import parse_compact
from tolerant_compact import parse_delimited_compact
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def metric(rows,preds,idx):
    tp=fp=tn=fn=0;mae=[]
    for i in idx:
        gold=[it['gold'] for it in rows[i]['items']];p=preds[i];assert len(gold)==len(p)
        mae.append(abs(sum(x=='Y' for x in p)/len(p)-sum(x=='Y' for x in gold)/len(gold)))
        for g,y in zip(gold,p):tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn) and (tn+fp) else None,
            'ratio_mae':sum(mae)/len(mae) if mae else None}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ablation',type=Path,default=HERE/'run_compact_prompt_ablation_smollm')
    ap.add_argument('--esci-run',type=Path,default=HERE/'run_full_pools_smollm_250');ap.add_argument('--wands-run',type=Path,default=HERE/'run_wands_smollm_160')
    ap.add_argument('--out',type=Path,default=Path('run_compact_parse_audit_smollm'));a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    summary={'protocol':'PROTOCOL_compact_parse_audit.md','model':'SmolLM2-1.7B-Instruct','datasets':{}}
    for name,source in [('esci',a.esci_run),('wands',a.wands_run)]:
        rows=lines(source/'inputs.jsonl');original=lines(source/'outputs.jsonl');removed=lines(a.ablation/(name+'_outputs.jsonl'))
        assert len(rows)==len(original)==len(removed)
        cases={'original_strict':[],'original_delimited':[],'removed_strict':[],'removed_delimited':[]}
        valid={k:0 for k in cases};chunk_counts={k:0 for k in cases}
        for row,old,new in zip(rows,original,removed):
            assert row['query_id']==old['query_id']==new['query_id'];original_chunks=old['plans']['two_stage_compact_fallback']['raw'];new_chunks=new['raw']
            for name_case,raw,parser in [('original_strict',original_chunks,parse_compact),('original_delimited',original_chunks,parse_delimited_compact),
                                          ('removed_strict',new_chunks,parse_compact),('removed_delimited',new_chunks,parse_delimited_compact)]:
                pred=[None]*len(row['items']);occupied=set()
                for c in raw:
                    ids=set(range(c['start'],c['start']+c['size']));assert not occupied&ids and max(ids)<len(pred);occupied|=ids
                    decoded=parser(c['final'],c['size']);chunk_counts[name_case]+=1;valid[name_case]+=decoded is not None
                    if decoded is not None:pred[c['start']:c['start']+c['size']]=decoded
                cases[name_case].append(pred if None not in pred else None)
            assert cases['original_strict'][-1]==old['plans']['two_stage_compact_fallback']['pred']
            assert cases['removed_strict'][-1]==new['pred']
        ds={'attempted_pools':len(rows),'cases':{},'pairwise_intersections':{}}
        with (a.out/(name+'_predictions.jsonl')).open('w',encoding='utf8') as f:
            for i,row in enumerate(rows):f.write(json.dumps({'query_id':row['query_id'],'plans':{k:v[i] for k,v in cases.items()}})+'\n')
        for case,preds in cases.items():
            idx=[i for i,p in enumerate(preds) if p is not None]
            stats={'complete_pools':len(idx),'valid_final_chunks':valid[case],'attempted_chunks':chunk_counts[case]}
            if idx:
                stats['quality_on_own_complete']=metric(rows,preds,idx)
                for label,tag in [('Y','always_yes'),('N','always_no')]:
                    constants=[[label]*len(row['items']) for row in rows]
                    stats[tag+'_on_same_pools']=metric(rows,constants,idx)
            ds['cases'][case]=stats
        pairs=[('original_strict','original_delimited'),('removed_strict','removed_delimited'),('original_delimited','removed_delimited')]
        for left,right in pairs:
            idx=[i for i in range(len(rows)) if cases[left][i] is not None and cases[right][i] is not None]
            entry={'common_pools':len(idx)}
            if idx:entry['left_quality']=metric(rows,cases[left],idx);entry['right_quality']=metric(rows,cases[right],idx)
            ds['pairwise_intersections'][left+'__'+right]=entry
        summary['datasets'][name]=ds
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
