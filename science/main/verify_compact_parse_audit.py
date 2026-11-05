#!/usr/bin/env python3
"""Verify each saved parser-audit decision against the retained final raw text."""
import argparse,json
from pathlib import Path
from run_validation import parse_compact
from tolerant_compact import parse_delimited_compact
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_compact_parse_audit_smollm')
    ap.add_argument('--ablation',type=Path,default=HERE/'run_compact_prompt_ablation_smollm')
    ap.add_argument('--esci-run',type=Path,default=HERE/'run_full_pools_smollm_250');ap.add_argument('--wands-run',type=Path,default=HERE/'run_wands_smollm_160');a=ap.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());assert summary['protocol']=='PROTOCOL_compact_parse_audit.md'
    for dataset,source in [('esci',a.esci_run),('wands',a.wands_run)]:
        rows=lines(source/'inputs.jsonl');old=lines(source/'outputs.jsonl');new=lines(a.ablation/(dataset+'_outputs.jsonl'));saved=lines(a.run/(dataset+'_predictions.jsonl'))
        assert len(rows)==len(old)==len(new)==len(saved)
        cases=['original_strict','original_delimited','removed_strict','removed_delimited'];valid={c:0 for c in cases};complete={c:0 for c in cases}
        counts={c:[0,0,0,0,0.,0] for c in cases}
        for row,prior,ablation,result in zip(rows,old,new,saved):
            assert row['query_id']==prior['query_id']==ablation['query_id']==result['query_id']
            raw={'original':prior['plans']['two_stage_compact_fallback']['raw'],'removed':ablation['raw']}
            for case in cases:
                prompt,parser_name=case.split('_');parser=parse_compact if parser_name=='strict' else parse_delimited_compact
                pred=[None]*len(row['items'])
                for chunk in raw[prompt]:
                    p=parser(chunk['final'],chunk['size']);valid[case]+=p is not None
                    if p is not None:pred[chunk['start']:chunk['start']+chunk['size']]=p
                if None in pred:pred=None
                assert pred==result['plans'][case]
                if pred is not None:
                    complete[case]+=1;gold=[it['gold'] for it in row['items']];c=counts[case]
                    c[4]+=abs(sum(y=='Y' for y in pred)/len(pred)-sum(g=='Y' for g in gold)/len(gold));c[5]+=1
                    for g,y in zip(gold,pred):c[0]+=g=='Y' and y=='Y';c[1]+=g=='N' and y=='Y';c[2]+=g=='N' and y=='N';c[3]+=g=='Y' and y=='N'
        for case in cases:
            s=summary['datasets'][dataset]['cases'][case];assert complete[case]==s['complete_pools'] and valid[case]==s['valid_final_chunks']
            if complete[case]:
                tp,fp,tn,fn,mae,n=counts[case];q=s['quality_on_own_complete']
                assert [tp,fp,tn,fn]==[q['tp'],q['fp'],q['tn'],q['fn']]
                assert abs(mae/n-q['ratio_mae'])<1e-10
        print(dataset,{k:complete[k] for k in cases},'OK')
    print('Parser-audit decisions, completeness and confusion counts verified.')
if __name__=='__main__':main()
