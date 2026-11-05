#!/usr/bin/env python3
"""Diagnose completion bounds on failed pools; no semantic coverage is guaranteed."""
import argparse,json,statistics
from pathlib import Path
from run_experiment import parse_batch
from run_validation import parse_compact
from ratio_bounds import ratio_bounds
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    rows=lines(a.run/'inputs.jsonl');outs=lines(a.run/'outputs.jsonl');assert len(rows)==len(outs)
    methods=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback'];result={};details=[]
    for method in methods:
        failed=[]
        for row,out in zip(rows,outs):
            assert row['query_id']==out['query_id'];entry=out['plans'][method]
            if entry['pred'] is not None:continue
            n=len(row['items']);accepted=[None]*n
            if method=='pointwise':
                for i,x in enumerate(entry['raw']):
                    if x['final'] in ('Y','N'):accepted[i]=x['final']
            else:
                parser=parse_compact if 'compact' in method else parse_batch;occupied=set()
                for c in entry['raw']:
                    indices=set(range(c['start'],c['start']+c['size']))
                    assert min(indices,default=0)>=0 and max(indices,default=-1)<n and not occupied&indices
                    occupied.update(indices);v=parser(c['final'],c['size'])
                    if v is not None:accepted[c['start']:c['start']+c['size']]=v
            m=sum(x is not None for x in accepted);p=sum(x=='Y' for x in accepted);assert m<n
            reference=sum(it['gold']=='Y' for it in row['items'])
            wrong=sum(v is not None and v!=it['gold'] for v,it in zip(accepted,row['items']))
            lower,upper=ratio_bounds(n,m,p,k=0)
            rec={'method':method,'query_id':row['query_id'],'n':n,'accepted':m,'accepted_positive':p,
                 'lower':lower,'upper':upper,'width':upper-lower,
                 'reference_positive':reference,'reference_inside':p<=reference<=p+n-m,
                 'accepted_mistakes':wrong}
            failed.append(rec);details.append(rec)
        if failed:
            count=len(failed);acc=sum(x['accepted'] for x in failed);mistakes=sum(x['accepted_mistakes'] for x in failed)
            result[method]={'failed_pools':count,'accepted_labels':acc,'accepted_label_mistakes':mistakes,
                'accepted_label_error_rate':mistakes/acc if acc else None,
                'mean_interval_width':statistics.mean(x['width'] for x in failed),
                'median_interval_width':statistics.median(x['width'] for x in failed),
                'reference_ratio_inside_count':sum(x['reference_inside'] for x in failed),
                'reference_ratio_inside_fraction':sum(x['reference_inside'] for x in failed)/count,
                'pools_with_any_accepted_label':sum(x['accepted']>0 for x in failed)}
        else:result[method]={'failed_pools':0}
    (a.run/'failure_bounds.json').write_text(json.dumps({'description':'Model-label completion intervals on failed pools; reference inclusion is descriptive and is not a semantic guarantee.','methods':result},indent=2)+'\n')
    with (a.run/'failure_bounds_details.jsonl').open('w',encoding='utf8') as f:
        for x in details:f.write(json.dumps(x)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
