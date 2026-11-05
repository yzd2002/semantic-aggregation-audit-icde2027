#!/usr/bin/env python3
"""Plot both fixed model outcomes and simple/supervised references on new ESCI pools."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen',type=Path,required=True)
    p.add_argument('--smollm',type=Path,required=True)
    p.add_argument('--supervised',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out-prefix',type=Path,required=True)
    a=p.parse_args()
    q=json.loads((a.qwen/'summary.json').read_text());s=json.loads((a.smollm/'summary.json').read_text())
    ref=json.loads((a.supervised/'summary.json').read_text())
    with (a.selection/'inputs.jsonl').open(encoding='utf8') as f:
        inputs=[json.loads(line) for line in f]
    positives=sum(x['gold']=='Y' for r in inputs for x in r['items'])
    candidates=sum(len(r['items']) for r in inputs)
    if q['common_query_ids']!=s['common_query_ids'] or q['common_complete_pools']!=250:
        raise ValueError('This figure requires identical complete model subsets.')
    if [r['query_id'] for r in inputs]!=q['common_query_ids'] or candidates!=q['candidate_count']:
        raise ValueError('Figure inputs differ from saved analyses.')
    for suffix in ('.png','.pdf'):
        if a.out_prefix.with_suffix(suffix).exists():raise FileExistsError('Use a new plot path.')
    methods=[('original','Keyed original'),('actual_fallback','Selective fallback'),('pointwise','Full pointwise'),
             ('lexical','All query words'),('always_Y','All Y'),('supervised','Supervised (50k labels)')]
    colors=['#4c78a8','#008b77','#595959','#b0b0b0','#bf8f32','#8663a8']
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.9),sharex=True)
    for row,(model,key,result) in enumerate((('Qwen','qwen',q),('SmolLM2','smollm',s))):
        quality=dict(result['quality_common']);quality['supervised']=ref['models'][key]['supervised_quality_common']
        contrast=result['comparisons']['actual_fallback_minus_original']
        for col,(metric,title) in enumerate((('f1','Candidate F1 ↑'),('ratio_mae','Query-macro ratio MAE ↓'))):
            ax=axes[row,col]
            vals=[quality[k][metric] for k,_ in methods]
            bars=ax.barh(range(len(methods)),vals,color=colors,height=.65)
            ax.set_yticks(range(len(methods)),[label for _,label in methods] if col==0 else [])
            ax.invert_yaxis();ax.set_xlim(0,1.06);ax.set_xticks([0,.2,.4,.6,.8,1.])
            ax.grid(axis='x',color='#d9d9d9',linewidth=.6);ax.set_axisbelow(True)
            for bar,value in zip(bars,vals):ax.text(value+.015,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center',fontsize=8)
            for side in ('top','right','left'):ax.spines[side].set_visible(False)
            ax.tick_params(axis='y',length=0,labelsize=8);ax.tick_params(axis='x',labelsize=8)
            if row==0:ax.set_title(title,fontsize=10,weight='bold')
            if col==0:ax.text(-.69,1.08,model,transform=ax.transAxes,fontsize=10,weight='bold')
            if col==1:
                lo,hi=contrast['ratio_mae_difference_ci'];d=contrast['ratio_mae_difference']
                ax.text(.01,-.19,f'Selective − keyed: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]',
                        transform=ax.transAxes,fontsize=8)
    fig.suptitle('New ESCI pools: accuracy, ratio error and reference plans',fontsize=13,weight='bold',y=.98)
    fig.text(.5,.02,f'250 shared complete pools · {candidates:,} labeled candidates, {positives:,} E/S positive · 95% paired query-bootstrap intervals',
             ha='center',fontsize=8)
    fig.text(.5,.003,'Public relevance proxy; selected judged pools, not live purchase satisfaction. Two models use different families and sizes.',
             ha='center',fontsize=7.4)
    fig.subplots_adjust(left=.24,right=.98,top=.88,bottom=.105,wspace=.10,hspace=.56)
    a.out_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out_prefix.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(a.out_prefix.with_suffix('.png'),dpi=180,bbox_inches='tight')
    print('Saved figure with both models and all fixed reference plans.')


if __name__=='__main__':
    main()
