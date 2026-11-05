#!/usr/bin/env python3
"""Plot the unchanged nine model contrasts with larger, non-overlapping labels."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS=[('Qwen3-4B','#0072B2','o'),('Qwen2.5-7B','#D55E00','s'),('SmolLM2-1.7B','#009E73','^')]
DATASETS=['ESCI','WANDS','TREC-COVID']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 source=json.loads(a.data.read_text());d=source['values']
 if source.get('fixture') is not False or set(d)!=set(DATASETS):raise ValueError('Expected the nine actual sensitivity comparisons')
 for dataset in DATASETS:
  if set(d[dataset])!={x[0] for x in MODELS}:raise ValueError('Missing a model')
  pools={d[dataset][m]['pools'] for m,_,_ in MODELS}
  if len(pools)!=1 or next(iter(pools))<=0:raise ValueError('All models must have the same positive common-pool count in each panel')
  for m,_,_ in MODELS:
   x=d[dataset][m]
   if len(x['interval'])!=2 or x['interval'][0]>x['interval'][1] or not all(math.isfinite(v) for v in [x['difference'],*x['interval'],x['keyed_mae'],x['selective_mae']]) or not all(0<=x[v]<=1 for v in ('keyed_mae','selective_mae')) or not math.isclose(x['selective_mae']-x['keyed_mae'],x['difference'],abs_tol=1e-10):raise ValueError('Inconsistent contrast')
 vals=[0.0]+[v for group in d.values() for x in group.values() for v in [x['difference'],*x['interval']]]
 lo,hi=min(vals),max(vals);span=max(hi-lo,.01);limits=(lo-.14*span,hi+1.12*span)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
 fig,axes=plt.subplots(1,3,figsize=(7.3,2.7),sharex=True,sharey=True);checks=[];heading_checks=[]
 for ax,dataset in zip(axes,DATASETS):
  n=next(iter({d[dataset][m]['pools'] for m,_,_ in MODELS}));ax.axvline(0,color='#555555',lw=.7,zorder=0)
  ax.set_title(f'{dataset} (n={n})',fontsize=9,fontweight='bold',pad=9)
  subtitle=ax.text(.99,.94,'keyed → selective',transform=ax.transAxes,ha='right',va='bottom',fontsize=8.2,color='#444444');heading_checks.append((dataset,ax.title,subtitle))
  for i,(model,color,marker) in enumerate(MODELS):
   y=2-i;row=d[dataset][model];low,high=row['interval'];x=row['difference']
   ax.hlines(y,low,high,color=color,lw=1.3,zorder=2);ax.plot([low,high],[y,y],linestyle='none',marker='|',color=color,markersize=5)
   ax.scatter([x],[y],marker=marker,s=28,color=color,zorder=3)
   label=ax.text(.99,y,f"{row['keyed_mae']:.3f}→{row['selective_mae']:.3f}",transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8.2,color='#444444')
   checks.append((dataset,model,ax,high,y,label))
  ax.set_xlim(*limits);ax.set_yticks([2,1,0]);ax.set_ylim(-.6,2.6);ax.grid(axis='x',color='#dddddd',lw=.5)
  ax.tick_params(axis='both',length=0,labelsize=8.2);ax.set_xlabel('Selective − keyed ratio MAE',fontsize=8.2,labelpad=5)
 axes[0].set_yticklabels([m[0] for m in MODELS]);fig.subplots_adjust(left=.17,right=.99,top=.79,bottom=.24,wspace=.16)
 fig.suptitle('Paired query-level differences; lower is better',fontsize=9,y=.98)
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for dataset,title,subtitle in heading_checks:
  if title.get_window_extent(renderer).overlaps(subtitle.get_window_extent(renderer)):raise ValueError('Panel title overlaps annotation heading: '+dataset)
 for dataset,model,ax,high,y,label in checks:
  line_right=ax.transData.transform((high,y))[0];label_left=label.get_window_extent(renderer).x0
  if line_right+1>=label_left:raise ValueError(f'Confidence bar overlaps numeric label: {dataset} {model}')
 a.out.mkdir(parents=True);fig.savefig(a.out/'model_sensitivity.pdf',bbox_inches='tight');fig.savefig(a.out/'model_sensitivity.png',dpi=300,bbox_inches='tight');plt.close(fig)
 (a.out/'source_data_sha256.txt').write_text(__import__('hashlib').sha256(a.data.read_bytes()).hexdigest()+'\n')
 print('Saved readable nine-contrast figure; labels and bars do not overlap.')
if __name__=='__main__':main()
