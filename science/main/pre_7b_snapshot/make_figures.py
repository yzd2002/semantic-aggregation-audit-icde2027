#!/usr/bin/env python3
"""Build the paper's vector figures from the recorded validation results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
P=Path(__file__).resolve().parent
out=P.parent/'01_manuscript/figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.05,2.15));ax.set_xlim(0,100);ax.set_ylim(0,35);ax.axis('off')
def box(x,y,w,h,txt,fill='#edf3f8',edge='#476b87'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.5',facecolor=fill,edgecolor=edge,linewidth=.8))
 ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=7.4,linespacing=1.25)
def arrow(a,b,style='-',color='#52616b'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,linewidth=.9,linestyle=style,color=color))
box(1,22,12,8,'Query $q$','white','#88949c')
box(20,22,20,8,'Specification\ngeneration')
box(47,22,20,8,'Validate specification\n(retry once)')
box(1,4,12,8,'Candidate\npool $C_q$','white','#88949c')
box(20,4,20,8,'Batch judgment\n(indexed or compact)','#f9f1e4','#9b7950')
box(47,4,20,8,'Validate labels\n(retry once)','#f9f1e4','#9b7950')
box(75,4,23,8,'Ratio, or explicit\nnon-estimable','white','#88949c')
arrow((13,26),(20,26));arrow((40,26),(47,26));arrow((13,8),(20,8));arrow((40,8),(47,8));arrow((67,8),(75,8))
arrow((7,22),(20,12),'--');ax.text(8,17,'direct',fontsize=6.8,ha='center',color='#52616b')
arrow((49,22),(39,12));ax.text(31,18,'spec or fallback',fontsize=6.8,ha='center',color='#52616b')
ax.text(58,0.5,'invalid labels after retry: failure',fontsize=6.8,ha='center',color='#52616b')
fig.savefig(out/'operator.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'operator.png',dpi=200,bbox_inches='tight',pad_inches=.04);plt.close(fig)
report=json.loads((P/'run_full_pools_250/paired_analysis.json').read_text())
supervised=json.loads((P/'run_supervised_baseline_1000/summary.json').read_text())['qwen_common_207']['methods']['supervised_lexical']
lotus=json.loads((P/'run_lotus_baseline/summary.json').read_text())['qwen_common_207']['methods']['lotus_sem_filter']
keys=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']
names=['Pointwise','Batch / JSON','Two-stage / JSON','Two-stage / compact + fallback','LOTUS filter (post hoc)','Supervised lexical (post hoc)']
fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),gridspec_kw={'width_ratios':[1.15,1]},sharey=True)
y=list(range(len(names)))[::-1]
for axis,metric,title in [(axes[0],'f1','Candidate F1 (higher is better)'),(axes[1],'mae','Pool-ratio MAE (lower is better)')]:
 for yi,k in zip(y,keys):
  m=report['methods'][k];v=m[metric];lo,hi=m[metric+'_ci'];axis.plot([lo,hi],[yi,yi],color='#466d86',lw=1.4,zorder=2);axis.scatter(v,yi,s=18,color='#224e6b',zorder=3)
 sk='ratio_mae' if metric=='mae' else metric
 for yi,data,color in [(y[-2],lotus,'#36866c'),(y[-1],supervised,'#a96555')]:
  v=data[sk];lo,hi=data[sk+'_ci'];axis.plot([lo,hi],[yi,yi],color=color,lw=1.4,zorder=2);axis.scatter(v,yi,s=18,color=color,zorder=3)
 axis.axvline(report['methods']['always_yes'][metric],color='#a96555',linestyle='--',linewidth=0.9,label='Always Y')
 axis.axvline(report['methods']['lexical_all_terms'][metric],color='#777777',linestyle=':',linewidth=1.1,label='Lexical')
 axis.set_title(title,fontsize=8.2,pad=7);axis.grid(axis='x',color='#e3e7e9',linewidth=.6);axis.set_axisbelow(True);axis.tick_params(labelsize=7,length=0)
 axis.set_ylim(-.6,len(names)-.4)
axes[0].set_yticks(y,names,fontsize=7.1);axes[1].tick_params(labelleft=False)
axes[0].legend(loc='lower left',frameon=True,framealpha=0.9,fontsize=6.2);axes[1].legend(loc='upper right',frameon=True,framealpha=0.9,fontsize=6.2)
axes[0].set_xlim(0,1);axes[1].set_xlim(0,.8)
fig.tight_layout(pad=.6,w_pad=1.4)
fig.savefig(out/'quality.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'quality.png',dpi=200,bbox_inches='tight',pad_inches=.04);plt.close(fig)
wands=json.loads((P/'run_wands_160/paired_analysis.json').read_text())
names=['Pointwise','Batch / JSON','Two-stage / JSON','Two-stage / compact + fallback']
fig,axes=plt.subplots(1,2,figsize=(7.05,2.35),gridspec_kw={'width_ratios':[1.15,1]},sharey=True)
y=list(range(len(keys)))[::-1]
for axis,metric,title in [(axes[0],'f1','Candidate F1 (higher is better)'),(axes[1],'mae','Pool-ratio MAE (lower is better)')]:
 for yi,k in zip(y,keys):
  m=wands['methods'][k];v=m[metric];lo,hi=m[metric+'_ci'];axis.plot([lo,hi],[yi,yi],color='#466d86',lw=1.4,zorder=2);axis.scatter(v,yi,s=18,color='#224e6b',zorder=3)
 axis.axvline(wands['methods']['always_no'][metric],color='#a96555',linestyle='--',linewidth=.9,label='Always N')
 axis.axvline(wands['methods']['lexical_all_terms'][metric],color='#777777',linestyle=':',linewidth=1.1,label='All query words')
 axis.set_title(title,fontsize=8.2,pad=7);axis.grid(axis='x',color='#e3e7e9',linewidth=.6);axis.set_axisbelow(True);axis.tick_params(labelsize=7,length=0);axis.set_ylim(-.6,len(keys)-.4)
axes[0].set_yticks(y,names,fontsize=7.1);axes[1].tick_params(labelleft=False)
axes[0].legend(loc='upper left',frameon=True,framealpha=.9,fontsize=6.2);axes[1].legend(loc='upper right',frameon=True,framealpha=.9,fontsize=6.2)
axes[0].set_xlim(0,.8);axes[1].set_xlim(0,.45);fig.tight_layout(pad=.6,w_pad=1.4)
fig.savefig(out/'wands_quality.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'wands_quality.png',dpi=200,bbox_inches='tight',pad_inches=.04);plt.close(fig)
order_qwen=json.loads((P/'run_reverse_order_qwen_v1/paired_analysis.json').read_text())['datasets']
order_smol=json.loads((P/'run_reverse_order_smollm_v1/paired_analysis.json').read_text())['datasets']
studies=[('Qwen / ESCI',order_qwen['esci'],'#224e6b'),('Qwen / WANDS',order_qwen['wands'],'#224e6b'),
         ('SmolLM2 / ESCI',order_smol['esci'],'#a96555'),('SmolLM2 / WANDS',order_smol['wands'],'#a96555')]
fig,axes=plt.subplots(1,2,figsize=(7.05,2.25),gridspec_kw={'width_ratios':[1.15,1]},sharey=True)
y=list(range(len(studies)))[::-1]
for axis,metric,interval,title in [(axes[0],'candidate_label_change_fraction','candidate_label_change_fraction_ci','Candidate labels changed (%)'),
                                   (axes[1],'mean_absolute_pool_ratio_change','mean_absolute_pool_ratio_change_ci','Mean absolute ratio change (pp)')]:
 for yi,(_,row,color) in zip(y,studies):
  value=100*row[metric];lo,hi=(100*x for x in row['paired_bootstrap'][interval])
  axis.plot([lo,hi],[yi,yi],color=color,lw=1.4,zorder=2);axis.scatter(value,yi,s=23,color=color,zorder=3)
 axis.set_title(title,fontsize=8.2,pad=7);axis.grid(axis='x',color='#e3e7e9',linewidth=.6);axis.set_axisbelow(True)
 axis.tick_params(labelsize=7,length=0);axis.set_ylim(-.6,len(studies)-.4)
 limit=max(row['paired_bootstrap'][interval][1] for _,row,_ in studies)*100
 axis.set_xlim(0,min(100,max(10,((int(limit/10)+1)*10))))
axes[0].set_yticks(y,[name for name,_,_ in studies],fontsize=7.3);axes[1].tick_params(labelleft=False)
fig.tight_layout(pad=.6,w_pad=1.4)
fig.savefig(out/'order_stability.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'order_stability.png',dpi=200,bbox_inches='tight',pad_inches=.04);plt.close(fig)
print('Wrote vector and preview figures to',out)
