#!/usr/bin/env python3
"""Build the paper's vector figures from the recorded validation results."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
P=Path(__file__).resolve().parent
out=Path(os.environ.get('INTENTSAT_FIGURE_OUTPUT_DIR',P.parent/'01_manuscript/figures'));out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.05,2.28));ax.set_xlim(-1,101);ax.set_ylim(0,100);ax.axis('off')
blue='#426781'; gold='#967043';ink='#25343d';muted='#53636d'
def box(x,y,w,h,label,fill,edge):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.6',facecolor=fill,edgecolor=edge,linewidth=.85))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=7.4,linespacing=1.15,color=ink)
def arrow(x1,y1,x2,y2):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,linewidth=.9,color=muted))
ax.text(0,96,'A  Batch plans and output contract',fontsize=8.5,fontweight='bold',color=ink,va='center')
box(0,76,20,14,'Query $q$','white','#8999a2')
box(0,58,20,14,'Candidate pool $C_q$','white','#8999a2')
box(26,76,20,14,'Optional spec.\nvalidate / retry','#edf3f8',blue)
box(52,76,20,14,'Batch judgments\nvalidate / retry','#f9f1e4',gold)
box(78,76,22,14,'Complete: ratio\notherwise: abstain','white','#8999a2')
for x in (20,46,72):arrow(x,83,x+6,83)
arrow(20,65,52,76)
ax.text(50,51,'Invalid spec: direct fallback or abstain, depending on plan.',ha='center',fontsize=6.9,color=muted)
ax.text(0,40,'B  Selective plan: same candidates, chunks and stable IDs',fontsize=8.5,fontweight='bold',color=ink,va='center')
box(0,15,20,18,'Keyed outputs:\noriginal / reverse','#edf3f8',blue)
box(26,15,20,18,'Align labels\nby candidate','#edf3f8',blue)
box(52,15,20,18,'Agree: keep label\nElse: pointwise','#f9f1e4',gold)
box(78,15,22,18,'Assemble labels\nratio if complete','white','#8999a2')
for x in (20,46,72):arrow(x,24,x+6,24)
ax.text(50,4,'Invalid keyed output or unresolved required judgment after retry: abstain.',ha='center',fontsize=6.9,color=muted)
fig.savefig(out/'operator.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'operator.png',dpi=300,bbox_inches='tight',pad_inches=.04);plt.close(fig)
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
 axis.set_title(title,fontsize=9.1,pad=7);axis.grid(axis='x',color='#e3e7e9',linewidth=.6);axis.set_axisbelow(True);axis.tick_params(labelsize=8.1,length=0)
 axis.set_ylim(-.6,len(names)-.4)
axes[0].set_yticks(y,names,fontsize=8.6);axes[1].tick_params(labelleft=False)
axes[0].legend(loc='lower left',frameon=True,framealpha=0.9,fontsize=8.4);axes[1].legend(loc='upper right',frameon=True,framealpha=0.9,fontsize=8.4)
axes[0].set_xlim(0,1);axes[1].set_xlim(0,.8)
fig.tight_layout(pad=.6,w_pad=1.4)
fig.savefig(out/'quality.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'quality.png',dpi=300,bbox_inches='tight',pad_inches=.04);plt.close(fig)
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
 axis.set_title(title,fontsize=9.1,pad=7);axis.grid(axis='x',color='#e3e7e9',linewidth=.6);axis.set_axisbelow(True)
 axis.tick_params(labelsize=8.1,length=0);axis.set_ylim(-.6,len(studies)-.4)
 limit=max(row['paired_bootstrap'][interval][1] for _,row,_ in studies)*100
 axis.set_xlim(0,min(100,max(10,((int(limit/10)+1)*10))))
axes[0].set_yticks(y,[name for name,_,_ in studies],fontsize=8.6);axes[1].tick_params(labelleft=False)
fig.tight_layout(pad=.6,w_pad=1.4)
fig.savefig(out/'order_stability.pdf',bbox_inches='tight',pad_inches=.04)
fig.savefig(out/'order_stability.png',dpi=300,bbox_inches='tight',pad_inches=.04);plt.close(fig)
print('Wrote vector and preview figures to',out)
