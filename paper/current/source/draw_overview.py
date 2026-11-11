#!/usr/bin/env python3
"""Measured GLM Sports running example from frozen execution readouts."""
from __future__ import annotations
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'savefig.facecolor':'white','axes.linewidth':.6})
INK='#192B40'; CHECK='#664E82'; PLAN='#235D74'; GREY='#69727D'
RULE='#BBC3CC'; LIGHT='#F4F6F8'; PALE='#F2EFF6'; WHITE='#FFFFFF'
ALL_TEXT=[]

def txt(ax,x,y,s,size=8,**kw):
    t=ax.text(x,y,s,fontsize=size,color=kw.pop('color',INK),
              va=kw.pop('va','center'),**kw)
    ALL_TEXT.append((ax,t))
    return t

def box(ax,x,y,w,h,fill=WHITE,edge=RULE,lw=.7,ls='-'):
    p=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=ls)
    ax.add_patch(p)
    return p

def arrow(ax,x1,y1,x2,y2,color=GREY,lw=.8,ls='-'):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,
                      color=color,linewidth=lw,linestyle=ls)
    ax.add_patch(p)

def render(fig,stem):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    out=[]
    for ax,t in ALL_TEXT:
        if t.figure is not fig:
            continue
        extent=t.get_window_extent(renderer)
        f=fig.bbox
        if extent.x0 < f.x0-.5 or extent.x1 > f.x1+.5 or extent.y0 < f.y0-.5 or extent.y1 > f.y1+.5:
            out.append(t.get_text())
    if out:
        raise RuntimeError('text outside figure: '+repr(out))
    placed=[t for ax,t in ALL_TEXT if t.figure is fig]
    collisions=[]
    for i,t in enumerate(placed):
        a=t.get_window_extent(renderer)
        for u in placed[i+1:]:
            b=u.get_window_extent(renderer)
            dx=min(a.x1,b.x1)-max(a.x0,b.x0)
            dy=min(a.y1,b.y1)-max(a.y0,b.y0)
            if dx>1 and dy>1:
                collisions.append((t.get_text(),u.get_text()))
    if collisions:
        raise RuntimeError('text collision: '+repr(collisions))
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/(stem+'.'+ext),dpi=240)
    plt.close(fig)


def overview():
    import csv
    with (ROOT/'figure_inputs/all_main_policy_readouts.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    def record(policy):
        return next(x for x in rows if x['model']=='glm' and x['workload_id']=='agnews_count_sports' and x['n']=='2048' and x['prefix_cache']=='True' and x['contract']=='native' and x['policy']==policy)
    single=record('S1');batch=record('RANDOM8');checked=record('CURRENT_CP')
    N=int(checked['n']);base_calls=int(single['requests']);deploy_calls=int(batch['requests']);full_calls=int(checked['requests']);check_calls=full_calls-deploy_calls
    base_tokens=int(single['total_tokens']);deploy_tokens=int(batch['total_tokens']);full_tokens=int(checked['total_tokens']);check_tokens=full_tokens-deploy_tokens
    mismatches=round(N*float(checked['loss_to_S1']))
    assert (N,base_calls,deploy_calls,check_calls,full_calls)==(2048,2048,256,160,416)
    assert (base_tokens,deploy_tokens,check_tokens,full_tokens,mismatches)==(338365,255425,93120,348545,17)
    assert round(N*float(checked['loss_to_S2']))==17
    from figure_style import draw_overview
    draw_overview(OUT, {'base_tokens':base_tokens,'check_tokens':check_tokens,'deploy_tokens':deploy_tokens})



if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    overview()
    paths=[OUT/('contract_execution_overview.'+ext) for ext in ('pdf','svg','png')]
    manifest={'status':'MEASURED_SPORTS_RUNNING_EXAMPLE','empirical_values':True,
        'source_csv_sha256':hashlib.sha256((ROOT/'figure_inputs/all_main_policy_readouts.csv').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (OUT/'FIGURE1_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
