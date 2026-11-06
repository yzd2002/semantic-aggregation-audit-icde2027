#!/usr/bin/env python3
"""Original semantic-COUNT execution overview; identities are schematic."""
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
    fig,ax=plt.subplots(figsize=(7.16,2.46))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,7.16),ylim=(.72,3.0));ax.axis('off')
    # A specific, source-derived logical query, followed by its real operator input.
    box(ax,.02,2.46,7.12,.50,fill=LIGHT,edge=INK)
    txt(ax,.12,2.82,'Movie COUNT: a source-derived query variant',size=8.6,fontweight='bold')
    txt(ax,.12,2.61,'Prepared T (top-critic view)  →  sem_filter("clearly positive")  →  COUNT(*)',
        size=7.8,fontfamily='DejaVu Sans Mono')
    # Query/input strip connects both the checker and deployment.
    arrow(ax,1.12,2.45,1.12,2.27)
    arrow(ax,3.59,2.45,3.59,2.27)
    txt(ax,5.98,2.33,'Same full input T',size=7.5,ha='center',color=GREY)
    # Three horizontally aligned objects. The nominal object has a dashed outline.
    x1,x2,x3=.02,2.49,4.96
    box(ax,x1,.76,2.20,1.49,fill=PALE,edge=CHECK,ls='--')
    box(ax,x2,.76,2.20,1.49,fill=WHITE,edge=PLAN)
    box(ax,x3,.76,2.18,1.49,fill=WHITE,edge=INK)
    txt(ax,x1+.10,2.11,'a  Existing fidelity check',size=8.4,fontweight='bold',color=CHECK)
    txt(ax,x1+.10,1.88,'n fresh singleton references',size=7.8)
    # Abstract probe slots; these are identities, not verdicts.
    for row in range(2):
        yy=1.57-row*.26
        for j in range(5):
            is_probe=(j==(1 if row==0 else 3))
            box(ax,x1+.12+j*.29,yy,.24,.19,
                fill=WHITE,edge=CHECK if is_probe else RULE,lw=1 if is_probe else .55)
            txt(ax,x1+.24+j*.29,yy+.095,'r' if is_probe else '·',size=7.5,
                ha='center',fontweight='bold' if is_probe else 'normal')
        txt(ax,x1+1.66,yy+.095,'…',size=9)
    txt(ax,x1+.10,1.10,'One scored row per independent batch',size=7.1)
    txt(ax,x1+.10,.90,'Try 8, 4, 2; CP lower bound ≥ .95',size=7.0,fontweight='bold',color=CHECK)
    # Complete partitions depict a different object from artificial probe batches.
    txt(ax,x2+.10,2.11,'b  Complete executions',size=8.7,fontweight='bold',color=PLAN)
    txt(ax,x2+.10,1.88,'Pass: random k; no pass: singletons',size=7.1)
    for yy,lab in [(1.52,'Chosen k'),(1.18,'Fixed8')]:
        txt(ax,x2+.10,yy+.12,lab,size=7.4,fontweight='bold')
        for j in range(3):
            box(ax,x2+.80+j*.40,yy,.32,.23,fill=LIGHT,edge=PLAN,lw=.6)
            slots=8 if lab=='Fixed8' else 4
            for k in range(1,slots):
                ax.plot([x2+.80+j*.40+k*.32/slots]*2,[yy,yy+.23],color=RULE,lw=.4)
    txt(ax,x2+.10,1.04,'Random8 / Similar8: unchecked baselines',size=6.8)
    txt(ax,x2+.10,.85,'COUNT is computed outside the LLM',size=7.2)
    # Dashed link explicitly means that transport is an empirical question.
    arrow(ax,2.23,1.71,2.47,1.71,color=CHECK,ls='--')
    txt(ax,2.35,1.91,'select\nk',size=6.5,ha='center',color=CHECK)
    arrow(ax,4.70,1.71,4.94,1.71,color=PLAN)
    txt(ax,x3+.10,2.11,'c  Separate readouts',size=8.7,fontweight='bold')
    entries=[('Native / strict','Boolean delivery; format validity'),
             ('Fidelity','Full S1 and S2 executions'),
             ('Task COUNT','Supplied labels $g_i$; COUNT $G$'),
             ('Full bill','CPU; calibration; calls; tokens/cache')]
    for j,(title,desc) in enumerate(entries):
        yy=1.90-j*.30
        txt(ax,x3+.10,yy,title,size=7.5,fontweight='bold')
        txt(ax,x3+.10,yy-.13,desc,size=7.0)
    # S1/S2, supplied labels/count and schematic scope are defined in the caption.
    render(fig,'contract_execution_overview')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    overview()
    paths=[OUT/('contract_execution_overview.'+ext) for ext in ('pdf','svg','png')]
    manifest={'status':'ORIGINAL_VECTOR_EXECUTION_OVERVIEW', 'empirical_values':False,
              'illustration_scope':'declared query and execution objects; identities are schematic',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'files':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (OUT/'FIGURE1_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
