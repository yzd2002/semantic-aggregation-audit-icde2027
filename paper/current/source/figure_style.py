"""Shared publication styling and deterministic vector diagrams, from frozen data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

INK = '#24354B'
MUTED = '#657387'
RULE = '#D8DFE7'
BLUE = '#437DB0'
ORANGE = '#BE733E'
PURPLE = '#8065A3'
TEAL = '#247F77'
CHECK_FILL = '#BDA9D2'
PALE_TEAL = '#F0F7F5'
PALE_BLUE = '#F2F6FA'
PALE_PURPLE = '#F4F0F8'
COLORS = {'RANDOM8': BLUE, 'SIMILAR8': ORANGE, 'CURRENT_CP': PURPLE, 'S1': INK}

def setup():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':8, 'text.color':INK,
        'axes.labelcolor':INK, 'axes.edgecolor':RULE, 'xtick.color':MUTED,
        'ytick.color':INK, 'axes.linewidth':.6, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'none', 'legend.frameon':False,
        'savefig.facecolor':'white', 'axes.unicode_minus':True,
    })

def export(fig, folder, stem):
    """Fixed physical dimensions; enforce that all visible text fits the canvas."""
    fig.canvas.draw()
    ren=fig.canvas.get_renderer()
    outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():continue
        b=t.get_window_extent(ren)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1):
            # Shared-axis tick objects can exist while their parent artist is hidden.
            if t.axes is not None and not t.axes.get_visible():continue
            outside.append(t.get_text())
    if outside:raise RuntimeError((stem,'text outside figure',outside))
    Path(folder).mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(Path(folder)/(stem+'.'+ext),dpi=300,
                    metadata={'Creator':'Frozen-data vector figure renderer'} if ext=='pdf' else None)
    plt.close(fig)

def axes_style(ax, grid='x'):
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color(RULE)
    ax.tick_params(axis='both',length=0,pad=4,labelsize=7.7)
    ax.set_axisbelow(True)
    ax.grid(axis=grid,color=RULE,linewidth=.45,alpha=.7)

def draw_primary(r, rows):
    setup()
    cells=[(m,w) for m in ['qwen','glm'] for w in r.ORDER]
    # An extra row of space separates model groups without repeating model names.
    yy=np.array([12,11,10,9,8,7,5,4,3,2,1,0],dtype=float)
    fig,axes=plt.subplots(1,4,figsize=(7.16,4.12),sharey=True,
                         gridspec_kw={'width_ratios':[1.15,1.0,1.0,1.2],'wspace':.22})
    fig.subplots_adjust(left=.177,right=.978,top=.84,bottom=.19)
    for ax in axes:
        axes_style(ax)
        for y in yy[::2]:ax.axhspan(y-.48,y+.48,color='#F7F9FB',zorder=0)
        ax.axhline(6,color=RULE,lw=.8)
        ax.set_ylim(-.7,13.25)
    axes[0].axvspan(0,5,color=PALE_TEAL,zorder=.1)
    axes[3].axvspan(0,1,color=PALE_BLUE,zorder=.1)
    for j,p in enumerate(['RANDOM8','SIMILAR8','CURRENT_CP']):
        for idx,(model,w) in enumerate(cells):
            row=r.lookup(rows,model=model,workload_id=w,n=2048,prefix_cache=True,policy=p,contract='native')
            loss=100*row['max_loss_to_S1_S2'] if row['max_loss_to_S1_S2'] is not None else None
            values=[loss,loss,100*row['count_error_over_N'] if row['count_error_over_N'] is not None else None,row['total_tokens_ratio_vs_S1']]
            for k,(ax,value) in enumerate(zip(axes,values)):
                if value is None or (k==0 and value>5.25):continue
                fallback=p=='CURRENT_CP' and row['selected_k']==1
                ax.scatter(value,yy[idx]+(j-1)*.20,s=18,
                           marker=['o','s','D'][j],facecolors='white' if fallback else COLORS[p],
                           edgecolors=COLORS[p],linewidths=.85 if fallback else .45,zorder=4)
    for idx,(model,w) in enumerate(cells):
        base=r.lookup(rows,model=model,workload_id=w,n=2048,prefix_cache=True,policy='S1',contract='native')
        axes[2].scatter(100*base['count_error_over_N'],yy[idx]-.36,s=16,marker='x',color=INK,linewidths=.75,zorder=5)
    axes[0].set_yticks(yy,[r.NAMES[w] for _,w in cells],fontsize=8)
    axes[0].text(-.025,12.77,'Qwen',ha='right',va='center',fontsize=7.5,weight='bold',color=MUTED,transform=axes[0].get_yaxis_transform())
    axes[0].text(-.025,5.77,'GLM',ha='right',va='center',fontsize=7.5,weight='bold',color=MUTED,transform=axes[0].get_yaxis_transform())
    for ax in axes[1:]:ax.tick_params(labelleft=False)
    axes[0].set_xlim(-.18,5.28);axes[0].set_xticks([0,2.5,5])
    axes[1].set_xlim(-2,63);axes[1].set_xticks([0,30,60])
    axes[2].set_xlim(-2,78);axes[2].set_xticks([0,25,50,75])
    axes[3].set_xlim(0,1.36);axes[3].set_xticks([0,.5,1])
    for ax in axes[:2]:ax.axvline(5,color=TEAL,ls=(0,(3,2)),lw=.8,zorder=2)
    axes[3].axvline(1,color=MUTED,ls=(0,(3,2)),lw=.8,zorder=2)
    titles=['Reference loss','Reference loss','Source COUNT error','Complete policy cost']
    subs=['0–5% detail','full range','error / input rows','raw tokens / S1']
    for ax,title,sub in zip(axes,titles,subs):
        ax.set_title(title,fontsize=8.4,weight='bold',pad=25)
        ax.text(.5,1.035,sub,ha='center',va='bottom',transform=ax.transAxes,fontsize=7.5,color=MUTED)
    for ax in axes[:3]:ax.set_xlabel('Percent (%)',fontsize=7.5,labelpad=4)
    axes[3].set_xlabel('Ratio to single-row S1',fontsize=7.5,labelpad=4)
    handles=[Line2D([],[],ls='',marker=mk,markersize=4.3,color=COLORS[p],label=label)
             for p,mk,label in [('RANDOM8','o','Random8'),('SIMILAR8','s','Similar8')]]
    handles += [Line2D([],[],ls='',marker='D',markersize=4.3,color=PURPLE,markerfacecolor='white',label='Current80-S: fallback'),
                Line2D([],[],ls='',marker='D',markersize=4.3,color=PURPLE,label='Current80-S: accepted batch'),
                Line2D([],[],ls='',marker='x',markersize=4.3,color=INK,label='S1: source COUNT only')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=3,
               fontsize=7.6,columnspacing=1.3,handletextpad=.5,labelspacing=.65)
    export(fig,r.F,'primary_execution_readouts')

def draw_budget(r,brows,budget,rows):
    setup()
    fig,axes=plt.subplots(2,2,figsize=(7.16,3.10),sharex=True,sharey=True)
    fig.subplots_adjust(left=.083,right=.985,bottom=.19,top=.84,wspace=.20,hspace=.42)
    xs=[0,.72,1.90,2.62]
    data=[]
    for i,model in enumerate(['qwen','glm']):
        for j,w in enumerate(['agnews_count_science_technology','movie_count_positive_other_critics']):
            ax=axes[i,j];axes_style(ax,'y')
            baseline=r.lookup(rows,model=model,workload_id=w,n=7600,prefix_cache=True,policy='S1',contract='native')['total_tokens']
            for x,(rep,n) in zip(xs,[(1,80),(1,320),(2,80),(2,320)]):
                row=r.lookup(brows,model=model,workload_id=w,repeat=rep,budget=n,contract='native')
                c=r.lookup(budget['costs'],model=model,job_id=row['job_id'],phase='calibration')['total_tokens']/baseline
                e=r.lookup(budget['costs'],model=model,job_id=row['job_id'],phase='execution')['total_tokens']/baseline
                assert row['selected_k']==1
                ax.bar(x,e,width=.48,color=BLUE,zorder=3)
                ax.bar(x,c,bottom=e,width=.48,color=CHECK_FILL,edgecolor=PURPLE,lw=.35,zorder=3)
                ax.text(x,e+c+.027,f'{e+c:.2f}',fontsize=7.1,ha='center',va='bottom',color=INK)
                data.append({'model':model,'workload':w,'repeat':rep,'budget':n,'execution':e,'checking':c,'total':e+c})
            ax.axhline(1,color=MUTED,ls=(0,(3,2)),lw=.75,zorder=4)
            ax.set_title(r.MODELS[model]+' · '+r.NAMES[w],fontsize=8.4,weight='bold',pad=7)
            ax.set_xticks(xs,['80','320','80','320']);ax.tick_params(labelbottom=(i==1))
            ax.set_yticks([0,.5,1,1.5]);ax.set_ylim(0,1.50);ax.set_xlim(-.44,3.06)
            for middle,label in ([(.36,'Repeat 1'),(2.26,'Repeat 2')] if i==1 else []):
                ax.text(middle,-.26,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=7.2,color=MUTED)
    fig.text(.012,.51,'Complete raw tokens / S1',rotation=90,ha='left',va='center',fontsize=8)
    fig.text(.083,.955,'Higher checking budgets, unchanged singleton fallback',fontsize=9,weight='bold',ha='left')
    handles=[Rectangle((0,0),1,1,color=BLUE,label='Execution'),Rectangle((0,0),1,1,color=CHECK_FILL,label='Checking'),Line2D([],[],color=MUTED,ls='--',lw=.8,label='Single-row baseline')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.004),ncol=3,fontsize=7.7,columnspacing=2.2)
    (Path(r.F)/'budget_plot_values.json').write_text(json.dumps(data,indent=2)+'\n')
    export(fig,r.F,'budget_full_bill')

def draw_cache(r,rows):
    setup()
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.72),sharey=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.27,top=.84,wspace=.18)
    for ax,model in zip(axes,['qwen','glm']):
        axes_style(ax,'both')
        ax.add_patch(Rectangle((.50,1),.50,.34,fc='#FAF4EF',ec='none',zorder=0))
        ax.add_patch(Rectangle((.50,.50),.50,.50,fc=PALE_BLUE,ec='none',zorder=0))
        for w in ['movie_count_positive_top_critics','agnews_count_science_technology']:
            for policy in ['RANDOM8','SIMILAR8']:
                off=r.lookup(rows,model=model,workload_id=w,n=7600,prefix_cache=False,policy=policy,contract='native')
                on=r.lookup(rows,model=model,workload_id=w,n=7600,prefix_cache=True,policy=policy,contract='native')
                x0,y0=off['input_tokens_ratio_vs_S1'],off['uncached_tokens_ratio_vs_S1']
                x1,y1=on['input_tokens_ratio_vs_S1'],on['uncached_tokens_ratio_vs_S1']
                ax.plot([x0,x1],[y0,y1],color=MUTED,alpha=.40,lw=.8,zorder=2)
                for x,y,cache in [(x0,y0,False),(x1,y1,True)]:
                    ax.scatter(x,y,s=52 if policy=='RANDOM8' else 14,marker='o' if policy=='RANDOM8' else 's',
                               facecolors=COLORS[policy] if cache else 'white',edgecolors=COLORS[policy],lw=.9,
                               zorder=4 if policy=='RANDOM8' else 5)
            a=r.lookup(rows,model=model,workload_id=w,n=7600,prefix_cache=True,policy='RANDOM8',contract='native')
            ax.annotate(r.NAMES[w],(a['input_tokens_ratio_vs_S1'],a['uncached_tokens_ratio_vs_S1']),
                        xytext=(6,7),textcoords='offset points',fontsize=7.7,color=INK)
        ax.axhline(1,color=MUTED,ls=(0,(3,2)),lw=.8,zorder=3)
        ax.axvline(1,color=MUTED,ls=(0,(3,2)),lw=.8,zorder=3)
        ax.set_xlim(.50,1.02);ax.set_ylim(.50,1.34)
        ax.set_xticks([.5,.6,.7,.8,.9,1]);ax.set_yticks([.5,.7,.9,1,1.1,1.3])
        ax.set_title(r.MODELS[model],fontsize=9,weight='bold',pad=9)
        ax.set_xlabel('Raw input tokens / S1',fontsize=8,labelpad=6)
    axes[0].set_ylabel('Uncached input tokens / S1',fontsize=8,labelpad=6)
    axes[0].text(.505,1.015,'More uncached input',fontsize=7.2,color=ORANGE,va='bottom')
    axes[1].text(.985,.52,'Both counters lower',fontsize=7.2,color=BLUE,va='bottom',ha='right')
    handles=[Line2D([],[],ls='',marker='o',markersize=6,color=BLUE,label='Random8'),Line2D([],[],ls='',marker='s',markersize=4,color=ORANGE,label='Similar8'),
             Line2D([],[],ls='',marker='o',markersize=4.5,color=MUTED,label='Cache ON'),Line2D([],[],ls='',marker='o',markersize=4.5,color=MUTED,markerfacecolor='white',label='Cache OFF')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.025),ncol=4,fontsize=7.7,columnspacing=1.6)
    export(fig,r.F,'cache_input_cost_objects')

def draw_overview(folder, values):
    setup()
    fig=plt.figure(figsize=(7.16,3.00));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7.16),ylim=(0,3));ax.axis('off')
    def text(x,y,s,fs=8,**kw):
        return ax.text(x,y,s,fontsize=fs,color=kw.pop('color',INK),va='center',**kw)
    def line(x1,y1,x2,y2,color=RULE,lw=.65,**kw):ax.plot([x1,x2],[y1,y2],color=color,lw=lw,**kw)
    def box(x,y,w,h,fc='white',ec=RULE,rounding=.025,lw=.6):
        p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',fc=fc,ec=ec,lw=lw);ax.add_patch(p);return p
    def arrow(x1,y1,x2,y2,c=MUTED):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=6,lw=.7,color=c))
    def rows(x,y,w,h,n,highlight=None):
        box(x,y,w,h,ec='#AFBBC8')
        gap=.025;rh=(h-.08-(n-1)*gap)/n
        for i in range(n):
            yy=y+h-.04-(i+1)*rh-i*gap
            box(x+.045,yy,w-.09,rh,fc=CHECK_FILL if i==highlight else '#DCE3EA',ec='none',rounding=.006)
    text(.045,2.83,'Fewer calls, but more tokens after checking',fs=11.1,weight='bold')
    text(.045,2.59,'Sports COUNT  ·  2,048 news records  ·  GLM  ·  prefix cache ON',fs=8.2,color=MUTED)
    line(.045,2.43,7.115,2.43)
    for x in [2.25,4.42]:line(x,.15,x,2.27)
    text(.06,2.23,'Check a candidate',fs=9.2,weight='bold')
    text(2.40,2.23,'Execute the full query',fs=9.2,weight='bold')
    text(4.58,2.23,'Account for both stages',fs=9.2,weight='bold')
    text(.12,1.99,'One row',fs=7.7,color=MUTED)
    rows(.13,1.72,.53,.15,1,0)
    text(.12,1.52,'Eight rows',fs=7.7,color=MUTED)
    rows(.13,.96,.53,.47,8,3)
    for y in [1.79,1.19]:
        arrow(.68,y,.94,y)
        box(.97,y-.125,.43,.25,fc=PALE_BLUE,ec='#B4C7D7');text(1.185,y,'LLM',fs=7.8,ha='center')
        arrow(1.42,y,1.64,y)
        box(1.68,y-.10,.24,.20,fc=PALE_TEAL,ec='#8DB3AC');text(1.80,y,'T',fs=8.4,color=TEAL,weight='bold',ha='center')
    line(1.96,1.19,2.03,1.19,color=MUTED);line(2.03,1.19,2.03,1.79,color=MUTED);line(1.96,1.79,2.03,1.79,color=MUTED)
    text(2.08,1.49,'=',fs=9,ha='center')
    text(1.12,.80,'Compare one focal label per test',fs=7.6,ha='center',color=MUTED)
    box(.12,.48,2.00,.22,fc=PALE_PURPLE,ec='none');text(1.12,.59,'80 / 80 match → select size 8',fs=8,ha='center',color=PURPLE,weight='bold')
    line(.12,.38,2.12,.38)
    text(1.12,.245,'160 calls · 93,120 tokens',fs=8.2,weight='bold',ha='center',color=PURPLE)
    rows(2.42,1.51,.48,.51,8,None)
    text(3.12,1.82,'× 256',fs=9.5,ha='center',weight='bold');text(3.12,1.65,'batches',fs=7.4,ha='center',color=MUTED)
    arrow(3.40,1.76,3.60,1.76)
    box(3.65,1.51,.58,.50,fc=PALE_BLUE,ec='#B4C7D7');text(3.94,1.84,'COUNT',fs=7.7,ha='center');text(3.94,1.64,'519',fs=13.5,ha='center',weight='bold')
    text(3.32,1.32,'All 2,048 rows labeled',fs=8.1,ha='center')
    text(3.32,1.02,'99.17%',fs=20,ha='center',weight='bold',color=TEAL)
    text(3.32,.76,'agreement with single-row references',fs=7.5,ha='center')
    text(3.32,.55,'17 / 2,048 labels differ',fs=7.6,ha='center',color=MUTED)
    line(2.42,.38,4.23,.38)
    text(3.32,.245,'256 calls · 255,425 tokens',fs=8.2,weight='bold',ha='center',color=BLUE)
    # The true data, not the generative preview's pixel geometry, controls these bars.
    bx,bw,maximum=4.61,2.43,400000
    text(bx,1.99,'Raw tokens = input + output',fs=7.8,color=MUTED)
    base,check,execution=values['base_tokens'],values['check_tokens'],values['deploy_tokens']
    total=check+execution
    text(bx,1.76,'Single-row',fs=7.7);text(7.04,1.76,f'{base:,}',fs=7.8,ha='right')
    ax.add_patch(Rectangle((bx,1.50),bw*base/maximum,.16,fc='#B4BFCC',ec='none'))
    text(bx,1.31,'Checked policy',fs=7.7);text(7.04,1.31,f'{total:,}',fs=7.8,ha='right',weight='bold')
    ax.add_patch(Rectangle((bx,1.05),bw*check/maximum,.16,fc=CHECK_FILL,ec='none'))
    ax.add_patch(Rectangle((bx+bw*check/maximum,1.05),bw*execution/maximum,.16,fc=BLUE,ec='none'))
    line(bx+bw*base/maximum,1.00,bx+bw*base/maximum,1.68,color=MUTED,lw=.6,ls=(0,(2,2)))
    text(bx,.91,'0',fs=7,color=MUTED);text(bx+bw,.91,'400k',fs=7,ha='right',color=MUTED)
    for x,color,label in [(4.64,CHECK_FILL,'Checking'),(5.76,BLUE,'Execution')]:
        ax.add_patch(Rectangle((x,.718),.09,.075,fc=color,ec='none'));text(x+.15,.755,label,fs=7.4)
    text(5.18,.485,'−79.69%',fs=13.4,ha='center',weight='bold',color=BLUE)
    text(6.45,.485,'+3.01%',fs=13.4,ha='center',weight='bold',color=ORANGE)
    text(5.18,.28,'calls (416 vs 2,048)',fs=7.4,ha='center')
    text(6.45,.28,'total raw tokens',fs=7.4,ha='center')
    export(fig,folder,'contract_execution_overview')
