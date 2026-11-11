#!/usr/bin/env python3
"""Pure paper figures/tables from supplied readout dictionaries; no private study IO."""
from pathlib import Path
import csv,hashlib,json,sys
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
M=Path(__file__).resolve().parent
D=M/'generated_data'
F=M/'figures'
T=M/'draft/results'
NAMES={'agnews_count_world':'AG World','agnews_count_sports':'AG Sports',
       'agnews_count_business':'AG Business','agnews_count_science_technology':'AG Sci/Tech',
       'movie_count_positive_top_critics':'Movie top','movie_count_positive_other_critics':'Movie other'}
ORDER=['movie_count_positive_top_critics','movie_count_positive_other_critics',
       'agnews_count_world','agnews_count_sports','agnews_count_business','agnews_count_science_technology']
MODELS={'qwen':'Qwen','glm':'GLM'}
POLICIES=['S1','S2','RANDOM8','SIMILAR8','CURRENT_CP']
COLORS={'RANDOM8':'#3B7C99','SIMILAR8':'#9C6B30','CURRENT_CP':'#6B4E86','S1':'#263340'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                     'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65,
                     'savefig.facecolor':'white','legend.frameon':False})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def outcsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def ratio(a,b):return a/b if a is not None and b not in [None,0] else None
def fmt(v,d=2):return '--' if v is None else f'{v:.{d}f}'
def pc(v,d=2):return fmt(None if v is None else 100*v,d)
def integer(v):return '--' if v is None else f'{int(v):,}'
def lookup(rows,**keys):
    found=[r for r in rows if all(r.get(k)==v for k,v in keys.items())]
    if len(found)!=1:raise RuntimeError(('expected exactly one row',keys,len(found)))
    return found[0]
def pngpdf(fig,stem):
    fig.canvas.draw()
    for ext in ['pdf','svg','png']:
        fig.savefig(F/(stem+'.'+ext),dpi=240,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
def tables(rows,brows,budget):
    # Show the native S1 predicate-quality baseline for every primary workload,
    # so high fidelity cannot hide a poor named reference.
    lines=[r'\begin{table*}[t]',r'\centering\small',
           r'\caption{Native singleton reference S1 against supplied source labels in all primary configurations ($N=2,048$, cache ON). $G$: source COUNT; $C$: delivered native COUNT; D: default-True parses; R: source row-error percentage. Native delivery and reference fidelity do not imply task correctness.}',
           r'\label{tab:reference_quality}',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
           r'Workload & $G$ & Qwen $C$ & Qwen D/$N$ & Qwen R (\%) & GLM $C$ & GLM D/$N$ & GLM R (\%) \\',r'\midrule']
    for w in ORDER:
        a=lookup(rows,model='qwen',workload_id=w,n=2048,prefix_cache=True,policy='S1',contract='native')
        b=lookup(rows,model='glm',workload_id=w,n=2048,prefix_cache=True,policy='S1',contract='native')
        vals=[NAMES[w],integer(a['source_gold_count']),integer(a['predicted_count']),str(a['native_fallback_count'])+'/2048',pc(a['row_error_full']),integer(b['predicted_count']),str(b['native_fallback_count'])+'/2048',pc(b['row_error_full'])]
        lines.append(' & '.join(vals)+r' \\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (T/'reference_quality.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table*}[t]',r'\centering\small',
           r'\caption{Current80-S in all 12 primary configurations ($N=2,048$, cache ON). $V_s$: strict-valid plan rows; $C/G$: native/source COUNT; $e_N,e_G$: absolute COUNT error divided by $N,G$; $d_1,d_2$: native row disagreement with S1/S2. Complete calibration-plus-execution costs are divided by S1. A missing strict COUNT does not mean native failure.}',
           r'\label{tab:primary_results}',r'\begin{tabular}{lrrrrrrrr}',r'\toprule',
           r'Workload & $k$ & \shortstack{Native\\delivery} & $V_s/N$ & $C/G$ & $e_N\,|\,e_G$ (\%) & $d_1\,|\,d_2$ (\%) & Calls/S1 & \shortstack{Raw tokens\\/S1} \\',r'\midrule']
    for model in ['qwen','glm']:
        lines.append(r'\multicolumn{9}{l}{\textit{'+MODELS[model]+r'}} \\')
        for w in ORDER:
            r=lookup(rows,model=model,workload_id=w,n=2048,prefix_cache=True,policy='CURRENT_CP',contract='native')
            s=lookup(rows,model=model,workload_id=w,n=2048,prefix_cache=True,policy='CURRENT_CP',contract='strict')
            vals=[NAMES[w],str(r['selected_k']),'yes' if r['semantic_complete'] else 'no',str(s['valid_labels'])+'/2048',integer(r['predicted_count'])+'/'+integer(r['source_gold_count']),
                  pc(r['count_error_over_N'])+r' $\mid$ '+pc(r['count_error_over_G']),pc(r['loss_to_S1'])+r' $\mid$ '+pc(r['loss_to_S2']),
                  fmt(r['requests_ratio_vs_S1'],3),fmt(r['total_tokens_ratio_vs_S1'])]
            lines.append(' & '.join(vals)+r' \\')
        if model=='qwen':lines.append(r'\midrule')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (T/'primary_results.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table*}[t]',r'\centering\small',
           r'\caption{Every paired Current80-N$\to$Current320-N follow-up ($N=7,600$, cache ON). $R$: complete requests; $d_{\max}$: maximum native row disagreement with S1/S2. Raw-token and uncached-input ratios compare complete 320-probe with 80-probe policies. Repeats are descriptive; each budget has its own nominal scope.}',
           r'\label{tab:budget_results}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r'Model / workload & Rep. & $k$ & \shortstack{Calibration\\calls} & Full $R$ & \shortstack{Raw tokens\\$320/80$} & \shortstack{Uncached input\\$320/80$} & $d_{\max}$ (\%) \\',r'\midrule']
    for model in ['qwen','glm']:
        for w in ['agnews_count_science_technology','movie_count_positive_other_critics']:
            for rep in [1,2]:
                a=lookup(brows,model=model,workload_id=w,budget=80,repeat=rep,contract='native')
                b=lookup(brows,model=model,workload_id=w,budget=320,repeat=rep,contract='native')
                cal=[lookup(budget['costs'],model=model,job_id=x['job_id'],phase='calibration')['requests'] for x in [a,b]]
                vals=[MODELS[model]+' / '+NAMES[w],str(rep),str(a['selected_k'])+r'$\to$'+str(b['selected_k']),
                      str(cal[0])+r'$\to$'+str(cal[1]),integer(a['requests'])+r'$\to$'+integer(b['requests']),
                      fmt(ratio(b['total_tokens'],a['total_tokens'])),fmt(ratio(b['uncached_tokens'],a['uncached_tokens'])),
                      pc(a['max_loss_to_S1_S2'])+r'$\to$'+pc(b['max_loss_to_S1_S2'])]
                lines.append(' & '.join(vals)+r' \\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (T/'budget_results.tex').write_text('\n'.join(lines)+'\n')

def primary_policy_summary(rows):
    aggregate=[]
    for p in ['RANDOM8','SIMILAR8','CURRENT_CP']:
        flag='descriptive_flag' if p=='CURRENT_CP' else 'descriptive_fixed_plan_benefit'
        item={'policy':p,'certificate_policy':p=='CURRENT_CP','declared_primary_configurations':12}
        for contract in ['native','strict']:
            q=[r for r in rows if r['primary_matrix'] and r['policy']==p and r['contract']==contract]
            assert len(q)==12
            item[contract+'_delivery']=sum(r['semantic_complete'] for r in q)
            item[contract+'_descriptive_benefit']=sum(r[flag] is True for r in q)
            item[contract+'_benefit_unavailable']=sum(r[flag] is None for r in q)
            for field in ['max_loss_to_S1_S2','requests_ratio_vs_S1','total_tokens_ratio_vs_S1']:
                vals=[r[field] for r in q if r[field] is not None]
                item[contract+'_'+field+'_min']=min(vals) if vals else None
                item[contract+'_'+field+'_max']=max(vals) if vals else None
        aggregate.append(item)
    outcsv(D/'primary_policy_benefit_summary.csv',aggregate)
    lines=[r'\begin{table*}[t]',r'\centering\small',
           r"\caption{Three policies over the same 12 primary configurations. Benefit requires native delivery, $k>1$, both reference losses $\le.05$, fewer requests and fewer raw input-plus-output tokens than S1. Ranges use each case's S1. Strict benefit retains the 12-case denominator; it is unavailable if a full strict plan or S1/S2 reference is missing. For the fixed plans, all six unavailable cases arise from incomplete strict references. Fixed plans have no certificate; source correctness is separate.}",
           r'\label{tab:policy_summary}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Plan & Native delivery & Native $d_{\max}$ (\%) & Calls/S1 & Raw tokens/S1 & Native benefit & \shortstack{Strict benefit;\\unavailable of 12} \\',r'\midrule']
    for r in aggregate:
        vals=[{'RANDOM8':'Random8','SIMILAR8':'Similar8','CURRENT_CP':'Current80-S'}[r['policy']],str(r['native_delivery'])+'/12',
              pc(r['native_max_loss_to_S1_S2_min'])+'--'+pc(r['native_max_loss_to_S1_S2_max']),
              fmt(r['native_requests_ratio_vs_S1_min'],3)+'--'+fmt(r['native_requests_ratio_vs_S1_max'],3),
              fmt(r['native_total_tokens_ratio_vs_S1_min'],3)+'--'+fmt(r['native_total_tokens_ratio_vs_S1_max'],3),
              str(r['native_descriptive_benefit'])+'/12',str(r['strict_descriptive_benefit'])+'/12; '+str(r['strict_benefit_unavailable'])]
        lines.append(' & '.join(vals)+r' \\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (T/'primary_policy_summary.tex').write_text('\n'.join(lines)+'\n')
    return aggregate

def main_plot(rows):
    from figure_style import draw_primary
    draw_primary(sys.modules[__name__], rows)

def budget_plot(brows,budget,rows):
    from figure_style import draw_budget
    draw_budget(sys.modules[__name__], brows, budget, rows)

def cache_plot(rows):
    from figure_style import draw_cache
    draw_cache(sys.modules[__name__], rows)
    grouped=[]
    for model in ['qwen','glm']:
        for cache in [True,False]:
            for policy in ['RANDOM8','SIMILAR8','CURRENT_CP']:
                q=[r for r in rows if r['model']==model and r['prefix_cache']==cache and r['policy']==policy and r['contract']=='native']
                item={'model':model,'prefix_cache':cache,'policy':policy,'configurations':len(q)}
                for field in ['input_tokens_ratio_vs_S1','output_tokens_ratio_vs_S1','uncached_tokens_ratio_vs_S1','total_tokens_ratio_vs_S1']:
                    vals=[r[field] for r in q if r[field] is not None]
                    item[field+'_min']=min(vals) if vals else None;item[field+'_max']=max(vals) if vals else None
                    item[field+'_available']=len(vals)
                item['raw_input_lower_uncached_input_higher']=sum(r['input_tokens_ratio_vs_S1'] is not None and r['uncached_tokens_ratio_vs_S1'] is not None and r['input_tokens_ratio_vs_S1']<1<r['uncached_tokens_ratio_vs_S1'] for r in q)
                grouped.append(item)
    pairs=[]
    for model in ['qwen','glm']:
        for w in [ORDER[0],ORDER[1],ORDER[-1]]:
            for n in [2048,7600]:
                for policy in ['RANDOM8','SIMILAR8','CURRENT_CP']:
                    off=lookup(rows,model=model,workload_id=w,n=n,prefix_cache=False,policy=policy,contract='native')
                    on=lookup(rows,model=model,workload_id=w,n=n,prefix_cache=True,policy=policy,contract='native')
                    item={'model':model,'workload_id':w,'n':n,'policy':policy,
                          'raw_input_count_identical_ON_OFF':off['input_tokens']==on['input_tokens']}
                    for field in ['input_tokens','output_tokens','uncached_tokens','input_tokens_ratio_vs_S1','output_tokens_ratio_vs_S1','uncached_tokens_ratio_vs_S1']:
                        item[field+'_OFF']=off[field];item[field+'_ON']=on[field]
                    pairs.append(item)
    outcsv(D/'matched_cache_ON_OFF.csv',pairs)
    outcsv(D/'all_cache_cost_objects.csv',grouped)
    lines=[r'\begin{table*}[t]',r'\centering\small',
           r'\caption{Fixed-plan cost ratios to the corresponding separately executed S1. Cells give complete-matrix ranges; ON has 12 and OFF six configurations per model/plan. The last column counts raw-input ratio $P/P_{S1}<1$ but uncached-input ratio $U/U_{S1}>1$. Input, output and uncached input are distinct; uncached input is a prefill-work proxy.}',
           r'\label{tab:cache_costs}',r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Model/cache & Plan & Cases & Raw input/S1 & Output/S1 & Uncached input/S1 & \shortstack{$P/P_{S1}<1$,\\$U/U_{S1}>1$} \\',r'\midrule']
    for model in ['qwen','glm']:
        for cache in [True,False]:
            for policy in ['RANDOM8','SIMILAR8']:
                r=lookup(grouped,model=model,prefix_cache=cache,policy=policy)
                vals=[MODELS[model]+'/'+('ON' if cache else 'OFF'),'Random8' if policy=='RANDOM8' else 'Similar8',str(r['configurations'])]
                for field in ['input_tokens_ratio_vs_S1','output_tokens_ratio_vs_S1','uncached_tokens_ratio_vs_S1']:
                    vals.append(fmt(r[field+'_min'])+'--'+fmt(r[field+'_max']))
                vals.append(str(r['raw_input_lower_uncached_input_higher'])+'/'+str(r['configurations']))
                lines.append(' & '.join(vals)+r' \\')
        if model=='qwen':lines.append(r'\midrule')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (T/'cache_cost_ranges.tex').write_text('\n'.join(lines)+'\n')
    outcsv(D/'all_fixed_plan_input_output_cache.csv',[r for r in rows if r['policy'] in ['RANDOM8','SIMILAR8'] and r['contract']=='native'])


