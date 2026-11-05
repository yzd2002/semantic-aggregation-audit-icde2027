#!/usr/bin/env python3
"""Render completed controlled-audit observations; no inference or selection."""
from pathlib import Path
import argparse,hashlib,json,shutil

def main(bundle,out):
 out.mkdir(exist_ok=True)
 models={};summaries={}
 for model in ('qwen','smol'):
  p=bundle/f'run_{model}/analysis.json';a=json.loads(p.read_text())
  summary=json.loads((p.parent/'summary.json').read_text())
  assert summary['status']=='complete' and summary['jobs']==11894
  assert hashlib.sha256((p.parent/'responses.jsonl').read_bytes()).hexdigest()==a['responses_sha256']==summary['response_sha256']
  assert all(d['primary_common_pools']==d['batch_singleton_common_pools']==d['attempted_pools'] for d in a['datasets'].values())
  models[model]=a;summaries[model]=summary
  shutil.copy2(p,out/f'controlled_{model}_analysis.json')
 def value(model,dataset,key,metric):return models[model]['datasets'][dataset]['co_primary'][key][metric]['excess']
 def pp(x):return f"{100*x:.2f}"
 def qi(x):return '$['+','.join(pp(v) for v in x['unadjusted_query_interval_95'])+']$'
 qe=[value('qwen','esci',k,'absolute_ratio_change') for k in ('A','D')]
 se=[value('smol','esci',k,'absolute_ratio_change') for k in ('A','D')]
 qw=[value('qwen','wands',k,'absolute_ratio_change') for k in ('A','D')]
 sw=[value('smol','wands',k,'absolute_ratio_change') for k in ('A','D')]
 abstract=("On 128 previously unused ESCI pools, Qwen's row-order excess over replay is "
           f"{100*qe[0]['mean']:.1f}--{100*qe[1]['mean']:.1f} percentage points in absolute ratio change "
           "under two output-key orders, despite complete ID-aligned outputs.")
 (out/'CONTROLLED_ABSTRACT.tex').write_text(abstract+'\n')
 stats=[]
 for dataset in ('esci','wands'):
  for model in ('qwen','smol'):
   d=models[model]['datasets'][dataset]
   means=[sum(d['quality_all8'][c+str(k)]['ratio_mae'] for k in (0,1))/2 for c in ('AA','RA','AD','RD')]
   single=sum(d['singleton_quality_all10'][str(k)]['ratio_mae'] for k in (0,1))/2
   label=dataset.upper()+' / '+('Qwen' if model=='qwen' else 'SmolLM2')
   stats.append({'dataset':dataset,'model':model,'batch_condition_mean_min':min(means),'batch_condition_mean_max':max(means),
     'singleton_mean_mae':single,'always_y_mae':d['constants_all8']['Y']['ratio_mae'],'always_n_mae':d['constants_all8']['N']['ratio_mae'],
     'all8_and_all10_pools':d['primary_common_pools']})
 table_rows=[]
 for row in stats:
  label=row['dataset'].upper()+' / '+('Qwen' if row['model']=='qwen' else 'SmolLM2')
  table_rows.append(f"{label} & {row['batch_condition_mean_min']:.3f}--{row['batch_condition_mean_max']:.3f} & {row['singleton_mean_mae']:.3f} & {row['always_y_mae']:.3f} & {row['always_n_mae']:.3f} "+chr(92)*2)
 interaction=[]
 for model,a in models.items():
  for dataset,d in a['datasets'].items():
   for metric,x in d['effect_modification_D_minus_A'].items():
    lo,hi=x['unadjusted_query_interval_95']
    if not lo<=0<=hi:interaction.append((model,dataset,metric,x))
 if not interaction:
  interaction_text='All eight paired key-order effect-modification intervals span zero; this does not establish independence or equivalence.'
 else:
  interaction_text='Paired key-order effect-modification contrasts are retained in the artifact; intervals spanning zero do not establish independence or equivalence.'
 qints=[x for d in models['qwen']['datasets'].values() for x in d['effect_modification_D_minus_A'].values()]
 sints=[x for d in models['smol']['datasets'].values() for x in d['effect_modification_D_minus_A'].values()]
 if all(x['unadjusted_query_interval_95'][0]<=0<=x['unadjusted_query_interval_95'][1] for x in qints) and all(x['unadjusted_query_interval_95'][1]<0 for x in sints):
  ei=models['smol']['datasets']['esci']['effect_modification_D_minus_A']['absolute_ratio_change']
  wi=models['smol']['datasets']['wands']['effect_modification_D_minus_A']['absolute_ratio_change']
  interaction_text=(f"Qwen's four paired $D-A$ interaction intervals span zero. SmolLM2's ratio interactions are {pp(ei['mean'])} pp ({qi(ei)}) on ESCI and {pp(wi['mean'])} pp ({qi(wi)}) on WANDS; its candidate interactions are also negative. These unadjusted contrasts do not establish an internal mechanism.")
 valid=sum(x['valid_jobs'] for x in summaries.values());invalid=sum(x['invalid_jobs'] for x in summaries.values())
 assert invalid==0,'Report any incomplete/invalid observations explicitly instead of using this full-completion prose.'
 qs=models['qwen']['datasets']['esci']['singleton_quality_all10']
 qf=sum(qs[str(k)]['f1'] for k in (0,1))/2
 qb=sum(qs[str(k)]['balanced_accuracy'] for k in (0,1))/2
 qm=sum(qs[str(k)]['ratio_mae'] for k in (0,1))/2
 qy=models['qwen']['datasets']['esci']['constants_all8']['Y']['ratio_mae']
 result=fr'''All {valid:,} requests validate; every selected pool is complete under all eight batch and both singleton conditions. Figure~\ref{{fig:controlled_order}} reports both pre-specified distances. On the 128-pool ESCI controlled audit, Qwen's ratio-change excess is {pp(qe[0]['mean'])}/{pp(qe[1]['mean'])} percentage points (pp) for ascending/descending keys, with intervals {qi(qe[0])}/{qi(qe[1])}; SmolLM2's corresponding excesses are {pp(se[0]['mean'])}/{pp(se[1]['mean'])} pp. Both models also have positive excess on exposed WANDS. {interaction_text}

The secondary key-order analysis holds row order fixed; all 16 candidate/ratio excess intervals lie above zero across both models, datasets, and row orders (unadjusted). Table~\ref{{tab:controlled_quality}} separates stability from reference accuracy. Qwen's ESCI singleton is nonconstant (mean F1 {qf:.3f}; balanced accuracy {qb:.3f}), yet its mean MAE {qm:.3f} exceeds the {qy:.3f} all-Y baseline. SmolLM2's near-all-N ESCI batches have balanced accuracy near 0.5: low drift need not indicate quality. An order-induced ratio change is not a reference-error increase or evidence of a uniformly more accurate ordering.

\begin{{figure*}}[t]
\centering\includegraphics[width=0.98\textwidth]{{figures/controlled_order.pdf}}
\caption{{Controlled row-order variation minus same-condition replay, at each output-key order. Points and unadjusted 95\% query-bootstrap intervals average pools equally; fractions are shown in percentage points. Counts are all-eight-complete/attempted pools. Negative excess means replay variation is larger. These are stability distances, not accuracy gains.}}
\label{{fig:controlled_order}}
\end{{figure*}}

\begin{{table}}[t]
\caption{{Controlled-audit ratio MAE. Batch ranges span four condition means; singleton values average two repeats. All attempted pools complete. Both constant references are reported.}}
\label{{tab:controlled_quality}}
\centering\footnotesize
\begin{{tabular}}{{lrrrr}}
\toprule
Sample / model & Batch range & Singleton & All-Y & All-N \\
\midrule
'''+ '\n'.join(table_rows)+r'''
\bottomrule
\end{tabular}
\end{table}
'''
 (out/'CONTROLLED_RESULTS.tex').write_text(result)
 evidence={'scope':'Completed controlled audit; no hypothesis or sample selection in this renderer.',
   'source_analyses_sha256':{m:hashlib.sha256((out/f'controlled_{m}_analysis.json').read_bytes()).hexdigest() for m in models},
   'quality_table':stats,'effect_modification_intervals_excluding_zero':interaction,
   'requests':sum(x['jobs'] for x in summaries.values()),'valid_requests':valid,'invalid_requests':invalid}
 (out/'controlled_figure_data.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print('Controlled prose/quality table and figure inputs rendered from completed records.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();main(a.bundle,a.out)
