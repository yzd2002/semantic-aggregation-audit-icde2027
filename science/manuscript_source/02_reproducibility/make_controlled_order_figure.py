#!/usr/bin/env python3
"""Plot completed controlled-order analyses; never create placeholder results.

Inputs are the two real analyze_controlled.py JSON outputs. This file is fully
self-contained. No project modules, execution traces, or GPU models are imported.
The statistics are stability distances, not reference-label accuracy measures.
Negative excess and intervals crossing zero are retained without truncation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

CO_WIDTH, CO_HEIGHT = 7.10, 3.10
CO_INK, CO_MUTED = '#202D34', '#53616A'
CO_TEAL, CO_RUST = '#246B70', '#9B4B2A'
CO_METRICS = ('candidate_disagreement', 'absolute_ratio_change')


def co_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_controlled_analysis(path, expected_model):
    """Validate completed-analysis structure and means; use stored CIs unchanged."""
    path=Path(path)
    source=json.loads(path.read_text())
    if source.get('fixture') or source.get('model') != expected_model:
        raise ValueError(f'Expected real completed {expected_model} analysis, not a fixture')
    for key in ('responses_sha256','analysis_code_sha256'):
        if not re.fullmatch(r'[0-9a-fA-F]{64}',source.get(key,'')):
            raise ValueError(f'Missing provenance field: {key}')
    if set(source.get('datasets',{})) != {'esci','wands'}:
        raise ValueError('Both ESCI and WANDS analyses are required')
    for dataset,entry in source['datasets'].items():
        n,attempted=entry['primary_common_pools'],entry['attempted_pools']
        if not isinstance(n,int) or not (0<n<=attempted):
            raise ValueError(f'{dataset}: no positive all-eight-complete analysis sample')
        per_pool=entry['per_pool']
        if len(per_pool)!=n or len(entry['primary_common_query_ids'])!=n:
            raise ValueError(f'{dataset}: primary sample count mismatch')
        if [r['query_id'] for r in per_pool] != entry['primary_common_query_ids']:
            raise ValueError(f'{dataset}: primary query order mismatch')
        if set(entry['co_primary']) != {'A','D'}:
            raise ValueError(f'{dataset}: missing output-key condition')
        for key_order in ('A','D'):
            for metric in CO_METRICS:
                data=entry['co_primary'][key_order][metric]
                values=[]
                for row in per_pool:
                    p=row['orders'][key_order][metric]
                    if not math.isclose(p['row_change']-p['mean_replay'],p['excess'],abs_tol=1e-12):
                        raise ValueError('Per-pool excess is not row-change minus replay')
                    values.append(p['excess'])
                mean=data['excess']['mean'];lo,hi=data['excess']['unadjusted_query_interval_95']
                if not all(math.isfinite(v) for v in (mean,lo,hi)) or lo>hi:
                    raise ValueError('Nonfinite or reversed excess interval')
                if not all(-1-1e-12<=v<=1+1e-12 for v in (mean,lo,hi)):
                    raise ValueError('Distances/excess must be supplied as fractions, not already multiplied by 100')
                if not math.isclose(sum(values)/n,mean,abs_tol=1e-12):
                    raise ValueError('Recorded query-mean excess differs from per-pool data')
                if not math.isclose(data['row_change']['mean']-data['mean_replay']['mean'],mean,abs_tol=1e-12):
                    raise ValueError('Aggregate excess arithmetic differs')
    sibling=path.parent/'responses.jsonl'
    provenance={'analysis_sha256':co_sha(path),'recorded_responses_sha256':source['responses_sha256'],
                'recorded_analysis_code_sha256':source['analysis_code_sha256'],
                'sibling_response_file_hash_checked':False}
    if sibling.exists():
        if co_sha(sibling)!=source['responses_sha256']:raise ValueError('Analysis/response-file hash mismatch')
        provenance['sibling_response_file_hash_checked']=True
    return source,provenance


def co_limits(rows, metric):
    values=[0.0]
    for row in rows:
        d=row['statistics'][metric]['excess']
        values.extend([100*d['mean'],*[100*x for x in d['unadjusted_query_interval_95']]])
    low,high=min(values),max(values)
    pad=.12*max(high-low,1.0)
    return low-pad,high+pad


def controlled_order_plot(qwen_path, smol_path, out):
    sources={};provenance={}
    for model,path in [('qwen',qwen_path),('smol',smol_path)]:
        sources[model],provenance[model]=read_controlled_analysis(path,model)
    rows=[]
    names={'qwen':'Qwen3-4B','smol':'SmolLM2-1.7B'}
    for dataset in ('esci','wands'):
        for model in ('qwen','smol'):
            entry=sources[model]['datasets'][dataset]
            for key in ('A','D'):
                rows.append({'dataset':dataset.upper(),'model':names[model],'key_order':key,
                             'pools':entry['primary_common_pools'],'attempted_pools':entry['attempted_pools'],
                             'bootstrap_draws':entry['bootstrap_draws'],'bootstrap_seed':entry['bootstrap_seed'],
                             'statistics':entry['co_primary'][key]})
    # No output directory or plot is created until both real inputs validate.
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rc={'font.family':'DejaVu Sans','font.size':8.3,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','svg.hashsalt':'intentsat-controlled-order-v1',
        'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'dejavusans'}
    evidence={'scope':'Query-mean row-order distance minus mean identical-condition replay distance; stability, not semantic accuracy.',
              'intervals':'Unadjusted 95% query-bootstrap intervals copied from completed analyses.',
              'units':'percentage points; all source fractional values multiplied by exactly 100',
              'negative_values_retained':True,'zero_crossing_intervals_retained':True,
              'source_provenance':provenance,'points':[]}
    with plt.rc_context(rc):
        fig=plt.figure(figsize=(CO_WIDTH,CO_HEIGHT))
        overlay=fig.add_axes([0,0,1,1],zorder=5)
        overlay.set(xlim=(0,CO_WIDTH),ylim=(0,CO_HEIGHT));overlay.axis('off');overlay.patch.set_visible(False)
        def text(x,y,s,size=8.3,color=CO_INK,weight='normal',ha='left'):
            return overlay.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center')
        text(.07,2.98,'Excess row-order variation vs replay',size=9.4,weight='bold')
        for x,key,color,marker,label in [(5.03,'A',CO_TEAL,'o','keys ↑ (A)'),(6.13,'D',CO_RUST,'s','keys ↓ (D)')]:
            overlay.plot([x,x+.24],[2.98,2.98],color=color,lw=1)
            overlay.plot([x+.12],[2.98],marker=marker,color=color,ms=4.8,mec=CO_INK,mew=.35)
            text(x+.31,2.98,label,size=8.2)
        bottom,plot_h=.43,2.09
        def yin(y):return bottom+(y+.5)/8*plot_h
        for group_index in range(4):
            row=rows[2*group_index];y=6.5-2*group_index
            text(.07,yin(y)+.075,f"{row['dataset']} / {row['model']}",size=8.3,weight='bold')
            text(.07,yin(y)-.095,f"n={row['pools']} / {row['attempted_pools']} pools",size=8.2,color=CO_MUTED)
        for i,row in enumerate(rows):
            text(1.87,yin(7-i),row['key_order'],size=8.3,ha='center',
                 color=CO_TEAL if row['key_order']=='A' else CO_RUST)
        for y in (5.5,3.5,1.5):overlay.plot([.07,7.02],[yin(y),yin(y)],color='#CED6D9',lw=.55)
        for x,width,value_x,metric,title in [
            (2.10,1.90,4.37,'candidate_disagreement','A  Candidate disagreement'),
            (4.66,1.94,7.02,'absolute_ratio_change','B  Absolute ratio change')]:
            limits=co_limits(rows,metric)
            ax=fig.add_axes([x/CO_WIDTH,bottom/CO_HEIGHT,width/CO_WIDTH,plot_h/CO_HEIGHT],zorder=2)
            ax.set(xlim=limits,ylim=(-.5,7.5),yticks=[])
            locator=MaxNLocator(nbins=4,min_n_ticks=3,steps=[1,2,2.5,5,10])
            ax.set_xticks([v for v in locator.tick_values(*limits) if limits[0]<=v<=limits[1]])
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.tick_params(axis='x',labelsize=8.3,length=2.5,pad=3,width=.55,colors=CO_MUTED)
            for spine in ('left','right','top'):ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#9CA9AF');ax.spines['bottom'].set_linewidth(.55)
            ax.set_axisbelow(True);ax.grid(axis='x',color='#E0E6E8',lw=.5)
            # Neutral zero line: no favorable-side shading or accuracy claim.
            ax.axvline(0,color='#75888F',lw=.9,zorder=1)
            text(x,2.71,title,size=9.0,weight='bold');text(value_x,2.71,'pp',size=8.2,color=CO_MUTED,ha='right')
            for i,row in enumerate(rows):
                original=row['statistics'][metric]['excess'];value=100*original['mean']
                lo,hi=[100*v for v in original['unadjusted_query_interval_95']];y=7-i
                if not (limits[0]<min(lo,value)<=max(hi,value)<limits[1]):raise ValueError('An interval would be clipped')
                color,marker=(CO_TEAL,'o') if row['key_order']=='A' else (CO_RUST,'s')
                seg=ax.hlines(y,lo,hi,color=color,lw=1.15,zorder=3)
                ax.plot([lo,hi],[y,y],linestyle='none',marker='|',markersize=5,color=color,mew=1,zorder=3)
                dot,=ax.plot([value],[y],linestyle='none',marker=marker,ms=4.8,color=color,mec=CO_INK,mew=.4,zorder=4)
                if float(dot.get_xdata()[0])!=value or seg.get_segments()[0].tolist()!=[[lo,y],[hi,y]]:
                    raise ValueError('Plotted coordinates differ from source × 100')
                label=f'{value:+.2f}'.replace('-','−')
                text(value_x,yin(y),label,size=8.3,ha='right')
                evidence['points'].append({k:row[k] for k in ('dataset','model','key_order','pools','attempted_pools','bootstrap_draws','bootstrap_seed')}|
                    {'metric':metric,'source_mean':original['mean'],'source_interval':original['unadjusted_query_interval_95'],
                     'estimate_pp':value,'interval_pp':[lo,hi],'display_estimate_pp':label,'plot_coordinates_verified':True})
        text(4.53,.10,'Excess variation (percentage points)',size=8.5,ha='center')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
        texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        for i,t in enumerate(texts):
            bb=t.get_window_extent(renderer)
            if not(canvas.x0<=bb.x0<=bb.x1<=canvas.x1 and canvas.y0<=bb.y0<=bb.y1<=canvas.y1):
                raise ValueError(f'Text leaves canvas: {t.get_text()}')
            if t.get_fontsize()<8.2:raise ValueError('Font below 8.2 pt')
            for q in texts[i+1:]:
                if bb.overlaps(q.get_window_extent(renderer)):raise ValueError(f'Text collision: {t.get_text()} / {q.get_text()}')
        evidence.update({'canvas_inches':[CO_WIDTH,CO_HEIGHT],'minimum_nominal_font_pt':min(t.get_fontsize() for t in texts),
                         'minimum_font_pt_at_0.98_ieeetran_textwidth':8.2*(.98*43*12/72.27)/CO_WIDTH,
                         'all_text_inside_canvas':True,'text_boxes_nonoverlapping':True,
                         'no_fixture_or_placeholder_results_generated':True})
        for ext in ('pdf','svg','png'):
            if ext=='pdf':
                dt=datetime(2026,11,2,tzinfo=timezone.utc)
                metadata={'Creator':'IntentSat-2S controlled-order figure generator','CreationDate':dt,'ModDate':dt}
            elif ext=='svg':metadata={'Creator':'IntentSat-2S controlled-order figure generator','Date':'2026-11-02T00:00:00Z'}
            else:metadata={'Software':'IntentSat-2S controlled-order figure generator'}
            fig.savefig(out/f'controlled_order.{ext}',dpi=300,metadata=metadata)
        fig.savefig(out/'controlled_order_144dpi.png',dpi=144)
        (out/'controlled_order_checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
        manifest={'generator_sha256':co_sha(Path(__file__)),'source_analysis_sha256':{model:p['analysis_sha256'] for model,p in provenance.items()},
                  'figure_files_sha256':{f'controlled_order.{ext}':co_sha(out/f'controlled_order.{ext}') for ext in ('pdf','svg','png')}}
        (out/'CONTROLLED_FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        plt.close(fig)
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen-analysis',type=Path,required=True)
    p.add_argument('--smol-analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=controlled_order_plot(a.qwen_analysis,a.smol_analysis,a.out)
    print(json.dumps({'figures_generated_from_real_analyses':True,'plotted_estimates':len(r['points']),
                      'canvas_inches':r['canvas_inches'],'minimum_font_pt_at_paper_scale':r['minimum_font_pt_at_0.98_ieeetran_textwidth']},indent=2))
