#!/usr/bin/env python3
"""Plot all planned model-by-dataset selective-minus-keyed MAE contrasts."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=[('Qwen3-4B', '#0072B2','o'),('Qwen2.5-7B','#D55E00','s'),('SmolLM2-1.7B','#009E73','^')]
DATASETS=['ESCI','WANDS','TREC-COVID']

def load_json(path):return json.loads(path.read_text(encoding='utf8'))
def product(path):
    x=load_json(path);v=x['comparisons']['actual_fallback_minus_original']
    return {'difference':v['ratio_mae_difference'],'interval':v['ratio_mae_difference_ci'],'pools':x['common_complete_pools'],'keyed_mae':x['quality_common']['original']['ratio_mae'],'selective_mae':x['quality_common']['actual_fallback']['ratio_mae']}
def trec(path):
    x=load_json(path);v=x['primary_difference']
    return {'difference':v['selective_minus_original_ratio_mae'],'interval':v['query_bootstrap_95'],'pools':x['common_complete_topics'],'keyed_mae':x['metrics']['original']['ratio_mae'],'selective_mae':x['metrics']['selective']['ratio_mae']}
def normalize(x):
    if x is None:return None
    x={'difference':float(x['difference']),'interval':[float(y) for y in x['interval']],'pools':int(x['pools']),'keyed_mae':float(x['keyed_mae']),'selective_mae':float(x['selective_mae'])}
    if len(x['interval'])!=2 or x['interval'][0]>x['interval'][1] or x['pools']<=0 or not all(math.isfinite(y) for y in [x['difference'],*x['interval']]):
        raise ValueError('Invalid comparison or interval')
    if not all(0 <= x[k] <= 1 for k in ('keyed_mae','selective_mae')) or not math.isclose(x['selective_mae']-x['keyed_mae'],x['difference'],abs_tol=1e-10):raise ValueError('Absolute MAEs disagree with contrast')
    return x

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--artifact',type=Path);ap.add_argument('--new-root',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fixture',action='store_true');ap.add_argument('--fixture-missing',action='store_true');a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if a.fixture or a.fixture_missing:
        d={dataset:{model:{'difference':(-.035+.009*i+.004*j),'interval':[-.045+.009*i+.004*j,-.025+.009*i+.004*j],'pools':50+j,'keyed_mae':.3+.1*i,'selective_mae':(.3+.1*i-.035+.009*i+.004*j)} for i,(model,_,_) in enumerate(MODELS)} for j,dataset in enumerate(DATASETS)}
        if a.fixture_missing:d['WANDS']['Qwen2.5-7B']=None
    else:
        if not a.artifact or not a.new_root:ap.error('provide --artifact and --new-root')
        b=a.artifact;w=a.new_root
        def new_result(path,summary,parser):
            if not path.exists():return None
            if not summary.is_file() or load_json(summary)['model']!='Qwen2.5-7B-Instruct':raise ValueError('Unexpected new model')
            return parser(path)
        d={'ESCI':{'Qwen3-4B':product(b/'analysis_esci_selective_holdout_qwen/summary.json'),
                   'Qwen2.5-7B':new_result(w/'analysis_esci_7b/summary.json',w/'run_esci_7b/summary.json',product),
                   'SmolLM2-1.7B':product(b/'analysis_esci_selective_holdout_smollm/summary.json')},
           'WANDS':{'Qwen3-4B':product(b/'run_wands_selective_execution_qwen_v1/paired_analysis.json'),
                    'Qwen2.5-7B':new_result(w/'run_wands_7b/paired_analysis.json',w/'run_wands_7b/summary.json',product),
                    'SmolLM2-1.7B':product(b/'run_wands_selective_execution_smollm_v1/paired_analysis.json')},
           'TREC-COVID':{'Qwen3-4B':trec(b/'TREC_COVID_TRANSFER_ANALYSIS.json'),
                         'Qwen2.5-7B':new_result(w/'trec_7b_analysis.json',w/'run_trec_7b/summary.json',trec),
                         'SmolLM2-1.7B':trec(b/'TREC_COVID_SMOL_ANALYSIS.json')}}
    d={dataset:{model:normalize(record) for model,record in values.items()} for dataset,values in d.items()}
    extrema=[0.0]+[v for group in d.values() for record in group.values() if record is not None for v in [record['difference'],*record['interval']]]
    lo,hi=min(extrema),max(extrema);span=max(hi-lo,0.01);limits=(lo-.14*span,hi+.72*span)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.7),sharex=True,sharey=True)
    for ax,dataset in zip(axes,DATASETS):
        ax.axvline(0,color='#555555',lw=.7,zorder=0);ax.set_title(dataset,fontsize=9,fontweight='bold',pad=7);ax.text(.98,1.02,'keyed → selective',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.1,color='#444444')
        for i,(model,color,marker) in enumerate(MODELS):
            y=2-i;row=d[dataset][model]
            if row is None:
                ax.text(.5,y,'unavailable',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=7,color='#555555')
                continue
            low,high=row['interval'];x=row['difference']
            ax.hlines(y,low,high,color=color,lw=1.3,zorder=2);ax.plot([low,high],[y,y],linestyle='none',marker='|',color=color,markersize=5)
            ax.scatter([x],[y],marker=marker,s=28,color=color,zorder=3)
            ax.text(.98,y,f"{row['keyed_mae']:.3f}→{row['selective_mae']:.3f}\nn={row['pools']}",transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.1,color='#444444',linespacing=1.1)
        ax.set_xlim(*limits);ax.set_yticks([2,1,0]);ax.set_ylim(-.6,2.6);ax.grid(axis='x',color='#dddddd',lw=.5)
        ax.tick_params(axis='both',length=0,labelsize=7);ax.set_xlabel('Selective − keyed ratio MAE',fontsize=7.3,labelpad=5)
    axes[0].set_yticklabels([m[0] for m in MODELS]);fig.subplots_adjust(left=.16,right=.99,top=.79,bottom=.24,wspace=.16)
    fig.suptitle(('SYNTHETIC LAYOUT CHECK — ' if (a.fixture or a.fixture_missing) else '')+'Paired query-level differences; lower is better',fontsize=9,y=.98)
    a.out.mkdir(parents=True)
    (a.out/'figure_data.json').write_text(json.dumps({'values':d,'limits':limits,'fixture':bool(a.fixture or a.fixture_missing)},indent=2)+'\n',encoding='utf8')
    fig.savefig(a.out/'model_sensitivity_3datasets.pdf',bbox_inches='tight')
    fig.savefig(a.out/'model_sensitivity_3datasets.png',dpi=240,bbox_inches='tight')
    plt.close(fig)
    print('Saved 3-dataset figure and its numerical data:',a.out)
if __name__=='__main__':main()
