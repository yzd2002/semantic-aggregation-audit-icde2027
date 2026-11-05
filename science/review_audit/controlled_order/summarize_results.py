#!/usr/bin/env python3
"""Post-run display only: produce compact tables from independently verified analyses."""
import argparse,csv,json
from pathlib import Path
from common import ROOT

def estimate(x):
    low,high=x['unadjusted_query_interval_95']
    return f"{100*x['mean']:.2f} [{100*low:.2f}, {100*high:.2f}]"

def main(models,out):
    out.mkdir(exist_ok=False);primary=[];secondary=[];quality=[];notes=[];tex=[]
    for model in models:
        folder=ROOT/f'run_{model}';a=json.loads((folder/'analysis.json').read_text());b=json.loads((folder/'supplementary_analysis.json').read_text())
        summary=json.loads((folder/'summary.json').read_text());audit=json.loads((folder/'EXECUTION_AUDIT.json').read_text())
        assert audit['status']=='PASS'
        notes.append(f"{model}: {summary['jobs']} requests, {summary['valid_jobs']} valid, mixed workload {summary['mixed_workload_seconds']:.2f} s; actual input/output tokens {summary['actual_tokens']}.")
        for dataset,entry in a['datasets'].items():
            notes.append(f"{model}/{dataset}: attempted {entry['attempted_pools']}; all8={entry['primary_common_pools']}; all10={entry['batch_singleton_common_pools']}.")
            for order in ('A','D'):
                item={'model':model,'dataset':dataset,'n':entry['primary_common_pools'],'output_order':order}
                for metric,data in entry['co_primary'][order].items():
                    for kind,value in data.items():
                        item[metric+'_'+kind]=value['mean']
                        item[metric+'_'+kind+'_low'],item[metric+'_'+kind+'_high']=value['unadjusted_query_interval_95']
                primary.append(item)
                candidate=entry['co_primary'][order]['candidate_disagreement'];ratio=entry['co_primary'][order]['absolute_ratio_change']
                notes.append(f"  Output {order}: candidate row/replay {100*candidate['row_change']['mean']:.3f}/{100*candidate['mean_replay']['mean']:.3f} pp; excess {estimate(candidate['excess'])}; ratio row/replay {100*ratio['row_change']['mean']:.3f}/{100*ratio['mean_replay']['mean']:.3f} pp; excess {estimate(ratio['excess'])}.")
                notes.append('  Reference ratio-MAE reverse minus canonical (pp): '+estimate(entry['reference_quality_row_order'][order]['reference_ratio_mae_reverse_minus_canonical']))
                tex.append(f"{model} & {dataset} & {order} & {estimate(candidate['excess'])} & {estimate(ratio['excess'])} \\\\")
            notes.append('  Excess interaction D minus A, candidate/ratio (pp): '+', '.join(estimate(entry['effect_modification_D_minus_A'][metric]) for metric in ('candidate_disagreement','absolute_ratio_change')))
            for order,data in b['datasets'][dataset]['secondary_key_order'].items():
                item={'model':model,'dataset':dataset,'input_order':order,'n':entry['primary_common_pools']}
                for metric in ('candidate_disagreement','absolute_ratio_change'):
                    stats=data[metric]
                    for kind,value in stats.items():
                        item[metric+'_'+kind]=value['mean'];item[metric+'_'+kind+'_low'],item[metric+'_'+kind+'_high']=value['unadjusted_query_interval_95']
                secondary.append(item)
                notes.append(f"  Key-order secondary at input {order}, candidate/ratio excess (pp): "+', '.join(estimate(data[metric]['excess']) for metric in ('candidate_disagreement','absolute_ratio_change')))
                notes.append('  Reference ratio-MAE descending minus ascending (pp): '+estimate(data['reference_ratio_mae_D_minus_A']))
            for condition in ('AA','RA','AD','RD'):
                notes.append(f"  Batch-singleton {condition}, candidate/ratio (pp): "+', '.join(estimate(entry['batch_singleton'][condition][metric]) for metric in ('candidate_disagreement','absolute_ratio_change')))
            for condition in ('AA','RA','AD','RD','S'):
                values=[entry['quality_all8'][condition+str(k)] if condition!='S' else entry['singleton_quality_own_common'][str(k)] for k in (0,1)]
                means={key:sum(value[key] for value in values)/2 for key in ('ratio_mae','f1','balanced_accuracy','candidate_error_macro')}
                notes.append(f"  {condition} reference quality, mean over two replays: "+', '.join(f'{key}={value:.6f}' for key,value in means.items()))
            notes.append('  Constant reference ratio-MAE: '+', '.join(f"{key}={value['ratio_mae']:.6f}" for key,value in entry['constants_all8'].items()))
            notes.append('  Singleton replay candidate/ratio (pp): '+', '.join(estimate(entry['singleton_replay'][metric]) for metric in ('candidate_disagreement','absolute_ratio_change')))
            for plan,values in entry['quality_all8'].items():quality.append({'model':model,'dataset':dataset,'plan':plan,**values,**{'tokens_'+k:v for k,v in entry['actual_tokens'][plan].items()}})
            for rep,values in entry['singleton_quality_own_common'].items():quality.append({'model':model,'dataset':dataset,'plan':'S'+rep,**values,**{'tokens_'+k:v for k,v in entry['actual_tokens']['S'+rep].items()}})
    for name,rows in [('co_primary',primary),('key_order_secondary',secondary),('quality_and_tokens',quality)]:
        with (out/(name+'.tsv')).open('x') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    (out/'RESULTS_summary.txt').write_text('All percentages below are percentage points. Intervals are unadjusted paired query-bootstrap descriptions, 4000 draws.\n\n'+'\n'.join(notes)+'\n')
    (out/'co_primary_rows.tex').write_text('% Columns: model, dataset, output order, candidate excess pp [95% CI], absolute ratio-change excess pp [95% CI]\n'+'\n'.join(tex)+'\n')
    print('Wrote report-only tables:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',choices=['qwen','smol'],default=['qwen','smol']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.models,a.out)
