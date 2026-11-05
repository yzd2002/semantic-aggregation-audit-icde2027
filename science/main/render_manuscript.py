#!/usr/bin/env python3
"""Inject recorded, rounded metrics into the IEEE manuscript template."""
import json
from pathlib import Path
base=Path(__file__).resolve().parent.parent
p=base/'02_reproducibility/run_validation_500';s=json.loads((p/'summary.json').read_text());a=json.loads((p/'paired_analysis.json').read_text())
f=base/'02_reproducibility/run_full_pools_250';fs=json.loads((f/'summary.json').read_text());fa=json.loads((f/'paired_analysis.json').read_text())
sm=base/'02_reproducibility/run_full_pools_smollm_250'; ss=json.loads((sm/'summary.json').read_text()); sc=json.loads((sm/'crossmodel_analysis.json').read_text()); sj=json.loads((sm/'json_ablation.json').read_text())
sb=base/'02_reproducibility/run_supervised_baseline_1000'; sbj=json.loads((sb/'summary.json').read_text()); sbq=sbj['qwen_common_207']; sbm=sbq['methods']['supervised_lexical']
lb=base/'02_reproducibility/run_lotus_baseline'; lbj=json.loads((lb/'summary.json').read_text()); lba=json.loads((lb/'parse_audit.json').read_text()); lbq=lbj['qwen_common_207']; lbm=lbq['methods']['lotus_sem_filter']
lc=json.loads((base/'02_reproducibility/run_lotus_wands_cascade/summary.json').read_text())
lcp=json.loads((base/'02_reproducibility/run_lotus_wands_cascade/proxy_position_audit.json').read_text())
assert lc['pools']==52 and lc['candidates']==1664 and lcp['first_generated_tokens']=={'Answer':1664} and lcp['first_both_label_positions']=={'0':1664}
wq=base/'02_reproducibility/run_wands_160'; wqs=json.loads((wq/'summary.json').read_text()); wqa=json.loads((wq/'paired_analysis.json').read_text())
wsm=base/'02_reproducibility/run_wands_smollm_160'; wsms=json.loads((wsm/'summary.json').read_text()); wsma=json.loads((wsm/'paired_analysis.json').read_text()); wc=json.loads((wsm/'crossmodel_analysis.json').read_text())
fb=json.loads((f/'failure_bounds.json').read_text())['methods']['two_stage_compact_fallback']; wb=json.loads((wq/'failure_bounds.json').read_text())['methods']['two_stage_compact_fallback']
pb=json.loads((base/'02_reproducibility/run_partial_aggregate_bias/summary.json').read_text())['datasets']
assert pb['esci_250']['failed_pools']==fb['failed_pools'] and pb['wands_160']['failed_pools']==wb['failed_pools']
assert all(pb[k]['failed_pools']==pb[k]['available_only_defined_pools'] for k in ('esci_250','wands_160'))
fl=json.loads((base/'02_reproducibility/run_failure_lexical_fallback/summary.json').read_text())['datasets']
assert fl['esci_qwen']['failed_pools']==fb['failed_pools'] and fl['wands_qwen']['failed_pools']==wb['failed_pools']
assert fl['esci_smollm']['failed_pools']==250 and fl['wands_smollm']['failed_pools']==160
tc=json.loads((base/'02_reproducibility/TREC_COVID_TRANSFER_ANALYSIS.json').read_text())
tcm=json.loads((base/'02_reproducibility/trec_covid_transfer_selection/selection.json').read_text())
tcs=json.loads((base/'02_reproducibility/run_trec_covid_transfer/summary.json').read_text())
tcsm=json.loads((base/'02_reproducibility/TREC_COVID_SMOL_ANALYSIS.json').read_text())
tcsmrun=json.loads((base/'02_reproducibility/run_trec_covid_smollm/summary.json').read_text())
assert tc['common_complete_topics']==tc['topics_attempted']==tcm['included_topics']==50
assert all(x==50 for x in tc['completion'].values()) and tcm['sampled_candidates']==1600
assert tcs['selection_inputs_sha256']==tcm['inputs_sha256'] and tcm['nonstandard_grade_minus_one_excluded']==2
assert tcsm['common_topic_ids']==tc['common_topic_ids'] and all(x==50 for x in tcsm['completion'].values())
assert tcsmrun['protocol']=='PROTOCOL_trec_covid_smollm.md' and tcsmrun['selection_inputs_sha256']==tcm['inputs_sha256']
tr=json.loads((base/'02_reproducibility/ESCI_TRUNCATION_ANALYSIS.json').read_text())
trmeta=json.loads((base/'02_reproducibility/esci_truncation_selection/selection.json').read_text())
assert tr['selection_inputs_sha256']==trmeta['inputs_sha256']
assert tr['common_complete_pools']==tr['attempted_pools']==trmeta['pools']==250
assert tr['attempted_candidates']==trmeta['candidates']==5160 and all(x==250 for x in tr['completion'].values())
wr=json.loads((base/'02_reproducibility/run_offline_wands_replay/summary.json').read_text())
assert wr['attempted_queries']==16 and wr['attempted_chunks_per_mode']==64
assert all(x['complete_pools']==16 and x['valid_chunks']==64 for x in wr['completion'].values())
rf=json.loads((base/'02_reproducibility/run_ratio_instability/summary.json').read_text())
assert rf['common_complete_pools']==160 and rf['candidates_per_pool']==32
ab=json.loads((base/'02_reproducibility/run_compact_prompt_ablation_smollm/summary.json').read_text())['datasets']
pa=json.loads((base/'02_reproducibility/run_compact_parse_audit_smollm/summary.json').read_text())['datasets']
gq=json.loads((base/'02_reproducibility/run_guided_compact_qwen_v1/paired_analysis.json').read_text())['datasets']
gs=json.loads((base/'02_reproducibility/run_guided_compact_smollm_v1/paired_analysis.json').read_text())['datasets']
oq=json.loads((base/'02_reproducibility/run_reverse_order_qwen_v1/paired_analysis.json').read_text())['datasets']
om=json.loads((base/'02_reproducibility/run_reverse_order_smollm_v1/paired_analysis.json').read_text())['datasets']
ocq=json.loads((base/'02_reproducibility/run_order_controls_qwen_v1/paired_analysis.json').read_text())['datasets']
ocm=json.loads((base/'02_reproducibility/run_order_controls_smollm_v1/paired_analysis.json').read_text())['datasets']
holdout=json.loads((base/'02_reproducibility/wands_holdout_selection/selection.json').read_text())
hq=json.loads((base/'02_reproducibility/run_wands_holdout_qwen_v1/paired_analysis.json').read_text())
hs=json.loads((base/'02_reproducibility/run_wands_holdout_smollm_v1/paired_analysis.json').read_text())
assert rf['common_complete_pools']==hq['common_complete_pools'] and rf['candidate_disagreements']==hq['comparisons']['reverse_stable_ids']['changed_candidate_count']
random_q=json.loads((base/'02_reproducibility/analysis_random_order_qwen/summary.json').read_text())
random_s=json.loads((base/'02_reproducibility/analysis_random_order_smollm/summary.json').read_text())
assert random_q['common_complete_pools']==random_s['common_complete_pools']==160
assert random_q['common_query_ids']==random_s['common_query_ids']==hq['common_query_ids']
assert all(x['complete_pools']==160 and x['valid_chunks']==640 for j in (random_q,random_s) for x in j['completion'].values())
assert all(j['comparisons'][key]['changed_candidates']==0 for j in (random_q,random_s) for key in ('baseline_b_vs_baseline_a','baseline_c_vs_baseline_a'))
random_keys=('perm_1_vs_baseline_a','perm_2_vs_baseline_b','perm_3_vs_baseline_c')
random_qrange=[100*random_q['comparisons'][k]['change_fraction'] for k in random_keys]
random_srange=[100*random_s['comparisons'][k]['change_fraction'] for k in random_keys]
hq2=json.loads((base/'02_reproducibility/run_wands_holdout_qwen_v1/two_way_analysis.json').read_text())
hs2=json.loads((base/'02_reproducibility/run_wands_holdout_smollm_v1/two_way_analysis.json').read_text())
kq=json.loads((base/'02_reproducibility/run_keyed_order_qwen_v1/paired_analysis.json').read_text())
ks=json.loads((base/'02_reproducibility/run_keyed_order_smollm_v1/paired_analysis.json').read_text())
assert kq['common_query_ids']==ks['common_query_ids']
fvmeta=json.loads((base/'02_reproducibility/wands_fallback_validation_selection/selection.json').read_text())
fvs=json.loads((base/'02_reproducibility/run_wands_fallback_validation_qwen_v1/summary.json').read_text())
fva=json.loads((base/'02_reproducibility/run_wands_fallback_validation_qwen_v1/paired_analysis.json').read_text())
wc96=json.loads((base/'02_reproducibility/run_wands96_composition/summary.json').read_text())
wcp=json.loads((base/'02_reproducibility/pool_error_calibration/summary.json').read_text())
fv2=json.loads((base/'02_reproducibility/run_wands_fallback_validation_qwen_v1/two_way_analysis.json').read_text())
assert fva['common_complete_pools']==fva['attempted_pools']==fvmeta['sample_size']==96
svmeta=json.loads((base/'02_reproducibility/wands_selective_execution_selection/selection.json').read_text())
svs=json.loads((base/'02_reproducibility/run_wands_selective_execution_qwen_v1/summary.json').read_text())
sva=json.loads((base/'02_reproducibility/run_wands_selective_execution_qwen_v1/paired_analysis.json').read_text())
sv2=json.loads((base/'02_reproducibility/run_wands_selective_execution_qwen_v1/two_way_analysis.json').read_text())
sms=json.loads((base/'02_reproducibility/run_wands_selective_execution_smollm_v1/summary.json').read_text())
sma=json.loads((base/'02_reproducibility/run_wands_selective_execution_smollm_v1/paired_analysis.json').read_text())
sm2=json.loads((base/'02_reproducibility/run_wands_selective_execution_smollm_v1/two_way_analysis.json').read_text())
wsl=json.loads((base/'02_reproducibility/run_wands_supervised_reference/summary.json').read_text())
wcv=json.loads((base/'02_reproducibility/run_wands_calibrated_lexical/summary.json').read_text())
wrb=json.loads((base/'02_reproducibility/run_wands_ratio_baseline/summary.json').read_text())
wrps=json.loads((base/'02_reproducibility/run_wands_ratio_product_sensitivity/summary.json').read_text())
assert len(wcv['folds'])==5 and all(not x['convergence_warning'] for x in wcv['folds'])
wen=json.loads((base/'02_reproducibility/run_wands_embedding_reference/summary.json').read_text())
wer=json.loads((base/'02_reproducibility/run_wands_embedding_ratio_reference/summary.json').read_text())
svc=json.loads((base/'02_reproducibility/run_local_service_benchmark_resumed/summary.json').read_text())
assert len(svc['conditions'])==12 and all(x['attempted_pools']==52 for x in svc['conditions'])
assert wen['training']['candidates']==5120 and wen['representation']['dimension']==1024
assert sva['common_complete_pools']==sva['attempted_pools']==svmeta['sample_size']==52
assert sma['common_query_ids']==sva['common_query_ids'] and sma['common_complete_pools']==52
ehmeta=json.loads((base/'02_reproducibility/esci_selective_holdout_selection/selection.json').read_text())
ehq=json.loads((base/'02_reproducibility/analysis_esci_selective_holdout_qwen/summary.json').read_text())
ehs=json.loads((base/'02_reproducibility/analysis_esci_selective_holdout_smollm/summary.json').read_text())
ehqr=json.loads((base/'02_reproducibility/run_esci_selective_holdout_qwen/summary.json').read_text())
ehref=json.loads((base/'02_reproducibility/analysis_esci_selective_supervised_reference/summary.json').read_text())
ehover=json.loads((base/'02_reproducibility/ESCI_SELECTIVE_OVERLAP_ANALYSIS.json').read_text())
ehsens=json.loads((base/'02_reproducibility/analysis_esci_selective_text_overlap/summary.json').read_text())
labelmap=json.loads((base/'02_reproducibility/posthoc_esci_label_mapping/result.json').read_text())
varfloor=json.loads((base/'02_reproducibility/posthoc_variable_pool_floor/summary.json').read_text())
assert varfloor['pools']==250 and varfloor['candidates']==4428

assert labelmap['query_pools']==250 and labelmap['candidate_count']==4428

assert ehmeta['selected_queries']==ehq['attempted_pools']==ehs['attempted_pools']==250
assert ehmeta['candidates']==ehq['candidate_count']==ehs['candidate_count']==4428
assert ehq['selection_inputs_sha256']==ehs['selection_inputs_sha256']==ehmeta['inputs_sha256']
assert ehq['common_query_ids']==ehs['common_query_ids'] and ehq['common_complete_pools']==ehs['common_complete_pools']==250
assert all(n==250 for result in (ehq,ehs) for n in result['completion'].values())
assert ehover['new_candidate_rows_with_any_exact_text_overlap']==20 and ehover['new_queries_with_any_exact_text_overlap']==18
assert ehsens['included_queries']==232 and ehsens['included_candidates']==4126
assert wsl['training']['pools']==160 and wsl['training']['candidates']==5120 and not wsl['training']['convergence_warning']
assert all(x['training_query_overlap']==0 and x['training_product_overlap']==0 for x in wsl['samples'].values())
sv_cross=sv2['actual_fallback_minus_original_ratio_mae_query_product_ci']
sv_cross_abstract='spans zero' if sv_cross[0]<=0<=sv_cross[1] else 'does not span zero'
assert sc['common_pools']==0 and ss['plans']['two_stage_compact_fallback']['complete_pools']==0
assert ss['plans']['pointwise']['complete_pools']==250 and ss['plans']['pointwise']['tp']==0 and ss['plans']['pointwise']['fp']==0
assert not sbj['convergence_warning'] and sbq['pools']==207 and round(sbm['f1'],3)==0.836 and round(sbm['balanced_accuracy'],3)==0.557
assert lbq['pools']==207 and lba['postprocessed_defaulted_candidates']==0 and lba['strict_incomplete_pools']==0 and round(lbm['f1'],3)==0.556
assert wqa['common_complete_pools']==141 and wqs['grade_counts']=={'Partial':3282,'Exact':781,'Irrelevant':1057}
assert ab['esci']['corrected_complete_pools']==0 and ab['wands']['corrected_complete_pools']==0
order=[('pointwise','Pointwise (P)'),('batch_json','Batch JSON (J)'),('two_stage_json','Two-stage JSON (TJ)'),('batch_compact','Batch compact (C)'),('two_stage_compact_fallback','Two-stage compact + fallback (TC)')]
fullorder=[('pointwise','Pointwise (P)'),('batch_json','Batch JSON (J)'),('two_stage_json','Two-stage JSON (TJ)'),('two_stage_compact_fallback','Two-stage compact + fallback (TC)')]
def table(summary,analysis,order,n,constants):
 rows=[]
 for k,name in order:
  x=summary['plans'][k];y=analysis['methods'][k]
  rows.append(f"{name} & {x['complete_pools']}/{n} & {y['f1']:.3f} & {y['balanced_accuracy']:.3f} & {y['mae']:.3f} & {x['calls']:,} & {x['prompt_tokens']/1000:.1f}k & {x['output_tokens']/1000:.1f}k & {x['seconds']:.2f} " + chr(92)*2)
 for k,name in constants:
  y=analysis['methods'][k]
  rows.append(f"{name} & {n}/{n} & {y['f1']:.3f} & {y['balanced_accuracy']:.3f} & {y['mae']:.3f} & -- & -- & -- & -- " + chr(92)*2)
 return '\n'.join(rows)
def model_sensitivity():
 counts=[wsms['plans'][key]['complete_pools'] for key,_ in fullorder]
 sentence=f"For SmolLM2, pointwise, direct JSON, two-stage JSON, and compact fallback complete {counts[0]}, {counts[1]}, {counts[2]}, and {counts[3]} of 160 WANDS pools. "
 pt=wsms['plans']['pointwise']
 if counts[0]==160 and pt['tp']==0 and pt['fp']==0:
  sentence+=f"Pointwise labels all 5,120 candidates N, giving F1 {pt['f1']:.3f} and ratio MAE {pt['ratio_mae']:.3f} on all pools. "
 if wsma['common_complete_pools']==0:
  sentence+='No pool is completed by all four SmolLM2 plans, so model-specific four-plan paired quality is unavailable. '
 else:
  sentence+=f"Its four-plan common subset has {wsma['common_complete_pools']} pools; conditional pointwise and two-stage JSON F1 are {wsma['methods']['pointwise']['f1']:.3f} and {wsma['methods']['two_stage_json']['f1']:.3f}. "
 if wc['common_pools']==0:
  sentence+='The pre-specified intersection across both models and all four plans is empty. '
 else:
  sentence+=f"The pre-specified intersection across both models and all four plans contains {wc['common_pools']} pools. "
 return sentence+'Different model family, size, tokenizer, and template change together, and the prompts were not separately tuned; these observations are not an isolated architecture comparison.'
def guided_comparison():
 qe,qw=gq['esci'],gq['wands']
 eu,eg=qe['quality_common_complete']['unconstrained'],qe['quality_common_complete']['guided']
 wu,wg=qw['quality_common_complete']['unconstrained'],qw['quality_common_complete']['guided']
 ec=qe['paired_bootstrap']['f1_guided_minus_unconstrained_ci'];wc=qw['paired_bootstrap']['f1_guided_minus_unconstrained_ci']
 text=(f"In a further post-hoc comparison using the example-removed prompt, regex-guided decoding changes Qwen's ESCI completion from {qe['unconstrained_complete_pools']}/250 to {qe['guided_complete_pools']}/250 and WANDS completion from {qw['unconstrained_complete_pools']}/160 to {qw['guided_complete_pools']}/160. "
       f"On their respective common subsets of {qe['common_complete_pools']} ESCI and {qw['common_complete_pools']} WANDS pools, unconstrained versus guided F1 is {eu['f1']:.3f} versus {eg['f1']:.3f} and {wu['f1']:.3f} versus {wg['f1']:.3f}; ratio MAE is {eu['ratio_mae']:.3f} versus {eg['ratio_mae']:.3f} and {wu['ratio_mae']:.3f} versus {wg['ratio_mae']:.3f}. "
       f"The query-bootstrap intervals for guided-minus-unconstrained F1 are $[{ec[0]:.3f},{ec[1]:.3f}]$ and $[{wc[0]:.3f},{wc[1]:.3f}]$, respectively. ")
 se,sw=gs['esci'],gs['wands']
 text+=(f"For SmolLM2, unconstrained versus guided completion is {se['unconstrained_complete_pools']}/250 versus {se['guided_complete_pools']}/250 on ESCI and {sw['unconstrained_complete_pools']}/160 versus {sw['guided_complete_pools']}/160 on WANDS. ")
 if se['guided_complete_pools'] and sw['guided_complete_pools']:
  a=se['quality_own_complete']['guided'];b=sw['quality_own_complete']['guided']
  text+=(f"On the guided plan's own completed pools, F1 is {a['f1']:.3f}, balanced accuracy {a['balanced_accuracy']:.3f}, and ratio MAE {a['ratio_mae']:.3f} on ESCI; corresponding WANDS values are {b['f1']:.3f}, {b['balanced_accuracy']:.3f}, and {b['ratio_mae']:.3f}. ")
  if 'quality_all_attempted' in se and 'quality_all_attempted' in sw:
   text+=(f"On these full samples, all-Y has ESCI ratio MAE {se['quality_all_attempted']['always_Y']['ratio_mae']:.3f}, and all-N has WANDS ratio MAE {sw['quality_all_attempted']['always_N']['ratio_mae']:.3f}. ")
 if not se['common_complete_pools'] and not sw['common_complete_pools']:
  text+='There are no common completed SmolLM2 pools for a paired quality comparison. '
 else:
  text+=(f"Its common completed subsets contain {se['common_complete_pools']} ESCI and {sw['common_complete_pools']} WANDS pools. ")
 return text+'The negative control confirms that the V1 constraint is active, but syntactic completion does not validate the predicted labels. These exploratory comparisons do not isolate model architecture or establish performance on new pools.'
def order_results():
 qe,qw,me,mw=oq['esci'],oq['wands'],om['esci'],om['wands']
 assert qe['common_complete_pools']==me['common_complete_pools']==qe['attempted_pools']==me['attempted_pools']
 assert qw['common_complete_pools']==mw['common_complete_pools']==qw['attempted_pools']==mw['attempted_pools']
 ac,aw=qe['agreement_only_diagnostic'],qw['agreement_only_diagnostic']
 return fr'''Both models complete all {qe['common_complete_pools']} ESCI and {qw['common_complete_pools']} WANDS pools under both orders. Qwen changes {qe['changed_candidate_count']}/{qe['common_candidate_count']} ESCI and {qw['changed_candidate_count']}/{qw['common_candidate_count']} WANDS labels; SmolLM2 changes {me['changed_candidate_count']}/{me['common_candidate_count']} and {mw['changed_candidate_count']}/{mw['common_candidate_count']} (Fig.~\ref{{fig:order_stability}}). Despite label changes, {qe['pools_changed_labels_same_ratio']} ESCI and {qw['pools_changed_labels_same_ratio']} WANDS Qwen pools retain their ratio. Qwen F1 changes from {qe['original_quality_common']['f1']:.3f} to {qe['reversed_quality_common']['f1']:.3f} on ESCI and {qw['original_quality_common']['f1']:.3f} to {qw['reversed_quality_common']['f1']:.3f} on WANDS; aggregate quality can conceal candidate instability.

Post hoc, retaining only labels where both orders agree leaves {ac['accepted_reference_disagreements']}/{ac['accepted_candidates']} ESCI and {aw['accepted_reference_disagreements']}/{aw['accepted_candidates']} WANDS Qwen labels inconsistent with the reference. The $k=0$ completion interval contains the reference ratio in {ac['reference_ratio_inside_k0_count']}/{ac['attempted_pools']} and {aw['reference_ratio_inside_k0_count']}/{aw['attempted_pools']} pools; agreement does not justify zero label error, and inclusion may reflect cancellation. The initial reversal changes positions and IDs together.'''
def order_control_rows():
 rows=[]
 for name,data,prior in [('Qwen / ESCI',ocq['esci'],oq['esci']),('Qwen / WANDS',ocq['wands'],oq['wands']),('SmolLM2 / ESCI',ocm['esci'],om['esci']),('SmolLM2 / WANDS',ocm['wands'],om['wands'])]:
  assert data['common_complete_pools']>0
  vals=[]
  for key in ['identical_prompt_replay','reverse_stable_ids','reassign_ids_on_reversal','reverse_reassigned_ids']:
   x=data['comparisons'][key];lo,hi=x['candidate_label_change_fraction_ci']
   if key=='reverse_reassigned_ids' and data['common_query_ids']==prior['common_query_ids']:
    assert x['changed_candidate_count']==prior['changed_candidate_count']
    lo,hi=prior['paired_bootstrap']['candidate_label_change_fraction_ci']
   vals.append(f"{100*x['candidate_label_change_fraction']:.1f} [{100*lo:.1f},{100*hi:.1f}]")
  rows.append(f"{name} ({data['common_complete_pools']}/{data['attempted_pools']})"+' & '+' & '.join(vals)+' '+chr(92)*2)
 return '\n'.join(rows)
def order_control_results():
 qe,qw,me,mw=ocq['esci'],ocq['wands'],ocm['esci'],ocm['wands']
 def p(row,key):return 100*row['comparisons'][key]['candidate_label_change_fraction']
 def floor(row,key):
  x=row['comparisons'][key];return ((x['changed_candidate_count']+1)//2)/x['candidate_count']
 assert all(x['comparisons']['identical_prompt_replay']['changed_candidate_count']==0 for x in (qe,qw,me,mw))
 return fr'''Post hoc, identical-prompt replay changes no labels for either model or dataset. Stable-ID reversal changes {p(qe,'reverse_stable_ids'):.1f}\% and {p(qw,'reverse_stable_ids'):.1f}\% for Qwen, and {p(me,'reverse_stable_ids'):.1f}\% and {p(mw,'reverse_stable_ids'):.1f}\% for SmolLM2 on ESCI and WANDS (Table~\ref{{tab:order_controls}}). For a fixed reference, Eq.~\eqref{{eq:stability}} puts the worse Qwen pooled candidate error at least {floor(qe,'reverse_stable_ids'):.3f} on ESCI and {floor(qw,'reverse_stable_ids'):.3f} on WANDS; this does not bound ratio error. Fixed sequence and one replay cannot exclude numerical execution or prompt-context effects.'''
def holdout_method():
 return fr'''After inspecting the earlier order results, we freeze a new protocol and sample {holdout['sample_size']} WANDS queries from {holdout['eligible_query_ids_after_product_exclusion']} eligible IDs after excluding prior normalized query strings and the {holdout['prior_wands_sampled_product_ids_excluded']:,} product IDs sampled previously in WANDS. We uniformly sample 32 nonconflicting judgments per query without grade stratification: {holdout['grade_counts']['Exact']:,} Exact, {holdout['grade_counts']['Partial']:,} Partial, and {holdout['grade_counts']['Irrelevant']:,} Irrelevant. The {holdout['candidate_count']:,} judgments involve {holdout['unique_sampled_product_ids']:,} distinct products; products can recur across holdout queries. This is a disjoint sample from the same public dataset, not a new source or proof of pretraining exclusion.

For each model we generate a fresh specification or fallback and run original order, an identical replay, and stable-ID reversal on the same pools using V1. The earlier WANDS specifications used V0, so between-sample differences may reflect both sample and specification generation. The single primary contrast is Qwen replay versus stable-ID reversal: before sampling or inference, we set a 10\% practical threshold for the lower endpoint of a 95\% query-bootstrap interval. SmolLM2 is secondary. After sampling but before inspecting outcomes, we also fixed a secondary crossed query--product resampling sensitivity \cite{{pigeonhole}}; neither interval guarantees population coverage under this sampling design.'''
def holdout_table_rows():
 rows=[]
 for name,x in [('Qwen',hq),('SmolLM2',hs)]:
  assert x['common_complete_pools']>0
  r=x['comparisons']['identical_prompt_replay'];v=x['comparisons']['reverse_stable_ids'];q=x['quality_common']['original'];lo,hi=v['candidate_label_change_fraction_ci']
  rows.append(f"{name} ({x['common_complete_pools']}/{x['attempted_pools']}) & {100*r['candidate_label_change_fraction']:.1f} & {100*v['candidate_label_change_fraction']:.1f} [{100*lo:.1f},{100*hi:.1f}] & {v['mean_absolute_pool_ratio_change']:.3f} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} "+chr(92)*2)
 return '\n'.join(rows)
def holdout_results():
 q=hq['comparisons']['reverse_stable_ids'];s=hs['comparisons']['reverse_stable_ids'];qr=hq['comparisons']['identical_prompt_replay'];sr=hs['comparisons']['identical_prompt_replay']
 lo,hi=q['candidate_label_change_fraction_ci'];twlo,twhi=hq2['query_product_ci'];stwlo,stwhi=hs2['query_product_ci'];oq=hq['quality_common']['original'];lex=hq['quality_common']['lexical_all_terms'];an=hq['quality_common']['always_N']
 criterion='exceeds' if hq['primary_criterion_met'] else 'does not exceed'
 assert all(v==hq['attempted_pools'] for v in hq['completion'].values()) and all(v==hs['attempted_pools'] for v in hs['completion'].values())
 return fr'''Qwen completes all {hq['attempted_pools']} pools in each condition, with {hq['specification_fallback_pools']} specification fallbacks (Table~\ref{{tab:holdout}}). Identical replay changes {qr['changed_candidate_count']}/{qr['candidate_count']} labels; stable-ID reversal changes {q['changed_candidate_count']}/{q['candidate_count']} ({100*q['candidate_label_change_fraction']:.1f}\%, 95\% query-bootstrap [{100*lo:.1f}\%,{100*hi:.1f}\%]). Its lower endpoint {criterion} the pre-specified 10\% threshold. The crossed query--product sensitivity interval [{100*twlo:.1f}\%,{100*twhi:.1f}\%] is not exact coverage. For a fixed reference, Eq.~\eqref{{eq:stability}} gives the worse candidate-error rate a floor of {((q['changed_candidate_count']+1)//2)/q['candidate_count']:.3f}. On these equal-sized pools, the positive-count gaps sum to {rf['positive_count_gap_total']}; Eq.~\eqref{{eq:ratio-floor}} gives a worse macro-MAE floor of {rf['sharp_worse_macro_mae_floor_float']:.4f}.

Three further post-hoc seeded permutations on these exposed pools change {min(random_qrange):.1f}--{max(random_qrange):.1f}\% of Qwen and {min(random_srange):.1f}--{max(random_srange):.1f}\% of SmolLM2 parsed labels; two identical-prompt replays per model change none. Fixed sequence and prompt context limit attribution.

Qwen's original-order F1 and ratio MAE are {oq['f1']:.3f} and {oq['ratio_mae']:.3f}; all-query-words obtains {lex['f1']:.3f} and {lex['ratio_mae']:.3f}, and all-N has MAE {an['ratio_mae']:.3f}. SmolLM2 completes all {hs['attempted_pools']} pools in each condition, with {hs['specification_fallback_pools']} specification fallback{'s' if hs['specification_fallback_pools'] != 1 else ''}; replay changes {sr['changed_candidate_count']}/{sr['candidate_count']} labels and stable-ID reversal changes {s['changed_candidate_count']}/{s['candidate_count']} ({100*s['candidate_label_change_fraction']:.1f}\%). Its crossed sensitivity interval is [{100*stwlo:.1f}\%,{100*stwhi:.1f}\%]. These are sampled relevance judgments, not purchase outcomes; differences in prompts, models, executions and pools limit generalization.

Post hoc, on {wr['attempted_queries']} exposed pools with saved specifications, a same-GPU local-copy isolated run changes {wr['comparisons_to_prior']['original']['changed_candidates']}/{wr['comparisons_to_prior']['original']['comparable_candidates']} original-order labels from the earlier run; its identical replay changes {wr['current_replay_vs_original']['changed_candidates']}/{wr['current_replay_vs_original']['comparable_candidates']}. Runtime and batch size differ.'''
def keyed_output_rows():
 rows=[]
 for name,x,original in [('Qwen',kq,hq),('SmolLM2',ks,hs)]:
  assert x['common_complete_pools']>0
  co=original['comparisons']['reverse_stable_ids']; replay=x['comparisons']['keyed_identical_replay']; rev=x['comparisons']['keyed_stable_reverse']; quality=x['quality_common']['keyed_original']
  assert x['common_query_ids']==original['common_query_ids']
  assert co['changed_candidate_count']==x['comparisons']['compact_instructed_list_order']['changed_candidate_count']
  def with_ci(z):
   lo,hi=z['candidate_label_change_fraction_ci']
   return f"{100*z['candidate_label_change_fraction']:.1f} [{100*lo:.1f},{100*hi:.1f}]"
  rows.append(f"{name} ({x['common_complete_pools']}/{x['attempted_pools']}) & {with_ci(co)} & {100*replay['candidate_label_change_fraction']:.1f} & {with_ci(rev)} & {quality['f1']:.3f} & {quality['ratio_mae']:.3f} "+chr(92)*2)
 return '\n'.join(rows)
def keyed_output_results():
 qr=kq['comparisons']['keyed_stable_reverse']; sr=ks['comparisons']['keyed_stable_reverse']
 qre=kq['comparisons']['keyed_identical_replay']; sre=ks['comparisons']['keyed_identical_replay']
 qa=kq['comparisons']['compact_counterfactual_id_order']; sa=ks['comparisons']['compact_counterfactual_id_order']
 qf=kq['quality_common']['keyed_original']; sf=ks['quality_common']['keyed_original']
 return fr'''Both models use the same {kq['common_complete_pools']} previously exposed pools for all compact and keyed conditions (Table~\ref{{tab:keyed}}). Keyed Qwen's identical replay changes {qre['changed_candidate_count']}/{qre['candidate_count']} labels; stable-ID reversal changes {qr['changed_candidate_count']}/{qr['candidate_count']} ({100*qr['candidate_label_change_fraction']:.1f}\%). The corresponding SmolLM2 counts are {sre['changed_candidate_count']}/{sre['candidate_count']} and {sr['changed_candidate_count']}/{sr['candidate_count']} ({100*sr['candidate_label_change_fraction']:.1f}\%). Original-order keyed F1 and ratio MAE are {qf['f1']:.3f} and {qf['ratio_mae']:.3f} for Qwen, and {sf['f1']:.3f} and {sf['ratio_mae']:.3f} for SmolLM2.

The original compact prompt required labels in listed order. If its reversed strings were instead decoded in numeric-ID order, the apparent Qwen and SmolLM2 disagreement rates would be {100*qa['candidate_label_change_fraction']:.1f}\% and {100*sa['candidate_label_change_fraction']:.1f}\%, respectively. This counterfactual contradicts the prompt and does not reveal the model's intended association. Keyed decoding fixes parser associations, not semantic correctness; the changed prompt and grammar prevent attributing the contrast to format or ordering alone. All results concern parsed system outputs, not the model's latent decisions.'''
def fallback_method():
 return fr'''After the preceding analyses, but before selecting or inferring this sample, we fixed a further Qwen evaluation on {fvmeta['sample_size']} of {fvmeta['eligible_query_ids_after_product_exclusion']} eligible WANDS queries. We exclude prior query strings and {fvmeta['prior_wands_sampled_product_ids_excluded']:,} products sampled across either earlier WANDS study, then sample 32 judged products per query without grade stratification. The {fvmeta['candidate_count']:,} judgments contain {fvmeta['grade_counts']['Exact']:,} Exact, {fvmeta['grade_counts']['Partial']:,} Partial and {fvmeta['grade_counts']['Irrelevant']:,} Irrelevant labels. This is another sample from the same source, not an independent dataset; products can recur across its queries and pretraining exposure is unknown.

Each query receives one fresh specification or direct fallback; we run keyed original, identical replay, stable-ID reversal, and pointwise judgment for all candidates. We compute the agreement-fallback and lexical-fallback plans from these saved outputs. The pre-specified primary contrast is pointwise-fallback minus canonical-keyed query-macro ratio MAE, with a paired query-bootstrap interval; its upper endpoint below zero is a sample-specific criterion. The crossed query--product resampling sensitivity was fixed after selection and inference began but before inspecting outcomes, and reweights within-query candidate pools \cite{{pigeonhole}}. Both analyses retain the relevance-proxy limitations.'''
def fallback_rows():
 n=fva['attempted_pools']; spec=fvs['specifications']; start=fvs['modes']['original']; reverse=fvs['modes']['stable_reverse']; point=fvs['modes']['pointwise']
 spec_calls=spec['initial_calls']+spec['retry_calls']; spec_input=spec['logical_tokens']['input']
 def calls(x):return x['initial_calls']+x['retry_calls']
 values=[('96: Canonical keyed','original',spec_calls+calls(start),spec_input+start['logical_tokens']['input']),
         ('96: Pointwise fallback (virtual)','pointwise_fallback',fva['disagreement']['virtual_total_calls_with_spec_and_two_keyed_passes'],fva['disagreement']['virtual_logical_tokens']['input']),
         ('96: Lexical fallback (virtual)','lexical_fallback',spec_calls+calls(start)+calls(reverse),spec_input+start['logical_tokens']['input']+reverse['logical_tokens']['input']),
         ('96: All pointwise','pointwise',spec_calls+calls(point),spec_input+point['logical_tokens']['input']),
         ('96: All query words','lexical',0,0)]
 rows=[]
 for label,key,ncalls,ntokens in values:
  q=fva['quality_common'][key]
  rows.append(f"{label} & {fva['completion'][key]}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & {ncalls:,} & {ntokens:,} "+chr(92)*2)
 q=wsl['samples']['fallback96']['quality']['supervised']
 rows.append(f"96: Supervised, threshold 0.5 & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & 0 & 0 "+chr(92)*2)
 q=wcv['samples']['fallback96']['quality']['calibrated']
 rows.append(f"96: Supervised, CV threshold & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & 0 & 0 "+chr(92)*2)
 q=wen['samples']['fallback96']['quality']['embedding']
 rows.append(f"96: Calibrated embedding & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & -- & -- "+chr(92)*2)
 spec2=svs['specifications']; start2=svs['modes']['original']; point2=svs['modes']['pointwise']; base_calls=spec2['initial_calls']+spec2['retry_calls']; base_input=spec2['logical_tokens']['input']
 values2=[('52: Canonical keyed','original',base_calls+calls(start2),base_input+start2['logical_tokens']['input']),
          ('52: Actual selective fallback','actual_fallback',sva['selective']['actual_total_calls_with_spec_and_two_keyed_passes'],sva['selective']['actual_logical_tokens']['input']),
          ('52: All pointwise','pointwise',base_calls+calls(point2),base_input+point2['logical_tokens']['input']),
          ('52: All query words','lexical',0,0)]
 for label,key,ncalls,ntokens in values2:
  q=sva['quality_common'][key]
  rows.append(f"{label} & {sva['completion'][key]}/{sva['attempted_pools']} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & {ncalls:,} & {ntokens:,} "+chr(92)*2)
 q=wsl['samples']['selective52']['quality']['supervised'];n=sva['attempted_pools']
 rows.append(f"52: Supervised, threshold 0.5 & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & 0 & 0 "+chr(92)*2)
 q=wcv['samples']['selective52']['quality']['calibrated']
 rows.append(f"52: Supervised, CV threshold & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & 0 & 0 "+chr(92)*2)
 q=wen['samples']['selective52']['quality']['embedding']
 rows.append(f"52: Calibrated embedding & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & -- & -- "+chr(92)*2)
 for label,key,calls,tokens in [
   ('52: LOTUS filter','lotus_base',lc['logical_requests']['base_main'],lc['usage']['base_main']['physical']['prompt_tokens']),
   ('52: LOTUS cascade','lotus_cascade',sum(lc['logical_requests'][k] for k in ('cascade_helper','cascade_calibration_main','cascade_routed_main')),
    lc['usage']['cascade_helper']['physical']['prompt_tokens']+lc['usage']['cascade_main']['physical']['prompt_tokens'])]:
  q=lc['quality'][key]
  rows.append(f"{label} & {n}/{n} & {q['f1']:.3f} & {q['balanced_accuracy']:.3f} & {q['ratio_mae']:.3f} & {calls:,} & {tokens:,} "+chr(92)*2)
 return '\n'.join(rows)
def fallback_results():
 c=fva['comparisons']['pointwise_fallback_minus_original']; lo,hi=c['ratio_mae_difference_ci']
 cross=fv2['pointwise_fallback_minus_original_ratio_mae_query_product_ci']
 d=fva['disagreement']; orig=fva['quality_common']['original']; fb=fva['quality_common']['pointwise_fallback']; lexical=fva['quality_common']['lexical_fallback']
 c0=wc96['groups']['all_negative'];cm=wc96['groups']['mixed'];cl,ch=cm['unadjusted_posthoc_query_bootstrap_95']
 return fr'''All four conditions complete {fva['common_complete_pools']}/{fva['attempted_pools']} pools; {fva['specification_fallback_pools']} specifications fall back to direct judgments. Keyed/pointwise-fallback MAEs are {orig['ratio_mae']:.3f}/{fb['ratio_mae']:.3f} (Table~\ref{{tab:fallback}}); the pre-specified gap is {c['ratio_mae_difference']:.4f}, with 95\% query-bootstrap interval $[{lo:.4f},{hi:.4f}]$. The secondary query--product range $[{cross[0]:.4f},{cross[1]:.4f}]$ reweights pools and is not a fixed-pool interval. Lexical fallback attains {lexical['ratio_mae']:.3f}. Post hoc, {c0['pools']} all-negative pools contribute {c0['weighted_contribution_to_overall']:+.4f} of the total difference; {cm['pools']} mixed pools have difference {cm['fallback_minus_keyed']:+.4f}, with unadjusted 95\% query interval $[{cl:.4f},{ch:.4f}]$.

The keyed orders disagree on {d['candidates']}/{fva['candidate_count']:,} candidates; replay changes {fva['keyed_replay_changed_candidates']} keyed labels and {d['pointwise_replay_changed']}/{d['valid_pointwise_replays']} pointwise replays on disagreements. Cached-vector symmetry does not imply deterministic inference. The fallback represents {d['virtual_total_calls_with_spec_and_two_keyed_passes']:,} logical calls and {d['virtual_logical_tokens']['input']:,} input tokens derived from traces; all pointwise calls were executed, so these do not measure selective-scheduler latency. Results are conditional on sampled pools, one model and prompts.'''
def selective_method():
 return fr'''After the 96-query outcome, we fixed a follow-up on all {svmeta['sample_size']} remaining eligible WANDS queries, excluding prior query strings and {svmeta['prior_wands_sampled_product_ids_excluded']:,} sampled product IDs. The 32 judged candidates per query contain {svmeta['grade_counts']['Exact']:,} Exact, {svmeta['grade_counts']['Partial']:,} Partial and {svmeta['grade_counts']['Irrelevant']:,} Irrelevant labels. The scheduler requests pointwise judgments only for valid keyed disagreements; keyed replay and full pointwise calls are diagnostics. We specified the paired query-macro MAE contrast before product selection and inference. A crossed query--product sensitivity, fixed after selection and inference began but before outcome inspection, reweights within-query candidates. After the Qwen outcome, a secondary SmolLM2 run uses the same queries with fresh specifications; it is not a new sample or online benchmark.'''
def selective_results():
 c=sva['comparisons']['actual_fallback_minus_original'];lo,hi=c['ratio_mae_difference_ci'];cross=sv2['actual_fallback_minus_original_ratio_mae_query_product_ci']
 orig=sva['quality_common']['original'];actual=sva['quality_common']['actual_fallback'];d=sva['selective']
 cross_note='includes zero' if cross[0]<=0<=cross[1] else 'does not include zero'
 smc=sma['comparisons']['actual_fallback_minus_original'];smlo,smhi=smc['ratio_mae_difference_ci']
 smorig=sma['quality_common']['original'];smactual=sma['quality_common']['actual_fallback']
 return fr'''On all {sva['common_complete_pools']} common-complete queries, selective fallback has ratio MAE {actual['ratio_mae']:.3f} versus {orig['ratio_mae']:.3f} for keyed output (Table~\ref{{tab:fallback}}). The pre-specified difference is {c['ratio_mae_difference']:.4f}, with 95\% query-bootstrap interval $[{lo:.4f},{hi:.4f}]$. The crossed interval $[{cross[0]:.4f},{cross[1]:.4f}]$ reweights pools and {cross_note}; neither interval guarantees population coverage.

The scheduler sends {d['actual_initial_calls']} pointwise requests and {d['actual_total_calls_with_spec_and_two_keyed_passes']:,} calls overall; its {d['offline_batch_wall_seconds']:.2f}-second offline batch is not online latency. Keyed replay changes {sva['keyed_replay_changed_candidates']} labels; selective and later pointwise outputs differ on {d['changed_selective_vs_full_pointwise']}/{d['matched_selective_and_full_pointwise']} matched requests. Full-pointwise versus selective F1 is {sva['quality_common']['pointwise']['f1']:.3f} versus {actual['f1']:.3f}.

SmolLM2 completes all {sma['common_complete_pools']} queries; selective and keyed MAEs are {smactual['ratio_mae']:.3f} and {smorig['ratio_mae']:.3f}, with difference {smc['ratio_mae_difference']:+.4f} and query interval $[{smlo:.4f},{smhi:.4f}]$. Fallback F1 is {smactual['f1']:.3f}; full pointwise predicts N for all {sma['candidate_count']:,} candidates. Model and prompt changes confound comparison.'''
def esci_selective_holdout_method():
 return fr'''\textbf{{New ESCI pools.}} After earlier outcomes, we fixed a follow-up before sampling or inference and selected {ehmeta['selected_queries']} ESCI test queries from {ehmeta['eligible_query_ids']:,} eligible IDs, excluding prior ESCI and training query strings and product IDs in the original pools or supervised training. The {ehmeta['candidates']:,} judged candidates have {ehmeta['minimum_pool_size']}--{ehmeta['maximum_pool_size']} per pool. A post-hoc audit finds {ehover['new_candidate_rows_with_any_exact_text_overlap']} serialized product texts shared with prior samples or training, across {ehover['new_queries_with_any_exact_text_overlap']} queries. We pre-specified Qwen's selective-minus-keyed query-macro MAE contrast, with SmolLM2 and the {50000:,}-label supervised reference secondary.'''
def esci_selective_holdout_results():
 q=ehq['quality_common'];s=ehs['quality_common'];qc=ehq['comparisons']['actual_fallback_minus_original'];sc=ehs['comparisons']['actual_fallback_minus_original']
 qlo,qhi=qc['ratio_mae_difference_ci'];slo,shi=sc['ratio_mae_difference_ci'];supervised=ehref['models']['qwen']['supervised_quality_common']
 qp=ehqr['modes']['specifications'];qf=ehqr['modes']['pointwise'];fullcalls=qp['initial_calls']+qp['retry_calls']+qf['initial_calls']+qf['retry_calls']
 fulltokens=qp['logical_tokens']['input']+qf['logical_tokens']['input'];sel=ehq['selective']
 qpost=ehsens['models']['qwen']['selective_minus_keyed']['ratio_mae_difference'];spost=ehsens['models']['smollm']['selective_minus_keyed']['ratio_mae_difference']
 return fr'''\textbf{{New ESCI results.}} Both models complete all {ehq['common_complete_pools']} pools. Qwen's keyed, selective and full-pointwise ratio MAEs are {q['original']['ratio_mae']:.3f}, {q['actual_fallback']['ratio_mae']:.3f} and {q['pointwise']['ratio_mae']:.3f}; selective minus keyed is {qc['ratio_mae_difference']:+.3f}, with 95\% query-bootstrap interval $[{qlo:.3f},{qhi:.3f}]$. All-Y and the supervised reference achieve {q['always_Y']['ratio_mae']:.3f} and {supervised['ratio_mae']:.3f}; the latter uses training labels. Qwen's selective plan makes {sel['actual_total_calls_with_spec_and_two_keyed_passes']:,} calls versus {fullcalls:,} for full pointwise, but uses {sel['actual_logical_tokens']['input']/1e6:.2f}M versus {fulltokens/1e6:.2f}M logical input tokens. SmolLM2's keyed and selective MAEs are {s['original']['ratio_mae']:.3f} and {s['actual_fallback']['ratio_mae']:.3f} (difference {sc['ratio_mae_difference']:+.3f}, interval $[{slo:.3f},{shi:.3f}]$). Post hoc, excluding the {ehover['new_queries_with_any_exact_text_overlap']} text-overlap queries yields differences {qpost:+.3f} and {spost:+.3f}. Post hoc, rescoring fixed Qwen outputs with only E positive changes selective minus keyed to {labelmap['models']['qwen']['selective_minus_keyed']['E_only']['mean']:+.3f} $[{labelmap['models']['qwen']['selective_minus_keyed']['E_only']['ci95'][0]:.3f},{labelmap['models']['qwen']['selective_minus_keyed']['E_only']['ci95'][1]:.3f}]$; the prompt still treats substitutes as positive. For keyed versus reversed outputs, the unequal-pool floor is {varfloor['models']['qwen']['floor']['decimal']:.3f} for Qwen and {varfloor['models']['smollm']['floor']['decimal']:.3f} for SmolLM2. These selected pools do not establish population coverage or a speedup.'''
def wands_supervised_method():
 return fr'''After observing WANDS outcomes, we train a feature-limited classifier on {wsl['training']['candidates']:,} labeled candidates from {wsl['training']['pools']} earlier queries; normalized test queries, sampled product IDs and exact texts are disjoint from training. Five query folds exclude validation products and exact texts from training and select thresholds for pooled candidate F1 and query-macro ratio MAE; thresholds transfer to a full-training model. We also average its scores as a ratio-only reference. Selection-set metrics are not independent estimates; scores are not calibrated probabilities.'''

def wands_supervised_results():
 h=wsl['samples']['holdout160']['quality'];cv=wcv['samples']['holdout160']['quality']['calibrated'];x=wrb['samples'];t=wrb['threshold_selection']['threshold']
 rt='/'.join(f"{x[n]['ratio_results']['ratio_threshold']['ratio_mae']:.3f}" for n in ('holdout160','fallback96','selective52'))
 qm='/'.join(f"{x[n]['binary_metrics']['qwen']['ratio_mae']:.3f}" for n in ('holdout160','fallback96','selective52'))
 sm='/'.join(f"{x[n]['ratio_results']['score_mean']['ratio_mae']:.3f}" for n in ('holdout160','fallback96','selective52'))
 lo,hi=x['selective52']['ratio_results']['ratio_threshold']['query_bootstrap_95']
 plo,phi=wrps['samples']['selective52']['ranges']['hard_minus_qwen']
 return fr'''On the exposed 160-query holdout, supervised F1/MAE is {h['supervised']['f1']:.3f}/{h['supervised']['ratio_mae']:.3f}, keyed Qwen {h['qwen']['f1']:.3f}/{h['qwen']['ratio_mae']:.3f} and all-terms {h['lexical']['f1']:.3f}/{h['lexical']['ratio_mae']:.3f}. F1 threshold {wcv['calibration']['threshold']:.4f} yields {cv['f1']:.3f}/{cv['ratio_mae']:.3f}; F1 and MAE also rise on 96/52 queries (Table~\ref{{tab:fallback}}). Ratio-MAE threshold {t:.3f} yields {rt} on 160/96/52 queries versus Qwen's {qm}; mean scores yield {sm}. On 52, unadjusted 95\% threshold-minus-Qwen ranges are $[{lo:.3f},{hi:.3f}]$ (query bootstrap) and $[{plo:.3f},{phi:.3f}]$ (product-reweighted pools).'''

def wands_embedding_method():
 return fr'''Qwen3-Embedding-0.6B \cite{{qwen_embed}} encodes each query and truncated product text. On {wen['training']['candidates']:,} labeled candidates, we select an F1 threshold; after observing test results, we select a query-macro ratio-MAE threshold. Neither uses test labels or task fine-tuning. Train/test have disjoint normalized queries, sampled product IDs and exact serialized texts.'''
def wands_embedding_results():
 h=wen['samples']['holdout160']['quality']['embedding'];x=wer['samples'];names=('holdout160','fallback96','selective52')
 rm='/'.join(f"{x[n]['metrics']['ratio_threshold']['ratio_mae']:.3f}" for n in names)
 fm='/'.join(f"{x[n]['metrics']['f1_threshold']['ratio_mae']:.3f}" for n in names)
 lo,hi=x['selective52']['query_bootstrap']['ratio_minus_qwen']['interval_95'];plo,phi=x['selective52']['product_reweighted_ranges']['ratio_minus_qwen']
 return fr'''F1-threshold embedding has F1/MAE {h['f1']:.3f}/{h['ratio_mae']:.3f} on 160 queries (Table~\ref{{tab:fallback}}: 96/52). The ratio threshold {wer['threshold_selection']['threshold']:.3f} yields MAE {rm} on 160/96/52, versus {fm} for the F1 threshold. On 52, F1 falls from {x['selective52']['metrics']['f1_threshold']['f1']:.3f} to {x['selective52']['metrics']['ratio_threshold']['f1']:.3f}; ratio-threshold minus Qwen has 95\% query-bootstrap interval $[{lo:.3f},{hi:.3f}]$ and product-reweighted range $[{plo:.3f},{phi:.3f}]$. Comparisons are unadjusted and post hoc.'''
def service_rows():
 rows=[]
 for plan,label in [('pointwise','Pointwise'),('keyed','Keyed'),('selective','Selective')]:
  for c in (1,4):
   x=sorted((y for y in svc['conditions'] if y['plan']==plan and y['concurrency']==c),key=lambda y:y['repeat'])
   assert len(x)==2 and all(y['active_latency_p50_seconds'] is not None and y['active_latency_p95_seconds'] is not None for y in x)
   def spread(key):
    values=sorted(y[key] for y in x)
    return f"{values[0]:.2f}--{values[1]:.2f}"
   completed=','.join(str(y['complete_pools']) for y in x)
   rows.append(f"{label} ({c}) & {spread('active_latency_p50_seconds')} & {spread('active_latency_p95_seconds')} & {completed} "+chr(92)*2)
 return '\n'.join(rows)
def service_method():
 first=svc['conditions'][0]
 assert (first['plan'],first['concurrency'],first['complete_pools'],first['http_failures'])==('pointwise',4,49,3)
 return r'''After earlier outcomes, we test pointwise, keyed and selective pipelines on the same 52 WANDS pools via a one-GPU vLLM HTTP server, with 16 sequences and no prefix caching. Plans use fresh specifications and one or four active queries, repeated in reverse order. Active-query time includes local HTTP, queueing, generation and parsing but excludes startup, warmup and admission wait. Three read errors caused abstentions in the first condition; after preserving its records, we resumed the schedule with a new client pool. This amendment was post hoc; errors were not retried or imputed.'''
def service_results():
 complete=sum(x['complete_pools'] for x in svc['conditions'])
 fail=sum(x['http_failures'] for x in svc['conditions'])
 return fr'''The local workload completes {complete}/624 pipelines with {fail} HTTP or transport failures (Table~\ref{{tab:local_service}}); percentiles exclude failures. This warm local server omits admission wait, Internet latency and production traffic. Fresh specifications confound timing and quality contrasts.'''
p7e=json.loads((base/'02_reproducibility/POSTHOC_7B_ESCI.json').read_text())
p7w=json.loads((base/'02_reproducibility/POSTHOC_7B_WANDS.json').read_text())
p7t=json.loads((base/'02_reproducibility/POSTHOC_7B_TREC.json').read_text())
p7eg=json.loads((base/'02_reproducibility/POSTHOC_7B_ESCI_GUIDED.json').read_text())
p7wg=json.loads((base/'02_reproducibility/POSTHOC_7B_WANDS_GUIDED.json').read_text())
p7egstats=json.loads((base/'02_reproducibility/POSTHOC_7B_ESCI_GUIDED_STATS.json').read_text())
p7wgstats=json.loads((base/'02_reproducibility/POSTHOC_7B_WANDS_GUIDED_STATS.json').read_text())
p7prod=json.loads((base/'02_reproducibility/posthoc_product_weight/product_weight_summary.json').read_text())['datasets']
assert p7prod['ESCI']['pools']==p7eg['common_pools'] and p7prod['WANDS']['pools']==p7wg['common_pools']
assert abs(p7prod['ESCI']['difference_new_minus_old']-p7eg['comparison']['difference_new_minus_old'])<1e-12
assert abs(p7prod['WANDS']['difference_new_minus_old']-p7wg['comparison']['difference_new_minus_old'])<1e-12
assert p7eg['common_pools']>0 and p7wg['common_pools']>0
assert p7e['common_complete_pools']==p7e['attempted_pools']==p7e['specification_fallback_pools']==250
assert p7w['common_complete_pools']==p7w['attempted_pools']==p7w['specification_fallback_pools']==52
assert p7t['common_complete_topics']==p7t['topics_attempted']==50
vals={
"@@THIRD_MODEL_METHOD@@":r'''\textbf{Post-hoc third-model sensitivity.} We repeat the keyed, reversed and selective protocols on the same 250 ESCI, 52 WANDS and 50 TREC pools with Qwen2.5-7B-Instruct \cite{qwen25}. Family, tokenizer, chat template and execution differ, precluding isolated size comparisons. A further post-hoc product run constrains specifications to a JSON schema, retaining the strict parser, prompts, retry and fallback rules. This compares whole pipelines; TREC uses the official narrative.''',
"@@THIRD_MODEL_RESULTS@@":fr'''\subsection{{Third-Model Sensitivity}}
All product specifications in the unconstrained 7B runs are fenced and fail strict validation after retry; all {p7e['specification_fallback_pools']} ESCI and {p7w['specification_fallback_pools']} WANDS pools use direct fallback. On the {p7e['common_complete_pools']} ESCI pools, keyed and selective ratio MAEs are {p7e['quality_common']['original']['ratio_mae']:.3f} and {p7e['quality_common']['actual_fallback']['ratio_mae']:.3f}, while all-Y gives {p7e['quality_common']['always_Y']['ratio_mae']:.3f}. The corresponding WANDS MAEs are {p7w['quality_common']['original']['ratio_mae']:.3f} and {p7w['quality_common']['actual_fallback']['ratio_mae']:.3f}. On TREC, selective MAE rises from {p7t['metrics']['original']['ratio_mae']:.3f} to {p7t['metrics']['selective']['ratio_mae']:.3f}; its paired interval is $[{p7t['primary_difference']['query_bootstrap_95'][0]:.3f},{p7t['primary_difference']['query_bootstrap_95'][1]:.3f}]$. Figure~\ref{{fig:model_sensitivity}} shows unadjusted intervals and absolute errors for all three models. These exposed-sample comparisons are descriptive, not causal.''',
"@@SPEC_GUIDED_RESULTS@@":fr'''Schema guidance leaves {p7egstats['specification_fallback_pools']}/250 ESCI and {p7wgstats['specification_fallback_pools']}/52 WANDS pools on direct fallback. Selective MAE changes from {p7eg['comparison']['old_unguided_mae']:.3f} to {p7eg['comparison']['new_schema_guided_mae']:.3f} on ESCI (difference {p7eg['comparison']['difference_new_minus_old']:+.3f}, unadjusted 95\% query interval $[{p7eg['comparison']['paired_query_95_percentile_ci'][0]:.3f},{p7eg['comparison']['paired_query_95_percentile_ci'][1]:.3f}]$), and from {p7wg['comparison']['old_unguided_mae']:.3f} to {p7wg['comparison']['new_schema_guided_mae']:.3f} on WANDS ($[{p7wg['comparison']['paired_query_95_percentile_ci'][0]:.3f},{p7wg['comparison']['paired_query_95_percentile_ci'][1]:.3f}]$). Exponential product reweighting yields 95\% sensitivity ranges $[{p7prod['ESCI']['sensitivity_range_95'][0]:.3f},{p7prod['ESCI']['sensitivity_range_95'][1]:.3f}]$ and $[{p7prod['WANDS']['sensitivity_range_95'][0]:.3f},{p7prod['WANDS']['sensitivity_range_95'][1]:.3f}]$; these are not fixed-pool confidence intervals. These post-hoc pipeline comparisons reuse observed queries; valid format does not certify meaning.''',
"@@ESCI_TRUNCATION_RESULTS@@":fr'''Post hoc, on {tr['attempted_pools']} exposed pools we run guided pointwise Qwen in fixed 600-, 1,200-, 600-character replay order. The longer prefix changes {trmeta['changed_candidates']:,}/{trmeta['candidates']:,} inputs; {trmeta['source_text_over_1200_characters']:,} source texts still exceed 1,200 characters. Ratio MAE is {tr['metrics']['short']['ratio_mae']:.3f} and {tr['metrics']['long']['ratio_mae']:.3f} (long minus short {tr['primary_long_minus_short']['ratio_mae_difference']:.4f}; paired 95\% query-bootstrap $[{tr['primary_long_minus_short']['query_bootstrap_95'][0]:.4f},{tr['primary_long_minus_short']['query_bootstrap_95'][1]:.4f}]$); balanced accuracy falls from {tr['metrics']['short']['balanced_accuracy']:.3f} to {tr['metrics']['long']['balanced_accuracy']:.3f}. Replay changes {tr['changes']['replay_vs_short']['all']['changed']}/{trmeta['candidates']:,} labels. A query--product reweighting sensitivity range $[{tr['product_weight_sensitivity']['reweighted_range_95'][0]:.4f},{tr['product_weight_sensitivity']['reweighted_range_95'][1]:.4f}]$ changes the estimand. Fixed order, replay variation and the exposed sample limit attribution and generalization.''',
'@@LOTUS_CASCADE_METHOD@@':r'''On the already exposed 52 WANDS pools, we additionally compare a Qwen-only LOTUS filter with its built-in SmolLM2-helper cascade, using the same shuffled 1,664 candidates and predicate. The cascade samples 10\% of candidates with replacement for Qwen-based threshold learning; its nominal 0.8 precision and recall targets concern the reference-model procedure, not human labels. Both models share one GPU; prompts, parsing and execution differ from our plans. We retain calibration, routes, helper log probabilities and both models' outputs.''',
'@@LOTUS_CASCADE_RESULTS@@':fr'''The LOTUS cascade has F1 {lc['quality']['lotus_cascade']['f1']:.3f} versus {lc['quality']['lotus_base']['f1']:.3f} for its separately run full filter (Table~\ref{{tab:fallback}}); the paired query-bootstrap interval for the difference is $[{lc['cascade_minus_lotus_base']['f1_difference_ci'][0]:.3f},{lc['cascade_minus_lotus_base']['f1_difference_ci'][1]:.3f}]$. Ratio MAE is {lc['quality']['lotus_cascade']['ratio_mae']:.3f} versus {lc['quality']['lotus_base']['ratio_mae']:.3f}, with difference interval $[{lc['cascade_minus_lotus_base']['ratio_mae_difference_ci'][0]:.3f},{lc['cascade_minus_lotus_base']['ratio_mae_difference_ci'][1]:.3f}]$. The cascade uses {lc['logical_requests']['cascade_calibration_main']} calibration and {lc['logical_requests']['cascade_routed_main']} routed Qwen calls plus {lc['logical_requests']['cascade_helper']:,} helper calls. Its empirical precision and recall against the separate full-filter run are {lc['cascade_vs_separately_run_base']['precision']:.3f} and {lc['cascade_vs_separately_run_base']['recall']:.3f}; fresh Qwen calls and the exposed sample preclude a guarantee test. In all {lc['candidates']:,} helper outputs, LOTUS first finds both label tokens in top log probabilities at the initial generated token, which is ``Answer''; these scores are not necessarily conditioned on the answer position. Replaying proxy scores is NumPy-version-sensitive at quantile ties. The result is configuration-specific and neither establishes a causal mechanism nor generalizes to the complete optimizer.''',
'@@LOCAL_SERVICE_METHOD@@':service_method(),
'@@LOCAL_SERVICE_RESULTS@@':service_results(),
'@@LOCAL_SERVICE_ROWS@@':service_rows(),
"@@FULL_ABSTRACT@@":fr'''A language-model predicate can silently change a semantic aggregate's denominator when a batch omits candidate judgments. We define an execution contract for the fraction of a fixed product pool satisfying a query: validate judgments, retry invalid responses, and abstain if any remain unresolved. We derive sharp ratio bounds conditional on an accepted-label error budget and finite-pool candidate- and ratio-error floors for aligned outputs under a fixed binary reference. We evaluate pointwise, batched and two-stage plans on two product-relevance datasets and an adapted biomedical-relevance workload, reporting quality, completion, cost and local-service timing alongside lexical, supervised, embedding and LOTUS references. On a disjoint WANDS holdout, stable-ID reversal changes {100*hq['comparisons']['reverse_stable_ids']['candidate_label_change_fraction']:.1f}\% of Qwen's compact labels and {100*kq['comparisons']['keyed_stable_reverse']['candidate_label_change_fraction']:.1f}\% with keyed JSON, without isolating causes. On {ehmeta['selected_queries']} new ESCI pools, Qwen selective ratio MAE is {ehq['quality_common']['actual_fallback']['ratio_mae']:.3f} versus {ehq['quality_common']['original']['ratio_mae']:.3f} keyed and {ehq['quality_common']['always_Y']['ratio_mae']:.3f} for all-Y. In an unconstrained post-hoc third-model run, direct fallback is used on every product pool after specification validation fails: selective MAE is {p7e['quality_common']['actual_fallback']['ratio_mae']:.3f} on ESCI and {p7w['quality_common']['actual_fallback']['ratio_mae']:.3f} on WANDS; biomedical MAE rises from {p7t['metrics']['original']['ratio_mae']:.3f} keyed to {p7t['metrics']['selective']['ratio_mae']:.3f} selective. These relevance-proxy results do not establish purchase satisfaction or deployment-scale superiority.''',
'@@WANDS_RESULTS@@':fr'''On WANDS, the four Qwen plans jointly complete {wqa['common_complete_pools']} of 160 sampled pools. On these selected pools, pointwise candidate F1 is {wqa['methods']['pointwise']['f1']:.3f} with ratio MAE {wqa['methods']['pointwise']['mae']:.3f}; two-stage JSON obtains F1 {wqa['methods']['two_stage_json']['f1']:.3f} and MAE {wqa['methods']['two_stage_json']['mae']:.3f} (Table~\ref{{tab:wands}}). The 95\% query-bootstrap interval for two-stage minus pointwise F1 is $[{wqa['differences']['two_stage_json']['f1_ci'][0]:.3f},{wqa['differences']['two_stage_json']['f1_ci'][1]:.3f}]$; the corresponding ratio-MAE interval is $[{wqa['differences']['two_stage_json']['mae_ci'][0]:.3f},{wqa['differences']['two_stage_json']['mae_ci'][1]:.3f}]$. These are unadjusted descriptive contrasts after the ESCI analysis, and cannot isolate the specification or data-source effect. The simple all-query-words comparator attains F1 {wqa['methods']['lexical_all_terms']['f1']:.3f} and MAE {wqa['methods']['lexical_all_terms']['mae']:.3f}; all-N has MAE {wqa['methods']['always_no']['mae']:.3f}. Exact positives comprise {wqa['methods']['always_yes']['tp']}/{wqa['common_complete_pools']*32} decisions in this subset. Thus neither candidate F1 nor ratio accuracy can be interpreted without the pool's label mix.''',
'@@WANDS_COMMON@@':str(wqa['common_complete_pools']),
'@@WANDS_TABLE_ROWS@@':table(wqs,wqa,fullorder,160,[('always_yes','Always Y'),('always_no','Always N'),('lexical_all_terms','All query words')]),
'@@WANDS_SENSITIVITY@@':model_sensitivity(),
'@@GUIDED_COMPARISON@@':guided_comparison(),
'@@ORDER_RESULTS@@':order_results(),
'@@ORDER_CONTROL_RESULTS@@':order_control_results(),
'@@ORDER_CONTROL_TABLE_ROWS@@':order_control_rows(),
'@@HOLDOUT_METHOD@@':holdout_method(),
'@@HOLDOUT_RESULTS@@':holdout_results(),
'@@HOLDOUT_TABLE_ROWS@@':holdout_table_rows(),
'@@KEYED_OUTPUT_RESULTS@@':keyed_output_results(),
'@@KEYED_OUTPUT_TABLE_ROWS@@':keyed_output_rows(),
'@@FALLBACK_METHOD@@':fallback_method(),
'@@FALLBACK_RESULTS@@':fallback_results(),
'@@FALLBACK_TABLE_ROWS@@':fallback_rows(),
'@@SELECTIVE_METHOD@@':selective_method(),
'@@SELECTIVE_RESULTS@@':selective_results(),
'@@ESCI_SELECTIVE_HOLDOUT_METHOD@@':esci_selective_holdout_method(),
'@@ESCI_SELECTIVE_HOLDOUT_RESULTS@@':esci_selective_holdout_results(),
'@@WANDS_SUPERVISED_METHOD@@':wands_supervised_method(),
'@@WANDS_SUPERVISED_RESULTS@@':wands_supervised_results(),
'@@WANDS_EMBEDDING_METHOD@@':wands_embedding_method(),
'@@WANDS_EMBEDDING_RESULTS@@':wands_embedding_results(),
'@@COMPACT_ABLATION@@':f'''The original compact prompt requests no punctuation but contains the example ``YNNY.'' for four products even when a chunk has eight. After observing SmolLM2 failures, we remove this example and reuse the saved specifications and candidate order. Under the original strict parser, the revised prompt still completes {ab['esci']['corrected_complete_pools']}/250 ESCI and {ab['wands']['corrected_complete_pools']}/160 WANDS pools; final ESCI chunks consisting only of YNNY, with or without a period, fall from {ab['esci']['literal_example_final_original']} to {ab['esci']['literal_example_final_corrected']}. As a separate post-hoc analysis, a delimiter-tolerant parser accepts whitespace, commas, and one terminal period only when exactly the requested number of Y/N labels remains. Applied to revised responses, it completes {pa['esci']['cases']['removed_delimited']['complete_pools']}/250 ESCI pools and {pa['wands']['cases']['removed_delimited']['complete_pools']}/160 WANDS pools. On those selected ESCI pools, F1 is {pa['esci']['cases']['removed_delimited']['quality_on_own_complete']['f1']:.3f}, balanced accuracy {pa['esci']['cases']['removed_delimited']['quality_on_own_complete']['balanced_accuracy']:.3f}, and ratio MAE {pa['esci']['cases']['removed_delimited']['quality_on_own_complete']['ratio_mae']:.3f}. No pool is jointly complete under the original and revised delimiter-tolerant outputs. This targeted audit changes the prompt and parser and remains conditional on success; it neither certifies the recovered labels nor characterizes other prompts or models.''',
'@@BOUNDS_DIAGNOSTIC@@':fr'''On failed compact pools, $k=0$ bounds cover {fb['reference_ratio_inside_count']}/{fb['failed_pools']} ESCI and {wb['reference_ratio_inside_count']}/{wb['failed_pools']} WANDS reference ratios; {fb['accepted_label_mistakes']}/{fb['accepted_labels']} and {wb['accepted_label_mistakes']}/{wb['accepted_labels']} accepted labels disagree. Post-hoc $p/m$ MAE is {pb['esci_250']['available_only_mae']:.3f}/{pb['wands_160']['available_only_mae']:.3f}; N-filling yields {pb['esci_250']['zero_fill_mae']:.3f}/{pb['wands_160']['zero_fill_mae']:.3f}. Errors can cancel.

Post hoc, the {wcp['calibration']['order_statistic_rank']}th score (90\% nominal) among {wcp['calibration']['pools']} exposed complete WANDS pools gives thresholds {wcp['calibration']['candidate_error_threshold']} candidate errors and {wcp['calibration']['ratio_residual_threshold']} ratio-count residuals. On {wcp['test']['test96']['pools']}/{wcp['test']['test52']['pools']} later pools, error-budget intervals cover {wcp['test']['test96']['candidate_error']['interval_covered']}/{wcp['test']['test96']['pools']} and {wcp['test']['test52']['candidate_error']['interval_covered']}/{wcp['test']['test52']['pools']}, with mean widths {wcp['test']['test96']['candidate_error']['mean_width']:.3f}/{wcp['test']['test52']['candidate_error']['mean_width']:.3f}. Direct ratio intervals cover {wcp['test']['test96']['ratio_residual']['interval_covered']}/{wcp['test']['test96']['pools']} and {wcp['test']['test52']['ratio_residual']['interval_covered']}/{wcp['test']['test52']['pools']}, with widths {wcp['test']['test96']['ratio_residual']['mean_width']:.3f}/{wcp['test']['test52']['ratio_residual']['mean_width']:.3f}. Sequential exclusions and exposure preclude asserting exchangeability or future coverage.''',
'@@TREC_TRANSFER@@':fr'''After the product results, we fixed a TREC-COVID Complete transfer check \cite{{treccovid}} before sampling or inference: {tcm['included_topics']} topics each receive 32 judged documents; {tcm['sampled_grade_counts']['2']:,}/{tcm['sampled_candidates']:,} are fully relevant. Only fully relevant maps to Y; titles and abstracts are truncated to 200 and 800 characters. Two nonstandard $-1$ judgments are excluded. The official narrative replaces the generated specification.

All Qwen conditions complete {tc['common_complete_topics']}/{tc['topics_attempted']} pools. Stable-ID reversal changes {tc['stable_reverse_changes']['changed']}/{tc['stable_reverse_changes']['total']} labels and identical replay changes {tc['replay_changes']['changed']}. Keyed and selective MAEs are {tc['metrics']['original']['ratio_mae']:.4f} and {tc['metrics']['selective']['ratio_mae']:.4f}; their pre-specified difference is {tc['primary_difference']['selective_minus_original_ratio_mae']:+.4f}, with query-bootstrap interval $[{tc['primary_difference']['query_bootstrap_95'][0]:.4f},{tc['primary_difference']['query_bootstrap_95'][1]:.4f}]$. A query--document sensitivity $[{tc['crossed_sensitivity']['interval_95'][0]:.4f},{tc['crossed_sensitivity']['interval_95'][1]:.4f}]$ reweights pools.

After observing Qwen, we repeat on the same sample with SmolLM2. All conditions complete {tcsm['common_complete_topics']}/{tcsm['topics_attempted']} pools; keyed and selective MAEs are {tcsm['metrics']['original']['ratio_mae']:.4f} and {tcsm['metrics']['selective']['ratio_mae']:.4f}, with query interval $[{tcsm['primary_difference']['query_bootstrap_95'][0]:.4f},{tcsm['primary_difference']['query_bootstrap_95'][1]:.4f}]$. Models and untuned prompts differ; truncated abstracts may omit needed information. These public samples do not establish generalization or clinical utility.''',
'@@LOTUS_TEXT@@':f'''LOTUS returns decisions for all 250 pools; all 5,160 raw answers contain an expected decision token and the parser defaults on {lba['postprocessed_defaulted_candidates']} candidates. On the 207-pool intersection, its F1, balanced accuracy and ratio MAE are {lbm['f1']:.3f}, {lbm['balanced_accuracy']:.3f} and {lbm['ratio_mae']:.3f} (Table~\\ref{{tab:full}}). Paired intervals for LOTUS minus pointwise are $[{lbq['lotus_minus_pointwise_ci']['f1'][0]:.3f},{lbq['lotus_minus_pointwise_ci']['f1'][1]:.3f}]$ for F1 and $[{lbq['lotus_minus_pointwise_ci']['ratio_mae'][0]:.3f},{lbq['lotus_minus_pointwise_ci']['ratio_mae'][1]:.3f}]$ for MAE. Different prompts and contracts prevent attribution to the engine alone; this single filter excludes cascades and optimization, and its online time is not comparable to offline timings.''',
'@@SUPERVISED_TEXT@@':f'''On the 207 pools completed by every Qwen plan, the supervised lexical classifier has candidate F1 {sbm['f1']:.3f}, balanced accuracy {sbm['balanced_accuracy']:.3f}, and pool-ratio MAE {sbm['ratio_mae']:.3f}; pointwise Qwen obtains 0.815, 0.666, and 0.205. The paired query-bootstrap intervals for supervised minus pointwise are $[{sbq['supervised_minus_pointwise_ci']['f1'][0]:.3f},{sbq['supervised_minus_pointwise_ci']['f1'][1]:.3f}]$ for F1 and $[{sbq['supervised_minus_pointwise_ci']['ratio_mae'][0]:.3f},{sbq['supervised_minus_pointwise_ci']['ratio_mae'][1]:.3f}]$ for ratio MAE; neither interval clearly distinguishes the plans on this selected sample. On all 250 pools, the classifier obtains F1 {sbj['all_250']['methods']['supervised_lexical']['f1']:.3f} and MAE {sbj['all_250']['methods']['supervised_lexical']['ratio_mae']:.3f}. Its label access and the repeated product IDs limit direct interpretation against zero-shot plans. The second fit converged in {sbj['solver_iterations']} iterations; its 200-iteration predecessor did not converge and is retained in the artifact.''',
'@@SENSITIVITY_TEXT@@':f'''For SmolLM2, pointwise, direct JSON, two-stage JSON, and compact two-stage with fallback complete {ss['plans']['pointwise']['complete_pools']}, {ss['plans']['batch_json']['complete_pools']}, {ss['plans']['two_stage_json']['complete_pools']}, and {ss['plans']['two_stage_compact_fallback']['complete_pools']} of 250 pools, respectively. Every compact-plan pool has at least one unresolved chunk after retry, so the pre-specified intersection across all four plans and both models is empty. A four-plan paired quality comparison is therefore unavailable. On all 250 pools, the pointwise plan returns N for all 5,160 candidates, yielding F1 {ss['plans']['pointwise']['f1']:.3f}, balanced accuracy {ss['plans']['pointwise']['balanced_accuracy']:.3f}, and pool-ratio MAE {ss['plans']['pointwise']['ratio_mae']:.3f}. Structural completeness alone does not prevent such a degenerate prediction pattern.

As a post-hoc narrower contrast, the direct and two-stage JSON plans both complete {sj['common_pools']} pools, where their candidate F1 scores are {sj['metrics']['batch_json']['f1']:.3f} and {sj['metrics']['two_stage_json']['f1']:.3f}, respectively; the query-bootstrap interval for two-stage minus direct F1 is $[{sj['two_stage_minus_batch']['f1_ci'][0]:.3f},{sj['two_stage_minus_batch']['f1_ci'][1]:.3f}]$. This selected subset excludes failures and does not isolate the effect of the specification. The follow-up used the same prompt and token limits rather than model-specific tuning; model family, size, tokenizer, and chat template changed together, so the results do not establish a causal architecture effect.''',
'@@NCAND@@':str(sum(s['grade_counts'].values())),
'@@F1POINT@@':f"{a['methods']['pointwise']['f1']:.3f}",
'@@GRADES@@':'E=500, S=1,965, C=268, and I=849',
'@@COMMON@@':str(a['common_complete_pools']),
'@@TABLE_ROWS@@':table(s,a,order,500,[('always_yes','Always Y (post hoc)'),('always_no','Always N (post hoc)')]),
'@@FULL_COMMON@@':str(fa['common_complete_pools']),
'@@FULL_TABLE_ROWS@@':table(fs,fa,fullorder,250,[('always_yes','Always Y'),('always_no','Always N'),('lexical_all_terms','All query words')])+'\n'+f"LOTUS filter (post hoc) & 250/250 & {lbm['f1']:.3f} & {lbm['balanced_accuracy']:.3f} & {lbm['ratio_mae']:.3f} & 5,160 & -- & -- & -- " + chr(92)*2+'\n'+f"Supervised lexical (post hoc) & 250/250 & {sbm['f1']:.3f} & {sbm['balanced_accuracy']:.3f} & {sbm['ratio_mae']:.3f} & -- & -- & -- & -- " + chr(92)*2,
'@@RESULTS_TEXT@@':r'''All five plans completed 424 of 500 pools; Table~\ref{tab:main} reports quality on this common subset. The 95\% paired query-bootstrap interval for two-stage JSON minus pointwise F1 is $[-0.250,-0.195]$. These conditional results do not describe the excluded failures.''',
'@@INTERPRETATION@@':r'''On the same common subset, the post-hoc all-Y predictor attains F1 0.821 but balanced accuracy 0.500 and ratio MAE 0.301. Pointwise has lower F1, higher balanced accuracy (0.621), and MAE 0.284. The paired bootstrap interval for the all-Y minus pointwise MAE difference is $[-0.017,0.051]$, so this sample does not clearly distinguish their ratio error. The constant baseline emphasizes that F1 and overall accuracy depend on the artificially positive-heavy pool.

The gap is not confined to formatting success. Among the common pools, pointwise labels 1,169 of 1,668 S examples positive, compared with 548 of 1,668 for two-stage JSON; conversely, the latter correctly rejects 160 of 236 C examples versus 79 for pointwise. These patterns expose a precision--recall trade-off under the inherited E/S mapping, rather than evidence that either plan measures real customer satisfaction. Our design does not isolate whether the difference arises from the specification, listwise context, the prompt, or their interaction.''',
'@@EFFICIENCY@@':r'''On all 500 attempted pools, pointwise makes 3,582 calls in 27.25 seconds; direct indexed JSON makes 501 calls in 57.23 seconds (Table~\ref{tab:main}). These single-run, fixed-order GPU batch timings are not online latency; concurrency and prefix caching limit interpretation. The indexed two-stage plan has 31 invalid specifications and two other failed pools. Compact fallback issues direct judgments for those specifications but still fails 30 pools; the formats differ, so this does not isolate the fallback effect. Failed raw responses are retained.''',
'@@FULL_RESULTS@@':r'''All four plans complete 207 of 250 original pools; Table~\ref{tab:full} and Fig.~\ref{fig:quality} report quality on this intersection, excluding 43 attempted pools. The paired query-bootstrap interval for two-stage JSON minus pointwise F1 is $[-0.221,-0.164]$. On the 237 pools completed by both JSON plans, their F1 scores are 0.682 without and 0.631 with specification, with a difference interval of $[-0.070,-0.033]$. The prompts and call counts differ, so this contrast does not isolate the mechanism.''',
"@@FULL_INTERPRETATION@@":r'''On the common subset, all-Y has higher F1 but lower balanced accuracy than pointwise (Table~\ref{tab:full}); their MAE difference interval $[-0.031,0.045]$ includes zero. The high E/S share inflates all-Y F1. Direct JSON precision and recall are 0.911 and 0.541 versus 0.860 and 0.774 for pointwise, illustrating different operating points.

Two-stage JSON has 11 invalid specifications and two other failed pools; compact fallback runs for those 11 but completes only 220 pools. Post hoc, fallback completes 174/192 pools with 8--16 candidates and 42/54 with at least 25; these groups do not establish a causal size effect. Offline times do not establish online performance.'''
}
template=(base/'01_manuscript/main.template.tex').read_text(encoding='utf8')
for k,v in vals.items():template=template.replace(k,v)
assert '@@' not in template
(base/'01_manuscript/main.tex').write_text(template,encoding='utf8')
print('Rendered IEEE manuscript from saved metrics.')
