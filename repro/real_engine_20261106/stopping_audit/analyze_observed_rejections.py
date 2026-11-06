#!/usr/bin/env python3
"""Describe issued probe prefixes; no new calls, q estimate or unreached outcomes."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json
M=Path(__file__).resolve().parent;U=M.parent;A=M/'analysis';D=A/'derived';T=M/'draft/results'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main(root,main_summary,budget_summary,output,tex_output):
    global U,D,T
    U=Path(root).resolve();D=Path(output).resolve();T=Path(tex_output).resolve()
    D.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
    sm=read(main_summary);sb=read(budget_summary)
    assert sm['analysis_status']=='complete' and sm['completed_configurations']==36 and sm['integrity_error_count']==0
    assert sb['status']=='complete' and sb['completed_policies']==16 and sb['integrity_error_count']==0
    matrix=sm['case_matrix'];jobs=sb['job_matrix']
    candidates=[];fingerprints=[]
    def handle(model,identity,family,budget,result,refs=None):
        allowed=0 if budget==80 else 7
        for decision in result['calibration']:
            probes=decision['probes'];assert len(probes)==decision['actually_checked']
            count=Counter();m=len(probes)
            for i,p in enumerate(probes):
                native=p.get('reference_native_label') if refs is None else refs[i]['native_label']
                strict_valid=p.get('reference_strict_valid') if refs is None else refs[i]['strict_valid']
                assert type(native)is bool
                batch_valid=bool(p['strict_valid']) and type(p['label'])is bool
                native_mismatch=batch_valid and p['label']!=native
                expected=batch_valid and not native_mismatch and (strict_valid if family=='Original80-S' else True)
                assert p['agreement']==expected
                count['strict_invalid_reference_trials']+=not strict_valid
                if not expected:
                    if not batch_valid:count['batch_contract_failure']+=1
                    elif native_mismatch:count['native_boolean_mismatch']+=1
                    else:
                        assert family=='Original80-S' and not strict_valid
                        count['strict_only_reference_format_rejection']+=1
            failures=sum(count[k] for k in ['batch_contract_failure','native_boolean_mismatch','strict_only_reference_format_rejection'])
            assert failures==m-decision['agreements']
            passed=decision['passed'];assert passed==(m==budget and failures<=allowed)
            candidates.append({'model':model,'policy_id':identity,'family':family,'budget':budget,'k':decision['k'],
                'issued_probe_trials':m,'agreements':decision['agreements'],'actual_failure_flags':failures,
                **{k:count[k] for k in ['native_boolean_mismatch','strict_only_reference_format_rejection','batch_contract_failure','strict_invalid_reference_trials']},
                'passed':passed,'rejected':not passed,'allowed_failures_at_fixed_full_budget':allowed,
                'native_mismatches_alone_make_pass_impossible':not passed and count['native_boolean_mismatch']>allowed,
                'strict_only_decisive_after_removing_format_gate':not passed and count['native_boolean_mismatch']+count['batch_contract_failure']<=allowed,
                'outcome_role':'observed issued prefix; not an unbiased q estimate'})
    for c in matrix:
        dest=U/f'run_{c["model"]}_cache{int(c["prefix_cache"])}'/c['case_id']
        rp=dest/'CURRENT_CP_RESULT.json';sp=dest/'CURRENT_CP/calibration_singletons/native_result.json'
        fingerprints.extend([{'path':str(p.relative_to(U)),'sha256':sha(p)} for p in [rp,sp]])
        result=read(rp);refs=read(sp)['rows'];assert len(refs)==80
        handle(c['model'],c['case_id'],'Original80-S',80,result,refs)
    for c in jobs:
        p=U/f'budget_run_{c["model"]}'/c['job_id']/'POLICY_RESULT.json'
        fingerprints.append({'path':str(p.relative_to(U)),'sha256':sha(p)})
        handle(c['model'],c['job_id'],'Followup'+str(c['budget'])+'-N',c['budget'],read(p))
    aggregates=[]
    for family in ['Original80-S','Followup80-N','Followup320-N']:
        for model in ['qwen','glm']:
            rs=[r for r in candidates if r['family']==family and r['model']==model]
            planned=54 if family=='Original80-S' else 12
            aggregates.append({'family':family,'model':model,'planned_candidates':planned,'observed_candidates':len(rs),
                'unreached_candidates':planned-len(rs),'passed_candidates':sum(r['passed'] for r in rs),'rejected_candidates':sum(r['rejected'] for r in rs),
                **{k:sum(r[k] for r in rs) for k in ['issued_probe_trials','native_boolean_mismatch','strict_only_reference_format_rejection','batch_contract_failure','strict_invalid_reference_trials','native_mismatches_alone_make_pass_impossible','strict_only_decisive_after_removing_format_gate']}})
    assert len([r for r in candidates if r['family']=='Original80-S'])==106
    assert len(candidates)==154
    outcsv(D/'observed_probe_rejection_candidates.csv',candidates);outcsv(D/'observed_probe_rejection_summary.csv',aggregates)
    lines=[r'\begin{table*}[t]',r'\centering\small',
        r'\caption{Observed candidate prefixes only. O/P/R counts observed/passed/rejected candidates; M is a native-Boolean mismatch, F an additional strict-only reference-format rejection, and B a batch-contract failure. Failure flags are classified B first, then M, then F; a native mismatch may also have a non-strict reference. The last column counts rejected prefixes whose M flags alone make passing the fixed full budget impossible. Probe totals include all issued chunks. Two unreached original candidates remain unobserved. These stopped prefixes do not estimate population mismatch probability.}',
        r'\label{tab:rejection_evidence}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
        r'Family & Model & O/P/R & Issued probes & M flags & F flags & B flags & \shortstack{Native-sufficient\\stops / R} \\',r'\midrule']
    for r in aggregates:
        vals=[r['family'],{'qwen':'Qwen','glm':'GLM'}[r['model']],'/'.join(str(r[k]) for k in ['observed_candidates','passed_candidates','rejected_candidates']),
              f"{r['issued_probe_trials']:,}",str(r['native_boolean_mismatch']),str(r['strict_only_reference_format_rejection']),str(r['batch_contract_failure']),
              str(r['native_mismatches_alone_make_pass_impossible'])+'/'+str(r['rejected_candidates'])]
        lines.append(' & '.join(vals)+r' \\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (T/'observed_rejection_evidence.tex').write_text('\n'.join(lines)+'\n')
    result={'status':'OBSERVED_PREFIX_READOUT_COMPLETE','original_observed_candidates':106,'followup_observed_candidates':48,
        'aggregates':aggregates,'source_files':fingerprints,'source_sha256':sha(__file__),
        'input_summary_sha256':{'main':sha(main_summary),'budget':sha(budget_summary)},
        'no_new_inference':True,'unreached_candidates_have_no_assigned_outcomes':True,'q_estimation_claimed':False}
    (D/'OBSERVED_REJECTION_READOUT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(aggregates,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=U,help='Frozen study root containing run_* and budget_run_*')
    p.add_argument('--main-summary',type=Path,default=A/'main/summary.json')
    p.add_argument('--budget-summary',type=Path,default=A/'budget/summary.json')
    p.add_argument('--output',type=Path,default=D)
    p.add_argument('--tex-output',type=Path,default=T)
    q=p.parse_args();main(q.root,q.main_summary,q.budget_summary,q.output,q.tex_output)
