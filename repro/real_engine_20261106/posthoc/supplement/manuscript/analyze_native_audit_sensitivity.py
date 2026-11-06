#!/usr/bin/env python3
"""Post-hoc native-vector sensitivity; fixed existing HG API/design, no inference.

Canonical strict eligibility and outputs are retained. This adds a separately
named semantic readout on every available immutable native Boolean vector.
"""
from pathlib import Path
import csv,hashlib,json,random,sys
sys.dont_write_bytecode=True
M=Path(__file__).resolve().parent;U=M.parent;A=M/'analysis'
sys.path.insert(0,str(U/'novelty_theory'))
from finite_audit import audit_population,hg_interval

def source_path(p):
    q=Path(p);original=Path('/home/gpu/IntentSat2S_upgrade_20261106')
    return U/q.relative_to(original) if q.is_absolute() and q.is_relative_to(original) else U/q if not q.is_absolute() else q

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    receipt=read(A/'ANALYSIS_COMPLETE.json')
    assert receipt['main_configurations']==36 and receipt['main_integrity_errors']==0
    study=read(U/'prepared/STUDY.json');canon=read(A/'finite/FIXED_BUDGET_AUDITS.json')
    integrity=read(A/'main/integrity.json')
    fingerprints={str(source_path(f['path']).resolve()):f['sha256'] for f in integrity['fingerprints']}
    checked_files=set()
    def verified_read(p):
        p=Path(p).resolve();assert str(p) in fingerprints,('not independently fingerprinted',str(p))
        if p not in checked_files:
            assert sha(p)==fingerprints[str(p)],('changed after analysis snapshot',str(p))
            checked_files.add(p)
        return read(p)
    canonical={(r['model'],r['case_id'],r['policy'],r.get('reference')):r for r in canon['rows'] if r['status']=='complete'}
    bywork={w['workload_id']:w for w in study['workloads']}
    golds={}
    for name,w in bywork.items():
        gp=source_path(w['gold']).resolve();assert str(gp) in fingerprints and sha(gp)==fingerprints[str(gp)];checked_files.add(gp)
        raw=[json.loads(l) for l in gp.read_text().splitlines() if l.strip()]
        golds[name]=[bool(r['gold']) if w['domain']=='movie' else r['source_class']==w['gold_positive_class'] for r in raw]
    rows=[];details=[];checks=0
    for model in study['models']:
        for case in study['cases']:
            n=case['n'];m=min(1024,n);dest=U/f'run_{model}_cache{int(case["prefix_cache"])}'/case['case_id']
            assert (dest/'CASE_COMPLETE.json').exists()
            s1=verified_read(dest/'S1_RESULT.json');s2=verified_read(dest/'S2_RESULT.json')
            if s1['strict_complete']:assert s1['labels']==s1['strict_labels']
            targets=[('source_gold',golds[case['workload_id']][:n])]
            if all(type(x)is bool for x in s1['labels']):targets.append(('frozen_native_S1',s1['labels']))
            key='|'.join([study['config']['seed'],case['partition_key'],'fixed_budget_audit_order'])
            order=list(range(n));random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16],16)).shuffle(order)
            for policy in ['RANDOM8','SIMILAR8','CURRENT_CP']:
                result=verified_read(dest/f'{policy}_RESULT.json');b=result['labels']
                common={'model':model,**case,'policy':policy,'canonical_strict_complete':result['strict_complete'],
                        'analysis_role':'post_hoc_native_vector_sensitivity','new_inference':False}
                if len(b)!=n or not all(type(x)is bool for x in b):
                    rows.append({**common,'status':'native_vector_unavailable'});continue
                if policy in ['RANDOM8','SIMILAR8'] and result['strict_complete']:
                    assert all(type(x)is bool for x in b)
                for target,y in targets:
                    v=audit_population(b,y,order,looks=(m,),epsilon=.05,alpha=.05)
                    z=v['looks'][0];direct=hg_interval(n,m,sum(b[i]!=y[i] for i in order[:m]),.05)
                    g=sum(y);delta=sum(b)-g
                    row={**common,'reference':target,'status':'complete','oracle_labels':m,
                         'nonempty_prediction_strata':int(sum(b)>0)+int(sum(b)<n),
                         'raw_count_accepted':z['raw_count_accepted'],'joint_row_accepted':z['row_accepted'],
                         'direct_row_accepted':direct[1]<=.05*n,'count_interval_low':z['count_delta_interval'][0],
                         'count_interval_high':z['count_delta_interval'][1],
                         'actual_count_delta':delta,'actual_row_errors':sum(x!=t for x,t in zip(b,y)),
                         'actual_count_error_over_N':abs(delta)/n,
                         'actual_count_error_over_target_count':abs(delta)/g if g>0 else None,
                         'target_count':g,'predicted_count':sum(b),'target_zero':g==0,
                         'same_batch_delta_to_native_S2':sum(b)-sum(s2['labels']) if all(type(x)is bool for x in s2['labels']) else None}
                    rows.append(row);details.append({'model':model,'case_id':case['case_id'],'policy':policy,'reference':target,'audit':v})
                    oldname='source_gold' if target=='source_gold' else 'frozen_S1_research_replay'
                    old=canonical.get((model,case['case_id'],policy,oldname))
                    if old is not None and (target=='source_gold' or s1['strict_complete']):
                        assert (old['count_interval_low'],old['count_interval_high'],old['raw_count_accepted'],old['joint_count_row_accepted'])==(row['count_interval_low'],row['count_interval_high'],row['raw_count_accepted'],row['joint_row_accepted'])
                        checks+=1
    out=A/'native_sensitivity';out.mkdir(exist_ok=True)
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'NATIVE_VECTOR_AUDITS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    result={'status':'POST_HOC_NATIVE_VECTOR_SENSITIVITY_COMPLETE','role':'explicit post-hoc semantic sensitivity; not a new primary experiment',
            'planned_configurations':36,'scope':'all 36; three existing plans; source gold and frozen native S1',
            'same_frozen_srs_seed_budget_epsilon_alpha':True,'canonical_strict_output_unmodified':True,
            'matching_canonical_pair_checks':checks,'independently_fingerprinted_input_files_rechecked':len(checked_files),'source_api_sha256':sha(U/'novelty_theory/finite_audit.py'),
            'analysis_receipt_sha256':sha(A/'ANALYSIS_COMPLETE.json'),'script_sha256':sha(__file__),
            'rows':rows,'details':details,'future_reference_coverage_claimed':False}
    (out/'NATIVE_VECTOR_AUDITS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','planned_configurations','matching_canonical_pair_checks']},indent=2))
if __name__=='__main__':main()
