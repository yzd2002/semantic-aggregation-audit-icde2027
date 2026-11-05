#!/usr/bin/env python3
"""Independent strict parsing, token accounting and paired-result verification."""
import argparse,collections,hashlib,json,math,re
from pathlib import Path
import numpy as np
from batch_prompts import prompt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def labels(text,n):
    if not isinstance(text,str):return None
    m=re.fullmatch(r'\{'+','.join('"'+str(i)+'":"([YN])"' for i in range(1,n+1))+r'\}',text)
    return list(m.groups()) if m else None
def close(x,y):return x is None and y is None or x is not None and y is not None and math.isclose(x,y,abs_tol=1e-12,rel_tol=1e-12)
def stats(gold,pred):
    c=collections.Counter((a,b) for g,p in zip(gold,pred) for a,b in zip(g,p))
    tp,fp,fn,tn=c['Y','Y'],c['N','Y'],c['Y','N'],c['N','N']
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
            'ratio_mae':sum(abs(g.count('Y')-p.count('Y'))/32 for g,p in zip(gold,pred))/len(gold)}

def main(selection,preflight,recovery,prior,analysis):
    rows=lines(selection/'inputs.jsonl');pf=lines(preflight/'prompts.jsonl');f=json.loads((recovery/'summary.json').read_text());old=json.loads((prior/'summary.json').read_text());a=json.loads(analysis.read_text())
    meta=json.loads((preflight/'summary.json').read_text());checks=json.loads((recovery/'guidance_sanity.json').read_text())
    assert f['completed'] and old['completed'] and all(checks['checks'].values())
    assert sha(selection/'inputs.jsonl')==f['selection_sha256']==old['selection_inputs_sha256']==a['selection_sha256']
    assert sha(preflight/'prompts.jsonl')==meta['prompt_records_sha256'] and sha(recovery/'summary.json')==a['recovery_summary_sha256'] and sha(prior/'summary.json')==a['prior_summary_sha256']
    ref={(x['pool_id'],x['size'],x['start'],x['mode']):x for x in pf};assert len(ref)==24*len(rows)==len(pf)
    modes={'eight_original':8,'four_original':4,'four_stable_reverse':4,'four_replay':4,'eight_replay':8};vectors={};audit={}
    for name,size in modes.items():
        path=recovery/(name+'_responses.jsonl');rec=lines(path);assert len(rec)==len(rows)*32//size and sha(path)==f['modes'][name]['responses_sha256']
        counts=collections.Counter();tot=collections.Counter();v={};reverse=name=='four_stable_reverse';per=32//size
        for i,r in enumerate(rec):
            row=rows[i//per];start=i%per*size;text,ids=prompt(row,start,size,reverse);q=ref[row['pool_id'],size,start,'stable_reverse' if reverse else 'original']
            assert (r['pool_id'],r['size'],r['start'],r['presented_ids'])==(row['pool_id'],size,start,ids)
            assert q['presented_ids']==ids and q['prompt_sha256']==r['prompt_sha256']==hashlib.sha256(text.encode()).hexdigest()
            assert r['first_input_tokens']==q['prompt_tokens']
            first,last=labels(r['first'],size),labels(r['final'],size)
            if first is None:
                counts['retry_calls']+=1;assert r['retry_input_tokens']==q['retry_prompt_tokens'] and r['retry_finish_reason'] is not None
            else:assert r['final']==r['first'] and r['retry_finish_reason'] is None and not r['retry_input_tokens'] and not r['retry_output_tokens']
            counts['valid_first']+=first is not None;counts['valid_final']+=last is not None;counts['finish:'+str(r['first_finish_reason'])]+=1
            for k in ('input','output'):tot[k]+=r['first_'+k+'_tokens']+r['retry_'+k+'_tokens']
            v.setdefault(row['pool_id'],[]).append(last)
        saved=f['modes'][name];assert saved['initial_calls']==len(rec) and all(saved[k]==counts[k] for k in ('retry_calls','valid_first','valid_final'))
        assert saved['logical_tokens']==dict(tot) and saved['first_finish_reasons']=={k[7:]:v for k,v in counts.items() if k.startswith('finish:')}
        vectors[name]={pid:sum(groups,[]) if all(x is not None for x in groups) else None for pid,groups in v.items()}
        audit[name]={'initial_calls':len(rec),'retry_calls':counts['retry_calls'],'logical_tokens':dict(tot)}
    def prior_records(name,size):
        rec=lines(prior/(name+'_responses.jsonl'));per=32//size;assert len(rec)==len(rows)*per;v={}
        for i,r in enumerate(rec):
            row=rows[i//per];assert r['pool_id']==row['pool_id'] and r['start']==i%per*size
            x=labels(r['final'],size) if size>1 else ([r['final']] if r['final'] in ('Y','N') else None)
            v.setdefault(row['pool_id'],[]).append(x)
        return {pid:sum(groups,[]) if all(x is not None for x in groups) else None for pid,groups in v.items()}
    old8=prior_records('original',8);point=prior_records('pointwise',1)
    for scenario,seed in (('case_control',20261113),('uniform',20261114)):
        attempted=[r for r in rows if r['scenario']==scenario];chosen=[r for r in attempted if vectors['four_original'][r['pool_id']] is not None and vectors['eight_original'][r['pool_id']] is not None];x=a['scenarios'][scenario]
        assert x['attempted_pools']==len(attempted) and x['common_complete_pools']==len(chosen)
        for mode,v in vectors.items():assert x['completion'][mode]==sum(v[r['pool_id']] is not None for r in attempted)
        for mode,base in (('four_stable_reverse','four_original'),('four_replay','four_original'),('eight_replay','eight_original')):
            common=[r for r in attempted if vectors[mode][r['pool_id']] is not None and vectors[base][r['pool_id']] is not None]
            changed=sum(a!=b for r in common for a,b in zip(vectors[mode][r['pool_id']],vectors[base][r['pool_id']]))
            assert x[mode+'_changed_labels']==changed and x[mode+'_compared_labels']==32*len(common)
        common=[r for r in attempted if vectors['eight_original'][r['pool_id']] is not None and old8[r['pool_id']] is not None]
        assert x['prior_eight_vs_recovery_compared_labels']==32*len(common)
        assert x['prior_eight_vs_recovery_changed_labels']==sum(a!=b for r in common for a,b in zip(vectors['eight_original'][r['pool_id']],old8[r['pool_id']]))
        if not chosen:continue
        assert x['common_pool_ids']==[r['pool_id'] for r in chosen]
        gold=[[item['gold'] for item in r['items']] for r in chosen];p4=[vectors['four_original'][r['pool_id']] for r in chosen];p8=[vectors['eight_original'][r['pool_id']] for r in chosen]
        pred={'four_original':p4,'eight_original':p8,'always_N':[['N']*32 for _ in chosen]}
        if all(point[r['pool_id']] is not None for r in chosen):pred['prior_full_pointwise']=[point[r['pool_id']] for r in chosen]
        for name,guess in pred.items():
            result=stats(gold,guess);assert set(result)==set(x['metrics'][name]) and all(close(v,x['metrics'][name][k]) for k,v in result.items())
        delta=[(abs(g.count('Y')-p.count('Y'))-abs(g.count('Y')-q.count('Y')))/32 for g,p,q in zip(gold,p4,p8)]
        draws=np.random.default_rng(seed).integers(0,len(delta),size=(4000,len(delta)))
        boot=[sum(delta[int(i)] for i in draw)/len(delta) for draw in draws]
        assert close(x['four_minus_eight_ratio_mae'],sum(delta)/len(delta))
        assert all(close(v,z) for v,z in zip(x['query_bootstrap_95'],np.quantile(boot,[.025,.975])))
    assert a['all_pool_original_costs']=={'four':f['modes']['four_original'],'eight':f['modes']['eight_original']}
    print(json.dumps({'verified':True,'pool_count':len(rows),'modes':audit,'analysis_sha256':sha(analysis)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('selection','preflight','recovery','prior','analysis'):p.add_argument('--'+n,type=Path,required=True)
    x=p.parse_args();main(x.selection,x.preflight,x.recovery,x.prior,x.analysis)
