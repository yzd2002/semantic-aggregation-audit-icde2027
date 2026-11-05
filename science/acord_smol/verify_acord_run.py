#!/usr/bin/env python3
"""Independently verify saved ACORD prompts, routing and paired statistics."""
import argparse,collections,hashlib,json,math,re
from pathlib import Path
import numpy as np
from acord_prompts import IDS,SYSTEM,RETRY_SUFFIX,keyed_prompt,point_prompt
from run_keyed_order_audit import parse_keyed as strict_parse

def lines(p):
    with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
def f1(tp,fp,fn):return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0

def main(selection,preflight,run,analysis):
    rows=lines(selection/'inputs.jsonl');sel=json.loads((selection/'selection.json').read_text());summary=json.loads((run/'summary.json').read_text());result=json.loads(analysis.read_text())
    assert digest(selection/'inputs.jsonl')==sel['inputs_sha256']==summary['selection_inputs_sha256']==result['selection_sha256']
    assert summary.get('completed') and len({r['pool_id'] for r in rows})==len(rows)
    controls=json.loads((run/'guidance_sanity.json').read_text())['checks'];assert all(controls.values())
    pf=lines(preflight/'prompts.jsonl');pfmap={(x['pool_id'],x['mode'],x['start']):x for x in pf};assert len(pfmap)==len(pf)
    modes=('original','stable_reverse','selective_pointwise','replay','pointwise');events={k:lines(run/(k+'_responses.jsonl')) for k in modes};byid={r['pool_id']:r for r in rows}
    idx={}
    for mode,recs in events.items():
        idx[mode]={}
        for rec in recs:
            key=(rec['pool_id'],rec['start']);assert key not in idx[mode] and rec['pool_id'] in byid
            row=byid[rec['pool_id']]
            if mode in ('original','stable_reverse','replay'):
                reverse=mode=='stable_reverse';prompt,presented=keyed_prompt(row,rec['start'],reverse)
                pkey=(rec['pool_id'],'keyed_reverse' if reverse else 'keyed_original',rec['start'])
                assert rec['size']==8 and rec['presented_ids']==presented
            else:
                prompt=point_prompt(row,rec['start']);pkey=(rec['pool_id'],'pointwise',rec['start']);assert rec['size']==1
            p=pfmap[pkey];h=hashlib.sha256(prompt.encode('utf8')).hexdigest()
            assert h==p['prompt_sha256']==rec['prompt_sha256'] and rec['first_input_tokens']==p['prompt_tokens']
            if rec['retry_finish_reason'] is None:assert rec['retry_input_tokens']==0
            else:assert rec['retry_input_tokens']==p['retry_prompt_tokens']
            idx[mode][key]=rec
        m=summary['modes'][mode]
        assert m['initial_calls']==len(recs) and m['retry_calls']==sum(x['retry_finish_reason'] is not None for x in recs)
        for kind in ('input','output'):
            assert m['logical_tokens'][kind]==sum(x['first_'+kind+'_tokens']+x['retry_'+kind+'_tokens'] for x in recs)
    def label_vector(row,mode):
        out=[]
        for start in (0,8,16,24):
            x=strict_parse(idx[mode][(row['pool_id'],start)]['final'],IDS)
            if x is None:return None
            out.extend(x[i] for i in IDS)
        return out
    vectors={r['pool_id']:{k:label_vector(r,k) for k in ('original','stable_reverse','replay')} for r in rows}
    need=set();selected={}
    for r in rows:
        x=vectors[r['pool_id']];a=x['original'];b=x['stable_reverse']
        if a is None or b is None:selected[r['pool_id']]=None;continue
        v=[]
        for j,(p,q) in enumerate(zip(a,b)):
            if p==q:v.append(p)
            else:
                need.add((r['pool_id'],j));raw=idx['selective_pointwise'][(r['pool_id'],j)]['final'];v.append(raw if raw in ('Y','N') else None)
        selected[r['pool_id']]=v if None not in v else None
    assert need==set(idx['selective_pointwise'])
    saved=lines(run/'selective_outputs.jsonl');assert [(x['pool_id'],x['pred']) for x in saved]==[(r['pool_id'],selected[r['pool_id']]) for r in rows]
    assert set(idx['pointwise'])=={(r['pool_id'],i) for r in rows for i in range(32)}
    for scenario,seed in (('case_control',20261101),('uniform',20261102)):
        common=[r for r in rows if r['scenario']==scenario and all(vectors[r['pool_id']][k] is not None for k in ('original','stable_reverse')) and selected[r['pool_id']] is not None]
        info=result['scenarios'][scenario];assert [r['pool_id'] for r in common]==info.get('common_pool_ids',[])
        if not common:continue
        errors={k:[] for k in ('original','midpoint','lexical_fallback','selective')};arrays=[];docs=[]
        for r in common:
            a=vectors[r['pool_id']]['original'];b=vectors[r['pool_id']]['stable_reverse'];s=selected[r['pool_id']];g=[i['gold'] for i in r['items']]
            words=set(re.findall(r'\w+',r['query'].casefold()))
            lex=['Y' if words and words.issubset(set(re.findall(r'\w+',it['text'].casefold()))) else 'N' for it in r['items']]
            tie=[p if p==q else t for p,q,t in zip(a,b,lex)]
            pos=g.count('Y');pa=a.count('Y');pb=b.count('Y');ps=s.count('Y');pt=tie.count('Y')
            for k,v in [('original',abs(pa-pos)/32),('midpoint',abs(pa+pb-2*pos)/64),('lexical_fallback',abs(pt-pos)/32),('selective',abs(ps-pos)/32)]:errors[k].append(v)
            arrays.append(([z=='Y' for z in g],[z=='Y' for z in a],[z=='Y' for z in b],[z=='Y' for z in s],[z=='Y' for z in tie]))
            docs.append([it['doc_id'] for it in r['items']])
        n=len(common);assert close(sum(errors['midpoint'])/n,info['midpoint_ratio_mae'])
        rng=np.random.default_rng(seed);draw=rng.integers(0,n,size=(4000,n))
        for k in ('original','midpoint','lexical_fallback'):
            delta=[x-y for x,y in zip(errors['selective'],errors[k])];stored=info['selective_minus'][k]
            assert close(sum(delta)/n,stored['ratio_mae'])
            boot=[sum(delta[i] for i in d)/n for d in draw]
            assert all(close(x,y) for x,y in zip(np.quantile(boot,[.025,.975]),stored['query_bootstrap_95']))
        ids=sorted({x for group in docs for x in group});lookup={x:i for i,x in enumerate(ids)};rng=np.random.default_rng(seed+1000);weighted={k:[] for k in ('original','midpoint','lexical_fallback')}
        for _ in range(4000):
            qdraw=rng.integers(0,n,size=n);weights=rng.exponential(size=len(ids));diffs={k:[] for k in weighted}
            for group,(g,a,b,s,tie) in zip(docs,arrays):
                den=gg=aa=bb=ss=tt=0.0
                for d,g0,a0,b0,s0,t0 in zip(group,g,a,b,s,tie):
                    w=float(weights[lookup[d]]);den+=w;gg+=w*g0;aa+=w*a0;bb+=w*b0;ss+=w*s0;tt+=w*t0
                base=abs(ss/den-gg/den)
                for k,x in [('original',aa),('midpoint',(aa+bb)/2),('lexical_fallback',tt)]:diffs[k].append(base-abs(x/den-gg/den))
            for k in weighted:weighted[k].append(sum(diffs[k][i] for i in qdraw)/n)
        for k in weighted:
            assert all(close(x,y) for x,y in zip(np.quantile(weighted[k],[.025,.975]),info['selective_minus'][k]['clause_reweighted_range_95']))
        for name,vecpos in [('original',1),('reverse',2),('selective',3),('lexical_fallback',4)]:
            tp=fp=fn=tn=0
            for v in arrays:
                g=v[0];p=v[vecpos]
                tp+=sum(x and y for x,y in zip(g,p));fp+=sum(not x and y for x,y in zip(g,p));fn+=sum(x and not y for x,y in zip(g,p));tn+=sum(not x and not y for x,y in zip(g,p))
            m=info['metrics'][name];assert (tp,fp,fn,tn)==(m['tp'],m['fp'],m['fn'],m['tn']) and close(f1(tp,fp,fn),m['f1'])
        print(scenario,'verified',n,'complete pools; strict ID parser, request hashes, costs and paired intervals',flush=True)
    print('Independent verification passed; no inference was rerun.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);a=p.parse_args();main(a.selection,a.preflight,a.run,a.analysis)
