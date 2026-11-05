#!/usr/bin/env python3
"""Independently reconstruct prompts, output associations and all truncation statistics."""
import argparse
import collections
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

SYSTEM=("You judge whether each product can satisfy a shopper's query. "
        "Mark Y for an exact match or a functional substitute, and N for a mere accessory, "
        "an unrelated product, or a product lacking a stated essential requirement. "
        "Use only the supplied information. Do not assume unknown product properties.")
SUFFIX='\nOutput one character: Y or N.'


def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def parse(s):return s if s in ('Y','N') else None


def near(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(near(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(near(x,y) for x,y in zip(a,b))
    if a is None or b is None:return a is None and b is None
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)<1e-12
    return a==b


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--analysis',type=Path,required=True)
    ap.add_argument('--model',type=Path,help='optional local tokenizer for all prompt hashes and token lengths')
    a=ap.parse_args();rows=read(a.selection/'inputs.jsonl');sel=json.loads((a.selection/'selection.json').read_text())
    summary=json.loads((a.run/'summary.json').read_text());report=json.loads(a.analysis.read_text())
    digest=hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()
    assert digest==sel['inputs_sha256']==summary['selection_inputs_sha256']==report['selection_inputs_sha256']
    assert summary['protocol']==report['protocol']=='PROTOCOL_esci_truncation_sensitivity.md'
    assert summary['system']==SYSTEM and summary['retry_suffix']==SUFFIX
    ctrl=json.loads((a.run/'guidance_sanity.json').read_text())
    assert parse(ctrl['ordinary']['text']) is None and parse(ctrl['guided']['text']) is not None
    assert all(ctrl['checks'].values())
    tokenizer=None
    if a.model:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
    modes=('short','long','replay');flat=[(r,it) for r in rows for it in r['items']]
    records={m:read(a.run/(m+'_responses.jsonl')) for m in modes};by_query={m:collections.defaultdict(list) for m in modes}
    for mode in modes:
        assert len(records[mode])==len(flat)
        for (row,it),rec in zip(flat,records[mode]):
            assert (rec['query_id'],rec['item_id'],rec['product_id'],rec['text_changed'])==(
                row['query_id'],it['id'],it['product_id'],it['text_changed'])
            field='long' if mode=='long' else 'short'
            prompt='Query: '+row['query']+'\nProduct: '+it[field]+'\nAnswer ONLY Y or N.'
            assert sha(prompt)==rec['prompt_sha256']
            invalid=parse(rec['first']) is None
            assert (rec['retry_finish_reason'] is not None)==invalid
            if not invalid:assert rec['first']==rec['final'] and rec['retry_input_tokens']==rec['retry_output_tokens']==0
            else:assert rec['retry_input_tokens']>0
            if tokenizer:
                def wrap(text,tokenize=False):
                    return tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],
                                                         tokenize=tokenize,add_generation_prompt=True)
                assert sha(wrap(prompt))==rec['wrapped_sha256']
                assert len(wrap(prompt,True))==rec['first_input_tokens']
                if invalid:assert len(wrap(prompt+SUFFIX,True))==rec['retry_input_tokens']
            by_query[mode][row['query_id']].append(parse(rec['final']))
        stat=summary['modes'][mode];rs=records[mode]
        assert stat['initial_calls']==len(rs) and stat['retry_calls']==sum(x['retry_finish_reason'] is not None for x in rs)
        assert stat['valid_first']==sum(parse(x['first']) is not None for x in rs)
        assert stat['valid_final']==sum(parse(x['final']) is not None for x in rs)
        assert stat['first_finish_reasons']==dict(collections.Counter(x['first_finish_reason'] for x in rs))
        for kind in ('input','output'):
            assert stat['logical_tokens'][kind]==sum(x['first_'+kind+'_tokens']+x['retry_'+kind+'_tokens'] for x in rs)
        assert stat['max_retry_prompt_plus_cap']<=4096
    completion={m:sum(len(by_query[m][r['query_id']])==len(r['items']) and None not in by_query[m][r['query_id']]
                      for r in rows) for m in modes}
    assert report['completion']==completion and all(summary['modes'][m]['complete_pools']==completion[m] for m in modes)
    common=[r for r in rows if all(len(by_query[m][r['query_id']])==len(r['items'])
                                  and None not in by_query[m][r['query_id']] for m in modes)]
    assert common and report['attempted_pools']==len(rows) and report['attempted_candidates']==len(flat)
    assert report['common_complete_pools']==len(common) and report['common_query_ids']==[r['query_id'] for r in common]
    assert report['changed_text_candidates_all']==sel['changed_candidates']
    assert report['changed_text_candidates_common']==sum(it['text_changed'] for r in common for it in r['items'])
    preds={m:[by_query[m][r['query_id']] for r in common] for m in modes};errors={}
    for mode in modes:
        pairs=[(it['gold'],p) for r,ps in zip(common,preds[mode]) for it,p in zip(r['items'],ps)]
        c=collections.Counter(pairs);tp=c['Y','Y'];tn=c['N','N'];fp=c['N','Y'];fn=c['Y','N']
        e=[abs(Fraction(ps.count('Y')-sum(it['gold']=='Y' for it in r['items']),len(ps)))
           for r,ps in zip(common,preds[mode])]
        expected={'candidates':len(pairs),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
            'ratio_mae':float(sum(e)/len(e))}
        assert near(report['metrics'][mode],expected),mode
        errors[mode]=np.array([float(x) for x in e])
    delta=errors['long']-errors['short'];rng=np.random.default_rng(20271101)
    draws=rng.integers(0,len(common),size=(4000,len(common)));boot=delta[draws].mean(axis=1)
    assert near(report['primary_long_minus_short'],{'ratio_mae_difference':float(delta.mean()),
        'query_bootstrap_95':[float(x) for x in np.percentile(boot,[2.5,97.5])],'draws':4000,'seed':20271101})
    for mode in ('long','replay'):
        subsets={'all':[0,0],'text_changed':[0,0],'text_unchanged':[0,0]}
        for r,sp,other in zip(common,preds['short'],preds[mode]):
            for it,x,y in zip(r['items'],sp,other):
                key='text_changed' if it['text_changed'] else 'text_unchanged'
                for sub in ('all',key):subsets[sub][0]+=x!=y;subsets[sub][1]+=1
        expected={k:{'changed':v[0],'total':v[1],'fraction':v[0]/v[1] if v[1] else None} for k,v in subsets.items()}
        assert near(report['changes'][mode+'_vs_short'],expected)
    docs=sorted({it['product_id'] for r in common for it in r['items']});didx={x:i for i,x in enumerate(docs)}
    qi=[];di=[];gold=[];short=[];long=[]
    for q,(r,sp,lp) in enumerate(zip(common,preds['short'],preds['long'])):
        for it,x,y in zip(r['items'],sp,lp):
            qi.append(q);di.append(didx[it['product_id']]);gold.append(int(it['gold']=='Y'))
            short.append(int(x=='Y'));long.append(int(y=='Y'))
    qi=np.array(qi);di=np.array(di);gold=np.array(gold);short=np.array(short);long=np.array(long)
    rng=np.random.default_rng(20271102);sensitivity=[]
    for _ in range(4000):
        qdraw=rng.integers(0,len(common),len(common));weights=rng.exponential(1.,len(docs))[di]
        den=np.bincount(qi,weights=weights,minlength=len(common));assert np.all(den>0)
        g=np.bincount(qi,weights=weights*gold,minlength=len(common))/den
        s=np.bincount(qi,weights=weights*short,minlength=len(common))/den
        l=np.bincount(qi,weights=weights*long,minlength=len(common))/den
        sensitivity.append(float(np.mean((np.abs(l-g)-np.abs(s-g))[qdraw])))
    assert near(report['product_weight_sensitivity'],{'distinct_products':len(docs),'draws':4000,'seed':20271102,
        'reweighted_range_95':[float(x) for x in np.percentile(sensitivity,[2.5,97.5])]})
    print('All prompt hashes, candidate associations, retries, completions, metrics and both resampling analyses independently verified.')
    if tokenizer:print('Wrapped prompt hashes and every input-token count independently verified.')


if __name__=='__main__':main()
