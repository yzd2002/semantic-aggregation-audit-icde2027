#!/usr/bin/env python3
"""Paired query differences across two saved ACORD runs, not a causal model comparison."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from acord_prompts import IDS, parse_keyed


def rows(path):
    with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def plans(dataset, run):
    records=rows(run/'original_responses.jsonl');ids=[(r['pool_id'],n) for r in dataset for n in (0,8,16,24)]
    if [(x['pool_id'],x['start']) for x in records]!=ids:raise ValueError('Original request order mismatch')
    selected=rows(run/'selective_outputs.jsonl')
    if [x['pool_id'] for x in selected]!=[x['pool_id'] for x in dataset]:raise ValueError('Selective pool order mismatch')
    output={}
    for i,row in enumerate(dataset):
        chunks=[parse_keyed(rec['final']) for rec in records[i*4:i*4+4]]
        orig=[x[j] for x in chunks for j in IDS] if all(x is not None for x in chunks) else None
        sel=selected[i]['pred']
        if sel is not None and (len(sel)!=32 or set(sel)-{'Y','N'}):raise ValueError('Invalid selective vector')
        output[row['pool_id']]={'original':orig,'selective':sel}
    return output


def main(selection,qwen,smol,out):
    if out.exists():raise FileExistsError('Use a new output file')
    inputs=selection/'inputs.jsonl';dataset=rows(inputs);meta=json.loads((selection/'selection.json').read_text())
    if sha(inputs)!=meta['inputs_sha256']:raise ValueError('Input hash differs')
    summaries=[json.loads((p/'summary.json').read_text()) for p in (qwen,smol)]
    if not all(s.get('completed') and s['selection_inputs_sha256']==sha(inputs) for s in summaries):
        raise ValueError('Run incomplete or selection differs')
    models={'qwen':plans(dataset,qwen),'smol':plans(dataset,smol)}
    result={'scope':'Exploratory matched-query follow-up after observing Qwen; not a causal architecture comparison.',
            'selection_sha256':sha(inputs),'qwen_summary_sha256':sha(qwen/'summary.json'),
            'smol_summary_sha256':sha(smol/'summary.json'),'resamples':4000,'scenarios':{}}
    for scenario,seed in (('case_control',20261111),('uniform',20261112)):
        attempted=[r for r in dataset if r['scenario']==scenario]
        complete={k:sum(all(v[r['pool_id']][m] is not None for m in ('original','selective')) for r in attempted) for k,v in models.items()}
        shared=[r for r in attempted if all(all(v[r['pool_id']][m] is not None for m in ('original','selective')) for v in models.values())]
        entry={'attempted':len(attempted),'complete_by_model':complete,'shared_complete':len(shared),'query_bootstrap_seed':seed}
        if not shared:result['scenarios'][scenario]=entry;continue
        errors={k:{m:[] for m in ('original','selective')} for k in models}
        for r in shared:
            gold=sum(x['gold']=='Y' for x in r['items'])
            for k,v in models.items():
                for m in ('original','selective'):
                    errors[k][m].append(abs(sum(x=='Y' for x in v[r['pool_id']][m])-gold)/32)
        entry['ratio_mae']={k:{m:float(np.mean(e)) for m,e in model.items()} for k,model in errors.items()}
        q=np.asarray(errors['qwen']['selective'])-errors['qwen']['original']
        s=np.asarray(errors['smol']['selective'])-errors['smol']['original'];delta=s-q
        rng=np.random.default_rng(seed);draw=rng.integers(0,len(shared),size=(4000,len(shared)));boot=delta[draw].mean(axis=1)
        entry['qwen_selective_minus_original']=float(q.mean());entry['smol_selective_minus_original']=float(s.mean())
        entry['difference_of_differences']=float(delta.mean());entry['query_bootstrap_95']=[float(v) for v in np.quantile(boot,[.025,.975])]
        result['scenarios'][scenario]=entry
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Computed descriptive paired cross-model effects on shared complete queries; no inference rerun.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--qwen',type=Path,required=True);p.add_argument('--smol',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.selection,a.qwen,a.smol,a.out)
