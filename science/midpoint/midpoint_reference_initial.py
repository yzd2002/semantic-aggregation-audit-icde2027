#!/usr/bin/env python3
"""Reconstruct an exploratory fractional-ratio comparator from saved traces."""
import argparse, hashlib, json, tarfile
from pathlib import Path
import numpy as np

CONFIG = [
    ('wands96_qwen', 'wands_fallback_validation_selection', 'run_wands_fallback_validation_qwen_v1', None),
    ('wands52_qwen', 'wands_selective_execution_selection', 'run_wands_selective_execution_qwen_v1', None),
    ('wands52_smollm', 'wands_selective_execution_selection', 'run_wands_selective_execution_smollm_v1', None),
    ('esci250_qwen', 'esci_selective_holdout_selection', 'run_esci_selective_holdout_qwen', None),
    ('esci250_smollm', 'esci_selective_holdout_selection', 'run_esci_selective_holdout_smollm', None),
    ('trec50_qwen', 'trec_covid_transfer_selection', 'run_trec_covid_transfer', None),
    ('trec50_smollm', 'trec_covid_transfer_selection', 'run_trec_covid_smollm', None),
    ('esci250_7b', 'esci_selective_holdout_selection', None, 'posthoc/run_esci_7b'),
    ('esci250_7b_schema', 'esci_selective_holdout_selection', None, 'posthoc/run_esci_7b_specguided'),
    ('wands52_7b', 'wands_selective_execution_selection', None, 'posthoc/run_wands_7b'),
    ('wands52_7b_schema', 'wands_selective_execution_selection', None, 'posthoc/run_wands_7b_specguided'),
    ('trec50_7b', 'trec_covid_transfer_selection', None, 'posthoc/run_trec_7b'),
]

def lines(raw):
    return [json.loads(x) for x in raw.decode('utf8').splitlines() if x.strip()]

def pairs_no_duplicate(pairs):
    out = {}
    for k,v in pairs:
        if k in out:
            raise ValueError('duplicate JSON key')
        out[k] = v
    return out

def parse_keyed(text, size):
    try:
        x=json.loads(text, object_pairs_hook=pairs_no_duplicate)
        wanted={str(i) for i in range(1,size+1)}
        if not isinstance(x,dict) or set(x)!=wanted or any(type(v) is not str or v not in ('Y','N') for v in x.values()):
            return None
        return [x[str(i)] for i in range(1,size+1)]
    except (ValueError,TypeError):
        return None

def keyed(rows, raw, id_field, mode):
    rec=lines(raw)
    expected=[(row[id_field], start, min(8,len(row['items'])-start)) for row in rows for start in range(0,len(row['items']),8)]
    if [(x[id_field], x['start'], x['size']) for x in rec] != expected:
        raise ValueError('response query, chunk start, or size differs')
    by_id={row[id_field]: [None]*len(row['items']) for row in rows}
    if len(by_id)!=len(rows):raise ValueError('duplicate query/topic ID')
    for r in rec:
        expected_ids=list(range(1,r['size']+1))
        order=r['presented_ids']
        if order != (expected_ids if mode=='original' else expected_ids[::-1]):
            raise ValueError('stable-ID presentation order mismatch')
        parsed=parse_keyed(r['final'],r['size'])
        if parsed is not None:
            by_id[r[id_field]][r['start']:r['start']+r['size']]=parsed
    return [v if None not in v else None for v in (by_id[row[id_field]] for row in rows)]

def selection_outputs(rows, raw, id_field):
    records=lines(raw)
    if [x[id_field] for x in records] != [r[id_field] for r in rows]:
        raise ValueError('selective output ID order mismatch')
    values=[]
    for row,rec in zip(rows,records):
        v=rec['pred']
        if v is not None and (len(v)!=len(row['items']) or any(x not in ('Y','N') for x in v)):
            raise ValueError('invalid selective vector')
        values.append(v)
    return values

def virtual_outputs(rows, original, reverse, raw, id_field):
    records=lines(raw)
    want=[(row[id_field],i) for row in rows for i in range(len(row['items']))]
    if [(r[id_field],r['start']) for r in records] != want:
        raise ValueError('pointwise record order mismatch')
    point={(r[id_field],r['start']):r for r in records}
    labels=[];calls=0;tokens=0
    for row,a,b in zip(rows,original,reverse):
        if a is None or b is None:
            labels.append(None);continue
        v=[]
        for i,(x,y) in enumerate(zip(a,b)):
            if x==y:v.append(x);continue
            rec=point[(row[id_field],i)];z=rec['final']
            v.append(z if z in ('Y','N') else None)
            calls+=1+int(rec['retry_finish_reason'] is not None)
            tokens+=rec['first_input_tokens']+rec['retry_input_tokens']
        labels.append(v if None not in v else None)
    return labels,calls,tokens

def costs(summary):
    modes=summary['modes']
    base=[modes['original'],modes['stable_reverse']]
    spec=summary.get('specifications',modes.get('specifications'))
    if spec is not None:base.append(spec)
    return {'calls':sum(x['initial_calls']+x['retry_calls'] for x in base),
            'input_tokens':sum(x['logical_tokens']['input'] for x in base)}

def read_expected(root,name,archive):
    if name=='wands96_qwen':p=root/'run_wands_fallback_validation_qwen_v1/paired_analysis.json'
    elif name.startswith('wands52_') and not name.endswith(('7b','schema')):
        p=root/('run_wands_selective_execution_'+('qwen' if name.endswith('qwen') else 'smollm')+'_v1/paired_analysis.json')
    elif name.startswith('esci250_') and not name.endswith(('7b','schema')):
        p=root/('analysis_esci_selective_holdout_'+('qwen' if name.endswith('qwen') else 'smollm')+'/summary.json')
    elif name=='trec50_qwen':p=root/'TREC_COVID_TRANSFER_ANALYSIS.json'
    elif name=='trec50_smollm':p=root/'TREC_COVID_SMOL_ANALYSIS.json'
    else:
        key={'esci250_7b':'esci','esci250_7b_schema':'esci_specguided',
             'wands52_7b':'wands','wands52_7b_schema':'wands_specguided','trec50_7b':'trec'}[name]
        candidates={'esci':'posthoc/analysis_esci_7b/summary.json',
                    'esci_specguided':'posthoc/analysis_esci_7b_specguided/summary.json',
                    'wands':'posthoc/run_wands_7b/paired_analysis.json',
                    'wands_specguided':'posthoc/run_wands_7b_specguided/paired_analysis.json',
                    'trec':'posthoc/trec_7b_analysis.json'}
        value=json.load(archive.extractfile(candidates[key]))
        if key=='trec':return value['metrics']['original']['ratio_mae'],value['metrics']['selective']['ratio_mae']
        return value['quality_common']['original']['ratio_mae'],value['quality_common']['actual_fallback']['ratio_mae']
    v=json.loads(p.read_text())
    if name.startswith('trec'):return v['metrics']['original']['ratio_mae'],v['metrics']['selective']['ratio_mae']
    return v['quality_common']['original']['ratio_mae'],v['quality_common']['pointwise_fallback' if name=='wands96_qwen' else 'actual_fallback']['ratio_mae']

def calculate(root,out):
    archive_path=root/'posthoc_7b/posthoc_7b_artifacts.tar.gz'
    results={'scope':'post-hoc, exposed samples; ratio-only comparator; no new inference',
             'protocol':'PROTOCOL.md','settings':{}}
    with tarfile.open(archive_path,'r:gz') as archive:
        for idx,(name,selection,run,member) in enumerate(CONFIG):
            input_bytes=(root/selection/'inputs.jsonl').read_bytes();rows=lines(input_bytes)
            id_field='topic_id' if name.startswith('trec') else 'query_id'
            if len({r[id_field] for r in rows})!=len(rows):raise ValueError('duplicate query/topic')
            def get(filename):
                return (root/run/filename).read_bytes() if run else archive.extractfile(member+'/'+filename).read()
            summary=json.loads(get('summary.json'))
            if summary['selection_inputs_sha256'] != hashlib.sha256(input_bytes).hexdigest():raise ValueError('selection hash mismatch '+name)
            original=keyed(rows,get('original_responses.jsonl'),id_field,'original')
            reverse=keyed(rows,get('stable_reverse_responses.jsonl'),id_field,'reverse')
            base=costs(summary)
            if name=='wands96_qwen':
                selected,extra_calls,extra_tokens=virtual_outputs(rows,original,reverse,get('pointwise_responses.jsonl'),id_field)
                scost={'calls':base['calls']+extra_calls,'input_tokens':base['input_tokens']+extra_tokens}
            else:
                selected=selection_outputs(rows,get('selective_outputs.jsonl'),id_field)
                p=summary['selective_pipeline'];scost={'calls':p['initial_calls']+p['retry_calls'],'input_tokens':p['logical_tokens']['input']}
                mode=summary['modes']['selective_pointwise'];extra={'calls':mode['initial_calls']+mode['retry_calls'],'input_tokens':mode['logical_tokens']['input']}
                if any(scost[k]!=base[k]+extra[k] for k in base):raise ValueError('pipeline cost mismatch '+name)
            common=[i for i in range(len(rows)) if all(x[i] is not None for x in (original,reverse,selected))]
            if not common:raise ValueError('no common complete pools '+name)
            gold=[];aa=[];bb=[];ss=[];ids=[]
            for i in common:
                row=rows[i];g=np.asarray([x['gold']=='Y' for x in row['items']],dtype=float)
                if any(x['gold'] not in ('Y','N') for x in row['items']):raise ValueError('non-binary gold')
                av=np.asarray([x=='Y' for x in original[i]],dtype=float)
                bv=np.asarray([x=='Y' for x in reverse[i]],dtype=float)
                sv=np.asarray([x=='Y' for x in selected[i]],dtype=float)
                gold.append(g);aa.append(av);bb.append(bv);ss.append(sv)
                ids.append([x['doc_id'] if id_field=='topic_id' else x['product_id'] for x in row['items']])
            maes={k:np.asarray([abs(float(v.mean()-g.mean())) for v,g in zip(vectors,gold)])
                  for k,vectors in [('original',aa),('reverse',bb),('midpoint',[(x+y)/2 for x,y in zip(aa,bb)]),('selective',ss)]}
            old_orig,old_sel=read_expected(root,name,archive)
            if abs(maes['original'].mean()-old_orig)>1e-10 or abs(maes['selective'].mean()-old_sel)>1e-10:
                raise ValueError('published metric mismatch '+name)
            if any(mid>(x+y)/2+1e-12 for mid,x,y in zip(maes['midpoint'],maes['original'],maes['reverse'])):
                raise ValueError('convexity violated '+name)
            delta=maes['selective']-maes['midpoint'];n=len(common)
            rng=np.random.default_rng(20261101+idx);sample=delta[rng.integers(0,n,size=(4000,n))].mean(axis=1)
            unique=sorted({z for row in ids for z in row});index={v:i for i,v in enumerate(unique)}
            counts=np.array([len(x) for x in ids]);offset=np.r_[0,np.cumsum(counts)[:-1]]
            ix=np.asarray([index[z] for row in ids for z in row]);g=np.concatenate(gold);a=np.concatenate(aa);b=np.concatenate(bb);s=np.concatenate(ss)
            rng=np.random.default_rng(20262101+idx);cross=[]
            for _ in range(4000):
                qdraw=rng.integers(0,n,size=n);w=rng.exponential(size=len(unique))[ix]
                den=np.add.reduceat(w,offset);rg=np.add.reduceat(w*g,offset)/den
                rm=np.add.reduceat(w*(a+b)/2,offset)/den;rs=np.add.reduceat(w*s,offset)/den
                cross.append(float(np.mean((abs(rs-rg)-abs(rm-rg))[qdraw])))
            result={'attempted':len(rows),'common':n,'completion':{'original':sum(x is not None for x in original),
                    'reverse':sum(x is not None for x in reverse),'selective':sum(x is not None for x in selected)},
                    'mean_absolute_ratio_error':{k:float(v.mean()) for k,v in maes.items()},
                    'selective_minus_midpoint':float(delta.mean()),
                    'query_bootstrap_95':[float(x) for x in np.quantile(sample,[.025,.975])],
                    'product_reweighted_range_95':[float(x) for x in np.quantile(cross,[.025,.975])],
                    'comparison_counts':{'selective_lower_error':int(sum(delta< -1e-12)),
                                         'equal_error':int(sum(abs(delta)<=1e-12)),
                                         'midpoint_lower_error':int(sum(delta>1e-12))},
                    'midpoint_cost':base,'selective_cost':scost,'virtual_selective':name=='wands96_qwen',
                    'query_seed':20261101+idx,'product_weight_seed':20262101+idx,
                    'query_id_field':id_field,'query_ids':[rows[i][id_field] for i in common]}
            results['settings'][name]=result
            print(name,'n',n,'midpoint',round(float(maes['midpoint'].mean()),4),'selective',round(float(maes['selective'].mean()),4),
                  'delta CI', [round(float(x),4) for x in np.quantile(sample,[.025,.975])],flush=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Choose a new output path; do not overwrite results')
    calculate(args.artifact,args.out)
