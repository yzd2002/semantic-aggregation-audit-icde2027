#!/usr/bin/env python3
"""Feature-only reconstruction of the fixed-design transfer panel; CPU only."""
import collections, csv, json, re
from common import *

BASE=ROOT.parent
DECISION=BASE/'reconstruction/decision_audit'
CONTEXT=BASE/'recovered/context_audit'
FIELDS=('title','authors','venue','year')
TRANSFER_HASHSEED=2026110210

def normalize(s):
    return ' '.join(s.casefold().split())

def citation_fingerprint(c):
    return digest(compact([normalize(c[k]) for k in FIELDS]))

def fingerprint(r):
    f=r['features']
    ordered=([[normalize(f[s][k]) for k in FIELDS] for s in ('left','right')]
             if r['task']=='dblp_acm' else [normalize(f[k]) for k in ('question','passage')])
    return digest(compact(ordered))

def identities(r):
    if r['task']=='dblp_acm':
        result={('citation',citation_fingerprint(r['features'][s])) for s in ('left','right')}
        for field in ('left_id','right_id'):
            if field in r: result.add((field,r[field]))
        return result
    return {(k,normalize(r['features'][k])) for k in ('question','passage')}

def source_features(source):
    """Never expose labels in returned records. Exact original feature field order."""
    tables=[]; malformed=[]; result={t:[] for t in TASKS}
    for name in ('tableA.csv','tableB.csv'):
        with (source/'dblp_acm'/name).open(newline='',encoding='utf8') as f:
            rows=list(csv.DictReader(f))
        assert len({r['id'] for r in rows})==len(rows)
        tables.append({r['id']:{k:r[k] for k in FIELDS} for r in rows})
    with (source/'dblp_acm/test.csv').open(newline='',encoding='utf8') as f:
        for index,row in enumerate(csv.DictReader(f)):
            a,b=row['ltable_id'],row['rtable_id']; sid=f'test:{index:05d}:L{a}:R{b}'
            if a not in tables[0] or b not in tables[1]:
                malformed.append({'task':'dblp_acm','source_id':sid,'reason':'nonjoinable'});continue
            if not tables[0][a]['title'].strip() or not tables[1][b]['title'].strip():
                malformed.append({'task':'dblp_acm','source_id':sid,'reason':'empty_title'});continue
            result['dblp_acm'].append({'task':'dblp_acm','source_split':'test','source_index':index,
                'source_id':sid,'left_id':a,'right_id':b,'features':{'left':tables[0][a],'right':tables[1][b]}})
    for index,r in enumerate(read_lines(source/'boolq/validation.jsonl')):
        assert index==r['row_idx']; sid=f'validation:{index:05d}'
        if any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('question','passage')):
            malformed.append({'task':'boolq','source_id':sid,'reason':'malformed_or_empty_question_passage'});continue
        result['boolq'].append({'task':'boolq','source_split':'validation','source_index':index,
            'source_id':sid,'features':{k:r[k] for k in ('question','passage')}})
    return result,malformed

def records_in_prompt(text):
    """Read complete JSON record lines, never substring-match citation fields."""
    for line in text.splitlines():
        if not line.startswith('{'): continue
        try: r=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(r,dict) and isinstance(r.get('features'),dict):
            f=r['features']; task='dblp_acm' if set(f)=={'left','right'} else 'boolq' if set(f)=={'question','passage'} else None
            if task is None: raise ValueError('Unrecognized structured record in retained prompt')
            yield {'task':task,'features':f}

def historical_exposure():
    known_ids={t:set() for t in TASKS}; known_fp={t:set() for t in TASKS}; known_identity={t:set() for t in TASKS}
    exposures=[]; manifests=[]; seen=set()
    def add(r,origin,prompt_id=None):
        t=r['task']; known_fp[t].add(fingerprint(r)); known_identity[t].update(identities(r))
        if r.get('source_id'): known_ids[t].add(r['source_id'])
        key=(origin,prompt_id,t,r.get('source_id'),fingerprint(r))
        if key in seen:return
        seen.add(key);exposures.append({'origin':origin,'prompt_id':prompt_id,'task':t,'source_id':r.get('source_id'),
            'features':r['features'],'normalized_full_feature_sha256':fingerprint(r),
            'identities':sorted(identities(r))})
    # The context targets include every actual contextual background, not the unused train bank.
    p=CONTEXT/'selection/targets.jsonl'; manifests.append({'path':str(p),'sha256':sha(p),'role':'science inputs'})
    for row in read_lines(p):
        add(row['target'],str(p),row['target_id'])
        for records in row['contexts'].values():
            for r in records:add(r,str(p),row['target_id'])
    # Every decision dev/test record is requested, including independent P and S.
    p=DECISION/'inputs/relations.jsonl';manifests.append({'path':str(p),'sha256':sha(p),'role':'science inputs'})
    for r in read_lines(p):
        r=dict(r)
        if r['task']=='dblp_acm':
            m=re.fullmatch(r'(?:test|train|valid):\d+:L(.+):R(.+)',r['source_id'])
            if not m:raise ValueError('Unknown DBLP source ID')
            r['left_id'],r['right_id']=m.groups()
        add(r,str(p),r['phase'])
    # Verify all retained prepared decision scientific prompts, including historical plans.
    paths=sorted((DECISION/'prepared').glob('*_requests.jsonl'))
    paths+=sorted((DECISION/'protocol_history').rglob('*_requests.jsonl'))
    # Context prepared token files were lost; selection contains the exact feature payloads.
    for p in paths:
        count=0
        with p.open() as f:
            for line in f:
                row=json.loads(line); prompt=row.get('unwrapped_user',row.get('wrapped_prompt',''))
                for r in records_in_prompt(prompt):
                    if fingerprint(r) not in known_fp[r['task']]:raise ValueError('Unmapped retained scientific record')
                    count+=1
        manifests.append({'path':str(p),'sha256':sha(p),'role':'science prompt verification','record_occurrences':count})
    # All structured controls in both original models and all decision phases.
    paths=sorted(CONTEXT.glob('run_*/diagnostics.jsonl'))+sorted(DECISION.glob('run_*/diagnostics.jsonl'))
    for p in paths:
        count=0
        for row in read_lines(p):
            prompt=row.get('wrapped_prompt','')
            for r in records_in_prompt(prompt):add(r,str(p),row['request_id']);count+=1
        manifests.append({'path':str(p),'sha256':sha(p),'role':'structured controls','record_occurrences':count})
    return known_ids,known_fp,known_identity,exposures,manifests

def select_records(data,lengths,known_ids,known_fp,known_identity):
    eligible={};rejects=[];report={}
    for task in TASKS:
        selected=[];seen=set();counts=collections.Counter()
        for r in sorted(data[task],key=lambda r:r['source_id']):
            key=(task,r['source_id']); ls=lengths[key]; reasons=[]
            if any(n>512 for n in ls.values()):reasons.append('feature_over_512_tokens')
            if r['source_id'] in known_ids[task]:reasons.append('known_source_id')
            if fingerprint(r) in known_fp[task]:reasons.append('known_normalized_complete_features')
            overlap=identities(r)&known_identity[task]
            if overlap:reasons.append('known_identity_overlap')
            # Greedy uniqueness is after the complete historical and length exclusions.
            if not reasons and identities(r)&seen:reasons.append('source_id_order_greedy_identity_overlap')
            if reasons:
                counts.update(reasons);rejects.append({'task':task,'source_id':r['source_id'],'reasons':reasons,
                    'known_overlapping_identities':sorted(overlap),'token_lengths':ls});continue
            r={**r,'feature_sha256':digest(feature_string(r)),'normalized_feature_sha256':fingerprint(r),'token_lengths':ls}
            selected.append(r);seen.update(identities(r))
        eligible[task]=selected
        ranked=sorted(selected,key=lambda r:(digest(compact([TRANSFER_HASHSEED,task,r['source_id']])),r['source_id']))
        count=min(20,len(ranked)//128); kept=ranked[:count*128]
        report[task]={'source_records':len(data[task]),'eligible_after_source_id_order_greedy':len(selected),
            'known_source_ids':len(known_ids[task]),'known_full_feature_fingerprints':len(known_fp[task]),
            'known_identity_count':len(known_identity[task]),'exclusion_reason_counts_nonexclusive':dict(counts),
            'relations':count,'selected_records':len(kept),'unused_eligible':len(selected)-len(kept),
            'hash_order_rule':'sha256(compact([2026110210,task,source_id])), then source_id',
            'eligible_sha256':digest(compact([r['source_id'] for r in selected])),
            'ranked_source_ids':[r['source_id'] for r in ranked]}
    return eligible,rejects,report

def main():
    from transformers import AutoTokenizer
    data,malformed=source_features(ROOT/'sources')
    loc=json.loads((ROOT/'model_locations.json').read_text()); lengths={}
    records=[r for t in TASKS for r in data[t]]
    for model in MODELS:
        tok=AutoTokenizer.from_pretrained(loc[model],local_files_only=True,trust_remote_code=False)
        for offset in range(0,len(records),256):
            batch=records[offset:offset+256]
            out=tok([feature_string(r) for r in batch],add_special_tokens=False,truncation=False,padding=False)['input_ids']
            for r,ids in zip(batch,out):lengths.setdefault((r['task'],r['source_id']),{})[model]=len(ids)
    ki,kf,kn,exposures,manifest=historical_exposure()
    eligible,rejects,report=select_records(data,lengths,ki,kf,kn)
    rows=[]
    for task in TASKS:
        ranked=sorted(eligible[task],key=lambda r:(digest(compact([TRANSFER_HASHSEED,task,r['source_id']])),r['source_id']))
        for index,r in enumerate(ranked[:report[task]['relations']*128]):
            relation_id=f'{task}/{index//128:02d}'
            rows.append({**r,'phase':'transfer','relation_id':relation_id,'relation_index':index//128,
                'occurrence_id':index%128+1,'reserve_ordinal':None})
    write_lines(ROOT/'inputs/relations.jsonl',rows)
    write_lines(ROOT/'inputs/historical_exposed_records.jsonl',exposures)
    write_lines(ROOT/'inputs/exclusions.jsonl',rejects)
    write_lines(ROOT/'inputs/eligible_records.jsonl',[r for task in TASKS for r in eligible[task]])
    write_lines(ROOT/'inputs/all_feature_lengths.jsonl',[{'task':t,'source_id':sid,'lengths':v} for (t,sid),v in sorted(lengths.items())])
    write_json(ROOT/'provenance/HISTORICAL_EXPOSURE_MANIFEST.json',{'files':manifest,
        'context_complete_prepared_prompt_files_available':False,'context_scientific_features_reconstructed_from_exact_retained_targets':True})
    write_json(ROOT/'prepared/SOURCE_IDENTITY_EXCLUSION.json',{'status':'CPU-only new real replication selection',
        'historical_expectation':{'dblp_acm':{'eligible':420,'relations':3},'boolq':{'eligible':2571,'relations':20}},
        'matches_old_eligible_counts':all(report[t]['eligible_after_source_id_order_greedy']==e for t,e in [('dblp_acm',420),('boolq',2571)]),
        'old_membership_recovered':False,'new_selection_rule_declared_before_new_inference':True,
        'hashseed':TRANSFER_HASHSEED,'tasks':report,'malformed':malformed,
        'gold_used_for_selection':False,'relations_sha256':sha(ROOT/'inputs/relations.jsonl')})
    print(compact({t:{k:report[t][k] for k in ('source_records','eligible_after_source_id_order_greedy','relations','selected_records')} for t in TASKS}),flush=True)

if __name__=='__main__':main()
