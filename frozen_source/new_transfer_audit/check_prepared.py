#!/usr/bin/env python3
"""Full prepared integrity check independent of tokenizer/model execution."""
import collections
from common import *
from identity_selection import identities,fingerprint,records_in_prompt

def verify():
    rows=read_lines(ROOT/'inputs/relations.jsonl');lookup={(r['relation_id'],r['occurrence_id']):r for r in rows}
    exposed=read_lines(ROOT/'inputs/historical_exposed_records.jsonl');known={t:set() for t in TASKS};seen={t:set() for t in TASKS};knownids={t:set() for t in TASKS}
    for r in exposed:
        known[r['task']].update(tuple(x) for x in r['identities'])
        if r['source_id']:knownids[r['task']].add(r['source_id'])
    for r in rows:
        t=r['task'];ids=identities(r)
        assert r['source_id'] not in knownids[t] and not ids&known[t] and not ids&seen[t]
        assert digest(feature_string(r))==r['feature_sha256'];seen[t].update(ids)
    assert len(rows)==23*128 and len(lookup)==len(rows)
    relationids={t:sorted({r['relation_id'] for r in rows if r['task']==t}) for t in TASKS}
    assert len(relationids['dblp_acm'])==3 and len(relationids['boolq'])==20
    parts=read_lines(ROOT/'inputs/partitions.jsonl')
    for p in parts:
        assert sorted(i for batch in p['batches'] for i in batch)==list(range(1,129))
        assert all(len(b)==p['k'] for b in p['batches']) and digest(compact(p['batches']))==p['batch_membership_sha256']
    budget=json.loads((ROOT/'prepared/BUDGET.json').read_text());preflight=json.loads((ROOT/'prepared/BOTH_MODEL_CONTEXT_PREFLIGHT.json').read_text())
    counts={};permutations=json.loads((ROOT/'inputs/RANDOM_BASE_PERMUTATIONS.json').read_text())
    for task in TASKS:
        expected=list(range(1,129));import random
        random.Random(int(digest(compact([2026110207,'random',task,'test'])),16)).shuffle(expected)
        assert all(permutations[r]['permutation']==expected for r in relationids[task])
    for model in MODELS:
        schedule=read_lines(ROOT/'prepared'/f'{model}_schedule.jsonl');jobs={j['request_id']:j for j in schedule}
        assert len(jobs)==budget['by_model'][model]['science_calls'];coverage=collections.defaultdict(list);p_science={}
        for index,j in enumerate(schedule):
            assert j['ordinal']==index and j['block']==index//64
            coverage[(j['relation_id'],j['configuration_id'],j['repeat'])].extend(j['occurrence_ids'])
        for key,ids in coverage.items():assert sorted(ids)==list(range(1,129))
        for relation in lookup:
            rel,i=relation
            assert (rel,'singleton_execution',1) in coverage and (rel,'singleton_execution',2) in coverage and (rel,'singleton',1) in coverage and (rel,'singleton',2) in coverage
        seen=set();summed=0;maximum=0
        with (ROOT/'prepared'/f'{model}_requests.jsonl').open() as f:
            for index,line in enumerate(f):
                r=json.loads(line);j=schedule[index];assert all(r[k]==v for k,v in j.items());seen.add(r['request_id'])
                rendered=render(j['task'],[lookup[j['relation_id'],i] for i in j['occurrence_ids']],j['execution_type'])
                assert all(r[k]==v for k,v in rendered.items())
                assert r['prompt_sha256']==digest(r['wrapped_prompt']) and r['grammar']==grammar(r['output_ids'])
                assert r['grammar_sha256']==digest(r['grammar']) and r['id_map_sha256']==digest(compact(r['id_map']))
                expected_ids=sorted(j['occurrence_ids']);expected_out=list(reversed(expected_ids)) if j['execution_type']=='O' else expected_ids
                assert r['requested_ids']==expected_ids and r['output_ids']==expected_out
                extracted=list(records_in_prompt(r['unwrapped_user']));expected_order=list(reversed(expected_ids)) if j['execution_type']=='I' else expected_ids
                assert [x['features'] for x in extracted]==[lookup[j['relation_id'],i]['features'] for i in expected_order]
                for i in expected_ids:
                    m=r['id_map'][str(i)];assert m['source_id']==lookup[j['relation_id'],i]['source_id'] and m['relation_id']==j['relation_id']
                assert len(r['prompt_token_ids'])==r['prompt_tokens_preflight'] and len(r['prompt_token_ids'])+128<=4096
                summed+=len(r['prompt_token_ids']);maximum=max(maximum,len(r['prompt_token_ids']))
                if j['execution_type'] in ('P','S'):
                    k=(j['relation_id'],expected_ids[0]);sig=(r['prompt_sha256'],r['grammar_sha256'],r['id_map_sha256'])
                    if k in p_science:assert p_science[k]==sig
                    p_science[k]=sig
        assert seen==set(jobs) and summed==preflight[model]['preflight_science_input_tokens'] and maximum==preflight[model]['max_prompt_tokens']
        assert sha(ROOT/'prepared'/f'{model}_requests.jsonl')==preflight[model]['requests_sha256']
        controls=read_lines(ROOT/'prepared'/f'{model}_controls.jsonl');assert len(controls)==20
        counts[model]={'calls':len(jobs),'requested_item_judgments':sum(len(j['occurrence_ids']) for j in schedule),'preflight_input_tokens':summed,'controls':20}
    return {'status':'PASS','relation_records':len(rows),'relation_count':23,'models':counts,
        'producer_sha256':sha(ROOT/'check_prepared.py'),
        'P1_P2_S1_S2_separate_physical_request_ids':True,'historical_and_within_panel_identity_disjoint':True,
        'all_relations_have_exact_128_target_coverage_per_plan_replay':True,'scientific_calls':sum(x['calls'] for x in counts.values()),'control_calls':40}

if __name__=='__main__':
    result=verify();write_json(ROOT/'verification/CPU_PREPARED_CHECK.json',result);print(compact(result))
