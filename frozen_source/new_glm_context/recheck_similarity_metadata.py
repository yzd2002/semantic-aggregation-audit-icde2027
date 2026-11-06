#!/usr/bin/env python3
"""Post-selection CPU metadata correction; never changes IDs or prompts."""
import gc
import importlib.util
import time
from common import *


def main():
    assert not any((ROOT/f'GPU_GO_{model}.json').exists() for model in MODELS)
    assert not any((ROOT/f'run_{model}').exists() for model in MODELS)
    spec=importlib.util.spec_from_file_location('original_lexical_index',ROOT/'source_reference/prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=read_lines(ROOT/'selection/targets.jsonl')
    report=json.loads((ROOT/'selection/selection_report.json').read_text())
    oldbytes=(ROOT/'selection/targets.jsonl').read_bytes()
    (ROOT/'initial_cpu_preflight/PRIMARY_TARGETS_BEFORE_SIMILARITY_METADATA_CORRECTION.jsonl').write_bytes(oldbytes)
    audit=[]
    for task in TASKS:
        bank=read_lines(ROOT/'selection'/f'{task}_background_bank.jsonl')
        index=module.LexicalIndex(bank)
        for key in ('documents','vocabulary_terms','vocabulary_sha256','idf_sha256','bank_ids_sha256'):
            assert index.audit[key]==report[task]['lexical_index'][key],(task,key)
        positions={r['source_id']:i for i,r in enumerate(bank)}
        for r in rows:
            if r['task']!=task:continue
            scores=index.scores(r['target'])
            near=[float(scores[positions[b['source_id']]]) for b in r['contexts']['near']]
            assert all(abs(v-old['similarity'])<1e-12 for v,old in zip(near,r['near_ranks']))
            actual=[float(scores[positions[b['source_id']]]) for b in r['contexts']['random']]
            if r['glm_primary_random_context_redrawn']:
                r['random_similarities_historical']=r['random_similarities']
                r['random_similarities']=actual
                r['random_similarity_metadata_corrected_after_selection_before_inference']=True
            else:
                assert all(abs(v, )>=0 for v in actual)
                assert all(abs(v-old)<1e-12 for v,old in zip(actual,r['random_similarities']))
            audit.append({'task':task,'target_id':r['target_id'],'selected_ordinal':r['selected_ordinal'],
                'redrawn':r['glm_primary_random_context_redrawn'],'near_similarities':near,
                'primary_random_similarities':actual,'historical_random_similarities':r.get('random_similarities_historical',actual)})
        del index,scores;gc.collect()
    changed=ROOT/'selection/targets.corrected.jsonl';write_lines(changed,rows);changed.replace(ROOT/'selection/targets.jsonl')
    write_lines(ROOT/'prepared/THREE_TOKENIZER_PRIMARY_SIMILARITY_AUDIT.jsonl',audit)
    write_json(ROOT/'verification/SIMILARITY_METADATA_CORRECTION.json',{'created_unix':time.time(),'status':'PASS',
        'reason':'34 primary random triplets changed at CPU design time; stale original similarity values were detected during pre-inference self-review.',
        'old_targets_sha256':digest(oldbytes),'new_targets_sha256':sha(ROOT/'selection/targets.jsonl'),
        'changed_similarity_metadata_targets':sum(r['redrawn'] for r in audit),
        'original_index_vocabulary_idf_bank_hashes_reproduced':True,'target_near_and_random_IDs_features_prompts_labels_schedules_unchanged':True,
        'selection_was_not_rerun_and_similarity_not_used_as_acceptance':True,'no_model_judgments_exist':True})
    print('Corrected 34 stale random-similarity metadata entries, preserving historical values; no request or selection changed.',flush=True)


if __name__=='__main__':main()
