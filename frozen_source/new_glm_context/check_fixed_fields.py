#!/usr/bin/env python3
"""Independent CPU rejoin of all preserved complete source fields and labels."""
import collections
import importlib.util
import time
from common import *


def main():
    spec=importlib.util.spec_from_file_location('source_loader',ROOT/'source_reference/prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source,malformed=module.load_features(ROOT/'sources')
    gold=module.load_gold(ROOT/'sources')
    lookup={(r['task'],r['source_id']):r for task in TASKS for rows in source[task].values() for r in rows}
    original=read_lines(ROOT/'historical_context_arm/selection/targets.jsonl')
    primary=read_lines(ROOT/'selection/targets.jsonl')
    assert len(original)==len(primary)==256
    banks={task:{r['source_id']:r for r in read_lines(ROOT/'selection'/f'{task}_background_bank.jsonl')} for task in TASKS}
    changed=collections.Counter();feature_checks=0;empty_authors=0
    for old,new in zip(original,primary):
        for key in ('task','target_id','reserve_ordinal','selected_ordinal','target','gold'):
            assert new[key]==old[key]
        assert new['contexts']['near']==old['contexts']['near']
        changed[new['task']]+=new['contexts']['random']!=old['contexts']['random']
        for context,records in [('target',[new['target']]),*new['contexts'].items()]:
            for record in records:
                original_source=lookup[new['task'],record['source_id']]
                assert record['features']==original_source['features']
                assert digest(feature_string(record))==record['feature_sha256']
                feature_checks+=1
                if context!='target':
                    assert record['features']==banks[new['task']][record['source_id']]['features']
                if new['task']=='dblp_acm':
                    for side in ('left','right'):
                        assert tuple(record['features'][side])==FIELDS
                        empty_authors+=record['features'][side]['authors']==''
            if context!='target':
                expected=[gold[new['task'],r['source_id']] for r in records]
                assert new['background_gold'][context]==expected
        assert new['gold']==gold[new['task'],new['target_id']]
        near_ids=set().union(*(identities(r) for r in new['contexts']['near']))
        used=set()
        for record in new['contexts']['random']:
            assert not identities(record)&near_ids and not identities(record)&used
            used.update(identities(record))
    assert changed=={'dblp_acm':4,'boolq':30}
    snapshot=json.loads((ROOT/'provenance/SOURCE_SNAPSHOT.json').read_text())
    for relative,want in snapshot['snapshot_files'].items():
        if relative!='selection/targets.jsonl':assert sha(ROOT/relative)==want
    assert sha(ROOT/'historical_context_arm/selection/targets.jsonl')==snapshot['snapshot_files']['selection/targets.jsonl']
    write_json(ROOT/'verification/COMPLETE_SOURCE_FIELD_REJOIN.json',{'created_unix':time.time(),'status':'PASS',
        'all_targets':256,'changed_primary_random_targets':dict(changed),'complete_record_occurrences_rejoined':feature_checks,
        'empty_author_field_occurrences_preserved':empty_authors,'all_sources_and_original_selection_files_preserved':True,
        'original_target_file_preserved_under_historical_arm':True,'gold_used_for_verification_only_after_rematch':True})
    print('Independent complete-field source rejoin, identity exclusions and label verification PASS',flush=True)


if __name__=='__main__':main()
