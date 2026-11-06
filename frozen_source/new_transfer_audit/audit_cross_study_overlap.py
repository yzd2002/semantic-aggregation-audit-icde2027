#!/usr/bin/env python3
"""Feature-only overlap disclosure; never changes the frozen-design membership."""
from collections import Counter
from common import *
from identity_selection import identities,fingerprint

def audit():
    fresh_path=ROOT.parent/'new_glm_context/selection/targets.jsonl'
    old_path=ROOT.parent/'recovered/context_audit/selection/targets.jsonl'
    before=sha(fresh_path);fresh=read_lines(fresh_path);old={(r['task'],r['target_id']):r for r in read_lines(old_path)}
    changed_slots=[];changed_groups=0;all_new=[]
    for row in fresh:
        key=(row['task'],row['target_id']);old_records=old[key]['contexts']['random'];now=row['contexts']['random']
        changed=any((a['source_id'],fingerprint(a))!=(b['source_id'],fingerprint(b)) for a,b in zip(old_records,now))
        if changed:
            changed_groups+=1
            # "34 new random backgrounds" denotes redrawn three-record contexts.
            for ix,r in enumerate(now):all_new.append({'context_target_id':row['target_id'],'context_task':row['task'],'slot':ix,'record':r})
        for ix,(a,b) in enumerate(zip(old_records,now)):
            if (a['source_id'],fingerprint(a))!=(b['source_id'],fingerprint(b)):
                changed_slots.append({'context_target_id':row['target_id'],'context_task':row['task'],'slot':ix,'record':b})
    assert before==sha(fresh_path),'GLM selection changed while read; audit must use a stable snapshot'
    panel=read_lines(ROOT/'inputs/relations.jsonl')
    def intersections(backgrounds):
        pairs=[];touched=set();identity_types=Counter()
        for p in panel:
            for b in backgrounds:
                r=b['record']
                if p['task']!=r['task']:continue
                overlap=identities(p)&identities(r);same_id=p['source_id']==r['source_id'];same_complete=fingerprint(p)==fingerprint(r)
                if overlap or same_id or same_complete:
                    touched.add((p['task'],p['source_id']));identity_types.update(k for k,v in overlap)
                    pairs.append({'task':p['task'],'panel_relation_id':p['relation_id'],'panel_source_id':p['source_id'],
                        'glm_context_target_id':b['context_target_id'],'glm_background_source_id':r['source_id'],'slot':b['slot'],
                        'same_source_id':same_id,'same_complete_normalized_features':same_complete,'overlapping_identities':sorted(overlap)})
        return {'unique_panel_records_with_any_overlap':len(touched),'pair_count_including_reused_background_slots':len(pairs),
            'overlapping_identity_type_counts':dict(identity_types),'pairs':pairs}
    return {'status':'feature-only disclosure, membership unchanged','new_glm_selection_path':str(fresh_path),'new_glm_selection_sha256':before,
        'old_context_selection_sha256':sha(old_path),'new_transfer_relations_sha256':sha(ROOT/'inputs/relations.jsonl'),
        'redrawn_random_context_count':changed_groups,'changed_random_record_slot_count':len(changed_slots),
        'random_record_slots_in_all_redrawn_contexts':len(all_new),'changed_slots_overlap':intersections(changed_slots),
        'all_slots_in_redrawn_contexts_overlap':intersections(all_new),
        'exclusion_scope':'Original historical context/decision science and structured controls only; new GLM/SNLI research inputs were not added to the primary exclusion set.',
        'independence_limit':'Two new studies use shared captured sources. Any disclosed background/panel identity intersections prevent treating their record samples as disjoint independent samples.',
        'model_outputs_read':False,'gold_used':False,'membership_adjusted':False}

if __name__=='__main__':
    result=audit();write_json(ROOT/'prepared/CROSS_STUDY_FEATURE_OVERLAP.json',result)
    print(compact({k:result[k] for k in ('redrawn_random_context_count','changed_random_record_slot_count','random_record_slots_in_all_redrawn_contexts')}))
    print(compact({k:result[k]['unique_panel_records_with_any_overlap'] for k in ('changed_slots_overlap','all_slots_in_redrawn_contexts_overlap')}))
