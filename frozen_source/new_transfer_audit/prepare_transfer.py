#!/usr/bin/env python3
"""Full CPU plan/token preflight for new execution; never load a model or generate."""
import collections, csv, importlib.metadata as metadata, os, platform, random, subprocess, sys
from common import *
from lexical import LexicalIndex
import numpy as np

def policy_id(p):return '/'.join(p[k] for k in ('model','task','tolerance','selector'))

def plan_roles(policies,model,task):
    roles=collections.defaultdict(list)
    for p in policies:
        if (p['model'],p['task'])==(model,task):
            config=p['chosen']['configuration_id']
            roles['singleton_execution' if config=='singleton' else config].append('policy/'+policy_id(p))
    roles['k8/random/C'].append('fixed/random_k8_C')
    roles['k8/diversity/C'].append('fixed/diversity_k8_C')
    roles['singleton_execution'].append('independent_P_baseline')
    roles['singleton'].append('common_S_reference')
    return dict(roles)

def relation_partitions(rows):
    result=[];audits={};bases={}
    for task in TASKS:
        bank=read_lines(ROOT/'sources'/f'{task}_background_bank.jsonl');index=LexicalIndex(bank);audits[task]=index.audit
        for relation in sorted({r['relation_id'] for r in rows if r['task']==task}):
            records=sorted((r for r in rows if r['relation_id']==relation),key=lambda r:r['occurrence_id'])
            assert [r['occurrence_id'] for r in records]==list(range(1,129))
            matrix=index.encode_counts([index.terms(feature_string(r)) for r in records]);cosine=(matrix@matrix.T).toarray()
            # Literal original "test" salt. A relation ID must never enter these seeds.
            seed=digest(compact([SEEDS['grouping'],'random',task,'test']))
            shuffled=list(range(1,129));random.Random(int(seed,16)).shuffle(shuffled)
            bases[relation]={'seed_sha256':seed,'permutation':shuffled,'salt':'test'}
            ranks={i:digest(compact([SEEDS['grouping'],'diversity-anchor',task,'test',i])) for i in range(1,129)}
            for k in KS:
                for grouping in GROUPS:
                    if grouping=='random':batches=[shuffled[start:start+k] for start in range(0,128,k)]
                    else:
                        available=set(range(1,129));batches=[]
                        while available:
                            anchor=min(available,key=lambda i:(ranks[i],i));batch=[anchor];available.remove(anchor)
                            while available and len(batch)<k:
                                chosen=min(available,key=lambda i:(max(float(cosine[i-1,j-1]) for j in batch),i))
                                batch.append(chosen);available.remove(chosen)
                            batches.append(batch)
                    assert sorted(i for batch in batches for i in batch)==list(range(1,129)) and all(len(batch)==k for batch in batches)
                    result.append({'task':task,'relation_id':relation,'k':k,'grouping':grouping,'batches':batches,
                        'batch_membership_sha256':digest(compact(batches)),
                        'mean_within_batch_cosine':float(np.mean([cosine[i-1,j-1] for batch in batches for ix,i in enumerate(batch) for j in batch[ix+1:]]))})
    return result,audits,bases

def schedules(rows,parts,policies):
    lookup={(p['relation_id'],p['k'],p['grouping']):p for p in parts};schedules={};roles_by_model={}
    for model in MODELS:
        roles_by_model[model]={};jobs=[]
        for task in TASKS:
            roles=plan_roles(policies,model,task);roles_by_model[model][task]=roles
            for relation in sorted({r['relation_id'] for r in rows if r['task']==task}):
                for config,configroles in sorted(roles.items()):
                    if config in ('singleton','singleton_execution'):
                        k=1;grouping='none';kind='S' if config=='singleton' else 'P';batches=[[i] for i in range(1,129)];membership=None
                    else:
                        ks,grouping,kind=config.split('/');k=int(ks[1:]);part=lookup[relation,k,grouping];batches=part['batches'];membership=part['batch_membership_sha256']
                    for repeat in (1,2):
                        for ix,ids in enumerate(batches):
                            jobs.append({'phase':'transfer','model':model,'task':task,'relation_id':relation,
                                'configuration_id':config,'k':k,'grouping':grouping,'execution_type':kind,
                                'repeat':repeat,'batch_index':ix,'occurrence_ids':sorted(ids),'roles':configroles,
                                'batch_membership_sha256':membership,'request_id':f'transfer/{model}/{relation}/{config}/r{repeat}/b{ix:03d}'})
        # Fixed ordering and shuffling are documented; results never change the order.
        random.Random(2026110211).shuffle(jobs)
        for ordinal,j in enumerate(jobs):j.update(ordinal=ordinal,block=ordinal//64)
        assert len({j['request_id'] for j in jobs})==len(jobs)
        schedules[model]=jobs
    return schedules,roles_by_model

def synthetic_records(task):
    ids=[1,8,9,10,11,64,127,128];records=[]
    for ix,i in enumerate(ids):
        if task=='dblp_acm':
            a={'title':'Example database study '+str(ix),'authors':'Alice Example','venue':'Illustrative Conference','year':'2020'}
            b=a.copy() if ix%2==0 else {'title':'Distinct weather study '+str(ix),'authors':'Bob Example','venue':'Illustrative Journal','year':'2019'}
            f={'left':a,'right':b}
        else:f={'question':'Is the lamp on?','passage':'The lamp is '+('on.' if ix%2==0 else 'off.')}
        records.append({'task':task,'phase':'diagnostic','source_id':'synthetic:'+str(ix),'reserve_ordinal':None,
            'occurrence_id':i,'features':f,'feature_sha256':digest(compact(f))})
    return records

def controls(tok):
    wrapped=tok.apply_chat_template([{'role':'system','content':'Follow the user instruction exactly.'},
        {'role':'user','content':'Reply with only the word BANANA.'}],tokenize=False,add_generation_prompt=True)
    base={'wrapped_prompt':wrapped,'prompt_token_ids':tok.encode(wrapped,add_special_tokens=False),'prompt_sha256':digest(wrapped)}
    out=[{**base,'request_id':'diagnostic/ordinary_banana','name':'ordinary_banana','output_ids':[],'guided':False,'grammar':None}]
    ids=[1,8,9,10,11,64,127,128]
    for ix,spec in enumerate(([128],[9,10,127,128],[128,127,10,9],ids,list(reversed(ids)))):
        out.append({**base,'request_id':f'diagnostic/guided_banana/{ix}','name':'guided_banana',
            'output_ids':spec,'guided':True,'grammar':grammar(spec),'grammar_sha256':digest(grammar(spec))})
    for task in TASKS:
        records=synthetic_records(task)
        for k,kind in [(2,'C'),(2,'I'),(2,'O'),(4,'C'),(4,'I'),(4,'O'),(8,'C')]:
            p=prepare_prompt(tok,task,records[-k:],kind)
            out.append({**p,'request_id':f'diagnostic/smoke/{task}/k{k}/{kind}',
                'name':'synthetic_format_smoke','task':task,'execution_type':kind,'guided':True})
    assert len(out)==20
    return out

def attach_gold(rows):
    lookup={}
    with (ROOT/'sources/dblp_acm/test.csv').open(newline='',encoding='utf8') as f:
        for i,r in enumerate(csv.DictReader(f)):
            assert r['label'] in ('0','1');lookup['dblp_acm',f'test:{i:05d}:L{r["ltable_id"]}:R{r["rtable_id"]}']=int(r['label'])
    for r in read_lines(ROOT/'sources/boolq/validation.jsonl'):
        assert type(r['answer']) is bool;lookup['boolq',f'validation:{r["row_idx"]:05d}']=int(r['answer'])
    return [{k:r[k] for k in ('task','relation_id','occurrence_id','source_id')}|{'gold':lookup[r['task'],r['source_id']]} for r in rows]

def main():
    from transformers import AutoTokenizer
    rows=read_lines(ROOT/'inputs/relations.jsonl');lock=json.loads((ROOT/'POLICY_LOCK.json').read_text());policies=lock['payload']['policies']
    assert len(policies)==36 and sha(ROOT/'POLICY_LOCK.json')==sha(ROOT/'source_reference/decision_POLICY_LOCK.json')
    parts,audits,bases=relation_partitions(rows);jobs,roles=schedules(rows,parts,policies)
    write_lines(ROOT/'inputs/partitions.jsonl',parts);write_json(ROOT/'inputs/LEXICAL_INDEX_AUDIT.json',audits)
    write_json(ROOT/'inputs/RANDOM_BASE_PERMUTATIONS.json',bases);write_json(ROOT/'prepared/ROLE_PLAN_MAP.json',roles)
    feature_plan=digest(compact({'relations':rows,'partitions':parts,'schedules':jobs}))
    write_json(ROOT/'inputs/FEATURE_ONLY_PLAN_HASH.json',{'sha256':feature_plan,'gold_used_for_plan_construction':False})
    # Gold attached only after all feature-only relation/partition/schedule choices.
    write_lines(ROOT/'inputs/gold.jsonl',attach_gold(rows))
    assert feature_plan==digest(compact({'relations':rows,'partitions':parts,'schedules':jobs}))
    budget={'new_replication':True,'old_expected_scientific_calls':28800,'old_expected_controls':40,'by_model':{}}
    for model in MODELS:
        current=jobs[model];budget['by_model'][model]={'science_calls':len(current),'control_calls':20,
            'requested_item_judgments':sum(len(j['occurrence_ids']) for j in current),'by_task':{}}
        for task in TASKS:
            subset=[j for j in current if j['task']==task];num=len({j['relation_id'] for j in subset})
            budget['by_model'][model]['by_task'][task]={'relations':num,'science_calls':len(subset),
                'calls_per_relation':len(subset)//num if num else None,'physical_plan_count':len(roles[model][task]),
                'calls_P':sum(j['execution_type']=='P' for j in subset),'calls_S':sum(j['execution_type']=='S' for j in subset)}
        write_lines(ROOT/'prepared'/f'{model}_schedule.jsonl',current)
        write_json(ROOT/'prepared'/f'{model}_block_membership.json',[[j['request_id'] for j in current[start:start+64]] for start in range(0,len(current),64)])
    budget.update(science_calls=sum(len(v) for v in jobs.values()),control_calls=40)
    write_json(ROOT/'prepared/BUDGET.json',budget)
    locations=json.loads((ROOT/'model_locations.json').read_text());lookup={(r['relation_id'],r['occurrence_id']):r for r in rows};preflight={}
    for model in MODELS:
        tok=AutoTokenizer.from_pretrained(locations[model],local_files_only=True,trust_remote_code=False)
        path=ROOT/'prepared'/f'{model}_requests.jsonl';seen_prompt={};max_tokens=0;max_output=0;total_input=0
        with path.open('x',encoding='utf8') as f:
            for j in jobs[model]:
                records=[lookup[j['relation_id'],i] for i in j['occurrence_ids']]
                p=prepare_prompt(tok,j['task'],records,j['execution_type'])
                p['id_map']={k:{**v,'relation_id':j['relation_id']} for k,v in p['id_map'].items()}
                p['id_map_sha256']=digest(compact(p['id_map']))
                assert p['prompt_tokens_preflight']+128<=4096
                longest=compact({str(i):'Y' for i in p['output_ids']});assert len(longest)+1<=128
                sizes=[len(tok.encode(compact({str(i):v for i in p['output_ids']}),add_special_tokens=False)) for v in ('Y','N')]
                max_output=max(max_output,*sizes);max_tokens=max(max_tokens,p['prompt_tokens_preflight']);total_input+=p['prompt_tokens_preflight']
                # Verify independent P/S replays really have identical frozen prompt, map and grammar.
                if j['execution_type'] in ('P','S'):
                    key=(j['relation_id'],j['occurrence_ids'][0]);contract=(p['prompt_sha256'],p['grammar_sha256'],p['id_map_sha256'])
                    if key in seen_prompt:assert seen_prompt[key]==contract
                    seen_prompt[key]=contract
                f.write(compact({**j,**p,'canonical_output_token_lengths_allY_allN':sizes})+'\n')
        ctrl=controls(tok)
        for p in ctrl:assert len(p['prompt_token_ids'])+128<=4096
        write_lines(ROOT/'prepared'/f'{model}_controls.jsonl',ctrl)
        preflight[model]={'science_calls':len(jobs[model]),'controls':len(ctrl),'max_prompt_tokens':max_tokens,
            'max_prompt_plus_128':max_tokens+128,'max_canonical_output_tokens':max_output,'preflight_science_input_tokens':total_input,
            'preflight_control_input_tokens':sum(len(p['prompt_token_ids']) for p in ctrl),
            'output_token_cap':128,'requests_sha256':sha(path),'P1_P2_S1_S2_prompt_map_grammar_identity':'PASS',
            'physical_calls_remain_independent_despite_identical_prompts':True}
        print(model,compact(preflight[model]),flush=True)
    write_json(ROOT/'prepared/BOTH_MODEL_CONTEXT_PREFLIGHT.json',preflight)
    # Verify restored model/tokenizer bytes against the original captured manifest.
    old=json.loads((DECISION_MANIFEST:=ROOT.parent/'reconstruction/decision_audit/prepared/MODEL_FILE_SHA256.json').read_text())
    checked={}
    for model in MODELS:
        checks={name:sha(Path(locations[model])/name) for name in old[model]['files']}
        assert checks==old[model]['files'];checked[model]={'path':locations[model],'files':checks}
    write_json(ROOT/'prepared/MODEL_FILE_SHA256.json',checked)
    write_json(ROOT/'provenance/MODEL_MANIFEST_ORIGIN.json',{'path':str(DECISION_MANIFEST),'sha256':sha(DECISION_MANIFEST),'all_restored_bytes_match':True})
    deps=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (ROOT/'prepared/pip_check.log').write_text(deps.stdout+deps.stderr)
    config={'python':platform.python_version(),'python_executable':sys.executable,'packages':{p:metadata.version(p) for p in PACKAGES},
        'pip_check_exit':deps.returncode,'environment_status':'inherited environment; conflicts retained; fresh experiment-local persistent cache',
        'dtype':'bfloat16','temperature':0,'max_model_len':4096,'max_tokens':128,'max_num_seqs':16,'tensor_parallel_size':1,
        'enforce_eager':True,'prefix_caching':False,'guided_backend':'xgrammar','disable_guidance_fallback':True,
        'gpu_memory_utilization':.70,'block_requests':64,'groupingseed':2026110207,'grouping_salt':'test','transfer_hashseed':2026110210,
        'schedule_seed':2026110211,'FLASHINFER_WORKSPACE_BASE':str(ROOT/'runtime_cache/flashinfer_base')}
    write_json(ROOT/'prepared/runtime_configuration.json',config)
    write_json(ROOT/'inputs/SOURCE_FILE_SHA256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'sources').rglob('*')) if p.is_file()})
    write_json(ROOT/'prepared/PARSER_CHECKS.json',{'status':'PASS','checks':parser_checks()})
    print('Full prepared new replication. No model loaded and no inference performed.',flush=True)

if __name__=='__main__':main()
