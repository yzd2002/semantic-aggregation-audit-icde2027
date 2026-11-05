#!/usr/bin/env python3
"""Independent read-only check of the prompt/order manipulation and token preflight."""
import argparse,hashlib,json,re
from common import ROOT,MODELS,SYSTEMS,read,write_json,sha

def digest(value):return hashlib.sha256(value.encode('utf8')).hexdigest()

def main(model,retokenize):
    pool={(dataset,row['query_id']):row for dataset in ('esci','wands') for row in read(ROOT/'selection'/f'{dataset}_inputs.jsonl')}
    requests=read(ROOT/'prepared'/f'{model}_requests.jsonl');schedule=read(ROOT/'selection/schedule.jsonl')
    groups={};seen_tokens={};maximum=0
    if retokenize:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(MODELS[model],local_files_only=True,trust_remote_code=False)
    for request,job in zip(requests,schedule):
        assert all(request[key]==value for key,value in job.items())
        ids=job['ids'];row=pool[job['dataset'],job['query_id']]
        assert ids==list(range(job['start']+1,job['start']+job['size']+1))
        displayed=ids[::-1] if job['condition'].startswith('R') else ids
        output=sorted(ids,reverse=job['condition'].endswith('D'))
        prefix='Query: '+row['query']+'\nProducts in order:\n'
        suffix='\nReturn only a JSON object mapping each given product id to Y or N for that product. Use the given ids, not positions; include every id exactly once. Either key order is allowed. Do not add spaces or explanations.'
        expected=prefix+'\n'.join(json.dumps({'id':i,'text':row['items'][i-1]['text']},ensure_ascii=False) for i in displayed)+suffix
        assert request['user_prompt']==expected
        assert request['user_prompt_sha256']==digest(expected)
        wrapped='<|im_start|>system\n'+SYSTEMS[job['dataset']]+'<|im_end|>\n<|im_start|>user\n'+expected+'<|im_end|>\n<|im_start|>assistant\n'
        assert request['wrapped_prompt']==wrapped and request['prompt_sha256']==digest(wrapped)
        rule=r'\{'+','.join('"'+str(i)+'":"[YN]"' for i in output)+r'\}'
        assert request['grammar']==rule and request['grammar_sha256']==digest(rule)
        for label in ('Y','N'):
            valid='{'+','.join('"'+str(i)+'":"'+label+'"' for i in output)+'}'
            assert re.fullmatch(rule,valid) and not re.fullmatch(rule,' '+valid)
        mapping=request['id_map'];assert mapping['presented_ids']==displayed and mapping['output_ids']==output
        assert set(mapping['occurrences'])==set(map(str,ids))
        for i in ids:
            assert row['items'][i-1]['id']==i
            assert mapping['occurrences'][str(i)]=={'pool_index':i-1,'product_id':row['items'][i-1].get('product_id'),'example_id':row['items'][i-1].get('example_id')}
        assert request['id_map_sha256']==digest(json.dumps(mapping,sort_keys=True,separators=(',',':')))
        group=(job['dataset'],job['query_id'],job['start'],job['size'],'S' if job['condition']=='S' else job['condition'][0])
        signature=(wrapped,request['prompt_token_ids'])
        if group in groups:assert groups[group]==signature
        else:groups[group]=signature
        tokens=request['prompt_token_ids'];assert len(tokens)==request['prompt_tokens_preflight']
        maximum=max(maximum,len(tokens)+128);assert len(tokens)+128<=4096
        if retokenize:
            if wrapped not in seen_tokens:seen_tokens[wrapped]=tok.encode(wrapped,add_special_tokens=False)
            assert tokens==seen_tokens[wrapped]
    assert len(requests)==len(schedule)==11894
    recorded=json.loads((ROOT/'prepared/BOTH_MODEL_CONTEXT_PREFLIGHT.json').read_text())[model]
    assert recorded['max_prompt_plus_cap']==maximum
    result={'model':model,'status':'PASS','requests':len(requests),'distinct_prompt_groups':len(groups),
        'prompt_unchanged_across_output_orders_and_replays':True,'numeric_original_pool_ids':True,
        'all_prompt_grammar_idmap_hashes_recomputed':True,'exact_template_and_original_candidate_texts':True,
        'max_prompt_plus_128_cap':maximum,'retokenized_from_local_model_assets':retokenize,
        'request_file_sha256':sha(ROOT/'prepared'/f'{model}_requests.jsonl')}
    write_json(ROOT/f'run_{model}/PREPARED_REQUESTS_AUDIT.json',result)
    print('PASS independent prepared-prompt, row-order, grammar, ID, hash and context checks:',model,'retokenized',retokenize)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol'],required=True);p.add_argument('--retokenize',action='store_true');a=p.parse_args();main(a.model,a.retokenize)
