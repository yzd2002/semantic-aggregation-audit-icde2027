#!/usr/bin/env python3
"""CPU-only Qwen tokenizer preflight; never loads model weights or runs inference."""
import argparse,collections,hashlib,importlib.metadata,json
from pathlib import Path
from acord_prompts import SYSTEM,RETRY_SUFFIX,keyed_prompt,point_prompt

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(selection,model,out):
    if out.exists():raise FileExistsError('Choose a new output directory')
    if model.name!='Qwen3-4B-Instruct-2507':raise ValueError('Unexpected tokenizer path')
    meta=json.loads((selection/'selection.json').read_text())
    if sha(selection/'inputs.jsonl')!=meta['inputs_sha256']:raise ValueError('Selection SHA mismatch')
    with (selection/'inputs.jsonl').open(encoding='utf8') as f:rows=[json.loads(x) for x in f if x.strip()]
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    def tokens(prompt):return len(tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True))
    records=[];overflow=set();maxima=collections.defaultdict(int)
    for row in rows:
        tasks=[]
        for start in (0,8,16,24):
            for mode,reverse in (('keyed_original',False),('keyed_reverse',True)):
                text,presented=keyed_prompt(row,start,reverse);tasks.append((mode,start,text,128,presented))
        for index in range(32):tasks.append(('pointwise',index,point_prompt(row,index),4,None))
        for mode,start,text,cap,presented in tasks:
            length=tokens(text);with_retry=tokens(text+RETRY_SUFFIX);worst=max(length,with_retry)+cap
            maxima[mode]=max(maxima[mode],worst)
            if worst>4096:overflow.add(row['pool_id'])
            records.append({'pool_id':row['pool_id'],'mode':mode,'start':start,'presented_ids':presented,
                            'prompt_sha256':hashlib.sha256(text.encode('utf8')).hexdigest(),
                            'prompt_tokens':length,'retry_prompt_tokens':with_retry,'output_cap':cap,'worst_total':worst})
    out.mkdir(parents=True,exist_ok=False);dest=out/'prompts.jsonl'
    with dest.open('w',encoding='utf8',newline='\n') as f:
        for x in records:f.write(json.dumps(x,separators=(',',':'))+'\n')
    assets={name:sha(model/name) for name in ('tokenizer.json','tokenizer_config.json','config.json')}
    summary={'scope':'CPU tokenizer preflight only; no model or GPU inference','selection_sha256':meta['inputs_sha256'],
             'prompt_code_sha256':sha(Path(__file__).with_name('acord_prompts.py')),'tokenizer_assets_sha256':assets,
             'transformers':importlib.metadata.version('transformers'),'tokenizers':importlib.metadata.version('tokenizers'),
             'pool_count':len(rows),'prompt_count':len(records),'max_with_retry_and_output_cap':dict(maxima),
             'overflow_pool_count':len(overflow),'overflow_pools':sorted(overflow),'prompt_records_sha256':sha(dest)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Preflight:',len(rows),'pools,',len(records),'prompts; maxima',dict(maxima),'overflow pools',len(overflow))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.selection,a.model,a.out)
