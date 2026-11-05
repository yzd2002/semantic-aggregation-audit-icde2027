#!/usr/bin/env python3
"""CPU tokenizer check for both keyed chunk sizes before inference."""
import argparse,hashlib,importlib.metadata as md,json
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from batch_prompts import SYSTEM,RETRY_SUFFIX,prompt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(selection,model,protocol,out):
    if out.exists():raise FileExistsError('Use a fresh preflight output directory')
    data=selection/'inputs.jsonl';meta=json.loads((selection/'selection.json').read_text())
    if sha(data)!=meta['inputs_sha256'] or meta.get('prefix_length')!=2400:raise ValueError('Selection changed')
    rows=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    if len(rows)!=114 or model.name!='Qwen3-4B-Instruct-2507':raise ValueError('Unexpected selection or model')
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    def tokens(text):return len(tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True))
    rec=[];maxima=defaultdict(int);overflow=set()
    for row in rows:
        for size in (4,8):
            for start in range(0,32,size):
                for mode,reverse in (('original',False),('stable_reverse',True)):
                    text,ids=prompt(row,start,size,reverse);first=tokens(text);retry=tokens(text+RETRY_SUFFIX);total=retry+128
                    maxima[str(size)+'_'+mode]=max(maxima[str(size)+'_'+mode],total)
                    if total>4096:overflow.add(row['pool_id'])
                    rec.append({'pool_id':row['pool_id'],'size':size,'start':start,'mode':mode,'presented_ids':ids,
                                'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':first,'retry_prompt_tokens':retry,'worst_total':total})
    if len(rec)!=2736:raise ValueError('Unexpected prompt count')
    out.mkdir();dest=out/'prompts.jsonl';dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rec))
    result={'selection_sha256':sha(data),'protocol_sha256':sha(protocol),'prompt_code_sha256':sha(Path(__file__).with_name('batch_prompts.py')),
            'base_prompt_code_sha256':sha(Path(__file__).with_name('acord_prompts.py')),'transformers':md.version('transformers'),'tokenizers':md.version('tokenizers'),
            'tokenizer_assets_sha256':{n:sha(model/n) for n in ('tokenizer.json','tokenizer_config.json','config.json')},
            'pool_count':len(rows),'prompt_count':len(rec),'max_with_retry_and_output_cap':dict(maxima),
            'overflow_pool_count':len(overflow),'prompt_records_sha256':sha(dest)}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CPU preflight:',len(rows),'pools,',len(rec),'prompts; maxima',dict(maxima),'overflow pools',len(overflow))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('selection','model','protocol','out'):p.add_argument('--'+n,type=Path,required=True)
    x=p.parse_args();main(x.selection,x.model,x.protocol,x.out)
