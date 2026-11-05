#!/usr/bin/env python3
"""Check every short/long pointwise prompt plus retry against the context limit."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
import run_experiment as base

SUFFIX='\nOutput one character: Y or N.'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
    p=a.selection/'inputs.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
    maxima={'short':0,'long':0};counts={'short':0,'long':0}
    for row in rows:
        for item in row['items']:
            for mode in ('short','long'):
                prompt=base.p_point(row,{'text':item[mode]})+SUFFIX
                n=len(tok.apply_chat_template([{'role':'system','content':base.SYSTEM},
                       {'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True))+4
                maxima[mode]=max(maxima[mode],n);counts[mode]+=1
    info={'selection_inputs_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'model':a.model.name,'requests_considered':counts,'max_retry_prompt_plus_cap':maxima,
          'context_limit':4096,'all_within_limit':max(maxima.values())<=4096}
    a.out.write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))
    if not info['all_within_limit']:raise RuntimeError('Prompt exceeds context')


if __name__=='__main__':main()
