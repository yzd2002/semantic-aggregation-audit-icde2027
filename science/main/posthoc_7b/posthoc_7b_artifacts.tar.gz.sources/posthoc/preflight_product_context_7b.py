#!/usr/bin/env python3
"""Check proposed 7B product prompts using earlier observed specifications."""
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
import run_experiment as base
import run_wands
from run_full_pools import chunks
from run_keyed_order_audit import keyed_prompt
from run_esci_7b_posthoc import point_prompt,RETRY,SPEC_RETRY

def lines(p):
    with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True,trust_remote_code=False)
    families=[('esci',base.SYSTEM,'esci_selective_holdout_selection/inputs.jsonl',
       ['run_esci_selective_holdout_qwen/specifications_responses.jsonl','run_esci_selective_holdout_smollm/specifications_responses.jsonl']),
      ('wands',run_wands.SYSTEM,'wands_selective_execution_selection/inputs.jsonl',
       ['run_wands_selective_execution_qwen_v1/specifications.jsonl','run_wands_selective_execution_smollm_v1/specifications.jsonl'])]
    reports={}
    for name,system,source,paths in families:
        rows=lines(a.artifact/source);saved=[]
        for path in paths:
            spec_records=lines(a.artifact/path)
            if [x['query_id'] for x in spec_records]!=[x['query_id'] for x in rows]:raise ValueError('Specification/pool alignment changed')
            saved.append([base.parse_spec(x['final']) for x in spec_records])
        maxes={'specification':0,'keyed_original':0,'keyed_reverse':0,'pointwise':0};counts={k:0 for k in maxes}
        def update(mode,text,cap):
            suffix=SPEC_RETRY if mode=='specification' else RETRY
            n=len(tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text+suffix}],tokenize=True,add_generation_prompt=True))+cap
            maxes[mode]=max(maxes[mode],n);counts[mode]+=1
        for i,row in enumerate(rows):
            update('specification',base.p_spec(row),160)
            for spec in (saved[0][i],saved[1][i],None):
                for part in chunks(row):
                    update('keyed_original',keyed_prompt(part,spec),128)
                    rev={**part,'items':list(reversed(part['items']))}
                    update('keyed_reverse',keyed_prompt(rev,spec),128)
                for item in row['items']:update('pointwise',point_prompt(row,item,spec),4)
        reports[name]={'pools':len(rows),'requests_considered':counts,'max_retry_prompt_plus_cap':maxes,
            'context_limit':4096,'all_checked_within_limit':max(maxes.values())<=4096}
    report={'model':'Qwen2.5-7B-Instruct','samples':reports,
        'limitation':'Uses two previous models’ saved specifications and direct fallback. The new model’s specifications are unknown until inference; the runner must still check every actual prompt.'}
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
