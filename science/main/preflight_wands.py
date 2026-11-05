#!/usr/bin/env python3
"""Check token lengths for frozen WANDS prompts before running inference."""
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
import run_wands,run_experiment as base,run_full_pools as harness
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wands-dir',type=Path,default=Path('/home/gpu/intent2s_wands_official_2026'))
    ap.add_argument('--model',default='/root/models/Qwen/Qwen3-4B-Instruct-2507');ap.add_argument('--out',type=Path)
    a=ap.parse_args();rows,_=run_wands.sample(a.wands_dir);base.SYSTEM=run_wands.SYSTEM
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    def n(prompt):return len(tok.apply_chat_template([{'role':'system','content':base.SYSTEM},{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True))
    maximum=max(n(base.p_batch(part)) for row in rows for part in harness.chunks(row))
    spec=max(n(base.p_spec(row)) for row in rows);point=max(n(base.p_point(row,it)) for row in rows for it in row['items'])
    result={'max_direct_batch_prompt_tokens':maximum,'max_spec_prompt_tokens':spec,'max_pointwise_prompt_tokens':point,
            'conservative_extra_spec_tokens':320,'max_batch_generation_tokens':192,'max_context':4096,
            'batch_with_spec_reserve_within_context':maximum+320+192<4096}
    assert result['batch_with_spec_reserve_within_context']
    if a.out:a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
