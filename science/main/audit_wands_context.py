#!/usr/bin/env python3
"""Reconstruct generated WANDS requests and check the exact token budget after inference."""
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
import run_wands,run_experiment as base,run_full_pools as harness
from run_validation import compact_prompt,parse_compact
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_wands_160')
    ap.add_argument('--model',default='/root/models/Qwen/Qwen3-4B-Instruct-2507');a=ap.parse_args()
    rows=lines(a.run/'inputs.jsonl');outs=lines(a.run/'outputs.jsonl');summary=json.loads((a.run/'summary.json').read_text());assert len(rows)==len(outs)
    base.SYSTEM=run_wands.SYSTEM;tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    def length(text):return len(tok.apply_chat_template([{'role':'system','content':base.SYSTEM},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True))
    result={'max_context':4096,'methods':{}}
    for method in ['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']:
        max_prompt=0;max_total=0;calls=0
        def observe(prompt,cap):
            nonlocal max_prompt,max_total,calls
            size=length(prompt);calls+=1;max_prompt=max(max_prompt,size);max_total=max(max_total,size+cap)
        for row,out in zip(rows,outs):
            entry=out['plans'][method]
            if method=='pointwise':
                for item,record in zip(row['items'],entry['raw']):
                    p=base.p_point(row,item);observe(p,4)
                    if record['first'] not in ('Y','N'):observe(p+'\nOutput one character: Y or N.',4)
                continue
            specification=method.startswith('two_stage');spec=None
            if specification:
                raw=entry['spec'];assert raw is not None
                s=base.p_spec(row);observe(s,160)
                if base.parse_spec(raw['first']) is None:observe(s+'\nReturn valid JSON only.',160)
                spec=base.parse_spec(raw['final'])
            chunk_map={part['start']:part for part in harness.chunks(row)}
            compact='compact' in method;parser=parse_compact if compact else base.parse_batch;cap=32 if compact else 192
            for raw in entry['raw']:
                part=chunk_map[raw['start']];assert len(part['items'])==raw['size']
                p=compact_prompt(part,spec) if compact else base.p_batch(part,spec)
                observe(p,cap)
                if parser(raw['first'],raw['size']) is None:
                    observe(p+'\nYour previous answer had the wrong format; follow the output instructions exactly.',cap)
        result['methods'][method]={'reconstructed_calls':calls,'max_prompt_tokens':max_prompt,
                                   'max_prompt_plus_generation_cap':max_total,'within_context':max_total<=4096}
        assert calls==summary['plans'][method]['calls'],(method,calls,summary['plans'][method]['calls'])
        assert max_total<=4096,(method,max_total)
    (a.run/'context_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
