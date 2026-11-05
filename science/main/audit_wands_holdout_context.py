#!/usr/bin/env python3
"""Reconstruct actual and worst-case holdout prompts and validate context budgets."""
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer
import run_experiment as base
import run_full_pools as harness
import run_wands
from run_compact_prompt_ablation import lines, prompt_without_example
from run_validation import parse_compact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--model', required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    specs = lines(a.run / 'specifications.jsonl')
    summary = json.loads((a.run / 'summary.json').read_text())
    assert len(rows) == len(specs) == 160 and summary['protocol'] == 'PROTOCOL_wands_holdout.md'
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    def length(text):
        return len(tok.apply_chat_template([{'role': 'system', 'content': run_wands.SYSTEM},
                                            {'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
    suffix = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
    spec_suffix = '\nReturn valid JSON only.'
    spec_calls = 0;spec_worst = spec_actual = 0
    for row, spec in zip(rows, specs):
        assert row['query_id'] == spec['query_id']
        text = base.p_spec(row)
        spec_worst = max(spec_worst, length(text + spec_suffix) + 160)
        spec_actual = max(spec_actual, length(text) + 160);spec_calls += 1
        if base.parse_spec(spec['first']) is None:
            spec_actual = max(spec_actual, length(text + spec_suffix) + 160);spec_calls += 1
    assert spec_calls == summary['specifications']['initial_calls'] + summary['specifications']['retry_calls']
    assert spec_worst == summary['specifications']['max_prompt_with_retry_plus_cap'] and spec_worst <= 4096
    result = {'max_context': 4096, 'specification': {'calls': spec_calls, 'actual_max_prompt_plus_cap': spec_actual,
              'worst_retry_prompt_plus_cap': spec_worst}, 'modes': {}}
    global_worst = 0
    for mode in ('original', 'replay', 'stable_reverse'):
        outputs = lines(a.run / (mode + '_outputs.jsonl'))
        assert len(outputs) == len(rows)
        calls = actual = worst = 0
        for row, spec, out in zip(rows, specs, outputs):
            assert row['query_id'] == spec['query_id'] == out['query_id']
            parts = harness.chunks(row)
            assert len(parts) == len(out['raw'])
            for part, raw in zip(parts, out['raw']):
                n = len(part['items'])
                assert raw['start'] == part['start'] and raw['size'] == n
                if mode == 'stable_reverse':part = {**part, 'items': [dict(x) for x in reversed(part['items'])]}
                text = prompt_without_example(part, spec['parsed'])
                actual = max(actual, length(text) + 32);calls += 1
                worst = max(worst, length(text + suffix) + 32)
                if parse_compact(raw['first'], n) is None:
                    actual = max(actual, length(text + suffix) + 32);calls += 1
        st = summary['modes'][mode]
        assert calls == st['chunks'] + st['retry_chunk_calls'] and worst <= 4096
        global_worst = max(global_worst, worst)
        result['modes'][mode] = {'calls': calls, 'actual_max_prompt_plus_cap': actual, 'worst_retry_prompt_plus_cap': worst}
    assert global_worst == summary['max_candidate_prompt_with_retry_plus_cap']
    (a.run / 'context_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':main()
