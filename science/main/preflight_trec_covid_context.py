#!/usr/bin/env python3
"""Check all potential TREC-COVID prompts against the model context before GPU use."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from run_trec_covid_transfer import RETRY_SUFFIX, SYSTEM, keyed_prompt, point_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    rows = [json.loads(x) for x in (a.selection / 'inputs.jsonl').read_text(encoding='utf8').splitlines()]
    maxima = {'original': 0, 'stable_reverse': 0, 'pointwise': 0}
    counts = {k: 0 for k in maxima}
    for row in rows:
        requests = [('original', keyed_prompt(row, i)[0], 128) for i in range(0, 32, 8)]
        requests += [('stable_reverse', keyed_prompt(row, i, True)[0], 128) for i in range(0, 32, 8)]
        requests += [('pointwise', point_prompt(row, i), 4) for i in range(32)]
        for name, p, cap in requests:
            length = len(tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': p + RETRY_SUFFIX}], tokenize=True,
                         add_generation_prompt=True)) + cap
            maxima[name] = max(maxima[name], length)
            counts[name] += 1
    report = {'topics': len(rows), 'requests_considered': counts, 'max_retry_prompt_plus_cap': maxima,
              'context_limit': 4096, 'all_within_limit': max(maxima.values()) <= 4096}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if not report['all_within_limit']:
        raise RuntimeError('At least one prompt exceeds the recorded context')


if __name__ == '__main__':
    main()
