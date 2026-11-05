#!/usr/bin/env python3
"""Re-run the exposed WANDS holdout with ID-keyed, regex-guided answers."""
import argparse
import collections
import hashlib
import json
import os
import re
import time
from pathlib import Path

import run_experiment as base
import run_full_pools as harness
import run_wands
from run_compact_prompt_ablation import lines

PROTOCOL = 'PROTOCOL_keyed_order_audit.md'
SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def keyed_regex(ids):
    ids = sorted(ids)
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError('Expected unique, consecutive chunk-local IDs starting at 1.')
    return r'\{' + ','.join(re.escape(json.dumps(str(i))) + r':"[YN]"' for i in ids) + r'\}'


def parse_keyed(text, ids):
    if re.fullmatch(keyed_regex(ids), text) is None:
        return None
    obj = json.loads(text)
    if list(obj) != [str(i) for i in sorted(ids)] or set(obj.values()) - {'Y', 'N'}:
        return None
    return {int(k): v for k, v in obj.items()}


def keyed_prompt(part, spec):
    text = 'Query: ' + part['query'] + '\n'
    if spec is not None:
        text += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    text += 'Products in order:\n' + '\n'.join(
        json.dumps({'id': x['id'], 'text': x['text']}, ensure_ascii=False) for x in part['items'])
    text += ('\nReturn only a JSON object mapping each given product id to Y or N for that product. '
             'Use the given ids, not positions; include every id once, in increasing numeric order. '
             'Do not add spaces or explanations.')
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1':
        raise RuntimeError('The protocol requires VLLM_USE_V1=1.')
    rows = lines(a.selection / 'inputs.jsonl')
    selection_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    prior_summary = json.loads((a.prior / 'summary.json').read_text())
    specs = lines(a.prior / 'specifications.jsonl')
    if (len(rows) != 160 or len(specs) != len(rows) or any(len(r['items']) != 32 for r in rows)
            or prior_summary['selection_inputs_sha256'] != selection_sha
            or prior_summary['model'] != os.path.basename(a.model)):
        raise ValueError('Unexpected selection, prior run or model.')
    for row, s in zip(rows, specs):
        if row['query_id'] != s['query_id'] or s['parsed'] != base.parse_spec(s['final']) or s['fallback'] != (s['parsed'] is None):
            raise ValueError('Saved specification does not match its raw response.')
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = LLM(model=a.model, tokenizer=a.model, trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tok = llm.get_tokenizer()
    def wrap(system, text, tokenize=False):
        return tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
                                       tokenize=tokenize, add_generation_prompt=True)
    ids = list(range(1, 9))
    rx = keyed_regex(ids)
    def guided():
        return SamplingParams(temperature=0, max_tokens=128, guided_decoding=GuidedDecodingParams(regex=rx))
    control = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')
    ordinary = llm.generate([control], SamplingParams(temperature=0, max_tokens=128), use_tqdm=False)[0].outputs[0].text.strip()
    constrained = llm.generate([control], guided(), use_tqdm=False)[0].outputs[0].text.strip()
    sanity = {'unconstrained': ordinary, 'guided_keyed': constrained,
              'unconstrained_valid': parse_keyed(ordinary, ids) is not None,
              'guided_valid': parse_keyed(constrained, ids) is not None, 'regex': rx}
    (a.out / 'guidance_sanity.json').write_text(json.dumps(sanity, indent=2) + '\n')
    if sanity['unconstrained_valid'] or not sanity['guided_valid']:
        raise RuntimeError('Keyed guidance negative control failed; no candidate evaluation was run.')

    system = run_wands.SYSTEM
    modes = {name: {'prompts': [], 'locations': [], 'metadata': []} for name in ('original', 'replay', 'stable_reverse')}
    max_prompt = 0
    for i, (row, record) in enumerate(zip(rows, specs)):
        for part in harness.chunks(row):
            start = part['start']
            if [x['id'] for x in part['items']] != ids:
                raise ValueError('Unexpected chunk-local IDs.')
            reversed_part = {**part, 'items': [dict(it) for it in reversed(part['items'])]}
            original = keyed_prompt(part, record['parsed'])
            reverse = keyed_prompt(reversed_part, record['parsed'])
            for mode, text, presented in [('original', original, ids), ('replay', original, ids),
                                           ('stable_reverse', reverse, list(reversed(ids)))]:
                max_prompt = max(max_prompt, len(wrap(system, text + SUFFIX, tokenize=True)))
                modes[mode]['prompts'].append(wrap(system, text))
                modes[mode]['locations'].append((i, start))
                modes[mode]['metadata'].append({'text': text, 'presented_ids': presented,
                                                'condition_prompt_sha256': digest(text),
                                                'original_prompt_sha256': digest(original)})
    if max_prompt + 128 > 4096:
        raise RuntimeError('Prompt plus output cap exceeds the context window.')
    summary = {'protocol': PROTOCOL, 'model': os.path.basename(a.model), 'selection': a.selection.name,
               'prior': a.prior.name, 'selection_inputs_sha256': selection_sha,
               'prior_specifications_sha256': hashlib.sha256((a.prior / 'specifications.jsonl').read_bytes()).hexdigest(),
               'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__,
                           'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'max_model_len': 4096,
                           'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection'},
               'max_prompt_with_retry_plus_cap': max_prompt + 128, 'mode_order': list(modes), 'modes': {}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for mode, data in modes.items():
        tick = time.perf_counter()
        results = llm.generate(data['prompts'], guided(), use_tqdm=False)
        first = [r.outputs[0].text.strip() for r in results]
        final = first[:]
        reasons = [r.outputs[0].finish_reason for r in results]
        retry_reasons = [None] * len(first)
        bad = [j for j, s in enumerate(first) if parse_keyed(s, ids) is None]
        retry = llm.generate([wrap(system, data['metadata'][j]['text'] + SUFFIX) for j in bad], guided(), use_tqdm=False) if bad else []
        for j, result in zip(bad, retry):
            final[j] = result.outputs[0].text.strip()
            retry_reasons[j] = result.outputs[0].finish_reason
        records = [{'query_id': row['query_id'], 'pred': [None] * len(row['items']), 'raw': []} for row in rows]
        for j, (i, start) in enumerate(data['locations']):
            meta = data['metadata'][j]
            answer = parse_keyed(final[j], ids)
            records[i]['raw'].append({'start': start, 'size': len(ids), 'presented_ids': meta['presented_ids'],
                                      'original_prompt_sha256': meta['original_prompt_sha256'],
                                      'condition_prompt_sha256': meta['condition_prompt_sha256'],
                                      'first': first[j], 'final': final[j], 'first_finish_reason': reasons[j],
                                      'retry_finish_reason': retry_reasons[j]})
            if answer is not None:
                for identifier, label in answer.items():
                    records[i]['pred'][start + identifier - 1] = label
        for rec in records:
            if None in rec['pred']:
                rec['pred'] = None
        with (a.out / (mode + '_outputs.jsonl')).open('w', encoding='utf8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        mode_stats = {'attempted_pools': len(rows), 'complete_pools': sum(r['pred'] is not None for r in records),
                      'chunks': len(first), 'retry_chunk_calls': len(bad),
                      'valid_first_chunks': sum(parse_keyed(s, ids) is not None for s in first),
                      'valid_final_chunks': sum(parse_keyed(s, ids) is not None for s in final),
                      'finish_reasons': dict(collections.Counter(reasons)),
                      'offline_generation_seconds': time.perf_counter() - tick}
        summary['modes'][mode] = mode_stats
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(mode, json.dumps(mode_stats), flush=True)
    print('Completed keyed order audit.', flush=True)


if __name__ == '__main__':
    main()
