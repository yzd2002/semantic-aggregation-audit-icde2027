#!/usr/bin/env python3
"""Evaluate the frozen disjoint WANDS workload under three guided order conditions."""
import argparse, collections, hashlib, json, os, time
from pathlib import Path
import run_experiment as base
import run_full_pools as harness
import run_wands
from run_compact_prompt_ablation import lines, prompt_without_example
from run_validation import parse_compact

def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1':
        raise RuntimeError('The frozen holdout protocol requires VLLM_USE_V1=1.')
    a.out.mkdir(parents=True, exist_ok=False)
    rows = lines(a.selection / 'inputs.jsonl')
    selection = json.loads((a.selection / 'selection.json').read_text())
    if selection['protocol'] != 'PROTOCOL_wands_holdout.md' or len(rows) != 160 or any(len(r['items']) != 32 for r in rows):
        raise ValueError('Unexpected holdout selection.')
    selection_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    import torch, transformers, vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=a.model, tokenizer=a.model, trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tok = llm.get_tokenizer()
    def wrap(system, text):
        return tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
                                       tokenize=False, add_generation_prompt=True)
    def token_count(system, text):
        return len(tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
                                           tokenize=True, add_generation_prompt=True))
    def guided(n):
        return SamplingParams(temperature=0, max_tokens=32, guided_decoding=GuidedDecodingParams(regex='[YN]{%d}' % n))
    control = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')
    raw_control = llm.generate([control], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)[0].outputs[0].text.strip()
    guided_control = llm.generate([control], guided(4), use_tqdm=False)[0].outputs[0].text.strip()
    sanity = {'unconstrained': raw_control, 'guided_regex_4': guided_control,
              'unconstrained_valid': parse_compact(raw_control, 4) is not None,
              'guided_valid': parse_compact(guided_control, 4) is not None}
    (a.out / 'guidance_sanity.json').write_text(json.dumps(sanity, indent=2) + '\n')
    if sanity['unconstrained_valid'] or not sanity['guided_valid']:
        raise RuntimeError('Guidance negative control failed; no candidate evaluations were run.')

    system = run_wands.SYSTEM
    spec_prompts = [base.p_spec(row) for row in rows]
    max_spec_with_retry = max(token_count(system, p + '\nReturn valid JSON only.') for p in spec_prompts)
    if max_spec_with_retry + 160 > 4096:
        raise RuntimeError('Specification prompt plus generation cap exceeds context.')
    tick = time.perf_counter()
    spec_results = llm.generate([wrap(system, p) for p in spec_prompts], SamplingParams(temperature=0, max_tokens=160), use_tqdm=False)
    spec_first = [r.outputs[0].text.strip() for r in spec_results]
    spec_final = spec_first[:]
    spec_bad = [i for i, value in enumerate(spec_first) if base.parse_spec(value) is None]
    spec_retry_results = llm.generate([wrap(system, spec_prompts[i] + '\nReturn valid JSON only.') for i in spec_bad],
                                       SamplingParams(temperature=0, max_tokens=160), use_tqdm=False) if spec_bad else []
    for i, result in zip(spec_bad, spec_retry_results):
        spec_final[i] = result.outputs[0].text.strip()
    specs = [base.parse_spec(value) for value in spec_final]
    spec_records = [{'query_id': row['query_id'], 'prompt_sha256': digest(prompt), 'first': first, 'final': final,
                     'parsed': spec, 'fallback': spec is None}
                    for row, prompt, first, final, spec in zip(rows, spec_prompts, spec_first, spec_final, specs)]
    with (a.out / 'specifications.jsonl').open('w', encoding='utf8') as f:
        for rec in spec_records:f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    spec_seconds = time.perf_counter() - tick
    summary = {'protocol': 'PROTOCOL_wands_holdout.md', 'model': os.path.basename(a.model),
               'selection': a.selection.name, 'selection_inputs_sha256': selection_sha,
               'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__,
                           'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'max_model_len': 4096,
                           'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection'},
               'specifications': {'initial_calls': len(rows), 'retry_calls': len(spec_bad),
                                  'fallback_pools': sum(x is None for x in specs), 'offline_generation_seconds': spec_seconds,
                                  'max_prompt_with_retry_plus_cap': max_spec_with_retry + 160},
               'mode_order': ['original', 'replay', 'stable_reverse'], 'modes': {}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    modes = {name: {'prompts': [], 'locations': [], 'metadata': []}
             for name in ('original', 'replay', 'stable_reverse')}
    suffix = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
    max_prompt_with_retry = 0
    for i, (row, spec) in enumerate(zip(rows, specs)):
        for part in harness.chunks(row):
            start, n = part['start'], len(part['items'])
            reversed_part = {**part, 'items': [dict(it) for it in reversed(part['items'])]}
            original_text = prompt_without_example(part, spec)
            reversed_text = prompt_without_example(reversed_part, spec)
            for mode, text, positions in [('original', original_text, list(range(start, start+n))),
                                          ('replay', original_text, list(range(start, start+n))),
                                          ('stable_reverse', reversed_text, list(range(start+n-1, start-1, -1)))]:
                max_prompt_with_retry = max(max_prompt_with_retry, token_count(system, text + suffix))
                modes[mode]['prompts'].append(wrap(system, text))
                modes[mode]['locations'].append((i, start, n, positions))
                modes[mode]['metadata'].append({'text': text, 'original_prompt_sha256': digest(original_text),
                                                'condition_prompt_sha256': digest(text)})
    if max_prompt_with_retry + 32 > 4096:
        raise RuntimeError('Candidate prompt plus generation cap exceeds context.')
    summary['max_candidate_prompt_with_retry_plus_cap'] = max_prompt_with_retry + 32
    for mode, data in modes.items():
        prompts, locations, metadata = data['prompts'], data['locations'], data['metadata']
        tick = time.perf_counter()
        results = llm.generate(prompts, [guided(loc[2]) for loc in locations], use_tqdm=False)
        first = [r.outputs[0].text.strip() for r in results]
        final = first[:]
        first_reason = [r.outputs[0].finish_reason for r in results]
        retry_reason = [None] * len(first)
        bad = [j for j, value in enumerate(first) if parse_compact(value, locations[j][2]) is None]
        retries = llm.generate([wrap(system, metadata[j]['text'] + suffix) for j in bad],
                               [guided(locations[j][2]) for j in bad], use_tqdm=False) if bad else []
        for j, result in zip(bad, retries):
            final[j] = result.outputs[0].text.strip()
            retry_reason[j] = result.outputs[0].finish_reason
        seconds = time.perf_counter() - tick
        records = [{'pred': [None] * len(r['items']), 'raw': []} for r in rows]
        for j, (i, start, n, positions) in enumerate(locations):
            labels = parse_compact(final[j], n)
            rec = records[i]
            rec['raw'].append({'start': start, 'size': n, 'original_indices_in_prompt_order': positions,
                               'original_prompt_sha256': metadata[j]['original_prompt_sha256'],
                               'condition_prompt_sha256': metadata[j]['condition_prompt_sha256'],
                               'first': first[j], 'final': final[j], 'first_finish_reason': first_reason[j],
                               'retry_finish_reason': retry_reason[j]})
            if labels is not None:
                for index, label in zip(positions, labels):rec['pred'][index] = label
        saved = []
        for row, rec in zip(rows, records):
            pred = None if None in rec['pred'] else rec['pred']
            saved.append({'query_id': row['query_id'], 'pred': pred, 'raw': rec['raw']})
        with (a.out / (mode + '_outputs.jsonl')).open('w', encoding='utf8') as f:
            for rec in saved:f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        chunks = [c for row in saved for c in row['raw']]
        summary['modes'][mode] = {'attempted_pools': len(saved), 'complete_pools': sum(r['pred'] is not None for r in saved),
                                  'chunks': len(chunks), 'retry_chunk_calls': len(bad), 'offline_generation_seconds': seconds,
                                  'valid_first_chunks': sum(parse_compact(c['first'], c['size']) is not None for c in chunks),
                                  'valid_final_chunks': sum(parse_compact(c['final'], c['size']) is not None for c in chunks),
                                  'finish_reasons': dict(collections.Counter(c['first_finish_reason'] for c in chunks))}
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(mode, json.dumps(summary['modes'][mode], ensure_ascii=False), flush=True)
    print('Completed all holdout conditions.', flush=True)
if __name__ == '__main__':
    main()
