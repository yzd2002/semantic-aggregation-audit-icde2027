#!/usr/bin/env python3
"""Apply the fixed example-removed prompt with a per-chunk regex constraint."""
import argparse, collections, json, os, time
from pathlib import Path
import run_experiment as base
import run_full_pools as harness
import run_wands
from run_validation import parse_compact
from run_compact_prompt_ablation import lines, prompt_without_example

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1':
        raise RuntimeError('The corrected guided comparison requires VLLM_USE_V1=1 for the V1 engine.')
    a.out.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for name, path, system in [('esci', a.esci_run, base.SYSTEM), ('wands', a.wands_run, run_wands.SYSTEM)]:
        rows = lines(path / 'inputs.jsonl')
        previous = lines(path / 'outputs.jsonl')
        assert len(rows) == len(previous) and all(x['query_id'] == y['query_id'] for x, y in zip(rows, previous))
        datasets[name] = {'rows': rows, 'previous': previous, 'system': system}

    import torch, transformers, vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=a.model, tokenizer=a.model, trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16,
              enforce_eager=True, enable_prefix_caching=True, disable_log_stats=True)
    tok = llm.get_tokenizer()
    def wrap(system, text):
        return tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
                                       tokenize=False, add_generation_prompt=True)
    def params(n):
        return SamplingParams(temperature=0, max_tokens=32,
                              guided_decoding=GuidedDecodingParams(regex='[YN]{%d}' % n))
    # Fail closed: this must change an intentionally incompatible answer into
    # an output accepted by the same exact-length parser before real inference.
    control = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')
    raw_control = llm.generate([control], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)[0].outputs[0].text.strip()
    guided_control = llm.generate([control], params(4), use_tqdm=False)[0].outputs[0].text.strip()
    sanity = {'unconstrained': raw_control, 'guided_regex_4': guided_control,
              'unconstrained_valid': parse_compact(raw_control, 4) is not None,
              'guided_valid': parse_compact(guided_control, 4) is not None}
    (a.out / 'guidance_sanity.json').write_text(json.dumps(sanity, indent=2) + '\n')
    if sanity['unconstrained_valid'] or not sanity['guided_valid']:
        raise RuntimeError('Guidance negative-control check failed; do not interpret this run as constrained.')
    prompts, specs, locations = [], [], []
    for name, d in datasets.items():
        for i, (row, old) in enumerate(zip(d['rows'], d['previous'])):
            entry = old['plans']['two_stage_compact_fallback']
            spec = base.parse_spec(entry['spec']['final'])
            assert bool(entry['fallback']) == (spec is None)
            for part in harness.chunks(row):
                text = prompt_without_example(part, spec)
                prompts.append(wrap(d['system'], text))
                specs.append((d['system'], text))
                locations.append((name, i, part['start'], len(part['items'])))
    tick = time.perf_counter()
    results = llm.generate(prompts, [params(x[3]) for x in locations], use_tqdm=False)
    first = [r.outputs[0].text.strip() for r in results]
    final = first[:]
    reasons = [r.outputs[0].finish_reason for r in results]
    tokens_in = sum(len(r.prompt_token_ids) for r in results)
    tokens_out = sum(len(r.outputs[0].token_ids) for r in results)
    bad = [i for i, val in enumerate(first) if parse_compact(val, locations[i][3]) is None]
    retry_prompts = [wrap(specs[j][0], specs[j][1] + '\nYour previous answer had the wrong format; follow the output instructions exactly.') for j in bad]
    retry_results = llm.generate(retry_prompts, [params(locations[j][3]) for j in bad], use_tqdm=False) if bad else []
    retry_reasons = [None] * len(first)
    for j, res in zip(bad, retry_results):
        final[j] = res.outputs[0].text.strip()
        retry_reasons[j] = res.outputs[0].finish_reason
    tokens_in += sum(len(r.prompt_token_ids) for r in retry_results)
    tokens_out += sum(len(r.outputs[0].token_ids) for r in retry_results)
    elapsed = time.perf_counter() - tick
    records = {name: [{'pred': [None] * len(r['items']), 'raw': []} for r in d['rows']] for name, d in datasets.items()}
    for j, (name, i, start, size) in enumerate(locations):
        parsed = parse_compact(final[j], size)
        e = records[name][i]
        e['raw'].append({'start': start, 'size': size, 'first': first[j], 'final': final[j],
                         'first_finish_reason': reasons[j], 'retry_finish_reason': retry_reasons[j]})
        if parsed is not None:
            e['pred'][start:start + size] = parsed
    summary = {'protocol': 'PROTOCOL_guided_compact.md', 'model': os.path.basename(a.model),
               'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__,
                           'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'max_model_len': 4096,
                           'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection',
                           'guidance_sanity_file': 'guidance_sanity.json'},
               'initial_chunk_calls': len(first), 'retry_chunk_calls': len(bad),
               'prompt_tokens': tokens_in, 'output_tokens': tokens_out, 'offline_generation_seconds': elapsed,
               'datasets': {}}
    for name, d in datasets.items():
        saved = []
        for row, e in zip(d['rows'], records[name]):
            pred = e['pred'] if None not in e['pred'] else None
            saved.append({'query_id': row['query_id'], 'pred': pred, 'raw': e['raw']})
        with (a.out / (name + '_outputs.jsonl')).open('w', encoding='utf8') as f:
            for row in saved:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        chunks = [c for row in saved for c in row['raw']]
        summary['datasets'][name] = {'attempted_pools': len(saved), 'complete_pools': sum(x['pred'] is not None for x in saved),
                                     'chunks': len(chunks), 'valid_first_chunks': sum(parse_compact(c['first'], c['size']) is not None for c in chunks),
                                     'valid_final_chunks': sum(parse_compact(c['final'], c['size']) is not None for c in chunks),
                                     'finish_reasons': dict(collections.Counter(c['first_finish_reason'] for c in chunks))}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
