#!/usr/bin/env python3
"""Replay identical prompts, then reverse each chunk while preserving product IDs."""
import argparse, collections, json, os, time
from pathlib import Path
import run_experiment as base
import run_full_pools as harness
import run_wands
from run_compact_prompt_ablation import lines, prompt_without_example
from run_reverse_order import digest, reverse_part
from run_validation import parse_compact

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--renumbered', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1':
        raise RuntimeError('This protocol requires the verified V1 engine: VLLM_USE_V1=1.')
    a.out.mkdir(parents=True, exist_ok=False)
    baseline_summary = json.loads((a.baseline / 'summary.json').read_text())
    renumbered_summary = json.loads((a.renumbered / 'summary.json').read_text())
    assert baseline_summary['model'] == renumbered_summary['model'] == os.path.basename(a.model)
    assert baseline_summary['runtime']['engine'] == renumbered_summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    assert renumbered_summary['baseline'] == a.baseline.name
    for folder in (a.baseline, a.renumbered):
        sanity = json.loads((folder / 'guidance_sanity.json').read_text())
        assert not sanity['unconstrained_valid'] and sanity['guided_valid']
    datasets = {}
    for name, source, system in [('esci', a.esci_run, base.SYSTEM), ('wands', a.wands_run, run_wands.SYSTEM)]:
        rows = lines(source / 'inputs.jsonl')
        previous = lines(source / 'outputs.jsonl')
        baseline = lines(a.baseline / (name + '_outputs.jsonl'))
        renumbered = lines(a.renumbered / (name + '_outputs.jsonl'))
        assert len(rows) == len(previous) == len(baseline) == len(renumbered)
        assert all(x['query_id'] == y['query_id'] == z['query_id'] == w['query_id'] for x, y, z, w in zip(rows, previous, baseline, renumbered))
        datasets[name] = {'rows': rows, 'previous': previous, 'baseline': baseline, 'renumbered': renumbered, 'system': system}

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
    def params(n):
        return SamplingParams(temperature=0, max_tokens=32, guided_decoding=GuidedDecodingParams(regex='[YN]{%d}' % n))
    control = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')
    raw_control = llm.generate([control], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)[0].outputs[0].text.strip()
    guided_control = llm.generate([control], params(4), use_tqdm=False)[0].outputs[0].text.strip()
    sanity = {'unconstrained': raw_control, 'guided_regex_4': guided_control,
              'unconstrained_valid': parse_compact(raw_control, 4) is not None,
              'guided_valid': parse_compact(guided_control, 4) is not None}
    (a.out / 'guidance_sanity.json').write_text(json.dumps(sanity, indent=2) + '\n')
    if sanity['unconstrained_valid'] or not sanity['guided_valid']:
        raise RuntimeError('Guidance negative control failed; stop before evaluating candidates.')

    modes = {'replay': {'prompts': [], 'locations': [], 'metadata': []},
             'stable_reverse': {'prompts': [], 'locations': [], 'metadata': []}}
    retry_suffix = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
    max_prompt_with_retry = 0
    for name, d in datasets.items():
        for i, (row, previous, original, old_reversal) in enumerate(zip(d['rows'], d['previous'], d['baseline'], d['renumbered'])):
            entry = previous['plans']['two_stage_compact_fallback']
            spec = base.parse_spec(entry['spec']['final'])
            assert bool(entry['fallback']) == (spec is None)
            parts = harness.chunks(row)
            assert len(parts) == len(original['raw']) == len(old_reversal['raw'])
            for part, old_base, old_reverse in zip(parts, original['raw'], old_reversal['raw']):
                start, n = part['start'], len(part['items'])
                assert start == old_base['start'] == old_reverse['start'] and n == old_base['size'] == old_reverse['size']
                stable = {**part, 'items': [dict(x) for x in reversed(part['items'])]}
                renumbered, reverse_indices = reverse_part(part)
                for left, right in zip(stable['items'], renumbered['items']):
                    assert {k: v for k, v in left.items() if k != 'id'} == {k: v for k, v in right.items() if k != 'id'}
                original_text = prompt_without_example(part, spec)
                stable_text = prompt_without_example(stable, spec)
                renumbered_text = prompt_without_example(renumbered, spec)
                assert digest(original_text) == old_reverse['original_prompt_sha256']
                assert digest(renumbered_text) == old_reverse['reversed_prompt_sha256']
                for mode, text, indices in [('replay', original_text, list(range(start, start + n))),
                                            ('stable_reverse', stable_text, reverse_indices)]:
                    max_prompt_with_retry = max(max_prompt_with_retry, token_count(d['system'], text + retry_suffix))
                    modes[mode]['prompts'].append(wrap(d['system'], text))
                    modes[mode]['locations'].append((name, i, start, n, indices))
                    modes[mode]['metadata'].append({'system': d['system'], 'text': text,
                        'original_prompt_sha256': digest(original_text), 'condition_prompt_sha256': digest(text),
                        'renumbered_prompt_sha256': digest(renumbered_text)})
    if max_prompt_with_retry + 32 > 4096:
        raise RuntimeError('Prompt plus generation cap exceeds the context window.')
    summary = {'protocol': 'PROTOCOL_order_controls.md', 'model': os.path.basename(a.model),
               'baseline': a.baseline.name, 'renumbered': a.renumbered.name,
               'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__,
                           'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'max_model_len': 4096,
                           'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection'},
               'max_prompt_with_retry_tokens': max_prompt_with_retry, 'max_prompt_plus_output_cap': max_prompt_with_retry + 32,
               'mode_order': ['replay', 'stable_reverse'], 'modes': {}}
    for mode, dmode in modes.items():
        prompts, locations, metadata = dmode['prompts'], dmode['locations'], dmode['metadata']
        tick = time.perf_counter()
        results = llm.generate(prompts, [params(x[3]) for x in locations], use_tqdm=False)
        first = [r.outputs[0].text.strip() for r in results]
        final = first[:]
        first_reason = [r.outputs[0].finish_reason for r in results]
        retry_reason = [None] * len(first)
        bad = [j for j, s in enumerate(first) if parse_compact(s, locations[j][3]) is None]
        retries = llm.generate([wrap(metadata[j]['system'], metadata[j]['text'] + retry_suffix) for j in bad],
                               [params(locations[j][3]) for j in bad], use_tqdm=False) if bad else []
        for j, result in zip(bad, retries):
            final[j] = result.outputs[0].text.strip()
            retry_reason[j] = result.outputs[0].finish_reason
        seconds = time.perf_counter() - tick
        records = {name: [{'pred': [None] * len(r['items']), 'raw': []} for r in data['rows']] for name, data in datasets.items()}
        for j, (name, i, start, n, indices) in enumerate(locations):
            parsed = parse_compact(final[j], n)
            rec = records[name][i]
            rec['raw'].append({'start': start, 'size': n, 'original_indices_in_prompt_order': indices,
                               'original_prompt_sha256': metadata[j]['original_prompt_sha256'],
                               'condition_prompt_sha256': metadata[j]['condition_prompt_sha256'],
                               'renumbered_prompt_sha256': metadata[j]['renumbered_prompt_sha256'],
                               'first': first[j], 'final': final[j], 'first_finish_reason': first_reason[j],
                               'retry_finish_reason': retry_reason[j]})
            if parsed is not None:
                for ix, label in zip(indices, parsed):
                    rec['pred'][ix] = label
        mode_summary = {'initial_chunk_calls': len(first), 'retry_chunk_calls': len(bad),
                        'offline_generation_seconds': seconds, 'datasets': {}}
        for name, data in datasets.items():
            saved = []
            for row, rec in zip(data['rows'], records[name]):
                pred = None if None in rec['pred'] else rec['pred']
                saved.append({'query_id': row['query_id'], 'pred': pred, 'raw': rec['raw']})
            with (a.out / (mode + '_' + name + '_outputs.jsonl')).open('w', encoding='utf8') as f:
                for row in saved:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            chunks = [c for row in saved for c in row['raw']]
            mode_summary['datasets'][name] = {'attempted_pools': len(saved), 'complete_pools': sum(x['pred'] is not None for x in saved),
                'chunks': len(chunks), 'valid_first_chunks': sum(parse_compact(c['first'], c['size']) is not None for c in chunks),
                'valid_final_chunks': sum(parse_compact(c['final'], c['size']) is not None for c in chunks),
                'finish_reasons': dict(collections.Counter(c['first_finish_reason'] for c in chunks))}
        summary['modes'][mode] = mode_summary
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(mode, json.dumps(mode_summary, ensure_ascii=False), flush=True)
    print('Completed both order controls.', flush=True)

if __name__ == '__main__':
    main()
