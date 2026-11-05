#!/usr/bin/env python3
"""Independently verify control prompts, identity mapping, metrics and intervals."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import run_experiment as base
import run_full_pools as harness
from run_compact_prompt_ablation import lines, prompt_without_example
from run_validation import parse_compact
from verify_compact_prompt_ablation import metrics

def digest(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()

def close(a, b):
    for k, v in a.items():
        assert (v is None and b[k] is None) or (v is not None and abs(v - b[k]) < 1e-10), (k, v, b[k])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--renumbered', type=Path, required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    assert summary['protocol'] == analysis['protocol'] == 'PROTOCOL_order_controls.md'
    assert summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)' and summary['mode_order'] == ['replay', 'stable_reverse']
    assert summary['baseline'] == a.baseline.name and summary['renumbered'] == a.renumbered.name
    assert parse_compact(sanity['unconstrained'], 4) is None and parse_compact(sanity['guided_regex_4'], 4) is not None
    assert summary['max_prompt_plus_output_cap'] <= 4096
    definitions = [('identical_prompt_replay', 'original', 'replay'), ('reverse_stable_ids', 'replay', 'stable_reverse'),
                   ('reassign_ids_on_reversal', 'stable_reverse', 'renumbered_reverse'),
                   ('reverse_reassigned_ids', 'original', 'renumbered_reverse')]
    for name, source in [('esci', a.esci_run), ('wands', a.wands_run)]:
        rows, previous = lines(source / 'inputs.jsonl'), lines(source / 'outputs.jsonl')
        outputs = {'original': lines(a.baseline / (name + '_outputs.jsonl')),
                   'renumbered_reverse': lines(a.renumbered / (name + '_outputs.jsonl')),
                   'replay': lines(a.run / ('replay_' + name + '_outputs.jsonl')),
                   'stable_reverse': lines(a.run / ('stable_reverse_' + name + '_outputs.jsonl'))}
        assert all(len(v) == len(rows) for v in [previous, *outputs.values()])
        assert all(all(r['query_id'] == o['query_id'] for r, o in zip(rows, v)) for v in [previous, *outputs.values()])
        recovered = {}
        for mode in ('replay', 'stable_reverse'):
            saved = outputs[mode];predictions = [];first_valid = final_valid = 0
            for row, prior, baseline, renumbered, result in zip(rows, previous, outputs['original'], outputs['renumbered_reverse'], saved):
                entry = prior['plans']['two_stage_compact_fallback']
                spec = base.parse_spec(entry['spec']['final'])
                assert bool(entry['fallback']) == (spec is None)
                parts = harness.chunks(row)
                assert len(parts) == len(result['raw']) == len(baseline['raw']) == len(renumbered['raw'])
                pred = [None] * len(row['items'])
                for part, raw, prev in zip(parts, result['raw'], renumbered['raw']):
                    start, n = part['start'], len(part['items'])
                    assert raw['start'] == prev['start'] == start and raw['size'] == prev['size'] == n
                    original = part
                    stable = {'query': part['query'], 'start': start,
                              'items': [dict(part['items'][j]) for j in range(n - 1, -1, -1)]}
                    shifted = {'query': part['query'], 'start': start,
                               'items': [{**part['items'][j], 'id': k + 1} for k, j in enumerate(range(n - 1, -1, -1))]}
                    assert [x['id'] for x in stable['items']] == list(range(n, 0, -1))
                    assert all({k: v for k, v in x.items() if k != 'id'} == {k: v for k, v in y.items() if k != 'id'}
                               for x, y in zip(stable['items'], shifted['items']))
                    ot = prompt_without_example(original, spec)
                    st = prompt_without_example(stable, spec)
                    rt = prompt_without_example(shifted, spec)
                    assert digest(ot) == raw['original_prompt_sha256'] == prev['original_prompt_sha256']
                    assert digest(rt) == raw['renumbered_prompt_sha256'] == prev['reversed_prompt_sha256']
                    assert digest(ot if mode == 'replay' else st) == raw['condition_prompt_sha256']
                    positions = list(range(start, start + n)) if mode == 'replay' else list(range(start + n - 1, start - 1, -1))
                    assert raw['original_indices_in_prompt_order'] == positions
                    first, final = parse_compact(raw['first'], n), parse_compact(raw['final'], n)
                    first_valid += first is not None;final_valid += final is not None
                    if first is not None:assert raw['first'] == raw['final'] and raw['retry_finish_reason'] is None
                    if final is not None:
                        for ix, label in zip(positions, final):
                            assert pred[ix] is None
                            pred[ix] = label
                pred = None if None in pred else pred
                assert pred == result['pred']
                predictions.append(pred)
            saved_summary = summary['modes'][mode]['datasets'][name]
            assert saved_summary['valid_first_chunks'] == first_valid and saved_summary['valid_final_chunks'] == final_valid
            assert saved_summary['complete_pools'] == sum(x is not None for x in predictions)
            recovered[mode] = predictions
        preds = {'original': [x['pred'] for x in outputs['original']], 'renumbered_reverse': [x['pred'] for x in outputs['renumbered_reverse']], **recovered}
        idx = [i for i in range(len(rows)) if all(v[i] is not None for v in preds.values())]
        d = analysis['datasets'][name]
        assert d['common_complete_pools'] == len(idx) and d['common_query_ids'] == [rows[i]['query_id'] for i in idx]
        assert d['completion'] == {k: sum(x is not None for x in vals) for k, vals in preds.items()}
        if idx:
            for k, vals in preds.items():close(metrics(rows, vals, idx), d['quality_common'][k])
            rng = np.random.default_rng(20271121)
            draw = rng.integers(0, len(idx), size=(4000, len(idx)))
            for label, left, right in definitions:
                c = d['comparisons'][label]
                assert (c['left_condition'], c['right_condition']) == (left, right)
                items = []
                for i in idx:
                    x, y = preds[left][i], preds[right][i]
                    diff = sum(k != v for k, v in zip(x, y))
                    items.append((diff, len(x), abs(x.count('Y') - y.count('Y')) / len(x), diff > 0,
                                  x.count('Y') != y.count('Y')))
                arr = np.asarray(items, dtype=float)
                assert c['candidate_count'] == int(arr[:, 1].sum()) and c['changed_candidate_count'] == int(arr[:, 0].sum())
                assert c['pools_with_any_label_change'] == int(arr[:, 3].sum()) and c['pools_with_changed_ratio'] == int(arr[:, 4].sum())
                assert abs(c['candidate_label_change_fraction'] - arr[:, 0].sum() / arr[:, 1].sum()) < 1e-12
                assert abs(c['mean_absolute_pool_ratio_change'] - arr[:, 2].mean()) < 1e-12
                flip = arr[draw, 0].sum(axis=1) / arr[draw, 1].sum(axis=1)
                shift = arr[draw, 2].mean(axis=1)
                assert np.allclose(np.quantile(flip, [.025, .975]), c['candidate_label_change_fraction_ci'], rtol=0, atol=1e-12)
                assert np.allclose(np.quantile(shift, [.025, .975]), c['mean_absolute_pool_ratio_change_ci'], rtol=0, atol=1e-12)
            prior = json.loads((a.renumbered / 'paired_analysis.json').read_text())['datasets'][name]
            if prior['common_query_ids'] == d['common_query_ids']:
                assert prior['changed_candidate_count'] == d['comparisons']['reverse_reassigned_ids']['changed_candidate_count']
        print(name, 'common pools',len(idx), 'control contrasts verified')
    print('Stable IDs, identical replay prompts, candidate alignment and paired bootstrap independently verified.')
if __name__ == '__main__':
    main()
