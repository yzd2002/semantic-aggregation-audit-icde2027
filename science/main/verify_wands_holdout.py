#!/usr/bin/env python3
"""Independently verify holdout prompts, labels, baselines, and the frozen contrast."""
import argparse, collections, hashlib, json, re
from pathlib import Path
import numpy as np
import run_experiment as base
import run_full_pools as harness
from run_compact_prompt_ablation import lines, prompt_without_example
from run_validation import parse_compact
from verify_compact_prompt_ablation import metrics

def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def close(a, b):
    for k, v in a.items():
        assert (v is None and b[k] is None) or (v is not None and abs(v-b[k]) < 1e-10), (k, v, b[k])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    selection = json.loads((a.selection / 'selection.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    rows = lines(a.selection / 'inputs.jsonl')
    selection_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    assert summary['protocol'] == analysis['protocol'] == selection['protocol'] == 'PROTOCOL_wands_holdout.md'
    assert summary['selection_inputs_sha256'] == analysis['selection_inputs_sha256'] == selection_sha
    assert summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)' and summary['mode_order'] == ['original', 'replay', 'stable_reverse']
    assert parse_compact(sanity['unconstrained'], 4) is None and parse_compact(sanity['guided_regex_4'], 4) is not None
    assert summary['specifications']['max_prompt_with_retry_plus_cap'] <= 4096
    assert summary['max_candidate_prompt_with_retry_plus_cap'] <= 4096
    specs = lines(a.run / 'specifications.jsonl')
    assert len(specs) == len(rows) == 160
    first_bad = final_bad = 0
    for row, rec in zip(rows, specs):
        assert row['query_id'] == rec['query_id'] and rec['prompt_sha256'] == digest(base.p_spec(row))
        first = base.parse_spec(rec['first']);final = base.parse_spec(rec['final'])
        first_bad += first is None;final_bad += final is None
        if first is not None:assert rec['first'] == rec['final']
        assert rec['parsed'] == final and rec['fallback'] == (final is None)
    assert summary['specifications']['retry_calls'] == first_bad and summary['specifications']['fallback_pools'] == final_bad
    outputs = {mode: lines(a.run / (mode + '_outputs.jsonl')) for mode in summary['mode_order']}
    assert all(len(v) == len(rows) for v in outputs.values())
    preds = {}
    for mode, saved in outputs.items():
        completed = [];first_valid = final_valid = 0
        for row, spec, result in zip(rows, specs, saved):
            assert row['query_id'] == result['query_id']
            parts = harness.chunks(row)
            assert len(parts) == len(result['raw']) == 4
            pred = [None] * len(row['items'])
            for part, raw in zip(parts, result['raw']):
                start, n = part['start'], len(part['items'])
                assert raw['start'] == start and raw['size'] == n
                rev = {'query': part['query'], 'start': start, 'items': [dict(part['items'][j]) for j in range(n-1, -1, -1)]}
                assert [it['id'] for it in rev['items']] == list(range(n, 0, -1))
                original_prompt = prompt_without_example(part, spec['parsed'])
                condition_prompt = prompt_without_example(rev if mode == 'stable_reverse' else part, spec['parsed'])
                assert raw['original_prompt_sha256'] == digest(original_prompt)
                assert raw['condition_prompt_sha256'] == digest(condition_prompt)
                positions = list(range(start+n-1, start-1, -1)) if mode == 'stable_reverse' else list(range(start, start+n))
                assert raw['original_indices_in_prompt_order'] == positions
                first, final = parse_compact(raw['first'], n), parse_compact(raw['final'], n)
                first_valid += first is not None;final_valid += final is not None
                if first is not None:assert raw['first'] == raw['final'] and raw['retry_finish_reason'] is None
                if final is not None:
                    for i, label in zip(positions, final):
                        assert pred[i] is None
                        pred[i] = label
            pred = None if None in pred else pred
            assert pred == result['pred']
            completed.append(pred)
        stats = summary['modes'][mode]
        assert stats['valid_first_chunks'] == first_valid and stats['valid_final_chunks'] == final_valid
        assert stats['retry_chunk_calls'] == stats['chunks'] - first_valid
        assert stats['complete_pools'] == sum(x is not None for x in completed)
        preds[mode] = completed
    for row, original, replay in zip(rows, outputs['original'], outputs['replay']):
        assert all(a['condition_prompt_sha256'] == b['condition_prompt_sha256'] for a, b in zip(original['raw'], replay['raw']))
    preds['always_Y'] = [['Y'] * len(r['items']) for r in rows]
    preds['always_N'] = [['N'] * len(r['items']) for r in rows]
    preds['lexical_all_terms'] = []
    for row in rows:
        terms = set(re.findall(r'\w+', row['query'].casefold()))
        preds['lexical_all_terms'].append(['Y' if terms and terms.issubset(set(re.findall(r'\w+', it['text'].casefold()))) else 'N' for it in row['items']])
    idx = [i for i in range(len(rows)) if all(preds[k][i] is not None for k in summary['mode_order'])]
    assert analysis['common_complete_pools'] == len(idx) and analysis['common_query_ids'] == [rows[i]['query_id'] for i in idx]
    assert analysis['completion'] == {k: sum(x is not None for x in preds[k]) for k in summary['mode_order']}
    assert analysis['grade_counts'] == dict(collections.Counter(it['grade'] for row in rows for it in row['items']))
    assert analysis['specification_fallback_pools'] == final_bad and analysis['resamples'] == 4000 and analysis['seed'] == 20271123
    assert analysis['primary_threshold'] == .10 and analysis['primary_model'] == (summary['model'] == 'Qwen3-4B-Instruct-2507')
    if idx:
        for key, vals in preds.items():close(metrics(rows, vals, idx), analysis['quality_common'][key])
        rng = np.random.default_rng(20271123)
        draw = rng.integers(0, len(idx), size=(4000, len(idx)))
        for label, left, right in [('identical_prompt_replay', 'original', 'replay'), ('reverse_stable_ids', 'replay', 'stable_reverse')]:
            c = analysis['comparisons'][label]
            assert c['left_condition'] == left and c['right_condition'] == right
            values = []
            for i in idx:
                x, y = preds[left][i], preds[right][i]
                changed = sum(u != v for u, v in zip(x, y))
                values.append((changed, len(x), abs(x.count('Y')-y.count('Y')) / len(x), changed > 0, x.count('Y') != y.count('Y')))
            vals = np.array(values, dtype=float)
            assert c['candidate_count'] == int(vals[:, 1].sum()) and c['changed_candidate_count'] == int(vals[:, 0].sum())
            assert c['pools_with_any_label_change'] == int(vals[:, 3].sum()) and c['pools_with_changed_ratio'] == int(vals[:, 4].sum())
            assert abs(c['candidate_label_change_fraction'] - vals[:, 0].sum()/vals[:, 1].sum()) < 1e-12
            assert abs(c['mean_absolute_pool_ratio_change'] - vals[:, 2].mean()) < 1e-12
            assert abs(c['fixed_reference_max_candidate_error_floor'] - vals[:, 0].sum()/(2*vals[:, 1].sum())) < 1e-12
            rate = vals[draw, 0].sum(axis=1)/vals[draw, 1].sum(axis=1)
            shift = vals[draw, 2].mean(axis=1)
            assert np.allclose(np.quantile(rate, [.025,.975]), c['candidate_label_change_fraction_ci'], rtol=0, atol=1e-12)
            assert np.allclose(np.quantile(shift, [.025,.975]), c['mean_absolute_pool_ratio_change_ci'], rtol=0, atol=1e-12)
        expected = analysis['comparisons']['reverse_stable_ids']['candidate_label_change_fraction_ci'][0] > .10
        assert analysis['primary_criterion_met'] == (expected if analysis['primary_model'] else None)
    else:assert analysis['primary_criterion_met'] is None
    print('Holdout specs, exact prompts, stable IDs, candidate mapping, completions, metrics and bootstrap independently verified.')
    print('Common pools:',len(idx),'fallback specifications:',final_bad,'model:',summary['model'])
if __name__ == '__main__':main()
