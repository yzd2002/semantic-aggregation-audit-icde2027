#!/usr/bin/env python3
"""Independently reconstruct positions, prompts, labels and paired statistics."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import run_experiment as base
import run_full_pools as harness
from run_compact_prompt_ablation import lines, prompt_without_example
from run_validation import parse_compact
from verify_compact_prompt_ablation import metrics

def close_dict(a, b):
    for key, value in a.items():
        if value is None:
            assert b[key] is None
        else:
            assert abs(value - b[key]) < 1e-10, (key, value, b[key])

def digest(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    control = json.loads((a.run / 'guidance_sanity.json').read_text())
    assert summary['protocol'] == analysis['protocol'] == 'PROTOCOL_order_sensitivity.md'
    assert summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)' and summary['baseline'] == a.baseline.name
    assert parse_compact(control['unconstrained'], 4) is None and parse_compact(control['guided_regex_4'], 4) is not None
    assert summary['max_prompt_plus_output_cap'] <= 4096
    for name, source in [('esci', a.esci_run), ('wands', a.wands_run)]:
        rows = lines(source / 'inputs.jsonl')
        previous = lines(source / 'outputs.jsonl')
        old = lines(a.baseline / (name + '_outputs.jsonl'))
        new = lines(a.run / (name + '_outputs.jsonl'))
        assert len(rows) == len(previous) == len(old) == len(new)
        recovered, first_valid, final_valid = [], 0, 0
        for row, prior, b, result in zip(rows, previous, old, new):
            assert row['query_id'] == prior['query_id'] == b['query_id'] == result['query_id']
            entry = prior['plans']['two_stage_compact_fallback']
            spec = base.parse_spec(entry['spec']['final'])
            assert bool(entry['fallback']) == (spec is None)
            expected = harness.chunks(row)
            assert len(expected) == len(result['raw']) == len(b['raw'])
            pred = [None] * len(row['items'])
            for part, raw, original_raw in zip(expected, result['raw'], b['raw']):
                start, n = part['start'], len(part['items'])
                assert raw['start'] == original_raw['start'] == start and raw['size'] == original_raw['size'] == n
                positions = [start + j for j in range(n - 1, -1, -1)]
                assert raw['original_indices_in_prompt_order'] == positions
                rev = {'query': part['query'], 'start': start,
                       'items': [{**part['items'][j], 'id': k + 1} for k, j in enumerate(range(n - 1, -1, -1))]}
                assert raw['original_prompt_sha256'] == digest(prompt_without_example(part, spec))
                assert raw['reversed_prompt_sha256'] == digest(prompt_without_example(rev, spec))
                first = parse_compact(raw['first'], n)
                final = parse_compact(raw['final'], n)
                first_valid += first is not None
                final_valid += final is not None
                if first is not None:
                    assert raw['final'] == raw['first'] and raw['retry_finish_reason'] is None
                if final is not None:
                    for pos, label in zip(positions, final):
                        assert pred[pos] is None
                        pred[pos] = label
            pred = None if None in pred else pred
            assert pred == result['pred']
            recovered.append(pred)
        assert summary['datasets'][name]['valid_first_chunks'] == first_valid
        assert summary['datasets'][name]['valid_final_chunks'] == final_valid
        before = [x['pred'] for x in old]
        idx = [i for i, (x, y) in enumerate(zip(before, recovered)) if x is not None and y is not None]
        d = analysis['datasets'][name]
        assert len(idx) == d['common_complete_pools']
        assert d['common_query_ids'] == [rows[i]['query_id'] for i in idx]
        assert sum(x is not None for x in recovered) == d['reversed_complete_pools'] == summary['datasets'][name]['complete_pools']
        if idx:
            close_dict(metrics(rows, before, idx), d['original_quality_common'])
            close_dict(metrics(rows, recovered, idx), d['reversed_quality_common'])
            old_labels = [z for i in idx for z in before[i]]
            new_labels = [z for i in idx for z in recovered[i]]
            changed = sum(x != y for x, y in zip(old_labels, new_labels))
            any_changed = sum(before[i] != recovered[i] for i in idx)
            ratio_changed = sum(before[i].count('Y') != recovered[i].count('Y') for i in idx)
            ratios = np.array([abs(before[i].count('Y') - recovered[i].count('Y')) / len(before[i]) for i in idx])
            assert len(old_labels) == d['common_candidate_count'] and changed == d['changed_candidate_count']
            assert any_changed == d['pools_with_any_label_change'] and ratio_changed == d['pools_with_changed_ratio']
            assert any_changed - ratio_changed == d['pools_changed_labels_same_ratio']
            assert abs(changed / len(old_labels) - d['candidate_label_change_fraction']) < 1e-12
            assert abs(float(ratios.mean()) - d['mean_absolute_pool_ratio_change']) < 1e-12
            assert abs(float(np.quantile(ratios, .95)) - d['p95_absolute_pool_ratio_change']) < 1e-12
            assert sum(x == 'Y' and y == 'N' for x, y in zip(old_labels, new_labels)) == d['yes_to_no_changes']
            assert sum(x == 'N' and y == 'Y' for x, y in zip(old_labels, new_labels)) == d['no_to_yes_changes']
            agree = [i for i, (x, y) in enumerate(zip(old_labels, new_labels)) if x == y]
            flattened_gold = [it['gold'] for i in idx for it in rows[i]['items']]
            agreement = d['agreement_only_diagnostic']
            wrong = sum(old_labels[i] != flattened_gold[i] for i in agree)
            assert agreement['accepted_candidates'] == len(agree)
            assert agreement['abstained_candidates'] == len(old_labels) - len(agree)
            assert agreement['accepted_reference_disagreements'] == wrong
            assert (agreement['accepted_reference_error_rate'] is None if not agree else abs(agreement['accepted_reference_error_rate'] - wrong / len(agree)) < 1e-12)
            inside, widths = 0, []
            for i in idx:
                g = [it['gold'] for it in rows[i]['items']]
                matched = [x for x, y in zip(before[i], recovered[i]) if x == y]
                p = matched.count('Y'); n = len(g); m = len(matched)
                inside += p <= g.count('Y') <= p + n - m
                widths.append((n - m) / n)
            assert agreement['reference_ratio_inside_k0_count'] == inside
            assert agreement['attempted_pools'] == len(idx)
            assert abs(agreement['mean_k0_interval_width'] - sum(widths) / len(widths)) < 1e-12
            # Rebuild the paired query-cluster bootstrap from candidate labels,
            # independently of the analysis program's record construction.
            blocks = []
            for i in idx:
                q = [it['gold'] for it in rows[i]['items']]
                def one(v):
                    return [sum(g == 'Y' and p == 'Y' for g, p in zip(q, v)),
                            sum(g == 'N' and p == 'Y' for g, p in zip(q, v)),
                            sum(g == 'Y' and p == 'N' for g, p in zip(q, v)),
                            abs(v.count('Y') / len(v) - q.count('Y') / len(q))]
                blocks.append([*one(before[i]), *one(recovered[i]),
                               sum(x != y for x, y in zip(before[i], recovered[i])), len(q),
                               abs(before[i].count('Y') - recovered[i].count('Y')) / len(q)])
            block = np.array(blocks, dtype=float)
            rng = np.random.default_rng(20271119)
            draw = rng.integers(0, len(block), size=(4000, len(block)))
            def bootf(offset):
                c = block[draw, offset:offset + 3].sum(axis=1)
                den = 2 * c[:, 0] + c[:, 1] + c[:, 2]
                return np.divide(2 * c[:, 0], den, out=np.zeros_like(den), where=den != 0)
            boot = d['paired_bootstrap']
            intervals = {'f1_reversed_minus_original_ci': bootf(4) - bootf(0),
                         'mae_reversed_minus_original_ci': block[draw, 7].mean(axis=1) - block[draw, 3].mean(axis=1),
                         'candidate_label_change_fraction_ci': block[draw, 8].sum(axis=1) / block[draw, 9].sum(axis=1),
                         'mean_absolute_pool_ratio_change_ci': block[draw, 10].mean(axis=1)}
            assert boot['resamples'] == 4000 and boot['seed'] == 20271119
            for key, values in intervals.items():
                assert np.allclose(np.quantile(values, [.025, .975]), boot[key], rtol=0, atol=1e-12), key
        print(name, 'common pools', len(idx), 'changed decisions', d.get('changed_candidate_count'), 'OK')
    print('Prompt reversal, exact candidate mapping, strict parsing, completion and paired metrics independently verified.')
if __name__ == '__main__':
    main()
