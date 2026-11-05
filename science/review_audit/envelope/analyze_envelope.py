#!/usr/bin/env python3
"""Post-hoc consensus count-envelope diagnostic; standard library, no inference."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path('/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility')
AUDIT = Path('/dev/shm/intentsat-review-20261102/methods/consensus_audit.json')
GROUPS = ('collapsed_exact', 'at_or_below_lower', 'at_or_above_upper', 'strictly_inside')
POLICIES = ('and', 'or', 'actual_selective')
SETTINGS = (
    ('WANDS/Qwen', 'wands_selective_execution_selection', 'run_wands_selective_execution_qwen_v1', 52),
    ('WANDS/SmolLM2', 'wands_selective_execution_selection', 'run_wands_selective_execution_smollm_v1', 52),
    ('ESCI/Qwen', 'esci_selective_holdout_selection', 'run_esci_selective_holdout_qwen', 250),
    ('ESCI/SmolLM2', 'esci_selective_holdout_selection', 'run_esci_selective_holdout_smollm', 250),
)
TRACE_FILES = ('original_responses.jsonl', 'stable_reverse_responses.jsonl',
               'selective_pointwise_responses.jsonl', 'selective_outputs.jsonl')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_lines(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]

def save(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def exact(value):
    return [value.numerator, value.denominator]

def batch_vectors(rows, records, reversed_input):
    wanted = [(r['query_id'], start, min(8, len(r['items'])-start))
              for r in rows for start in range(0, len(r['items']), 8)]
    assert [(r['query_id'], r['start'], r['size']) for r in records] == wanted
    vectors = {r['query_id']: [None] * len(r['items']) for r in rows}
    for record in records:
        ids = list(range(1, record['size']+1))
        assert record['presented_ids'] == (ids[::-1] if reversed_input else ids)
        pairs = json.loads(record['final'], object_pairs_hook=list)
        assert [k for k, _ in pairs] == list(map(str, ids))
        assert all(v in ('Y', 'N') for _, v in pairs)
        for key, label in pairs:
            pos = record['start'] + int(key)-1
            assert vectors[record['query_id']][pos] is None
            vectors[record['query_id']][pos] = int(label == 'Y')
    assert all(None not in v for v in vectors.values())
    return vectors

def category(lower, upper, gold):
    if lower == upper == gold:
        return 'collapsed_exact'
    if gold <= lower:
        return 'at_or_below_lower'
    if gold >= upper:
        return 'at_or_above_upper'
    assert lower < gold < upper
    return 'strictly_inside'

def aggregate(rows):
    count = len(rows)
    if not count:
        return {'pools': 0, 'mean_floor': None, 'mean_floor_exact': None,
                'policies': {p: {'mean_error': None, 'mean_regret': None,
                                  'mean_error_exact': None, 'mean_regret_exact': None,
                                  'floor_attaining_pools': 0} for p in POLICIES}}
    floor = sum((Fraction(*r['floor_exact']) for r in rows), Fraction()) / count
    result = {'pools': count, 'candidates': sum(r['n'] for r in rows),
              'mean_floor': float(floor), 'mean_floor_exact': exact(floor),
              'exact_ratio_attainable_pools': sum(r['floor_count'] == 0 for r in rows),
              'strictly_outside_pools': sum(r['G'] < r['L'] or r['G'] > r['U'] for r in rows),
              'no_disagreement_pools': sum(r['L'] == r['U'] for r in rows), 'policies': {}}
    for policy in POLICIES:
        err = sum((Fraction(*r['policies'][policy]['error_exact']) for r in rows), Fraction()) / count
        reg = sum((Fraction(*r['policies'][policy]['regret_exact']) for r in rows), Fraction()) / count
        assert err - floor == reg
        result['policies'][policy] = {'mean_error': float(err), 'mean_regret': float(reg),
            'mean_error_exact': exact(err), 'mean_regret_exact': exact(reg),
            'floor_attaining_pools': sum(r['policies'][policy]['regret_count'] == 0 for r in rows)}
    return result

def main():
    old = json.loads(AUDIT.read_text(encoding='utf8'))
    assert set(old['settings']) == {s[0] for s in SETTINGS}
    paths = {AUDIT, HERE/'PROTOCOL_consensus_envelope.md', HERE/'analyze_envelope.py', HERE/'verify_envelope.py'}
    for name, selection, run, count in SETTINGS:
        assert old['settings'][name]['selection'] == selection and old['settings'][name]['run'] == run
        paths.add(BASE/selection/'inputs.jsonl')
        paths.update(BASE/run/f for f in TRACE_FILES)
    source_hashes = {str(p): digest(p) for p in sorted(paths)}
    for path, h in source_hashes.items():
        p = Path(path)
        if p.is_relative_to(BASE):
            assert old['source_sha256'][str(p.relative_to(BASE))] == h
    save(HERE/'PRECOMPUTATION_SHA256.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Protocol and calculation/verification code fixed before this diagnostic was computed; source outcomes already exposed.',
        'files': source_hashes})
    all_rows = []
    summary = {'design': 'Post-hoc descriptive count envelope with human-reference-dependent hindsight floor; not an online selector or new theorem.',
               'aggregation': 'query-macro within each setting; no confidence intervals',
               'classification_order': list(GROUPS), 'settings': {}, 'source_sha256': source_hashes}
    for name, selection, run, count in SETTINGS:
        rows = read_lines(BASE/selection/'inputs.jsonl')
        assert len(rows) == count and len({r['query_id'] for r in rows}) == count
        a = batch_vectors(rows, read_lines(BASE/run/TRACE_FILES[0]), False)
        b = batch_vectors(rows, read_lines(BASE/run/TRACE_FILES[1]), True)
        point = read_lines(BASE/run/TRACE_FILES[2])
        needed = [(r['query_id'], i) for r in rows for i in range(len(r['items']))
                  if a[r['query_id']][i] != b[r['query_id']][i]]
        assert [(r['query_id'], r['start']) for r in point] == needed
        assert all(r['size'] == 1 and r['final'] in ('Y', 'N') for r in point)
        point_map = {(r['query_id'], r['start']): int(r['final'] == 'Y') for r in point}
        assert len(point_map) == len(point)
        saved = read_lines(BASE/run/TRACE_FILES[3])
        assert [r['query_id'] for r in saved] == [r['query_id'] for r in rows]
        derived = []
        for original, stored, previous in zip(rows, saved, old['settings'][name]['per_pool']):
            qid = original['query_id']; n = len(original['items'])
            assert previous['query_id'] == qid and previous['n'] == n
            assert [i['id'] for i in original['items']] == list(range(1, n+1))
            positives = {'Exact'} if name.startswith('WANDS/') else {'E', 'S'}
            allowed = {'Exact', 'Partial', 'Irrelevant'} if name.startswith('WANDS/') else set('ESCI')
            assert all(i['grade'] in allowed for i in original['items'])
            gold = [int(i['grade'] in positives) for i in original['items']]
            assert gold == [int(i['gold'] == 'Y') for i in original['items']]
            x, y = a[qid], b[qid]
            selective = [u if u == v else point_map[qid, i] for i, (u, v) in enumerate(zip(x, y))]
            assert stored['pred'] == ['Y' if p else 'N' for p in selective]
            assert all(h == u for u, v, h in zip(x, y, selective) if u == v)
            lower = sum(u and v for u, v in zip(x, y)); upper = sum(u or v for u, v in zip(x, y))
            g = sum(gold); p = sum(selective); assert 0 <= lower <= p <= upper <= n
            floor_count = max(lower-g, 0, g-upper)
            result = {'setting': name, 'query_id': qid, 'n': n, 'L': lower, 'U': upper, 'G': g, 'P': p,
                'disagreements': upper-lower, 'group': category(lower, upper, g), 'floor_count': floor_count,
                'floor': float(Fraction(floor_count, n)), 'floor_exact': exact(Fraction(floor_count, n)),
                'labels': {'canonical': ''.join(map(str, x)), 'reversed': ''.join(map(str, y)),
                           'actual_selective': ''.join(map(str, selective)), 'reference': ''.join(map(str, gold))},
                'policies': {}}
            for policy, value in [('and', lower), ('or', upper), ('actual_selective', p)]:
                error_count = abs(value-g); regret_count = error_count-floor_count
                assert regret_count >= 0
                result['policies'][policy] = {'positive_count': value, 'error_count': error_count,
                    'regret_count': regret_count, 'error': error_count/n, 'regret': regret_count/n,
                    'error_exact': exact(Fraction(error_count, n)), 'regret_exact': exact(Fraction(regret_count, n))}
                legacy = 'selective' if policy == 'actual_selective' else policy
                assert abs(previous['metrics'][legacy]['ratio_mae']-error_count/n) < 1e-12
            derived.append(result)
        setting = aggregate(derived)
        setting['groups'] = {g: {**aggregate([r for r in derived if r['group'] == g]),
                                    'share': sum(r['group'] == g for r in derived)/count} for g in GROUPS}
        assert sum(v['pools'] for v in setting['groups'].values()) == count
        for policy, values in setting['policies'].items():
            legacy = 'selective' if policy == 'actual_selective' else policy
            assert abs(values['mean_error']-old['settings'][name]['metrics'][legacy]['ratio_mae']) < 1e-12
        summary['settings'][name] = setting; all_rows.extend(derived)
        print(name, json.dumps({'groups': {k: v['pools'] for k, v in setting['groups'].items()},
              'mean_floor': setting['mean_floor'], 'policies': setting['policies']}, sort_keys=True))
    assert len(all_rows) == 604
    summary['pool_model_instances'] = len(all_rows)
    summary['distinct_dataset_query_pools'] = len({(r['setting'].split('/')[0], r['query_id']) for r in all_rows})
    assert summary['distinct_dataset_query_pools'] == 302
    with (HERE/'envelope_per_pool.jsonl').open('x', encoding='utf8') as f:
        for row in all_rows: f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    save(HERE/'envelope_summary.json', summary)

if __name__ == '__main__':
    main()
