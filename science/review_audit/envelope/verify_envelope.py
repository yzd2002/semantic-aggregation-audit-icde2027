#!/usr/bin/env python3
"""Independent position maps and attainable-count DP, without analyzer imports."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility')
LEGACY = Path('/dev/shm/intentsat-review-20261102/methods/consensus_audit.json')
JOBS = (
    ('WANDS/Qwen', 'wands_selective_execution_selection', 'run_wands_selective_execution_qwen_v1', 52),
    ('WANDS/SmolLM2', 'wands_selective_execution_selection', 'run_wands_selective_execution_smollm_v1', 52),
    ('ESCI/Qwen', 'esci_selective_holdout_selection', 'run_esci_selective_holdout_qwen', 250),
    ('ESCI/SmolLM2', 'esci_selective_holdout_selection', 'run_esci_selective_holdout_smollm', 250),
)

def load(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]

def same_number(value, saved):
    assert abs(float(value)-saved) < 1e-12, (value, saved)

def compare_fraction(value, saved):
    assert value == Fraction(*saved), (value, saved)

def keyed_positions(path, reverse):
    mapped = {}; chunks = set()
    for rec in load(path):
        chunk = (rec['query_id'], rec['start']); assert chunk not in chunks; chunks.add(chunk)
        parsed = json.loads(rec['final'], object_pairs_hook=lambda pairs: pairs)
        assert isinstance(parsed, list) and all(isinstance(x, tuple) and len(x) == 2 for x in parsed)
        keys = [int(k) for k, _ in parsed]
        assert keys == list(range(1, rec['size']+1)) and all(v in ('Y', 'N') for _, v in parsed)
        assert rec['presented_ids'] == sorted(keys, reverse=reverse)
        for key, value in parsed:
            position = (rec['query_id'], rec['start'] + int(key)-1)
            assert position not in mapped; mapped[position] = value == 'Y'
    return mapped

def verify():
    frozen = json.loads((ROOT/'PRECOMPUTATION_SHA256.json').read_text())
    for path, want in frozen['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == want
    summary = json.loads((ROOT/'envelope_summary.json').read_text()); prior = json.loads(LEGACY.read_text())
    saved = load(ROOT/'envelope_per_pool.jsonl'); table = {(r['setting'], r['query_id']): r for r in saved}
    assert len(table) == len(saved) == 604
    assert summary['source_sha256'] == frozen['files']
    verified = 0
    for name, selection, run, size in JOBS:
        rows = load(DATA/selection/'inputs.jsonl'); assert len(rows) == size
        expected = {(row['query_id'], i) for row in rows for i in range(len(row['items']))}
        left = keyed_positions(DATA/run/'original_responses.jsonl', False)
        right = keyed_positions(DATA/run/'stable_reverse_responses.jsonl', True)
        assert set(left) == set(right) == expected
        disagree = {key for key in expected if left[key] != right[key]}
        checker = {}
        for rec in load(DATA/run/'selective_pointwise_responses.jsonl'):
            key = (rec['query_id'], rec['start']); assert key not in checker and rec['final'] in ('Y', 'N') and rec['size'] == 1
            checker[key] = rec['final'] == 'Y'
        assert set(checker) == disagree
        actual_saved = {r['query_id']: r['pred'] for r in load(DATA/run/'selective_outputs.jsonl')}
        prior_rows = {r['query_id']: r for r in prior['settings'][name]['per_pool']}
        computed = []
        for row in rows:
            qid = row['query_id']; n = len(row['items']); old = table[name, qid]
            truth = [int(i['grade'] in ({'Exact'} if name.startswith('WANDS') else {'E', 'S'})) for i in row['items']]
            assert truth == [int(i['gold'] == 'Y') for i in row['items']]
            a = [int(left[qid, i]) for i in range(n)]; b = [int(right[qid, i]) for i in range(n)]
            actual = [int(checker.get((qid, i), left[qid, i])) for i in range(n)]
            assert actual_saved[qid] == ['Y' if v else 'N' for v in actual]
            lower = sum(1 for u, v in zip(a, b) if u+v == 2)
            upper = sum(1 for u, v in zip(a, b) if u+v >= 1); gold = sum(truth); pred = sum(actual)
            for key, value in {'n': n, 'L': lower, 'U': upper, 'G': gold, 'P': pred, 'disagreements': sum(u != v for u, v in zip(a, b))}.items():
                assert old[key] == value
            for field, vector in [('canonical', a), ('reversed', b), ('actual_selective', actual), ('reference', truth)]:
                assert old['labels'][field] == ''.join(map(str, vector))
            # Independent DP: each agreement has one possible decision; each disagreement has two.
            possible = {0}
            for u, v in zip(a, b):
                choices = {u} if u == v else {0, 1}
                possible = {p+c for p in possible for c in choices}
            assert possible == set(range(lower, upper+1)) and pred in possible
            optimum_count = min(abs(gold-p) for p in possible); floor = Fraction(optimum_count, n)
            flags = {'collapsed_exact': lower == upper == gold,
                'at_or_below_lower': gold <= lower and not lower == upper == gold,
                'at_or_above_upper': gold >= upper and not lower == upper == gold,
                'strictly_inside': lower < gold < upper}
            assert sum(flags.values()) == 1; group = next(k for k, v in flags.items() if v)
            assert old['group'] == group and old['floor_count'] == optimum_count
            compare_fraction(floor, old['floor_exact']); same_number(floor, old['floor'])
            item = {'group': group, 'n': n, 'L': lower, 'U': upper, 'G': gold, 'floor': floor, 'policies': {}}
            for label, p in [('and', lower), ('or', upper), ('actual_selective', pred)]:
                loss = Fraction(abs(gold-p), n); regret = loss-floor; target = old['policies'][label]
                assert target['positive_count'] == p and target['error_count'] == abs(gold-p) and target['regret_count'] == abs(gold-p)-optimum_count
                compare_fraction(loss, target['error_exact']); compare_fraction(regret, target['regret_exact'])
                same_number(loss, target['error']); same_number(regret, target['regret'])
                old_label = 'selective' if label == 'actual_selective' else label
                same_number(loss, prior_rows[qid]['metrics'][old_label]['ratio_mae'])
                item['policies'][label] = (loss, regret)
            computed.append(item); verified += 1
        def check_summary(items, reported):
            assert reported['pools'] == len(items)
            if not items:
                assert reported['mean_floor'] is None and reported['mean_floor_exact'] is None
                for p in ('and', 'or', 'actual_selective'):
                    assert reported['policies'][p]['mean_error'] is None and reported['policies'][p]['mean_regret'] is None
                    assert reported['policies'][p]['floor_attaining_pools'] == 0
                return
            assert reported['candidates'] == sum(x['n'] for x in items)
            average = sum(x['floor'] for x in items)/len(items)
            compare_fraction(average, reported['mean_floor_exact']); same_number(average, reported['mean_floor'])
            assert reported['exact_ratio_attainable_pools'] == sum(x['floor'] == 0 for x in items)
            assert reported['strictly_outside_pools'] == sum(not x['L'] <= x['G'] <= x['U'] for x in items)
            assert reported['no_disagreement_pools'] == sum(x['L'] == x['U'] for x in items)
            for p in ('and', 'or', 'actual_selective'):
                target = reported['policies'][p]
                for j, tag in enumerate(('error', 'regret')):
                    avg = sum(x['policies'][p][j] for x in items)/len(items)
                    compare_fraction(avg, target['mean_'+tag+'_exact']); same_number(avg, target['mean_'+tag])
                assert target['floor_attaining_pools'] == sum(x['policies'][p][1] == 0 for x in items)
        report = summary['settings'][name]; check_summary(computed, report)
        for group, entry in report['groups'].items():
            subset = [x for x in computed if x['group'] == group]; check_summary(subset, entry)
            same_number(Fraction(len(subset), len(computed)), entry['share'])
        assert sum(x['pools'] for x in report['groups'].values()) == size
        for p in ('and', 'or', 'actual_selective'):
            label = 'selective' if p == 'actual_selective' else p
            same_number(sum(x['policies'][p][0] for x in computed)/size, prior['settings'][name]['metrics'][label]['ratio_mae'])
        print('PASS', name, ': source vectors, every integer count, attainable-count DP, exclusive groups, rational losses/regrets, all summaries and legacy MAEs.')
    assert verified == summary['pool_model_instances'] == 604
    assert len({(r['setting'].split('/')[0], r['query_id']) for r in saved}) == summary['distinct_dataset_query_pools'] == 302
    print('PASS: all 604 pool-model records / 302 distinct pools independently verified; all frozen sources and code unchanged.')

if __name__ == '__main__':
    verify()
