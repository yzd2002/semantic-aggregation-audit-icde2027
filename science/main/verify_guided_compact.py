#!/usr/bin/env python3
"""Independently reconstruct strict outputs and compare recorded metrics."""
import argparse, json
from pathlib import Path
from run_compact_prompt_ablation import lines
from run_full_pools import chunks
from run_validation import parse_compact
from verify_compact_prompt_ablation import metrics

def equal_metrics(a, b):
    for k, v in a.items():
        if v is None:
            assert b[k] is None
        else:
            assert abs(v - b[k]) < 1e-10, (k, v, b[k])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guided-run', type=Path, required=True)
    ap.add_argument('--unconstrained-run', type=Path, required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.guided_run / 'summary.json').read_text())
    paired = json.loads((a.guided_run / 'paired_analysis.json').read_text())
    assert summary['protocol'] == paired['protocol'] == 'PROTOCOL_guided_compact.md'
    assert summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    sanity = json.loads((a.guided_run / 'guidance_sanity.json').read_text())
    assert parse_compact(sanity['unconstrained'], 4) is None
    assert parse_compact(sanity['guided_regex_4'], 4) is not None
    baseline_log = Path(str(a.unconstrained_run) + '.log')
    if baseline_log.exists():
        assert 'V1 LLM engine' in baseline_log.read_text(encoding='utf8'), 'Unconstrained engine is not the matched V1 engine.'
    for name, source in [('esci', a.esci_run), ('wands', a.wands_run)]:
        rows = lines(source / 'inputs.jsonl')
        previous = lines(source / 'outputs.jsonl')
        modes = {'guided': lines(a.guided_run / (name + '_outputs.jsonl')),
                 'unconstrained': lines(a.unconstrained_run / (name + '_outputs.jsonl'))}
        preds, first_count, final_count = {}, 0, 0
        for mode, outs in modes.items():
            assert len(rows) == len(outs)
            recovered = []
            for row, out in zip(rows, outs):
                assert row['query_id'] == out['query_id']
                expected = [(c['start'], len(c['items'])) for c in chunks(row)]
                assert [(c['start'], c['size']) for c in out['raw']] == expected
                labels = [None] * len(row['items'])
                for c in out['raw']:
                    x = parse_compact(c['final'], c['size'])
                    if mode == 'guided':
                        first_count += parse_compact(c['first'], c['size']) is not None
                        final_count += x is not None
                        if parse_compact(c['first'], c['size']) is not None:
                            assert c['first'] == c['final'] and c['retry_finish_reason'] is None
                    if x is not None:
                        labels[c['start']:c['start'] + c['size']] = x
                pred = None if None in labels else labels
                assert pred == out['pred']
                recovered.append(pred)
            preds[mode] = recovered
        assert first_count == summary['datasets'][name]['valid_first_chunks']
        assert final_count == summary['datasets'][name]['valid_final_chunks']
        idx = {k: [i for i, x in enumerate(val) if x is not None] for k, val in preds.items()}
        common = sorted(set(idx['guided']) & set(idx['unconstrained']))
        d = paired['datasets'][name]
        assert summary['datasets'][name]['complete_pools'] == d['guided_complete_pools'] == len(idx['guided'])
        assert d['unconstrained_complete_pools'] == len(idx['unconstrained'])
        assert d['common_complete_pools'] == len(common)
        for mode in preds:
            if idx[mode]:
                equal_metrics(metrics(rows, preds[mode], idx[mode]), d['quality_own_complete'][mode])
            if common:
                equal_metrics(metrics(rows, preds[mode], common), d['quality_common_complete'][mode])
        if common:
            pointwise = [x['plans']['pointwise']['pred'] for x in previous]
            equal_metrics(metrics(rows, pointwise, common), d['quality_common_complete']['pointwise'])
        print(name, 'guided complete', len(idx['guided']), 'unconstrained complete', len(idx['unconstrained']), 'common', len(common), 'OK')
    print('Guided outputs, chunk indices, completions, and paired quality independently verified.')
if __name__ == '__main__':
    main()
