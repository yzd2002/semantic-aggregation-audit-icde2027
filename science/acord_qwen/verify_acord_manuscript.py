#!/usr/bin/env python3
"""Check the ACORD manuscript excerpts against saved analyses, without inference."""
import argparse, json, re
from pathlib import Path


def main(manuscript, analysis, supplement, summary, reproducibility):
    tex = manuscript.read_text(encoding='utf8')
    a = json.loads(analysis.read_text());x = json.loads(supplement.read_text());s = json.loads(summary.read_text())
    c = a['scenarios']['case_control'];u = a['scenarios']['uniform'];cp = x['scenarios']['case_control']
    assert c['common_complete_pools'] == u['common_complete_pools'] == cp['joint_complete'] == 57
    for name in ('case_control', 'uniform'):
        assert all(v == 57 for v in a['scenarios'][name]['completion'].values())
    assert s['completed'] and s['selective_pipeline']['initial_calls'] == 1234 and s['modes']['pointwise']['initial_calls'] == 3648
    expected = [f"Qwen selective MAE is {c['metrics']['selective']['ratio_mae']:.3f} versus {c['metrics']['original']['ratio_mae']:.3f} keyed and {cp['metrics']['pointwise']['ratio_mae']:.3f} pointwise",
                f"keyed and selective MAE are {c['metrics']['original']['ratio_mae']:.3f} and {c['metrics']['selective']['ratio_mae']:.3f}",
                f"selective minus keyed {c['selective_minus']['original']['ratio_mae']:.4f}",
                f"[{c['selective_minus']['original']['query_bootstrap_95'][0]:.4f},{c['selective_minus']['original']['query_bootstrap_95'][1]:.4f}]",
                f"MAEs are {cp['metrics']['pointwise']['ratio_mae']:.3f} and {c['metrics']['lexical_fallback']['ratio_mae']:.3f}",
                f"selective MAE {u['metrics']['selective']['ratio_mae']:.3f} exceeds all-N {u['metrics']['always_N']['ratio_mae']:.3f}",
                '195 and 251 of 1,824 candidate occurrences remain truncated',
                'Selective uses 1,234 logical calls versus 3,648 pointwise']
    for text in expected:
        assert text in tex, text
    # Some previously rendered summaries are now static compact prose; verify
    # their rounded numbers against the same saved JSON as the original renderer.
    e = json.loads((reproducibility / 'POSTHOC_7B_ESCI.json').read_text())
    w = json.loads((reproducibility / 'POSTHOC_7B_WANDS.json').read_text())
    t = json.loads((reproducibility / 'POSTHOC_7B_TREC.json').read_text())
    eg = json.loads((reproducibility / 'POSTHOC_7B_ESCI_GUIDED.json').read_text())
    wg = json.loads((reproducibility / 'POSTHOC_7B_WANDS_GUIDED.json').read_text())
    weight = json.loads((reproducibility / 'posthoc_product_weight/product_weight_summary.json').read_text())['datasets']
    def mae(d, key): return f"{d['quality_common'][key]['ratio_mae']:.3f}"
    assert f"ESCI keyed/selective MAEs are {mae(e, 'original')}/{mae(e, 'actual_fallback')} versus {mae(e, 'always_Y')} all-Y" in tex
    assert f"WANDS MAEs are {mae(w, 'original')}/{mae(w, 'actual_fallback')}" in tex
    assert f"MAE rises from {t['metrics']['original']['ratio_mae']:.3f} to {t['metrics']['selective']['ratio_mae']:.3f}" in tex
    for d in (eg, wg):
        x1 = d['comparison'];assert f"from {x1['old_unguided_mae']:.3f} to {x1['new_schema_guided_mae']:.3f}" in tex
        assert f"[{x1['paired_query_95_percentile_ci'][0]:.3f},{x1['paired_query_95_percentile_ci'][1]:.3f}]" in tex
    for d in (weight['ESCI'], weight['WANDS']):
        assert f"[{d['sensitivity_range_95'][0]:.3f},{d['sensitivity_range_95'][1]:.3f}]" in tex
    body, refs = tex.split('\\begin{thebibliography}', 1)
    order = []
    for match in re.finditer(r'\\cite\{([^}]+)\}', body):
        for key in match.group(1).split(','):
            if key not in order:order.append(key)
    assert order == re.findall(r'\\bibitem\{([^}]+)\}', refs) and order[-1] == 'acord'
    print('Manuscript matches saved ACORD and compacted 7B metrics; all', len(order), 'citations follow first-mention order.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ('manuscript', 'analysis', 'supplement', 'summary', 'reproducibility'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args();main(a.manuscript, a.analysis, a.supplement, a.summary, a.reproducibility)
