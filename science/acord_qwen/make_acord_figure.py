#!/usr/bin/env python3
"""Plot verified ACORD fixed-pool summaries; no model inference."""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(analysis, supplement, output):
    primary = json.loads(analysis.read_text())['scenarios']
    extra = json.loads(supplement.read_text())['scenarios']
    for scenario in ('case_control', 'uniform'):
        if primary[scenario]['common_complete_pools'] != extra[scenario]['joint_complete']:
            raise ValueError('Cannot combine estimates from different completion subsets')
    colors = {'original': '#52738a', 'selective': '#b25837', 'pointwise': '#7e9373', 'always_N': '#777777'}
    labels = {'original': 'Keyed', 'selective': 'Selective', 'pointwise': 'Pointwise', 'always_N': 'All N'}
    methods = tuple(labels)
    fig, (ax, ci) = plt.subplots(1, 2, figsize=(8.0, 3.15), gridspec_kw={'width_ratios': [1.15, 1.0]}, layout='constrained')
    y = np.arange(2); height = .16
    for i, method in enumerate(methods):
        vals = [extra[s]['metrics'][method]['ratio_mae'] for s in ('case_control', 'uniform')]
        ax.barh(y + (i - 1.5) * height, vals, height, label=labels[method], color=colors[method])
        for yi, value in zip(y + (i - 1.5) * height, vals):
            ax.annotate(f'{value:.3f}', (value, yi), xytext=(3, 0), textcoords='offset points',
                        va='center', fontsize=7)
    ax.set_yticks(y, [f'Case-control (n={extra["case_control"]["joint_complete"]})',
                       f'Uniform-judged (n={extra["uniform"]["joint_complete"]})'])
    ax.invert_yaxis();ax.set_xlabel('Query-macro ratio MAE (lower is better)')
    ax.set_xlim(0, max(extra[s]['metrics'][m]['ratio_mae'] for s in extra for m in methods) * 1.23 + .01)
    ax.grid(axis='x', color='#dedede', linewidth=.7);ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=4, frameon=False, fontsize=8,
               loc='lower center', bbox_to_anchor=(.5, -.055))
    scenario = 'case_control'; p = primary[scenario]['selective_minus']; e = extra[scenario]
    comparisons = [('Keyed', p['original']['ratio_mae'], p['original']['query_bootstrap_95']),
                   ('Midpoint', p['midpoint']['ratio_mae'], p['midpoint']['query_bootstrap_95']),
                   ('Lexical tie-break', p['lexical_fallback']['ratio_mae'], p['lexical_fallback']['query_bootstrap_95']),
                   ('Pointwise', e['selective_minus_pointwise_ratio_mae'], e['query_bootstrap_95'])]
    for i, (_, value, interval) in enumerate(comparisons):
        ci.hlines(i, interval[0], interval[1], color='#b25837', linewidth=1)
        ci.plot(value, i, 'o', color='#b25837', markersize=4)
    ci.set_yticks(range(len(comparisons)), [x[0] for x in comparisons]);ci.invert_yaxis()
    ci.axvline(0, color='#555555', linewidth=.8, linestyle='--');ci.grid(axis='x', color='#dedede', linewidth=.7)
    ci.set_axisbelow(True);ci.set_xlabel('Selective minus reference MAE\n(unadjusted 95% query-bootstrap interval)')
    ax.set_title('Fixed-pool ratio error', loc='left', fontweight='bold')
    ci.set_title('Case-control paired differences', loc='left', fontweight='bold')
    for a in (ax, ci):
        a.spines[['top', 'right']].set_visible(False);a.tick_params(labelsize=8);a.xaxis.label.set_size(8)
    fig.suptitle('ACORD: one model, sampled judged pools', fontsize=10, fontweight='bold')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=220, bbox_inches='tight')
    print('Saved ACORD supplementary figure; intervals are unadjusted query bootstrap intervals.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--supplement', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();main(a.analysis, a.supplement, a.out)
