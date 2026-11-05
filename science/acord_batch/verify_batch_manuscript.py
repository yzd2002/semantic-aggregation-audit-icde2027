#!/usr/bin/env python3
"""Check the reported recovery contrast and cost against the audited data."""
import argparse,json
from pathlib import Path
def excerpt(data):
    s=data['scenarios']['case_control'];m=s['metrics'];c=data['all_pool_original_costs'];ci=s['query_bootstrap_95']
    return (f"In a post-hoc recovery with conflicting dependencies, stress-pool four/eight keyed MAEs are {m['four_original']['ratio_mae']:.3f}/{m['eight_original']['ratio_mae']:.3f} "
            f"(difference ${s['four_minus_eight_ratio_mae']:.4f}$, unadjusted 95\\% query interval $[{ci[0]:.4f},{ci[1]:.4f}]$); "
            f"across scenarios they use {c['four']['initial_calls']:,}/{c['eight']['initial_calls']:,} calls and "
            f"{c['four']['logical_tokens']['input']/1e6:.3f}M/{c['eight']['logical_tokens']['input']/1e6:.3f}M input tokens. "
            'Partitions and grammars also differ.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--manuscript',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.analysis.read_text());source=a.manuscript.read_text();assert data['scenarios']['case_control']['common_complete_pools']==57
    assert data['scenarios']['uniform']['common_complete_pools']==57 and all(x['retry_calls']==0 for x in data['all_pool_original_costs'].values())
    assert source.count(excerpt(data))==1,'Printed contrast or cost disagrees with audited records'
    print('Manuscript recovery contrast, confidence interval and token/call totals match the independently checked analysis.')
