#!/usr/bin/env python3
"""Check planned plotting input extraction on the six already observed analyses."""
import argparse,math
from pathlib import Path
from plot_7b_sensitivity import product,trec,normalize
ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parent);b=ap.parse_args().artifact
checks=[('analysis_esci_selective_holdout_qwen/summary.json',product,-.034242926632228665,250),
 ('analysis_esci_selective_holdout_smollm/summary.json',product,.01082698412698413,250),
 ('run_wands_selective_execution_qwen_v1/paired_analysis.json',product,-.01682692307692306,52),
 ('run_wands_selective_execution_smollm_v1/paired_analysis.json',product,.005408653846153855,52),
 ('TREC_COVID_TRANSFER_ANALYSIS.json',trec,-.000625,50),('TREC_COVID_SMOL_ANALYSIS.json',trec,.015625,50)]
for path,parser,difference,n in checks:
 x=normalize(parser(b/path))
 assert math.isclose(x['difference'],difference,abs_tol=1e-13) and x['pools']==n and len(x['interval'])==2
assert normalize(None) is None
for x in [{'difference':float('nan'),'interval':[0,1],'pools':1}, {'difference':0,'interval':[1,-1],'pools':1}, {'difference':0,'interval':[0,1],'pools':0}]:
 try:normalize(x)
 except ValueError:pass
 else:raise AssertionError('Invalid figure data accepted')
print('All six previous plotted contrasts, sample sizes, missing values and invalid data checks passed.')
