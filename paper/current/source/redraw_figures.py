#!/usr/bin/env python3
"""Redraw the four paper figures from supplied CSVs; preserve frozen TeX."""
from pathlib import Path
import csv,json,re,subprocess,sys
sys.dont_write_bytecode=True
import paper_figures as r
M=Path(__file__).resolve().parent
def typed(v):
    if v=='':return None
    if v in ['True','False']:return v=='True'
    if re.fullmatch(r'-?\d+',v):return int(v)
    try:return float(v)
    except ValueError:return v
def read(name):
    with (M/'figure_inputs'/name).open() as f:return [{k:typed(v) for k,v in row.items()} for row in csv.DictReader(f)]
rows=read('all_main_policy_readouts.csv');brows=read('all_budget_policy_readouts.csv');costs=read('all_budget_cost_components.csv')
assert len(rows)==360 and len({(x['model'],x['case_id']) for x in rows})==36
assert len(brows)==32 and len({(x['model'],x['job_id']) for x in brows})==16 and len(costs)==48
r.D=M/'regenerated/data_exports';r.T=M/'regenerated/table_exports'
for p in [r.D,r.T,r.F]:p.mkdir(parents=True,exist_ok=True)
r.main_plot(rows);r.budget_plot(brows,{'costs':costs},rows);r.cache_plot(rows)
subprocess.run([sys.executable,'-B',str(M/'draw_overview.py')],check=True)
print('Four figures redrawn; frozen TeX captions and author block preserved.')
