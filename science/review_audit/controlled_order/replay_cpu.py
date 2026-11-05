#!/usr/bin/env python3
"""Portable CPU re-analysis of retained traces, without model weights or source tables."""
import argparse,importlib,json,shutil,sys
from pathlib import Path

def same(a,b,path='root'):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),path
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(a-b)<=1e-12,(path,a,b)
    else:assert a==b,(path,a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--models',nargs='+',choices=['qwen','smol'],default=['qwen','smol']);a=p.parse_args()
    bundle=a.bundle.resolve();work=a.out.resolve();work.mkdir(parents=True,exist_ok=False)
    for pattern in ('*.py','*.md','*FROZEN_SHA256.json'):
        for f in bundle.glob(pattern):
            if not (work/f.name).exists():shutil.copyfile(f,work/f.name)
    for name in ('selection','prepared'):(work/name).symlink_to(bundle/name,target_is_directory=True)
    for model in a.models:
        source=bundle/f'run_{model}';destination=work/f'run_{model}';destination.mkdir()
        for f in source.iterdir():
            if f.is_file() and f.name not in ('analysis.json','supplementary_analysis.json','EXECUTION_AUDIT.json'):(destination/f.name).symlink_to(f)
    sys.path.insert(0,str(work))
    import common
    common.ROOT=work
    # Raw-trace re-analysis intentionally does not require the inherited source,
    # model or artifact locations. Source-selection verification is a separate step.
    analysis=importlib.import_module('analyze_controlled');supplement=importlib.import_module('supplementary_analysis')
    verify=importlib.import_module('verify_controlled');extra=importlib.import_module('verify_supplement')
    execution=importlib.import_module('audit_execution')
    for model in a.models:
        execution.audit(model);analysis.main(model);supplement.main(model);verify.run_verify(model);extra.main(model)
        for name in ('analysis.json','supplementary_analysis.json','EXECUTION_AUDIT.json'):
            same(json.loads((work/f'run_{model}'/name).read_text()),json.loads((bundle/f'run_{model}'/name).read_text()))
        print('PASS portable CPU replay exactly matches saved results (numeric tolerance 1e-12):',model,flush=True)
if __name__=='__main__':main()
