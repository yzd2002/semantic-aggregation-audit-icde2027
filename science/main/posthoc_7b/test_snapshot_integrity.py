#!/usr/bin/env python3
"""Positive and negative checks for the frozen pre-7B plotting-script snapshot.

Creates a new, retained output directory (about 46 MB); never edits artifact files.
Requires a POSIX system with symlink support. No model or GPU is required.
"""
import argparse,os,shutil,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact',type=Path,required=True,help='extracted reproducibility directory')
p.add_argument('--out',type=Path,required=True,help='new output directory; retained for audit')
a=p.parse_args();artifact=a.artifact.resolve();extractor=artifact/'posthoc_7b/extract_and_verify.py';snapshot=artifact/'pre_7b_snapshot/make_figures.py'
if not extractor.is_file() or not snapshot.is_file():raise FileNotFoundError('Artifact extractor or historical snapshot missing')
if a.out.exists() or a.out.is_symlink():raise FileExistsError('Use a new output directory')
a.out.mkdir(parents=True,exist_ok=False);out=a.out.resolve();shadow=out/'tampered_artifact';shadow.mkdir()
for source in artifact.iterdir():
 target=shadow/source.name
 if source.name=='pre_7b_snapshot':shutil.copytree(source,target)
 else:target.symlink_to(source,target_is_directory=source.is_dir())
copy=shadow/'pre_7b_snapshot/make_figures.py';copy.write_bytes(copy.read_bytes()+b'\n# Deliberately modified negative control.\n')
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
def execute(name,art,dest):
 cmd=[sys.executable,str(extractor),'--artifact',str(art),'--out',str(dest)]
 r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (out/(name+'.log')).write_text(r.stdout,encoding='utf8');return r
valid=execute('valid',artifact,out/'valid_extraction')
if valid.returncode:raise RuntimeError('Unmodified artifact failed; inspect valid.log')
negative=execute('tampered',shadow,out/'tampered_extraction')
expected='PRE_7B_SOURCE_SHA256.json: frozen file mismatch: artifact/make_figures.py'
if negative.returncode==0 or expected not in negative.stdout:raise RuntimeError('Tampered snapshot was not rejected as expected; inspect tampered.log')
print('PASS: unchanged frozen snapshot verifies, and altered snapshot is rejected.')
print('Results and extracted files retained in',out)
