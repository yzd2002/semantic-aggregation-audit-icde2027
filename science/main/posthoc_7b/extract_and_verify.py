#!/usr/bin/env python3
"""Safely extract and check the saved post-hoc third-model experiment."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--artifact',type=Path,required=True,help='extracted reproducibility directory')
 p.add_argument('--out',type=Path,required=True,help='new, non-existing extraction directory')
 a=p.parse_args();artifact=a.artifact.resolve();archive=Path(__file__).resolve().parent/'posthoc_7b_artifacts.tar.gz'
 expected=(archive.parent/'POSTHOC_7B_ARCHIVE.sha256').read_text().split()[0]
 if digest(archive)!=expected:raise ValueError('Third-model archive SHA-256 mismatch')
 if a.out.exists() or a.out.is_symlink():raise FileExistsError('Output path must not already exist')
 with tarfile.open(archive,'r:gz') as tar:
  items=tar.getmembers();names=[x.name for x in items]
  if len(names)!=len(set(names)):raise ValueError('Repeated archive member')
  for item in items:
   name=PurePosixPath(item.name)
   if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!='posthoc' or not (item.isfile() or item.isdir()):
    raise ValueError('Unsafe archive member: '+item.name)
  a.out.mkdir(parents=True,exist_ok=False)
  for item in items:
   target=a.out.joinpath(*PurePosixPath(item.name).parts)
   if item.isdir():target.mkdir(parents=True,exist_ok=True)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(item) as source, target.open('xb') as output:shutil.copyfileobj(source,output,1024*1024)
 work=(a.out/'posthoc').resolve()
 saved=json.loads((work/'FINAL_7B_SHA256.json').read_text())
 found={str(x.relative_to(work)) for x in work.rglob('*') if x.is_file() and x.name!='FINAL_7B_SHA256.json'}
 if found!=set(saved):raise ValueError('Saved-file inventory mismatch')
 for name,want in saved.items():
  if digest(work/name)!=want:raise ValueError('Saved-file hash mismatch: '+name)
 snapshots={'render_manuscript.py','make_editable_word.py','make_figures.py'}
 def verify_manifest(name,skip_posthoc=False,skip_keys=()):
  obj=json.loads((work/name).read_text());count=0
  for key,want in obj.items():
   if key in skip_keys or (skip_posthoc and key.startswith('posthoc/')):continue
   if key.startswith('artifact/'):
    relative=key[len('artifact/'):]
    path=artifact/'pre_7b_snapshot'/relative if relative in snapshots else artifact/relative
   elif key.startswith('posthoc/'):path=work/key[len('posthoc/'):]
   else:path=work/key
   if not path.is_file():raise FileNotFoundError(f'{name}: missing {key}; if using the reduced TREC package, first reconstruct its sample')
   if digest(path)!=want:raise ValueError(f'{name}: frozen file mismatch: {key}')
   count+=1
  print(name,':',count,'frozen files verified')
 for name in ['PRE_7B_SOURCE_SHA256.json','PRE_7B_MULTIDOMAIN_SHA256.json','PRE_7B_CROSSED_SHA256.json','PRE_7B_GUARDED_BATCH_SHA256.json','PRE_7B_CROSSMODEL_AMENDMENT_SHA256.json','PRE_7B_CONTEXT_SHA256.json','PRE_7B_DRIVER_SHA256.json','PRE_7B_FIGURE_SHA256.json','PRE_7B_FIGURE_TEST_AMENDMENT_SHA256.json','SPEC_GUIDED_FROZEN_SHA256.json']:
  verify_manifest(name)
 verify_manifest('PRE_7B_CROSSMODEL_SHA256.json',skip_posthoc=True)
 verify_manifest('PRE_7B_FIGURE_TEST_SHA256.json',skip_keys={'test_7b_figure_sources.py'})
 comparisons={'POSTHOC_7B_ESCI.json':'analysis_esci_7b/summary.json','POSTHOC_7B_WANDS.json':'run_wands_7b/paired_analysis.json','POSTHOC_7B_TREC.json':'trec_7b_analysis.json','POSTHOC_7B_FIGURE_DATA.json':'figure_7b_absolute/figure_data.json','POSTHOC_7B_ESCI_GUIDED.json':'esci_specguided_comparison.json','POSTHOC_7B_WANDS_GUIDED.json':'wands_specguided_comparison.json'}
 for outer,inner in comparisons.items():
  if digest(artifact/outer)!=digest(work/inner):raise ValueError('Manuscript data differs from saved run: '+outer)
 print('All archived files, frozen experiment inputs and manuscript data agree.')
 print('Full model weights are not included. For analysis commands see posthoc_7b/README.md.')
if __name__=='__main__':main()
