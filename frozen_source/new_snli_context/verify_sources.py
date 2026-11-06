#!/usr/bin/env python3
"""Verify immutable Parquet bytes, declared license and schema before selection."""
from common import *
cpu_environment()
import collections
import pyarrow as pa
import pyarrow.parquet as pq

def verify():
 source=ROOT/'sources'
 tree={r['path']:r for r in json.loads((source/'hf_tree_fixed_cdb.json').read_text())}
 readme=source/'hf_README_fixed_cdb.md'
 body=readme.read_text()
 assert 'cc-by-sa-4.0' in body
 assert 'The Stanford Natural Language Inference Corpus by The Stanford NLP Group is licensed under a' in body
 assert 'Creative Commons Attribution-ShareAlike 4.0 International License' in body
 git_blob=hashlib.sha1(b'blob '+str(readme.stat().st_size).encode()+b'\0'+readme.read_bytes()).hexdigest()
 assert git_blob==tree['README.md']['oid']
 result={'status':'PASS','source_repository':'stanfordnlp/snli','revision':SNLI_REVISION,
  'transport':'third-party hf-mirror.com resolver plus public Hugging Face Xet CDN; not original ZIP',
  'official_zip_byte_comparison':False,'declared_dataset_license':'CC-BY-SA-4.0',
  'declared_license_source':f'https://huggingface.co/datasets/stanfordnlp/snli/blob/{SNLI_REVISION}/README.md',
  'license_card_sha256':sha(readme),'license_card_git_blob_oid':git_blob,
  'license_card_lines':[i+1 for i,s in enumerate(body.splitlines()) if 'Corpus by The Stanford NLP Group is licensed under' in s],
  'paper':'https://aclanthology.org/D15-1075/','splits':{}}
 for split in ('test','train'):
  path=source/f'{split}.parquet';entry=tree[f'plain_text/{split}-00000-of-00001.parquet']
  pointer=(source/f'{split}_lfs_pointer.txt').read_text()
  assert f'oid sha256:{SOURCE_SHA256[split]}' in pointer and f'size {SOURCE_SIZES[split]}' in pointer
  assert path.stat().st_size==entry['size']==entry['lfs']['size']==SOURCE_SIZES[split]
  assert sha(path)==entry['lfs']['oid']==SOURCE_SHA256[split]
  table=pq.read_table(path)
  assert table.num_rows==SOURCE_ROWS[split]
  assert table.schema.names==['premise','hypothesis','label']
  assert all(pa.types.is_string(table.schema.field(k).type) for k in SOURCE_FIELDS)
  assert pa.types.is_integer(table.schema.field('label').type)
  hf=json.loads(table.schema.metadata[b'huggingface'])
  assert hf['info']['features']['label']['names']==['entailment','neutral','contradiction']
  labels=table['label'].to_pylist()
  assert set(labels)<=set([-1,0,1,2])
  result['splits'][split]={'rows':table.num_rows,'schema':str(table.schema),'sha256':sha(path),
   'bytes':path.stat().st_size,'lfs_pointer_verified':True,'valid_label_metadata':sum(x in (0,1,2) for x in labels),
   'missing_label_metadata':sum(x==-1 for x in labels),'source_id_contract':f'{split}:zero-based six-digit parquet row index'}
 write_json(source/'SOURCE_LICENSE_SCHEMA_VERIFIED.json',result)
 print(compact({k:v for k,v in result.items() if k!='splits'}))
 for split,info in result['splits'].items():print(split,info['rows'],info['valid_label_metadata'],info['missing_label_metadata'])

if __name__=='__main__':verify()
