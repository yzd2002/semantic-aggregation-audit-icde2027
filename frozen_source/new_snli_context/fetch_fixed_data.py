#!/usr/bin/env python3
from fetch_provenance import fetch,SOURCES
from concurrent.futures import ThreadPoolExecutor
import json
REV='cdb5c3d5eed6ead6e5a341c8e56e669bb666725b'
BASE='https://hf-mirror.com/datasets/stanfordnlp/snli'
URLS={
 'hf_tree_fixed_cdb.json':f'https://hf-mirror.com/api/datasets/stanfordnlp/snli/tree/{REV}?recursive=true&expand=false',
 'hf_README_fixed_cdb.md':f'{BASE}/resolve/{REV}/README.md',
 'test_lfs_pointer.txt':f'{BASE}/raw/{REV}/plain_text/test-00000-of-00001.parquet',
 'train_lfs_pointer.txt':f'{BASE}/raw/{REV}/plain_text/train-00000-of-00001.parquet',
 'test.parquet':f'{BASE}/resolve/{REV}/plain_text/test-00000-of-00001.parquet',
 'train.parquet':f'{BASE}/resolve/{REV}/plain_text/train-00000-of-00001.parquet',
 'D15-1075.pdf':'https://aclanthology.org/D15-1075.pdf',
}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:
  results=[]
  for row in pool.map(fetch,URLS.items()):results.append(row);print(json.dumps(row),flush=True)
 (SOURCES/'fixed_data_retrieval.json').write_text(json.dumps(results,indent=2)+'\n')
