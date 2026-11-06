#!/usr/bin/env python3
from fetch_provenance import fetch,SOURCES
from concurrent.futures import ThreadPoolExecutor
import json
URLS={
 'hf_mirror_api_snli.json':'https://hf-mirror.com/api/datasets/stanfordnlp/snli',
 'hf_mirror_README_main.md':'https://hf-mirror.com/datasets/stanfordnlp/snli/resolve/main/README.md',
 'stanford_index.html':'https://nlp.stanford.edu/projects/snli/index.html',
 'tfds_snli_checksums_current.txt':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/snli/checksums.tsv',
 'tfds_snli_source.py':'https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/snli/snli_dataset_builder.py',
 'hf_README_fixed_944ee.md':'https://hf-mirror.com/datasets/stanfordnlp/snli/resolve/944ee2aa07ada80ae04d586218befe1b7ddc52f9/README.md',
}
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:
  results=[]
  for row in pool.map(fetch,URLS.items()):results.append(row);print(json.dumps(row),flush=True)
 (SOURCES/'mirror_retrieval.json').write_text(json.dumps(results,indent=2)+'\n')
