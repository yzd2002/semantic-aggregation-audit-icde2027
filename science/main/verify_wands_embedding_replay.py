#!/usr/bin/env python3
"""Recompute the recorded independent embedding-vector replay differences."""
import argparse,json
from pathlib import Path
import numpy as np
OFFSETS=(0,3200,6400,9600,12800)
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);a=p.parse_args()
 with np.load(a.prior/'embeddings.npz',allow_pickle=False) as f:q=f['queries'];d=f['documents']
 with np.load(a.replay/'replay_vectors.npz',allow_pickle=False) as f:rq=f['queries'];rd=f['documents']
 orig=np.concatenate([q[:32],*[d[i:i+32] for i in OFFSETS]],axis=0);again=np.concatenate([rq,rd],axis=0)
 assert orig.shape==again.shape==(192,1024)
 diff=np.abs(orig-again);cos=np.sum(orig*again,axis=1,dtype=np.float32)
 summary=json.loads((a.replay/'summary.json').read_text());records=[json.loads(s) for s in (a.replay/'differences.jsonl').read_text().splitlines()]
 assert len(records)==192 and summary['document_offsets']==list(OFFSETS)
 for i,r in enumerate(records):
  assert r['index']==i and abs(r['max_abs_difference']-float(diff[i].max()))<1e-12
  assert abs(r['mean_abs_difference']-float(diff[i].mean()))<1e-12 and abs(r['cosine']-float(cos[i]))<1e-12
  assert r['exact']==bool(np.array_equal(orig[i],again[i]))
 assert summary['exact_vectors']==sum(r['exact'] for r in records)
 assert abs(summary['max_absolute_difference']-float(diff.max()))<1e-12
 assert abs(summary['mean_absolute_difference']-float(diff.mean()))<1e-12
 assert abs(summary['minimum_cosine']-float(cos.min()))<1e-12
 print('Independently recomputed differences for',len(records),'fixed replay vectors; exact',summary['exact_vectors'],'maximum absolute difference',summary['max_absolute_difference'])
if __name__=='__main__':main()
