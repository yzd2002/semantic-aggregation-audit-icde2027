#!/usr/bin/env python3
"""Rerun preselected original query and document batches in a fresh process."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from run_wands_embedding_reference import MODEL_NAMES,QUERY_PREFIX,sha,source_data
OFFSETS=(0,3200,6400,9600,12800)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 for name,h in MODEL_NAMES.items():
  if sha(a.model/name)!=h:raise ValueError('Model hash differs: '+name)
 datasets=source_data();qs=[];docs=[];qseen=set();dseen=set()
 for name,rows in datasets:
  for row in rows:
   q=QUERY_PREFIX+row['query']
   if q not in qseen:qseen.add(q);qs.append(q)
   for i in row['items']:
    if i['text'] not in dseen:dseen.add(i['text']);docs.append(i['text'])
 with np.load(a.prior/'embeddings.npz',allow_pickle=False) as f:oq=f['queries'];od=f['documents']
 assert len(qs)==len(oq) and len(docs)==len(od) and all(j+32<=len(docs) for j in OFFSETS)
 a.out.mkdir(parents=True,exist_ok=False)
 tok=AutoTokenizer.from_pretrained(a.model,padding_side='left',local_files_only=True)
 model=AutoModel.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
 def encode(strings):
  x=tok(strings,padding=True,truncation=False,return_tensors='pt')
  assert int(x['attention_mask'].sum(dim=1).max())<=1024
  assert bool((x['attention_mask'][:,-1]==1).all()) and bool((x['input_ids'][:,-1]==model.config.eos_token_id).all())
  with torch.inference_mode():
   z=model(**x.to('cuda'),use_cache=False).last_hidden_state[:,-1,:].float()
   return F.normalize(z,p=2,dim=1).cpu().numpy()
 t=time.monotonic();rq=encode(qs[:32]);rd=np.concatenate([encode(docs[j:j+32]) for j in OFFSETS],axis=0);torch.cuda.synchronize();elapsed=time.monotonic()-t
 baseline=np.concatenate([oq[:32],*(od[j:j+32] for j in OFFSETS)],axis=0);replay=np.concatenate([rq,rd],axis=0)
 assert baseline.shape==replay.shape==(192,1024)
 dif=np.abs(baseline-replay);cos=np.sum(baseline*replay,axis=1,dtype=np.float32)
 np.savez_compressed(a.out/'replay_vectors.npz',queries=rq,documents=rd)
 records=[]
 for i in range(len(replay)):
  records.append({'index':i,'max_abs_difference':float(dif[i].max()),'mean_abs_difference':float(dif[i].mean()),'cosine':float(cos[i]),'exact':bool(np.array_equal(baseline[i],replay[i]))})
 (a.out/'differences.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
 summary={'protocol':'PROTOCOL_wands_embedding_replay.md','query_count':32,'document_offsets':list(OFFSETS),'documents_per_batch':32,'vectors':192,
  'exact_vectors':sum(x['exact'] for x in records),'max_absolute_difference':float(dif.max()),'mean_absolute_difference':float(dif.mean()),
  'minimum_cosine':float(cos.min()),'offline_encoding_seconds':elapsed,'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0)}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
