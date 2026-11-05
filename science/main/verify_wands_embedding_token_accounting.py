#!/usr/bin/env python3
"""Independently reconstruct token strings through saved indices and AutoTokenizer."""
import argparse,hashlib,importlib.metadata,json,math
from pathlib import Path
from transformers import AutoTokenizer
PREFIX='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
S=[('train160','run_wands_160'),('holdout160','wands_holdout_selection'),('fallback96','wands_fallback_validation_selection'),('selective52','wands_selective_execution_selection')]
def read(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args()
 if importlib.metadata.version('transformers')!='4.57.1' or importlib.metadata.version('tokenizers')!='0.22.0':raise ValueError('Different tokenizer libraries')
 ref=json.loads(a.result.read_text())
 def digest_file(path):
  h=hashlib.sha256()
  with path.open('rb') as f:
   for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
  return h.hexdigest()
 for name,expected in ref['model_asset_hashes'].items():
  if digest_file(a.model/name)!=expected:raise ValueError('Model asset mismatch: '+name)
 for name,expected in ref['input_sha256'].items():
  if digest_file(a.artifact/name)!=expected:raise ValueError('Input mismatch: '+name)
 idx=read(a.artifact/'run_wands_embedding_reference/embedding_index.jsonl');offset=0;q={};d={};groups={}
 for name,path in S:
  sample=read(a.artifact/path/'inputs.jsonl');qset=set();dset=set()
  for row in sample:
   r=idx[offset];offset+=1
   if r['sample']!=name or r['query_id']!=row['query_id'] or len(r['document_indices'])!=len(row['items']):raise ValueError('Alignment')
   qi=r['query_index'];qtext=PREFIX+row['query']
   if qi in q and q[qi]!=qtext:raise ValueError('Inconsistent query vector')
   q[qi]=qtext;qset.add(qi)
   for di,item in zip(r['document_indices'],row['items']):
    if di in d and d[di]!=item['text']:raise ValueError('Inconsistent document vector')
    d[di]=item['text'];dset.add(di)
  groups[name]=(qset,dset)
 if offset!=len(idx) or set(q)!=set(range(len(q))) or set(d)!=set(range(len(d))):raise ValueError('Non-contiguous index')
 model=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left',use_fast=True)
 def calc(records):
  lengths=[];h=hashlib.sha256();padded=0
  for start in range(0,len(records),32):
   enc=model(records[start:start+32],padding=True,truncation=False)
   for ids,mask in zip(enc['input_ids'],enc['attention_mask']):
    selected=[i for i,m in zip(ids,mask) if m]
    if not selected or selected[-1]!=151643 or mask[-1]!=1:raise ValueError('EOS or padding mismatch')
    lengths.append(len(selected));padded+=len(ids);h.update(len(selected).to_bytes(4,'little'))
    for i in selected:h.update(int(i).to_bytes(4,'little'))
  return lengths,h.hexdigest(),padded
 ql,qh,qp=calc([q[i] for i in range(len(q))]);dl,dh,dp=calc([d[i] for i in range(len(d))]);joint=ref['joint_original_order']
 expected={'queries':len(q),'unique_texts':len(d),'query_tokens':sum(ql),'document_tokens':sum(dl),'total_unpadded_tokens':sum(ql)+sum(dl),'padded_batch_positions':qp+dp,'query_token_ids_sha256':qh,'document_token_ids_sha256':dh,'maximum_tokens':max(ql+dl),'batch_size':32,'batches':math.ceil(len(q)/32)+math.ceil(len(d)/32)}
 for k,v in expected.items():
  if joint[k]!=v:raise ValueError('Joint mismatch: '+k)
 for name,(qi,di) in groups.items():
  current={'queries':len(qi),'unique_texts':len(di),'query_tokens':sum(ql[i] for i in qi),'document_tokens':sum(dl[i] for i in di),'total_unpadded_tokens':sum(ql[i] for i in qi)+sum(dl[i] for i in di)}
  if current!=ref['standalone_samples_reconstructed'][name]:raise ValueError('Sample mismatch: '+name)
 print('AutoTokenizer reproduced every reconstructed token ID digest, EOS, batch positions and sample totals.')
if __name__=='__main__':main()
