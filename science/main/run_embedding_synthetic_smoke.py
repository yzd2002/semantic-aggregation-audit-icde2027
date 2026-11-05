#!/usr/bin/env python3
"""Unrelated, fixed model-card-style sanity check; no WANDS examples are used."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from run_wands_embedding_reference import MODEL_NAMES,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 for name,want in MODEL_NAMES.items():
  if sha(a.model/name)!=want:raise ValueError('Model hash differs: '+name)
 if a.out.exists():raise ValueError('Use a new output file')
 strings=['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?',
          'The capital of China is Beijing.','Gravity is a force that attracts two bodies toward each other.']
 t=time.monotonic();tok=AutoTokenizer.from_pretrained(a.model,padding_side='left',local_files_only=True)
 model=AutoModel.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
 x=tok(strings,padding=True,truncation=False,return_tensors='pt');assert bool((x['input_ids'][:,-1]==model.config.eos_token_id).all())
 with torch.inference_mode():
  y=F.normalize(model(**x.to('cuda'),use_cache=False).last_hidden_state[:,-1,:].float(),p=2,dim=1)
  scores=(y[0]@y[1:].T).cpu().tolist()
 result={'query':'What is the capital of China?','documents':strings[1:],'similarities':scores,
  'relevant_ranks_first':bool(scores[0]>scores[1]),'max_tokens':int(x['attention_mask'].sum(dim=1).max()),
  'model_config_terminal_token_id':int(model.config.eos_token_id),'tokenizer_eos_token_id':int(tok.eos_token_id),
  'elapsed_load_and_encoding_seconds':time.monotonic()-t,'gpu':torch.cuda.get_device_name(0)}
 a.out.write_text(json.dumps(result,indent=2)+'\n');assert result['relevant_ranks_first'];print(json.dumps(result,indent=2))
if __name__=='__main__':main()
