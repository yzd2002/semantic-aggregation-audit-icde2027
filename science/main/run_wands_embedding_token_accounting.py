#!/usr/bin/env python3
"""Reconstruct token work from the saved, jointly encoded WANDS inputs."""
import argparse,hashlib,importlib.metadata,json,math
from pathlib import Path
from tokenizers import Tokenizer
PREFIX='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
SELECTIONS=[('train160','run_wands_160'),('holdout160','wands_holdout_selection'),('fallback96','wands_fallback_validation_selection'),('selective52','wands_selective_execution_selection')]
TOKENIZER_HASHES={'tokenizer.json':'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a','tokenizer_config.json':'253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0','config.json':'b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9'}
def hashfile(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(path):
 with path.open(encoding='utf8') as f:return [json.loads(line) for line in f if line.strip()]
def sources(artifact,model):
 data=[];index=[];qs=[];docs=[];qmap={};dmap={};groups={}
 for (name,path),n in zip(SELECTIONS,[160,160,96,52]):
  source=rows(artifact/path/'inputs.jsonl')
  if len(source)!=n or any(len(r['items'])!=32 for r in source):raise ValueError('Unexpected pools: '+name)
  groupq=set();groupd=set()
  for row in source:
   key=PREFIX+row['query']
   if key not in qmap:qmap[key]=len(qs);qs.append(key)
   qi=qmap[key];groupq.add(qi);docix=[]
   for item in row['items']:
    text=item['text']
    if text not in dmap:dmap[text]=len(docs);docs.append(text)
    di=dmap[text];docix.append(di);groupd.add(di)
   index.append({'sample':name,'query_id':row['query_id'],'query_index':qi,'document_indices':docix})
  groups[name]={'queries':sorted(groupq),'documents':sorted(groupd)}
  data.append(artifact/path/'inputs.jsonl')
 actual=rows(artifact/'run_wands_embedding_reference/embedding_index.jsonl')
 if index!=actual:raise ValueError('Reconstructed token source differs from saved vector index')
 return qs,docs,groups,data

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 if importlib.metadata.version('tokenizers')!='0.22.0':raise ValueError('Use original tokenizers 0.22.0')
 for name,h in TOKENIZER_HASHES.items():
  if hashfile(a.model/name)!=h:raise ValueError('Changed model asset: '+name)
 qs,docs,groups,files=sources(a.artifact,a.model)
 orig=json.loads((a.artifact/'run_wands_embedding_reference/summary.json').read_text())['representation'];config=json.loads((a.model/'config.json').read_text())
 if config['eos_token_id']!=151643:raise ValueError('Unexpected model EOS')
 tok=Tokenizer.from_file(str(a.model/'tokenizer.json'));tok.no_truncation();tok.no_padding()
 def calc(values):
  lengths=[];hasher=hashlib.sha256();padded=0
  for i in range(0,len(values),32):
   encoded=tok.encode_batch(values[i:i+32],add_special_tokens=True);batch=[]
   for e in encoded:
    if not e.ids or e.ids[-1]!=151643:raise ValueError('Bad special token')
    batch.append(len(e.ids));hasher.update(len(e.ids).to_bytes(4,'little'))
    for token in e.ids:hasher.update(token.to_bytes(4,'little'))
   lengths.extend(batch);padded+=max(batch)*len(batch)
  return lengths,hasher.hexdigest(),padded
 ql,qsha,qp=calc(qs);dl,dsha,dp=calc(docs)
 if len(qs)!=orig['query_count'] or len(docs)!=orig['unique_documents'] or max(ql+dl)!=orig['max_input_tokens'] or max(ql+dl)>1024:raise ValueError('Mismatch against recorded GPU-run summary')
 sample={}
 for name,v in groups.items():
  qi=v['queries'];di=v['documents'];sample[name]={'queries':len(qi),'unique_texts':len(di),'query_tokens':sum(ql[i] for i in qi),'document_tokens':sum(dl[i] for i in di),'total_unpadded_tokens':sum(ql[i] for i in qi)+sum(dl[i] for i in di)}
 inputs={str(p.relative_to(a.artifact)):hashfile(p) for p in files+[a.artifact/'run_wands_embedding_reference/embedding_index.jsonl',a.artifact/'run_wands_embedding_reference/summary.json',a.artifact/'run_wands_embedding_reference.py']}
 result={'status':'post-hoc tokenizer reconstruction; no model inference',
 'versions':{'tokenizers':importlib.metadata.version('tokenizers'),'original_transformers':'4.57.1'},'model_asset_hashes':TOKENIZER_HASHES,'input_sha256':inputs,
 'joint_original_order':{'queries':len(qs),'unique_texts':len(docs),'query_tokens':sum(ql),'document_tokens':sum(dl),'total_unpadded_tokens':sum(ql)+sum(dl),'padded_batch_positions':qp+dp,'query_token_ids_sha256':qsha,'document_token_ids_sha256':dsha,'maximum_tokens':max(ql+dl),'batch_size':32,'batches':math.ceil(len(qs)/32)+math.ceil(len(docs)/32),'original_offline_seconds':orig['offline_encoding_seconds'],'original_peak_allocated_mib':orig['peak_allocated_mib']},
 'standalone_samples_reconstructed':sample}
 with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'joint':result['joint_original_order'],'sample52':sample['selective52']},indent=2))
if __name__=='__main__':main()
