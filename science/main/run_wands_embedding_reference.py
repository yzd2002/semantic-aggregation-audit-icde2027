#!/usr/bin/env python3
"""Encode WANDS pools with a fixed embedding model and train one threshold."""
import argparse,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
import torch,transformers
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from analyze_order_fallback_validation import paired
from run_compact_prompt_ablation import metrics
from run_wands_supervised_reference import SETS,check,ids,lines,queries

HERE=Path(__file__).resolve().parent
MODEL_NAMES={'model.safetensors':'0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd',
 'config.json':'b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9',
 'tokenizer.json':'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a',
 'tokenizer_config.json':'253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0',
 'merges.txt':'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
 'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
QUERY_PREFIX='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
PROTOCOL='PROTOCOL_wands_embedding_reference.md'
SEEDS={'holdout160':20271221,'fallback96':20271222,'selective52':20271223}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def fixed_threshold(scores,labels):
    order=np.argsort(-scores,kind='stable');s=scores[order];y=labels[order]
    end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[end];n=end+1;den=n+int(y.sum());f=2*tp/den
    best=int(np.argmax(f));return float(s[end[best]]),int(tp[best]),int(n[best]-tp[best]),int(y.sum()-tp[best]),float(f[best])
def source_data():
    datasets=[('train160',lines(HERE/'run_wands_160/inputs.jsonl'))]
    datasets.extend((name,lines(HERE/selection/'inputs.jsonl')) for name,selection,_,_ in SETS)
    for (name,rows),n in zip(datasets,[160,160,96,52]):check(rows,n)
    tr=datasets[0][1]
    for name,rows in datasets[1:]:
        if queries(tr)&queries(rows) or ids(tr)&ids(rows) or {i['text'] for r in tr for i in r['items']}&{i['text'] for r in rows for i in r['items']}:raise ValueError('Training overlap: '+name)
    return datasets
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for this recorded run')
    for name,wanted in MODEL_NAMES.items():
        if sha(a.model/name)!=wanted:raise ValueError('Model local hash mismatch: '+name)
    datasets=source_data();qs=[];docs=[];qmap={};dmap={};index=[]
    for name,rows in datasets:
        for row in rows:
            q=row['query'];qkey=QUERY_PREFIX+q
            if qkey not in qmap:qmap[qkey]=len(qs);qs.append(qkey)
            ix=[]
            for item in row['items']:
                text=item['text']
                if text not in dmap:dmap[text]=len(docs);docs.append(text)
                ix.append(dmap[text])
            index.append({'sample':name,'query_id':row['query_id'],'query_index':qmap[qkey],'document_indices':ix})
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'embedding_index.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in index),encoding='utf8')
    tok=AutoTokenizer.from_pretrained(a.model,padding_side='left',local_files_only=True)
    model=AutoModel.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
    torch.cuda.reset_peak_memory_stats();gpu_start=time.monotonic();max_tokens=0
    def encode(strings,label):
        nonlocal max_tokens
        chunks=[]
        for start in range(0,len(strings),32):
            enc=tok(strings[start:start+32],padding=True,truncation=False,return_tensors='pt')
            length=int(enc['attention_mask'].sum(dim=1).max());max_tokens=max(max_tokens,length)
            if length>1024:raise ValueError('Input token length exceeds fixed cap')
            if not (enc['attention_mask'][:,-1]==1).all():raise ValueError('Expected left padding')
            if not (enc['input_ids'][:,-1]==model.config.eos_token_id).all():raise ValueError('Expected model-config terminal token')
            enc=enc.to('cuda')
            with torch.inference_mode():
                out=model(**enc,use_cache=False).last_hidden_state[:,-1,:].float()
                vectors=F.normalize(out,p=2,dim=1).cpu().numpy()
            chunks.append(vectors)
            if (start//32)%50==0:print(label,start,'/',len(strings),flush=True)
        return np.concatenate(chunks,axis=0)
    qv=encode(qs,'queries');dv=encode(docs,'documents');torch.cuda.synchronize()
    inference=time.monotonic()-gpu_start;peak=round(torch.cuda.max_memory_allocated()/2**20,2)
    np.savez_compressed(a.out/'embeddings.npz',queries=qv,documents=dv)
    if not np.allclose(np.linalg.norm(qv,axis=1),1,atol=1e-5) or not np.allclose(np.linalg.norm(dv,axis=1),1,atol=1e-5):raise ValueError('Invalid vector normalization')
    scores={};row_index=0
    for name,rows in datasets:
        vals=[]
        for row in rows:
            ix=index[row_index];row_index+=1
            vals.append(np.sum(dv[ix['document_indices']]*qv[ix['query_index']],axis=1,dtype=np.float32))
        scores[name]=np.stack(vals)
    y=np.asarray([int(i['gold']=='Y') for row in datasets[0][1] for i in row['items']],dtype=np.int8)
    threshold,tp,fp,fn,trainf1=fixed_threshold(scores['train160'].ravel(),y)
    prior=lines(HERE/'run_wands_supervised_reference/predictions.jsonl');out=[];result={'protocol':PROTOCOL,
      'model':'Qwen/Qwen3-Embedding-0.6B','model_revision':'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
      'representation':{'query_count':len(qs),'unique_documents':len(docs),'dimension':int(qv.shape[1]),'max_input_tokens':max_tokens,
      'batch_size':32,'dtype':'bfloat16','attention':'sdpa','offline_encoding_seconds':inference,'peak_allocated_mib':peak},
      'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)},
      'training':{'pools':len(datasets[0][1]),'candidates':int(y.size),'positive':int(y.sum()),'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'f1':trainf1},'samples':{}}
    for name,rows in datasets[1:]:
        previous=[r for r in prior if r['sample']==name]
        if len(previous)!=len(rows) or any(x['query_id']!=y['query_id'] for x,y in zip(previous,rows)):raise ValueError('Prior reference alignment')
        embed=[['Y' if float(v)>=threshold else 'N' for v in score] for score in scores[name]]
        pred={'embedding':embed,'qwen':[r['qwen'] for r in previous],'lexical':[r['lexical'] for r in previous],
              'supervised_lexical':[r['supervised'] for r in previous]}
        common=list(range(len(rows)))
        if any(v is None or len(v)!=32 for v in pred['qwen']):raise ValueError('Unexpected Qwen completion')
        draw=np.random.default_rng(SEEDS[name]).integers(0,len(rows),size=(4000,len(rows)))
        result['samples'][name]={'pools':len(rows),'query_bootstrap_seed':SEEDS[name],'resamples':4000,
          'quality':{k:metrics(rows,v,common) for k,v in pred.items()},
          'embedding_minus_qwen':paired(rows,pred,common,'qwen','embedding',draw)}
        for j,row in enumerate(rows):out.append({'sample':name,'query_id':row['query_id'],'scores':[float(x) for x in scores[name][j]],'embedding':embed[j]})
    with (a.out/'predictions.jsonl').open('w',encoding='utf8') as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
