#!/usr/bin/env python3
"""Evaluate full, original ESCI query candidate lists; see PROTOCOL_full_pools.md."""
import argparse,collections,hashlib,json,os,random,time
from pathlib import Path
import run_experiment as base
from run_validation import compact_prompt,parse_compact
SEED=20271103
HERE=Path(__file__).resolve().parent

def sample_official(examples_path,products_path):
    import pandas as pd
    import pyarrow.dataset as ds
    prior=set()
    for folder in ('run_test_160','run_validation_500'):
        prior.update(json.loads(x)['query'] for x in (HERE/folder/'inputs.jsonl').open())
    ex=pd.read_parquet(examples_path,columns=['example_id','query','query_id','product_id','product_locale','esci_label','small_version','split'],
                       filters=[('product_locale','=','us'),('small_version','=',1),('split','=','test')])
    ex=ex[ex.esci_label.isin(['E','S','C','I']) & ex['query'].notna() & (~ex['query'].isin(prior))].copy()
    if not (ex.groupby('query_id')['query'].nunique()==1).all():raise RuntimeError('A query ID maps to multiple strings')
    ids=sorted(ex['query_id'].unique().tolist());rng=random.Random(SEED)
    chosen=rng.sample(ids,250);ex=ex[ex.query_id.isin(chosen)].copy()
    pids=sorted(ex.product_id.dropna().unique().tolist())
    fields=['product_id','product_locale','product_title','product_description','product_bullet_point','product_brand','product_color']
    dataset=ds.dataset(products_path,format='parquet')
    prod=dataset.to_table(columns=fields,filter=(ds.field('product_locale')=='us') & ds.field('product_id').isin(pids)).to_pandas()
    if prod.duplicated(['product_id','product_locale']).any():raise RuntimeError('Duplicate product keys; refusing ambiguous join')
    merged=ex.merge(prod,on=['product_id','product_locale'],how='left',indicator=True,validate='many_to_one')
    missing=int((merged['_merge']!='both').sum());rows=[]
    for qid in chosen:
        group=merged[merged.query_id==qid].sort_values('example_id')
        if group.empty:raise RuntimeError('Selected query has no examples')
        query=str(group['query'].iloc[0]);items=[]
        for _,x in group.iterrows():
            fields=('product_title','product_description','product_bullet_point','product_brand','product_color')
            text=' '.join(x[k] for k in fields if isinstance(x[k],str) and x[k])[:600]
            grade=str(x.esci_label)
            items.append({'text':text,'grade':grade,'gold':'Y' if grade in ('E','S') else 'N',
                          'example_id':str(x.example_id),'product_id':str(x.product_id),'join_missing':x['_merge']!='both'})
        qr=random.Random(int.from_bytes(hashlib.sha256((str(SEED)+'\0'+str(qid)).encode()).digest()[:8],'big'));qr.shuffle(items)
        for i,it in enumerate(items,1):it['id']=i
        rows.append({'query_id':str(qid),'query':query,'items':items})
    sizes=[len(x['items']) for x in rows]
    return rows,{'eligible_query_ids':len(ids),'excluded_prior_query_strings':len(prior),'sample_size':len(rows),'candidate_count':sum(sizes),
                 'pool_min':min(sizes),'pool_max':max(sizes),'pool_mean':sum(sizes)/len(sizes),'missing_product_joins':missing,
                 'empty_candidate_texts':sum(not it['text'] for x in rows for it in x['items']),
                 'distinct_sample_query_strings':len({x['query'] for x in rows})}

def chunks(row):
    out=[]
    for start in range(0,len(row['items']),8):
        part=[]
        for local,it in enumerate(row['items'][start:start+8],1):part.append({**it,'id':local})
        out.append({'query':row['query'],'items':part,'start':start})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--examples',default='/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_examples.parquet')
    ap.add_argument('--products',default='/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_products.parquet')
    ap.add_argument('--model',default='/root/models/Qwen/Qwen3-4B-Instruct-2507');ap.add_argument('--out',default='run_full_pools_250');a=ap.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    rows,selection=sample_official(a.examples,a.products);base.write_json(out/'selection.json',selection)
    with (out/'inputs.jsonl').open('w',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Selection:',selection,flush=True)
    import torch,transformers,vllm
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,tokenizer=a.model,trust_remote_code=False,dtype='bfloat16',max_model_len=4096,
            gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=True,disable_log_stats=True)
    tok=llm.get_tokenizer()
    def wrap(p):return tok.apply_chat_template([{'role':'system','content':base.SYSTEM},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
    def stats():return {'seconds':0.,'calls':0,'prompt_tokens':0,'output_tokens':0,'fallback_pools':0}
    def run(prompts,cap,st,label):
        if not prompts:return []
        start=time.perf_counter();rs=llm.generate([wrap(x) for x in prompts],SamplingParams(temperature=0,max_tokens=cap),use_tqdm=False)
        sec=time.perf_counter()-start;st['seconds']+=sec;st['calls']+=len(prompts)
        st['prompt_tokens']+=sum(len(x.prompt_token_ids) for x in rs);st['output_tokens']+=sum(len(x.outputs[0].token_ids) for x in rs)
        print(label,'calls',len(prompts),'seconds',round(sec,2),flush=True)
        return [x.outputs[0].text.strip() for x in rs]
    run(['Answer ONLY Y or N.'],4,stats(),'warmup')
    result={};stat={}
    # One model request per candidate, all served concurrently by the same offline engine.
    st=stats();prompts=[];places=[]
    for i,row in enumerate(rows):
        for j,item in enumerate(row['items']):prompts.append(base.p_point(row,item));places.append((i,j))
    first=run(prompts,4,st,'pointwise');final=first[:];bad=[i for i,x in enumerate(first) if x not in ('Y','N')]
    retry=run([prompts[i]+'\nOutput one character: Y or N.' for i in bad],4,st,'pointwise retry')
    for i,v in zip(bad,retry):final[i]=v
    pred=[[None]*len(r['items']) for r in rows];raw=[[None]*len(r['items']) for r in rows]
    for k,(i,j) in enumerate(places):pred[i][j]=final[k] if final[k] in ('Y','N') else None;raw[i][j]={'first':first[k],'final':final[k]}
    result['pointwise']=[{'pred':x if None not in x else None,'raw':r} for x,r in zip(pred,raw)];stat['pointwise']=st
    def batch(name,mode,specification=False,fallback=False):
        st=stats();specs=[None]*len(rows);specraw=[None]*len(rows)
        eligible=list(range(len(rows)))
        if specification:
            prompts=[base.p_spec(r) for r in rows];first=run(prompts,160,st,name+' spec')
            specs=[base.parse_spec(x) for x in first];specraw=[{'first':x,'final':x} for x in first]
            bad=[i for i,x in enumerate(specs) if x is None]
            retry=run([prompts[i]+'\nReturn valid JSON only.' for i in bad],160,st,name+' spec retry')
            for i,v in zip(bad,retry):specs[i]=base.parse_spec(v);specraw[i]['final']=v
            if fallback:st['fallback_pools']=sum(x is None for x in specs)
            else:eligible=[i for i in eligible if specs[i] is not None]
        prompts=[];locations=[]
        for i in eligible:
            for part in chunks(rows[i]):
                p=base.p_batch(part,specs[i] if specification else None) if mode=='json' else compact_prompt(part,specs[i] if specification else None)
                prompts.append(p);locations.append((i,part['start'],len(part['items'])))
        parser=base.parse_batch if mode=='json' else parse_compact;cap=192 if mode=='json' else 32
        first=run(prompts,cap,st,name);final=first[:];parsed=[parser(v,loc[2]) for v,loc in zip(first,locations)]
        bad=[j for j,x in enumerate(parsed) if x is None]
        retry=run([prompts[j]+'\nYour previous answer had the wrong format; follow the output instructions exactly.' for j in bad],cap,st,name+' retry')
        for j,v in zip(bad,retry):final[j]=v;parsed[j]=parser(v,locations[j][2])
        pred=[[None]*len(r['items']) for r in rows];raw=[[] for r in rows]
        for j,(i,start,n) in enumerate(locations):
            raw[i].append({'start':start,'size':n,'first':first[j],'final':final[j]})
            if parsed[j] is not None:pred[i][start:start+n]=parsed[j]
        result[name]=[{'pred':p if None not in p else None,'raw':r,'spec':specraw[i],
                       'fallback':bool(fallback and specs[i] is None)} for i,(p,r) in enumerate(zip(pred,raw))]
        stat[name]=st
    batch('batch_json','json')
    batch('two_stage_json','json',specification=True)
    batch('two_stage_compact_fallback','compact',specification=True,fallback=True)
    with (out/'outputs.jsonl').open('w',encoding='utf8') as f:
        for i,row in enumerate(rows):f.write(json.dumps({'query_id':row['query_id'],'query':row['query'],
                                  'plans':{k:v[i] for k,v in result.items()}},ensure_ascii=False)+'\n')
    summary={'protocol':'PROTOCOL_full_pools.md','selection':selection,'model':os.path.basename(a.model),
        'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name(0),'max_model_len':4096,'dtype':'bfloat16'},
        'grade_counts':dict(collections.Counter(it['grade'] for row in rows for it in row['items'])),'plans':{}}
    for name,rs in result.items():
        complete=[(r,o) for r,o in zip(rows,rs) if o['pred'] is not None];tp=fp=tn=fn=0;maes=[];gc=collections.Counter();gt=collections.Counter()
        for row,o in complete:
            gold=[it['gold'] for it in row['items']];pred=o['pred']
            maes.append(abs(sum(x=='Y' for x in gold)/len(gold)-sum(x=='Y' for x in pred)/len(pred)))
            for it,y in zip(row['items'],pred):
                g=it['gold'];gt[it['grade']]+=1;gc[it['grade']]+=int(g==y)
                tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
        n=tp+fp+tn+fn
        summary['plans'][name]={**stat[name],'complete_pools':len(complete),'attempted_pools':len(rows),'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'accuracy':(tp+tn)/n if n else None,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn) and (tn+fp) else None,
            'ratio_mae':sum(maes)/len(maes) if maes else None,'grade_correct':dict(gc),'grade_total':dict(gt)}
    base.write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
