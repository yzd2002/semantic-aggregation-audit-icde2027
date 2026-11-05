#!/usr/bin/env python3
"""Run the disjoint validation in PROTOCOL_validation.md using the pilot code."""
import argparse, collections, hashlib, io, json, math, os, random, time
from pathlib import Path
import run_experiment as base
SEED=20271101

def sample_rows(path,excluded,n=500):
    import zstandard
    rng=random.Random(SEED);seen=set();sample=[];eligible=bad=0
    with open(path,'rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as z:
        for lineno,line in enumerate(io.TextIOWrapper(z,encoding='utf-8',errors='strict'),1):
            try:x=json.loads(line)
            except json.JSONDecodeError:bad+=1;continue
            q=x.get('query');pos=x.get('doc');neg=x.get('neg');scores=x.get('negscore')
            if not (isinstance(q,str) and q.strip() and isinstance(pos,str) and pos.strip() and
                    isinstance(neg,list) and isinstance(scores,list) and len(neg)==len(scores) and neg):continue
            if q in seen or q in excluded:continue
            labels=[];good=True
            for d,s in zip(neg,scores):
                if not isinstance(d,str) or not d.strip() or not isinstance(s,(int,float)):good=False;break
                label=next((k for k,v in [('S',.1),('C',.01),('I',0.)] if math.isclose(s,v,abs_tol=1e-6)),None)
                if label is None:good=False;break
                labels.append(label)
            if not good:continue
            seen.add(q);eligible+=1
            rec={'line':lineno,'query':q,'positive':pos,'negative':neg,'grades':labels}
            if len(sample)<n:sample.append(rec)
            else:
                j=rng.randrange(eligible)
                if j<n:sample[j]=rec
    if len(sample)<n:raise RuntimeError('Not enough eligible query strings')
    rows=[]
    for row in sample:
        q=row['query'];qr=random.Random(int.from_bytes(hashlib.sha256((str(SEED)+'\0'+q).encode()).digest()[:8],'big'))
        ids=qr.sample(range(len(row['negative'])),min(7,len(row['negative'])))
        items=[{'text':row['positive'][:600],'grade':'E'}]+[{'text':row['negative'][i][:600],'grade':row['grades'][i]} for i in ids]
        qr.shuffle(items)
        for i,it in enumerate(items,1):it['id']=i;it['gold']='Y' if it['grade'] in ('E','S') else 'N'
        rows.append({'source_line':row['line'],'query':q,'items':items})
    return rows,{'eligible_distinct_strings_after_exclusion':eligible,'malformed_json_lines':bad,'sample_size':len(rows),'excluded_pilot_queries':len(excluded)}

def compact_prompt(row,spec=None):
    s='Query: '+row['query']+'\n'
    if spec is not None:s+='Query specification: '+json.dumps(spec,ensure_ascii=False)+'\n'
    s+='Products in order:\n'+'\n'.join(json.dumps({'id':it['id'],'text':it['text']},ensure_ascii=False) for it in row['items'])
    s+=f'\nReturn exactly {len(row["items"])} characters, one Y or N per product in the listed order. No spaces, punctuation, or explanations. Example for 4 products: YNNY.'
    return s

def parse_compact(s,n):
    return list(s) if len(s)==n and set(s)<=set('YN') else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/data/esci/test.jsonl.zst');ap.add_argument('--model',default='/root/models/Qwen/Qwen3-4B-Instruct-2507');ap.add_argument('--out',default='run_validation_500');a=ap.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    pilot=Path(__file__).resolve().parent/'run_test_160/inputs.jsonl'
    excluded={json.loads(x)['query'] for x in pilot.open()};rows,selection=sample_rows(a.data,excluded)
    assert not ({x['query'] for x in rows}&excluded)
    base.write_json(out/'selection.json',selection)
    with (out/'inputs.jsonl').open('w',encoding='utf8') as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print('Selection:',selection,flush=True)
    import torch,transformers,vllm
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,tokenizer=a.model,trust_remote_code=False,dtype='bfloat16',max_model_len=4096,
            gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=True,disable_log_stats=True)
    tok=llm.get_tokenizer()
    def wrap(p):return tok.apply_chat_template([{'role':'system','content':base.SYSTEM},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
    def st():return {'seconds':0.,'calls':0,'prompt_tokens':0,'output_tokens':0,'fallback_pools':0}
    def run(prompts,cap,stats,label):
        if not prompts:return []
        start=time.perf_counter();rs=llm.generate([wrap(x) for x in prompts],SamplingParams(temperature=0,max_tokens=cap),use_tqdm=False)
        elapsed=time.perf_counter()-start;stats['seconds']+=elapsed;stats['calls']+=len(prompts)
        stats['prompt_tokens']+=sum(len(x.prompt_token_ids) for x in rs);stats['output_tokens']+=sum(len(x.outputs[0].token_ids) for x in rs)
        print(label,'calls',len(prompts),'sec',round(elapsed,2),flush=True)
        return [x.outputs[0].text.strip() for x in rs]
    run(['Answer ONLY Y or N.'],4,st(),'warmup')
    allresults={};allstats={}
    # Independent pointwise plan.
    stat=st();prompts=[];places=[]
    for i,row in enumerate(rows):
        for j,item in enumerate(row['items']):prompts.append(base.p_point(row,item));places.append((i,j))
    first=run(prompts,4,stat,'pointwise');final=first[:];bad=[i for i,v in enumerate(first) if v not in ('Y','N')]
    retry=run([prompts[i]+'\nOutput one character: Y or N.' for i in bad],4,stat,'pointwise retry')
    for i,v in zip(bad,retry):final[i]=v
    pred=[[None]*len(r['items']) for r in rows];raw=[[None]*len(r['items']) for r in rows]
    for k,(i,j) in enumerate(places):pred[i][j]=final[k] if final[k] in ('Y','N') else None;raw[i][j]={'first':first[k],'final':final[k]}
    allresults['pointwise']=[{'pred':p if None not in p else None,'raw':r} for p,r in zip(pred,raw)];allstats['pointwise']=stat
    # Shared generation helper for batch plans.
    def batch_plan(label,mode,specs=None,fallback=False):
        stat=st();spec_raw=[None]*len(rows);spec_values=[None]*len(rows);eligible=list(range(len(rows)))
        if specs:
            sp=[base.p_spec(x) for x in rows];sr=run(sp,160,stat,label+' spec')
            spec_values=[base.parse_spec(s) for s in sr];spec_raw=[{'first':s,'final':s} for s in sr]
            bad=[i for i,v in enumerate(spec_values) if v is None]
            retry=run([sp[i]+'\nReturn valid JSON only.' for i in bad],160,stat,label+' spec retry')
            for i,v in zip(bad,retry):spec_values[i]=base.parse_spec(v);spec_raw[i]['final']=v
            if not fallback:eligible=[i for i in eligible if spec_values[i] is not None]
            else:stat['fallback_pools']=sum(x is None for x in spec_values)
        prompts=[]
        for i in eligible:
            spec=spec_values[i] if specs else None
            prompts.append(base.p_batch(rows[i],spec) if mode=='json' else compact_prompt(rows[i],spec))
        parser=base.parse_batch if mode=='json' else parse_compact;cap=192 if mode=='json' else 32
        first=run(prompts,cap,stat,label);final=first[:];pred=[parser(v,len(rows[i]['items'])) for i,v in zip(eligible,first)]
        bad=[j for j,p in enumerate(pred) if p is None]
        retry=run([prompts[j]+'\nYour previous answer had the wrong format; follow the output instructions exactly.' for j in bad],cap,stat,label+' retry')
        for j,v in zip(bad,retry):final[j]=v;pred[j]=parser(v,len(rows[eligible[j]]['items']))
        result=[{'pred':None,'raw':None,'spec':spec_raw[i]} for i in range(len(rows))]
        for j,i in enumerate(eligible):result[i]={'pred':pred[j],'raw':{'first':first[j],'final':final[j]},'spec':spec_raw[i]}
        allresults[label]=result;allstats[label]=stat
    batch_plan('batch_json','json')
    batch_plan('two_stage_json','json',specs=True)
    batch_plan('batch_compact','compact')
    batch_plan('two_stage_compact_fallback','compact',specs=True,fallback=True)
    with (out/'outputs.jsonl').open('w',encoding='utf8') as f:
        for i,row in enumerate(rows):f.write(json.dumps({'source_line':row['source_line'],'query':row['query'],'plans':{k:v[i] for k,v in allresults.items()}},ensure_ascii=False)+'\n')
    summary={'protocol':'PROTOCOL_validation.md','selection':selection,'model':os.path.basename(a.model),
             'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'max_model_len':4096,'dtype':'bfloat16'},
             'grade_counts':dict(collections.Counter(it['grade'] for row in rows for it in row['items'])),'plans':{}}
    for name,results in allresults.items():
        complete=[(r,o) for r,o in zip(rows,results) if o['pred'] is not None]
        tp=fp=tn=fn=0;mae=[];gc=collections.Counter();gt=collections.Counter()
        for row,o in complete:
            gold=[it['gold'] for it in row['items']];pred=o['pred']
            mae.append(abs(sum(x=='Y' for x in gold)/len(gold)-sum(x=='Y' for x in pred)/len(pred)))
            for it,y in zip(row['items'],pred):
                g=it['gold'];gt[it['grade']]+=1;gc[it['grade']]+=int(g==y)
                tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
        total=tp+fp+tn+fn
        summary['plans'][name]={**allstats[name],'complete_pools':len(complete),'attempted_pools':len(rows),'tp':tp,'fp':fp,'tn':tn,'fn':fn,
           'accuracy':(tp+tn)/total if total else None,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
           'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'ratio_mae':sum(mae)/len(mae) if mae else None,'grade_correct':dict(gc),'grade_total':dict(gt)}
    base.write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
