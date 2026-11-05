#!/usr/bin/env python3
"""Reconstruct a small ESCI-derived test workload and compare three LM plans.
See PROTOCOL.md. Do not overwrite an existing run directory.
"""
import argparse, collections, hashlib, io, json, math, os, random, time
from pathlib import Path

SEED=20271030
SYSTEM=("You judge whether each product can satisfy a shopper's query. "
        "Mark Y for an exact match or a functional substitute, and N for a mere accessory, "
        "an unrelated product, or a product lacking a stated essential requirement. "
        "Use only the supplied information. Do not assume unknown product properties.")

def read_rows(path, n=160):
    import zstandard
    rng=random.Random(SEED); seen=set(); sample=[]; eligible=0; malformed=0
    with open(path,'rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as z:
        for lineno,line in enumerate(io.TextIOWrapper(z,encoding='utf-8',errors='strict'),1):
            try: x=json.loads(line)
            except (json.JSONDecodeError, UnicodeError): malformed+=1; continue
            q=x.get('query'); pos=x.get('doc'); neg=x.get('neg'); scores=x.get('negscore')
            if not (isinstance(q,str) and q.strip() and isinstance(pos,str) and pos.strip() and
                    isinstance(neg,list) and isinstance(scores,list) and len(neg)==len(scores) and neg): continue
            if q in seen: continue
            labels=[]; good=True
            for d,s in zip(neg,scores):
                if not isinstance(d,str) or not d.strip() or not isinstance(s,(float,int)):
                    good=False; break
                match=next((k for k,v in [('S',.1),('C',.01),('I',0.)] if math.isclose(s,v,abs_tol=1e-6)),None)
                if match is None: good=False; break
                labels.append(match)
            if not good: continue
            seen.add(q); eligible+=1
            rec={'line':lineno,'query':q,'positive':pos,'negative':neg,'grades':labels}
            if len(sample)<n: sample.append(rec)
            else:
                j=rng.randrange(eligible)
                if j<n: sample[j]=rec
    if len(sample)<n: raise RuntimeError(f'Only {len(sample)} eligible distinct strings')
    out=[]
    for row in sample:
        q=row['query']
        qseed=int.from_bytes(hashlib.sha256((str(SEED)+'\0'+q).encode()).digest()[:8],'big')
        qr=random.Random(qseed)
        ids=qr.sample(range(len(row['negative'])),min(7,len(row['negative'])))
        items=[{'text':row['positive'][:600], 'grade':'E'}]
        items += [{'text':row['negative'][i][:600], 'grade':row['grades'][i]} for i in ids]
        qr.shuffle(items)
        for j,it in enumerate(items,1):
            it['id']=j; it['gold']='Y' if it['grade'] in ('E','S') else 'N'
        out.append({'source_line':row['line'],'query':q,'items':items})
    return out, {'unique_eligible_strings':eligible,'malformed_json_lines':malformed,'sample_size':len(out)}

def no_duplicates(pairs):
    obj={}
    for k,v in pairs:
        if k in obj: raise ValueError('duplicate object key')
        obj[k]=v
    return obj

def parse_spec(text):
    try:
        x=json.loads(text,object_pairs_hook=no_duplicates)
        if set(x)!= {'product_type','required','preferred'}: return None
        if not isinstance(x['product_type'],str) or not x['product_type'].strip(): return None
        if any(not isinstance(x[k],list) or any(not isinstance(y,str) for y in x[k]) for k in ('required','preferred')): return None
        return x
    except (ValueError,TypeError,AttributeError): return None

def parse_batch(text,n):
    try:
        arr=json.loads(text,object_pairs_hook=no_duplicates)
        if not isinstance(arr,list) or len(arr)!=n: return None
        labels={}
        for x in arr:
            if not isinstance(x,dict) or set(x)!={'id','label'}: return None
            if type(x['id']) is not int or not 1<=x['id']<=n or x['id'] in labels: return None
            if x['label'] not in ('Y','N'): return None
            labels[x['id']]=x['label']
        return [labels[i] for i in range(1,n+1)]
    except (ValueError,TypeError,KeyError): return None

def p_spec(row):
    return ('Describe only requirements explicitly supported by this shopping query. '
            'Return ONLY JSON with exactly these keys: "product_type" (a nonempty string), '
            '"required" (an array of strings), and "preferred" (an array of strings). '
            'Do not invent unstated attributes. Empty arrays are allowed.\nQuery: '+row['query'])

def p_point(row,item):
    return 'Query: '+row['query']+'\nProduct: '+item['text']+'\nAnswer ONLY Y or N.'

def p_batch(row,spec=None):
    prompt='Query: '+row['query']+'\n'
    if spec is not None: prompt+='Query specification: '+json.dumps(spec,ensure_ascii=False)+'\n'
    prompt+='Products:\n'+'\n'.join(json.dumps({'id':x['id'],'text':x['text']},ensure_ascii=False) for x in row['items'])
    prompt+=('\nReturn ONLY a JSON array with one object for EVERY product. Each object must have exactly '
             '"id" (integer) and "label" ("Y" or "N"). Include every id exactly once, even for N. '
             'Example for two items: [{"id":1,"label":"Y"},{"id":2,"label":"N"}].')
    return prompt

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='/data/esci/test.jsonl.zst')
    ap.add_argument('--model',default='/root/models/Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--out',default='run_test_160'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    rows, selection=read_rows(a.data)
    write_json(out/'selection.json',selection)
    with (out/'inputs.jsonl').open('w',encoding='utf8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Selection:',selection,flush=True)
    import torch, transformers, vllm
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,tokenizer=a.model,trust_remote_code=False,dtype='bfloat16',
            max_model_len=4096,gpu_memory_utilization=0.70,max_num_seqs=16,enforce_eager=True,
            enable_prefix_caching=True,disable_log_stats=True)
    tok=llm.get_tokenizer()
    def wrap(user):
        return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':user}],
                                       tokenize=False,add_generation_prompt=True)
    def run(prompts,max_tokens,stats,tag):
        if not prompts:return []
        tick=time.perf_counter()
        result=llm.generate([wrap(p) for p in prompts],SamplingParams(temperature=0,max_tokens=max_tokens),use_tqdm=False)
        elapsed=time.perf_counter()-tick
        stats['seconds']+=elapsed; stats['calls']+=len(prompts)
        stats['prompt_tokens']+=sum(len(r.prompt_token_ids) for r in result)
        stats['output_tokens']+=sum(len(r.outputs[0].token_ids) for r in result)
        print(tag,'calls',len(prompts),'sec',round(elapsed,2),flush=True)
        return [r.outputs[0].text.strip() for r in result]
    # One short unmeasured warmup for model/allocator initialization.
    warm={'seconds':0,'calls':0,'prompt_tokens':0,'output_tokens':0}
    run(['Answer ONLY Y or N.'],'4' if False else 4,warm,'warmup')
    all_results={}; all_stats={}
    def st():return {'seconds':0.0,'calls':0,'prompt_tokens':0,'output_tokens':0}
    # Pointwise, with at most one retry per invalid output.
    stats=st(); prompts=[]; places=[]
    for i,row in enumerate(rows):
        for j,item in enumerate(row['items']): prompts.append(p_point(row,item));places.append((i,j))
    first=run(prompts,4,stats,'pointwise'); final=first[:]
    bad=[i for i,x in enumerate(first) if x not in ('Y','N')]
    rer=run([prompts[i]+'\nOutput one character: Y or N.' for i in bad],4,stats,'pointwise retry')
    for i,val in zip(bad,rer):final[i]=val
    preds=[[None]*len(x['items']) for x in rows]; logs=[[None]*len(x['items']) for x in rows]
    for k,(i,j) in enumerate(places):preds[i][j]=final[k] if final[k] in ('Y','N') else None;logs[i][j]={'first':first[k],'final':final[k]}
    all_results['pointwise']=[{'pred':p if None not in p else None,'raw':l} for p,l in zip(preds,logs)]
    all_stats['pointwise']=stats
    # Direct listwise followed by refined listwise.
    for method in ('batch','two_stage'):
        stats=st(); specs=[None]*len(rows); spec_raw=[None]*len(rows)
        if method=='two_stage':
            sp=[p_spec(row) for row in rows]; sr=run(sp,160,stats,'spec')
            specs=[parse_spec(s) for s in sr]; spec_raw=[{'first':s,'final':s} for s in sr]
            bad=[i for i,x in enumerate(specs) if x is None]
            rer=run([sp[i]+'\nReturn valid JSON only.' for i in bad],160,stats,'spec retry')
            for i,val in zip(bad,rer):spec_raw[i]['final']=val;specs[i]=parse_spec(val)
            eligible=[i for i in range(len(rows)) if specs[i] is not None]
        else: eligible=list(range(len(rows)))
        bprompts=[p_batch(rows[i],specs[i] if method=='two_stage' else None) for i in eligible]
        raw=run(bprompts,192,stats,method); parsed=[parse_batch(s,len(rows[i]['items'])) for i,s in zip(eligible,raw)]
        final=raw[:];bad=[j for j,x in enumerate(parsed) if x is None]
        rer=run([bprompts[j]+'\nYour previous answer did not satisfy the format. Return a complete JSON array only.' for j in bad],192,stats,method+' retry')
        for j,val in zip(bad,rer):final[j]=val;parsed[j]=parse_batch(val,len(rows[eligible[j]]['items']))
        result=[{'pred':None,'raw':None,'spec':spec_raw[i]} for i in range(len(rows))]
        for j,i in enumerate(eligible):result[i]={'pred':parsed[j],'raw':{'first':raw[j],'final':final[j]},'spec':spec_raw[i]}
        all_results[method]=result;all_stats[method]=stats
    with (out/'outputs.jsonl').open('w',encoding='utf8') as f:
        for i,row in enumerate(rows):
            f.write(json.dumps({'source_line':row['source_line'],'query':row['query'],
                                'plans':{k:v[i] for k,v in all_results.items()}},ensure_ascii=False)+'\n')
    summary={'protocol':'PROTOCOL.md','selection':selection,'model':os.path.basename(a.model),
             'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,
                        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'max_model_len':4096,'dtype':'bfloat16'},
             'grade_counts':dict(collections.Counter(x['grade'] for row in rows for x in row['items'])),'plans':{}}
    for method in all_results:
        complete=[(row,res) for row,res in zip(rows,all_results[method]) if res['pred'] is not None]
        tp=fp=tn=fn=0; mae=[]; grade_correct=collections.Counter();grade_total=collections.Counter()
        for row,res in complete:
            gold=[x['gold'] for x in row['items']];pred=res['pred']
            mae.append(abs(sum(x=='Y' for x in gold)/len(gold)-sum(x=='Y' for x in pred)/len(pred)))
            for item,y in zip(row['items'],pred):
                g=item['gold'];grade_total[item['grade']]+=1;grade_correct[item['grade']]+=int(g==y)
                tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
        precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        n=tp+fp+tn+fn
        stats=all_stats[method]
        summary['plans'][method]={**stats,'complete_pools':len(complete),'attempted_pools':len(rows),
             'tp':tp,'fp':fp,'tn':tn,'fn':fn,'accuracy':(tp+tn)/n if n else None,
             'precision':precision,'recall':recall,'f1':f1,'ratio_mae':sum(mae)/len(mae) if mae else None,
             'grade_correct':dict(grade_correct),'grade_total':dict(grade_total)}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
