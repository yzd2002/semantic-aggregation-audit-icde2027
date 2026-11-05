#!/usr/bin/env python3
"""Remove one misleading compact-format example while retaining the prior specs and pools."""
import argparse,collections,json,os,time
from pathlib import Path
import run_experiment as base
import run_full_pools as harness
import run_wands
from run_validation import compact_prompt,parse_compact

HERE=Path(__file__).resolve().parent
SENTENCE='Example for 4 products: YNNY.'
ESCI_SYSTEM=base.SYSTEM
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def prompt_without_example(part,spec):
    original=compact_prompt(part,spec)
    if not original.endswith(SENTENCE) or original.count(SENTENCE)!=1:raise ValueError('Unexpected original compact prompt')
    return original[:-len(SENTENCE)].rstrip()
def metrics(rows,labels,indices):
    tp=fp=tn=fn=0;mae=[]
    for i in indices:
        row=rows[i];pred=labels[i];gold=[it['gold'] for it in row['items']];assert len(pred)==len(gold)
        mae.append(abs(sum(x=='Y' for x in pred)/len(pred)-sum(x=='Y' for x in gold)/len(gold)))
        for g,y in zip(gold,pred):tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
    den=2*tp+fp+fn
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'f1':2*tp/den if den else 0.0,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if (tp+fn) and (tn+fp) else None,
            'ratio_mae':sum(mae)/len(mae) if mae else None}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct')
    ap.add_argument('--esci-run',type=Path,default=HERE/'run_full_pools_smollm_250')
    ap.add_argument('--wands-run',type=Path,default=HERE/'run_wands_smollm_160')
    ap.add_argument('--out',type=Path,default=Path('run_compact_prompt_ablation_smollm'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    datasets={}
    for name,path,system in [('esci',a.esci_run,ESCI_SYSTEM),('wands',a.wands_run,run_wands.SYSTEM)]:
        rows=lines(path/'inputs.jsonl');original=lines(path/'outputs.jsonl');assert len(rows)==len(original)
        assert all(x['query_id']==y['query_id'] for x,y in zip(rows,original))
        datasets[name]={'rows':rows,'original':original,'system':system}
    import torch,transformers,vllm
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,tokenizer=a.model,trust_remote_code=False,dtype='bfloat16',max_model_len=4096,
            gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=True,disable_log_stats=True)
    tok=llm.get_tokenizer()
    def wrap(system,p):return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
    prompts=[];locations=[]
    for name,d in datasets.items():
        for i,(row,previous) in enumerate(zip(d['rows'],d['original'])):
            e=previous['plans']['two_stage_compact_fallback'];assert e['spec'] is not None
            spec=base.parse_spec(e['spec']['final']);assert bool(e['fallback'])==(spec is None)
            for part in harness.chunks(row):
                prompts.append(wrap(d['system'],prompt_without_example(part,spec)))
                locations.append((name,i,part['start'],len(part['items'])))
    assert prompts
    start=time.perf_counter();first_results=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)
    first=[r.outputs[0].text.strip() for r in first_results];final=first[:]
    bad=[i for i,x in enumerate(first) if parse_compact(x,locations[i][3]) is None]
    # The original retry appends the message to the user prompt, not to the wrapped chat string.
    retries=[]
    for j in bad:
        name,i,startidx,_=locations[j];d=datasets[name];row=d['rows'][i];entry=d['original'][i]['plans']['two_stage_compact_fallback']
        spec=base.parse_spec(entry['spec']['final']);part=next(x for x in harness.chunks(row) if x['start']==startidx)
        retries.append(wrap(d['system'],prompt_without_example(part,spec)+'\nYour previous answer had the wrong format; follow the output instructions exactly.'))
    retry_results=llm.generate(retries,SamplingParams(temperature=0,max_tokens=32),use_tqdm=False) if retries else []
    for j,res in zip(bad,retry_results):final[j]=res.outputs[0].text.strip()
    seconds=time.perf_counter()-start
    new={name:[{'pred':[None]*len(r['items']),'raw':[]} for r in d['rows']] for name,d in datasets.items()}
    for j,(name,i,startidx,size) in enumerate(locations):
        parsed=parse_compact(final[j],size);e=new[name][i]
        e['raw'].append({'start':startidx,'size':size,'first':first[j],'final':final[j]})
        if parsed is not None:e['pred'][startidx:startidx+size]=parsed
    result={'protocol':'PROTOCOL_compact_prompt_ablation.md','model':os.path.basename(a.model),
            'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,
                       'gpu':torch.cuda.get_device_name(0),'dtype':'bfloat16','max_model_len':4096},
            'initial_chunk_calls':len(first),'retry_chunk_calls':len(bad),'offline_generation_seconds':seconds,'datasets':{}}
    for name,d in datasets.items():
        rows=d['rows'];prior=d['original'];preds=[];raw=[]
        for row,e in zip(rows,new[name]):
            pred=e['pred'] if None not in e['pred'] else None;preds.append(pred)
            raw.append({'query_id':row['query_id'],'pred':pred,'raw':e['raw']})
        with (a.out/(name+'_outputs.jsonl')).open('w',encoding='utf8') as f:
            for x in raw:f.write(json.dumps(x,ensure_ascii=False)+'\n')
        idx=[i for i,p in enumerate(preds) if p is not None]
        prioridx=[i for i,o in enumerate(prior) if o['plans']['two_stage_compact_fallback']['pred'] is not None]
        common=sorted(set(idx)&set(prioridx))
        chunks=[x for record in raw for x in record['raw']]
        previous_chunks=[x for record in prior for x in record['plans']['two_stage_compact_fallback']['raw']]
        stats={'attempted_pools':len(rows),'original_complete_pools':len(prioridx),'corrected_complete_pools':len(idx),
               'common_complete_pools':len(common),'chunks':len(chunks),
               'valid_first_chunks':sum(parse_compact(x['first'],x['size']) is not None for x in chunks),
               'valid_final_chunks':sum(parse_compact(x['final'],x['size']) is not None for x in chunks),
               'literal_example_final_original':sum(x['final'].strip() in ('YNNY','YNNY.') for x in previous_chunks),
               'literal_example_final_corrected':sum(x['final'].strip() in ('YNNY','YNNY.') for x in chunks),
               'specification_fallback_pools':sum(o['plans']['two_stage_compact_fallback']['fallback'] for o in prior)}
        if idx:
            stats['corrected_quality_on_own_complete']=metrics(rows,preds,idx)
            for label,name2 in [('Y','always_yes'),('N','always_no')]:
                constants=[[label]*len(row['items']) for row in rows]
                stats[name2+'_on_corrected_complete']=metrics(rows,constants,idx)
        if common:
            original_preds=[o['plans']['two_stage_compact_fallback']['pred'] for o in prior]
            stats['corrected_quality_on_common']=metrics(rows,preds,common)
            stats['original_quality_on_common']=metrics(rows,original_preds,common)
        result['datasets'][name]=stats
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
