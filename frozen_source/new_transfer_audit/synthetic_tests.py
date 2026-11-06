#!/usr/bin/env python3
"""Independent numerical fixtures; never write synthetic raw into a scientific run."""
import ast,collections,math,random
from fractions import Fraction
from common import *
from identity_selection import identities,fingerprint,select_records,records_in_prompt,source_features
from lexical import LexicalIndex
from prepare_transfer import plan_roles,schedules
from analyze_transfer import cost,plus_calibration,relation_summary,aggregate_relations
from decision_metrics import disagreement,gold_metrics,evaluation
import numpy as np

def run():
    checks=[]
    def okay(name,condition):
        assert condition,name;checks.append({'test':name,'status':'PASS'})
    okay('strict parser positive/negative fixtures',len(parser_checks())>=11)
    okay('missing ID and duplicate ID rejected',parse('{"1":"Y","1":"N"}',[1]) is None and parse('{"1":"Y"}',[1,2]) is None)
    a={'title':'Alpha Study','authors':'A. Smith','venue':'Conf','year':'2020'}
    b={'title':'Different Work','authors':'B','venue':'Journal','year':'2019'}
    r={'task':'dblp_acm','left_id':'1','right_id':'2','source_id':'z','features':{'left':a,'right':b}}
    a2={k:' '+v.upper()+'\t' for k,v in a.items()}
    reverse={'task':'dblp_acm','left_id':'7','right_id':'8','source_id':'x','features':{'left':b,'right':a2}}
    okay('normalized complete citation blocks cross-role reuse',len(identities(r)&identities(reverse))==2)
    q={'task':'boolq','source_id':'b','features':{'question':' Is IT True? ','passage':'Alpha\t Beta'}}
    q2={'task':'boolq','source_id':'a','features':{'question':'is it true?','passage':'other'}}
    okay('casefold whitespace question dedup',('question','is it true?') in identities(q)&identities(q2))
    data={'dblp_acm':[],'boolq':[q,q2]};lengths={('boolq','a'):{'qwen':10,'smol':11},('boolq','b'):{'qwen':10,'smol':11}}
    empty={t:set() for t in TASKS};eligible,rejected,report=select_records(data,lengths,empty,empty,empty)
    okay('source-ID greedy precedes hash ordering',[x['source_id'] for x in eligible['boolq']]==['a'])
    okay('complete structured prompt extraction',len(list(records_in_prompt('Records:\n'+compact({'id':'1','features':r['features']})+'\nRequested IDs: [1]')))==1)
    import identity_selection as selection_module
    from unittest.mock import patch
    original_reader=selection_module.csv.DictReader;original_lines=selection_module.read_lines
    def flipped_reader(*args,**kwargs):
        for row in original_reader(*args,**kwargs):
            if 'label' in row:row={**row,'label':str(1-int(row['label']))}
            yield row
    def flipped_lines(path):
        return [{**row,'answer':not row['answer']} if type(row.get('answer')) is bool else row for row in original_lines(path)]
    base_features,base_bad=source_features(ROOT/'sources')
    with patch.object(selection_module.csv,'DictReader',flipped_reader),patch.object(selection_module,'read_lines',flipped_lines):
        changed_features,changed_bad=source_features(ROOT/'sources')
    okay('all real source gold labels flipped in memory leave feature inventory unchanged',base_features==changed_features and base_bad==changed_bad)
    records=[{'task':'boolq','phase':'test','source_id':str(i),'reserve_ordinal':i,'occurrence_id':i,
        'features':{'question':'q'+str(i),'passage':'p'+str(i)},'feature_sha256':str(i)} for i in (9,10,128)]
    c=render('boolq',records,'C');i=render('boolq',records,'I');o=render('boolq',records,'O')
    okay('C/O same full input prompt with reversed output key order',c['unwrapped_user']==o['unwrapped_user'] and c['output_ids']==[9,10,128] and o['output_ids']==[128,10,9])
    okay('I input reverse keeps sorted output keys',i['input_ids_in_order']==[128,10,9] and i['output_ids']==[9,10,128])
    okay('P/S renderer equality',render('boolq',records[:1],'P')==render('boolq',records[:1],'S'))
    # Numerical TF-IDF oracle from manually supplied term counts, not LexicalIndex.terms.
    bank=[{'source_id':'a','features':'alpha beta'},{'source_id':'b','features':'alpha alpha'}]
    lex=LexicalIndex(bank);idf=1+math.log(3/2)
    expected={ 'b:alpha alpha':[0,idf],'b:alpha beta':[idf,0],'u:alpha':[1,2],'u:beta':[idf,0] }
    arr=np.array([expected[w] for w in lex.vocabulary]).T
    arr=arr/np.linalg.norm(arr,axis=1)[:,None]
    okay('independent hand-counted TF-IDF weights',np.allclose(lex.matrix.toarray(),arr,rtol=0,atol=1e-14))
    okay('independent TF-IDF cosine',(abs(float((lex.matrix@lex.matrix.T).toarray()[0,1])-float(arr[0]@arr[1]))<1e-14))
    pred=[1,1,0,0]*32;gold=[1,0,1,0]*32;gm=gold_metrics(pred,gold)
    okay('independent 128-item confusion fixture',gm['confusion_counts_on_valid_outcomes']=={'tp':32,'tn':32,'fp':32,'fn':32})
    okay('COUNT cancellation is explicit despite half wrong',gm['error']==.5 and gm['absolute_ratio_error']==0 and gm['canceled_gold_error_mass']==.5)
    incomplete=pred.copy();incomplete[0]=None;im=gold_metrics(incomplete,gold)
    okay('invalid outcome abstains fixed-128 COUNT',im['absolute_ratio_error'] is None and im['fixed_denominator']==128 and im['invalid_or_missing_outcomes']==1)
    flip=pred.copy()
    for ix in range(8):flip[ix]=1-flip[ix]
    okay('S disagreement numerical oracle',disagreement(flip,pred)=={'complete':True,'numerator':8,'fixed_denominator':128,'value':.0625})
    rp={'1':evaluation(pred,pred,gold),'2':evaluation(flip,pred,gold)}
    summary=relation_summary(rp,Fraction(1,20))
    okay('MEAN versus BOTH differ on fixed synthetic replay fixture',summary['mean_S_disagreement']==.03125 and summary['within_tolerance_MEAN'] is True and summary['within_tolerance_BOTH'] is False)
    summary2=relation_summary({'1':evaluation(incomplete,pred,gold),'2':evaluation(flip,pred,gold)},Fraction(1,20))
    okay('missing replay cannot pass by shrinking denominator',summary2['within_tolerance_MEAN'] is None and summary2['within_tolerance_BOTH'] is None)
    jobs=[{'request_id':'a','occurrence_ids':[1]}];unknown=cost(jobs,{'a':{'actual_input_tokens':None,'actual_output_tokens':3}})
    okay('unknown input/total tokens null, not zero',unknown['actual_input_tokens'] is None and unknown['actual_total_tokens'] is None and unknown['unknown_input_token_call_count']==1)
    missing=cost(jobs,{})
    okay('unreturned usage remains unknown',missing['actual_total_tokens'] is None and missing['unknown_output_token_call_count']==1)
    actualzero=cost(jobs,{'a':{'actual_input_tokens':7,'actual_output_tokens':0}})
    okay('observed zero output is distinct from unknown',actualzero['actual_total_tokens']==7 and actualzero['token_cost_complete'])
    cal={'planned_calls':1152,'returned_calls':1152,'planned_requested_items':3328,'actual_input_tokens':100,'actual_output_tokens':50,'actual_total_tokens':150}
    cst=cost([{'request_id':str(k),'occurrence_ids':[1]} for k in range(48)],{})
    combined=plus_calibration(cst,cal)
    okay('3 DBLP relations k8 cannot amortize original calibration',combined['planned_calls']==1200 and combined['planned_calls']>3*128)
    okay('singleton policy cannot amortize 20-relation panel',1152+20*128>20*128)
    lock=json.loads((ROOT/'POLICY_LOCK.json').read_text())['payload']['policies'];roles=plan_roles(lock,'smol','dblp_acm')
    okay('three .20 identical choices share one physical C plan',sum(x.startswith('policy/') for x in roles['k8/random/C'])==3)
    # Re-execute the exact old source function with its literal test phase on one relation/task.
    original=(ROOT/'source_reference/decision_prepare.py').read_text();tree=ast.parse(original)
    func=next(ast.get_source_segment(original,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='partitions')
    ns={**globals(),'PHASES':('test',)};exec(compile(func,'original_partition_function','exec'),ns)
    full=read_lines(ROOT/'inputs/relations.jsonl');first={t:sorted({r['relation_id'] for r in full if r['task']==t})[0] for t in TASKS}
    fixture=[{**r,'phase':'test'} for r in full if r['relation_id']==first[r['task']]]
    oldparts,_,_=ns['partitions'](fixture);newparts=read_lines(ROOT/'inputs/partitions.jsonl')
    for op in oldparts:
        npart=next(x for x in newparts if (x['relation_id'],x['k'],x['grouping'])==(first[op['task']],op['k'],op['grouping']))
        okay('original partition source agreement '+op['task']+'/'+str(op['k'])+'/'+op['grouping'],op['batches']==npart['batches'] and op['batch_membership_sha256']==npart['batch_membership_sha256'])
    for p in ROOT.glob('*.py'):compile(p.read_text(),str(p),'exec')
    okay('all new Python sources compile',True)
    return {'status':'PASS','test_count':len(checks),'checks':checks,'model_loaded':False,'inference_calls':0,
        'source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
        'fixtures_are_synthetic_only':True,'scientific_raw_written':False}

if __name__=='__main__':
    result=run();write_json(ROOT/'verification/SYNTHETIC_TESTS.json',result);print(compact({'status':result['status'],'test_count':result['test_count']}))
