#!/usr/bin/env python3
"""Exercise paired analysis and independent verification on fabricated traces."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from analyze_esci_truncation import measure
from run_esci_truncation import SUFFIX,parse,sha
import run_experiment as base


def write_lines(p,rows):p.write_text(''.join(json.dumps(x)+'\n' for x in rows))


def main():
    tmp=Path(tempfile.mkdtemp(prefix='esci-truncation-test-',dir='/dev/shm'))
    selection=tmp/'selection';run=tmp/'run';selection.mkdir();run.mkdir()
    rows=[]
    for q in range(3):
        items=[]
        for i in range(4):
            short='sample product '+str(i);long=short+' extra' if i%2==0 else short
            items.append({'id':i+1,'product_id':'p'+str(q*2+i),'example_id':str(q*4+i),
                'grade':'E' if i%3==0 else 'I','gold':'Y' if i%3==0 else 'N',
                'short':short,'long':long,'source_characters':len(long),'text_changed':short!=long})
        rows.append({'query_id':'q'+str(q),'query':'sample query','items':items})
    write_lines(selection/'inputs.jsonl',rows)
    selection_hash=hashlib.sha256((selection/'inputs.jsonl').read_bytes()).hexdigest()
    (selection/'selection.json').write_text(json.dumps({'inputs_sha256':selection_hash,'changed_candidates':6}))
    row=rows[0];item=row['items'][0]
    prompt=base.p_point(row,{'text':item['short']})
    altered={**row,'items':[dict(x,gold='N',grade='secret') for x in row['items']]}
    assert base.p_point(altered,{'text':item['short']})==prompt
    assert parse('Y')=='Y' and parse('Y\n') is None and parse('banana') is None
    metrics,errors=measure([row],[['Y','N','N','N']])
    assert metrics['tp']==1 and metrics['fn']==1 and metrics['tn']==2 and errors.tolist()==[.25]
    modes=('short','long','replay');stats={}
    for mode in modes:
        records=[]
        for r in rows:
            for it in r['items']:
                field='long' if mode=='long' else 'short';p=base.p_point(r,{'text':it[field]})
                first='Y' if (it['id']+int(mode=='long' and it['text_changed']))%3==0 else 'N'
                records.append({'query_id':r['query_id'],'item_id':it['id'],'product_id':it['product_id'],
                    'text_changed':it['text_changed'],'prompt_sha256':sha(p),'wrapped_sha256':'not-used',
                    'first':first,'final':first,'first_finish_reason':'stop','retry_finish_reason':None,
                    'first_input_tokens':1,'first_output_tokens':1,'retry_input_tokens':0,'retry_output_tokens':0})
        write_lines(run/(mode+'_responses.jsonl'),records);n=len(records)
        stats[mode]={'initial_calls':n,'retry_calls':0,'valid_first':n,'valid_final':n,'complete_pools':3,
            'first_finish_reasons':{'stop':n},'logical_tokens':{'input':n,'output':n},
            'max_retry_prompt_plus_cap':100,'offline_generation_seconds':0}
    (run/'summary.json').write_text(json.dumps({'protocol':'PROTOCOL_esci_truncation_sensitivity.md',
        'selection_inputs_sha256':selection_hash,'system':base.SYSTEM,'retry_suffix':SUFFIX,'modes':stats}))
    (run/'guidance_sanity.json').write_text(json.dumps({'ordinary':{'text':'BANANA'},'guided':{'text':'Y'},
                                                       'checks':{'ordinary_invalid':True,'guided_valid':True}}))
    directory=Path(__file__).resolve().parent;analysis=tmp/'analysis.json'
    subprocess.run([sys.executable,str(directory/'analyze_esci_truncation.py'),'--selection',str(selection),
                    '--run',str(run),'--out',str(analysis)],check=True,stdout=subprocess.DEVNULL)
    verify=[sys.executable,str(directory/'verify_esci_truncation.py'),'--selection',str(selection),
            '--run',str(run),'--analysis',str(analysis)]
    subprocess.run(verify,check=True,stdout=subprocess.DEVNULL)
    path=run/'long_responses.jsonl';records=[json.loads(x) for x in path.read_text().splitlines()]
    records[0]['prompt_sha256']='0'*64;write_lines(path,records)
    assert subprocess.run(verify,capture_output=True).returncode!=0
    print('Synthetic paired metrics, prompt isolation, independent verification and tampering check passed.')


if __name__=='__main__':main()
