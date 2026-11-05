#!/usr/bin/env python3
"""Check saved-response provenance without printing queries or responses."""
import argparse,ast,collections,json
from pathlib import Path


def has_text_strip(path):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='strip'
               and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='text'
               for n in ast.walk(tree))


def has_raw_record(path):
    tree=ast.parse(path.read_text(encoding='utf8'))
    for n in ast.walk(tree):
        if not isinstance(n,ast.Dict):continue
        fields={k.value:v for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant) and isinstance(k.value,str)}
        first,final=fields.get('first'),fields.get('final')
        if (isinstance(first,ast.Attribute) and isinstance(first.value,ast.Name) and first.value.id=='output' and first.attr=='text'
            and isinstance(final,ast.Attribute) and isinstance(final.value,ast.Name) and final.value.id=='last' and final.attr=='text'):
            return True
    return False


def main(repro,acord,out):
    if out.exists():raise FileExistsError('Use a fresh output path')
    early=('run_experiment.py','run_validation.py','run_full_pools.py','run_keyed_order_audit.py')
    assert all(has_text_strip(repro/name) for name in early)
    assert has_raw_record(repro/'run_trec_covid_transfer.py')
    assert has_raw_record(acord/'run_acord_holdout.py')
    stats=collections.Counter()
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('first','final') and isinstance(v,str):
                    stats['response_strings']+=1;stats['with_surrounding_whitespace']+=v!=v.strip()
                walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    with (repro/'run_validation_500/outputs.jsonl').open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line);stats['pools']+=1;walk(row.get('plans',{}))
    assert stats['pools']==500 and stats['response_strings']==13102 and stats['with_surrounding_whitespace']==0
    result={'scope':'Static source inspection and saved-string counts only; no model inference.',
            'early_wrapper_files':list(early),'validation_500':dict(stats),
            'later_raw_record_files':['run_trec_covid_transfer.py','run_acord_holdout.py'],
            'limitation':'The check cannot recover or determine leading/trailing whitespace in the original early model responses.'}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Checked early response wrappers, 500 saved pools, and later raw-record code; no response text printed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repro',type=Path,required=True);p.add_argument('--acord',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repro,a.acord,a.out)
