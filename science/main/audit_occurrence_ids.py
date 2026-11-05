#!/usr/bin/env python3
"""Report within-pool repeated source identifiers without changing samples."""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCES=(('esci_original','run_full_pools_250/inputs.jsonl','product_id'),
         ('esci_new','esci_selective_holdout_selection/inputs.jsonl','product_id'),
         ('wands_initial','run_wands_160/inputs.jsonl','product_id'),
         ('wands_holdout','wands_holdout_selection/inputs.jsonl','product_id'),
         ('wands_fallback','wands_fallback_validation_selection/inputs.jsonl','product_id'),
         ('wands_selective','wands_selective_execution_selection/inputs.jsonl','product_id'),
         ('trec_covid','trec_covid_transfer_selection/inputs.jsonl','doc_id'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a new output path.')
    result={}
    for name,relative,key in SOURCES:
        with (ROOT/relative).open(encoding='utf8') as f:rows=[json.loads(line) for line in f]
        duplicates=[]
        for row in rows:
            if not row['items'] or any(key not in item for item in row['items']):
                raise ValueError('Missing source identity in '+name)
            counts=Counter(str(item[key]) for item in row['items'])
            duplicates.append(sum(n-1 for n in counts.values() if n>1))
        result[name]={'pools':len(rows),'candidates':sum(len(row['items']) for row in rows),
                      'pools_with_repeated_source_id':sum(x>0 for x in duplicates),
                      'repeated_occurrences_within_pools':sum(duplicates)}
    result['limitations']=['Derived ESCI pilot and validation do not retain original product IDs.',
        'Distinct IDs can still share identical text or represent the same product.',
        'The audit does not impose duplicate-consistency constraints on model outputs.']
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Audited source IDs within',sum(x['pools'] for x in result.values() if isinstance(x,dict)),'pools.')


if __name__=='__main__':main()
