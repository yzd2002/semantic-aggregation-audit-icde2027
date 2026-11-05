#!/usr/bin/env python3
"""Independently reconstruct keyed counts and check exact mixed-pool bounds."""
import argparse,hashlib,json,re
from fractions import Fraction
from pathlib import Path


def lines(p):
    with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def fraction(record):
    value=Fraction(record['numerator'],record['denominator'])
    if abs(float(value)-record['decimal'])>1e-14:raise ValueError('Inconsistent numeric record')
    return value


def reconstruct(rows,records,reverse):
    pos=0;output=[]
    for row in rows:
        vals=[]
        for start in range(0,len(row['items']),8):
            count=min(8,len(row['items'])-start)
            if pos>=len(records):raise ValueError('Missing response')
            event=records[pos];pos+=1
            ids=list(range(1,count+1))
            if (event['query_id'],event['start'],event['size'],event['presented_ids']) != (
                    row['query_id'],start,count,ids[::-1] if reverse else ids):raise ValueError('Response association mismatch')
            pattern=r'\{' + ','.join(r'"'+str(i)+r'":"[YN]"' for i in ids)+r'\}'
            if re.fullmatch(pattern,event['final']) is None:raise ValueError('Invalid keyed response')
            obj=json.loads(event['final']);vals.extend(obj[str(i)] for i in ids)
        output.append(vals)
    if pos!=len(records):raise ValueError('Extra responses')
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--analysis-dir',type=Path,required=True)
    a=ap.parse_args();root=a.root;here=a.analysis_dir
    frozen=json.loads((here/'PRE_ANALYSIS_SHA256.json').read_text())
    local={'PROTOCOL_variable_pool_floor.md','variable_pool_floor.py','test_variable_pool_floor.py','analyze_variable_pool_floor.py'}
    for name,wanted in frozen.items():
        if sha((here if name in local else root)/name)!=wanted:raise ValueError('Changed frozen input: '+name)
    expected=json.loads((here/'summary.json').read_text())
    rows=lines(root/'esci_selective_holdout_selection/inputs.jsonl')
    if len(rows)!=250 or sum(len(r['items']) for r in rows)!=4428:raise ValueError('Unexpected sample')
    for name,wanted in expected['input_sha256'].items():
        if sha(root/name)!=wanted:raise ValueError('Changed result input: '+name)
    for model in ('qwen','smollm'):
        base=root/f'run_esci_selective_holdout_{model}'
        first=reconstruct(rows,lines(base/'original_responses.jsonl'),False)
        second=reconstruct(rows,lines(base/'stable_reverse_responses.jsonl'),True)
        saved=expected['models'][model];summed=Fraction();wa=Fraction();wb=Fraction();aa=Fraction();ab=Fraction();changed=0
        if len(saved['pool_counts'])!=len(rows):raise ValueError('Pool count differs')
        for row,x,y,item in zip(rows,first,second,saved['pool_counts'],strict=True):
            n=len(row['items']);pa=x.count('Y');pb=y.count('Y');r=item['synthetic_reference_positive']
            if (row['query_id'],n,pa,pb)!=(item['query_id'],item['n'],item['original_positive'],item['reverse_positive']):
                raise ValueError('Saved positive counts differ from parsed responses')
            if not min(pa,pb)<=r<=max(pa,pb):raise ValueError('Invalid witness reference count')
            gold=sum(v['gold']=='Y' for v in row['items'])
            summed+=Fraction(abs(pa-pb),n);changed+=pa!=pb
            wa+=Fraction(abs(pa-r),n);wb+=Fraction(abs(pb-r),n)
            aa+=Fraction(abs(pa-gold),n);ab+=Fraction(abs(pb-gold),n)
        q=len(rows);floor=summed/(2*q);wa/=q;wb/=q;aa/=q;ab/=q
        slack=Fraction(1,2*q*min(len(row['items']) for row in rows))
        if saved['pools_with_changed_positive_count']!=changed:raise ValueError('Changed count differs')
        checks={'floor':floor,'weighted_count_gap_sum':summed,'witness_mae':max(wa,wb),
                'witness_mae_first':wa,'witness_mae_second':wb,'guaranteed_witness_slack':slack,
                'observed_ES_positive_mae_original':aa,'observed_ES_positive_mae_reversed':ab}
        for name,value in checks.items():
            if fraction(saved[name])!=value:raise ValueError('Exact statistic mismatch: '+name)
        if not floor<=max(wa,wb)<=floor+slack or max(aa,ab)<floor:raise ValueError('Bound failed')
    print('Independently parsed both models\' keyed responses and verified exact rational floors and witnesses.')


if __name__=='__main__':main()
