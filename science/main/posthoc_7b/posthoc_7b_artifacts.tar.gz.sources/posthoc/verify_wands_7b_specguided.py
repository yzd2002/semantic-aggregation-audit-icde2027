#!/usr/bin/env python3
"""Independently verify the actual selective scheduler and paired evaluation."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import run_experiment as base
import run_wands
from verify_order_fallback_validation import close, digest, key_prompt, keyed, lines, point, single_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--model', help='Optional local tokenizer for independent prompt length and input-token checks')
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    metadata = json.loads((a.selection / 'selection.json').read_text())
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    assert len(rows) == 52 and all(len(r['items']) == 32 for r in rows)
    assert metadata['protocol'] == 'PROTOCOL_selective_execution.md'
    assert summary['protocol'] == analysis['protocol'] == 'PROTOCOL_7b_specguided_posthoc.md'
    assert summary['selection_inputs_sha256'] == analysis['selection_inputs_sha256'] == sha
    assert summary['model'] == analysis['model'] == 'Qwen2.5-7B-Instruct'
    assert keyed(sanity['unconstrained']) is None and point(sanity['unconstrained']) is None
    assert keyed(sanity['json_guided']) is not None and point(sanity['point_guided']) is not None
    assert not sanity['unconstrained_valid_json'] and not sanity['unconstrained_valid_point']
    assert sanity['guided_json_valid'] and sanity['guided_point_valid']
    modes = ['original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise']
    assert summary['mode_order'] == modes and summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    records = {mode: lines(a.run / (mode + '_responses.jsonl')) for mode in modes}
    specs = lines(a.run / 'specifications.jsonl')
    assert len(specs) == len(rows)
    first_bad = final_bad = 0
    for row, rec in zip(rows, specs):
        assert rec['query_id'] == row['query_id'] and rec['prompt_sha256'] == digest(base.p_spec(row))
        x, y = base.parse_spec(rec['first']), base.parse_spec(rec['final'])
        first_bad += x is None; final_bad += y is None
        if x is not None:assert rec['first'] == rec['final']
        assert rec['parsed'] == y and rec['fallback'] == (y is None)
    assert summary['specifications']['initial_calls'] == 52
    assert summary['specifications']['retry_calls'] == first_bad and summary['specifications']['fallback_pools'] == final_bad
    tok = None
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False, local_files_only=True)
    def tokens(text):
        return len(tok.apply_chat_template([{'role': 'system', 'content': run_wands.SYSTEM},
                                            {'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
    suffix = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
    if tok:
        mx = max(tokens(base.p_spec(row) + '\nReturn valid JSON only.') + 160 for row in rows)
        spec_inputs = sum(tokens(base.p_spec(row)) + (tokens(base.p_spec(row)+'\nReturn valid JSON only.')
                          if base.parse_spec(rec['first']) is None else 0) for row, rec in zip(rows, specs))
        assert mx == summary['specifications']['max_prompt_with_retry_plus_cap'] <= 4096
        assert spec_inputs == summary['specifications']['logical_tokens']['input']
    pred = {}; point_prompts = {}
    for mode in ('original', 'stable_reverse', 'replay', 'pointwise'):
        data = records[mode]; per = 32 if mode == 'pointwise' else 4
        assert len(data) == len(rows)*per
        vectors = []; max_context = 0
        for q, (row, spec) in enumerate(zip(rows, specs)):
            vector = []
            for j in range(per):
                rec = data[q*per+j]; start = j if mode == 'pointwise' else 8*j
                assert rec['query_id'] == row['query_id'] and rec['start'] == start
                if mode == 'pointwise':
                    assert rec['size'] == 1 and rec['presented_ids'] is None
                    prompt = single_prompt(row, row['items'][start], spec['parsed']); parser, cap = point, 4
                    point_prompts[(row['query_id'], start)] = prompt
                else:
                    assert rec['size'] == 8
                    assert rec['presented_ids'] == (list(range(8, 0, -1)) if mode == 'stable_reverse' else list(range(1, 9)))
                    prompt = key_prompt(row, row['items'][start:start+8], spec['parsed'], mode == 'stable_reverse'); parser, cap = keyed, 128
                assert rec['prompt_sha256'] == digest(prompt)
                first, final = parser(rec['first']), parser(rec['final'])
                if first is None:assert rec['retry_finish_reason'] is not None
                else:assert rec['first'] == rec['final'] and rec['retry_finish_reason'] is None
                vector.extend((final if final is not None else [None]*8) if mode != 'pointwise' else [final])
                if tok:
                    assert rec['first_input_tokens'] == tokens(prompt)
                    assert rec['retry_input_tokens'] == (tokens(prompt+suffix) if first is None else 0)
                    max_context = max(max_context, tokens(prompt+suffix)+cap)
            vectors.append(vector if None not in vector else None)
        pred[mode] = vectors
        stats = summary['modes'][mode]; parser = point if mode == 'pointwise' else keyed
        assert stats['initial_calls'] == len(data) and stats['retry_calls'] == sum(parser(x['first']) is None for x in data)
        assert stats['valid_first'] == sum(parser(x['first']) is not None for x in data)
        assert stats['valid_final'] == sum(parser(x['final']) is not None for x in data)
        assert stats['finish_reasons'] == dict(collections.Counter(x['first_finish_reason'] for x in data))
        for key in ('input', 'output'):
            assert stats['logical_tokens'][key] == sum(x['first_'+key+'_tokens']+x['retry_'+key+'_tokens'] for x in data)
        if tok:assert stats['max_prompt_with_retry_plus_cap'] == max_context <= 4096
    need = []
    for q, row in enumerate(rows):
        for c in range(4):
            x = keyed(records['original'][q*4+c]['final']); y = keyed(records['stable_reverse'][q*4+c]['final'])
            if x is not None and y is not None:
                need.extend((q, 8*c+i) for i in range(8) if x[i] != y[i])
    selective = records['selective_pointwise']
    assert [(x['query_id'],x['start']) for x in selective] == [(rows[q]['query_id'],i) for q,i in need]
    selected = {}; max_context = 0; matched = changed = 0
    for (q,i), rec in zip(need, selective):
        key = (rows[q]['query_id'],i); prompt = point_prompts[key]
        assert rec['size'] == 1 and rec['presented_ids'] is None and rec['prompt_sha256'] == digest(prompt)
        first, final = point(rec['first']), point(rec['final'])
        if first is None:assert rec['retry_finish_reason'] is not None
        else:assert rec['first'] == rec['final'] and rec['retry_finish_reason'] is None
        selected[key] = final
        full = records['pointwise'][q*32+i]
        assert full['prompt_sha256'] == rec['prompt_sha256']
        if final is not None and point(full['final']) is not None:
            matched += 1; changed += final != full['final']
        if tok:
            assert rec['first_input_tokens'] == tokens(prompt)
            assert rec['retry_input_tokens'] == (tokens(prompt+suffix) if first is None else 0)
            max_context = max(max_context,tokens(prompt+suffix)+4)
    stats = summary['modes']['selective_pointwise']
    assert stats['initial_calls'] == len(need) and stats['retry_calls'] == sum(point(x['first']) is None for x in selective)
    assert stats['valid_first'] == sum(point(x['first']) is not None for x in selective)
    assert stats['valid_final'] == sum(point(x['final']) is not None for x in selective)
    assert stats['finish_reasons'] == dict(collections.Counter(x['first_finish_reason'] for x in selective))
    for key in ('input','output'):
        assert stats['logical_tokens'][key] == sum(x['first_'+key+'_tokens']+x['retry_'+key+'_tokens'] for x in selective)
    if tok:assert stats['max_prompt_with_retry_plus_cap'] == max_context <= 4096
    full_partial = [[point(x['final']) for x in records['pointwise'][q*32:(q+1)*32]] for q in range(len(rows))]
    sparse = [[selected.get((row['query_id'],i)) for i in range(32)] for row in rows]
    def combine(x,y,h):
        if x is None or y is None:return None
        z = [a if a == b else c for a,b,c in zip(x,y,h)]
        return z if None not in z else None
    pred['actual_fallback'] = [combine(x,y,h) for x,y,h in zip(pred['original'],pred['stable_reverse'],sparse)]
    pred['full_trace_fallback'] = [combine(x,y,h) for x,y,h in zip(pred['original'],pred['stable_reverse'],full_partial)]
    saved = lines(a.run / 'selective_outputs.jsonl')
    assert len(saved) == len(rows) and all(r['query_id'] == x['query_id'] and x['pred'] == p
                                           for r,x,p in zip(rows,saved,pred['actual_fallback']))
    pred['always_N'] = [['N']*32 for row in rows]
    pred['lexical'] = []
    for row in rows:
        words = set(re.findall(r'\w+',row['query'].casefold()))
        pred['lexical'].append(['Y' if words and words <= set(re.findall(r'\w+',x['text'].casefold())) else 'N' for x in row['items']])
    common = [i for i in range(len(rows)) if all(pred[k][i] is not None for k in
              ('original','stable_reverse','replay','pointwise','actual_fallback','full_trace_fallback'))]
    assert common and analysis['common_complete_pools'] == len(common)
    assert analysis['common_query_ids'] == [rows[i]['query_id'] for i in common]
    assert analysis['attempted_pools'] == 52 and analysis['candidate_count'] == 1664
    assert analysis['resamples'] == 4000 and analysis['seed'] == 20271202
    assert analysis['completion'] == {k: sum(x is not None for x in v) for k,v in pred.items()}
    assert analysis['grade_counts'] == dict(collections.Counter(x['grade'] for row in rows for x in row['items']))
    assert analysis['specification_fallback_pools'] == final_bad
    for key,vectors in pred.items():
        tp=fp=tn=fn=0;errors=[]
        for i in common:
            g=[x['gold'] for x in rows[i]['items']];y=vectors[i]
            errors.append(abs(y.count('Y')-g.count('Y'))/len(g))
            for p,t in zip(y,g):
                tp += p == t == 'Y';tn += p == t == 'N';fp += p == 'Y' and t == 'N';fn += p == 'N' and t == 'Y'
        vals={'tp':tp,'tn':tn,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
              'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
              'ratio_mae':sum(errors)/len(errors)}
        assert all(close(v,analysis['quality_common'][key][k]) for k,v in vals.items())
    draw=np.random.default_rng(20271202).integers(0,len(common),size=(4000,len(common)))
    for name,data in analysis['comparisons'].items():
        left,right=data['left'],data['right'];points=[]
        for i in common:
            gold=[x['gold'] for x in rows[i]['items']]
            def stat(y):
                tp=sum(p==g=='Y' for p,g in zip(y,gold));fp=sum(p=='Y' and g=='N' for p,g in zip(y,gold))
                fn=sum(p=='N' and g=='Y' for p,g in zip(y,gold));return tp,fp,fn,abs(y.count('Y')-gold.count('Y'))/len(gold)
            points.append(stat(pred[left][i])+stat(pred[right][i]))
        arr=np.asarray(points,dtype=float)[draw];totals=arr.sum(axis=1)
        def f1(j):
            n=2*totals[:,j];d=n+totals[:,j+1]+totals[:,j+2]
            return np.divide(n,d,out=np.zeros_like(n),where=d!=0)
        fd=f1(4)-f1(0);md=arr[:,:,7].mean(axis=1)-arr[:,:,3].mean(axis=1)
        assert close(data['f1_difference'],analysis['quality_common'][right]['f1']-analysis['quality_common'][left]['f1'])
        assert close(data['ratio_mae_difference'],analysis['quality_common'][right]['ratio_mae']-analysis['quality_common'][left]['ratio_mae'])
        assert close(data['f1_difference_ci'],np.quantile(fd,[.025,.975]).tolist())
        assert close(data['ratio_mae_difference_ci'],np.quantile(md,[.025,.975]).tolist())
    pipeline=summary['selective_pipeline'];stages=[summary['modes'][m] for m in ('original','stable_reverse','selective_pointwise')]
    spec=summary['specifications']
    assert pipeline['initial_calls'] == spec['initial_calls']+sum(x['initial_calls'] for x in stages)
    assert pipeline['retry_calls'] == spec['retry_calls']+sum(x['retry_calls'] for x in stages)
    for key in ('input','output'):
        assert pipeline['logical_tokens'][key] == spec['logical_tokens'][key]+sum(x['logical_tokens'][key] for x in stages)
    assert pipeline['complete_pools'] == sum(x is not None for x in pred['actual_fallback'])
    assert pipeline['offline_batch_wall_seconds'] >= spec['offline_generation_seconds']+sum(x['offline_generation_seconds'] for x in stages)
    result=analysis['selective']
    assert result['disagreeing_candidates'] == len(need) == result['actual_initial_calls']
    assert result['actual_retry_calls'] == stats['retry_calls'] and result['actual_complete_pools'] == pipeline['complete_pools']
    assert result['actual_total_calls_with_spec_and_two_keyed_passes'] == pipeline['initial_calls']+pipeline['retry_calls']
    assert result['actual_logical_tokens'] == pipeline['logical_tokens'] and close(result['offline_batch_wall_seconds'],pipeline['offline_batch_wall_seconds'])
    assert result['matched_selective_and_full_pointwise'] == matched and result['changed_selective_vs_full_pointwise'] == changed
    assert analysis['keyed_replay_changed_candidates'] == sum(sum(x!=y for x,y in zip(pred['original'][i],pred['replay'][i])) for i in common)
    assert analysis['secondary_no_success_threshold'] is True and 'primary_upper_endpoint_below_zero' not in analysis
    print('Selective requests, mappings, output, actual call totals, prompt hashes, metrics and paired bootstrap independently verified.')
    print('Common queries:',len(common),'actual selective requests:',len(need),'matched later full-pointwise changes:',changed,'tokenizer context audit:',bool(tok))


if __name__ == '__main__':
    main()
