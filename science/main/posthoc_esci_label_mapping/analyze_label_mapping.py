#!/usr/bin/env python3
"""Score frozen ESCI predictions under original and alternate label mappings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

SEED = 20271031
DRAWS = 4000
NAMES = ('qwen', 'smollm')
MODES = ('original', 'selective', 'pointwise', 'supervised', 'all_y', 'all_n')
SOURCE_FILES = ['esci_selective_holdout_selection/inputs.jsonl',
                'run_esci_selective_supervised_reference/predictions.jsonl'] + [
    f'run_esci_selective_holdout_{model}/{file}' for model in NAMES
    for file in ('original_responses.jsonl', 'selective_outputs.jsonl', 'pointwise_responses.jsonl')]


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def keyed(text, size):
    pattern = r'\{' + ','.join(r'"' + str(i) + r'":"[YN]"' for i in range(1, size + 1)) + r'\}'
    if re.fullmatch(pattern, text) is None:
        raise ValueError('Invalid keyed output')
    obj = json.loads(text)
    return [obj[str(i)] for i in range(1, size + 1)]


def load_vectors(rows, root, model):
    base = root / f'run_esci_selective_holdout_{model}'
    orig, selected, point = (lines(base / name) for name in
                             ('original_responses.jsonl', 'selective_outputs.jsonl', 'pointwise_responses.jsonl'))
    expected = [(r['query_id'], start, min(8, len(r['items'])-start)) for r in rows
                for start in range(0, len(r['items']), 8)]
    if [(x['query_id'], x['start'], x['size']) for x in orig] != expected:
        raise ValueError('Original chunk order differs from selected pools')
    if [(x['query_id'], x['start'], x['size']) for x in point] != [
        (r['query_id'], i, 1) for r in rows for i in range(len(r['items']))]:
        raise ValueError('Pointwise order differs from selected pools')
    if len(selected) != len(rows) or [x['query_id'] for x in selected] != [r['query_id'] for r in rows]:
        raise ValueError('Selective pool order differs')
    original, selective, pointwise = [], [], []
    pos = pt = 0
    for row, saved in zip(rows, selected):
        n = len(row['items']); labels = []
        for start in range(0, n, 8):
            labels.extend(keyed(orig[pos]['final'], min(8, n-start)))
            pos += 1
        one = [x['final'] for x in point[pt:pt+n]]; pt += n
        fallback = saved['pred']
        if not isinstance(fallback, list) or len(fallback) != n or any(x not in ('Y','N') for x in fallback):
            raise ValueError('Missing or malformed selective output')
        if any(x not in ('Y','N') for x in one):
            raise ValueError('Missing or malformed pointwise output')
        original.append(labels); selective.append(fallback); pointwise.append(one)
    return {'original': original, 'selective': selective, 'pointwise': pointwise}


def quality(rows, vectors, positive):
    if len(rows) != len(vectors) or not rows:
        raise ValueError('Invalid pool count')
    tp=fp=tn=fn=0; per_query=[]
    for row, pred in zip(rows, vectors):
        items = row['items']; n = len(items)
        if not n or len(pred) != n or any(x not in ('Y','N') for x in pred):
            raise ValueError('Invalid prediction vector')
        gold = [x['grade'] in positive for x in items]
        tp += sum(p == 'Y' and g for p,g in zip(pred,gold))
        fp += sum(p == 'Y' and not g for p,g in zip(pred,gold))
        tn += sum(p == 'N' and not g for p,g in zip(pred,gold))
        fn += sum(p == 'N' and g for p,g in zip(pred,gold))
        per_query.append(abs(pred.count('Y') - sum(gold))/n)
    if not (tp+fn and tn+fp):
        raise ValueError('Both classes are required for balanced accuracy')
    den=2*tp+fp+fn
    return {'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn,
            'f1': 2*tp/den if den else 0.0,
            'balanced_accuracy': (tp/(tp+fn)+tn/(tn+fp))/2,
            'ratio_mae':float(np.mean(per_query))}, np.array(per_query)


def run(root):
    rows = lines(root / SOURCE_FILES[0])
    if len(rows) != 250 or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Unexpected pool selection')
    for r in rows:
        for x in r['items']:
            if x['grade'] not in ('E','S','C','I') or x['gold'] != ('Y' if x['grade'] in ('E','S') else 'N'):
                raise ValueError('Grade or original mapping mismatch')
    supervised = lines(root / SOURCE_FILES[1])
    if [x['query_id'] for x in supervised] != [r['query_id'] for r in rows]:
        raise ValueError('Supervised pool order differs')
    common = {'supervised':[x['pred'] for x in supervised],
              'all_y':[['Y']*len(r['items']) for r in rows],
              'all_n':[['N']*len(r['items']) for r in rows]}
    draw = np.random.default_rng(SEED).integers(0, len(rows), size=(DRAWS,len(rows)))
    out = {'analysis':'post-hoc fixed-output mapping sensitivity', 'seed':SEED, 'resamples':DRAWS,
           'query_pools':len(rows), 'candidate_count':sum(len(r['items']) for r in rows),
           'grade_counts':{g:sum(x['grade']==g for r in rows for x in r['items']) for g in 'ESCI'},
           'input_sha256':{f:digest(root/f) for f in SOURCE_FILES}, 'models':{},
           'limitation':'E-only rescoring changes the target; prompts and supervised training remain E/S-positive. No new inference.'}
    for model in NAMES:
        pred={**load_vectors(rows,root,model),**common}
        result={}; errors={}
        for mapping, positive in [('E_or_S',{'E','S'}),('E_only',{'E'})]:
            result[mapping]={}; errors[mapping]={}
            for name in MODES:
                result[mapping][name], errors[mapping][name] = quality(rows,pred[name],positive)
        old=json.loads((root/f'analysis_esci_selective_holdout_{model}/summary.json').read_text())['quality_common']
        for this, prior in [('original','original'),('selective','actual_fallback'),('pointwise','pointwise'),('all_y','always_Y'),('all_n','always_N')]:
            for metric in ('tp','fp','tn','fn','f1','balanced_accuracy','ratio_mae'):
                if abs(result['E_or_S'][this][metric]-old[prior][metric]) > 1e-12:
                    raise ValueError(f'Original mapping differs from saved result: {model} {this} {metric}')
        effects={m:errors[m]['selective']-errors[m]['original'] for m in errors}
        effects['E_only_minus_E_or_S'] = effects['E_only']-effects['E_or_S']
        def interval(a):
            return {'mean':float(np.mean(a)), 'ci95':[float(x) for x in np.quantile(a[draw].mean(axis=1),[.025,.975])]}
        out['models'][model]={'quality':result,'selective_minus_keyed':{m:interval(a) for m,a in effects.items()},
                              'original_mapping_matches_archived_results':True}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=run(args.root)
    with args.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Analyzed',result['query_pools'],'fixed pools and',result['candidate_count'],'candidates for both mappings.')


if __name__=='__main__':main()
