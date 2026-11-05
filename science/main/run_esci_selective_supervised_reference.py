#!/usr/bin/env python3
"""Refit the previously frozen ESCI supervised reference and score new pools."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import run_supervised_baseline as base
from select_esci_selective_holdout import EXAMPLES_SHA, PRODUCTS_SHA, sha

HERE = Path(__file__).resolve().parent
PROTOCOL = 'PROTOCOL_esci_selective_supervised_reference.md'


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--products', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory.')
    versions = {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                'scikit_learn': sklearn.__version__}
    if versions != {'numpy': '2.2.6', 'pandas': '2.3.3', 'scipy': '1.16.2', 'scikit_learn': '1.7.2'}:
        raise ValueError('Use the recorded original training library versions.')
    if sha(a.examples) != EXAMPLES_SHA or sha(a.products) != PRODUCTS_SHA:
        raise ValueError('Pinned official source hash differs.')
    new = lines(a.selection/'inputs.jsonl')
    meta = json.loads((a.selection/'selection.json').read_text())
    if len(new) != 250 or sha(a.selection/'inputs.jsonl') != meta['inputs_sha256']:
        raise ValueError('New selection differs.')
    old = lines(HERE/'run_full_pools_250/inputs.jsonl')
    old_predictions = lines(HERE/'run_supervised_baseline_1000/predictions.jsonl')
    manifest = lines(HERE/'run_supervised_baseline_1000/train_manifest.jsonl')
    training_queries = {x['query'].strip().casefold() for x in manifest}
    training_products = {x['product_id'] for x in manifest}
    if ({x['query'].strip().casefold() for x in new} & training_queries or
            {x['product_id'] for r in new for x in r['items']} & training_products):
        raise ValueError('New sample overlaps the recorded training queries or product IDs.')
    prior = {r['query'].strip().casefold() for name in ('run_test_160', 'run_validation_500', 'run_full_pools_250')
             for r in lines(HERE/name/'inputs.jsonl')}
    training, eligible = base.train_data(a.examples, a.products, prior)
    rebuilt = [{'example_id': str(x.example_id), 'query_id': str(x.query_id), 'query': x.query,
                'product_id': str(x.product_id), 'grade': x.esci_label} for x in training.itertuples(index=False)]
    if rebuilt != manifest or len(rebuilt) != 50000:
        raise ValueError('Refitted model would use different training examples or order.')
    docs = [(x.query, ' '.join(getattr(x, k) for k in base.FIELDS
             if isinstance(getattr(x, k), str) and getattr(x, k))[:600]) for x in training.itertuples(index=False)]
    labels = np.asarray([int(x.esci_label in ('E', 'S')) for x in training.itertuples(index=False)])
    hasher = FeatureHasher(n_features=2**18, input_type='dict', alternate_sign=False, dtype=np.float32)
    matrix = hasher.transform(base.features(q, text) for q, text in docs)
    classifier = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, penalty='l2',
                                    class_weight=None, random_state=20271106)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        with threadpool_limits(limits=2):
            classifier.fit(matrix, labels)
    convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    def score(pools):
        features = (base.features(pool['query'], item['text']) for pool in pools for item in pool['items'])
        values = hasher.transform(features)
        with threadpool_limits(limits=2):
            probs = classifier.predict_proba(values)[:, 1]
        records, i = [], 0
        for pool in pools:
            n = len(pool['items'])
            p = probs[i:i+n]
            records.append({'query_id': pool['query_id'], 'pred': ['Y' if v >= .5 else 'N' for v in p],
                            'positive_probability': [float(v) for v in p]})
            i += n
        if i != len(probs):
            raise ValueError('Scoring alignment mismatch.')
        return records

    old_refit = score(old)
    if len(old_refit) != len(old_predictions):
        raise ValueError('Original prediction count mismatch.')
    maximum = 0.
    for x, y in zip(old_refit, old_predictions):
        if x['query_id'] != y['query_id'] or x['pred'] != y['pred'] or len(x['positive_probability']) != len(y['positive_probability']):
            raise ValueError('Refitted original predictions differ.')
        maximum = max(maximum, *(abs(v-w) for v, w in zip(x['positive_probability'], y['positive_probability'])))
    if maximum > 1e-6:
        raise ValueError('Refitted original probabilities differ by more than 1e-6.')
    new_predictions = score(new)
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out/'predictions.jsonl').open('x', encoding='utf8') as f:
        for row in new_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    np.save(a.out/'coefficient.npy', classifier.coef_, allow_pickle=False)
    np.save(a.out/'intercept.npy', classifier.intercept_, allow_pickle=False)
    info = {'protocol': PROTOCOL, 'selection_inputs_sha256': meta['inputs_sha256'],
            'training_manifest_sha256': sha(HERE/'run_supervised_baseline_1000/train_manifest.jsonl'),
            'old_predictions_sha256': sha(HERE/'run_supervised_baseline_1000/predictions.jsonl'),
            'training_examples': len(manifest), 'eligible_original_training_examples': eligible,
            'training_query_overlap': 0, 'training_product_overlap': 0,
            'maximum_refit_original_probability_difference': maximum, 'old_prediction_mismatches': 0,
            'solver_iterations': int(classifier.n_iter_[0]), 'convergence_warning': convergence,
            'threshold': .5, 'versions': versions}
    (a.out/'summary.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf8')
    print('Prior reference refitted; original predictions match; new pools scored without using their labels.')
    print('Maximum prior probability difference:', maximum, 'convergence warning:', convergence)


if __name__ == '__main__':
    main()
