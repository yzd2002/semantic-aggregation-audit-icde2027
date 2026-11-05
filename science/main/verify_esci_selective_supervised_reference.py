#!/usr/bin/env python3
"""Independently re-create training and check saved supervised predictions."""
import argparse
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FIELDS = ('product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color')
EXAMPLES_SHA = '4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'
PRODUCTS_SHA = '25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f]


def features(query, text):
    q, p = set(re.findall(r'\w+', query.casefold())), set(re.findall(r'\w+', text.casefold()))
    shared = q.intersection(p)
    d = {prefix + word: 1. for prefix, values in
         (('q:', q), ('p:', p), ('shared:', shared), ('missing:', q-p)) for word in values}
    d.update({'numeric:query_coverage': len(shared)/len(q) if q else 0.,
              'numeric:product_coverage': len(shared)/len(p) if p else 0.,
              'numeric:jaccard': len(shared)/len(q | p) if q | p else 0.,
              'numeric:log_query_count': math.log1p(len(q)), 'numeric:log_product_count': math.log1p(len(p)),
              'numeric:constant': 1.})
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--products', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--refit', action='store_true')
    a = p.parse_args()
    if digest(a.examples) != EXAMPLES_SHA or digest(a.products) != PRODUCTS_SHA:
        raise ValueError('Pinned input source differs.')
    meta = json.loads((a.reference/'summary.json').read_text())
    sample = read(a.selection/'inputs.jsonl')
    saved = read(a.reference/'predictions.jsonl')
    old = read(ROOT/'run_full_pools_250/inputs.jsonl')
    old_saved = read(ROOT/'run_supervised_baseline_1000/predictions.jsonl')
    manifest = read(ROOT/'run_supervised_baseline_1000/train_manifest.jsonl')
    if (meta['protocol'] != 'PROTOCOL_esci_selective_supervised_reference.md' or
            meta['selection_inputs_sha256'] != digest(a.selection/'inputs.jsonl') or
            meta['training_manifest_sha256'] != digest(ROOT/'run_supervised_baseline_1000/train_manifest.jsonl') or
            meta['old_predictions_sha256'] != digest(ROOT/'run_supervised_baseline_1000/predictions.jsonl')):
        raise ValueError('Reference or input hash differs.')
    train_queries = {r['query'].strip().casefold() for r in manifest}
    train_products = {r['product_id'] for r in manifest}
    if ({r['query'].strip().casefold() for r in sample} & train_queries or
            {x['product_id'] for r in sample for x in r['items']} & train_products):
        raise ValueError('New sample overlaps training queries or product IDs.')
    prior = {r['query'].strip().casefold() for name in ('run_test_160', 'run_validation_500', 'run_full_pools_250')
             for r in read(ROOT/name/'inputs.jsonl')}
    table = pd.read_parquet(a.examples, columns=['example_id','query','query_id','product_id','product_locale',
        'esci_label','small_version','split'],
        filters=[('product_locale','=','us'),('small_version','=',1),('split','=','train')])
    table = table[table['esci_label'].isin(['E','S','C','I']) & table['query'].notna()].copy()
    normalized = table['query'].str.strip().str.casefold()
    table = table[normalized.ne('') & ~normalized.isin(prior)].sort_values('example_id').sample(n=50000,random_state=20271106)
    rebuilt = [{'example_id': str(x.example_id), 'query_id': str(x.query_id), 'query': x.query,
                'product_id': str(x.product_id), 'grade': x.esci_label} for x in table.itertuples(index=False)]
    if rebuilt != manifest:
        raise ValueError('Independently reconstructed training sample differs.')
    coef = np.load(a.reference/'coefficient.npy', allow_pickle=False)
    intercept = np.load(a.reference/'intercept.npy', allow_pickle=False)
    if coef.shape != (1, 2**18) or intercept.shape != (1,) or not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError('Unexpected saved fitted coefficient shapes or values.')
    hasher = FeatureHasher(n_features=2**18, input_type='dict', alternate_sign=False, dtype=np.float32)

    def check_predictions(pools, predictions):
        if len(pools) != len(predictions):
            raise ValueError('Prediction count differs.')
        x = hasher.transform(features(row['query'], item['text']) for row in pools for item in row['items'])
        with threadpool_limits(limits=2):
            probability = expit(np.asarray(x @ coef[0]).reshape(-1) + intercept[0])
        offset, maximum = 0, 0.
        for row, saved_row in zip(pools, predictions):
            n = len(row['items'])
            if row['query_id'] != saved_row['query_id'] or len(saved_row['pred']) != n or len(saved_row['positive_probability']) != n:
                raise ValueError('Saved reference item alignment differs.')
            values = probability[offset:offset+n]; offset += n
            for value, label, saved_value in zip(values, saved_row['pred'], saved_row['positive_probability']):
                if label != ('Y' if value >= .5 else 'N'):
                    raise ValueError('Thresholded reference label differs.')
                maximum = max(maximum, abs(float(value)-saved_value))
        if offset != len(probability) or maximum > 1e-6:
            raise ValueError('Saved reference probabilities differ.')
        return maximum

    old_max = check_predictions(old, old_saved)
    new_max = check_predictions(sample, saved)
    if a.refit:
        product_ids = sorted(set(str(x) for x in table['product_id']))
        product_rows = ds.dataset(a.products, format='parquet').to_table(columns=['product_id','product_locale',*FIELDS],
            filter=(ds.field('product_locale')=='us') & ds.field('product_id').isin(product_ids)).to_pandas()
        if product_rows.duplicated(['product_id','product_locale']).any():
            raise ValueError('Duplicate training product key.')
        data = table.merge(product_rows, on=['product_id','product_locale'], how='left', indicator=True, validate='many_to_one')
        if not (data['_merge']=='both').all():
            raise ValueError('Training product join incomplete.')
        def documents():
            for x in data.itertuples(index=False):
                text = ' '.join(getattr(x, k) for k in FIELDS if isinstance(getattr(x, k), str) and getattr(x, k))[:600]
                yield features(x.query, text)
        matrix = hasher.transform(documents())
        labels = np.array([int(x in ('E','S')) for x in data['esci_label']])
        fit = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, penalty='l2', class_weight=None, random_state=20271106)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            with threadpool_limits(limits=2):
                fit.fit(matrix, labels)
        warn = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if (warn != meta['convergence_warning'] or int(fit.n_iter_[0]) != meta['solver_iterations'] or
                np.max(np.abs(fit.coef_-coef)) > 1e-6 or np.max(np.abs(fit.intercept_-intercept)) > 1e-6):
            raise ValueError('Independent refit differs from saved coefficients.')
    print('Training selection and new-sample isolation independently reconstructed; predictions verified.')
    print('Maximum prior/new probability differences:', old_max, new_max, 'independent refit:', a.refit)


if __name__ == '__main__':
    main()
