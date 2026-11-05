#!/usr/bin/env python3
"""Check the independent supervised feature reconstruction on edge cases."""
import copy
from run_supervised_baseline import features as original
from verify_esci_selective_supervised_reference import features as independent

cases = [('', ''), ('Red chair', 'Red red armchair'), ('蓝 色 № 7', '蓝 色 № 7 家具'),
         ('aa-aa bb', 'AA aa cc'), ('aaa', 'aaa'), ('a_b 123', 'a_b 123 123')]
for query, text in cases:
    assert original(query, text) == independent(query, text)
record = {'query': 'blue chair', 'text': 'blue chair', 'gold': 'Y', 'grade': 'E'}
changed = copy.deepcopy(record); changed['gold'] = 'N'; changed['grade'] = 'NOT-IN-INPUT'
assert independent(record['query'], record['text']) == independent(changed['query'], changed['text'])
print('Independent feature reconstruction and label isolation passed on synthetic edge cases.')
