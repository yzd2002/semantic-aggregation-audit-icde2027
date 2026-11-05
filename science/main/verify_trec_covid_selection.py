#!/usr/bin/env python3
"""Independently reconstruct TREC-COVID selection from NIST and BEIR sources."""
import argparse
import collections
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def digest(p, name='sha256'):
    h = hashlib.new(name)
    with p.open('rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''): h.update(x)
    return h.hexdigest()


def norm(x):
    return re.sub(r'\s+', ' ', x).strip() if isinstance(x, str) else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topics', type=Path, required=True)
    ap.add_argument('--qrels', type=Path, required=True)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    info = json.loads((a.selection / 'selection.json').read_text())
    rows = [json.loads(x) for x in (a.selection / 'inputs.jsonl').read_text().splitlines()]
    assert info['source_sha256'] == {'topics': digest(a.topics), 'qrels': digest(a.qrels),
                                     'beir_archive': digest(a.archive)}
    assert digest(a.archive, 'md5') == info['beir_archive_md5'] == 'ce62140cb23feb9becf6270d0d1fe6d1'
    assert digest(a.selection / 'inputs.jsonl') == info['inputs_sha256']
    topics = {int(x.attrib['number']): {k: norm(x.findtext(k)) for k in ('query', 'question', 'narrative')}
              for x in ET.parse(a.topics).getroot().findall('topic')}
    assert set(topics) == set(range(1, 51))
    qrels = collections.defaultdict(dict)
    unknown = 0
    duplicate = 0
    for line in a.qrels.read_text().splitlines():
        t, _, doc, grade = line.split(); tid, grade = int(t), int(grade)
        if grade == -1:
            unknown += 1
            continue
        assert tid in topics and grade in (0, 1, 2)
        if doc in qrels[tid]:
            duplicate += 1
            assert qrels[tid][doc] == grade
        qrels[tid][doc] = grade
    assert unknown == info['nonstandard_grade_minus_one_excluded']
    assert duplicate == info['duplicate_identical_pairs']
    wanted = {doc for d in qrels.values() for doc in d}
    documents, all_ids = {}, set()
    with zipfile.ZipFile(a.archive) as z:
        name, = [x for x in z.namelist() if x.endswith('/corpus.jsonl')]
        with z.open(name) as f:
            for line in f:
                x = json.loads(line); doc = str(x['_id'])
                assert doc not in all_ids
                all_ids.add(doc)
                if doc in wanted:
                    documents[doc] = (norm(x.get('title')), norm(x.get('text')))
    assert len(all_ids) == info['corpus_document_count']
    assert sum(len(d) for d in qrels.values()) == info['qrel_entries']
    expected, eligibility = [], {}
    for tid in sorted(topics):
        eligible, c = [], collections.Counter()
        for doc in sorted(qrels[tid]):
            if doc not in documents: c['missing_corpus'] += 1
            elif not documents[doc][0]: c['missing_title'] += 1
            elif not documents[doc][1]: c['missing_abstract'] += 1
            else: eligible.append(doc)
        c['eligible'] = len(eligible)
        eligibility[str(tid)] = dict(c)
        if len(eligible) < 32: continue
        docs = random.Random(20271031 + tid).sample(eligible, 32)
        items = []
        for doc in docs:
            title, abstract = documents[doc]; g = qrels[tid][doc]
            items.append({'doc_id': doc, 'title': title[:200], 'abstract': abstract[:800],
                          'title_characters': len(title), 'abstract_characters': len(abstract),
                          'grade': g, 'gold': 'Y' if g == 2 else 'N'})
        expected.append({'topic_id': tid, **topics[tid], 'items': items})
    assert eligibility == info['eligibility']
    assert expected == rows
    grades = collections.Counter(str(i['grade']) for r in rows for i in r['items'])
    doccounts = collections.Counter(i['doc_id'] for r in rows for i in r['items'])
    assert dict(grades) == info['sampled_grade_counts']
    assert len(rows) == info['included_topics'] and sum(doccounts.values()) == info['sampled_candidates']
    assert len(doccounts) == info['distinct_sampled_documents']
    assert sum(x > 1 for x in doccounts.values()) == info['documents_repeated_across_topics']
    assert sum(i['title_characters'] > 200 for r in rows for i in r['items']) == info['title_truncated']
    assert sum(i['abstract_characters'] > 800 for r in rows for i in r['items']) == info['abstract_truncated']
    print('Independently reconstructed all topic fields, eligibility, seeded selections, labels and truncated texts.')
    print('Topics:', len(rows), 'judged candidates:', sum(doccounts.values()), 'unique documents:', len(doccounts))


if __name__ == '__main__':
    main()
