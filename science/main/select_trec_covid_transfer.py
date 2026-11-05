#!/usr/bin/env python3
"""Select grade-independent judged-document pools for the fixed TREC-COVID protocol."""
import argparse
import collections
import hashlib
import json
import random
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

PROTOCOL = 'PROTOCOL_trec_covid_transfer.md'
EXPECTED_MD5 = 'ce62140cb23feb9becf6270d0d1fe6d1'


def digest(path, alg='sha256'):
    h = hashlib.new(alg)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm(value):
    return ' '.join(value.split()) if isinstance(value, str) else ''


def build(topics_file, qrels_file, archive):
    if digest(archive, 'md5') != EXPECTED_MD5:
        raise ValueError('BEIR archive does not match its published MD5')
    topics = {}
    for elem in ET.parse(topics_file).getroot().findall('topic'):
        n = int(elem.attrib['number'])
        if n in topics:
            raise ValueError('duplicate topic ID')
        t = {field: norm(elem.findtext(field)) for field in ('query', 'question', 'narrative')}
        if not all(t.values()):
            raise ValueError('missing topic field')
        topics[n] = t
    if set(topics) != set(range(1, 51)):
        raise ValueError('expected 50 final topic IDs')

    judged = collections.defaultdict(dict)
    duplicate_pairs = 0
    with qrels_file.open(encoding='utf8') as f:
        for line in f:
            tid, iteration, doc_id, grade = line.split()
            tid, grade = int(tid), int(grade)
            if tid not in topics or grade not in (0, 1, 2):
                raise ValueError('unexpected qrels topic or grade')
            previous = judged[tid].get(doc_id)
            if previous is not None:
                if previous != grade:
                    raise ValueError('conflicting duplicate judgment')
                duplicate_pairs += 1
            judged[tid][doc_id] = grade
    targets = {doc for row in judged.values() for doc in row}
    corpus, all_ids = {}, set()
    with zipfile.ZipFile(archive) as z:
        names = [name for name in z.namelist() if name.endswith('/corpus.jsonl')]
        if len(names) != 1:
            raise ValueError('expected one corpus.jsonl')
        with z.open(names[0]) as src:
            for line in src:
                doc = json.loads(line)
                doc_id = str(doc['_id'])
                if doc_id in all_ids:
                    raise ValueError('duplicate corpus document ID')
                all_ids.add(doc_id)
                if doc_id in targets:
                    corpus[doc_id] = (norm(doc.get('title')), norm(doc.get('text')))

    result, counts = [], {}
    for tid in sorted(topics):
        categories = collections.Counter()
        eligible = []
        for doc_id in sorted(judged[tid]):
            if doc_id not in corpus:
                categories['missing_corpus'] += 1
            elif not corpus[doc_id][0]:
                categories['missing_title'] += 1
            elif not corpus[doc_id][1]:
                categories['missing_abstract'] += 1
            else:
                eligible.append(doc_id)
        categories['eligible'] = len(eligible)
        counts[str(tid)] = dict(categories)
        if len(eligible) < 32:
            continue
        chosen = random.Random(20271031 + tid).sample(eligible, 32)
        items = []
        for doc_id in chosen:
            title, abstract = corpus[doc_id]
            grade = judged[tid][doc_id]
            items.append({'doc_id': doc_id, 'title': title[:200], 'abstract': abstract[:800],
                          'title_characters': len(title), 'abstract_characters': len(abstract),
                          'grade': grade, 'gold': 'Y' if grade == 2 else 'N'})
        result.append({'topic_id': tid, **topics[tid], 'items': items})
    repeated = collections.Counter(item['doc_id'] for row in result for item in row['items'])
    grades = collections.Counter(str(item['grade']) for row in result for item in row['items'])
    info = {'protocol': PROTOCOL, 'source_sha256': {'topics': digest(topics_file), 'qrels': digest(qrels_file),
                'beir_archive': digest(archive)}, 'beir_archive_md5': EXPECTED_MD5,
            'total_topics': len(topics), 'included_topics': len(result),
            'omitted_topics': [tid for tid in sorted(topics) if len([r for r in result if r['topic_id'] == tid]) == 0],
            'qrel_entries': sum(len(v) for v in judged.values()), 'duplicate_identical_pairs': duplicate_pairs,
            'corpus_document_count': len(all_ids), 'eligibility': counts, 'sampled_grade_counts': dict(grades),
            'sampled_candidates': sum(len(r['items']) for r in result), 'distinct_sampled_documents': len(repeated),
            'documents_repeated_across_topics': sum(v > 1 for v in repeated.values()),
            'title_truncated': sum(i['title_characters'] > 200 for r in result for i in r['items']),
            'abstract_truncated': sum(i['abstract_characters'] > 800 for r in result for i in r['items'])}
    return result, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topics', type=Path, required=True)
    ap.add_argument('--qrels', type=Path, required=True)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows, info = build(a.topics, a.qrels, a.archive)
    a.out.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    (a.out / 'inputs.jsonl').write_text(payload, encoding='utf8')
    info['inputs_sha256'] = digest(a.out / 'inputs.jsonl')
    (a.out / 'selection.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('Selected topics:', info['included_topics'], 'judged candidates:', info['sampled_candidates'])
    print('Sample grade counts:', info['sampled_grade_counts'])
    print('Distinct documents:', info['distinct_sampled_documents'], 'repeated across topics:',
          info['documents_repeated_across_topics'])


if __name__ == '__main__':
    main()
