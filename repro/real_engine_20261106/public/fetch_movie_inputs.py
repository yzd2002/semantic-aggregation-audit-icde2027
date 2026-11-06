#!/usr/bin/env python3
"""Acquire the exact Kaggle-v4 movie source anonymously into a new local tree.

Retains the host-declared CC0 notice as hosted, without assigning additional
rights. Uses the unchanged pinned fetch/preparation scripts with explicit paths.
"""
import argparse
import json
from pathlib import Path
import shutil
from portable_common import compact, digest, import_file, lines, load, sha, write_json
from fetch_agnews import PLACEHOLDER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--from-zip', type=Path, help='Use already acquired exact version4 bytes; no network')
    a = p.parse_args()
    root, out = a.bundle.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fetch = import_file('unchanged_movie_fetch', root/'real_engine/adapters/fetch_movie_v4.py')
    fetch.ROOT = out/'real_engine'
    fetch.ARCHIVE = fetch.ROOT/'sources/movie-kaggle-v4.zip'
    fetch.SOURCE_DIR = fetch.ROOT/'sources/movie_v4'
    if a.from_zip:
        if sha(a.from_zip) != fetch.EXPECTED:
            raise ValueError('Provided source ZIP is not the exact pinned version4')
        fetch.ARCHIVE.parent.mkdir(parents=True)
        shutil.copyfile(a.from_zip, fetch.ARCHIVE)
    fetch.main()
    prepare = import_file('unchanged_movie_prepare', root/'real_engine/adapters/prepare_movie.py')
    prepare.ROOT = fetch.ROOT
    prepare.SOURCE = fetch.SOURCE_DIR/'rotten_tomatoes_movie_reviews.csv'
    prepare.OUT = fetch.ROOT/'prepared/movie'
    prepare.main()
    plan = load(root/'MOVIE_FETCH_PLAN.json')
    for template in plan['native_request_templates']:
        target = out/template['relative_output']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream:
            for record in lines(out/template['relative_records']):
                row = json.loads(json.dumps(template['row_template'], ensure_ascii=False))
                row['record_id'] = record['record_id']
                for message in row['messages']:
                    message['content'] = message['content'].replace(PLACEHOLDER, record['fields'][template['text_field']])
                row['messages_sha256'] = digest(json.dumps(row['messages'], ensure_ascii=False, sort_keys=True, separators=(',', ':')))
                stream.write((compact(row) if template['serialization'] == 'compact' else json.dumps(row, ensure_ascii=False))+'\n')
    study = load(root/'prepared/STUDY.json')
    expected = {}
    old_root = Path(load(root/'COMPACT_SCOPE.json')['original_root'])
    for workload in study['workloads']:
        if workload.get('domain') != 'movie':
            continue
        for field in ('records', 'gold', 'metadata'):
            if field in workload:
                rel = Path(workload[field]).relative_to(old_root)
                file = out/rel
                if sha(file) != workload[field+'_sha256']:
                    raise ValueError('Reconstructed complete movie view differs: '+str(rel))
                expected[str(rel)] = sha(file)
    for rel, expected_hash in plan['expected_reconstruction_sha256'].items():
        if sha(out/rel) != expected_hash:
            raise ValueError('Reconstructed full movie input/formatter bytes differ: '+rel)
        expected[rel] = expected_hash
    write_json(out/'EXACT_MOVIE_RECONSTRUCTION.json', {'status': 'PASS', 'source_zip_sha256': fetch.EXPECTED,
        'exact_view_files': expected, 'license_as_hosted': 'CC0: Public Domain',
        'additional_rights_warranty': False, 'new_model_inference': False,
        'source_and_preparation_code_modified': False, 'explicit_path_mapping': True})
    print('PASS: exact version4 complete-text movie views and native requests reconstructed locally; no model inference.')


if __name__ == '__main__':
    main()
