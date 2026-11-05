#!/usr/bin/env python3
"""Create a separate reproducibility ZIP whose TREC sampled texts are reconstructed locally."""
import argparse
import copy
import hashlib
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

OMIT = {'reproducibility/trec_covid_transfer_selection/inputs.jsonl',
        'reproducibility/trec_covid_transfer_selection/selection.json'}
NOTICE_NAME = 'reproducibility/NOTICE_RECONSTRUCT_TREC_SAMPLE.md'
NOTICE = '''# Reconstruct the omitted TREC-COVID sample

This separately created archive omits the two saved TREC-COVID selection files,
including sampled CORD-19-derived article titles and abstracts. It keeps NIST
topics and qrels, selection code, recorded responses, analysis and hashes. This
does not remove product text or other third-party material, assign licenses, or
establish that all remaining content may be publicly redistributed. Review all
contents, rights and notices before publishing anything.

In the extracted `reproducibility` directory, download the exact BEIR archive
from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
to a local path. Its expected MD5 is ce62140cb23feb9becf6270d0d1fe6d1 and its
SHA-256 is 120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0.
Then run, replacing the archive path:

```bash
python3 select_trec_covid_transfer_v2.py --topics trec_covid_sources/topics-rnd5.xml \\
  --qrels trec_covid_sources/qrels-covid_d5_j0.5-5.txt \\
  --archive /path/to/trec-covid.zip --out trec_covid_transfer_selection
sha256sum -c TREC_COVID_INFERENCE_FROZEN_SHA256.txt
sha256sum -c TREC_COVID_RESULTS_SHA256.txt
python3 verify_trec_covid_selection.py --topics trec_covid_sources/topics-rnd5.xml \\
  --qrels trec_covid_sources/qrels-covid_d5_j0.5-5.txt \\
  --archive /path/to/trec-covid.zip --selection trec_covid_transfer_selection
```

The selector refuses an archive whose MD5 differs. The frozen hashes also
verify that regenerated input bytes match those used in the experiments.
Use a new extraction; never overwrite an existing user's sample or directory.
See `README.md` and `trec_covid_sources/SOURCES.md` for further verification,
runtime limitations and underlying article-rights details.
'''


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.digest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='full local reproducibility ZIP')
    ap.add_argument('--out', type=Path, required=True, help='new ZIP, never overwritten')
    a = ap.parse_args()
    if a.source.resolve() == a.out.resolve() or a.out.exists():
        raise FileExistsError('Choose a new output path distinct from the source archive.')
    with ZipFile(a.source) as old:
        names = old.namelist()
        if len(names) != len(set(names)) or not OMIT <= set(names) or NOTICE_NAME in names:
            raise ValueError('Unexpected source archive content')
        for item in old.infolist():
            p = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'reproducibility':
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(mode):
                raise ValueError('Symlinks are not supported')
        if old.testzip() is not None:
            raise ValueError('Source ZIP failed CRC check')
        with a.out.open('xb') as destination:
            with ZipFile(destination, 'w', compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as new:
                for item in old.infolist():
                    if item.filename in OMIT:
                        continue
                    with old.open(item) as src, new.open(copy.copy(item), 'w') as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                new.writestr(NOTICE_NAME, NOTICE)
        with ZipFile(a.out) as new:
            if new.testzip() is not None or set(new.namelist()) != ((set(names) - OMIT) | {NOTICE_NAME}):
                raise ValueError('Reduced ZIP failed integrity check')
            for name in set(names) - OMIT:
                with old.open(name) as src, new.open(name) as dst:
                    if digest(src) != digest(dst):
                        raise ValueError('Copied file differs: ' + name)
    with a.out.open('rb') as f:
        result = digest(f).hex()
    print('Created separate ZIP with only the two selection files omitted; SHA-256:', result)
    print('Reconstruction and remaining rights-review instructions are inside the archive.')


if __name__ == '__main__':
    main()
