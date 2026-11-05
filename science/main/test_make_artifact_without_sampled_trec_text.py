#!/usr/bin/env python3
"""Test archive omission, preservation and path/overwrite guards on fabricated content."""
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from make_artifact_without_sampled_trec_text import NOTICE_NAME, OMIT


def main():
    tmp = Path(tempfile.mkdtemp(prefix='trec-archive-test-', dir='/dev/shm'))
    full, reduced = tmp / 'full.zip', tmp / 'reduced.zip'
    unchanged = 'reproducibility/other/data.txt'
    with ZipFile(full, 'w', ZIP_DEFLATED) as z:
        for name in OMIT: z.writestr(name, 'sample content')
        z.writestr(unchanged, 'preserved exactly')
    script = Path(__file__).with_name('make_artifact_without_sampled_trec_text.py')
    args = [sys.executable, str(script), '--source', str(full), '--out', str(reduced)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    with ZipFile(reduced) as z:
        assert set(z.namelist()) == {unchanged, NOTICE_NAME}
        assert z.read(unchanged) == b'preserved exactly'
        assert b'select_trec_covid_transfer_v2.py' in z.read(NOTICE_NAME)
    assert subprocess.run(args, capture_output=True).returncode != 0
    unsafe = tmp / 'unsafe.zip'
    with ZipFile(unsafe, 'w') as z:
        for name in OMIT: z.writestr(name, 'sample content')
        z.writestr('reproducibility/../outside.txt', 'unsafe')
    assert subprocess.run([sys.executable, str(script), '--source', str(unsafe),
                           '--out', str(tmp / 'unsafe-out.zip')], capture_output=True).returncode != 0
    print('Archive omission, byte preservation, reconstruction notice, overwrite and path guards passed.')


if __name__ == '__main__':
    main()
