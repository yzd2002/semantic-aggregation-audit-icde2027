#!/usr/bin/env python3
"""Bounded CPU-only integrity and archive tests on explicit synthetic fixtures."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'public'))
from portable_common import compact, load, sha, write_json, write_lines
import replay_artifact
import replay_compact


def refresh(root):
    selected = load(root/'COMPACT_SELECTED_FILES.json')
    selected['files'] = {rel: sha(root/rel) for rel in selected['files']}
    write_json(root/'COMPACT_SELECTED_FILES.json', selected)
    inventory = load(root/'FILE_SHA256.json')
    inventory['files'] = {rel: sha(root/rel) for rel in inventory['files']}
    write_json(root/'FILE_SHA256.json', inventory)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Test output must stay inside portable_artifact')
    out.mkdir(parents=True, exist_ok=False)
    bundle = a.fixture_build/'bundle'
    if load(bundle/'COMPACT_SCOPE.json')['payload_kind'] != 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE':
        raise ValueError('These mutation tests require a synthetic fixture')
    checks = []
    original = load(bundle/'EXPECTED_MATHEMATICS.json')
    native = [r for r in original['policy_metrics'] if r['policy']=='S2' and r['contract']=='native']
    strict = [r for r in original['policy_metrics'] if r['policy']=='S2' and r['contract']=='strict']
    assert native and all(r['complete'] is True for r in native) and all(r['complete'] is False for r in strict)
    checks.append('native_output_completion_is_separate_from_strict_format')
    assert len(original['failed_attempts']) == 3
    assert sorted(r['observed_returned_requests'] for r in original['failed_attempts']) == [0,0,10]
    checks.append('three_startup_attempts_each_counted_once_controls_separate')
    assert any(r.get('cached_tokens_unknown_requests',0)>0 and r.get('cached_tokens') is None for r in original['costs'])
    checks.append('unknown_cached_usage_remains_null')
    for kind in ('wrong_CP_bound', 'unknown_cache_changed_to_zero'):
        root = out/kind
        shutil.copytree(bundle, root)
        if kind == 'wrong_CP_bound':
            result_path = next(root.glob('run_*/*/CURRENT_CP_RESULT.json'))
            result = load(result_path);result['calibration'][0]['lower'] = 0.0;write_json(result_path,result)
            marker = result_path.parent/'CASE_COMPLETE.json';data=load(marker)
            data['results_sha256']['CURRENT_CP'] = sha(result_path);write_json(marker,data)
        else:
            response_path = next(root.glob('run_*/responses.jsonl'))
            rows = [json.loads(line) for line in response_path.read_text().splitlines()]
            row = next(r for r in rows if r['cached_input_tokens'] is None)
            row['cached_input_tokens']=0;row['observed_uncached_input_tokens']=row['actual_input_tokens']
            response_path.write_text(''.join(compact(r)+'\n' for r in rows))
            marker = response_path.parent/'ALL_WORK_COMPLETE.json';data=load(marker)
            data['responses_sha256']=sha(response_path);write_json(marker,data)
        refresh(root)
        try:
            replay_compact.replay(root,out/(kind+'_replay'))
        except (ValueError, KeyError):
            checks.append(kind+'_rejected_after_all_outer_hashes_rebound')
        else:
            raise AssertionError('Semantic integrity mutation accepted: '+kind)
    for kind, member_name, member_type in [('traversal','../escape.txt',tarfile.REGTYPE),
            ('absolute','/tmp/forbidden_escape.txt',tarfile.REGTYPE),
            ('symlink','a',tarfile.SYMTYPE),('hardlink','a',tarfile.LNKTYPE)]:
        parts=out/(kind+'_parts');parts.mkdir()
        part=parts/'fixture.part0001'
        with tarfile.open(part,'w:xz') as stream:
            member=tarfile.TarInfo(member_name);member.type=member_type
            if member.isfile():member.size=1;stream.addfile(member,io.BytesIO(b'x'))
            else:member.linkname='/tmp';stream.addfile(member)
        entry={'filename':part.name,'bytes':part.stat().st_size,'sha256':sha(part)}
        module={'module':'fixture','archive_bytes':part.stat().st_size,'archive_sha256':sha(part),
                'uncompressed_bytes':1 if member_type==tarfile.REGTYPE else 0,
                'files':{member_name:hashlib.sha256(b'x').hexdigest()},'parts':[entry]}
        try:replay_artifact.extract({'modules':[module]},parts,out/(kind+'_extract'))
        except ValueError:checks.append('unsafe_'+kind+'_archive_rejected')
        else:raise AssertionError('Unsafe archive accepted: '+kind)
    try:replay_artifact.extract({'modules':[{'module':'../../unsafe'}]},out,out/'bad_module')
    except ValueError:checks.append('unsafe_module_join_path_rejected')
    else:raise AssertionError('Unsafe module name accepted')
    write_json(out/'CPU_GUARD_TESTS.json',{'status':'PASS','scientific_model_evidence':False,
        'scope':'Synthetic transport/mathematics/archive tests only; no GPU or live model calls', 'checks':checks})
    print('PASS CPU guard checks:',len(checks))


if __name__ == '__main__':main()
