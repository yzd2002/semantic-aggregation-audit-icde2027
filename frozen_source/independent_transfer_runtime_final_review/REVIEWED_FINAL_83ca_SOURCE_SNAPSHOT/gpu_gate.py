"""Explicit fresh root release binds review, budget, selected GPU and the new freeze."""
from common import *
from freeze import check

def verify(model,gpu):
    check();review_path=ROOT/'verification/INDEPENDENT_REVIEW.json';release_path=ROOT/'GPU_RELEASE.json'
    if not review_path.is_file():raise RuntimeError('Independent CPU review is required before root GPU release')
    review=json.loads(review_path.read_text());frozen=sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json')
    assert review.get('status')=='PASS' and review.get('freeze_sha256')==frozen and review.get('verified_policy_count')==36
    if not release_path.is_file():raise RuntimeError('No fresh root GPU GO; inference is prohibited')
    release=json.loads(release_path.read_text());budget=json.loads((ROOT/'prepared/BUDGET.json').read_text())
    assert release.get('status')=='approved' and release.get('root_authorized') is True
    assert release.get('freeze_sha256')==frozen and release.get('policy_lock_sha256')==sha(ROOT/'POLICY_LOCK.json')
    assert release.get('independent_review_sha256')==sha(review_path)
    assert model in release.get('allowed_models',[]) and str(release.get('gpu_by_model',{}).get(model))==str(gpu)
    assert release.get('science_calls')==budget['science_calls'] and release.get('control_calls')==budget['control_calls']
    return release
