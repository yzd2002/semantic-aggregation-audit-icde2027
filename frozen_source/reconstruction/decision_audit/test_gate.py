"""Fail-closed evaluation gate; immutable dev evidence, 36 decisions, independent review, root release."""
from common import *
from lock_policies import derive_lock

def verify_test_release(bundle):
 bundle=Path(bundle)
 for name in ('POLICY_LOCK.json','POLICY_LOCK_SHA256.json','DEV_INDEPENDENT_REVIEW.json','TEST_RELEASE.json'):
  if not (bundle/name).is_file():raise RuntimeError('Evaluation gate missing '+name)
 lock=json.loads((bundle/'POLICY_LOCK.json').read_text());recorded=json.loads((bundle/'POLICY_LOCK_SHA256.json').read_text())
 actual_sha=sha(bundle/'POLICY_LOCK.json')
 if recorded.get('POLICY_LOCK.json')!=actual_sha or recorded.get('policy_count')!=36:raise RuntimeError('Decision-lock hash/count mismatch')
 payload=lock['payload']
 if payload.get('policy_count')!=36 or len(payload.get('policies',[]))!=36:raise RuntimeError('All36 policies are required')
 if digest(compact(payload))!=recorded.get('payload_sha256'):raise RuntimeError('Decision payload hash mismatch')
 if payload!=derive_lock(bundle):raise RuntimeError('Decision lock differs from frozen selector and current immutable development evidence')
 review=json.loads((bundle/'DEV_INDEPENDENT_REVIEW.json').read_text());release=json.loads((bundle/'TEST_RELEASE.json').read_text())
 if review.get('status')!='PASS' or review.get('policy_lock_sha256')!=actual_sha or review.get('verified_policy_count')!=36:
  raise RuntimeError('Independent development review is missing, stale or incomplete')
 if release.get('status')!='approved' or release.get('policy_lock_sha256')!=actual_sha or release.get('independent_review_sha256')!=sha(bundle/'DEV_INDEPENDENT_REVIEW.json'):
  raise RuntimeError('Root evaluation release does not bind the reviewed decision lock')
 if sorted(release.get('allowed_models',[]))!=sorted(MODELS):raise RuntimeError('Release does not cover the two frozen models')
 return {'policy_lock_sha256':actual_sha,'independent_review_sha256':sha(bundle/'DEV_INDEPENDENT_REVIEW.json'),
  'test_release_sha256':sha(bundle/'TEST_RELEASE.json'),'verified_policy_count':36}
