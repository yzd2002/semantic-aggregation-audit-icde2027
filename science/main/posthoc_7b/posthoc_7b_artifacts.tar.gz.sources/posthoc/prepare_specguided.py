#!/usr/bin/env python3
"""Create the pre-run schema-guided derivatives without changing prior runs."""
from pathlib import Path
w=Path('/dev/shm/intentsat-7b-followup-20261031')
def rep(s,old,new,n=1):
 if s.count(old)!=n:raise ValueError(f'Expected {n} occurrences, got {s.count(old)}: {old[:100]}')
 return s.replace(old,new)
def save(name,s):
 with (w/name).open('x',encoding='utf8') as f:f.write(s)
# ESCI: the old selection protocol remains in its metadata; posthoc field identifies this run.
s=(w/'run_esci_7b_posthoc.py').read_text()
s=rep(s,'from run_offline_wands_replay import assert_own_gpu, gpu_processes','from run_offline_wands_replay import assert_own_gpu, gpu_processes\nfrom spec_guidance_schema import SPEC_SCHEMA')
s=rep(s,"'posthoc_protocol': 'PROTOCOL_esci_7b_posthoc.md'", "'posthoc_protocol': 'PROTOCOL_7b_specguided_posthoc.md'")
s=rep(s,"spec_params = SamplingParams(temperature=0, max_tokens=160)","spec_params = SamplingParams(temperature=0, max_tokens=160, guided_decoding=GuidedDecodingParams(json=SPEC_SCHEMA))")
s=rep(s,"    sanity = {'unconstrained': unconstrained, 'guided_keyed': guided_json, 'guided_point': guided_one,", "    guided_spec = llm.generate([control], spec_params, use_tqdm=False)[0].outputs[0].text.strip()\n    sanity = {'unconstrained': unconstrained, 'guided_keyed': guided_json, 'guided_point': guided_one,\n              'guided_spec': guided_spec, 'unconstrained_valid_spec': base.parse_spec(unconstrained) is not None,\n              'guided_spec_valid': base.parse_spec(guided_spec) is not None,")
s=rep(s,"            not sanity['guided_keyed_valid'] or not sanity['guided_point_valid']):", "            not sanity['guided_keyed_valid'] or not sanity['guided_point_valid'] or\n            sanity['unconstrained_valid_spec'] or not sanity['guided_spec_valid']):")
save('run_esci_7b_specguided.py',s)
# WANDS: retain guarded batches while recording a distinct protocol.
s=(w/'run_wands_7b_posthoc.py').read_text()
s=rep(s,"PROTOCOL = 'PROTOCOL_7b_multidomain_posthoc.md'", "PROTOCOL = 'PROTOCOL_7b_specguided_posthoc.md'")
s=rep(s,'from run_offline_wands_replay import assert_own_gpu, gpu_processes','from run_offline_wands_replay import assert_own_gpu, gpu_processes\nfrom spec_guidance_schema import SPEC_SCHEMA')
old="    control = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')"
new="    spec_params = SamplingParams(temperature=0, max_tokens=160, guided_decoding=GuidedDecodingParams(json=SPEC_SCHEMA))\n"+old
s=rep(s,old,new)
s=rep(s,"    sanity = {'unconstrained': ordinary, 'json_guided': json_control, 'point_guided': point_control,", "    spec_control = guarded_generate([control], spec_params, use_tqdm=False)[0].outputs[0].text.strip()\n    sanity = {'unconstrained': ordinary, 'json_guided': json_control, 'point_guided': point_control,\n              'guided_spec': spec_control, 'unconstrained_valid_spec': base.parse_spec(ordinary) is not None,\n              'guided_spec_valid': base.parse_spec(spec_control) is not None,")
s=rep(s,"            or not sanity['guided_json_valid'] or not sanity['guided_point_valid']):", "            or not sanity['guided_json_valid'] or not sanity['guided_point_valid']\n            or sanity['unconstrained_valid_spec'] or not sanity['guided_spec_valid']):")
# Do not replace the schema-guided construction itself.
old='SamplingParams(temperature=0, max_tokens=160), use_tqdm=False)'
if s.count(old)!=2:raise ValueError('Expected two specification generation calls')
s=s.replace(old,'spec_params, use_tqdm=False)')
save('run_wands_7b_specguided.py',s)
for source,target in [('analyze_wands_7b_posthoc.py','analyze_wands_7b_specguided.py'),('verify_wands_7b_posthoc.py','verify_wands_7b_specguided.py'),('analyze_wands_7b_two_way.py','analyze_wands_7b_specguided_two_way.py'),('verify_wands_7b_two_way.py','verify_wands_7b_specguided_two_way.py')]:
 s=(w/source).read_text();s=rep(s,'PROTOCOL_7b_multidomain_posthoc.md','PROTOCOL_7b_specguided_posthoc.md');save(target,s)
print('Created schema-guided 7B runners and matching analysis/verifier derivatives.')
