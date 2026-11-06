#!/usr/bin/env python3
"""Collect already-completed CPU evidence and verify the final review inputs."""
import ast
import difflib
import subprocess
import sys
from common import *
from prepare_runtime_cpu import check_baseline


def main():
    baseline=check_baseline()
    assert not any((ROOT/f'run_{m}').exists() or (ROOT/f'GPU_GO_{m}.json').exists() for m in MODELS)
    for path in ROOT.glob('*.py'):ast.parse(path.read_text())
    for path in (ROOT/'runtime_override').glob('*.py'):ast.parse(path.read_text())
    for path in (ROOT/'global_primary').glob('*.py'):ast.parse(path.read_text())
    global_names=sorted(p.name for p in (ROOT/'global_primary').iterdir())
    assert len(global_names)==4
    for name in global_names:assert sha(ROOT/'global_primary'/name)==sha(ROOT.parent/'global_primary'/name)
    assert sha(ROOT/'generation_receipts.py')=='a58f21a1596759a7903fb23468aa8a72a3f827ad51295f513d04cfe14be6b31d'
    provenance=json.loads((ROOT/'verification/RUNTIME_SOURCE_PROVENANCE.json').read_text())
    for name,want in provenance['byte_identical_modules'].items():assert sha(ROOT/name)==want
    # Source-adaptation diffs must be exact and auditable, with no stale patch.
    for name,label,reference in [('run_gpu.py','RUNNER','glm_v3_run_gpu.py'),
            ('launch_gpu.py','LAUNCHER','glm_v3_launch_gpu.py'),
            ('check_generation_failure_retention.py','FAILURE_CHECK','glm_v3_check_generation_failure_retention.py')]:
        expected=''.join(difflib.unified_diff((ROOT/'source_reference'/reference).read_text().splitlines(True),
            (ROOT/name).read_text().splitlines(True),fromfile='source_reference/'+reference,tofile=name))
        assert (ROOT/'verification'/f'GLM_V3_TO_SNLI_{label}.patch').read_text()==expected
    checks={}
    evidence_paths=('verification/CPU_RUNTIME_PREPARATION.json','verification/NATIVE_IPC_CPU_CHECK.json',
        'verification/PRIMARY_FAMILY_CPU_CHECK.json','verification/SERIALIZATION_REUSE_PROVENANCE.json',
        'cpu_checks/runtime_source_recheck_final/RESULTS.json','cpu_checks/runtime_failure_retention_final/RESULTS.json',
        'cpu_checks/runtime_full_chain_final/RESULTS.json')
    for name in evidence_paths:
        record=json.loads((ROOT/name).read_text());assert record['status']=='PASS',name
        checks[name]={'status':'PASS','sha256':sha(ROOT/name)}
    failure=json.loads((ROOT/'cpu_checks/runtime_failure_retention_final/RESULTS.json').read_text())
    assert len(failure['scenarios'])==17 and failure['real_model_judgments']==0
    for case in failure['scenarios']:
        assert case['actual_runner_sha256']==sha(ROOT/'run_gpu.py')
        assert case['receipt_module_sha256']==sha(ROOT/'generation_receipts.py')
    full=json.loads((ROOT/'cpu_checks/runtime_full_chain_final/RESULTS.json').read_text())
    assert len(full['cases'])==4 and full['real_model_judgments']==0 and full['GPU_execution'] is False
    for case in full['cases']:
        assert case['actual_runner_sha256']==sha(ROOT/'run_gpu.py') and case['analysis_sha256']==sha(ROOT/'analyze_snli_runtime.py')
        assert case['science_calls']==2240 and case['diagnostic_calls']==20
    launch_guards={}
    for model in MODELS:
        cfg=json.loads((ROOT/'prepared'/f'runtime_configuration_{model}.json').read_text())
        assert cfg['native_process_environment']=={'VLLM_RPC_BASE_PATH':str(ROOT/'rpc'/model[0]),'VLLM_ALLOW_INSECURE_SERIALIZATION':'1'}
        assert cfg['seeds']==SEEDS and not cfg['cuda_initialized_after_cpu_checks']
        proc=subprocess.run([sys.executable,'-B',str(ROOT/'launch_gpu.py'),'--model',model,'--device','0'],capture_output=True,text=True)
        assert proc.returncode!=0 and 'Explicit root GPU_GO.json tied to the final freeze is required.' in proc.stderr
        assert not (ROOT/f'run_{model}').exists() and not (ROOT/f'GPU_GO_{model}.json').exists()
        launch_guards[model]={'blocked_without_final_freeze_and_root_GO':True,'returncode':proc.returncode}
    write_json(ROOT/'verification/AUTHOR_RUNTIME_CPU_CHECKS.json',{'status':'PASS','baseline':baseline,'evidence':checks,
        'launch_guards':launch_guards,'final_runner_source_sha256':sha(ROOT/'run_gpu.py'),
        'final_analysis_source_sha256':sha(ROOT/'analyze_snli_runtime.py'),'generation_receipts_sha256':sha(ROOT/'generation_receipts.py'),
        'global_primary_four_files_byte_identical':True,'runtime_source_diffs_exact':True,'python_syntax':'PASS',
        'original_scientific_inputs_and_statistics_unmodified':True,'real_model_judgments':0,'GPU_execution':False,
        'independent_SNLI_review_pending':True,'installed_packages_modified':False})
    print(compact({'status':'PASS','baseline_files_preserved':105,'failure_scenarios':17,
        'full_original_request_fake_chains':3,'fixed_COUNT_missing_target_case':1,'real_model_judgments':0}),flush=True)


if __name__=='__main__':main()
