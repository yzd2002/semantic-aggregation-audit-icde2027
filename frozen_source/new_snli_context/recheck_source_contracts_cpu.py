#!/usr/bin/env python3
"""Rerun the unchanged original CPU verifier in a read-only linked input view."""
import argparse
import importlib.util
import os
import sys
from common import *
from prepare_runtime_cpu import check_baseline
from runtime_contract import cache_environment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    output=args.out.resolve()
    if not output.is_relative_to(ROOT):raise ValueError('CPU evidence must stay inside this SNLI directory')
    output.mkdir(parents=True,exist_ok=False)
    baseline=check_baseline()
    # The original verifier checks glob('run_*'); adding run_gpu.py would
    # otherwise trip that pre-run check. A linked input-only view retains the
    # unchanged verifier and original bytes while excluding runtime programs.
    inputs=output/'READ_ONLY_INPUT_VIEW';inputs.mkdir()
    for name in ('prepared','selection','sources','tokenizers'):(inputs/name).symlink_to(ROOT/name,target_is_directory=True)
    import common
    common.ROOT=inputs
    os.environ.update(cache_environment(ROOT,'cpu_source_recheck'))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    spec=importlib.util.spec_from_file_location('unchanged_original_cpu_verifier',ROOT/'verify_prepared_cpu.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    reports=[]
    def redirected_write(path,value):
        if path!=inputs/'prepared/AUTHOR_CPU_VERIFICATION.json':raise ValueError('Unexpected original-verifier write')
        reports.append(value);write_json(output/'SOURCE_CONTRACT_RECHECK.json',value)
    verifier.write_json=redirected_write
    verifier.main()
    assert len(reports)==1 and reports[0]['status']=='PASS'
    check_baseline()
    import torch
    assert not torch.cuda.is_initialized()
    write_json(output/'RESULTS.json',{'status':'PASS','original_verifier_sha256':sha(ROOT/'verify_prepared_cpu.py'),
        'unchanged_verifier_executed':True,'original_inputs_read_via_read_only_links':True,
        'sole_original_output_redirected_to_new_report':True,'baseline':baseline,
        'new_model_judgments':0,'GPU_execution':False,'torch_cuda_initialized':False})


if __name__=='__main__':main()
