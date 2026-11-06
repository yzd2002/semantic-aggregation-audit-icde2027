"""Fixed CP80/CP320 follow-up with native LOTUS Boolean semantics."""
from experiment_common import *
from run_study import native_call, batch_plan, owned, processes
from native_runtime import durable
import argparse
import os
import traceback


def execute_policy(backend, workload, job, out):
    from scipy.stats import beta
    n, budget, key = job["n"], job["budget"], job["sampling_key"]
    records = read_lines(workload["records"])[:n]
    requests = read_lines(workload["native_requests"])[:n]
    dest = out / job["job_id"]
    dest.mkdir(exist_ok=False)
    backend.reset_policy(job["job_id"], "NATIVE_CP")
    backend.stage = job["job_id"]+"/calibration"
    start = len(backend.responses)
    started = time.perf_counter()
    indices = probes(n, key, "native_budget_probes", budget)
    ref = native_call(backend, [records[i] for i in indices], workload["predicate"],
                      dest / "calibration_singletons",
                      expected_messages=[requests[i]["messages"] for i in indices])
    assert len(ref["rows"]) == budget
    selected, decisions = 1, []
    for k in (8, 4, 2):
        groups, positions = probe_batches(n, k, key, "native_budget_batches", indices)
        checked = []
        for offset in range(0, budget, 16):
            rows = batch_plan(backend, requests, groups[offset:offset+16],
                              dest / f"candidate_{k}_chunk_{offset}",
                              probe_positions=positions[offset:offset+16])
            for j, row in enumerate(rows, offset):
                reference = ref["rows"][j]
                row.update(reference_native_label=reference["native_label"],
                           reference_strict_label=reference["strict_label"],
                           reference_strict_valid=reference["strict_valid"],
                           agreement=bool(row["strict_valid"] and row["label"] == reference["native_label"]))
                checked.append(row)
            possible_x = sum(r["agreement"] for r in checked) + budget-len(checked)
            possible_lower = float(beta.ppf(.05/3, possible_x, budget-possible_x+1)) if possible_x else 0.
            if possible_lower < .95:
                break
        x = sum(r["agreement"] for r in checked)
        lower = float(beta.ppf(.05/3, x, budget-x+1)) if x else 0.
        passed = len(checked) == budget and lower >= .95
        decisions.append({"k": k, "budget": budget, "actually_checked": len(checked),
                          "agreements": x, "lower": lower, "passed": passed, "probes": checked})
        if passed:
            selected = k
            break
    calibration_usage = backend.usage(start)
    calibration_seconds = time.perf_counter()-started
    execution_start = len(backend.responses)
    backend.stage = job["job_id"]+"/execution"
    if selected == 1:
        result = native_call(backend, records, workload["predicate"], dest / "execution_singletons",
                             expected_messages=[r["messages"] for r in requests])
        assert len(result["rows"]) == n
        labels = [r["native_label"] for r in result["rows"]]
        payload = {"labels": labels, "native_complete": True, "native_count": result["native_count"],
                   "strict_complete": result["strict_count"] is not None,
                   "strict_count": result["strict_count"], "strict_valid_rows": result["strict_valid_rows"]}
    else:
        result = batch_plan(backend, requests, random_partition(n, selected, key), dest / "execution_partition")
        payload = {**result, "native_complete": result["strict_complete"], "native_count": result["strict_count"]}
    payload.update({"job": job, "selected_k": selected, "calibration": decisions,
                    "calibration_usage": calibration_usage, "calibration_seconds": calibration_seconds,
                    "execution_usage": backend.usage(execution_start), "usage": backend.usage(start),
                    "elapsed_seconds": time.perf_counter()-started,
                    "native_ref_format_valid_rows": ref["strict_valid_rows"],
                    "native_ref_rows": budget,
                    "protected_target": "native parsed Boolean random-batch trial law, not source gold or a future realized reference"})
    write_json(dest / "POLICY_RESULT.json", payload)
    write_json(dest / "POLICY_COMPLETE.json", {"job": job, "result_sha256": sha(dest / "POLICY_RESULT.json"),
                                               "time_unix": time.time()})
    print(compact({"job": job["job_id"], "selected_k": selected,
                   "calls": payload["usage"]["requests"], "native_complete": payload["native_complete"],
                   "seconds": payload["elapsed_seconds"]}), flush=True)


def main(model, gpu):
    freeze = json.loads((ROOT / "BUDGET_FOLLOWUP_FREEZE.json").read_text())
    for rel, wanted in freeze["files"].items():
        assert sha(ROOT / rel) == wanted, rel
    config = json.loads((ROOT / "BUDGET_FOLLOWUP_STUDY.json").read_text())
    go = json.loads((ROOT / f"GPU_GO_BUDGET_{model}.json").read_text())
    assert go["independent_preflight_pass"] is True
    assert go["freeze_sha256"] == sha(ROOT / "BUDGET_FOLLOWUP_FREEZE.json")
    assert str(go["gpu"]) == str(gpu) and go["model"] == model
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(gpu)
    assert (ROOT / f"run_{model}_cache1/ALL_WORK_COMPLETE.json").exists(), "Finish original ON study first"
    if processes(gpu):
        raise RuntimeError("Selected GPU has another process before follow-up load")
    out = ROOT / f"budget_run_{model}"
    out.mkdir(exist_ok=False)
    try:
        write_json(out / "START.json", {"model": model, "gpu": gpu, "freeze_sha256": go["freeze_sha256"],
                   "GO": go, "time_unix": time.time(), "pid": os.getpid()})
        from vllm import LLM
        if model == "glm":
            from native_runtime_glm_v3 import NativeBackend
        else:
            from native_runtime import NativeBackend
        llm = LLM(model=MODELS[model], tokenizer=MODELS[model], trust_remote_code=model == "glm",
                  dtype="bfloat16", tensor_parallel_size=1, max_model_len=8192,
                  gpu_memory_utilization=.70, max_num_seqs=32, enforce_eager=True,
                  enable_prefix_caching=True, disable_log_stats=True,
                  guided_decoding_backend="xgrammar", guided_decoding_disable_fallback=True)
        backend = NativeBackend(llm, out, model, True, gpu, owned)
        backend.stage = "controls"
        ms = [[{"role": "user", "content": "Return exactly the word BANANA and nothing else."}]]
        raw = backend.generate(ms, max_output=32)
        guided = backend.generate(ms, grammar=batch_grammar(1), max_output=128)
        smoke = backend.generate(ms*8, grammar=batch_grammar(1), max_output=128)
        control = {"ordinary_banana": "BANANA" in raw[0]["raw"],
                   "guided": parse_batch(guided[0]["raw"], 1) is not None,
                   "smoke_format": all(parse_batch(r["raw"], 1) is not None for r in smoke),
                   "usage": backend.usage(0), "semantic_answers_not_retention_gates": True}
        write_json(out / "CONTROLS.json", control)
        assert control["ordinary_banana"] and control["guided"] and control["smoke_format"]
        workloads = {w["workload_id"]: w for w in config["selected_workloads"]}
        for job in config["jobs_per_model"]:
            execute_policy(backend, workloads[job["workload_id"]], job, out)
        write_json(out / "BUDGET_ALL_WORK_COMPLETE.json", {"status": "complete", "model": model,
                   "jobs_complete": 8, "all_usage": backend.usage(0), "time_unix": time.time(),
                   "responses_sha256": sha(out / "responses.jsonl"),
                   "raw_returns_sha256": sha(out / "raw_returns.jsonl")})
    except BaseException as error:
        write_json(out / "FAILURE.json", {"type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "automatic_retry": False, "time_unix": time.time()})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--gpu", required=True)
    a = p.parse_args()
    main(a.model, a.gpu)
