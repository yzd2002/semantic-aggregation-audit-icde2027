"""Live LOTUS/native-vLLM physical execution with prelabel-frozen choices.

No provider keys, response cache, automatic retries, label-dependent input
selection, or cross-policy reuse of inference responses are used.
"""
from experiment_common import *
from native_runtime import NativeBackend, durable
import argparse
import math
import os
import subprocess
import sys
import traceback


def processes(gpu):
    raw = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True)
    return [{"pid": int(l.split(",")[0]), "record": l} for l in raw.splitlines() if l.strip()]


def owned(gpu):
    import psutil
    allowed = {os.getpid()} | {p.pid for p in psutil.Process().children(recursive=True)}
    running = processes(gpu)
    if any(p["pid"] not in allowed for p in running):
        raise RuntimeError("Another task appeared on selected GPU; no more submissions")
    return running


def native_call(backend, records, predicate, phase_dir, *, official_q3=False, expected_messages=None):
    phase_dir.mkdir(parents=True, exist_ok=False)
    records_path = phase_dir / "records.jsonl"
    if not official_q3:
        write_lines(records_path, records)
    result_path = phase_dir / "native_result.json"
    base = ROOT / "real_engine"
    command = [str(base / ".venv/bin/python"),
               str(base / "adapters" / ("stdio_sembench_q3.py" if official_q3 else "stdio_native_driver.py")),
               "--output", str(result_path), "--model-name", MODELS[backend.model]]
    if not official_q3:
        command += ["--records", str(records_path), "--predicate", predicate]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONPATH": str(ROOT), "INTENTSAT_SINGLE_RANK_NCCL_COMPAT": "0",
           "INTENTSAT_CACHE_RESET_OBSERVER": "0"}
    started = time.perf_counter()
    with (phase_dir / "stderr.log").open("w") as errorlog:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=errorlog, text=True, env=env, bufsize=1)
        events = []
        child_exception = None
        request_cursor = 0
        try:
            assert child.stdin and child.stdout
            for line in child.stdout:
                event = json.loads(line)
                events.append({k: v for k, v in event.items() if k != "messages"})
                if event["type"] == "requests":
                    if expected_messages is not None:
                        actual_messages = event["messages"]
                        if actual_messages != expected_messages[request_cursor:request_cursor+len(actual_messages)]:
                            raise RuntimeError("Live native LOTUS messages differ from frozen formatter preflight")
                        request_cursor += len(actual_messages)
                    kwargs = event.get("kwargs", {})
                    if kwargs.get("logprobs"):
                        raise RuntimeError("Unplanned native cascade/logprobs request")
                    rows = backend.generate(event["messages"], max_output=CONFIG["native_max_output"])
                    child.stdin.write(compact({"raw_outputs": [r["raw"] for r in rows]}) + "\n")
                    child.stdin.flush()
                elif event["type"] != "done":
                    raise RuntimeError("Unrecognized LOTUS transport event")
            returncode = child.wait()
            if expected_messages is not None and request_cursor != len(expected_messages):
                raise RuntimeError("Native operator did not request every expected complete record")
        except BaseException as error:
            child_exception = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            child.stdin.close()
            child.stdout.close()
            if child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # This is only the CPU child created above, never another job.
                    child.terminate()
                    child.wait(timeout=10)
            returncode = child.returncode
            write_json(phase_dir / "transport_receipt.json", {"command": command,
                       "child_pid": child.pid, "child_exit": returncode, "events": events,
                       "exception": child_exception, "live_native_operator": True,
                       "elapsed_seconds": time.perf_counter()-started})
    if returncode != 0:
        raise RuntimeError(f"Actual LOTUS operator failed, exit {returncode}; raw native returns preserved")
    return json.loads(result_path.read_text())


def batch_plan(backend, requests, groups, phase_dir, *, probe_positions=None):
    phase_dir.mkdir(parents=True, exist_ok=False)
    labels = [None] * len(requests)
    all_calls = []
    by_k = {}
    # Complete partition calls have a single k, except its possible short tail.
    # Calibration groups always have the same k, so execution ordering is stable.
    for ordinal, group in enumerate(groups):
        by_k.setdefault(len(group), []).append((ordinal, group))
    rows_by_ordinal = {}
    for k, subset in by_k.items():
        messages = [batch_messages([requests[i]["messages"] for i in g]) for _, g in subset]
        values = backend.generate(messages, grammar=batch_grammar(k), max_output=CONFIG["batch_max_output"])
        for (ordinal, group), value in zip(subset, values):
            parsed = parse_batch(value["raw"], len(group))
            row = {**value, "group_ordinal": ordinal, "indices": group,
                   "strict_valid": parsed is not None, "labels": parsed}
            rows_by_ordinal[ordinal] = row
            if parsed is not None:
                for i, label in zip(group, parsed):
                    labels[i] = label
    for ordinal in range(len(groups)):
        all_calls.append(rows_by_ordinal[ordinal])
    write_lines(phase_dir / "batch_calls.jsonl", all_calls)
    if probe_positions is not None:
        return [{"probe_index": groups[j][pos], "probe_slot": pos,
                 "label": all_calls[j]["labels"][pos] if all_calls[j]["labels"] is not None else None,
                 "strict_valid": all_calls[j]["strict_valid"],
                 "logical_id": all_calls[j]["logical_id"]}
                for j, pos in enumerate(probe_positions)]
    assert sorted(i for g in groups for i in g) == list(range(len(requests)))
    return {"labels": labels, "strict_complete": all(x is not None for x in labels),
            "strict_count": sum(labels) if all(x is not None for x in labels) else None,
            "groups": groups}


def execute_case(backend, workload, case, out):
    from scipy.stats import beta
    n, case_id, key = case["n"], case["case_id"], case["partition_key"]
    records = read_lines(workload["records"])[:n]
    requests = read_lines(workload["native_requests"])[:n]
    assert len(records) == len(requests) == n
    case_out = out / case_id
    case_out.mkdir()
    results = {}
    # Fixed pre-outcome permutation avoids systematically timing the same policy last.
    policies = ["S1", "S2", "RANDOM8", "SIMILAR8", "CURRENT_CP"]
    rng_for(key, "policy_schedule").shuffle(policies)
    write_json(case_out / "START.json", {"case": case, "workload_id": workload["workload_id"],
               "policy_order": policies, "source_records_sha256": workload["records_sha256"],
               "prelabel_freeze_sha256": sha(ROOT / "PREINFERENCE_FREEZE.json"),
               "no_gold_read_by_runner": True, "time_unix": time.time()})
    for policy in policies:
        backend.reset_policy(case_id, policy)
        backend.stage = case_id + "/" + policy
        start_response = len(backend.responses)
        started = time.perf_counter()
        phase_dir = case_out / policy
        if policy in ("S1", "S2"):
            native = native_call(backend, records, workload["predicate"], phase_dir,
                                 expected_messages=[r["messages"] for r in requests])
            result = {"labels": [r["native_label"] for r in native["rows"]],
                      "strict_labels": [r["strict_label"] for r in native["rows"]],
                      "strict_complete": native["strict_count"] is not None,
                      "strict_count": native["strict_count"], "native_count": native["native_count"]}
        elif policy in ("RANDOM8", "SIMILAR8"):
            if policy == "RANDOM8":
                groups, constructor_seconds = random_partition(n, 8, key), 0.0
            else:
                groups, constructor_seconds = similarity_partition(records, 8)
            result = batch_plan(backend, requests, groups, phase_dir)
            result["group_constructor_seconds"] = constructor_seconds
        else:
            # Paper-defined NousDB Algorithm 2 given a feasible kmax=8; this
            # does not reproduce NousDB's histogram optimizer or engine.
            phase_dir.mkdir()
            indices = probes(n, key, "calibration_probes", CONFIG["probe_budget"])
            reference = native_call(backend, [records[i] for i in indices], workload["predicate"], phase_dir / "calibration_singletons",
                                    expected_messages=[requests[i]["messages"] for i in indices])
            singles = reference["rows"]
            selected = 1
            decisions = []
            for k in CONFIG["candidate_sizes"]:
                groups, positions = probe_batches(n, k, key, "calibration_batches", indices)
                checked = []
                for offset in range(0, CONFIG["probe_budget"], CONFIG["calibration_chunk"]):
                    group_slice = groups[offset:offset+CONFIG["calibration_chunk"]]
                    slot_slice = positions[offset:offset+CONFIG["calibration_chunk"]]
                    rows = batch_plan(backend, requests, group_slice,
                                      phase_dir / f"candidate_{k}_chunk_{offset}", probe_positions=slot_slice)
                    for j, row in enumerate(rows, offset):
                        row["reference_strict_label"] = singles[j]["strict_label"]
                        row["agreement"] = bool(row["strict_valid"] and singles[j]["strict_valid"]
                                                and row["label"] == singles[j]["strict_label"])
                        checked.append(row)
                    # At this minimum budget any disagreement makes passing
                    # impossible. Every over-issued call in the chunk is billed.
                    possible_agreements = sum(r["agreement"] for r in checked) + CONFIG["probe_budget"] - len(checked)
                    possible_lower = beta.ppf(CONFIG["delta"]/3, possible_agreements,
                                               CONFIG["probe_budget"]-possible_agreements+1) if possible_agreements else 0.
                    if possible_lower < 1-CONFIG["eta"]:
                        break
                x = sum(r["agreement"] for r in checked)
                lower = beta.ppf(CONFIG["delta"]/3, x, CONFIG["probe_budget"]-x+1) if x else 0.
                passed = len(checked) == CONFIG["probe_budget"] and lower >= 1-CONFIG["eta"]
                decision = {"k": k, "probe_budget": CONFIG["probe_budget"], "actually_checked": len(checked),
                            "agreements": x, "lower": float(lower), "passed": bool(passed), "probes": checked}
                decisions.append(decision)
                if passed:
                    selected = k
                    break
            calibration_usage = backend.usage(start_response)
            calibration_seconds = time.perf_counter() - started
            backend.stage = case_id + "/CURRENT_CP/execution"
            execution_start = len(backend.responses)
            if selected == 1:
                native = native_call(backend, records, workload["predicate"], phase_dir / "execution_singletons",
                                     expected_messages=[r["messages"] for r in requests])
                result = {"labels": [r["native_label"] for r in native["rows"]],
                          "strict_labels": [r["strict_label"] for r in native["rows"]],
                          "strict_complete": native["strict_count"] is not None,
                          "strict_count": native["strict_count"], "native_count": native["native_count"]}
            else:
                result = batch_plan(backend, requests, random_partition(n, selected, key), phase_dir / "execution_partition")
            result.update({"selected_k": selected, "calibration": decisions,
                           "calibration_usage": calibration_usage, "calibration_seconds": calibration_seconds,
                           "execution_usage": backend.usage(execution_start)})
        result.update({"policy": policy, "usage": backend.usage(start_response),
                       "elapsed_seconds": time.perf_counter()-started})
        results[policy] = result
        write_json(case_out / (policy + "_RESULT.json"), result)
        print(compact({"case_id": case_id, "policy": policy,
                       "requests": result["usage"]["requests"], "seconds": result["elapsed_seconds"],
                       "selected_k": result.get("selected_k"), "complete": result["strict_complete"]}), flush=True)
    # Independent, preseeded validation of the selected non-singleton size;
    # diagnostically separate from the actual policy's cost and decision.
    selected = results["CURRENT_CP"]["selected_k"]
    validation = {"status": "not_applicable_singleton_selected", "selected_k": selected}
    if selected > 1:
        backend.reset_policy(case_id, "INDEPENDENT_VALIDATION")
        backend.stage = case_id + "/INDEPENDENT_VALIDATION"
        start_response = len(backend.responses)
        indices = probes(n, key, "independent_validation_probes", CONFIG["validation_probes"])
        groups, positions = probe_batches(n, selected, key, "independent_validation_batches", indices)
        ref = native_call(backend, [records[i] for i in indices], workload["predicate"], case_out / "validation_singletons",
                          expected_messages=[requests[i]["messages"] for i in indices])
        batched = batch_plan(backend, requests, groups, case_out / "validation_batches", probe_positions=positions)
        validation = {"status": "complete", "selected_k": selected, "indices": indices,
                      "fresh_singletons": ref["rows"], "batched_probes": batched,
                      "usage": backend.usage(start_response),
                      "not_used_for_selection": True}
    write_json(case_out / "INDEPENDENT_VALIDATION.json", validation)
    write_json(case_out / "CASE_COMPLETE.json", {"case": case, "policies": policies,
               "results_sha256": {p: sha(case_out / (p + "_RESULT.json")) for p in policies},
               "validation_sha256": sha(case_out / "INDEPENDENT_VALIDATION.json"), "time_unix": time.time()})


def main(model, cache, gpu):
    study = json.loads((ROOT / "prepared/STUDY.json").read_text())
    assert study["config"] == CONFIG
    freeze = json.loads((ROOT / "PREINFERENCE_FREEZE.json").read_text())
    for rel, want in freeze["files"].items():
        assert sha(ROOT / rel) == want, rel
    go = json.loads((ROOT / f"GPU_GO_{model}_{int(cache)}.json").read_text())
    assert go["freeze_sha256"] == sha(ROOT / "PREINFERENCE_FREEZE.json")
    assert (go["model"], go["prefix_cache"], str(go["gpu"])) == (model, cache, str(gpu))
    assert go["independent_preflight_pass"] is True
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu)
    if processes(gpu):
        raise RuntimeError("Selected GPU is no longer idle; not loading")
    out = ROOT / f"run_{model}_cache{int(cache)}"
    out.mkdir(exist_ok=False)
    try:
        write_json(out / "START.json", {"model": model, "prefix_cache": cache, "gpu": gpu,
                   "pid": os.getpid(), "prelabel_freeze_sha256": sha(ROOT / "PREINFERENCE_FREEZE.json"),
                   "GPU_GO": go, "started_unix": time.time()})
        import single_rank_nccl_compat as compat
        assert compat.ACTIVE
        from vllm import LLM
        llm = LLM(model=MODELS[model], tokenizer=MODELS[model], trust_remote_code=model == "glm",
                  dtype="bfloat16", tensor_parallel_size=1, max_model_len=CONFIG["max_model_len"],
                  gpu_memory_utilization=.70, max_num_seqs=CONFIG["max_num_seqs"], enforce_eager=True,
                  enable_prefix_caching=cache, disable_log_stats=True,
                  guided_decoding_backend="xgrammar", guided_decoding_disable_fallback=True)
        backend = NativeBackend(llm, out, model, cache, gpu, owned)
        backend.stage = "controls"
        ordinary = [[{"role": "user", "content": "Return exactly the word BANANA and nothing else."}]]
        raw = backend.generate(ordinary, max_output=32)
        guided = backend.generate(ordinary, grammar=batch_grammar(1), max_output=128)
        smoke_messages = batch_messages([[{"role": "system", "content": "Test"},
                                         {"role": "user", "content": "Context: two plus two is four. Claim: two plus two is four."}]])
        smoke = backend.generate([smoke_messages]*8, grammar=batch_grammar(1), max_output=128)
        control = {"ordinary_banana": "BANANA" in raw[0]["raw"],
                   "guided_incompatible_instruction": parse_batch(guided[0]["raw"], 1) is not None,
                   "synthetic_format": all(parse_batch(r["raw"], 1) is not None for r in smoke),
                   "semantics_not_a_retention_gate": True, "usage": backend.usage(0)}
        write_json(out / "CONTROLS.json", control)
        if not all(control[k] for k in ("ordinary_banana", "guided_incompatible_instruction", "synthetic_format")):
            raise RuntimeError("Pre-science output contract controls failed")
        backend.reset_policy("official_q3", "native")
        backend.stage = "OFFICIAL_SEMBENCH_Q3"
        start_response = len(backend.responses)
        q3_requests = read_lines(ROOT / "real_engine/sources/sembench_q3_smoke/native_requests.jsonl")
        q3 = native_call(backend, [], "", out / "official_sembench_q3", official_q3=True,
                         expected_messages=[r["messages"] for r in q3_requests])
        write_json(out / "OFFICIAL_Q3_RECEIPT.json", {"actual_native_query": True,
                   "result_file": "official_sembench_q3/native_result.json", "usage": backend.usage(start_response),
                   "source_query_not_substituted_by_large_variants": True})
        cases = [c for c in study["cases"] if c["prefix_cache"] == cache]
        rng_for("case_schedule", cache).shuffle(cases)
        workloads = {w["workload_id"]: w for w in study["workloads"]}
        for case in cases:
            execute_case(backend, workloads[case["workload_id"]], case, out)
        write_json(out / "ALL_WORK_COMPLETE.json", {"status": "complete", "model": model,
                   "prefix_cache": cache, "cases_complete": len(cases), "all_usage": backend.usage(0),
                   "all_returned_requests": len(backend.responses), "time_unix": time.time(),
                   "responses_sha256": sha(out / "responses.jsonl"),
                   "raw_returns_sha256": sha(out / "raw_returns.jsonl"),
                   "effective_requests_sha256": sha(out / "effective_requests.jsonl")})
    except BaseException as e:
        write_json(out / "FAILURE.json", {"error": str(e), "type": type(e).__name__,
                   "traceback": traceback.format_exc(), "automatic_retry": False,
                   "time_unix": time.time(), "returned_raw_preserved": (out / "raw_returns.jsonl").exists()})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--cache", type=int, choices=(0, 1), required=True)
    p.add_argument("--gpu", required=True)
    a = p.parse_args()
    main(a.model, bool(a.cache), a.gpu)
