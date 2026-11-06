#!/usr/bin/env python3
"""Read-only extraction of the existing frozen Qwen/SciTech/7600/ON/rep1 pair.

Python standard library only; no model, GPU, network, source mutation or new trial.
--root accepts the scientific root or an expanded compact artifact root.
All emitted files are derived explanatory evidence, not protocol modifications.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


TOKEN_FIELDS = {
    "P_input": "actual_input_tokens", "O_output": "actual_output_tokens",
    "K_cached": "cached_input_tokens", "U_uncached": "observed_uncached_input_tokens",
}
LOTUS_COMMIT = "136ae4f4a344a2f75d89f811e516dfcb0de30e46"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprints = {}

    def track(rel, optional=False):
        p = root / rel
        if not p.exists() and optional:
            fingerprints[rel] = {"available": False}
            return None
        fingerprints[rel] = {"bytes": p.stat().st_size, "sha256": sha(p)}
        return p

    def load(rel):
        return json.loads(track(rel).read_text())

    def rows(rel):
        with track(rel).open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def bill(items):
        items = list(items)
        result = {"requests": len(items)}
        for name, field in TOKEN_FIELDS.items():
            values = [r[field] for r in items]
            assert all(isinstance(x, int) and x >= 0 for x in values), (name, values)
            result[name] = sum(values)
        assert result["P_input"] == result["K_cached"] + result["U_uncached"]
        return result

    def add(a, b):
        return {k: a[k] + b[k] for k in a}

    def subtract(a, b):
        result = {k: a[k] - b[k] for k in a}
        assert all(v >= 0 for v in result.values())
        return result

    def assert_policy_usage(derived, official):
        assert derived == {
            "requests": official["requests"], "P_input": official["input_tokens"],
            "O_output": official["output_tokens"], "K_cached": official["cached_tokens"],
            "U_uncached": official["uncached_tokens"],
        }, (derived, official)
        assert official["cached_tokens_unknown_requests"] == 0
        assert official["cached_tokens_observed_lower_bound"] == derived["K_cached"]

    study = load("prepared/STUDY.json")
    feasibility = load("prepared/TOKENIZER_FEASIBILITY.json")
    followup = load("BUDGET_FOLLOWUP_STUDY.json")
    freeze = load("BUDGET_FOLLOWUP_FREEZE.json")
    start = load("budget_run_qwen/START.json")
    assert freeze["followup_inference_started"] is False
    assert freeze["created_unix"] < start["time_unix"]
    assert start["freeze_sha256"] == fingerprints["BUDGET_FOLLOWUP_FREEZE.json"]["sha256"]
    for rel in ("prepare_study.py", "prepare_budget_followup.py", "experiment_common.py",
                "native_runtime.py", "native_runtime_glm_v3.py", "run_budget_followup.py",
                "run_study.py", "real_engine/adapters/lotus_native.py"):
        track(rel)
    track("novelty_theory/followup_design/REPORT.md", optional=True)
    track(f"real_engine/repos/lotus-{LOTUS_COMMIT}/lotus/sem_ops/postprocessors.py", optional=True)

    world = next(w for w in study["workloads"] if w["workload_id"] == "agnews_count_world")
    world_result_path = "run_qwen_cache1/agnews_count_world__n2048__cache1/S1/native_result.json"
    world_result = load(world_result_path)
    examples = []
    for raw in ("Answer: Sports", "Answer: Sci/Tech"):
        r = next(r for r in world_result["rows"] if r["raw_output"] == raw)
        assert r["native_label"] is True and r["strict_valid"] is False and r["strict_label"] is None
        examples.append({k: r[k] for k in ("record_id", "raw_output", "native_label", "strict_label", "strict_valid")})
    feasibility_summary = {}
    for model, entries in feasibility.items():
        native = [r["native_max_tokens"] for r in entries]
        batch = [r["longest_eight_batch_tokens"] for r in entries]
        assert len(entries) == 6
        assert all(r["complete_fields"] and not r["truncation"] for r in entries)
        assert max(batch) + study["config"]["batch_max_output"] <= study["config"]["max_model_len"]
        assert max(native) + study["config"]["native_max_output"] <= study["config"]["max_model_len"]
        feasibility_summary[model] = {"native_maxima_range": [min(native), max(native)],
            "longest_eight_batch_range": [min(batch), max(batch)], "per_workload": entries}

    original_case = "run_qwen_cache1/agnews_count_science_technology__n7600__cache1"
    trigger_result = load(original_case + "/CURRENT_CP_RESULT.json")
    trigger_refs = load(original_case + "/CURRENT_CP/calibration_singletons/native_result.json")
    assert trigger_result["selected_k"] == 1
    assert len(trigger_refs["rows"]) == 80 and all(r["strict_valid"] for r in trigger_refs["rows"])
    movie_hashes = {w["workload_id"]: hashlib.sha256(("2026110621|" + w["workload_id"]).encode()).hexdigest()
                    for w in study["workloads"] if w["domain"] == "movie"}
    movie_anchor = min(movie_hashes, key=movie_hashes.get)
    assert movie_anchor == "movie_count_positive_other_critics"
    ag_anchor = next(w for w in study["anchor_workloads"] if w.startswith("agnews_"))
    assert ag_anchor == "agnews_count_science_technology"
    allowed = {r["budget"]: r["maximum_disagreements_allowed"] for r in followup["exact_iid_power_not_model_measurements"]}
    assert allowed == {80: 0, 320: 7}

    ledger = rows("budget_run_qwen/responses.jsonl")
    trace, phases, checks = [], [], []
    for budget in (80, 320):
        job = f"agnews_count_science_technology__n7600__budget{budget}__rep1"
        rel = "budget_run_qwen/" + job
        policy = load(rel + "/POLICY_RESULT.json")
        assert policy["job"]["budget"] == budget and policy["selected_k"] == 1
        stage_cal, stage_exec = job + "/calibration", job + "/execution"
        cal = [r for r in ledger if r["stage"] == stage_cal]
        execution = [r for r in ledger if r["stage"] == stage_exec]
        by_id = {r["logical_id"]: r for r in cal}
        assert len(by_id) == len(cal)
        cal_bill, exec_bill = bill(cal), bill(execution)
        all_probe_ids, probe_bill = set(), bill([])

        def phase(name, value, stage, method):
            phases.append({"model": "qwen", "workload": "agnews_count_science_technology", "N": 7600,
                "cache": "ON", "repeat": 1, "budget": budget, "accounting_category": name,
                "actual_ledger_stage": stage, "derivation": method, **value})

        for c in policy["calibration"]:
            k = c["k"]
            chunks = sorted((root / rel).glob(f"candidate_{k}_chunk_*/batch_calls.jsonl"),
                            key=lambda p: int(p.parent.name.rsplit("_", 1)[1]))
            issued = []
            for p in chunks:
                chunk = rows(str(p.relative_to(root)))
                assert len(chunk) == 16
                issued.extend(chunk)
            probes = c["probes"]
            ids = [r["logical_id"] for r in issued]
            assert len(ids) == len(set(ids)) == c["actually_checked"] == len(probes)
            assert ids == [r["logical_id"] for r in probes]
            assert not all_probe_ids.intersection(ids)
            all_probe_ids.update(ids)
            for r, p in zip(issued, probes):
                canonical = by_id[r["logical_id"]]
                assert all(r[key] == canonical[key] for key in canonical)
                assert r["stage"] == stage_cal and len(r["indices"]) == k
                assert r["strict_valid"] == p["strict_valid"]
                if p["strict_valid"]:
                    assert r["labels"][p["probe_slot"]] == p["label"]
                assert p["agreement"] == bool(p["strict_valid"] and p["label"] == p["reference_native_label"])
            agrees = sum(p["agreement"] for p in probes)
            assert agrees == c["agreements"]
            failures, first_decisive, chunk_trace = 0, None, []
            for j, p in enumerate(probes, 1):
                failures += not p["agreement"]
                if first_decisive is None and failures > allowed[budget]:
                    first_decisive = j
                if j % 16 == 0:
                    chunk_trace.append({"issued_scored": j, "agreements": j - failures, "failures": failures})
            assert not c["passed"] and first_decisive is not None
            assert len(probes) - 16 < first_decisive <= len(probes)
            candidate_bill = bill(by_id[i] for i in ids)
            probe_bill = add(probe_bill, candidate_bill)
            trace.append({"model": "qwen", "workload": "agnews_count_science_technology", "N": 7600,
                "cache": "ON", "repeat": 1, "budget": budget, "candidate_k": k,
                "max_failures_allowed_at_full_budget": allowed[budget], "issued_batch_requests": len(issued),
                "scored_trials": len(probes), "agreements": agrees, "failures": failures,
                "strict_invalid_probe_outputs": sum(not p["strict_valid"] for p in probes),
                "valid_boolean_disagreements": sum(p["strict_valid"] and not p["agreement"] for p in probes),
                "first_decisive_trial_1based": first_decisive,
                "issued_after_first_decisive": len(probes) - first_decisive,
                "fully_scored_chunks": len(chunks), "passed": c["passed"],
                "chunk_cumulative_trace": json.dumps(chunk_trace, separators=(",", ":")),
                **{n: candidate_bill[n] for n in TOKEN_FIELDS}, "policy_result": rel + "/POLICY_RESULT.json"})
            phase(f"candidate_k{k}_probes", candidate_bill, stage_cal, "logical_id join to canonical responses")

        refs_bill = subtract(cal_bill, probe_bill)
        residual = [r for r in cal if r["logical_id"] not in all_probe_ids]
        assert refs_bill == bill(residual) and refs_bill["requests"] == budget
        refs = load(rel + "/calibration_singletons/native_result.json")
        assert len(refs["rows"]) == policy["native_ref_rows"] == budget
        assert sum(r["strict_valid"] for r in refs["rows"]) == policy["native_ref_format_valid_rows"]
        assert len(execution) == 7600
        full_bill = add(cal_bill, exec_bill)
        assert_policy_usage(cal_bill, policy["calibration_usage"])
        assert_policy_usage(exec_bill, policy["execution_usage"])
        assert_policy_usage(full_bill, policy["usage"])
        phase("singleton_references", refs_bill, stage_cal,
              "exact ledger decomposition: calibration total minus all candidate probes; residual-ID cross-check")
        phase("calibration_total", cal_bill, stage_cal, "canonical stage sum; agrees exactly with POLICY_RESULT")
        phase("execution_total", exec_bill, stage_exec, "canonical stage sum; agrees exactly with POLICY_RESULT")
        phase("full_policy", full_bill, stage_cal + " + " + stage_exec,
              "calibration plus execution; agrees exactly with POLICY_RESULT")
        checks.append({"budget": budget, "selected_k": policy["selected_k"], "native_ref_rows": budget,
            "native_ref_strict_valid_rows": policy["native_ref_format_valid_rows"],
            "reference_bill": refs_bill, "all_probe_bill": probe_bill,
            "calibration_bill": cal_bill, "execution_bill": exec_bill, "full_policy_bill": full_bill,
            "POLICY_RESULT_usage_exact_match": True,
            "native_complete": policy["native_complete"], "native_count": policy["native_count"],
            "strict_complete": policy["strict_complete"], "execution_strict_valid_rows": policy["strict_valid_rows"]})

    def write_csv(name, data):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    write_csv("candidate_stop_trace.csv", trace)
    write_csv("phase_exact_cost.csv", phases)
    evidence = {
        "scope": "Explanatory extraction of existing frozen artifacts only; no new trials or statistical conclusion.",
        "root": str(root),
        "DB10": {"predicate_template": world["predicate"], "model": study["models"]["qwen"],
                 "source": world_result_path, "reply_examples_no_news_text": examples,
                 "parser": "Pinned LOTUS substring/default=True; separate strict audit accepts only one True/False label."},
        "DB15": {"candidate_sizes": study["config"]["candidate_sizes"],
                 "batch_output_budget": study["config"]["batch_max_output"],
                 "native_output_budget": study["config"]["native_max_output"],
                 "context": study["config"]["max_model_len"], "feasibility": feasibility_summary,
                 "scope": "k=8 is the experiment-given tested feasible ceiling; no larger k was tested or proved infeasible/optimal.",
                 "precheck": "prepare_study.py:65-82; select eight longest native requests per workload/model and serialize their full batch.",
                 "runtime_guard": "native_runtime.py:123-126 and native_runtime_glm_v3.py:131-134 reject len(prompt_ids)+max_output>8192 without truncation."},
        "DB18": {"trigger": followup["trigger"], "original_refs": 80, "original_refs_strict_valid": 80,
                 "ag_anchor": ag_anchor, "movie_anchor": movie_anchor, "movie_hash_seed": "2026110621",
                 "movie_candidate_hashes": movie_hashes, "budget_multiple": "320 = 4 * 80",
                 "allowed_disagreements": allowed, "registration_status": followup["status"],
                 "freeze_created_unix": freeze["created_unix"], "qwen_followup_start_unix": start["time_unix"],
                 "freeze_before_own_calls": True},
        "MT14_MT24": checks,
        "interpretation_constraints": [
            "The ledger has calibration/execution stages for these jobs; reference/candidate rows are exact derived accounting categories, not invented logged stages.",
            "A chunk is 16 batch requests. The complete returned chunk is scored before rejection; all issued requests are billed.",
            "First-decisive positions are reconstructed in frozen trial order, not an online stopping point or an estimate of q.",
            "All CP80/CP320 calls are physically independent; pairing is of predeclared inputs and seeds, not response reuse.",
            "No inference about a general LOTUS bug or token-cap causation follows from the example.",
            "No need to guess historical names or procedures moved out of the active manuscript (MT26/MT27)."],
        "source_fingerprints": fingerprints,
    }
    (out / "FROZEN_EVIDENCE.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(out), "candidate_rows": len(trace), "phase_rows": len(phases),
                      "exact_usage_matches": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
