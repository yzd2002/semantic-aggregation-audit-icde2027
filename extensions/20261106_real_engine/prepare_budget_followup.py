"""One bounded, transparent follow-up after observing the first CP80 fallback.

The original 36 configurations are neither replaced nor relabeled. Workloads
are selected by predeclared feature-only hashes, not by measured outcomes.
"""
from experiment_common import *
from scipy.stats import beta, binom


def main():
    study = json.loads((ROOT / "prepared/STUDY.json").read_text())
    movie = [w for w in study["workloads"] if w["domain"] == "movie"]
    movie_anchor = min(movie, key=lambda w: digest("2026110621|"+w["workload_id"]))["workload_id"]
    ag_anchor = next(w for w in study["anchor_workloads"] if w.startswith("agnews_"))
    selected = [w for w in study["workloads"] if w["workload_id"] in (movie_anchor, ag_anchor)]
    jobs = []
    for workload in selected:
        for repeat in (1, 2):
            for budget in (80, 320):
                jobs.append({"job_id": f'{workload["workload_id"]}__n7600__budget{budget}__rep{repeat}',
                    "workload_id": workload["workload_id"], "n": 7600, "repeat": repeat,
                    "budget": budget, "sampling_key": f'budget_followup_20261106|{workload["workload_id"]}|rep{repeat}'})
    rng_for("budget_followup_20261106", "job_schedule").shuffle(jobs)
    power = []
    for budget in (80, 320):
        allowed = [errors for errors in range(budget+1)
                   if beta.ppf(.05/3, budget-errors, errors+1) >= .95]  # budget-errors=0 yields NaN -> fails.
        maximum = max(allowed)
        power.append({"budget": budget, "maximum_disagreements_allowed": maximum,
                      "iid_theoretical_pass_probabilities": {str(q): float(binom.cdf(maximum, budget, q))
                           for q in (0., .005, .01, .02, .03, .04, .05, .10)}})
    manifest = {"status": "PROSPECTIVE_FOR_THIS_FOLLOWUP_NOT_RETROACTIVE_REGISTRATION",
        "trigger": "The first completed original Qwen Sci/Tech N7600 case fell back at CP80; all its calibration references were strict valid. No hidden error-rate estimate is inferred from the early-stopped prefix.",
        "original_36_configurations_unchanged": True,
        "question": "Does a larger fixed certification budget improve complete-policy cost under the same native Boolean semantic contract?",
        "models": list(MODELS), "n": 7600, "prefix_cache": True,
        "selected_workloads": selected, "movie_anchor_hash_seed": "2026110621",
        "budgets": [80, 320], "repeats": [1, 2], "jobs_per_model": jobs,
        "model_policy_executions": 16, "eta": .05, "delta": .05, "candidates": [8, 4, 2],
        "probe_chunk": 16, "reference_contract": "Unchanged native LOTUS parsed Boolean, including disclosed default=True behavior; strict-format validity separately retained.",
        "physical_pairing": "Common feature/probe seed within workload/repeat across budgets, but all singleton and batch calls physically independent; no response reuse.",
        "evaluation_references": "Existing independently executed complete native S1/S2 on the same original ON input, plus source gold. They are evaluation data only, never borrowed for calibration.",
        "alpha_scope": "delta/3 across candidates within one predeclared budget/policy. No joint 95% guarantee across budgets/repeats/workloads and no certified post-hoc budget winner.",
        "no_new_online_HG_no_new_models_or_thresholds": True,
        "outcomes": "Report every job and failure, native COUNT and strict-format status, selected size, calibration/execution/full request and raw/cache-token bill, and fidelity/source errors. Two repeats are descriptive, not population inference or latency significance.",
        "exact_iid_power_not_model_measurements": power,
        "worst_case_scientific_requests": 134400,
        "scope": "This is the final bounded experimental addition in the current round; no outcome-dependent extension."}
    write_json(ROOT / "BUDGET_FOLLOWUP_STUDY.json", manifest)
    print(compact({"anchors": [w["workload_id"] for w in selected], "policy_runs": 16, "power": power}))


if __name__ == "__main__":
    main()
