# Observed rejection evidence: portable CPU entry

Run `analyze_observed_rejections.py` with Python 3 standard library only. It performs no model call, reads no gold labels, never invents an outcome for an unreached candidate, and does not estimate population q from stopped prefixes.

```bash
python3 analyze_observed_rejections.py \
  --root /path/to/unpacked/study \
  --main-summary /path/to/completed/main/summary.json \
  --budget-summary /path/to/completed/budget/summary.json \
  --output /path/to/rejection_readout \
  --tex-output /path/to/rejection_readout/tex
```

The two summaries must report complete 36/36 and 16/16 matrices and zero integrity errors. They supply case/job identities, model/cache roles and the full fixed denominators. The helper constructs these paths beneath `--root`:

- For each of the 36 main configurations: `run_{model}_cache{0,1}/{case_id}/CURRENT_CP_RESULT.json` and `CURRENT_CP/calibration_singletons/native_result.json`.
- For each of the 16 follow-up policies: `budget_run_{model}/{job_id}/POLICY_RESULT.json`.

No original absolute path inside the summaries is executed or read. The main native-reference file supplies native and strict labels in the recorded probe order; the follow-up result embeds its own native/strict reference labels. Every actually issued candidate prefix is classified, and stored agreement flags must match the reconstructed comparison event. Actual batch failure is classified before native mismatch, followed by strict-only format rejection; non-strict references that already mismatch natively remain in the native-mismatch category. A separate diagnostic counts all issued reference trials with strict-invalid format. Counts concern issued trials, including repeated source identities and reused references, not independent populations.

Outputs are `observed_probe_rejection_candidates.csv` (154 observed candidates), `observed_probe_rejection_summary.csv` (six family/model rows), `OBSERVED_REJECTION_READOUT.json` (scope/source hashes) and the generated TeX table. The two original candidates not reached after the Sports pass remain unobserved. `REJECTION_REPLAY_CHECK.json` records an actual explicit-CLI replay on the complete study, with byte-identical CSV outputs. The helper source hash and input file hashes are retained in the JSON; the CSV comparison remains stable when the study root is relocated.
