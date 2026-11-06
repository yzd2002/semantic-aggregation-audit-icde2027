# Fresh three-family context replication

**Current v3 entrypoint:** read V3_HANDOFF.md and global_primary/GLOBAL_PRIMARY_AMENDMENT.md first. The prospective nine-case 179/180 interval rule supersedes the v1/v2 98.75% primary wording. Legacy statistical files and outputs are preserved; actual v2 documentation is archived under initial_cpu_preflight/v2_failure_retention_baseline.

**Built with glm-4.** The model's required license copy is [LICENSE](LICENSE). No weights are distributed in this experiment bundle.

This is a new, forward-frozen replication designed after the previous Qwen/Smol context and Qwen/Smol/GLM decision results were known. All model outputs in the new scientific ledgers must come from fresh inference. It is not a preregistration of the original research.

The authoritative design is PROTOCOL_fresh_three_family_context_replication.md and prepared/REQUEST_ENVELOPE.json. The original 256 targets and all near contexts are kept. Thirty-four primary random triplets were redrawn without labels to satisfy the original 24-token difference under all three native tokenizers; all original random triplets remain in a secondary exact-historical arm. Matching attempts, inversion replay, complete fields, source IDs, native token prefixes, schedules, model-file verification, CPU runtime controls and adapter diff are preserved.

Each model has 3584 primary mixed, 204 historical-extra mixed, 896 primary serial and 66 historical-extra serial scientific requests, plus the original 20 diagnostics. This is 4750 scientific calls and 20 diagnostics per model; the three-model total is 14250+60 calls. Sharing within a model is explicit in the historical call maps; no old observation or cross-model observation is reused. The original statistical functions remain byte-identical, including legacy 98.75% outputs. The authoritative current primary components use the root nine-case 179/180 amendment; no exact population/FWER guarantee is claimed.

No model may load without a root-issued GPU_GO_<model>.json tied to the final freeze SHA, with authorized_by=root, model and device fields. After root review, the explicit launcher takes --model qwen|smol|glm and --device <approved physical index>. It checks authorization before importing the model runtime. Separate per-model caches remain inside this persistent directory. Other tasks' processes are not stopped or changed.

The initial GLM-only CPU reports and superseded envelope are retained as CPU history. They contain no model judgments. PREINFERENCE_FROZEN_SHA256.json is a new current freeze; missing historical certificates are never recreated under old names.
