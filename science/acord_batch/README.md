# Post-hoc ACORD batch comparison: reconstruction candidate

This supplement records a same-pool four/eight-item comparison performed **after earlier outcomes were observed** and after a restart erased the first four-item run's per-response traces. It is not an independent or blinded study. The recovery runtime had documented vLLM dependency conflicts; both keyed sizes were rerun in the same process, and guidance controls passed. The earlier full-pointwise reference was produced in a different run/environment. The plans also differ in chunk partitions, local IDs, grammar lengths and scheduling, so this is not a causal batch-size estimate. `PROTOCOL_recovery_batch.md`, `RECOVERY_HISTORY.md`, and `pip_check_before_recovery.log` provide the chronology and limits.

The supplement does not directly include ACORD clause text, query text, or candidate labels. Its unsalted pool pseudonyms can be linked back to the public upstream data and are **not anonymization**. Character-shape screening of model outputs is not a license, privacy or redistribution review. Model weights are not bundled.

First reconstruct the original Qwen ACORD study using the main reproduction artifact and the separate Qwen ACORD reconstruction supplement. Its fresh working directory must contain `selection_2400_corrected/` and `restored_results/`. Then execute, with compatible NumPy 2.2.6, transformers 4.57.1 and tokenizers 0.22.0:

```sh
bash rebuild_batch_supplement.sh QWEN_REBUILT_WORK TOKENIZER_MODEL_DIR NEW_WORK_DIR
```

The tokenizer directory must be named `Qwen3-4B-Instruct-2507` and contain the matching tokenizer/configuration files. The new work directory must not exist; set `INTENTSAT_PYTHON` to choose Python. The script verifies this supplement's checksums, recomputes tokenizer preflight, restores pseudonymous records against original file hashes, recomputes both scenario comparisons, and runs independent accounting/statistical checks. It does not rerun a GPU model or establish a clean-machine GPU installation.

## 中文说明

这是事后同池、非盲法的恢复运行，原执行环境重启导致首轮逐条轨迹丢失；恢复时的软件依赖存在冲突。两种分块在同一次恢复进程中执行，不能把差异单独归因于批次大小。无原文候选中的伪名可以反查，发布前须审核上游数据、输出、权利和隐私；本包尚未公开。
