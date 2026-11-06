# SNLI runtime independent CPU review interface

当前阶段只允许 CPU。不得创建顶层 `GPU_GO_*.json`，不得加载模型权重或执行真实 GPU inference；测试必须使用假引擎。只写本 SNLI 目录的新 review 文件或新独立检查子目录；保留原候选、作者检查及已有 fixture 原件。

审查入口是 `RUNTIME_REVIEW_CANDIDATE_SHA256.json` 及其 sidecar。先核对该候选中每个文件，再独立核对原 `CPU_REVIEW_CANDIDATE_SHA256.json` 的 `e0c61d…` 和所列 105 个原件。global_primary 的四个文件须与根目录冻结版本逐字节一致。原 analyze.py、secondary.py 和协议仍在原候选内；新 amendment 必须显式 supersede 旧 98.75% primary wording。

建议独立检查：

1. 核对完整 SNLI source rejoin、gold entailment→Y 映射、prompt/grammar/native token、block 和 serial 复制，目标/背景不变。可用 `recheck_source_contracts_cpu.py --out <本目录内新独立目录>` 执行未修改的 `verify_prepared_cpu.py`：它只在 input-only 链接视图运行，将唯一报告重定向到新目录，避免原 verifier 的 `glob('run_*')` 把新增 run_gpu.py 误当运行目录。它读取 CPU tokenizer 及固定 Parquet，不加载模型，不重写原 AUTHOR_CPU_VERIFICATION。
2. 审查 `launch_gpu.py`、`run_gpu.py`、`runtime_contract.py`、`prepared/runtime_configuration_*.json` 和 IPC 检查；要求 root/freeze/device gate、完整模型哈希、native worker、greedy/EOS、guidance fallback、owned GPU guard、短 IPC 路径、持久独立缓存、无系统包更改。序列化兼容报告已复制纳入：变量只显式作用于本运行及其本地子进程，但启用通用 pickle/cloudpickle fallback，并非函数白名单；实际协议只传可信冻结 inspection callable、不接外部 payload。
3. 审查 `generation_receipts.py` 的固定 hash、字节复用与 runner diff；确认所有 diagnostics/science 的每个实际返回和所有 completion 先 fsync，再 count/order/token/identity/parse。缺失/多余/乱序/重复/未知 ID/多 completion/异常均不丢 raw，不自动重试，未知 compute 成本为 null。
4. 用新的输出目录运行 `check_generation_failure_retention.py --out <新目录>`；它执行实际 runner 配假引擎，不导入真实 torch/vLLM。审查 canonical→durable receipt 链及篡改拒绝。可独立添加相关反例。
5. 用新的输出目录运行 `check_runtime_full_chain_cpu.py --out <新目录>`。它按原 token IDs 执行三个模型各 2,260 个假调用，并核对 e-vector、原点估计、common12、完整 COUNT128/serial32、缺一目标弃权。输出均为 synthetic，汇总接口必须拒绝将其当真实结果。
6. 检查 `analyze_snli_runtime.py` 只从 mixed full-first common12 形成单模型 SNLI primary；原分析函数不改、旧结果独立保留。复制的根模块须为 179/180、tails 1/360/359/360、4,000 draws 及正确 case seed。`assemble_snli_primary.py` 只能聚合三个实际 SNLI component，不制造另六个。

正式独立报告请写为 `verification/INDEPENDENT_RUNTIME_REVIEW.json`，至少包含：

```json
{
  "status": "PASS",
  "reviewer": "independent reviewer identity",
  "candidate_manifest_sha256": "actual candidate hash",
  "runner_source_sha256": "actual run_gpu.py hash",
  "generation_receipts_source_sha256": "actual generation_receipts.py hash",
  "analysis_source_sha256": "actual analyze_snli_runtime.py hash",
  "checks": {
    "baseline_105_preserved": true,
    "receipt_retention_and_identity": true,
    "authoritative_primary_and_COUNT": true,
    "no_gpu_and_no_dependency_changes": true,
    "native_runtime_and_short_ipc": true
  },
  "tests_run": [],
  "issues": [],
  "real_model_judgments": 0,
  "GPU_execution": false
}
```

FAIL 或未证实之处请如实记载，不能改候选文件后仍声明原候选 PASS。若确需修正，保留旧候选及检查，用新的前瞻 amendment、新候选重新独立审查。

根任务确认独立报告后，运行 `python3 -B freeze_runtime.py --final --independent-review verification/INDEPENDENT_RUNTIME_REVIEW.json`。脚本先重新验证候选和原 105 文件，再把该独立报告和候选证据纳入最终 freeze。它不创建 GPU GO，不调用推理；最终授权仍由根任务给出。
