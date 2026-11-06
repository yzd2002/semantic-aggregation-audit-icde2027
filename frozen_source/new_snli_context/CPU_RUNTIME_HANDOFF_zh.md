# SNLI runtime CPU 候选交接

本目录新增了可执行的原生 GPU runner、完整 raw-return 账本、严格分析链和最终 freeze 接口。**当前仍 NO_GPU_GO，真实模型判断 0；独立 SNLI CPU review 后才能由根任务授权运行。** 原 `CPU_HANDOFF_zh.md` 及其全部 105 文件保持原字节；原候选 SHA-256 为 `e0c61d331338e544939ac28d3c656a61eee79659ab8a098d0f6bdef92fa93832`。

科学调用计划不变：每模型 mixed 1,792 + serial 448 + diagnostic 20 = 2,260；三模型科学调用 6,720、诊断 60。未改变 source、完整 premise/hypothesis、gold、选样、背景、prompt、grammar、token IDs、顺序、block 或采样。原 analyze.py 和 secondary.py 不改。

`PREINFERENCE_RUNTIME_AMENDMENT_20261106.md` 明确 supersede 旧 98.75% primary wording。根目录 global_primary 的四个文件逐字节纳入；唯一权威新 primary 是 mixed/full-first/common12 的逐目标 excess，179/180 CI、tails 1/360 与 359/360、4,000 draws、九项 family。此目录只能提供三个实际 SNLI component，另外六个必须来自实际 context 运行。COUNT 主分母固定 128、serial 固定 32，缺任一个有效目标即对应条件/比较弃权。

运行组件采用已审查 GLM v3 的原生实现路径与采样、停止 token、GPU ownership guard。`generation_receipts.py` 逐字节复用，SHA-256 为 `a58f21a1596759a7903fb23468aa8a72a3f827ad51295f513d04cfe14be6b31d`；runtime_override 也逐字节复用。源参考、来源 hash 和适配 diff 均在 source_reference/ 与 verification/ 中。所有真实返回及每个 completion 先完整 append/flush/fsync，再验数量、位置、完整 token、身份和 parse；没有自动重试。未知成本为 null，不能用 canonical 已解析行代表全部返回。

本机技术兼容设置在推理前明确纳入：独占持久 `rpc/q`、`rpc/s`、`rpc/g`，实际 UUID endpoint 99 bytes，均通过 CPU AF_UNIX 与 ZMQ bind；per-process `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 由自身本地子进程继承，启用通用 pickle/cloudpickle fallback，并非函数白名单。实际协议只传可信冻结的 worker_identity，不接外来序列化 payload。独立兼容报告、函数 source hash 对齐证据已纳入；该报告不替代完整 SNLI review。

作者必要 CPU 检查的固定证据：

- `verification/CPU_RUNTIME_PREPARATION.json`：三个 native config、greedy/EOS/stop、实际安装包/源码身份；CUDA 未初始化，包未修改，继承的 pip conflicts 另记日志。
- `cpu_checks/runtime_source_recheck_final/RESULTS.json`：原 verifier 未修改地重跑，重新读取固定 Parquet、完整字段/gold、原生 tokenizer、全部原请求和 serial 复制；唯一报告写至新目录，原件不改。
- `cpu_checks/runtime_failure_retention_final/RESULTS.json`：17 个实际 runner + 假引擎反例，包括短/多/乱序/未知或重复 engine ID、错误 prompt、多 completion、无效 raw、parse 抛错、generate 抛错及空/None 返回；保存全返回和所有 completion、无 retry、未知成本 null，并拒绝 canonical 绑定/文本篡改。
- `cpu_checks/runtime_full_chain_final/RESULTS.json`：三个模型各自的原 2,260 调用完整假引擎链；另一个单目标无效 fixture 验证 common12=127 且 COUNT 仍固定 128 并弃权。所有输出都明确为 synthetic，真实 component 汇总接口会拒绝它们。
- `verification/AUTHOR_RUNTIME_CPU_CHECKS.json`：原 105 文件保全、四个 global 文件、源 diff、最终 runner hash、上述证据、语法以及缺少 GO 时 launcher 拒绝运行。

复审锁定 `RUNTIME_REVIEW_CANDIDATE_SHA256.json` 和 sidecar；详细步骤与正式报告 schema 见 `INDEPENDENT_RUNTIME_REVIEW_INTERFACE.md`。独立 PASS 报告须绑定候选 hash 和最终 runner/receipt/analysis hash。

完成独立报告后，由根任务执行：

```text
python3 -B freeze_runtime.py --final --independent-review verification/INDEPENDENT_RUNTIME_REVIEW.json
```

这会把候选和独立报告纳入不可替换的 `PREINFERENCE_FROZEN_SHA256.json`，不创建 GPU GO。根任务再按模型给出匹配最终 freeze hash 的 `GPU_GO_<model>.json` 和物理卡；授权完成才调用 `launch_gpu.py --model <model> --device <authorized_device>`。开始前 runner 会重验完整模型文件和冻结输入；缓存只写本目录持久独立子目录；不能影响其他卡或任务。

真实运行后，`analyze_snli_runtime.py --model <model> --out <新的结果目录>` 先验证 durable chain，再生成 legacy_main、legacy_secondary、成本和单模型 global_primary_component。`assemble_snli_primary.py --components <三个真实component路径> --out <新汇总文件>` 只生成三个 SNLI case；根任务需另提供六个 context case 才能组装完整九项 family。无真实 run-start 的目录不能产出新 primary case，任何 synthetic fixture 都不能进入真实汇总。
