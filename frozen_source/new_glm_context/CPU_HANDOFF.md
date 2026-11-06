# CPU 审查交接

**v3 主口径更新优先：** global_primary/GLOBAL_PRIMARY_AMENDMENT.md 统一三模型×DBLP/BoolQ/SNLI 的九个新主比较，采用179/180（99.4444%）区间；旧98.75%输出仅作legacy对照。请先读 V3_HANDOFF.md。v2原文已完整保存在 initial_cpu_preflight/v2_failure_retention_baseline；后文旧记录不能覆盖此前向amendment。

本目录是唯一写入范围。所有数据与脚本均位于持久 `/home/gpu/IntentSat2S_revision_20261106/new_glm_context`。尚无本轮真实模型判断或 GPU 模型加载；合成结果只在明确标记的 cpu_checks 内。

主协议：PROTOCOL_fresh_three_family_context_replication.md。新设计是在旧 Qwen/Smol context 和 Qwen/Smol/GLM decision 结果已知后制定，且在本轮任何模型判断前扩大为三家族全新推理。不能称为旧研究的预注册。

## 已测得的 CPU 事实

全部 256 target、全部 near、完整字段及输入位置保留。原 random 中 222 个保留，34 个按固定 seed 2026110603 重抽，44 次有限 proposal 即全部成功；标签反转实际重跑得到相同 IDs 和完整尝试日志。三模型逻辑输入逐条一致，原 main/serial 日程保留。原 random 完整保留在次级 historical arm，其共享关系逐 call 冻结。

| 模型 | 主输入最大 token +128 cap | 全部阶段输入 token | 科学调用 | 诊断 | 主 EOS / stop set |
|---|---:|---:|---:|---:|---|
| Qwen | 1465 | 2793538 | 4750 | 20 | 151645 / {151643,151645} |
| Smol | 1526 | 2946673 | 4750 | 20 | 2 / {2} |
| GLM | 1439 | 2688080 | 4750 | 20 | 151329 / {151329,151336,151338} |

每个模型、每个任务、每个 scope 的 near/random 最大长度差均为 24，违规 0；完整 feature 最大 447 token，均未超过 512。每模型科学 item labels 为 12970；总 14250 科学调用、60 诊断、38910 科学 item labels、8428291 输入 token。请求上限为每次 128 output tokens，但有效 JSON 的分词上界为 17，停止 token 另计。精确逐阶段预算见 prepared/REQUEST_ENVELOPE.json。

GLM 当前本地完整 24 文件、18802687293 字节全量匹配官方固定 revision bd8234fe5e0c09c48637a92abb0c797cb5fa0e73。官方 metadata 为根任务本轮经 Firecrawl 取得的 HTTP200 API 响应；本任务重新计算每个本地文件的 LFS SHA256 或 Git blob SHA1。Qwen 当前 16 文件、Smol 当前 9 文件全部匹配原保留 manifest 的 SHA256。所有均为新观察记录，未重造丢失的历史证书。

CPU 已实际确认 native config、native registry、greedy 归一化和各模型停止集；torch.cuda.is_initialized() 为 false。analyze.py、secondary.py 逐字节未改；common.py 仅扩展 MODELS。旧算术/严格解析器/缺失输出检查通过；新增 4750+20 全链路合成检查、历史共享成本、serial 缺失保留、错误 prompt hash 拒绝也通过。旧98.75%判据仅保留legacy输出；当前九比较主摘要使用root提供的179/180区间与新固定case seed，仍不声称精确总体FWER。

## GPU 资源与仍待验证

最近检查 GPU0/1/2 均为空闲，其他卡不受本实验检查/退出逻辑影响。每模型独占根任务指定的一张卡，BF16、TP1、4096 context、max_num_seqs16、.70 显存、eager、无 prefix cache。5090 的 .70 总预算约 22.29 GiB。

仅由 safetensors header 和 config 计算：Qwen 权重 7.49 GiB、16×4096 的 BF16 KV 理论上界 9 GiB；Smol 权重 3.19 GiB、KV 上界 12 GiB；GLM 权重 17.51 GiB、KV 上界 2.5 GiB。这不是实际显存测量；框架开销、实际分配和 kernel 可行性须在授权后由 native runtime 收据验证。当前磁盘约 74GB 可用；已有三个独立持久 FlashInfer cache 副本，不改系统依赖或旧 cache。

历史保存的 Qwen mixed+serial 用时约 127+204 秒，Smol 68+138 秒；这只是旧记录，可为新预算提供参照，不是本轮实测。建议并行时预留 Qwen/Smol 各 10–15 分钟、GLM 20–30 分钟，包含加载和潜在编译；GLM 吞吐尚未测定。实际时间、显存、每个 engine-processed 请求的 native/sampling 控制及全部 raw 输出由 runner 记录，不按结果提前停止。

## 审查与授权入口

重点审查 rematch_contexts.py、run_gpu.py、launch_gpu.py、analyze_fresh_replication.py、prepared/REQUEST_ENVELOPE.json、verification/NCCL_ADAPTER_PATH_DIFF.patch 及 CPU 检查收据。adapter 仅改变事件日志 ROOT，其原件、diff、新 SHA 和静态审查都保留；所有 runtime cache 按模型隔离。

根任务完成静态审查后，分别写 GPU_GO_qwen.json、GPU_GO_smol.json、GPU_GO_glm.json。各文件需含 authorized_by="root"、model、device、freeze_sha256（最终 PREINFERENCE_FROZEN_SHA256.json 的 SHA256）。随后用 launch_gpu.py --model <model> --device <同一物理卡> 启动。启动器在导入模型运行时前验证授权；runner 再核对当前 freeze、包版本、已审阅 vLLM 文件、所有模型文件、native worker 和实际采样。它只检查分配卡上的其他进程并在发现冲突时停止提交，不终止任何外部进程。

本轮 GPU 模型加载、20 个真实技术诊断和科学推理均待明确 GO。任何失败须保留尝试，另做 append-only amendment 后再审查，不能静默重跑覆盖。
