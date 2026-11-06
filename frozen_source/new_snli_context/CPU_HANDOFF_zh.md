# SNLI 新任务 CPU 交接

结论：许可、固定来源、字段与确定性选样可行；已准备 128 个完整 premise/hypothesis 目标，三个 tokenizer 全部通过。**目前 NO_GPU_GO，新模型判断 0，未加载模型权重、未使用 CUDA。** 本目录是待独立审查的 CPU 候选，尚不是最终推理冻结版本。

来源采用 `stanfordnlp/snli` 的固定提交 `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`。原论文确认 Attribution-ShareAlike，固定仓库卡明确 CC BY-SA 4.0；正文与分发数据保留出处、许可链接及二值标签改动说明。完整 test=10,000，实际有效标签 9,824/无标签 176；train=550,152，实际有效 549,367/无标签 785。没有使用 validation。第三方 hf-mirror.com 传输的 Parquet 已匹配固定 Git LFS SHA-256；原 Stanford ZIP 本机不可达，**没有声称与原 ZIP 做过字节比对**。该 Parquet 不提供原 pairID 或逐标注者投票，使用固定 split/行号 ID，不能评估标注者共识置信度。

筛选使用 seed 2026110611，先排无标签/无效字段，再检查所有完整字段在三个 tokenizer 下 ≤512，未发生长度排除或截断。预留 256 个无重复完整句子字段的 test 对；训练集排除所有 reserve 共享字段 934 条、规范化完整 pair 重复 649 条，余 547,784 条，再按 ID hash 截至 10,000。TF-IDF near3 与 seeded random3 匹配，最终 128/128 个目标全部可行。类别整体循环扰动后，从新 Parquet 重载并重跑选样，目标/背景/所有提案/提示/主与串行顺序均完全相同。

作者机械核验重新读取原始 Parquet，核对所有选定字段不被更改，重新原生编码保留库和 reserve 的全部字段，并核对所有请求、grammar、局部 ID1/首位置、目标前缀、full-first/full-last 提示字节与 token 相等、serial 完整复制及逐调用计划。独立审查仍待安排。

| 模型 | 主实验最大输入 token | 含128输出上限 | near/random 最大差 | 主/串行/诊断调用 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 323 | 451 | 24 | 1792 / 448 / 20 |
| SmolLM2-1.7B | 342 | 470 | 24 | 1792 / 448 / 20 |
| GLM-4-9B | 317 | 445 | 24 | 1792 / 448 / 20 |

合计 6,720 次科学调用 + 60 次诊断 = **6,780 次**；计划输入 1,709,911 token，科学 Y/N 输出槽位 18,240、诊断约束槽位 153，另有 3 个无约束 BANANA 输出。以上是请求计划，不是实测输出 token 或有效答案。每个模型 20 个诊断来自 1 普通 + 5 不兼容指令 grammar + 14 合成格式请求。主实验每模型 4,864 科学槽位，serial 为1,216。初始逐调用账本的所有项均为 not_submitted。

primary 仍是每模型 full-first replacement excess over replay，共3个新primary。均预声明98.75%双侧 percentile 区间，下界>0才符合阳性criterion；单项名义α=.0125，新family Bonferroni总上界.0375≤.05。未将.0125再除3，也不与旧任务合并声称统一FWER。4,000次 bootstrap 是选定目标集上的近似描述，不保证总体/因果覆盖。target-only、full-last、serial32、参考资格/transport、准确性和 COUNT 为预声明次级。COUNT 主分母固定128、serial固定32；背景标签不计入关系，缺任一目标即该条件/比较弃权，不能改分母。

原类别仅在 feature-only 选样和扰动检查后附加：47 entailment、39 neutral、42 contradiction，二值 gold Y=47。没有据此重采样。尚无任何模型表现、任何 primary 效应或成功证据；不据准备完成推断审稿等级或“Strong Accept”。这扩展到完整句对蕴含谓词，但仍受公开老基准、未知预训练暴露、标注噪声和小型固定目标集限制。

关键文件：

- `PROTOCOL_snli_context.md`：前瞻协议、alpha、质量、COUNT、诊断和成本定义。
- `sources/SOURCE_LICENSE_SCHEMA_VERIFIED.json`、`SNLI_DATA_LICENSE_AND_ATTRIBUTION.txt`：固定来源与许可。
- `selection/targets.jsonl`、`reserves.jsonl`、`background_bank.jsonl`、`random_proposals.jsonl`：完整选样与提案。
- `selection/GOLD_PERTURBATION_CHECK.json`：执行过的类别扰动不变性检查。
- `prepared/*_main_requests.jsonl`、`*_serial_requests.jsonl`、`*_diagnostic_requests.jsonl`：三个模型的完整输入、token IDs、scope/grammar keys。
- `prepared/ALPHA_PLAN.json`、`GOLD_MAPPING_AND_QUALITY_PLAN.json`、`PLANNED_COSTS.json`、`planned_call_ledger.jsonl`。
- `prepared/AUTHOR_CPU_VERIFICATION.json`：作者机械检查；`cpu_checks/no_inference_*`：空响应边界核验，未推理。
- `prepared/MODEL_FILE_SHA256.json`：透明复用并行GLM准备阶段的当前模型证据；本SNLI阶段没有重复扫描权重，tokenizer/config与该哈希已核对。未来推理runner仍需在判断前核对完整权重哈希。
- `analyze.py`、`secondary.py`：沿恢复context统计实现，附来源hash；任务/模型常量来自本地 common.py，新family声明已补齐。

余下门槛：独立CPU审查；指定空闲GPU与兼容性审查；推理runner及所需adapter的最终代码冻结；包含审查和runner的最终推理前hash清单；根任务明确GPU GO。本轮不启动GPU，也未修改其它代理目录。当前环境的已继承 pip 依赖冲突另记于 `prepared/pip_check.log`，没有修改安装包。
