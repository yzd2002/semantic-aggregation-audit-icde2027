# 从这里开始

本目录服务于最终论文：**[Experiment, Analysis, and Benchmark] Auditing Reference Fidelity and Checking Costs in Semantic COUNT**。

这份源码树与本地压缩包仍是待发布候选；拟用版本号 `v2026.11.06-final`。只有发布者上传后、匿名下载并核对实际字节，再生成的 `asset_index.json` 才是正式版本记录。不要把仅对应历史稿的 `v2026.11.06-recovered` 填作最终论文复现版本。

## 包内内容

| 模块 | 提供内容 | CPU 检查 |
|---|---|---|
| `context.tar.xz` | 新 DBLP–ACM/BoolQ、三模型；269 项冻结输入/代码/协议，完整实际 raw、分析、控制与全部六次失败尝试 | 原统计程序重跑并逐文件比较 |
| `snli.tar.xz` | 新 SNLI、三模型；163 项冻结资产、完整实际 raw、分析及来源/许可 | 原统计程序与完整字段/原标签源核对 |
| `transfer.tar.xz` | 新真实 transfer；158 项最终冻结资产，完整实际运行与成本账 | 运行完成、有效参数证据及分析全部齐全才准入 |
| `decision_dependency.tar.xz` | 原 DEV 36 政策锁、实际校准 raw/token、必要源输入与离线 tokenizer | 重推完整 36 锁/eligibility ledger、实际 tokens；从源重建输入与 DEV prompts |
| `independent_audit.tar.xz` | 九个 context case 的独立数学/代理代码、证书与逐条结果；全局主检验定义 | 从归档实际 raw 独立重算六例及 SNLI 三例，再逐文件比较 |
| `independent_transfer_audit.tar.xz` | 新真实 transfer 的独立数学源码、必要纯函数 helper、两份真实证书与派生结果；独立于模拟评审评分 | 显式路径适配后重算实际 raw，逐文件比较政策、前缀、配置、逐 target 与成本 |

模块是否实际建齐以清单中的 `complete_artifact_modules_built` 为准。未完成的 transfer 不能用空值、合成样例或计划数冒充真实结果。压缩包内 `_artifact_manifests/` 给出逐文件 SHA-256/字节数；源码公开白名单另见 `SOURCE_WHITELIST_SHA256.json`。

## 四步复现

1. 先从匹配 release 的附件下载正式 `asset_index.json`，再获取本目录五个 `.py` 工具及必要压缩包。不可变科学 tag 可以只含 `asset_index.template.json`；实际验收后的 index 随 release 附上，main 的便利副本不改写 tag。源码树 `README.md` 有完整命令。
2. 用 `artifact_cli.py verify` 校验压缩文件，再用 `extract` 安全展开到不存在的新目录。
3. 用 `cpu_replay.py --bundle-root 展开目录 --out 新结果目录` 重算。可用 `--modules context snli` 等限定模块。独立九例核验同时需要 context、SNLI 和 independent_audit 三个模块；独立 transfer 核验同时需要 transfer 和 independent_transfer_audit。
4. 查看新结果目录 `CPU_REPLAY_REPORT.json` 及每个实际日志。成功必须来自本次运行的 `status: PASS` 和逐文件相等；保留的历史 PASS 不代替本次检查。

CPU 结果复放无需 GPU、模型权重、网络或访问原服务器目录；decision 输入/tokenizer 重建使用随包离线资产。运行于已安装的 Python 3.12.11 / NumPy 2.2.6；SNLI 源检查用 PyArrow 22.0.0。首次另建环境时可参考 `requirements-cpu.txt`，不要擅自升级现有环境。

原科学代码中保留的服务器绝对路径属于历史 provenance。生产分析通过明确的 `--bundle`/`--root` 路径或文件相对定位运行；独立 SNLI 代码通过单独路径适配器映射，映射写入本次输出。没有依赖 `/opt/filebrowser`、原 RAM 文件或丢失的路径。

## 证据与许可边界

独立核验是独立编写的数学/代理代码核验，未宣称外部人类评审或洁净机器 GPU 重跑。原 decision 包只覆盖最终论文所需 DEV 依赖，明确不声称补齐旧 73/242 项历史归档与缺失证书。完整旧 TEST 不是本次论文的结果模块。

请读 `RIGHTS_AND_NOTICES.md`：BoolQ 为 CC BY-SA 3.0，SNLI 为 CC BY-SA 4.0，DBLP 基础 benchmark 的官方 CC BY 4.0 证据与实际 DeepMatcher split 来源分别保留。没有把软件许可扩展到第三方数据，也没有替作者决定整仓 MIT 许可。GLM 保留 **Built with glm-4** 及对应条款。模型权重、缓存、凭据、ACORD/TREC 旧源文及其他任务材料均不在选择范围。
