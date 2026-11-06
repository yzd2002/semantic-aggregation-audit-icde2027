# SNLI pre-inference runtime and primary-family amendment

本补充在 SNLI 尚无任何真实模型判断、尚无 SNLI GPU GO 时作出。它补齐运行和分析接口，不是对历史结果的重新预注册。原 `CPU_REVIEW_CANDIDATE_SHA256.json` 的 SHA-256 为 `e0c61d331338e544939ac28d3c656a61eee79659ab8a098d0f6bdef92fa93832`，所列 105 个原件全部逐字节保留。原协议、交接、ALPHA_PLAN、analyze.py 和 secondary.py 都保留为可核查的准备记录。

## 唯一权威的新 primary 规则

逐字节纳入根目录 `global_primary/` 的四个固定文件。该目录的 `GLOBAL_PRIMARY_AMENDMENT.md` 明确 supersede 原 SNLI 协议、CPU 交接、`prepared/ALPHA_PLAN.json` 和旧统计字段中的 **98.75% primary wording**。新 suite 的九项固定为 DBLP/BoolQ/SNLI × Qwen/Smol/GLM；本目录只提供三项实际 SNLI component，不生成另外六项。

每项只使用 **mixed、full-first、common12** 上 near-versus-random change 减同条件 replay 的逐目标 e-vector。名义 confidence 为 **179/180 = 99.444444…%**，分位数为 **1/360、359/360**；4,000 次 bootstrap，任务/模型 SHA-256 派生 seed 与根实现一致。严格正下界才满足该项 positive-primary criterion。缺失项仍属于九项 family，不可缩小 family 或替换 scope。空项 unavailable，不能当零。

`analyze_snli_runtime.py` 在验证 durable receipts 后调用原统计函数，点估计保持原定义；旧 98.75% 结果独立保留于 `legacy_main/analysis.json`，其阳性字段不再是本 suite 的权威结论。target-only、full-last、singleton、serial32、transport、correctness 和 COUNT 仍是次级。COUNT 主关系每条件固定 128，serial 固定 32；缺一个有效目标就 abstain，不能用 common12 的分母替代。bootstrap 是固定选定目标集的近似描述，不声称精确总体 coverage、精确 FWER 或因果效应。

## 固定请求与原生执行

每模型原请求为 mixed 1,792、serial 448、diagnostic 20，合计 2,260；三模型科学调用 6,720、诊断 60，合计 6,780。原 source、完整字段、gold、目标/背景、提案、prompt、grammar、token IDs、ID map、顺序、block 与 serial 复制均不更改。diagnostic 的分组来自原 request ID；仅补齐分组审计元数据，并明确普通无约束 BANANA 的 grammar hash 为 null，已有 guided hash 与原 grammar 重算核对。

原生 vLLM 模型为 Qwen3ForCausalLM、LlamaForCausalLM、ChatGLMForCausalLM，BF16，TP=1，4096 上下文，max_num_seqs=16，mixed 每次提交 64 请求、serial 每次 1 请求，同一 loaded engine 先诊断再 mixed 再 serial。采样维持 temperature=0、top_k=0、top_p=1、max_tokens=128、n=1；xgrammar 严格约束、关闭 fallback；各模型自身 EOS/stop 保持原配置。GLM 只信任固定本地 tokenizer 的代码；禁止 remote modeling/configuration 模块。开始判断前重验完整模型文件哈希，并在实际 worker 和每条 engine-processed request 记录所用原生实现与参数。

`generation_receipts.py` 原样复用已独立严格审查的 GLM v3，SHA-256 `a58f21a1596759a7903fb23468aa8a72a3f827ad51295f513d04cfe14be6b31d`。runtime_override 两个 Python 文件原样复用；它只替换已初始化单 rank NCCL world 中 ranks=[0] 的 CPU group transport，不伪造 collective 结果，异常通信应明确失败。源参考原件及 runner/launcher 的适配 diff 留在本目录，来源目录只读。

所有 blocking generate 的 invocation、processed engine ID、完整返回序列、每个 completion 的未裁剪文本和完整 token IDs，先 append/flush/fsync。随后才做返回数量、位置、engine 身份、完整 prompt tokens、grammar 和 parse 检查。位置候选在验证前明确为 UNVERIFIED；失败或多余返回保留原文，不归入科学条件。分析必须从 canonical row 验证至 durable receipt、processed binding、effective controls 和重新计算的身份验证结果。失配则停止权威分析。

未知、缺失、抛错或无法正确归属的 compute token 成本为 null，不计作零。raw-return accounting 是全部实际返回成本的权威，不能用已解析 canonical 行冒充所有返回。无自动重试、无自动恢复、无按输出改 prompt 或补样；异常后保留 FAILURE、所有已有 receipts 和成本证据，新的尝试须由根任务另行明确决定。

## 机器兼容性与授权

缓存各有本目录下持久独立子目录，不安装或改系统包，不删其他任务数据，不读取凭据。进程加载前要求指定物理 GPU 为空，每次提交前检查该卡上的进程均属本运行。只检查和使用根任务明确授权的卡。

本机 vLLM 默认从 TMPDIR 构造 ZMQ IPC 路径。根任务在其他运行的模型初始化阶段发现长路径超过本机 AF_UNIX 限制；因此在本 SNLI freeze 前显式设置原生 `VLLM_RPC_BASE_PATH` 为本目录下各模型独占的短持久路径 `rpc/q`、`rpc/s`、`rpc/g`。CPU 使用相同长度的 UUID socket 做真实 bind 核验，实际变量写入 START。该环境兼容修正不改采样、模型、科学请求或数据，不修改安装包。

根任务另在其他运行的 native worker inspection 阶段确认 V1 默认 msgspec 无法序列化函数。因此同样在 SNLI freeze 前，仅对该分配的本地运行进程显式设置原生 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，由其自己的本地 vLLM 子进程继承；START 记录该变量。**它对这些进程的 RPC codec 启用通用 pickle/cloudpickle fallback，并非单函数白名单。** 本协议实际只传递本目录冻结、可信的 `worker_identity` inspection callable，不从外部文件或网络读取序列化 payload，不启动公共 API，不把变量写入系统环境，不更改 generation、采样或模型。已逐字节纳入 `verification/CONTEXT_LOCAL_RPC_SERIALIZATION_REVIEW.json`：独立 CPU 审查确认 flag=0 拒绝函数、flag=1 对可信冻结函数 roundtrip 一致；本目录的函数 source hash 与该报告完全一致。这个兼容性 PASS 不替代 SNLI 的完整独立 CPU review。

最终 GPU launch 需要独立 CPU review 通过后产生的 `PREINFERENCE_FROZEN_SHA256.json`，以及根任务另行写入的 `GPU_GO_<model>.json`，其中须匹配 model、物理 device 与最终 freeze 哈希。假引擎测试里的授权只存在于清晰标注的 synthetic fixture，不是实际授权。真实模型判断在本补充准备时仍为 0。
