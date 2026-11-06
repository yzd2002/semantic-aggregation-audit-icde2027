# Runtime amendment candidate：须由 root 应用并由其他代理独立最终审查

状态：仅 CPU 候选，未改原95冻结目录，未创建 GPU release，未运行GPU。14个真实调用候选runner的CPU fake-engine fixtures已通过，包括native类/dtype错误、effective greedy/EOS/grammar/prompt错误、返回engineID/prompt错误、少/多返回、多completion、invalid raw。两个正常模型路径还自动执行了新增的离线effective回放验证。所有输出标明CPU SYNTHETIC。

## 应用文件

替换根目录 `run_gpu.py`、`launch_gpu.py`；新增根目录 `runtime_observer.py`、`verify_effective_runtime.py`、`check_runtime_amendment.py`；复制 `runtime_observation_contract.json` 到 `prepared/runtime_observation_contract.json`。checker已优先读取prepared下的正式contract，在本候选目录没有prepared时才回退读取同目录候选contract；应用后不需要根目录额外副本。

原始 `LLM`、`SamplingParams`、`GuidedDecodingParams` 构造调用经 AST 检查完全不变。所有模型文件、输入、36政策、分组、日程、输出grammar和分析数值不变。AUTO model_impl请求保留，native实现根据实际worker类/module/BF16参数验证。没有新增模型判断或新的calibration。

保留旧95-file freeze、原run/launch源、所有旧CPU报告和本次baseline95独立证据；新技术amendment明确新时间，不能回填旧历史。旧报告的旧source hash仍是历史事实，不可当成新runtime测试。

## 本次新增保障

原生身份先落盘再断言；每条processor有效控制含engineID、输入token hash、grammar hash、greedy/top_p/top_k/max_tokens/EOS/guidance落盘；原始return_journal先fsync整批所有实际返回，再按engineID/prompt绑定到逻辑ID。绑定包含原journal字节段位置、长度、SHA-256，避免反复读整个增长文件。成功末尾自动离线核验这些记录后才运行原分析；失败保持原raw、部分canonical及观察状态，绝不重试。

缓存全部分到 `runtime_cache/{qwen,smol}`；短IPC路径 `/home/gpu/i2sr1106/tq`、`/home/gpu/i2sr1106/ts`；VLLM_ALLOW_INSECURE_SERIALIZATION=1只在实验进程及其本地children生效。这开启通用pickle/cloudpickle fallback，不是单函数白名单；实际传送的是root审过并冻结的纯inspection callable。START和actual runtime均记录白名单环境字段，不记录凭据。原runtime_configuration.json的共享FLASHINFER路径只作为baseline95历史记录，该缓存路径被本次明确runtime contract取代。适配器仍以各PID独立事件文件写入共享adapter_events目录。

观察器逐条持久化会加入CPU/IO开销，可能影响请求抵达引擎的时间；不能因此声称和未插桩版本有位级相同输出或把混合研究运行时间说成孤立部署延迟。逻辑输入/政策/模型/解码构造保持一致，新执行结果必须全部如实报告。

## 冻结和审查

不要直接套用旧 freeze.create 的旧6-case门槛。旧 `freeze.check()` 可校验新的manifest与科学文件inventory；root应基于本次实际新文件建立新manifest，并绑定本次14-case报告、constructor AST、旧95-manifest和独立最终审查。原37个纯数值synthetic tests可在新树重跑；其报告包含所有.py的hash，不能沿用旧报告冒充新版。必须另外安排未参与本runtime候选实现的代理做独立最终CPU审查，PASS和最新freeze绑定后，才可root写新GPU_RELEASE。

可复制CPU检查命令（输出目录必须尚不存在）：
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -B check_runtime_amendment.py --base-science-root /home/gpu/IntentSat2S_revision_20261106/new_transfer_audit --out /new/CPU_SYNTHETIC/output`

现有本地候选验证：`cpu_fixture_results_v3/RUNTIME_AMENDMENT_CPU_CHECK.json`。v1是加入自动离线回放gate前版本；v2验证加入该gate的runner；v3额外修复checker应用到正式树后的prepared-contract路径，并绑定离线verifier源码。前三版报告都保留，没有改写历史。v3绑定当前run/launch/observer/contract/checker/verifier。`verify_effective_runtime.py`由v3正常runner实际调用，两正常模型均PASS；最终candidate manifest另绑定其字节。

当前CPU checker为候选编写者自己的检查，不声称是对候选的独立审查。baseline95独立核心证书在父目录BASELINE95_CPU_REVIEW_CERTIFICATE.json，只覆盖95旧版科学逻辑及真正独立重建，不能作为新版GPU PASS。

`TRUSTED_CALLABLE_CPU_CODEC_CHECK.json` 还实际调用候选worker_identity读取CPU dummy model，并用本机真实Msgpack codec复核flag0拒绝/flag1可信函数往返，torch.cuda.is_initialized保持False。
