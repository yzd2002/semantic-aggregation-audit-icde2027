# 最终实际数据核验

36/36 原配置、16/16 预算策略及六条实际请求链完整核验通过；完整性错误为0。108个原候选位置全部交代，未调用候选不伪造结果。

原CURRENT_CP选中批大小分布：{1: 35, 8: 1}。Native shadow同决定/同调用成本 36/36；需要faithful follow-up 0/36，pending为0。

Current80-S的固定12格primary：严格向量联合净收益 0/12（unavailable 6）；原生布尔语义口径 0/12（unavailable 0）。两种定义透明并列，不改原校准门槛。

同一固定12格的三策略native净收益：RANDOM8 1/12（unavailable 0）；SIMILAR8 2/12（unavailable 0）；Current80-S 0/12（unavailable 0）。条件均为k>1、对native S1/S2两者dF≤.05、完整requests及raw tokens都低于S1。RANDOM8/SIMILAR8是无证书固定计划；这些有限面板描述性结果不外推所有计划。

预算配对共8组：320预算比80预算少用完整策略请求的有0组，少用完整原始input+output tokens的有0组。均包含各自校准与执行，不给预算赢家联合95%保证。

精确CP口径为每个固定预算内3候选分配delta/3：80次完整探针最多允许0次不一致；320次最多允许7次。所有提前停止仍按实际发出chunk的完整调用记账。

| 模型 | 工作负载 | 预算 | 两次选k | 平均完整请求 | 平均完整raw input+output tokens | 平均完整uncached input tokens |
|---|---|---:|---|---:|---:|---:|
| qwen | agnews_count_science_technology | 80 | [1, 1] | 7752.0 | 1366081.5 | 835822.0 |
| qwen | agnews_count_science_technology | 320 | [1, 1] | 8232.0 | 1514521.0 | 943147.0 |
| qwen | movie_count_positive_other_critics | 80 | [1, 1] | 7736.0 | 984666.5 | 456042.0 |
| qwen | movie_count_positive_other_critics | 320 | [1, 1] | 8152.0 | 1078693.0 | 515157.5 |
| glm | agnews_count_science_technology | 80 | [1, 1] | 7760.0 | 1318715.0 | 780163.0 |
| glm | agnews_count_science_technology | 320 | [1, 1] | 8304.0 | 1548843.0 | 960682.5 |
| glm | movie_count_positive_other_critics | 80 | [1, 1] | 7752.0 | 939208.0 | 401592.0 |
| glm | movie_count_positive_other_critics | 320 | [1, 1] | 8168.0 | 1041226.0 | 468026.0 |

固定primary 12格的完整交付、参考保真与资源读数：

| 模型 | case | k | native COUNT / G | native dF→S1 / S2 | strict完整 CP/S1/S2 | 完整calls / S1 | 完整raw tokens / S1 | strict / native净收益 |
|---|---|---:|---|---|---|---:|---:|---|
| qwen | agnews_count_world__n2048__cache1 | 1 | 1999 / 514 | 0.000000 / 0.000000 | false/false/false | 1.062500 | 1.124161 | unavailable / false |
| qwen | agnews_count_sports__n2048__cache1 | 1 | 1750 / 517 | 0.011230 / 0.009766 | false/false/false | 1.062500 | 1.124947 | unavailable / false |
| qwen | agnews_count_business__n2048__cache1 | 1 | 613 / 498 | 0.000977 / 0.001465 | false/false/false | 1.078125 | 1.183788 | unavailable / false |
| qwen | agnews_count_science_technology__n2048__cache1 | 1 | 483 / 519 | 0.003418 / 0.002441 | false/false/false | 1.062500 | 1.123679 | unavailable / false |
| qwen | movie_count_positive_top_critics__n2048__cache1 | 1 | 896 / 1251 | 0.001465 / 0.001953 | true/true/true | 1.062500 | 1.114045 | false / false |
| qwen | movie_count_positive_other_critics__n2048__cache1 | 1 | 986 / 1418 | 0.001465 / 0.001465 | true/true/true | 1.062500 | 1.113825 | false / false |
| glm | agnews_count_world__n2048__cache1 | 1 | 621 / 514 | 0.002441 / 0.001953 | false/false/false | 1.085938 | 1.213482 | unavailable / false |
| glm | agnews_count_sports__n2048__cache1 | 8 | 519 / 517 | 0.008301 / 0.008301 | true/true/true | 0.203125 | 1.030086 | false / false |
| glm | agnews_count_business__n2048__cache1 | 1 | 845 / 498 | 0.000977 / 0.000977 | false/false/false | 1.101562 | 1.250737 | unavailable / false |
| glm | agnews_count_science_technology__n2048__cache1 | 1 | 499 / 519 | 0.001465 / 0.000488 | true/true/true | 1.062500 | 1.124793 | false / false |
| glm | movie_count_positive_top_critics__n2048__cache1 | 1 | 1025 / 1251 | 0.002441 / 0.002441 | true/true/true | 1.062500 | 1.115711 | false / false |
| glm | movie_count_positive_other_critics__n2048__cache1 | 1 | 1145 / 1418 | 0.003906 / 0.003906 | true/true/true | 1.062500 | 1.115413 | false / false |

原SemBench Q3原方法在四次主运行中真实执行，完整输入关系均为120行：

| 模型 | prefix cache | native COUNT | strict COUNT | 源标签 G | native defaults |
|---|---|---:|---|---:|---:|
| qwen | False | 7 | 7 | 14 | 0 |
| qwen | True | 7 | 7 | 14 | 0 |
| glm | False | 9 | 9 | 14 | 0 |
| glm | True | 9 | 9 | 14 | 0 |

原生COUNT/源标签FP-FN/行错误与strict状态均保留；COUNT误差按N及G两个分母给出。G=0且预测0时error/G为0，G=0但预测非0时为null并标原因。

四条主运行成功、两条budget运行成功；另保留三份失败启动archive（主launcher共7次保留尝试），不把GLM的零generate说成零初始化工作。六条成功运行合计返回557996个native请求，单位为一个prompt/RequestOutput，绝非完整查询次数；S1/S2、校准refs/probes、执行、研究validation、controls、官方Q3与budget分账见NATIVE_REQUEST_USAGE_BY_RUN_AND_PURPOSE.csv。

实际GLM的动态模块仅包含已固定tokenizer（SHA-256 90a9a1b739f834cdf7a2f40959f9eeff2a705b80acefb2cea9c19ff3958a9279）；模型与配置没有动态模块，native类/参数数/BF16已核对。缓存开启策略每次真实BlockPool空缓存/PID记录均匹配原生日志；缓存关闭运行核对实际engine profile与返回cached字段。

原始执行日志中的NCCL/nanobind/refcount/EngineCore teardown诊断全部保留并列出时间证据。Qwen OFF在完整标记后出现这些诊断，scheduler记录launcher exit0；其他运行分别记录其实际时间关系及可用exit证据。它们与三份启动失败archive及缺失科学结果分账，不因含ERROR字样而改写已核验结果。

仍存的两项研究风险：

1. 系统范围与参考质量：实际使用LOTUS参考算子，批处理为明确自定义物理算子、CP为论文定义复现；两模型、两域不等于完整优化器广泛复现。参考质量也有限：primary S1最高默认率出现在qwen agnews_count_world__n2048__cache1，为1936/2048，native COUNT=1999、源G=514、源标签行错误=0.743652；高native fidelity不能替代源标签正确性。
2. 新认识的幅度仍需与近邻比较：参考变异、批处理扰动与校准开销已有前例；更大固定预算的功效/完整成本对照及可核验实例能补证据，但不能据可靠性、请求数量或经典统计控制本身预设新颖性与接收结论。

所有数值的可引用统一版本见 COMPLETE_VERIFIED_NUMBERS.json；本文件不修改论文或发布材料。
