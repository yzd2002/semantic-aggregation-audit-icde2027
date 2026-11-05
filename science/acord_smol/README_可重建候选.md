# ACORD 第二模型无原文复现候选

本包依赖相邻的 `ACORD_无原文可重建复现候选.zip`：先按其说明从固定上游 ACORD 归档重建选池、Qwen 预检和已保存响应。本包不直接包含合同／查询原文或上游评分，包含第二模型生成的 Y/N、可反查的池 ID SHA-256 伪名、汇总结果、协议和代码。伪名不是匿名化；模型输出、短片段、数据重建方式和第三方权利仍需作者审核，本候选包尚未公开。

此 SmolLM2-1.7B-Instruct 敏感性检查在观察 Qwen 与同一公开数据结果之后、Smol 推理之前固定；两模型使用同一候选池和提示文本，但模型、tokenizer 和聊天模板不同。它不是新的独立样本，不支持架构因果解释、生产时延或法律实用性结论。原始方案、输出解析、负控、全部已完成请求、失败和实际修改历史均在包中记录；第一次启动因误用 Qwen 的预检哈希被保护性拒绝，未加载模型或预测，修正后才运行。`prior_attempt/` 保留失败日志、原代码及当时哈希。报告中的 bootstrap 是未校正的描述性查询重采样。

模型固定修订为 `31b70e2e869a7173562077fd711b654946d38674`；模型目录应命名为 `SmolLM2-1.7B-Instruct`，完整权重无需参与离线重建，tokenizer 和 config 文件须匹配 `preflight_summary.json` 中的哈希。原运行权重 SHA-256 记录在 `SMOL_WEIGHT_SHA256.txt`，模型卡及来源为 https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674 。

先重建 Qwen 候选包，再使用原主复现包的 `02_reproducibility`、与记录一致的 Python/numpy/transformers/tokenizers 和固定 Smol tokenizer。所有路径自行替换；新的输出目录必须不存在。该过程只读取已保存响应，不重新调用 GPU 或模型权重：

```bash
cd /path/to/acord_smol_reconstructable
sha256sum -c SHA256SUMS.txt
export INTENTSAT_PYTHON=/path/to/compatible/python
bash rebuild_smol_followup.sh /path/to/qwen_reconstructed_work /path/to/SmolLM2-1.7B-Instruct /path/to/02_reproducibility /path/to/new_smol_work
```

脚本重算全部 4,560 条 Smol tokenizer 预检，按本地重建的池 ID 复原响应并逐文件核对原始哈希；重算并独立核验单模型结果、全量逐条补充和共享查询上的两模型配对差异。`RESTORE_SHA256.json` 存有各项原始哈希，`PRE_INFERENCE_SMOL_SHA256.json` 为运行前本地记录。该流程在原服务器环境核验，不证明另一台机器已完成安装与模型推理。
