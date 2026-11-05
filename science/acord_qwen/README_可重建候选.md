# ACORD 无合同原文的可重建复现候选

此候选包不直接包含 ACORD 源归档、条款文本、查询原文、原始评分表或按查询列出的参考标签。它包含方案、采样／提示／分析代码、汇总结果及模型生成的 Y/N 响应。响应中的池 ID 已替换为固定前缀加原 ID 的 SHA-256；这是可从公开数据字典反查的**伪名，不是匿名化或隐私保证**。提示哈希、聚合统计和模型输出可能仍涉及第三方权利；作者必须在公开前审核上游及输出内容、代码授权、隐私和适用许可。本包未被发布，也未自动取得授权。

## 固定上游来源

使用作者仓库修订 `942b33a7c6fed7ead362c7c296401086f8bc0321` 的 `ACORD Dataset & ReadMe.zip`，Git blob 为 `6de8c6c01a5221fe7669bc3117cfc02e5bc02fbc`，文件 SHA-256 为 `c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3`。固定文件地址：

https://raw.githubusercontent.com/The-Atticus-Project/acord/942b33a7c6fed7ead362c7c296401086f8bc0321/ACORD%20Dataset%20%26%20ReadMe.zip

来源论文：https://doi.org/10.18653/v1/2025.acl-long.1206 。数据集 README 声明 CC BY 4.0，仓库代码 LICENSE 为 MIT；底层合同内容可能另有权利条件。请自行获取并核对哈希，不要把本候选包理解为原始合同的再发布许可。模型 Qwen3-4B-Instruct-2507 和 tokenizer 的固定资产哈希见预检摘要，权重清单仅记录原实验使用的哈希，不含权重。

## 重建、复核，不进行新推理

需要主复现包中的 `02_reproducibility/`、原记录使用的 Python、numpy 2.2.6、transformers 4.57.1、tokenizers 0.22.0，以及匹配哈希的本地 Qwen tokenizer 文件。先下载固定上游 ZIP，并将下面路径改为真实路径；新工作目录不能预先存在。当前程序不会下载或加载模型权重，不请求 GPU，也不会覆盖旧输出。

```bash
cd /path/to/acord_reconstructable
sha256sum -c SHA256SUMS.txt
export INTENTSAT_PYTHON=/path/to/compatible/python
bash rebuild_from_upstream.sh /path/to/ACORD_upstream.zip /path/to/Qwen3-4B-Instruct-2507 /path/to/02_reproducibility /path/to/new_work_directory
```

脚本从上游重新生成 600、1,200、1,800 和 2,400 码点输入，检查三个前缀的 4,560 条 tokenizer 预检及其哈希，依据固定规则重算选择；用本地重建的原始池 ID 将伪名响应复原，并对每个恢复文件核对执行前保存的哈希；最后独立重算主要分析和全量逐条补充分析、核验严格解析、调用与统计。原模型结果无法因发布伪名而改变，恢复得到的响应文件与私有实验原件逐字节一致。主分析与补充分析的哈希保存在 `RESTORE_SHA256.json`。

原协议在观察此前结果之后、ACORD 选池和推理前形成；两次前缀修订在查看数据长度后、任何 ACORD 推理前确定，不是公开预注册。补充逐条指标脚本在推理过程中、查看候选预测和完成逐条前补写，不改变主要比较。两场景共享查询及部分条款，压力池依参考评分构造；来源评分包含扩展负例，存在截断、已标注候选偏差和未知模型预训练暴露。统计区间为未校正、描述性查询 bootstrap；不能据此推断总体性能或法律实用性。
