# 从这里开始：当前论文与可复现材料

当前论文：**[Experiment, Analysis, and Benchmark] Auditing Batched Semantic COUNT: Fidelity Checks and Full Policy Cost**。论文已包含九位作者，14页由12页正文、1页参考文献和1页AI披露组成。

- [当前PDF](paper/current/ICDE2027_Semantic_COUNT.pdf)
- [clean source ZIP](paper/current/paper_source.zip)与[可编辑TeX](paper/current/source/draft/main.tex)
- [四张矢量图](paper/current/figures/)与[源码重建核验](verification/readability_paper_source/)
- [本版文件校验表](PUBLICATION_FILES_SHA256.json)与[材料定位表](ASSETS.md)

本版把采样批处理保真检查连接到完整COUNT结果与完整策略账单，提供36个原始配置、16次预算扩展执行及实际响应、分区、标签和成本证据。

## 一次下载，按顺序CPU重算

在GitHub选定本次不可变提交，下载该提交的完整仓库压缩包并解压。177个科学分块已在仓库中；这样无需逐块请求匿名API。使用Python3.12、NumPy2.2.6、SciPy1.16.2，从仓库根目录依次执行：

```bash
python3 -B repro/real_engine_20261106/public/replay_artifact.py --index repro/real_engine_20261106/public/artifact_index.json --parts-dir repro/real_engine_20261106/public/parts --out fresh_core
python3 -B repro/real_engine_20261106/public/replay_native_sensitivity.py --bundle fresh_core/expanded --supplement repro/real_engine_20261106/posthoc/supplement --out fresh_native_HG
python3 -B repro/real_engine_20261106/stopping_audit/analyze_observed_rejections.py --root fresh_core/expanded --main-summary fresh_core/cpu_replay/engine/summary.json --budget-summary fresh_core/cpu_replay/separate_author_budget_analysis/summary.json --output fresh_stopping --tex-output fresh_stopping/tex
python3 -B repro/real_engine_20261106/readability_evidence/recompute_frozen_evidence.py --root fresh_core/expanded --out fresh_readability_evidence
```

四步依次覆盖：8模块core的36+16数学与成本；已有原生Boolean向量的post-hoc HG；154个实际观察到的候选前缀；新稿解释用的停止轨迹与精确账单。最后两步分别生成两份CSV，均应与对应公开目录内CSV字节一致。它们读取冻结结果，不启动模型或产生新实验。

`bash paper/current/source/build.sh`用普通pdflatex及随包矢量图编译。可选重画入口`redraw_figures.py`读取随包CSV，并保留当前作者块和编辑后caption。交付前已真实解压、重画四图并重编译，逐页文本与渲染内容一致。

## 版本与权利边界

当前论文、源码统一在`paper/current/`。当前科学索引是`repro/real_engine_20261106/public/artifact_index.json`；当前仓库文件清单是根目录`PUBLICATION_FILES_SHA256.json`。根目录旧`asset_index.json`及旧SHA表只对应历史v2，不能用它们校验本版README或新稿。

旧[v2026.11.06-final发布](https://github.com/yzd2002/semantic-aggregation-audit-icde2027/releases/tag/v2026.11.06-final)保留六个legacy科学档案，供历史实验与离线复现使用。当前新增科学包有8模块、177分块，共90,066,144字节；详见[ASSETS.md](ASSETS.md)。

原生交付、严格格式、参考保真、源标签COUNT与完整成本分别报告。六个工作负载属于两个文本领域；预算扩展没有增加独立领域。停止前缀仅描述实际发出的比较，未触及候选保持未观察状态。数据文本、完整token数组和模型权重不随紧凑包重分发；固定源获取/重建方案与上游权利声明见[scientific README](repro/real_engine_20261106/public/README.md)和[rights/notices](repro/real_engine_20261106/public/RIGHTS_AND_NOTICES.md)。
