# 当前材料定位 / Asset map

## 当前论文与源文件

| 内容 | 路径 / 校验 |
|---|---|
| 十作者当前PDF | [paper/current/ICDE2027_Semantic_COUNT.pdf](paper/current/ICDE2027_Semantic_COUNT.pdf)，SHA-256 `bac7db7f7bd22377bc73818495f52ea374f8e10859289e6699662a3e0439bcd9` |
| 实测重建的clean source ZIP | [paper/current/paper_source.zip](paper/current/paper_source.zip)，SHA-256 `354f028e6f0c35b27b6eb48a600c2f041a257d94cf86613cddbdc9feb514b57f` |
| 活跃TeX与四图输入/源码 | [paper/current/source/](paper/current/source/) |
| 矢量PDF/SVG | [paper/current/figures/](paper/current/figures/) |
| 本版文件SHA清单 | [PUBLICATION_FILES_SHA256.json](PUBLICATION_FILES_SHA256.json) |

## 当前8模块科学负载

索引：[repro/real_engine_20261106/public/artifact_index.json](repro/real_engine_20261106/public/artifact_index.json)。分块目录：[repro/real_engine_20261106/public/parts/](repro/real_engine_20261106/public/parts/)。共177个分块、90,066,144字节；每个分块至多512KiB。索引列出每个part、合并压缩流和展开成员的实际字节数与SHA，part URL使用实际Git blob内容地址。8个模块合并重播36个main配置与16次独立budget policy执行。

| 模块 | 分块数 | 压缩流字节 | 内容 |
|---|---:|---:|---|
| foundation | 12 | 6,040,612 | 冻结源码、协议、输入身份/标签、数学接口与来源通知 |
| budget_run_glm | 17 | 8,537,352 | GLM独立预算扩展策略 |
| budget_run_qwen | 17 | 8,435,432 | Qwen独立预算扩展策略 |
| failed_attempts | 1 | 4,232 | 保留的失败尝试与实际返回账 |
| run_glm_cache0 | 30 | 15,375,080 | GLM缓存关闭配置 |
| run_glm_cache1 | 36 | 18,360,232 | GLM缓存开启配置 |
| run_qwen_cache0 | 29 | 15,192,852 | Qwen缓存关闭配置 |
| run_qwen_cache1 | 35 | 18,120,352 | Qwen缓存开启配置 |

额外公开计算入口：

- [post-hoc native-vector HG](repro/real_engine_20261106/posthoc/)：原作者脚本、bound结果/来源receipt及CPU重算。
- [stopping audit](repro/real_engine_20261106/stopping_audit/)：106+48个观察前缀，两份CSV与便携CLI。
- [readability evidence](repro/real_engine_20261106/readability_evidence/)：冻结Qwen/SciTech配对的真实停止轨迹、分阶段完整账单、两份CSV与重算脚本。
- [model sources](repro/real_engine_20261106/model_sources/)与[rights/notices](repro/real_engine_20261106/public/RIGHTS_AND_NOTICES.md)：模型/数据来源和权利界限，权重与源文本不在压缩包内。
- [verification/](verification/)：已完成的producer-side科学检查及本轮source重建证据；最终公开提交另作实际匿名取回与CPU验证。

## 六个legacy科学档案

[不可变v2 release](https://github.com/yzd2002/semantic-aggregation-audit-icde2027/releases/tag/v2026.11.06-final)和[BASELINE_V2_REFERENCES.json](repro/real_engine_20261106/public/BASELINE_V2_REFERENCES.json)保留六档的精确名称、URL、字节与SHA。它们承载历史实验与离线复现，本版不重复打包其字节。

历史根`asset_index.json`、旧SHA/checksum清单及v2论文记录继续保留为历史来源；当前稿使用`paper/current/`与新`PUBLICATION_FILES_SHA256.json`。请据各自版本的索引核对，保持当前8模块与legacy六档的范围分别清楚。
