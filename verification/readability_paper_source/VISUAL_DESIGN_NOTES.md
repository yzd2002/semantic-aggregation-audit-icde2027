# 本轮视觉阅读与设计记录

本轮核对官方奖项/口头报告页面，并实际查看作者公开 PDF 的图和表；下列页码为 PDF 页码。论文预印本可能与会议最终版不同，奖项对应论文而非某一版排版。只借鉴组织和视觉原则，全部本稿图形由本稿实际 CSV 重新绘制，没有复制他人图形或数据。

| 参考论文 | 已视觉查看的内容 | 采用的原则 |
|---|---|---|
| ICLR 2026 Outstanding Paper: LLMs Get Lost In Multi-Turn Conversation | 主图及第8页 Table 1 | 先交代任务和关键现象，分组表头，选择性强调重要结果 |
| ICML 2026 Outstanding Paper / Oral: The Flexibility Trap | 第2页 Figure 1、Figure 2，第8页 Table 1 | 一眼可读的主结论，有限色彩，直接标注；Figure 1无abc，Figure 2有(a)(b)，说明标签取决于内容而非档次 |
| ICLR 2026 Honorable Mention / Oral: The Polar Express | 第2页 Figure 1 | 简洁线图、共同坐标语义、清晰图例、留白 |
| NeurIPS 2025 Best Paper: Gated Attention for Large Language Models | 主图及第4页 Table 1 | 清晰机制分区、分组横线表、少量重点粗体 |

截至本次核查，没有找到可核验的 NeurIPS 2026 获奖/Oral 完整结果。官网 Press 页面列出的最新既有奖项含2025，故明确采用2025获奖论文作补充，不把2025误写为2026。

## 本稿最终设计

- 四张图统一深蓝文字、蓝色基准/随机方案、棕橙相似批处理、紫色检查策略、青色保真结果。所有正式图提供 PDF/SVG，PDF 嵌入字体，无位图画布。
- 主图以“Check a candidate → Execute the full query → Account for both stages”讲清具体任务、决策、完整结果和总账。标题给出实测结论；80/80、99.17%、416 vs 2,048和+3.01%均来自冻结数据。成本条从零起点按真实比例绘制。T标签为示意，caption明确标注。
- 主结果图保留全部12个配置、完整范围及0–5%细节；浅色行带、共同模型分组、阈值线和空心回退标记帮助比较。没有删除不利结果。
- 预算图保持共同纵轴，分开执行和检查成本，直接给出总比值。缓存图以相同标记位置显示真实坐标，以填充/空心区别缓存状态；“更少raw、更多uncached”的着色区域只覆盖对应象限。
- 不使用多余(a)(b)(c)。每个视图用有含义的标题；caption保留读图顺序、对象和分母。
- 七张编号表统一booktabs横线、表头粗体、适度行距；只对分组和一个接受部署做轻量底色。数值、缺失标记、精度和范围保留，千位分隔用于提高阅读速度。
- 老师指定的摘要逐字保留。仅改必要caption、少量衔接文字及AI披露中图形协助的范围。八作者与ORCID链接不变。

## 核验

正文12页，参考文献第13页，AI披露第14页；没有附录、负间距或缩页边距。四图已从源码ZIP真实重画，论文已重新编译，并逐页比较文本和渲染。具体证据见同包FINAL_PAGE_AUDIT、NUMERICAL_INTEGRITY及source_rebuild报告。未新增模型实验，历史科学复现报告保留其原执行范围。

## 原始来源

- [ICLR 2026官方奖项](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/)
- [LLMs Get Lost in Multi-Turn Conversation](https://arxiv.org/pdf/2505.06120)
- [ICML 2026官方奖项](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/)
- [The Flexibility Trap官方Oral页面](https://icml.cc/virtual/2026/oral/71086)
- [The Flexibility Trap](https://arxiv.org/pdf/2601.15165)
- [ICLR 2026 Oral列表](https://iclr.cc/virtual/2026/events/oral)
- [The Polar Express](https://arxiv.org/pdf/2505.16932)
- [NeurIPS 2025官方奖项](https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/)
- [Gated Attention](https://arxiv.org/pdf/2505.06708)
- [NeurIPS 2026 Press](https://neurips.cc/Conferences/2026/Press)
- [ICDE 2027 Research Track要求](https://icde2027.github.io/cf-research-papers.html)
