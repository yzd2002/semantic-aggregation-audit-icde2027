# 保存输出的来源与解释

公开复现材料时，请将本说明与主复现包及各补充包一同提供。早期商品实验的生成包装在保存之前对输出文本调用 `.strip()`；因此历史输出文件中名为 `raw`、`first` 和 `final` 的字段，是解析前但**已去除首尾空白**的字符串，而非可证明逐字节等同于模型返回的原始文本。未保存的首尾空白无法从这些记录恢复。

后续 TREC-COVID 和 ACORD 的键控、逐条执行分别保存并直接验证模型返回的原始输出字符串。LOTUS 过滤对照的 `raw_output_lotus` 由该接口单独返回；不能把早期商品包装的空白处理自动套用到这一对照。各项结果按各自运行时的实际解析规则计算，本说明不改变模型预测、完成率或统计。

对 500 查询验证的保存文件检查了 13,102 个 `first`／`final` 字符串，未发现首尾空白；这与保存前去空白的代码一致，但不能证明模型原始返回没有空白。检查方法和可执行脚本见同目录 `README.md`、`recorded_response_provenance_check.py` 和 `response_provenance_check.json`。不要将历史协议里要求保留原始输出的文字误读为已保存未经处理的字节。

## 英文复制说明

```text
In the early product-study runners, model response strings were stripped of leading and trailing whitespace before being saved. Fields named “raw”, “first”, or “final” in those records are therefore pre-parser strings, not necessarily verbatim model output; the removed whitespace cannot be recovered. The TREC-COVID and ACORD keyed and pointwise runs instead recorded and validated the untrimmed output strings. Reported metrics use each run's actual parser and saved records.
```
