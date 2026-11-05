# Parsing amendment after partial computation

The initial implementation printed the three WANDS results and then failed before producing a result file on the first ESCI JSONL record. Python `str.splitlines()` treats Unicode line separators occurring inside a JSON string as record boundaries. The original script and failure log are retained. The only intended correction is to split records on the JSONL LF byte represented by `\n`; the data, settings, comparison, and random seeds were not altered. The results remain post hoc and are not prospectively confirmatory.
