# Pre-inference runner correction, v2

The first Qwen launch failed before loading a model or generating controls/data: the local synthetic-smoke variable `metadata` shadowed the imported package-metadata module. The original frozen script, first log, and failure directory are preserved. No new model judgments were observed.

The v2 runner only renames the imported package-metadata module to `package_metadata` and verifies this additional frozen manifest. Inputs, schedule, grammar, prompts, controls, inference parameters, metrics and bootstrap definitions are unchanged. Both models will use v2. This is an implementation correction after selection, not a fresh experiment or a change of outcomes.
