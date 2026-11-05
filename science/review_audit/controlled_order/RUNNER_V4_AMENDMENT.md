# Pre-data guidance configuration correction, v4

The v3 engine loaded successfully with real singleton NCCL groups; no unexpected communication occurred. Before any data request, V1 rejected guided negative controls because its engine-level backend was auto while the explicitly requested backend was xgrammar. Installed processor.py requires the two to match.

The v4 runner explicitly initializes the engine with guided_decoding_backend=xgrammar and guided_decoding_disable_fallback=True, matching the original runtime_configuration intent. It immediately saves the ordinary BANANA control before submitting guided controls. The failed v3 attempt produced an ordinary synthetic control but did not persist its response before the validation exception; no scientific pool judgment occurred. That limitation and the failure log are retained.

The process-local adapter is unchanged except that diagnostic output goes to stderr and its event directory is located beside the adapter file. This avoids polluting JSON-producing Python subprocess stdout and improves portability; no inference or communication behavior is altered. The same v4 runner/adapter will be used for both models. All prompts, source pools, request order, code freezes and analyses remain unchanged.
