# Independent clean GPU environment diagnostic

This post-hoc diagnostic follows observation of all earlier outcomes. Its aim is to test whether a separately resolved Python environment can run the existing synthetic vLLM format checks on the same machine. It is not a fresh-machine reproduction, benchmark, timing comparison, semantic accuracy test, or replacement for any earlier result.

Create a new Python 3.12 virtual environment with system site packages disabled. Place the environment, package cache, temporary files and model-runtime caches on a separate temporary filesystem, leaving the existing Python and GPU installations unchanged. Record the Python and installer versions and install only binary packages from the public PyPI and official PyTorch CUDA 12.8 package indexes, pinning the following observed key versions before resolving other dependencies:

```
vllm==0.10.2
torch==2.8.0+cu128
transformers==4.57.1
numpy==2.2.6
xgrammar==0.1.23
outlines-core==0.2.11
setuptools==79.0.1
pandas==2.3.3
pyarrow==22.0.0
zstandard==0.25.0
```

The first seven versions match the existing synthetic runner's required values. The additional three support the data-processing scripts; their presence does not establish reproducibility of every analysis. Retain the installation log, full resolved package-version list and dependency-check output, including failures. Record index URLs but do not record credentials, environment variables or login caches. The resolution may select other transitive versions than earlier runs; do not call it the original environment.

Only if installation, version checks and an independent dependency check succeed, use the unmodified `run_vllm_declared_deps_smoke.py` and the recorded local Qwen3-4B-Instruct-2507 model. The runner requires an idle GPU before loading, checks GPU process ownership during generation, and stops its own run if another process appears. It uses V1, the recorded settings, an unconstrained incompatible negative control, two constrained formats, and two unrelated synthetic product prompts; preserve all raw outputs and logs. Independently verify the saved records and tokenizer lengths with `verify_vllm_declared_deps_smoke.py`. Compare synthetic responses with the prior overlay run only descriptively; parsing success or matching text cannot establish semantic accuracy or a reproduction on different hardware.

Freeze this protocol, the original smoke runner and independent verifier, their imported prompt/parsing files, the pinned model manifest, and the earlier synthetic records before beginning installation or inference. Preserve failed attempts and never overwrite earlier outputs. If installation, dependency checks, controls or inference fail, retain the failure and report it rather than changing the test and calling it a pass.
