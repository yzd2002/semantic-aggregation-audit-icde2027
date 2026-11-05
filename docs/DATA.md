# Fixed upstream inputs

The fixed upstream sources below are existing source links, not newly created
artifact-release URLs. Keep all downloads and reconstructed text outside Git.
Verify hashes before using a source. See the archive protocols for exact pool
construction, binary label mappings, truncation, and selection amendments.

| Source | Fixed location and verification |
| --- | --- |
| WANDS | [Publisher revision](https://github.com/wayfair/WANDS/tree/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5); `reproducibility/WANDS_DATA_SHA256.txt` identifies the used files. Preserve the MIT notice. |
| Original ESCI | [Publisher revision](https://github.com/amazon-science/esci-data/tree/7916cdf6ab75a462e77f20ab40428a10923998d5); `OFFICIAL_DATA_SHA256.txt` identifies the source files. Preserve Apache-2.0 and NOTICE. |
| Derived ESCI test file | [Pinned dataset revision](https://huggingface.co/datasets/nixiesearch/amazon-esci/tree/c092c898ccc036577f25a998fd429b84facfbd86); see `reproducibility/README.md` for conversion and file hashes. |
| TREC-COVID | [NIST collection](https://ir.nist.gov/covidSubmit/data.html) and the [fixed BEIR ZIP](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip). Expected SHA-256 `120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0`, MD5 `ce62140cb23feb9becf6270d0d1fe6d1`. Follow `NOTICE_RECONSTRUCT_TREC_SAMPLE.md` in the main archive before TREC analysis. |
| ACORD | [Pinned publisher revision](https://github.com/The-Atticus-Project/acord/tree/942b33a7c6fed7ead362c7c296401086f8bc0321), [source ZIP](https://raw.githubusercontent.com/The-Atticus-Project/acord/942b33a7c6fed7ead362c7c296401086f8bc0321/ACORD%20Dataset%20%26%20ReadMe.zip). Expected SHA-256 `c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3`. Obtain it directly and rebuild using the three reconstruction supplements. Do not add the downloaded ZIP or reconstructed text to this repository. |

The ACORD README describes dataset CC BY 4.0, while the repository LICENSE is
MIT. Those are separate publisher statements. Attribution is to the source
dataset and its [paper](https://doi.org/10.18653/v1/2025.acl-long.1206); the names
of this manuscript's authors must not be guessed from that upstream work.

## Models and tokenizer-only verification

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507): use the model and asset hashes in `MODEL_SHA256.txt`, `MODEL_QWEN_ASSETS_SHA256.txt`, and each preflight summary.
- [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674): fixed revision and hashes in the historical main/sensitivity records.
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28): the historical `posthoc_7b/` archive preserves model metadata, tokenizer and weight hashes, without the weights. This is not the unavailable later GLM study.
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3): see `MODEL_EMBEDDING_SHA256.txt`.

CPU tokenization checks require only the exact configuration/tokenizer assets
recorded by the applicable preflight, not weight loading. Do not use credentials
or model caches as release assets. Models and upstream datasets remain outside
the public repository and outside the selected release asset set.
