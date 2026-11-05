# Qwen3 asset provenance recheck

After the experiments, selected local Qwen3-4B model files were compared with
the official Hugging Face model metadata captured on 2026-10-31 at the fixed
revision recorded in `official_file_metadata.json`. The nine selected small
files matched their official Git blob hashes or, for LFS files, their official
LFS SHA-256 values. The seven assets listed in the original frozen
`MODEL_QWEN_ASSETS_SHA256.txt` also matched their earlier recorded local
SHA-256 values. This includes `merges.txt` and `vocab.json`.

The three previously recorded weight SHA-256 values matched the official LFS
metadata and the local file sizes matched, but the large local weight bytes
were **not re-hashed in this follow-up**. This is a selected-file comparison,
not a claim that every file in the local model directory is identical to the
official repository. The original pre-run manifest is preserved unchanged.
The metadata JSON is a captured snapshot, not a signed attestation: to repeat
the provenance check, independently retrieve the official API URL stored in
that file and compare the pinned revision and relevant file metadata.

From the reproducibility directory, use a fresh output path:

```bash
python3 qwen3_official_asset_recheck/test_qwen3_official_assets.py
python3 qwen3_official_asset_recheck/verify_qwen3_official_assets.py \
  --metadata qwen3_official_asset_recheck/official_file_metadata.json \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --assets-manifest MODEL_QWEN_ASSETS_SHA256.txt \
  --weights-manifest MODEL_SHA256.txt \
  --out /path/to/new_verification_result.json
```

The included `verification_result.json` records this follow-up's outcome.
The test checks matching inputs, same-size byte tampering, and size changes.
