#!/usr/bin/env python3
"""Compare selected local model assets with captured, revision-pinned HF metadata.

Large weight files are not re-hashed: their previously recorded SHA-256 values
are compared with the official LFS values, and local file sizes are checked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SOURCE = "https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507/revision/" + REVISION + "?blobs=true"
SMALL = {"LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
ASSETS = {"config.json", "generation_config.json", "merges.txt", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
WEIGHTS = {f"model-{i:05d}-of-00003.safetensors" for i in range(1, 4)}


def file_hash(path, algorithm, prefix=b""):
    h = hashlib.new(algorithm)
    h.update(prefix)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_asset(path, entry):
    size = path.stat().st_size
    if size != entry["size"]:
        raise ValueError(f"Unexpected size: {path.name}")
    local_sha256 = file_hash(path, "sha256")
    if "lfs_sha256" in entry:
        official_match = local_sha256 == entry["lfs_sha256"]
        comparison = "official_lfs_sha256"
    else:
        header = f"blob {size}\0".encode("ascii")
        official_match = file_hash(path, "sha1", header) == entry["git_blob_sha1"]
        comparison = "official_git_blob_sha1"
    if not official_match:
        raise ValueError(f"Official metadata mismatch: {path.name}")
    return {"name": path.name, "size": size, "local_sha256": local_sha256,
            "official_comparison": comparison, "official_match": True}


def read_manifest(path):
    result = {}
    for line in path.read_text(encoding="utf8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match:
            name = Path(match.group(2)).name
            if name in result:
                raise ValueError(f"Duplicate manifest file: {name}")
            result[name] = match.group(1)
    return result


def run(metadata, model, assets_manifest, weights_manifest):
    data = json.loads(metadata.read_text(encoding="utf8"))
    if data.get("revision") != REVISION or data.get("source_url") != SOURCE:
        raise ValueError("Unexpected source or revision")
    entries = data["files"]
    by_name = {e["name"]: e for e in entries}
    if len(by_name) != len(entries) or set(by_name) != SMALL | WEIGHTS:
        raise ValueError("Unexpected official metadata file list")
    for entry in entries:
        if not isinstance(entry["size"], int) or entry["size"] < 0:
            raise ValueError("Invalid metadata size")
        if not re.fullmatch(r"[0-9a-f]{40}", entry["git_blob_sha1"]):
            raise ValueError("Invalid Git blob digest")
        if "lfs_sha256" in entry and not re.fullmatch(r"[0-9a-f]{64}", entry["lfs_sha256"]):
            raise ValueError("Invalid LFS digest")
    expected_assets = read_manifest(assets_manifest)
    expected_weights = read_manifest(weights_manifest)
    if set(expected_assets) != ASSETS or set(expected_weights) != WEIGHTS:
        raise ValueError("Unexpected frozen-manifest file list")
    small = [check_asset(model / name, by_name[name]) for name in sorted(SMALL)]
    for asset in small:
        if asset["name"] in ASSETS:
            asset["matches_frozen_local_sha256"] = asset["local_sha256"] == expected_assets[asset["name"]]
            if not asset["matches_frozen_local_sha256"]:
                raise ValueError(f"Frozen local digest mismatch: {asset['name']}")
    weights = []
    for name in sorted(WEIGHTS):
        entry = by_name[name]
        if (model / name).stat().st_size != entry["size"]:
            raise ValueError(f"Weight size mismatch: {name}")
        if expected_weights[name] != entry.get("lfs_sha256"):
            raise ValueError(f"Recorded weight digest differs from official LFS: {name}")
        weights.append({"name": name, "size": entry["size"],
                        "recorded_sha256_matches_official_lfs": True,
                        "local_size_matches": True, "rehash_performed": False})
    return {"revision": REVISION, "metadata_source": SOURCE,
            "metadata_is_captured_snapshot": True, "verified_small_files": small,
            "weights_recorded_hash_comparison": weights,
            "scope": "Selected assets only; existing local files are not modified. Weight bytes are not re-hashed. The official metadata snapshot should be independently refreshed before relying on it."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.metadata, args.model, args.assets_manifest, args.weights_manifest)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Verified {len(result['verified_small_files'])} small files and compared {len(result['weights_recorded_hash_comparison'])} recorded weight hashes. Output: {args.out}")


if __name__ == "__main__":
    main()
