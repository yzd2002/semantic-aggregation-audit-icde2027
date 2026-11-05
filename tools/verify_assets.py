#!/usr/bin/env python3
"""Verify the closed selected asset set, hashes, sizes and ZIP CRCs offline."""
import argparse
import json
import zipfile
from pathlib import Path
from build_asset_index import hash_file


def verify(asset_dir, index_path):
    index = json.loads(index_path.read_text())
    names = [x["name"] for x in index["assets"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate asset names in index")
    allowed_extras = {"SHA256SUMS.txt", "asset_index.json", "ASSETS.md"}
    actual = {f.name for f in asset_dir.iterdir() if f.is_file()}
    missing = set(names) - actual
    extras = actual - set(names) - allowed_extras
    if missing or extras:
        raise ValueError("asset set mismatch: missing=" + repr(sorted(missing)) + "; unlisted=" + repr(sorted(extras)))
    for item in index["assets"]:
        f = asset_dir / item["name"]
        if f.is_symlink() or f.stat().st_size != item["bytes"] or hash_file(f) != item["sha256"]:
            raise ValueError("asset identity mismatch: " + item["name"])
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as archive:
                if archive.testzip() is not None:
                    raise ValueError("ZIP CRC mismatch: " + item["name"])
        print("PASS", item["name"], flush=True)
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path(__file__).resolve().parents[1] / "asset_index.json")
    a = p.parse_args()
    index = verify(a.asset_dir, a.index)
    print("PASS", len(index["assets"]), "asset identities and ZIP CRCs; no inference or network used")


if __name__ == "__main__":
    main()
