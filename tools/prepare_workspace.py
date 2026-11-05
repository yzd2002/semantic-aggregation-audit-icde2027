#!/usr/bin/env python3
"""Verify and safely extract the selected archives into a fresh workspace."""
import argparse
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from verify_assets import verify
from build_asset_index import hash_file


def compose_reproduction_inputs(workspace):
    """Map an unchanged delivered supplement input to its historical location."""
    source = workspace / "midpoint_ratio_supplement/routing_budget_result.json"
    target = workspace / "reproducibility/posthoc_routing_budget/routing_budget_result.json"
    consensus = json.loads((workspace / "review_audit/consensus/consensus_audit.json").read_text())
    expected = consensus["source_sha256"]["posthoc_routing_budget/routing_budget_result.json"]
    if hash_file(source) != expected:
        raise ValueError("routing supplement differs from frozen consensus dependency")
    if target.exists():
        raise FileExistsError("supplement mapping target must be absent")
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(source, target)
    return [{"source": str(source.relative_to(workspace)),
             "target": str(target.relative_to(workspace)), "sha256": expected,
             "reason": "The frozen consensus analyzer resolves this supplemental result below the historical artifact root; copied bytes are unchanged."}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path(__file__).resolve().parents[1] / "asset_index.json")
    a = p.parse_args()
    if a.out.exists() or a.out.is_symlink():
        raise FileExistsError("output path must not exist")
    index = verify(a.asset_dir, a.index)
    members = []
    names = set()
    for item in index["assets"]:
        if not item["name"].endswith(".zip"):
            continue
        with zipfile.ZipFile(a.asset_dir / item["name"]) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                if not parts or ".." in parts or info.filename.startswith(("/", "\\")) or "\\" in info.filename or ":" in parts[0] or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("unsafe ZIP member path")
                if info.filename in names:
                    raise ValueError("cross-archive path collision: " + info.filename)
                names.add(info.filename)
                members.append((item["name"], info.filename, info.file_size))
    required = sum(x[2] for x in members)
    import shutil
    parent = a.out.parent
    if not parent.is_dir():
        raise FileNotFoundError("output parent must exist")
    if shutil.disk_usage(parent).free < required + 256 * 1024**2:
        raise OSError("insufficient free space for extracted workspace and margin")
    a.out.mkdir(mode=0o700)
    for item in index["assets"]:
        if not item["name"].endswith(".zip"):
            continue
        with zipfile.ZipFile(a.asset_dir / item["name"]) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                target = a.out.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                mode = info.external_attr >> 16
                target.chmod(0o700 if mode & 0o111 else 0o600)
    mappings = compose_reproduction_inputs(a.out)
    receipt = {"mode": "offline_safe_extraction_and_explicit_supplement_mapping_no_inference", "archives": [x["name"] for x in index["assets"] if x["name"].endswith(".zip")], "archive_files": len(members), "expanded_archive_bytes": required, "supplement_mappings": mappings}
    (a.out / "EXTRACTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PASS safely extracted", len(members), "files to a fresh workspace")


if __name__ == "__main__":
    main()
