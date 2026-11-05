#!/usr/bin/env python3
"""Copy browsable scientific sources from verified assets, preserving bytes."""
import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from verify_assets import verify

CODE = {".py", ".sh", ".tex", ".cls"}
EXCLUDE = {"prepare_submission_authors.py", "test_prepare_submission_authors.py", "generate_submission_fields.py", "make_editable_word.py"}


def include(name):
    parts = PurePosixPath(name).parts
    leaf = parts[-1] if parts else ""
    low = leaf.lower()
    if not parts or ".." in parts or "03_submission" in parts or leaf in EXCLUDE:
        return False
    return (Path(leaf).suffix.lower() in CODE or
            (low.endswith(".md") and any(x in low for x in ("readme", "protocol", "amendment", "notice", "history", "clarification"))) or
            low.startswith(("license", "notice", "upstream_license", "requirements")) or
            (low.endswith(".txt") and "sha256" in low))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--index", type=Path, default=Path(__file__).resolve().parents[1] / "asset_index.json")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("source output must not exist")
    index = verify(a.asset_dir, a.index)
    a.out.mkdir()
    receipt = []

    def save(role, member, data, original):
        if not include(member):
            return
        path = a.out / role / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        receipt.append({"path": path.relative_to(a.out).as_posix(), "archive_member": original,
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "modified": False})

    for item in index["assets"]:
        if not item["name"].endswith(".zip"):
            continue
        with zipfile.ZipFile(a.asset_dir / item["name"]) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                    raise ValueError("unsafe source member")
                relative = PurePosixPath(*name.parts[1:]).as_posix()
                original = item.get("original_name", item["name"]) + "!" + info.filename
                if include(relative):
                    save(item["role"], relative, z.read(info), original)
                if info.filename.endswith((".tar.gz", ".tgz")):
                    with tarfile.open(fileobj=io.BytesIO(z.read(info)), mode="r:gz") as tar:
                        for member in tar:
                            if not member.isfile() or member.issym() or member.islnk():
                                continue
                            if PurePosixPath(member.name).is_absolute() or ".." in PurePosixPath(member.name).parts:
                                raise ValueError("unsafe nested source member")
                            if include(member.name):
                                stream = tar.extractfile(member)
                                if stream is None:
                                    raise ValueError("unreadable nested source")
                                save(item["role"], relative + ".sources/" + member.name, stream.read(), original + "!" + member.name)
    (a.out / "SOURCE_PROVENANCE.json").write_text(json.dumps({"snapshot": index["snapshot"], "files": receipt}, ensure_ascii=False, indent=2) + "\n")
    (a.out / "README.md").write_text("# Readable scientific source\n\nThese are byte-preserved source files copied from the selected historical release assets. `SOURCE_PROVENANCE.json` identifies their exact archive members and hashes. They are for browsing; use the fully extracted release assets for execution because this source-only tree intentionally omits dataset samples, responses, model assets and large records. No new license grant is implied; see the repository NOTICE and LICENSES.\n")
    print("PASS copied", len(receipt), "unchanged scientific source/notice files")


if __name__ == "__main__":
    main()
