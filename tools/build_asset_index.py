#!/usr/bin/env python3
"""Build SHA-256 and Markdown indexes from the exact selected release assets."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

ORIGINAL_NAMES = {'acord_qwen_textfree.zip': 'ACORD_无原文可重建复现候选.zip', 'acord_smollm2_textfree.zip': 'ACORD_SmolLM2_无原文复现候选.zip', 'acord_batch_textfree.zip': 'ACORD_批次敏感性_无原文复现候选.zip'}

ASSETS = [
    ("reproducibility_without_trec_article_text.zip", "main", "Main saved traces, analyses and fixed-source TREC reconstruction"),
    ("midpoint_ratio_supplement.zip", "midpoint", "Midpoint, lexical rule, cancellation and matched-budget routing"),
    ("response_provenance_supplement.zip", "provenance", "Historical response-recording correction"),
    ("figure_style_supplement.zip", "figures", "Current figures, generators and provenance"),
    ("acord_qwen_textfree.zip", "acord_qwen", "ACORD Qwen reconstruction without source contract/query text"),
    ("acord_smollm2_textfree.zip", "acord_smol", "ACORD SmolLM2 saved follow-up; requires Qwen reconstruction"),
    ("acord_batch_textfree.zip", "acord_batch", "ACORD replacement chunk-size comparison and disclosed history"),
    ("review_audit_supplement.zip", "review_audit", "Controlled order, consensus and count-envelope evidence"),
    ("IntentSat2S_LaTeX_source.zip", "manuscript_source", "Historical rendered manuscript and standalone editable source"),
    ("IntentSat2S_IEEE_review_draft.pdf", "manuscript_pdf", "Historical review PDF with pending author metadata"),
]


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--github-repo", help="Actual published OWNER/REPOSITORY; omit before creation")
    p.add_argument("--tag", help="Actual release tag; omit before creation")
    a = p.parse_args()
    if bool(a.github_repo) != bool(a.tag):
        p.error("--github-repo and --tag must be given together")
    if a.github_repo and (a.github_repo.count("/") != 1 or any(c in a.github_repo for c in " ?#\\")):
        p.error("repository must be actual OWNER/REPOSITORY")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, role, description in ASSETS:
        f = a.asset_dir / name
        if not f.is_file() or f.is_symlink():
            raise FileNotFoundError("missing or symlinked asset: " + name)
        url = None
        if a.github_repo:
            url = "https://github.com/" + a.github_repo + "/releases/download/" + quote(a.tag, safe="") + "/" + quote(name, safe="")
        entries.append({"name": name, "role": role, "bytes": f.stat().st_size,
                        "sha256": hash_file(f), "description": description,
                        "download_url": url, "original_name": ORIGINAL_NAMES.get(name, name)})
    index = {"schema_version": 1, "snapshot": "historical_recoverable_2026-11-02",
             "publication_status": "location_recorded_access_not_verified" if a.github_repo else "local_staging",
             "repository": a.github_repo, "release_tag": a.tag,
             "coverage": "One main variant plus seven supplements cover the saved historical review manuscript. Later missing transfer/GLM volumes are not included.",
             "assets": entries}
    (a.out_dir / "asset_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    manifest = "".join(x["sha256"] + "  " + x["name"] + "\n" for x in entries)
    (a.out_dir / "SHA256SUMS.txt").write_text(manifest)
    lines = ["# Selected artifact assets", "", "One main archive and all seven scientific supplements are required. The manuscript PDF/source identify the historical revision.", "", "Release locations are recorded from the created repository and release. Anonymous download verification is documented separately.", "", "| Asset | Bytes | SHA-256 | Purpose |", "| --- | ---: | --- | --- |"]
    for x in entries:
        label = "[" + x["name"] + "](" + x["download_url"] + ")" if x["download_url"] else "`" + x["name"] + "`"
        lines.append("| " + label + " | " + str(x["bytes"]) + " | `" + x["sha256"] + "` | " + x["description"] + " |")
    lines.extend(["", "Total selected bytes: " + str(sum(x["bytes"] for x in entries)) + ".", "", "The full TREC-article-text main variant is deliberately excluded from this set. Third-party notices, linkable pseudonyms, the pending project license status and the unavailable later volumes are described in the repository guides.", ""])
    (a.out_dir / "ASSETS.md").write_text("\n".join(lines))
    print("Indexed", len(entries), "actual assets; public location", "recorded" if a.github_repo else "unset")


if __name__ == "__main__":
    main()
