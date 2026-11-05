#!/usr/bin/env python3
"""Read every member of nominated archives; write hashes and flag-only findings.

This is a bounded heuristic audit, not proof of privacy or permission to publish.
It never prints matching credential values or private reference text. No files
are extracted, no environment credentials are read, and no network is used.
Optional PDF text extraction needs pypdf; missing PDF coverage fails closed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import stat
import tarfile
import warnings
import zipfile
from pathlib import Path, PurePosixPath

SECRET_PATTERNS = {
    "private_key_pem": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "openai_key_shape": rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b",
    "huggingface_token_shape": rb"\bhf_[A-Za-z0-9]{25,}\b",
    "github_token_shape": rb"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b",
    "aws_access_key_shape": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "slack_token_shape": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "google_api_key_shape": rb"\bAIza[A-Za-z0-9_-]{35}\b",
    "jwt_shape": rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b",
    "url_embedded_password": rb"\b(?:https?|postgres(?:ql)?|mysql|mongodb)://[^\s/:@]{1,128}:[^\s/@]{3,256}@",
    "bearer_header_shape": rb"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9_.~+/-]{24,}",
    "long_secret_assignment_shape": rb"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_+/.=-]{24,}[\"']",
}
PATTERNS = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
PRIVATE_PARTS = {".git", ".ssh", ".aws", ".cache", ".config", ".venv", "venv", "__pycache__", "site-packages"}
PRIVATE_NAMES = {".env", ".netrc", ".npmrc", "id_rsa", "id_ed25519", "credentials.json", "token.json", "author_metadata.json"}
MAX_ENTRY = 512 * 1024**2
MAX_EXPANSION = 8 * 1024**3
MAX_ENTRIES = 50000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--private-reference-jsonl", action="append", type=Path, default=[],
                        help="Optional local JSONL with query/items[].text; never copied into report")
    parser.add_argument("--private-reference-zip", action="append", default=[],
                        help="Optional ZIP::member JSONL reference, read only in memory")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None

    refs: dict[bytes, str] = {}
    reference_inputs = [file.read_bytes() for file in args.private_reference_jsonl]
    for reference in args.private_reference_zip:
        filename, member = reference.split("::", 1)
        with zipfile.ZipFile(filename) as z:
            reference_inputs.append(z.read(member))
    for payload in reference_inputs:
        for line in payload.splitlines():
            record = json.loads(line)
            query = record.get("query", "")
            if isinstance(query, str) and len(query) >= 8:
                refs[json.dumps(query, ensure_ascii=False).encode()] = "quoted_query"
                refs[json.dumps(query, ensure_ascii=True).encode()] = "quoted_query"
            for item in record.get("items", []):
                txt = item.get("text", "")
                if isinstance(txt, str) and len(txt) >= 160:
                    refs[txt[:160].encode()] = "clause_prefix_160"
                    refs[json.dumps(txt[:160], ensure_ascii=True)[1:-1].encode()] = "clause_prefix_160"

    all_summaries = []
    for archive in args.archives:
        entries = []
        findings = []
        counts = {"expanded_bytes": 0, "zip_containers": 0, "tar_containers": 0,
                  "gzip_streams": 0, "pdf_documents": 0, "pdf_failures": 0}

        def finding(path, category, **extra):
            findings.append({"member": path, "category": category, **extra})

        def inspect(path: str, data: bytes):
            for category, pattern in PATTERNS.items():
                if pattern.search(data):
                    finding(path, category)
            if "acord" in archive.name.lower():
                for value, category in refs.items():
                    if value in data:
                        finding(path, "private_reference_" + category,
                                reference_sha256=digest(value))

        def read(stream):
            data = stream.read(MAX_ENTRY + 1)
            if len(data) > MAX_ENTRY:
                raise ValueError("archive member exceeds bounded scan size")
            counts["expanded_bytes"] += len(data)
            if counts["expanded_bytes"] > MAX_EXPANSION:
                raise ValueError("total expansion exceeds bounded scan size")
            return data

        def visit(path: str, data: bytes, depth: int, container: bool = False):
            if depth > 6 or len(entries) > MAX_ENTRIES:
                raise ValueError("archive nesting or member count exceeds limit")
            entries.append({"member": path, "bytes": len(data), "sha256": digest(data), "depth": depth})
            inspect(path, data)
            low = path.lower()
            if low.endswith((".zip", ".docx", ".xlsx", ".pptx")):
                counts["zip_containers"] += 1
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    names = set()
                    for info in z.infolist():
                        if info.is_dir():
                            continue
                        child = path + "!" + info.filename
                        parts = PurePosixPath(info.filename).parts
                        if info.filename in names:
                            finding(child, "duplicate_member")
                        names.add(info.filename)
                        if info.filename.startswith(("/", "\\")) or ".." in parts or "\\" in info.filename:
                            finding(child, "unsafe_member_path")
                        if stat.S_ISLNK(info.external_attr >> 16):
                            finding(child, "symlink_member")
                        if any(part.lower() in PRIVATE_PARTS for part in parts) or Path(info.filename).name.lower() in PRIVATE_NAMES or info.filename.lower().endswith((".pyc", ".pyo")):
                            finding(child, "private_or_cache_path")
                        if info.flag_bits & 1:
                            raise ValueError("encrypted member cannot be audited")
                        if info.file_size > MAX_ENTRY:
                            raise ValueError("declared archive member exceeds scan limit")
                        with z.open(info) as stream:
                            visit(child, read(stream), depth + 1)
            elif low.endswith((".tar.gz", ".tgz", ".tar")):
                counts["tar_containers"] += 1
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
                    names = set()
                    for info in tar:
                        child = path + "!" + info.name
                        if info.issym() or info.islnk():
                            finding(child, "symlink_member")
                        if not info.isfile():
                            continue
                        if info.name in names:
                            finding(child, "duplicate_member")
                        names.add(info.name)
                        if info.name.startswith("/") or ".." in PurePosixPath(info.name).parts:
                            finding(child, "unsafe_member_path")
                        if any(part.lower() in PRIVATE_PARTS for part in PurePosixPath(info.name).parts) or Path(info.name).name.lower() in PRIVATE_NAMES or info.name.lower().endswith((".pyc", ".pyo")):
                            finding(child, "private_or_cache_path")
                        if info.size > MAX_ENTRY:
                            raise ValueError("declared tar member exceeds scan limit")
                        stream = tar.extractfile(info)
                        if stream is None:
                            raise ValueError("unreadable tar member")
                        with stream:
                            visit(child, read(stream), depth + 1)
            elif low.endswith(".gz"):
                counts["gzip_streams"] += 1
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                    visit(path + "!<expanded>", read(stream), depth + 1)
            elif low.endswith(".pdf"):
                counts["pdf_documents"] += 1
                try:
                    if PdfReader is None:
                        raise RuntimeError("PDF parser unavailable")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pdf = PdfReader(io.BytesIO(data), strict=True)
                        if pdf.is_encrypted:
                            raise ValueError("encrypted PDF")
                        inspect(path + "!metadata", json.dumps(dict(pdf.metadata or {}), default=str).encode())
                        for index, page in enumerate(pdf.pages):
                            inspect(path + "!page-" + str(index + 1), (page.extract_text() or "").encode())
                except Exception as exc:
                    counts["pdf_failures"] += 1
                    finding(path, "pdf_text_unchecked", exception_type=type(exc).__name__)

        payload = archive.read_bytes()
        visit(archive.name, payload, 0, True)
        summary = {"archive": archive.name, "bytes": len(payload), "sha256": digest(payload),
                   "entries_including_containers": len(entries), **counts,
                   "findings": findings, "private_reference_patterns": len(refs),
                   "status": "heuristic_clear" if not findings else "review_required"}
        (args.report_dir / (archive.name + ".members.json")).write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
        all_summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in {"findings"}}, ensure_ascii=False))
    report = {"method": "recursive per-member byte scan, CRC validation, SHA-256 inventory, PDF text/metadata, nominated private-reference exact substrings",
              "limitations": "No OCR, semantic privacy guarantee, arbitrary password detector, copyright permission, or completeness guarantee. Reference scan covers nominated strings only; pseudonyms can remain linkable.",
              "archives": all_summaries}
    (args.report_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if all(x["status"] == "heuristic_clear" for x in all_summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
