#!/usr/bin/env python3
"""Positive and tampering controls for local asset verification."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from verify_qwen3_official_assets import check_asset


class AssetChecks(unittest.TestCase):
    def test_git_blob_and_lfs_positive_and_same_size_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample"
            data = b"original file\n"
            path.write_bytes(data)
            git = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            entries = [
                {"size": len(data), "git_blob_sha1": git},
                {"size": len(data), "git_blob_sha1": "0" * 40,
                 "lfs_sha256": hashlib.sha256(data).hexdigest()},
            ]
            for entry in entries:
                self.assertTrue(check_asset(path, entry)["official_match"])
            path.write_bytes(b"modified file\n")
            for entry in entries:
                with self.assertRaises(ValueError):
                    check_asset(path, entry)
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                check_asset(path, entries[0])


if __name__ == "__main__":
    unittest.main()
