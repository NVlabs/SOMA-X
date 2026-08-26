#!/usr/bin/env python3

"""Verify a staged or downloaded SOMA-X Hugging Face release."""

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

MANIFEST_NAME = "SOMA-X-HF-MANIFEST.json"
HUB_INFRASTRUCTURE_FILES = {".gitattributes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(root: Path, allow_download_cache: bool = False) -> dict:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Release manifest must use schema_version 1")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Release manifest must contain a non-empty files list")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if allow_download_cache and relative.startswith(".cache/huggingface/"):
            continue
        if path.is_symlink():
            raise ValueError(f"Release contains a symlink: {relative}")
        if relative != MANIFEST_NAME and relative not in HUB_INFRASTRUCTURE_FILES:
            actual.add(relative)

    expected: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each release manifest file entry must be an object")
        relative = entry.get("path", "")
        path = PurePosixPath(relative)
        if not relative or path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError(f"Manifest contains an unsafe path: {relative!r}")
        if relative in expected:
            raise ValueError(f"Manifest contains a duplicate path: {relative}")
        expected.add(relative)
        file_path = root.joinpath(*path.parts)
        if not file_path.is_file():
            raise ValueError(f"Manifest file is missing: {relative}")
        if file_path.stat().st_size != entry.get("size"):
            raise ValueError(f"Size mismatch for {relative}")
        if _sha256(file_path) != entry.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {relative}")

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("Release file set mismatch: " + ", ".join(details))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--allow-download-cache",
        action="store_true",
        help="Ignore local .cache/huggingface metadata created by hf download.",
    )
    args = parser.parse_args()
    try:
        manifest = verify_assets(args.root, allow_download_cache=args.allow_download_cache)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Hugging Face verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Verified {len(manifest['files'])} files for {manifest['release_tag']} "
        f"from public commit {manifest['public_commit']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
