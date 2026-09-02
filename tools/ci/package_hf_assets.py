#!/usr/bin/env python3

"""Build a deterministic, allowlisted Hugging Face release stage."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

from check_release_version import check_versions

MANIFEST_NAME = "SOMA-X-HF-MANIFEST.json"
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
TAG_RE = re.compile(r"^v(?P<series>\d+\.\d+)\.\d+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must be a normalized relative path: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("Hugging Face asset contract must use schema_version 1")
    if not isinstance(contract.get("release_series"), str):
        raise ValueError("Hugging Face asset contract is missing release_series")
    if not isinstance(contract.get("files"), list) or not contract["files"]:
        raise ValueError("Hugging Face asset contract must contain a non-empty files list")
    return contract


def package_assets(
    root: Path,
    contract_path: Path,
    output: Path,
    release_tag: str,
    public_commit: str,
) -> dict:
    root = root.resolve()
    contract_path = contract_path.resolve()
    output = output.resolve()
    contract = _load_contract(contract_path)

    tag_match = TAG_RE.fullmatch(release_tag)
    if tag_match is None:
        raise ValueError(f"Release tag must be vMAJOR.MINOR.PATCH: {release_tag!r}")
    if tag_match.group("series") != contract["release_series"]:
        raise ValueError(
            f"Release tag {release_tag} does not match contract series "
            f"{contract['release_series']}"
        )
    check_versions(root, release_tag)
    if COMMIT_RE.fullmatch(public_commit) is None:
        raise ValueError("Public commit must be a full 40-character lowercase Git SHA")
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the repository root")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    seen_destinations: set[str] = set()
    for entry in contract["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Each contract file entry must be an object")
        source_rel = _safe_relative_path(entry.get("source", ""), "source")
        destination_rel = _safe_relative_path(entry.get("path", ""), "path")
        destination_name = destination_rel.as_posix()
        if destination_name == MANIFEST_NAME:
            raise ValueError(f"Contract cannot overwrite {MANIFEST_NAME}")
        if destination_name in seen_destinations:
            raise ValueError(f"Duplicate destination in contract: {destination_name}")
        seen_destinations.add(destination_name)

        source = root.joinpath(*source_rel.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Required source is missing or not a regular file: {source_rel}")
        with source.open("rb") as stream:
            if stream.read(len(LFS_POINTER_HEADER)) == LFS_POINTER_HEADER:
                raise ValueError(f"Required source is an unresolved Git LFS pointer: {source_rel}")

        destination = output.joinpath(*destination_rel.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append(
            {
                "path": destination_name,
                "sha256": _sha256(destination),
                "size": destination.stat().st_size,
                "source": source_rel.as_posix(),
            }
        )

    manifest = {
        "schema_version": 1,
        "release_tag": release_tag,
        "public_commit": public_commit,
        "files": sorted(files, key=lambda item: item["path"]),
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("hf_assets_v0.3.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--public-commit", required=True)
    args = parser.parse_args()

    try:
        manifest = package_assets(
            args.root,
            args.contract,
            args.output,
            args.release_tag,
            args.public_commit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Hugging Face staging failed: {exc}", file=sys.stderr)
        return 1

    total_bytes = sum(item["size"] for item in manifest["files"])
    print(
        f"Staged {len(manifest['files'])} files ({total_bytes} bytes) for "
        f"{manifest['release_tag']} at {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
