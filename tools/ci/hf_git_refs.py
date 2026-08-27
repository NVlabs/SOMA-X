#!/usr/bin/env python3

"""Exercise or recover Hugging Face Git tags with a scoped CI credential."""

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from publish_hf_assets import _create_tag_via_git, _delete_tag_via_git
from verify_hf_assets import MANIFEST_NAME

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SMOKE_TAG_RE = re.compile(r"^ci-oidc-[0-9]+-[0-9]+$")
VERSION_TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")


def _manifest_bytes(repo_id: str, revision: str, token: str) -> bytes:
    manifest = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=MANIFEST_NAME,
            revision=revision,
            token=token,
            force_download=True,
        )
    )
    return manifest.read_bytes()


def _tag_names(repo_id: str, token: str) -> set[str]:
    return {ref.name for ref in HfApi(token=token).list_repo_refs(repo_id).tags}


def smoke_git_auth(repo_id: str, revision: str, tag: str, token: str) -> None:
    """Create, verify, and remove a non-release tag without changing Hub content."""
    if SMOKE_TAG_RE.fullmatch(tag) is None:
        raise ValueError("Smoke tag must match ci-oidc-RUN_ID-RUN_ATTEMPT")
    if tag in _tag_names(repo_id, token):
        raise ValueError(f"Refusing to reuse existing smoke tag {tag}")

    source_manifest = _manifest_bytes(repo_id, revision, token)
    created = False
    try:
        _create_tag_via_git(repo_id, tag, revision, token)
        created = True
        if _manifest_bytes(repo_id, tag, token) != source_manifest:
            raise ValueError(f"Smoke tag {tag} does not resolve to {revision}")
    finally:
        if created:
            _delete_tag_via_git(repo_id, tag, token)

    if tag in _tag_names(repo_id, token):
        raise RuntimeError(f"Smoke tag cleanup failed: {tag}")
    print(f"Hugging Face Git authentication smoke passed for {repo_id}; removed {tag}.")


def create_verified_release_tag(
    repo_id: str,
    revision: str,
    tag: str,
    token: str,
) -> None:
    """Create an immutable release tag for an already-published, verified manifest."""
    if COMMIT_RE.fullmatch(revision) is None:
        raise ValueError("Recovery revision must be a full lowercase Git commit SHA")
    if VERSION_TAG_RE.fullmatch(tag) is None:
        raise ValueError("Recovery tag must match vMAJOR.MINOR.PATCH")

    source_manifest = _manifest_bytes(repo_id, revision, token)
    manifest = json.loads(source_manifest)
    if manifest.get("release_tag") != tag:
        raise ValueError(
            f"Hub manifest release {manifest.get('release_tag')!r} does not match {tag}"
        )

    if tag in _tag_names(repo_id, token):
        if _manifest_bytes(repo_id, tag, token) != source_manifest:
            raise ValueError(f"Hugging Face tag {tag} exists with a different manifest")
        print(f"Hugging Face tag {tag} already matches {revision}.")
        return

    _create_tag_via_git(repo_id, tag, revision, token)
    if _manifest_bytes(repo_id, tag, token) != source_manifest:
        raise ValueError(f"Hugging Face tag {tag} does not resolve to {revision}")
    print(f"Created and verified Hugging Face tag {repo_id}@{tag} at {revision}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="nvidia/SOMA-X")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--revision", default="main")
    smoke_parser.add_argument("--tag", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--revision", required=True)
    create_parser.add_argument("--tag", required=True)

    args = parser.parse_args()
    token = get_token()
    if token is None:
        print("Hugging Face credential is unavailable", file=sys.stderr)
        return 1

    try:
        if args.operation == "smoke":
            smoke_git_auth(args.repo_id, args.revision, args.tag, token)
        else:
            create_verified_release_tag(args.repo_id, args.revision, args.tag, token)
    except (HfHubHTTPError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Hugging Face Git ref operation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
