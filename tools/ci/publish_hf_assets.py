#!/usr/bin/env python3

"""Publish and immutably tag a verified SOMA-X Hugging Face stage."""

import argparse
import sys
import tempfile
from pathlib import Path

from verify_hf_assets import MANIFEST_NAME, verify_assets


def publish_assets(stage: Path, repo_id: str, release_tag: str) -> str:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    stage = stage.resolve()
    manifest = verify_assets(stage)
    if manifest["release_tag"] != release_tag:
        raise ValueError(
            f"Stage release tag {manifest['release_tag']} does not match {release_tag}"
        )

    api = HfApi()
    existing_tags = {ref.name: ref.target_commit for ref in api.list_repo_refs(repo_id).tags}
    if release_tag in existing_tags:
        remote_manifest = Path(
            hf_hub_download(repo_id=repo_id, filename=MANIFEST_NAME, revision=release_tag)
        )
        if remote_manifest.read_bytes() != (stage / MANIFEST_NAME).read_bytes():
            raise ValueError(
                f"Hugging Face tag {release_tag} already exists with a different manifest"
            )
        with tempfile.TemporaryDirectory() as download_dir:
            downloaded = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=release_tag,
                    local_dir=download_dir,
                )
            )
            verify_assets(downloaded, allow_download_cache=True)
        print(
            f"Hugging Face tag {release_tag} already matches the stage; "
            "publication is idempotently complete."
        )
        return existing_tags[release_tag]

    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=stage,
        delete_patterns="*",
        commit_message=f"Publish SOMA-X assets for {release_tag}",
        commit_description=f"Public GitHub commit: {manifest['public_commit']}",
    )
    api.create_tag(
        repo_id=repo_id,
        tag=release_tag,
        revision=commit.oid,
        tag_message=f"SOMA-X assets for {release_tag}",
    )
    print(f"Published {repo_id}@{release_tag} from Hub commit {commit.oid}")
    return commit.oid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--repo-id", default="nvidia/SOMA-X")
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    try:
        publish_assets(args.stage, args.repo_id, args.release_tag)
    except (OSError, ValueError) as exc:
        print(f"Hugging Face publication failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
