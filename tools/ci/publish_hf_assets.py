#!/usr/bin/env python3

"""Publish and immutably tag a verified SOMA-X Hugging Face stage."""

import argparse
import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from verify_hf_assets import MANIFEST_NAME, verify_assets


def _run_git(args: list[str], token: str | None = None) -> None:
    command = ["git"]
    env = os.environ.copy()
    encoded_credential = None
    if token is not None:
        encoded_credential = base64.b64encode(f"hf_user:{token}".encode()).decode()
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {encoded_credential}",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
    command.extend(args)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode == 0:
        return

    detail = result.stderr.strip()
    if token is not None:
        detail = detail.replace(token, "***")
    if encoded_credential is not None:
        detail = detail.replace(encoded_credential, "***")
    if detail:
        detail = f": {detail}"
    raise RuntimeError(f"git {args[0]} failed with exit code {result.returncode}{detail}")


def _create_tag_via_git(repo_id: str, tag: str, revision: str, token: str) -> None:
    """Create an annotated Hub tag with the repo-scoped OIDC credential."""
    with tempfile.TemporaryDirectory() as checkout_dir:
        repo = Path(checkout_dir)
        _run_git(["-C", str(repo), "init", "--quiet"])
        _run_git(
            [
                "-C",
                str(repo),
                "remote",
                "add",
                "origin",
                f"https://huggingface.co/{repo_id}.git",
            ]
        )
        _run_git(["-C", str(repo), "fetch", "--quiet", "--depth=1", "origin", revision])
        _run_git(
            [
                "-C",
                str(repo),
                "-c",
                "user.name=SOMA-X release automation",
                "-c",
                "user.email=soma-x-release@noreply.github.com",
                "tag",
                "--annotate",
                tag,
                "FETCH_HEAD",
                "--message",
                f"SOMA-X assets for {tag}",
            ]
        )
        _run_git(
            ["-C", str(repo), "push", "--quiet", "origin", f"refs/tags/{tag}"],
            token=token,
        )


def _delete_tag_via_git(repo_id: str, tag: str, token: str) -> None:
    """Delete a Hub tag with the same scoped Git credential used to create it."""
    with tempfile.TemporaryDirectory() as checkout_dir:
        repo = Path(checkout_dir)
        _run_git(["-C", str(repo), "init", "--quiet"])
        _run_git(
            [
                "-C",
                str(repo),
                "remote",
                "add",
                "origin",
                f"https://huggingface.co/{repo_id}.git",
            ]
        )
        _run_git(
            ["-C", str(repo), "push", "--quiet", "origin", f":refs/tags/{tag}"],
            token=token,
        )


def publish_assets(stage: Path, repo_id: str, release_tag: str) -> str:
    from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download

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
    token = get_token()
    if token is None:
        raise RuntimeError("Hugging Face credential is unavailable for tag creation")
    _create_tag_via_git(repo_id, release_tag, commit.oid, token)
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
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Hugging Face publication failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
