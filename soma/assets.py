# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset download helpers for SOMA-X model data."""

from pathlib import Path

REPO_ID = "nvidia/soma-x"


def get_assets_dir(revision: str = "v0.2.4", cache_dir: str | Path | None = None) -> Path:
    """Download (or retrieve from cache) the SOMA asset directory from HuggingFace.

    Uses ``huggingface_hub.snapshot_download`` which preserves the repository
    directory structure and caches downloads under ``~/.cache/huggingface/hub/``.
    Subsequent calls with the same *revision* are instant (no network access).

    Args:
        revision: Git revision (branch, tag, or commit hash) to download. Defaults
            to the immutable asset tag matching this package release.
        cache_dir: Override the default HuggingFace cache directory.

    Returns:
        Path to the local directory containing the downloaded assets.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
        )
    )
