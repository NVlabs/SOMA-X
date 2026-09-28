# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constructor asset lookup works without local model assets or network access."""

from unittest.mock import Mock

import pytest

from soma import SOMAHandLayer, SOMALayer


@pytest.mark.parametrize(
    "layer_type,asset_name",
    [(SOMALayer, "SOMA_neutral.npz"), (SOMAHandLayer, "SOMAHand.npz")],
    ids=["body", "hand"],
)
@pytest.mark.parametrize(
    "root_kind", ["omitted", "none", "missing_path", "missing_str", "existing_path", "existing_str"]
)
def test_constructor_resolves_asset_directory(
    layer_type, asset_name, root_kind, tmp_path, monkeypatch
):
    downloaded_root = tmp_path / "downloaded"
    downloaded_root.mkdir()
    explicit_root = tmp_path / "explicit"
    uses_download = not root_kind.startswith("existing")
    if not uses_download:
        explicit_root.mkdir()

    kwargs = {}
    if root_kind == "none":
        kwargs["data_root"] = None
    elif root_kind != "omitted":
        kwargs["data_root"] = str(explicit_root) if root_kind.endswith("str") else explicit_root

    download = Mock(return_value=str(downloaded_root))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    # Asset lookup does not need the hand constructor's eager Warp initialization.
    monkeypatch.setattr("soma.hand.soma.ensure_warp_initialized", lambda: None)

    # Empty directories stop construction at real asset validation, before model loading.
    with pytest.raises(FileNotFoundError) as exc_info:
        layer_type(identity_model_type="mhr", device="cpu", **kwargs)

    expected_root = downloaded_root if uses_download else explicit_root
    assert asset_name in str(exc_info.value)
    assert str(expected_root) in str(exc_info.value)
    if uses_download:
        download.assert_called_once()
    else:
        download.assert_not_called()
