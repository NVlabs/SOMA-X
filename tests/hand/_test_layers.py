# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached SOMAHandLayer construction for unit tests."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

from tests._optional_assets import hand_identity_skip_reason

_HAND_LAYER_CACHE_MAX_SIZE = 16
_HAND_LAYER_CACHE = OrderedDict()


def make_hand_layer(
    data_root: str | Path,
    hand_type: str,
    device,
    identity_model_type: str = "soma",
    mode: str = "warp",
    **layer_kwargs: Any,
):
    cache_key = (
        str(Path(data_root).resolve()),
        hand_type,
        str(device),
        identity_model_type,
        mode,
        tuple(sorted(layer_kwargs.items())),
    )
    if cache_key in _HAND_LAYER_CACHE:
        _HAND_LAYER_CACHE.move_to_end(cache_key)
        return _HAND_LAYER_CACHE[cache_key]

    skip_reason = hand_identity_skip_reason(
        data_root,
        identity_model_type,
        hand_type=hand_type,
    )
    if skip_reason is not None:
        return _remember_hand_layer(cache_key, (None, skip_reason))

    from soma import SOMAHandLayer

    try:
        layer = SOMAHandLayer(
            data_root=data_root,
            hand_type=hand_type,
            device=device,
            identity_model_type=identity_model_type,
            mode=mode,
            **layer_kwargs,
        ).to(device)
        return _remember_hand_layer(cache_key, (layer, None))
    except (FileNotFoundError, ImportError) as e:
        return _remember_hand_layer(cache_key, (None, f"Missing asset or dependency: {e}"))
    except Exception as e:
        return _remember_hand_layer(
            cache_key,
            (None, f"Could not create layer ({type(e).__name__}): {e}"),
        )


def _remember_hand_layer(cache_key, value):
    _HAND_LAYER_CACHE[cache_key] = value
    _HAND_LAYER_CACHE.move_to_end(cache_key)
    while len(_HAND_LAYER_CACHE) > _HAND_LAYER_CACHE_MAX_SIZE:
        _HAND_LAYER_CACHE.popitem(last=False)
    return value
