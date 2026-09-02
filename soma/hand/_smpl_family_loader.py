# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MANO asset loading helpers for hand-only SOMA-X code."""

from pathlib import Path
from typing import Any

import numpy as np

from .._smpl_family_loader import (
    _get_required,
    _read_model_file,
    _to_numpy,
    parent_ids_from_kintree,
)


def load_mano_pkl(
    data_root: str | Path,
    hand_type: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load MANO pickle data as plain NumPy arrays."""
    if hand_type not in ("left", "right"):
        raise ValueError(f"hand_type must be 'left' or 'right', got {hand_type!r}.")

    if model_path is None:
        pkl_path = Path(data_root) / "MANO" / f"MANO_{hand_type.upper()}.pkl"
    else:
        pkl_path = Path(model_path).expanduser()
        if not pkl_path.is_file():
            raise FileNotFoundError(f"MANO model not found at '{pkl_path}'")
    data = _read_model_file(pkl_path)
    j_reg = _to_numpy(_get_required(data, "J_regressor"), np.float32)

    return {
        "v_template": _to_numpy(_get_required(data, "v_template"), np.float32),
        "shapedirs": _to_numpy(_get_required(data, "shapedirs"), np.float32),
        "J_regressor": j_reg,
        "weights": _to_numpy(_get_required(data, "weights"), np.float32),
        "kintree_table": _to_numpy(_get_required(data, "kintree_table"), np.int64),
        "faces": _to_numpy(_get_required(data, "f"), np.int64),
        "posedirs": _to_numpy(_get_required(data, "posedirs"), np.float32),
        "hands_mean": _to_numpy(_get_required(data, "hands_mean"), np.float32),
        "hands_components": _to_numpy(_get_required(data, "hands_components"), np.float32),
    }


def mano_parent_ids(kintree_table: np.ndarray) -> list[int]:
    """Convert a MANO kintree table into parent column indices."""
    return parent_ids_from_kintree(kintree_table).tolist()
