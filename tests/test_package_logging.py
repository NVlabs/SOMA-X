# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
from pathlib import Path

import numpy as np

from soma.io import load_soma_npz, save_soma_npz
from soma.soma import SOMALayer

REPO_ROOT = Path(__file__).resolve().parents[1]
SOMA_ROOT = REPO_ROOT / "soma"


def test_package_modules_do_not_call_print():
    offenders = []
    for path in sorted(SOMA_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []


def test_save_soma_npz_logs_summary_without_stdout(tmp_path, caplog, capsys):
    out_path = tmp_path / "sample.npz"
    poses = np.zeros((1, 2, 3), dtype=np.float32)
    transl = np.zeros((1, 3), dtype=np.float32)
    identity_coeffs = np.zeros((1, 4), dtype=np.float32)

    with caplog.at_level(logging.INFO, logger="soma.io"):
        save_soma_npz(
            out_path,
            poses,
            transl,
            joint_names=["Root", "Hips"],
            identity_model_type="soma",
            identity_coeffs=identity_coeffs,
            keep_root=True,
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert out_path.is_file()
    assert "Saved:" in caplog.text
    assert "identity_model_type: soma" in caplog.text
    assert "joint_names: 2 joints" in caplog.text


def test_save_load_soma_npz_preserves_native_scale_params(tmp_path):
    out_path = tmp_path / "native_scales.npz"
    scale_params = np.linspace(
        0.75,
        1.25,
        SOMALayer.NUM_BONE_SCALE_PARAMS,
        dtype=np.float32,
    )[None]
    save_soma_npz(
        out_path,
        np.zeros((1, 2, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        joint_names=["Root", "Hips"],
        identity_model_type="soma",
        identity_coeffs=np.zeros((1, 128), dtype=np.float32),
        scale_params=scale_params,
        keep_root=True,
    )

    loaded = load_soma_npz(out_path)

    np.testing.assert_array_equal(loaded.scale_params, scale_params)
