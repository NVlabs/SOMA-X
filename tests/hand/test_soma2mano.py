# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the SOMAHand-to-MANO prototype tool."""

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
MANO_DIR = ASSETS_DIR / "MANO"
TEST_DATA_DIR = MANO_DIR / "test_data"


def test_mano_joints_with_fingertips_convention():
    from soma.hand.mano import (
        MANO_FINGERTIP_VERTEX_IDS,
        MANO_JOINT_NAMES,
        MANO_JOINT_NAMES_WITH_FINGERTIPS,
        MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS,
        MANO_TIP_VERTEX_IDS,
        build_mano_joints_with_fingertips,
    )

    assert len(MANO_JOINT_NAMES) == 16
    assert len(MANO_JOINT_NAMES_WITH_FINGERTIPS) == 21
    assert len(MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS) == 21

    names = MANO_JOINT_NAMES_WITH_FINGERTIPS
    for finger in ("Index", "Middle", "Pinky", "Ring", "Thumb"):
        tip_name = f"{finger}Tip"
        tip_idx = names.index(tip_name)
        distal_idx = names.index(f"{finger}3")
        assert MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS[tip_idx] == distal_idx

    assert MANO_TIP_VERTEX_IDS["index"] == MANO_FINGERTIP_VERTEX_IDS["IndexTip"]

    max_tip_id = max(MANO_FINGERTIP_VERTEX_IDS.values())
    vertices = torch.zeros(2, max_tip_id + 1, 3)
    joints16 = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    for tip_id in MANO_FINGERTIP_VERTEX_IDS.values():
        vertices[:, tip_id] = torch.tensor([float(tip_id), 1.0, 2.0])

    joints21 = build_mano_joints_with_fingertips(vertices, joints16)

    assert joints21.shape == (2, 21, 3)
    assert torch.equal(joints21[:, names.index("Index1")], joints16[:, 1])
    assert torch.equal(
        joints21[:, names.index("ThumbTip")],
        vertices[:, MANO_FINGERTIP_VERTEX_IDS["ThumbTip"]],
    )


@pytest.fixture(scope="module")
def mano_assets():
    required = [
        MANO_DIR / "MANO_RIGHT.pkl",
        MANO_DIR / "MANO_LEFT.pkl",
        MANO_DIR / "base_hand_right.obj",
        MANO_DIR / "base_hand_left.obj",
        MANO_DIR / "SOMA_wrap_right.obj",
        MANO_DIR / "SOMA_wrap_left.obj",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Missing MANO assets: {missing}")
    return ASSETS_DIR


@pytest.fixture(scope="module")
def mano_test_file():
    files = sorted(TEST_DATA_DIR.glob("*.npz"))
    if not files:
        pytest.skip(f"No MANO test data in {TEST_DATA_DIR}")
    return files[0]


def test_load_mano_pkl_with_chumpy_compat(mano_assets):
    from soma.hand._smpl_family_loader import load_mano_pkl, mano_parent_ids

    mano = load_mano_pkl(mano_assets, "right")

    assert mano["v_template"].shape == (778, 3)
    assert mano["shapedirs"].shape == (778, 3, 10)
    assert mano["J_regressor"].shape == (16, 778)
    assert mano["weights"].shape == (778, 16)
    assert mano["faces"].shape[1] == 3
    assert len(mano_parent_ids(mano["kintree_table"])) == 16
    assert np.allclose(mano["weights"].sum(axis=1), 1.0, atol=1e-4)


def test_mano_rig_layer_contract(mano_assets):
    from soma.hand.mano import MANOLayer

    layer = MANOLayer(mano_assets, "right", device="cpu", mode="warp")

    assert layer.bind_shape.shape == (778, 3)
    assert layer.bind_pose_world.shape == (16, 4, 4)
    assert layer.joint_parent_ids.shape == (16,)
    assert layer.excluded_vert_ids.numel() == 0
    assert layer.batched_skinning.get_bone_indices() is not None

    poses = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 16, 3, 3)
    out = layer.pose(poses, pose2rot=False, absolute_pose=True, fk_only=True)
    assert out["joints"].shape == (1, 16, 3)
    assert out["transforms"].shape == (1, 16, 4, 4)


def test_hand_pose_inversion_uses_layer_output_unit(mano_assets, tmp_path, monkeypatch):
    from soma.fitting.pose_inversion import PoseInversion
    from soma.hand import SOMAHandLayer
    from soma.units import Unit

    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
    layer = SOMAHandLayer(
        data_root=str(mano_assets),
        hand_type="right",
        device="cpu",
        identity_model_type="mano",
        mode="warp",
        output_unit=Unit.METERS,
    )

    betas = torch.zeros(1, layer.num_shape_components)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 25, 3, 3)
    root_translation = torch.tensor([[0.02, -0.03, 1.25]], dtype=torch.float32)

    with torch.no_grad():
        layer.prepare_identity(betas)
        target = layer.pose(
            rotations,
            pose2rot=False,
            absolute_pose=True,
            global_translation=root_translation,
        )["vertices"]

    inv = PoseInversion(layer, low_lod=False)
    with torch.no_grad():
        inv.prepare_identity(betas)
    # This test checks the root-translation unit contract. Keep the single
    # analytical full pass: Lie-GN can trade a small global translation drift
    # for lower vertex error, which is valid for fitting but not for this test.
    result = inv.fit(
        target,
        body_iters=0,
        finger_iters=0,
        full_iters=1,
        lie_iters=0,
        batch_size=1,
    )

    assert torch.allclose(result["root_translation"], root_translation, atol=1e-3)
    assert result["per_vertex_error"].mean() < 3e-3


@pytest.mark.parametrize("hand_type", ["left", "right"])
def test_process_sequence_cpu_roundtrip_smoke(
    mano_assets, mano_test_file, tmp_path, monkeypatch, hand_type
):
    from tools.hand.soma2mano import load_mano_data, process_sequence

    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
    seq = load_mano_data(mano_test_file)[0]

    stats = process_sequence(
        seq=seq,
        seq_idx=0,
        hand_type=hand_type,
        data_root=mano_assets,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        mode="warp",
        bcd_iters=1,
        lie_iters=0,
        lie_lambda=1e-1,
        batch_size=1,
        max_frames=1,
        export_usd=False,
        render=False,
        image_size=256,
        fps=30,
        max_render_frames=1,
    )

    assert stats["frames"] == 1
    for key in (
        "soma_fit",
        "direct_mano_fit",
        "mano_fit",
        "full_roundtrip",
        "source_mesh_roundtrip",
        "direct_mano_translation_roundtrip",
        "source_mano_translation_roundtrip",
    ):
        assert np.isfinite(stats[key]["mean_mm"])
    assert np.isfinite(stats["direct_mano_rotation_roundtrip"]["mean_deg"])
    assert np.isfinite(stats["source_mano_rotation_roundtrip"]["mean_deg"])

    for key in ("soma_fit", "direct_mano_fit", "mano_fit", "full_roundtrip"):
        assert stats[key]["mean_mm"] < 25.0

    assert list(tmp_path.glob(f"{hand_type}_0_*_mano_params.npz"))
    assert list(tmp_path.glob(f"{hand_type}_0_*_stats.json"))
