# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for private native-MHR pose inversion."""

from pathlib import Path

import pytest
import torch

from soma.fitting.pose_inversion_mhr import (
    _FLEXIBLE_SLICE,
    MHRPoseInversion,
    _redistribute_colocated_transforms,
)
from soma.geometry.transforms import euler_xyz_to_matrix, matrix_to_euler_xyz

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
MHR_DIR = ASSETS_DIR / "MHR"


@pytest.fixture(scope="module")
def mhr_inv():
    required = [
        MHR_DIR / "MHR_base_rig.npz",
        MHR_DIR / "parameter_transform.npz",
        MHR_DIR / "mhr_model_lod1.pt",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        pytest.skip(f"Missing MHR test assets: {missing}")
    return MHRPoseInversion(ASSETS_DIR, device="cpu", use_warp_for_rotations=False)


def test_native_mhr_rest_pose_inversion_smoke(mhr_inv):
    result = mhr_inv.fit(
        mhr_inv.bind_shape.unsqueeze(0),
        body_iters=0,
        finger_iters=0,
        full_iters=0,
        refine_iters=0,
    )

    assert result["pose_params"].shape == (1, 136)
    assert result["model_params"].shape == (1, 204)
    assert torch.isfinite(result["pose_params"]).all()
    assert torch.isfinite(result["per_vertex_error"]).all()
    assert result["pre_refine_per_vertex_error"].mean().item() < 0.05
    assert torch.allclose(result["pose_params"][:, _FLEXIBLE_SLICE], torch.zeros(1, 6))


def test_mhr_refinement_keeps_best_pose_and_freezes_flexibles(mhr_inv):
    model = mhr_inv._load_mhr_model()
    identity = torch.zeros(1, 45)
    scale = torch.zeros(1, 68)
    face = torch.zeros(1, 72)
    pose = torch.zeros(1, 136)
    pose[0, 3] = 0.05
    pose[0, 10] = 0.08

    with torch.no_grad():
        target, _ = model(identity, torch.cat([pose, scale], dim=1), face, False)

    result = mhr_inv.fit(
        target,
        identity_coeffs=identity,
        scale_params=scale,
        face_expr_coeffs=face,
        body_iters=0,
        finger_iters=0,
        full_iters=0,
        refine_iters=5,
        lr=1e-4,
        optimize_flexibles=False,
    )

    assert len(result["loss_history"]) == 5
    assert torch.allclose(result["pose_params"][:, _FLEXIBLE_SLICE], torch.zeros(1, 6))

    with torch.no_grad():
        pred, _ = model(identity, result["model_params"], face, False)
    final_mse = torch.nn.functional.mse_loss(pred, target).item()
    assert final_mse <= result["loss_history"][0] + 1e-8


def test_mhr_pose_params_to_local_transforms_rest_pose(mhr_inv):
    pose_local = mhr_inv.pose_params_to_local_transforms(torch.zeros(1, 136))

    assert pose_local.shape == (1, len(mhr_inv.joint_names), 4, 4)
    assert torch.allclose(pose_local[0, :, :3, :3], mhr_inv.bind_local[:, :3, :3], atol=1e-5)
    assert torch.allclose(
        pose_local[0, mhr_inv.root_joint_idx, :3, 3],
        mhr_inv.bind_local[mhr_inv.root_joint_idx, :3, 3],
        atol=1e-5,
    )


def test_mhr_model_skeleton_state_to_local_transforms_rest_pose(mhr_inv):
    model = mhr_inv._load_mhr_model()
    identity = torch.zeros(1, 45)
    scale = torch.zeros(1, 68)
    face = torch.zeros(1, 72)
    pose = torch.zeros(1, 136)

    with torch.no_grad():
        _, skel_state = model(identity, torch.cat([pose, scale], dim=1), face, False)
    pose_local = mhr_inv.model_skeleton_state_to_local_transforms(skel_state)

    assert pose_local.shape == (1, len(mhr_inv.joint_names), 4, 4)
    assert torch.allclose(pose_local[0, :, :3, :3], mhr_inv.bind_local[:, :3, :3], atol=1e-4)
    assert torch.allclose(pose_local[0, :, :3, 3], mhr_inv.bind_local[:, :3, 3], atol=1e-4)


def test_colocated_ankle_distribution_preserves_combined_rotation(mhr_inv):
    B = 2
    pose_local = mhr_inv.bind_local.unsqueeze(0).expand(B, -1, -1, -1).clone()
    for side in ("l", "r"):
        foot_idx = mhr_inv.joint_names.index(f"{side}_foot")
        talo_idx = mhr_inv.joint_names.index(f"{side}_talocrural")
        pose_local[:, foot_idx, :3, :3] = euler_xyz_to_matrix(
            torch.tensor([[0.3, -0.2, 0.4], [-0.1, 0.2, -0.5]])
        )
        pose_local[:, talo_idx, :3, :3] = euler_xyz_to_matrix(
            torch.tensor([[0.2, 0.1, -0.3], [0.5, -0.2, 0.1]])
        )
        before = pose_local[:, foot_idx, :3, :3] @ pose_local[:, talo_idx, :3, :3]
        _redistribute_colocated_transforms(pose_local, mhr_inv.joint_names, mhr_inv.pt_data)
        after = pose_local[:, foot_idx, :3, :3] @ pose_local[:, talo_idx, :3, :3]
        assert torch.allclose(after, before, atol=1e-5)


def test_euler_xyz_round_trip():
    euler = torch.tensor([[0.1, 0.2, -0.3], [0.5, -0.4, 0.25]])
    recovered = matrix_to_euler_xyz(euler_xyz_to_matrix(euler))
    assert torch.allclose(recovered, euler, atol=1e-5)
