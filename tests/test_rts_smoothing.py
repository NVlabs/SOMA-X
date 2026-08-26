# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from soma.geometry.rig_utils import precompute_joint_orient
from soma.geometry.transforms import (
    matrix_to_quaternion_xyzw,
    quaternion_conjugate_xyzw,
    quaternion_exp_xyzw,
    quaternion_log_xyzw,
    quaternion_multiply_xyzw,
    quaternion_xyzw_to_matrix,
)
from soma.rts_smoothing import (
    RTSSmoothingConfig,
    RTSSmoothingGains,
    derive_smoothing_groups,
    euclidean_acceleration,
    rts_smooth_euclidean,
    rts_smooth_rotations,
    smooth_pose,
)


def _assert_valid_rotations(rotations: torch.Tensor) -> None:
    eye = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    should_be_eye = rotations.transpose(-2, -1) @ rotations
    assert torch.allclose(should_be_eye, eye, atol=1e-5)
    assert torch.allclose(
        torch.linalg.det(rotations),
        torch.ones_like(rotations[..., 0, 0]),
        atol=1e-5,
    )


def _geodesic_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    q_a = matrix_to_quaternion_xyzw(a)
    q_b = matrix_to_quaternion_xyzw(b)
    relative = quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_a), q_b)
    rotvec = quaternion_log_xyzw(relative.reshape(-1, 4)).reshape(relative.shape[:-1] + (3,))
    return torch.linalg.norm(rotvec, dim=-1)


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    return quaternion_xyzw_to_matrix(quaternion_exp_xyzw(rotvec))


def _slow_jointwise_euclidean(
    values: torch.Tensor,
    *,
    fps: float,
    gains: RTSSmoothingGains,
    joint_gains: dict[int, RTSSmoothingGains] | None = None,
) -> torch.Tensor:
    outs = []
    for joint_idx in range(values.shape[1]):
        joint_gain = gains if joint_gains is None else joint_gains.get(joint_idx, gains)
        outs.append(
            rts_smooth_euclidean(
                values[:, joint_idx : joint_idx + 1],
                fps=fps,
                gains=joint_gain,
            )
        )
    return torch.cat(outs, dim=1)


class _FakeSomaLayer:
    def __init__(self):
        self.public_joint_names = (
            "Root",
            "Hips",
            "LeftForeArm",
            "LeftHand",
            "LeftHandIndex1",
            "LeftHandPinkyEnd",
            "RightLeg",
        )
        self.public_joint_parent_ids = torch.tensor([0, 0, 1, 2, 3, 4, 1])
        self.output_joint_parent_ids = self.public_joint_parent_ids
        orient_rotvec = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, -0.1],
                [0.05, 0.0, 0.0],
                [0.0, 0.0, 0.03],
                [0.0, -0.05, 0.0],
            ]
        )
        self.t_pose_world = torch.eye(4).expand(7, 4, 4).clone()
        self.t_pose_world[:, :3, :3] = _rotvec_to_matrix(orient_rotvec)
        self.public_transform_joint_indices = torch.arange(7)


def test_so3_smoothing_preserves_identity_rotations():
    rotations = torch.eye(3).expand(6, 4, 3, 3).clone()
    smoothed = rts_smooth_rotations(rotations, fps=30.0, gains=RTSSmoothingGains())

    assert torch.allclose(smoothed, rotations)
    _assert_valid_rotations(smoothed)


def test_so3_smoothing_tracks_constant_angular_velocity():
    angles = torch.linspace(0.0, 0.6, 12)
    rotvec = torch.zeros(12, 1, 3)
    rotvec[:, 0, 2] = angles
    rotations = _rotvec_to_matrix(rotvec.reshape(-1, 3)).reshape(12, 1, 3, 3)

    smoothed = rts_smooth_rotations(
        rotations,
        fps=30.0,
        gains=RTSSmoothingGains(r_obs=1e-6, q_pos=1.0, q_vel=10.0),
    )

    assert float(_geodesic_distance(smoothed, rotations).max()) < 1e-3
    _assert_valid_rotations(smoothed)


def test_so3_smoothing_is_not_sensitive_to_quaternion_sign_flips():
    half = torch.tensor(0.35)
    base = torch.stack(
        [
            torch.tensor(0.0),
            torch.sin(half),
            torch.tensor(0.0),
            torch.cos(half),
        ]
    )
    quats = base.expand(10, 1, 4).clone()
    quats[1::2] *= -1.0
    rotations = quaternion_xyzw_to_matrix(quats)

    smoothed = rts_smooth_rotations(
        rotations,
        fps=30.0,
        gains=RTSSmoothingGains(r_obs=0.2, q_pos=0.01, q_vel=0.1),
    )

    assert float(_geodesic_distance(smoothed, rotations).max()) < 1e-4
    _assert_valid_rotations(smoothed)


def test_so3_smoothing_handles_near_pi_axis_extraction_case():
    angles = torch.linspace(torch.pi - 2e-4, torch.pi + 2e-4, 9, dtype=torch.float64)
    rotvec = torch.zeros(9, 1, 3, dtype=torch.float64)
    rotvec[:, 0, 0] = angles
    rotations = _rotvec_to_matrix(rotvec.reshape(-1, 3)).reshape(9, 1, 3, 3)

    smoothed = rts_smooth_rotations(
        rotations,
        fps=30.0,
        gains=RTSSmoothingGains(r_obs=1e-6, q_pos=1.0, q_vel=10.0),
    )

    assert torch.isfinite(smoothed).all()
    assert float(_geodesic_distance(smoothed, rotations).max()) < 1e-3
    _assert_valid_rotations(smoothed)


def test_euclidean_smoothing_reduces_root_acceleration():
    base = torch.linspace(0.0, 1.0, 12)
    jitter = torch.tensor(
        [0.0, 0.12, -0.12, 0.12, -0.12, 0.12, -0.12, 0.12, -0.12, 0.12, -0.12, 0.0]
    )
    root = torch.stack([base + jitter, torch.zeros_like(base), torch.zeros_like(base)], dim=-1)

    smoothed = rts_smooth_euclidean(
        root[:, None, :],
        fps=30.0,
        gains=RTSSmoothingGains(r_obs=0.5, q_pos=0.001, q_vel=0.001),
    )[:, 0]

    raw_accel = torch.linalg.norm(euclidean_acceleration(root), dim=-1).mean()
    smooth_accel = torch.linalg.norm(euclidean_acceleration(smoothed), dim=-1).mean()
    assert smooth_accel < raw_accel


def test_euclidean_grouped_vectorization_matches_jointwise_reference():
    values = torch.randn(14, 5, 3)
    default_gains = RTSSmoothingGains(r_obs=0.1, q_pos=0.2, q_vel=1.5)
    fast_gains = RTSSmoothingGains(r_obs=0.02, q_pos=0.2, q_vel=8.0)
    joint_gains = {1: fast_gains, 3: fast_gains}

    grouped = rts_smooth_euclidean(
        values,
        fps=24.0,
        gains=default_gains,
        joint_gains=joint_gains,
    )
    jointwise = _slow_jointwise_euclidean(
        values,
        fps=24.0,
        gains=default_gains,
        joint_gains=joint_gains,
    )

    assert torch.allclose(grouped, jointwise)


def test_smooth_pose_relative_convention_does_not_require_joint_orient():
    rotations = torch.eye(3).expand(5, 4, 3, 3).clone()
    root = torch.zeros(5, 3)

    smoothed_rot, smoothed_root = smooth_pose(
        rotations,
        root,
        rotation_convention="relative",
        joint_names=["Hips", "LeftHand", "LeftHandIndex1", "RightLeg"],
        fps=30.0,
    )

    assert torch.allclose(smoothed_rot, rotations)
    assert torch.allclose(smoothed_root, root)


def test_smooth_pose_absolute_convention_roundtrips_joint_orient():
    layer = _FakeSomaLayer()
    rel_rotvec = torch.zeros(5, 7, 3)
    rel_rotvec[:, :, 0] = torch.linspace(0.0, 0.2, 5)[:, None]
    relative = _rotvec_to_matrix(rel_rotvec.reshape(-1, 3)).reshape(5, 7, 3, 3)
    orient, orient_parent_t = precompute_joint_orient(
        layer.t_pose_world, layer.public_joint_parent_ids
    )
    absolute = orient_parent_t[None] @ relative @ orient[None]

    smoothed_abs, _ = smooth_pose(
        absolute,
        None,
        soma_layer=layer,
        rotation_convention="absolute",
        output_rotation_convention="absolute",
        fps=30.0,
    )
    smoothed_rel, _ = smooth_pose(
        relative,
        None,
        soma_layer=layer,
        rotation_convention="relative",
        output_rotation_convention="relative",
        fps=30.0,
    )
    roundtrip_rel = (
        orient_parent_t.transpose(-2, -1)[None] @ smoothed_abs @ orient.transpose(-2, -1)[None]
    )

    assert torch.allclose(roundtrip_rel, smoothed_rel, atol=1e-5)
    _assert_valid_rotations(smoothed_abs)


def test_smooth_pose_preserves_dtype_and_device_cpu():
    rotations = torch.eye(3, dtype=torch.float32).expand(4, 2, 3, 3).clone()
    root = torch.zeros(4, 3, dtype=torch.float32)

    smoothed_rot, smoothed_root = smooth_pose(
        rotations,
        root,
        rotation_convention="relative",
        fps=30.0,
    )

    assert smoothed_rot.dtype == rotations.dtype
    assert smoothed_root.dtype == root.dtype
    assert smoothed_rot.device == rotations.device
    assert smoothed_root.device == root.device


@pytest.mark.gpu
def test_smooth_pose_preserves_cuda_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rotations = torch.eye(3, device="cuda").expand(4, 2, 3, 3).clone()
    root = torch.zeros(4, 3, device="cuda")

    smoothed_rot, smoothed_root = smooth_pose(
        rotations,
        root,
        rotation_convention="relative",
        fps=30.0,
    )

    assert smoothed_rot.device.type == "cuda"
    assert smoothed_root.device.type == "cuda"


def test_joint_groups_derive_hands_from_names_without_forearms():
    layer = _FakeSomaLayer()

    groups = derive_smoothing_groups(layer.public_joint_names, layer.public_joint_parent_ids)
    name_to_idx = {name: idx for idx, name in enumerate(layer.public_joint_names)}

    assert name_to_idx["LeftHand"] in groups.hand_indices
    assert name_to_idx["LeftHandIndex1"] in groups.hand_indices
    assert name_to_idx["LeftHandPinkyEnd"] in groups.hand_indices
    assert name_to_idx["LeftForeArm"] not in groups.hand_indices
    assert name_to_idx["LeftForeArm"] in groups.limb_indices


def test_invalid_inputs_raise_clear_errors():
    rotations = torch.eye(3).reshape(1, 1, 3, 3)

    with pytest.raises(ValueError, match="fps must be positive"):
        rts_smooth_rotations(rotations, fps=0.0, gains=RTSSmoothingGains())
    with pytest.raises(ValueError, match="requires soma_layer"):
        smooth_pose(rotations, rotation_convention="absolute", fps=30.0)
    with pytest.raises(ValueError, match="Unknown RTS smoothing preset"):
        smooth_pose(rotations, rotation_convention="relative", preset="missing")


def test_config_can_disable_root_translation_smoothing():
    rotations = torch.eye(3).expand(4, 1, 3, 3).clone()
    root = torch.randn(4, 3)
    config = RTSSmoothingConfig(smooth_root_translation=False)

    _, smoothed_root = smooth_pose(
        rotations,
        root,
        rotation_convention="relative",
        config=config,
    )

    assert smoothed_root is root
