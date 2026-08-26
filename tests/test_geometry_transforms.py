# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from soma.geometry.lbs import batch_rodrigues
from soma.geometry.transforms import (
    euler_xyz_to_matrix,
    matrix_to_quaternion_xyzw,
    matrix_to_quaternion_xyzw_stable,
    project_rotations_to_so3,
    quaternion_conjugate_xyzw,
    quaternion_exp_xyzw,
    quaternion_log_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_twist_angle_xyzw,
    quaternion_xyzw_to_matrix,
    single_axis_rotation_matrices,
)


def test_matrix_to_quaternion_xyzw_round_trips_rotation_matrices():
    rotations = euler_xyz_to_matrix(
        torch.tensor(
            [
                [0.2, -0.4, 0.7],
                [-1.1, 0.3, 0.9],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    quaternions = matrix_to_quaternion_xyzw(rotations)
    recovered = quaternion_xyzw_to_matrix(quaternions)

    assert quaternions.shape == (3, 4)
    assert torch.all(quaternions[..., 3] >= 0.0)
    torch.testing.assert_close(recovered, rotations, atol=1e-6, rtol=1e-6)


def test_matrix_to_quaternion_xyzw_handles_axis_pi_rotations():
    rotations = batch_rodrigues(
        torch.tensor(
            [
                [torch.pi, 0.0, 0.0],
                [0.0, torch.pi, 0.0],
                [0.0, 0.0, torch.pi],
            ],
            dtype=torch.float32,
        )
    )

    quaternions = matrix_to_quaternion_xyzw(rotations)
    recovered = quaternion_xyzw_to_matrix(quaternions)

    assert torch.isfinite(quaternions).all()
    assert torch.all(quaternions[..., 3] >= 0.0)
    torch.testing.assert_close(recovered, rotations, atol=1e-6, rtol=1e-6)


def test_matrix_to_quaternion_xyzw_stable_matches_standard_converter():
    rotations = euler_xyz_to_matrix(
        torch.tensor(
            [
                [0.2, -0.4, 0.7],
                [-1.1, 0.3, 0.9],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    expected = matrix_to_quaternion_xyzw(rotations)
    actual = matrix_to_quaternion_xyzw_stable(rotations)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_single_axis_rotation_matrices_match_rodrigues():
    angles = torch.tensor([[0.7, -0.5, 1.2]], dtype=torch.float32)
    signs = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)

    for axis in (0, 1, 2):
        rotvec = torch.zeros(3, 3, dtype=torch.float32)
        rotvec[:, axis] = angles[0] * signs
        expected = batch_rodrigues(rotvec).unsqueeze(0)
        actual = single_axis_rotation_matrices(angles, axis, signs)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_quaternion_xyzw_multiply_matches_matrix_composition():
    left = euler_xyz_to_matrix(torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32))
    right = euler_xyz_to_matrix(torch.tensor([[-0.6, 0.3, 0.1]], dtype=torch.float32))

    q_left = matrix_to_quaternion_xyzw(left)
    q_right = matrix_to_quaternion_xyzw(right)
    q_composed = quaternion_multiply_xyzw(q_left, q_right)

    recovered = quaternion_xyzw_to_matrix(q_composed)
    torch.testing.assert_close(recovered, left @ right, atol=1e-6, rtol=1e-6)


def test_quaternion_xyzw_conjugate_is_inverse_for_unit_quaternions():
    rotations = euler_xyz_to_matrix(
        torch.tensor([[0.2, -0.4, 0.7], [-1.1, 0.3, 0.9]], dtype=torch.float32)
    )
    quaternions = matrix_to_quaternion_xyzw(rotations)
    identity = quaternion_multiply_xyzw(quaternions, quaternion_conjugate_xyzw(quaternions))

    torch.testing.assert_close(
        quaternion_normalize_xyzw(identity),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).expand_as(identity),
        atol=1e-6,
        rtol=1e-6,
    )


def test_quaternion_xyzw_log_exp_round_trips_shortest_arc():
    rotvecs = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [torch.pi - 1e-5, 0.0, 0.0],
            [0.0, -torch.pi + 1e-5, 0.0],
        ],
        dtype=torch.float64,
    )

    quaternions = quaternion_exp_xyzw(rotvecs)
    recovered = quaternion_log_xyzw(quaternions)

    torch.testing.assert_close(recovered, rotvecs, atol=1e-9, rtol=1e-9)


def test_quaternion_xyzw_log_uses_short_arc_across_sign_flips():
    rotvec = torch.tensor([[0.0, 0.0, 0.25]], dtype=torch.float64)
    quaternion = quaternion_exp_xyzw(rotvec)

    recovered = quaternion_log_xyzw(-quaternion)

    torch.testing.assert_close(recovered, rotvec, atol=1e-12, rtol=1e-12)


def test_project_rotations_to_so3_fixes_reflections():
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))

    projected = project_rotations_to_so3(reflection)

    torch.testing.assert_close(projected.transpose(-2, -1) @ projected, torch.eye(3).double())
    torch.testing.assert_close(torch.linalg.det(projected), torch.tensor(1.0, dtype=torch.float64))


def test_quaternion_twist_angle_xyzw_extracts_per_axis_twist():
    angles = torch.tensor([0.7, -0.5, 1.2], dtype=torch.float32)
    rotations = batch_rodrigues(
        torch.tensor(
            [
                [angles[0], 0.0, 0.0],
                [0.0, angles[1], 0.0],
                [0.0, 0.0, angles[2]],
            ],
            dtype=torch.float32,
        )
    )
    quaternions = matrix_to_quaternion_xyzw(rotations)
    extracted = quaternion_twist_angle_xyzw(quaternions, torch.tensor([0, 1, 2]))

    torch.testing.assert_close(extracted, angles, atol=1e-6, rtol=1e-6)
