# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from scipy.spatial.transform import Rotation

from soma.geometry.lbs import batch_rodrigues
from soma.geometry.transforms import (
    euler_xyz_to_matrix,
    matrix_to_quaternion_xyzw,
    matrix_to_quaternion_xyzw_stable,
    matrix_to_rotvec,
    project_rotations_to_so3,
    quaternion_conjugate_xyzw,
    quaternion_exp_xyzw,
    quaternion_log_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_twist_angle_xyzw,
    quaternion_xyzw_to_matrix,
    rotvec_to_matrix,
    single_axis_rotation_matrices,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_rotvec_to_matrix_matches_scipy(dtype, batched):
    rotvecs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5e-9],
            [0.0, 0.0, 5e-7],
            [0.0, 0.0, 1e-6],
            [1e-4, -2e-4, 3e-4],
            [0.2, -0.4, 0.7],
            [0.0, 0.0, torch.pi / 2],
            [torch.pi - 1e-5, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    expected = torch.tensor(Rotation.from_rotvec(rotvecs.numpy()).as_matrix(), dtype=dtype)
    if batched:
        actual = rotvec_to_matrix(rotvecs.reshape(2, 4, 3)).reshape(-1, 3, 3)
    else:
        actual = torch.stack([rotvec_to_matrix(vector) for vector in rotvecs])

    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        actual.transpose(-1, -2) @ actual,
        torch.eye(3, dtype=dtype).expand_as(actual),
        atol=tolerance,
        rtol=tolerance,
    )
    torch.testing.assert_close(
        torch.linalg.det(actual), torch.ones(8, dtype=dtype), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_matrix_to_rotvec_matches_scipy_for_small_and_regular_angles(dtype, batched):
    vectors = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5e-7],
        [1e-5, -2e-5, 3e-5],
        [0.0, 0.0, 1e-4],
        [0.0, 0.0, 9e-4],
        [0.0, 0.0, 1.1e-3],
        [0.2, -0.4, 0.7],
        [0.0, 0.0, torch.pi / 2],
    ]
    matrices = torch.tensor(Rotation.from_rotvec(vectors).as_matrix(), dtype=dtype)
    expected = torch.tensor(vectors, dtype=dtype)
    if batched:
        actual = matrix_to_rotvec(matrices.reshape(2, 4, 3, 3)).reshape(-1, 3)
    else:
        actual = torch.stack([matrix_to_rotvec(matrix) for matrix in matrices])

    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotvec_to_matrix_identity_jacobian(dtype):
    vector = torch.zeros(3, dtype=dtype, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(rotvec_to_matrix, vector)
    # Infinitesimal rotation about each axis acts by its cross product.
    basis = torch.eye(3, dtype=dtype)
    expected = torch.stack(
        [torch.linalg.cross(axis.expand_as(basis), basis, dim=-1).T for axis in basis],
        dim=-1,
    )

    torch.testing.assert_close(jacobian, expected)


def test_rotvec_to_matrix_gradcheck():
    vectors = torch.tensor(
        [[0.0, 0.0, 0.0], [1e-7, -2e-7, 3e-7], [0.2, -0.4, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(rotvec_to_matrix, (vectors,), eps=1e-8)


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


@pytest.mark.parametrize("converter", [matrix_to_quaternion_xyzw, matrix_to_quaternion_xyzw_stable])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrix_to_quaternion_round_trips_mixed_sign_pi_rotations(converter, dtype):
    axes = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 2.0, 3.0], [3.0, 1.0, -2.0]], dtype=dtype)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    # Symmetric matrices have no antisymmetric terms from which to infer axis signs.
    rotations = 2.0 * axes[..., :, None] * axes[..., None, :] - torch.eye(3, dtype=dtype)
    quaternions = converter(rotations)

    assert torch.all(quaternions[..., 3] >= 0.0)
    torch.testing.assert_close(quaternions.norm(dim=-1), torch.ones(3, dtype=dtype))
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(quaternions), rotations, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("converter", [matrix_to_quaternion_xyzw, matrix_to_quaternion_xyzw_stable])
def test_matrix_to_quaternion_round_trips_near_pi_and_mirrored_rotations(converter):
    axes = torch.tensor([[1.0, -2.0, 3.0], [-3.0, 1.0, 2.0]], dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    offsets = torch.tensor([-1e-3, -1e-5, -1e-7, 1e-7, 1e-5, 1e-3], dtype=torch.float64)
    rotations = batch_rodrigues(((torch.pi + offsets[:, None, None]) * axes).reshape(-1, 3))
    rotations = rotations.to(torch.float32).reshape(6, 2, 3, 3)
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    rotations = torch.stack((rotations, mirror @ rotations @ mirror))

    quaternions = converter(rotations)

    assert quaternions.shape == (2, 6, 2, 4)
    assert quaternions.dtype == rotations.dtype
    assert quaternions.device == rotations.device
    assert torch.all(quaternions[..., 3] >= 0.0)
    torch.testing.assert_close(
        quaternion_xyzw_to_matrix(quaternions), rotations, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("converter", [matrix_to_quaternion_xyzw, matrix_to_quaternion_xyzw_stable])
def test_matrix_to_quaternion_has_finite_gradients_at_identity_and_pi(converter):
    rotations = torch.stack(
        (
            torch.eye(3, dtype=torch.float64),
            torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64)),
            torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float64
            ),
        )
    ).requires_grad_(True)
    quaternions = converter(rotations)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=rotations.dtype)
    (quaternions * weights).sum().backward()

    assert torch.isfinite(rotations.grad).all()
    assert torch.all(rotations.grad.abs().sum(dim=(-2, -1)) > 0.0)


@pytest.mark.parametrize("converter", [matrix_to_quaternion_xyzw, matrix_to_quaternion_xyzw_stable])
def test_matrix_to_quaternion_gradcheck_away_from_branch_boundaries(converter):
    rotations = batch_rodrigues(
        torch.tensor(
            [[0.2, -0.4, 0.7], [2.7, 0.1, -0.2], [0.1, 2.7, -0.2], [0.1, -0.2, 2.7]],
            dtype=torch.float64,
        )
    ).requires_grad_(True)

    assert torch.autograd.gradcheck(converter, (rotations,))


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
