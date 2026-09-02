# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def _rotation_x(theta: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=theta.dtype, device=theta.device)
    one = torch.ones((), dtype=theta.dtype, device=theta.device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos_t, -sin_t]),
            torch.stack([zero, sin_t, cos_t]),
        ],
    )


def _rotation_z(theta: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=theta.dtype, device=theta.device)
    one = torch.ones((), dtype=theta.dtype, device=theta.device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([cos_t, -sin_t, zero]),
            torch.stack([sin_t, cos_t, zero]),
            torch.stack([zero, zero, one]),
        ],
    )


def _reflected_alignment_vectors(device: torch.device | str = "cpu"):
    source = torch.tensor(
        [
            [0.27950025, 0.90204942, -2.21577716],
            [0.55080229, 1.15626287, 0.40988064],
            [-0.75964117, -1.09359479, -0.82464337],
        ],
        device=device,
    )
    target = torch.tensor(
        [
            [1.23683381, 0.46486983, -1.38707960],
            [-0.89064091, -0.37236741, 0.49168995],
            [-1.53532076, -1.70456111, 0.36192819],
        ],
        device=device,
    )
    return target, source


def test_align_vectors_newton_schulz_falls_back_for_degenerate_covariance():
    from soma.geometry.transforms import align_vectors, kabsch

    target = torch.zeros(2, 3)
    source = torch.zeros(2, 3)

    rotation = align_vectors(target, source, method="newton-schulz")
    expected = kabsch(torch.zeros(3, 3))

    assert torch.isfinite(rotation).all()
    assert torch.allclose(rotation, expected)
    assert torch.allclose(rotation.mT @ rotation, torch.eye(3))
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0))


def test_align_vectors_auto_uses_identity_reference_for_rank_deficient_twist():
    from soma.geometry.transforms import align_vectors

    source = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    target = source.clone()

    rotation = align_vectors(target, source, method="auto")

    assert torch.allclose(rotation @ source.T, target.T, atol=1e-6)
    assert torch.allclose(rotation, torch.eye(3), atol=1e-5)


def test_align_vectors_auto_returns_valid_rotation_for_zero_covariance():
    from soma.geometry.transforms import align_vectors

    source = torch.zeros(3, 3)
    target = torch.zeros(3, 3)

    rotation = align_vectors(target, source, method="auto")

    assert torch.isfinite(rotation).all()
    assert torch.allclose(rotation.mT @ rotation, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(rotation, torch.eye(3), atol=1e-5)


def test_align_vectors_default_matches_auto():
    from soma.geometry.transforms import align_vectors

    source = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.7],
        ]
    )
    target = source @ _rotation_z(torch.tensor(0.3)).T

    assert torch.allclose(
        align_vectors(target, source), align_vectors(target, source, method="auto")
    )


def test_align_vectors_auto_rank_deficient_fallback_backpropagates():
    from soma.geometry.transforms import align_vectors

    source = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    target = source.detach().clone().requires_grad_(True)

    rotation = align_vectors(target, source, method="auto")
    loss = rotation.square().sum()
    loss.backward()

    assert target.grad is not None
    assert source.grad is not None
    assert torch.isfinite(target.grad).all()
    assert torch.isfinite(source.grad).all()


def test_align_vectors_auto_uses_kabsch_for_reflected_covariance():
    from soma.geometry.transforms import (
        AUTO_ROTATION_DEGENERATE_THRESHOLD,
        align_vectors,
        compute_covariance,
        kabsch,
        regularize_covariance_with_reference,
    )

    target, source = _reflected_alignment_vectors()
    covariance = compute_covariance(target, source)
    regularized_covariance = regularize_covariance_with_reference(
        covariance,
        rank_threshold=AUTO_ROTATION_DEGENERATE_THRESHOLD,
    )

    assert torch.linalg.det(regularized_covariance) < 0
    assert torch.allclose(
        align_vectors(target, source, method="auto"),
        kabsch(regularized_covariance),
        atol=1e-5,
    )


def test_auto_refit_alignment_matches_data_when_reference_agrees_for_full_rank():
    from soma.fitting.pose_inversion import _align_vectors_auto
    from soma.geometry.transforms import compute_covariance, newton_schulz

    source = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5],
            [-0.3, 0.2, 0.4],
        ]
    )
    rotation = _rotation_z(torch.tensor(0.7))
    target = source @ rotation.T

    expected = newton_schulz(compute_covariance(target, source))
    auto = _align_vectors_auto(target, source, expected)

    assert torch.allclose(auto, expected, atol=1e-6)


def test_auto_refit_alignment_uses_reference_for_rank_deficient_twist():
    from soma.fitting.pose_inversion import _align_vectors_auto

    source = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    target = source.clone()
    reference = _rotation_x(torch.tensor(1.1))

    auto = _align_vectors_auto(target, source, reference)

    assert torch.allclose(auto @ source.T, target.T, atol=1e-6)
    assert torch.allclose(auto, reference, atol=1e-5)


def test_auto_refit_alignment_uses_reference_for_zero_covariance():
    from soma.fitting.pose_inversion import _align_vectors_auto

    source = torch.zeros(3, 3)
    target = torch.zeros(3, 3)
    reference = _rotation_z(torch.tensor(-0.8))

    auto = _align_vectors_auto(target, source, reference)

    assert torch.isfinite(auto).all()
    assert torch.allclose(auto.mT @ auto, torch.eye(3), atol=1e-6)
    assert torch.allclose(torch.linalg.det(auto), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(auto, reference, atol=1e-5)


def test_auto_refit_alignment_uses_kabsch_for_reflected_covariance():
    from soma.fitting.pose_inversion import _AUTO_REFIT_PRIOR_STRENGTH, _align_vectors_auto
    from soma.geometry.transforms import (
        compute_covariance,
        kabsch,
        regularize_covariance_with_reference,
    )

    target, source = _reflected_alignment_vectors()
    reference = torch.eye(3)
    covariance = compute_covariance(target, source)
    regularized_covariance = regularize_covariance_with_reference(
        covariance,
        reference_rotation=reference,
        prior_strength=_AUTO_REFIT_PRIOR_STRENGTH,
    )

    assert torch.linalg.det(regularized_covariance) < 0
    assert torch.allclose(
        _align_vectors_auto(target, source, reference),
        kabsch(regularized_covariance),
        atol=1e-5,
    )


def test_align_vectors_newton_schulz_matches_newton_schulz_default_iterations():
    from soma.geometry.transforms import align_vectors, compute_covariance, newton_schulz

    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.1, 0.0, 0.5],
        ]
    )
    source = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 0.1, 0.5],
        ]
    )

    covariance = compute_covariance(target, source)

    assert torch.allclose(
        align_vectors(target, source, method="newton-schulz"),
        newton_schulz(covariance),
        atol=1e-6,
    )


def test_fused_refit_svd_matches_kabsch_alignment():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.geometry.fused_refit_warp import fused_refit_level
    from soma.geometry.transforms import align_vectors

    device = torch.device("cuda")
    source = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.2, 1.0, 0.3],
            [-0.4, 0.1, 0.8],
        ],
        device=device,
    )
    theta = torch.tensor(0.7, device=device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zero = torch.zeros((), device=device)
    one = torch.ones((), device=device)
    rotation = torch.stack(
        [
            torch.stack([cos_t, -sin_t, zero]),
            torch.stack([sin_t, cos_t, zero]),
            torch.stack([zero, zero, one]),
        ],
    )
    target = source @ rotation.T

    R_all = fused_refit_level(
        source,
        torch.ones(3, 1, device=device),
        torch.zeros(3, 1, dtype=torch.long, device=device),
        torch.zeros(3, 1, device=device),
        torch.zeros(3, 1, dtype=torch.long, device=device),
        torch.ones(3, device=device),
        torch.arange(3, device=device),
        torch.tensor([0], device=device),
        torch.tensor([3], device=device),
        torch.zeros(3, dtype=torch.long, device=device),
        torch.tensor([0], device=device),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        target.view(1, 3, 3),
        1,
        rotation_method="svd",
    )

    expected = align_vectors(target, source, method="kabsch")

    assert torch.allclose(R_all[0, 0], expected, atol=1e-4)


def test_fused_refit_auto_matches_pytorch_auto_alignment():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.fitting.pose_inversion import _align_vectors_auto
    from soma.geometry.fused_refit_warp import fused_refit_level

    device = torch.device("cuda")
    source = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.2, 1.0, 0.3],
            [-0.4, 0.1, 0.8],
        ],
        device=device,
    )
    theta = torch.tensor(0.7, device=device)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zero = torch.zeros((), device=device)
    one = torch.ones((), device=device)
    rotation = torch.stack(
        [
            torch.stack([cos_t, -sin_t, zero]),
            torch.stack([sin_t, cos_t, zero]),
            torch.stack([zero, zero, one]),
        ],
    )
    target = source @ rotation.T
    reference = rotation

    R_all = fused_refit_level(
        source,
        torch.ones(3, 1, device=device),
        torch.zeros(3, 1, dtype=torch.long, device=device),
        torch.zeros(3, 1, device=device),
        torch.zeros(3, 1, dtype=torch.long, device=device),
        torch.ones(3, device=device),
        torch.arange(3, device=device),
        torch.tensor([0], device=device),
        torch.tensor([3], device=device),
        torch.zeros(3, dtype=torch.long, device=device),
        torch.tensor([0], device=device),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        target.view(1, 3, 3),
        1,
        rotation_method="auto",
        reference_rotations=reference.view(1, 1, 3, 3),
    )

    expected = _align_vectors_auto(target, source, reference)

    assert torch.allclose(R_all[0, 0], expected, atol=1e-4)


def test_fused_refit_auto_uses_kabsch_for_reflected_covariance():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.fitting.pose_inversion import _align_vectors_auto
    from soma.geometry.fused_refit_warp import fused_refit_level

    device = torch.device("cuda")
    target, source = _reflected_alignment_vectors(device)
    reference = torch.eye(3, device=device)

    result = fused_refit_level(
        source,
        torch.ones(3, 1, device=device),
        torch.zeros(3, 1, dtype=torch.long, device=device),
        torch.zeros(3, 1, device=device),
        torch.zeros(3, 1, dtype=torch.long, device=device),
        torch.ones(3, device=device),
        torch.arange(3, device=device),
        torch.tensor([0], device=device),
        torch.tensor([3], device=device),
        torch.zeros(3, dtype=torch.long, device=device),
        torch.tensor([0], device=device),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        torch.eye(4, device=device).view(1, 1, 4, 4),
        target.view(1, 3, 3),
        1,
        rotation_method="auto",
        reference_rotations=reference.view(1, 1, 3, 3),
    )[0, 0]

    expected = _align_vectors_auto(target, source, reference)

    assert torch.allclose(result, expected, atol=1e-4)


def test_align_vectors_warp_auto_matches_torch_auto_and_backpropagates():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.geometry.align_vectors_warp import align_vectors_warp
    from soma.geometry.transforms import align_vectors

    device = torch.device("cuda")
    source = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.2],
            [-0.3, 0.1, 0.7],
            [0.4, -0.5, 0.2],
        ],
        device=device,
        requires_grad=True,
    )
    theta = torch.tensor(0.35, device=device)
    rotation = _rotation_z(theta)
    target = (source.detach() @ rotation.T).requires_grad_(True)

    offsets = torch.tensor([0], dtype=torch.int32, device=device)
    counts = torch.tensor([source.shape[0]], dtype=torch.int32, device=device)

    result = align_vectors_warp(target, source, offsets, counts, method="auto")[0]
    expected = align_vectors(target, source, method="auto")

    assert torch.allclose(result, expected, atol=1e-4)

    loss = result.square().sum()
    loss.backward()

    assert target.grad is not None
    assert source.grad is not None
    assert torch.isfinite(target.grad).all()
    assert torch.isfinite(source.grad).all()


def test_align_vectors_warp_auto_matches_torch_for_near_planar_reflection():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.geometry.align_vectors_warp import align_vectors_warp
    from soma.geometry.transforms import align_vectors

    device = torch.device("cuda")
    source = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.01],
        ],
        device=device,
    )
    target = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, -0.01],
        ],
        device=device,
    )
    target = target @ _rotation_z(torch.tensor(0.2, device=device)).T

    offsets = torch.tensor([0], dtype=torch.int32, device=device)
    counts = torch.tensor([source.shape[0]], dtype=torch.int32, device=device)

    result = align_vectors_warp(target, source, offsets, counts, method="auto")[0]
    expected = align_vectors(target, source, method="auto")

    assert torch.allclose(result, expected, atol=1e-4)


def test_align_vectors_warp_auto_uses_kabsch_for_reflected_covariance():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.geometry.align_vectors_warp import align_vectors_warp
    from soma.geometry.transforms import align_vectors

    device = torch.device("cuda")
    target, source = _reflected_alignment_vectors(device)
    target.requires_grad_(True)
    source.requires_grad_(True)
    offsets = torch.tensor([0], dtype=torch.int32, device=device)
    counts = torch.tensor([source.shape[0]], dtype=torch.int32, device=device)

    result = align_vectors_warp(target, source, offsets, counts, method="auto")[0]
    expected = align_vectors(target, source, method="kabsch")

    assert torch.allclose(result, expected, atol=1e-4)

    result.square().sum().backward()

    assert target.grad is not None
    assert source.grad is not None
    assert torch.isfinite(target.grad).all()
    assert torch.isfinite(source.grad).all()
