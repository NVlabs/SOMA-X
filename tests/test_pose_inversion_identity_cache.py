# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from soma.fitting import pose_inversion


class _DifferentiableIdentityLayer:
    def __init__(self):
        self.identity_weight = torch.nn.Parameter(torch.tensor(2.0))
        self.joint_parent_ids = torch.tensor([0, 0])
        self.skinning_weights = torch.full((3, 2), 0.5)
        self.t_pose_world = torch.eye(4).repeat(2, 1, 1)
        self.rig_data = {"joint_names": ["Root", "Hips"]}
        self.calls = 0

    def prepare_identity(self, identity_coeffs, scale_params, *, repose_to_bind_pose, kwargs):
        self.calls += 1
        value = identity_coeffs * self.identity_weight.square()
        self._cached_rest_shape = value.unsqueeze(-1).expand(-1, 3, 3)
        translation = torch.zeros(2, 4, 4)
        translation[1, 0, 3] = 1.0
        self._cached_bind_transforms_world = (
            self.t_pose_world[None] + value[..., None, None] * translation
        )


@pytest.mark.parametrize("autograd_layer", ["none", "shared", "separate"])
@pytest.mark.parametrize("public_rig", [False, True])
def test_pose_inversion_identity_caches_support_repeated_backward(
    monkeypatch, autograd_layer, public_rig
):
    layer = _DifferentiableIdentityLayer()
    other_layer = _DifferentiableIdentityLayer() if autograd_layer == "separate" else layer
    if public_rig:
        layer.public_rig_view = lambda: SimpleNamespace(
            bind_transforms_world=layer._cached_bind_transforms_world,
            joint_parent_ids=layer.joint_parent_ids,
            skinning_weights=layer.skinning_weights,
            t_pose_world=layer.t_pose_world,
        )

    inv = pose_inversion.PoseInversion.__new__(pose_inversion.PoseInversion)
    inv.soma = layer
    inv._autograd_soma = None if autograd_layer == "none" else other_layer
    inv._root_joint_idx = 1
    inv._cache = {}
    updated_binds = []
    inv._skel_transfer = SimpleNamespace(update_bind=lambda *args: updated_binds.append(args))
    # Structural geometry is independent of this identity-cache lifetime test.
    monkeypatch.setattr(pose_inversion, "_precompute_refit_cache", lambda *a, **kw: {})

    layers = [layer, other_layer] if autograd_layer == "separate" else [layer]
    for value in (1.0, 2.0):
        coefficients = torch.tensor([[value]], requires_grad=True)
        inv.prepare_identity(coefficients)
        pose = torch.nn.Parameter(torch.tensor(0.5))
        optimizer = torch.optim.Adam([pose], lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = sum((pose * item._cached_rest_shape).square().mean() for item in layers)
            loss.backward()
            assert torch.isfinite(pose.grad)
            optimizer.step()

        assert pose.item() < 0.5
        assert coefficients.grad is None
        for item in layers:
            assert item.identity_weight.grad is None
            assert not item._cached_rest_shape.requires_grad
            assert not item._cached_bind_transforms_world.requires_grad
            torch.testing.assert_close(item._cached_rest_shape, torch.full((1, 3, 3), value * 4))

    assert len(updated_binds) == 2
    assert all(item.calls == 2 for item in layers)
    # Direct layer preparation remains differentiable outside PoseInversion.
    layer.prepare_identity(coefficients, None, repose_to_bind_pose=True, kwargs=None)
    layer._cached_rest_shape.sum().backward()
    assert coefficients.grad is not None
    assert layer.identity_weight.grad is not None
