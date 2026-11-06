# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export joint orient must be sized to the rotations being exported (issue #18)."""

from types import SimpleNamespace

import pytest
import torch

from soma.geometry.rig_utils import precompute_joint_orient
from tools.conversion_utils import resolve_joint_orient


def _fake_procedural_layer(n_public=5, n_target=8):
    # public joints first, twist helpers appended; public parents form a chain
    t_pose_world = torch.eye(3).repeat(n_target, 1, 1)
    public_indices = torch.arange(n_public)
    public_parents = torch.tensor([0, 0, 1, 2, 3])
    target_parents = torch.cat([public_parents, torch.tensor([1, 2, 3])])
    orient, orient_parent_T = precompute_joint_orient(t_pose_world, target_parents)

    def public_joint_orient():
        return precompute_joint_orient(t_pose_world[public_indices], public_parents)

    return SimpleNamespace(
        _t_pose_orient=orient,
        _t_pose_orient_parent_T=orient_parent_T,
        rig_data={"joint_names": [f"t{i}" for i in range(n_target)]},
        public_joint_names=tuple(f"p{i}" for i in range(n_public)),
        public_joint_orient=public_joint_orient,
    )


def test_resolve_joint_orient_uses_public_orient_for_public_rotations():
    layer = _fake_procedural_layer()
    orient, orient_parent_T, names = resolve_joint_orient(layer, 5)
    assert orient.shape == (5, 3, 3) and orient_parent_T.shape == (5, 3, 3)
    assert names == ["p0", "p1", "p2", "p3", "p4"]


def test_resolve_joint_orient_keeps_target_orient_for_target_rotations():
    layer = _fake_procedural_layer()
    orient, _, names = resolve_joint_orient(layer, 8)
    assert orient.shape == (8, 3, 3)
    assert names == [f"t{i}" for i in range(8)]


def test_resolve_joint_orient_rejects_unknown_joint_count():
    with pytest.raises(ValueError):
        resolve_joint_orient(_fake_procedural_layer(), 7)
