# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from soma.body import SOMALayer
from soma.geometry.rig_utils import precompute_joint_orient
from soma.io import load_soma_npz
from soma.units import Unit
from tools.conversion_utils import export_soma_npz
from tools.convert_amass_to_soma import _save_conversion


def _export_layer(procedural, dtype):
    # Exercise real SOMALayer methods without loading identity model assets.
    layer = SOMALayer.__new__(SOMALayer)
    torch.nn.Module.__init__(layer)
    target_names = [
        "Root",
        "HelperA",
        "Hips",
        "LeftHip",
        "LeftKnee",
        "HelperB",
        "HelperC",
        "LeftFoot",
    ]
    parents = torch.tensor([0, 0, 0, 2, 2, 4, 3, 3])
    indices = torch.tensor([0, 2, 4, 3, 7])
    public_parents = torch.tensor([0, 0, 1, 1, 3])
    vectors = np.arange(24).reshape(8, 3) * 0.025
    vectors[0] = 0.0
    orientations = torch.tensor(Rotation.from_rotvec(vectors).as_matrix(), dtype=dtype)
    public_orientations = orientations[indices]
    layer._public_joint_names = np.array([target_names[i] for i in indices])
    if not procedural:
        orientations = public_orientations
        parents = public_parents
        target_names = list(layer._public_joint_names)
        indices = torch.arange(len(target_names))
    layer.t_pose_world = orientations
    layer.public_transform_joint_indices = indices
    layer.public_joint_parent_ids = public_parents
    layer._t_pose_orient, layer._t_pose_orient_parent_T = precompute_joint_orient(
        orientations, parents
    )
    layer.rig_data = {"joint_names": target_names}
    layer.identity_model_type = "soma"
    layer.output_unit = Unit.METERS
    return layer, public_orientations


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("procedural", [False, True])
def test_public_joint_orient_selects_public_order_and_parents(procedural, dtype):
    layer, expected = _export_layer(procedural, dtype)

    orient, parent_orient = layer.public_joint_orient()

    torch.testing.assert_close(orient, expected)
    torch.testing.assert_close(parent_orient, expected[[0, 0, 1, 1, 3]].transpose(-1, -2))
    assert orient.dtype == dtype
    assert orient.device == layer.t_pose_world.device


@pytest.mark.parametrize("exporter", ["shared", "amass"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("procedural", [False, True])
@pytest.mark.parametrize("keep_root", [False, True])
def test_npz_export_round_trips_public_rotations(tmp_path, exporter, dtype, procedural, keep_root):
    layer, orient = _export_layer(procedural, dtype)
    vectors = np.arange(30).reshape(2, 5, 3) * 0.01
    vectors[:, 0] = 0.0
    relative = torch.tensor(Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix(), dtype=dtype)
    relative = relative.reshape(2, 5, 3, 3)
    parent_orient = orient[[0, 0, 1, 1, 3]].transpose(-1, -2)
    rotations = parent_orient[None] @ relative @ orient[None]
    translation = torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.4, 0.8]], dtype=dtype)
    identity = torch.tensor([[0.1, 0.2]], dtype=dtype)
    errors = torch.tensor([[0.01, 0.02], [0.03, 0.04]], dtype=dtype)
    path = tmp_path / "motion.npz"

    if exporter == "shared":
        export_soma_npz(
            path,
            rotations,
            translation,
            layer,
            output_unit="centimeters",
            keep_root=keep_root,
            identity_coeffs=identity,
            extra_arrays={"per_vertex_error": errors},
        )
    else:
        _save_conversion(
            {
                "rotations": rotations,
                "root_transl": translation,
                "betas": identity,
                "per_vertex_error": errors,
            },
            SimpleNamespace(soma=layer),
            SimpleNamespace(output_unit="centimeters", keep_root=keep_root),
            path,
        )

    loaded = load_soma_npz(path)
    start = 0 if keep_root else 1
    assert loaded["joint_names"] == list(layer.public_joint_names[start:])
    np.testing.assert_allclose(loaded["poses"], vectors[:, start:], atol=1e-6)
    np.testing.assert_allclose(loaded["transl"], translation.numpy() * 100, atol=1e-6)
    np.testing.assert_allclose(loaded["joint_orient"], orient.numpy(), atol=1e-6)
    np.testing.assert_allclose(loaded["identity_coeffs"], identity.numpy())
    np.testing.assert_allclose(loaded["per_vertex_error"], errors.numpy())
    assert loaded["keep_root"] is keep_root
    assert loaded["absolute_pose"] is False
    assert loaded["rotation_repr"] == "rotvec"
    assert loaded["unit"] == "centimeters"
    assert loaded["identity_model_type"] == "soma"
