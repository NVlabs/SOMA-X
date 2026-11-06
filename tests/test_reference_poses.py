# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference conventions must affect articulation without changing bind geometry."""

import hashlib
import json
import shutil
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from soma import SOMALayer
from soma.geometry.batched_skinning import BatchedSkinning, FKTopology
from soma.geometry.lbs import batch_rodrigues
from soma.geometry.rig_utils import precompute_joint_orient
from soma.procedural_transforms import SOMAProceduralParameterTransform
from soma.reference_poses import ReferencePoseHistory, validate_reference_pose
from tests.test_procedural_transforms import (
    PROCEDURAL_TRANSFORM_DEFINITION,
    SOMA_TWIST_SEGMENTS,
    _minimal_source_names,
    _target_names,
    _target_t_pose_world,
)


class _PoseCorrectives(torch.nn.Module):
    """Small pose-dependent offsets expose conversion errors before skinning."""

    def __init__(self, num_vertices):
        super().__init__()
        self.num_vertices = num_vertices

    def forward(self, rotations):
        offsets = rotations[:, 2:, :, 0].mean(dim=1) * 0.02
        return {"out": offsets[:, None].expand(-1, self.num_vertices, -1)}


class _TinyLayer(SOMALayer):
    """Real pose/FK/LBS pipeline with a tiny deterministic identity fixture."""

    def prepare_identity(self, identity_coeffs, scale_params=None, **kwargs):
        batch = len(identity_coeffs)
        self._cached_rest_shape = self.bind_shape.unsqueeze(0).expand(batch, -1, -1)
        self._cached_bind_transforms_world = self.bind_pose_world.unsqueeze(0).expand(
            batch, -1, -1, -1
        )
        self._cached_scale_params = None
        self._cached_global_scale = kwargs.get("global_scale", 1.0)


def _tiny_layer(procedural=False, device="cpu", dtype=torch.float32):
    layer = _TinyLayer.__new__(_TinyLayer)
    torch.nn.Module.__init__(layer)
    layer.register_buffer("_default_reference_pose", None, persistent=False)
    names = _minimal_source_names()
    target_names = _target_names(names) if procedural else names
    parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12]
    target_parents = parents.copy()
    if procedural:
        target_parents.extend(
            names.index(segment.start_joint)
            for segment in SOMA_TWIST_SEGMENTS
            for _ in segment.twist_joints
        )
    world = _target_t_pose_world(names, _target_names(names))[: len(target_names)].to(dtype=dtype)
    # Construct saved-reference fixtures on CPU independently of CUDA matmul precision.
    angles = torch.arange(len(names) * 3, dtype=dtype).reshape(-1, 3) * 0.012
    angles[0] = 0
    world[: len(names), :3, :3] = batch_rodrigues(angles)
    for idx in range(len(names), len(target_names)):
        world[idx, :3, :3] = world[target_parents[idx], :3, :3]
    world = world.to(device)
    layer.device = torch.device(device)
    layer._public_joint_names = np.array(names)
    layer.identity_model_type = "soma"
    layer.identity_lod_transfer = None
    layer.public_joint_parent_ids = torch.tensor(parents, device=device)
    layer.public_transform_joint_indices = torch.arange(len(names), device=device)
    layer.joint_parent_ids = torch.tensor(target_parents, device=device)
    layer.t_pose_world = world
    layer.bind_pose_world = world.clone()
    layer.bind_shape = world[:, :3, 3] + world.new_tensor([0.1, 0.2, -0.15])
    layer.procedural_transforms = None
    source_fk = None
    if procedural:
        layer.procedural_transforms = SOMAProceduralParameterTransform(
            names,
            target_names,
            rotation_extraction_modes=PROCEDURAL_TRANSFORM_DEFINITION.rotation_extraction_modes,
            segments=tuple(replace(segment, parent_joint=None) for segment in SOMA_TWIST_SEGMENTS),
            rotation_entries=PROCEDURAL_TRANSFORM_DEFINITION.rotation_entries,
            translation_entries=PROCEDURAL_TRANSFORM_DEFINITION.translation_entries,
            target_t_pose_world=world.cpu(),
            target_joint_parent_ids=layer.joint_parent_ids.cpu(),
            target_bind_pose_world=world.cpu(),
        ).to(device=device, dtype=dtype)
        source_fk = FKTopology(
            parent_ids=parents,
            target_joint_indices=layer.public_transform_joint_indices,
            joint_orient=world[: len(names)],
            bind_world_transforms=world[: len(names)],
        )
    layer.batched_skinning = BatchedSkinning(
        target_parents,
        torch.eye(len(target_names), device=device, dtype=dtype),
        world,
        layer.bind_shape,
        joint_orient=world,
        mode="dense",
        source_fk=source_fk,
    )
    layer._t_pose_orient, layer._t_pose_orient_parent_T = precompute_joint_orient(
        world, target_parents
    )
    layer.correctives_model = _PoseCorrectives(len(target_names))
    layer.prepare_identity(torch.zeros(1, 1, device=device))
    return layer


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["legacy", "procedural"])
def layer(request, device):
    return _tiny_layer(request.param, device)


@pytest.fixture(params=[False, True], ids=["legacy", "procedural"])
def precision_layer(request, device):
    # Algebraic and gradient oracles compare different multiplication orders.
    # Float64 isolates conversion correctness from reduced-precision TF32 GEMMs.
    return _tiny_layer(request.param, device, dtype=torch.float64)


def _inputs(layer, batch=2):
    count = len(layer.public_joint_names)
    dtype = layer.bind_pose_world.dtype
    angles = torch.linspace(-0.35, 0.4, batch * (count - 1) * 3, device=layer.device, dtype=dtype)
    angles = angles.reshape(batch, count - 1, 3)
    reference_angles = torch.linspace(0.5, -0.25, count * 3, dtype=dtype).reshape(count, 3)
    reference_angles[0] = 0
    return angles, batch_rodrigues(reference_angles, dtype=dtype).to(layer.device)


def _world_fk_oracle(relative, reference, parents):
    """Accumulate motion deltas in world axes, then recover parent-local rotations.

    This deliberately avoids SOMA's orientation helpers and their direct local
    sandwich formula. Noncommuting parent and child rotations expose frame errors.
    """
    count = len(parents)
    motion_world = [
        torch.eye(3, device=relative.device, dtype=relative.dtype).expand(len(relative), 3, 3)
    ]
    world = [motion_world[0]]
    for joint in range(1, count):
        motion_world.append(motion_world[parents[joint]] @ relative[:, joint - 1])
        world.append(motion_world[joint] @ reference[joint])
    local = [torch.linalg.solve(world[parents[j]], world[j]) for j in range(1, count)]
    return torch.stack(local, dim=1), torch.stack(world, dim=1)


def _assert_output_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("matrix_input", [False, True])
def test_reference_matches_independent_world_fk(precision_layer, matrix_input):
    layer = precision_layer
    angles, reference = _inputs(layer)
    rotations = batch_rodrigues(angles.reshape(-1, 3)).reshape(*angles.shape[:2], 3, 3)
    absolute, expected_world = _world_fk_oracle(
        rotations, reference, layer.output_joint_parent_ids.tolist()
    )
    translation = angles.new_tensor([[0.1, -0.3, 0.25], [-0.2, 0.4, 0.6]])
    correctives = layer.procedural_transforms is not None
    expected = layer.pose(
        absolute,
        transl=translation,
        pose2rot=False,
        absolute_pose=True,
        apply_correctives=correctives,
    )
    actual = layer.pose(
        rotations if matrix_input else angles,
        transl=translation,
        pose2rot=not matrix_input,
        reference_pose=reference,
        apply_correctives=correctives,
    )
    _assert_output_close(actual, expected)
    torch.testing.assert_close(
        actual["transforms"][..., :3, :3], expected_world, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(actual["joints"][:, 0], translation)
    fk = layer.pose(angles, transl=translation, reference_pose=reference, fk_only=True)
    assert "vertices" not in fk
    torch.testing.assert_close(fk["transforms"], actual["transforms"])


def test_current_reference_default_and_state_isolation(layer):
    angles, other_reference = _inputs(layer)
    current = layer.t_pose_world[layer.public_transform_joint_indices].clone()
    original = layer.t_pose_world.clone()
    options = {"apply_correctives": layer.procedural_transforms is not None}
    default = layer.pose(angles, **options)
    _assert_output_close(layer.pose(angles, reference_pose=None, **options), default)
    _assert_output_close(layer.pose(angles, reference_pose=current, **options), default)
    layer.pose(angles, reference_pose=other_reference, **options)
    _assert_output_close(layer.pose(angles, **options), default)
    torch.testing.assert_close(layer.t_pose_world, original, rtol=0, atol=0)
    translated = current.clone()
    translated[:, :3, 3] += 100
    _assert_output_close(layer.pose(angles, reference_pose=translated, **options), default)


@pytest.mark.parametrize("identity_batch,pose_batch", [(1, 2), (2, 1)])
@pytest.mark.parametrize("custom_reference", [False, True])
def test_forward_and_identity_broadcast(layer, identity_batch, pose_batch, custom_reference):
    angles, reference = _inputs(layer, batch=pose_batch)
    if not custom_reference:
        reference = None
    identity = torch.zeros(identity_batch, 1, device=layer.device)
    options = {"apply_correctives": layer.procedural_transforms is not None}
    layer.prepare_identity(identity)
    expected = layer.pose(angles, reference_pose=reference, **options)
    actual = layer(angles, identity, reference_pose=reference, **options)
    _assert_output_close(actual, expected)
    assert actual["vertices"].shape[0] == 2


def test_reference_and_pose_gradients(precision_layer):
    layer = precision_layer
    angles, _ = _inputs(layer, batch=1)
    angles = angles.detach().requires_grad_()
    reference_angles = torch.linspace(
        -0.2, 0.3, len(layer.public_joint_names) * 3, device=layer.device, dtype=torch.float64
    ).reshape(-1, 3)
    reference_angles = reference_angles.detach().requires_grad_()
    reference = torch.cat(
        (
            torch.eye(3, device=layer.device, dtype=torch.float64)[None],
            batch_rodrigues(reference_angles[1:]),
        ),
        dim=0,
    )
    out = layer.pose(
        angles, reference_pose=reference, apply_correctives=layer.procedural_transforms is not None
    )
    weights = torch.linspace(0.2, 1.3, out["vertices"].numel(), device=layer.device)
    loss = (out["vertices"].reshape(-1) * weights).sum()
    actual = torch.autograd.grad(loss, (angles, reference_angles), retain_graph=True)
    assert all(torch.isfinite(gradient).all() for gradient in actual)
    assert all(gradient.abs().max() > 1e-4 for gradient in actual)
    rotations = batch_rodrigues(angles.reshape(-1, 3)).reshape(1, -1, 3, 3)
    absolute, _ = _world_fk_oracle(
        rotations, reference.to(angles), layer.output_joint_parent_ids.tolist()
    )
    expected_out = layer.pose(
        absolute,
        pose2rot=False,
        absolute_pose=True,
        apply_correctives=layer.procedural_transforms is not None,
    )
    expected_loss = (expected_out["vertices"].reshape(-1) * weights).sum()
    expected = torch.autograd.grad(expected_loss, (angles, reference_angles))
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "problem",
    ["batch", "count", "shape", "nan", "inf", "scale", "reflection", "shear", "root", "type"],
)
def test_invalid_reference(layer, problem):
    angles, reference = _inputs(layer)
    if problem == "batch":
        reference = reference[None]
    elif problem == "count":
        reference = reference[1:]
    elif problem == "shape":
        reference = reference[..., :2]
    elif problem in ("nan", "inf"):
        reference[2, 0, 0] = float(problem)
    elif problem == "scale":
        reference[2] *= 1.1
    elif problem == "reflection":
        reference[2, :, 0] *= -1
    elif problem == "shear":
        reference[2, 0, 1] += 0.2
    elif problem == "root":
        reference[0] = reference[2]
    elif problem == "type":
        reference = reference.cpu().numpy()
    error = TypeError if problem == "type" else ValueError
    with pytest.raises(error, match="reference_pose"):
        layer.pose(angles, reference_pose=reference, apply_correctives=False)


def test_conflicting_flags_rejected_before_identity_update(layer):
    angles, reference = _inputs(layer)
    cached_shape = layer._cached_rest_shape
    cached_bind = layer._cached_bind_transforms_world
    with pytest.raises(ValueError, match="absolute_pose"):
        layer(
            angles,
            torch.zeros(3, 1, device=layer.device),
            reference_pose=reference,
            absolute_pose=True,
            apply_correctives=False,
        )
    assert layer._cached_rest_shape is cached_shape
    assert layer._cached_bind_transforms_world is cached_bind


_REFERENCE_ID = "t_pose_world__v0.1.0"
_ASSET_REVISION = "sha256:" + "a" * 64
_ASSET_ID = "t_pose_world__sha256_" + "a" * 64


def _catalog_data(layer):
    _, reference = _inputs(layer)
    metadata = {
        "schema_version": 2,
        "conventions": {"orientation_space": "world"},
        "default_reference_id": _REFERENCE_ID,
        "references": [
            {
                "id": _REFERENCE_ID,
                "soma_version": "0.1.0",
                "data_key": "t_pose_world",
                "description": "Synthetic reference",
            }
        ],
    }
    return {
        "reference_pose_history_metadata": np.array(json.dumps(metadata)),
        _REFERENCE_ID: reference.cpu().numpy(),
        _REFERENCE_ID + "__joint_names": np.array(layer.public_joint_names),
        _REFERENCE_ID + "__parent_ids": layer.output_joint_parent_ids.cpu().numpy(),
    }


def test_catalog_lookup_reorders_and_returns_independent_values(layer):
    data = _catalog_data(layer)
    expected = torch.tensor(data[_REFERENCE_ID], device=layer.device)
    order = np.array([0, *range(len(layer.public_joint_names) - 1, 0, -1)])
    inverse = np.argsort(order)
    data[_REFERENCE_ID] = data[_REFERENCE_ID][order]
    data[_REFERENCE_ID + "__joint_names"] = data[_REFERENCE_ID + "__joint_names"][order]
    data[_REFERENCE_ID + "__parent_ids"] = inverse[data[_REFERENCE_ID + "__parent_ids"][order]]
    layer._reference_pose_history = ReferencePoseHistory(data)
    records = layer.list_reference_poses()
    assert records[0]["id"] == _REFERENCE_ID
    records[0]["description"] = "changed by caller"
    assert layer.list_reference_poses()[0]["description"] == "Synthetic reference"
    actual = layer.get_reference_pose(_REFERENCE_ID)
    torch.testing.assert_close(actual, expected)
    assert actual.device == layer.bind_pose_world.device
    assert actual.dtype == layer.bind_pose_world.dtype
    actual.zero_()
    torch.testing.assert_close(layer.get_reference_pose(_REFERENCE_ID), expected)
    layer.bind_pose_world = layer.bind_pose_world.double()
    assert layer.get_reference_pose(_REFERENCE_ID).dtype == torch.float64
    with pytest.raises(KeyError, match="Unknown reference pose"):
        layer.get_reference_pose("not-published")
    layer.public_joint_parent_ids[3] = 0
    with pytest.raises(ValueError, match="hierarchy"):
        layer.get_reference_pose(_REFERENCE_ID)


def test_old_npz_without_catalog_remains_usable(layer):
    layer._reference_pose_history = ReferencePoseHistory({"custom__vertices": np.zeros((1, 3))})
    assert layer.list_reference_poses() == []
    with pytest.raises(KeyError, match="current SOMA_neutral.npz"):
        layer.get_reference_pose(_REFERENCE_ID)
    angles, reference = _inputs(layer)
    out = layer.pose(angles, reference_pose=reference, apply_correctives=False)
    assert torch.isfinite(out["vertices"]).all()


@pytest.mark.parametrize("problem", ["partial", "schema", "default", "duplicate", "cycle", "names"])
def test_malformed_catalog_fails_clearly(problem):
    data = _catalog_data(_tiny_layer())
    if problem == "partial":
        del data[_REFERENCE_ID]
    elif problem in ("schema", "default", "duplicate"):
        metadata = json.loads(data["reference_pose_history_metadata"].item())
        if problem == "schema":
            metadata["schema_version"] = 99
        elif problem == "default":
            metadata["default_reference_id"] = "missing"
        else:
            metadata["references"] *= 2
        data["reference_pose_history_metadata"] = np.array(json.dumps(metadata))
    elif problem == "cycle":
        data[_REFERENCE_ID + "__parent_ids"][2:4] = [3, 2]
    elif problem == "names":
        data[_REFERENCE_ID + "__joint_names"][2] = "Hips"
    with pytest.raises(ValueError, match="[Rr]eference pose history"):
        ReferencePoseHistory(data)


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_current_asset_bundle_can_replay_historical_reference_offline(monkeypatch, tmp_path):
    """The shipped current assets alone contain the historical reference payloads."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    current = tmp_path / "current-assets"
    current.mkdir()
    for filename in (
        "SOMA_neutral.npz",
        "SOMA_template_rig.usda",
        "SOMA_procedural_transforms.json",
    ):
        shutil.copy2(assets / filename, current / filename)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf-cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fail_network(*args, **kwargs):
        raise AssertionError("Historical references must be available without network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    layer = SOMALayer(
        data_root=current,
        lod="xlo",
        device="cpu",
        mode="dense",
        identity_model_type="soma",
        correctives_model_path=None,
    )
    records = layer.list_reference_poses()
    release_ids = {f"t_pose_world__v{version}" for version in ("0.1.0", "0.2.0")}
    assert {record["id"] for record in records if "version" in record} == release_ids
    assert len(records) == 3
    historical = layer.get_reference_pose(version="0.1.0", data_key="t_pose_world")
    torch.testing.assert_close(historical, layer.get_reference_pose("t_pose_world__v0.1.0"))
    latest = layer.get_reference_pose(version="v0.3.0", data_key="t_pose_world")
    torch.testing.assert_close(
        latest, layer.get_reference_pose("t_pose_world__v0.2.0"), rtol=0, atol=0
    )
    for version in ("0.2.0", "0.2.1"):
        torch.testing.assert_close(layer.get_reference_pose(version=version), latest)
    early_record = next(record for record in records if "asset_revision" in record)
    torch.testing.assert_close(
        layer.get_reference_pose(asset_revision=early_record["asset_revision"]),
        layer.get_reference_pose(early_record["id"]),
    )
    assert (historical - latest).abs().max() > 0.01
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    layer.prepare_identity(identity)
    torch.testing.assert_close(
        latest, layer.public_rig_view().t_pose_world[..., :3, :3], atol=1e-4, rtol=0
    )
    rotations = batch_rodrigues(torch.linspace(-0.3, 0.4, 77 * 3).reshape(77, 3))[None]
    absolute, _ = _world_fk_oracle(rotations, latest, layer.output_joint_parent_ids.tolist())
    expected = layer.pose(absolute, pose2rot=False, absolute_pose=True, apply_correctives=False)
    selector = {"version": "v0.3.0", "data_key": "t_pose_world"}
    actual = layer.pose(rotations, pose2rot=False, reference_pose=selector, apply_correctives=False)
    _assert_output_close(actual, expected)
    _assert_output_close(
        layer(
            rotations, identity, pose2rot=False, reference_pose=selector, apply_correctives=False
        ),
        expected,
    )
    pinned_hashes = {
        early_record["id"]: "7ed68850ae80998d89f56cecd49c0f6b9a0e7580d86a3eea262f0cfff0dd9b72",
        "t_pose_world__v0.1.0": "0ec6244b4eb45630a9c3e60598035b47177bbdcbd3f1489d3d677c09578bcfa6",
        "t_pose_world__v0.2.0": "dd5284ab7c781e1d690bd980d8547b197852dc4f6ca8a49fdb8b4261c3f394fc",
    }
    with np.load(current / "SOMA_neutral.npz", allow_pickle=False) as data:
        metadata = json.loads(data["reference_pose_history_metadata"].item())
        assert metadata["schema_version"] == 2
        assert metadata["default_reference_id"] == "t_pose_world__v0.2.0"
        assert "t_pose_world__v0.2.1" not in data
        assert "t_pose_world__v0.3.0" not in data
        assert "lookups" not in metadata
        assert "reference_pose_history_world" not in data
        assert not any("soma-tpose" in key for key in data.files)
        np.testing.assert_array_equal(data["t_pose_world__v0.1.0"], historical.numpy())
        for reference_id, expected_hash in pinned_hashes.items():
            payload = data[reference_id].astype("<f4").tobytes()
            assert hashlib.sha256(payload).hexdigest() == expected_hash
            hierarchy = {
                "joint_names": data[reference_id + "__joint_names"].tolist(),
                "parent_ids": data[reference_id + "__parent_ids"].tolist(),
            }
            hierarchy_payload = json.dumps(
                hierarchy, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii")
            assert hashlib.sha256(hierarchy_payload).hexdigest() == (
                "d3b43163e91cb27c5456d0f10df4fdf964d11f1efdd7b88153c68fc62d689b64"
            )


def test_float32_reference_validation_with_tf32_enabled(layer):
    """Saved float32 references must validate even when CUDA GEMMs use TF32."""
    # Exercise the production precision setting; the context restores it afterwards.
    precision = "tf32" if layer.device.type == "cuda" else "none"
    with torch.backends.flags(fp32_precision=precision):
        angles, reference = _inputs(layer)
        assert reference.dtype == torch.float32
        torch.testing.assert_close(
            validate_reference_pose(reference, len(layer.public_joint_names)),
            reference,
            rtol=0,
            atol=0,
        )
        out = layer.pose(angles, reference_pose=reference, apply_correctives=False)
        assert all(torch.isfinite(value).all() for value in out.values())
        malformed = reference.clone()
        malformed[2] *= 1.001
        with pytest.raises(ValueError, match="orthonormal"):
            validate_reference_pose(malformed, len(layer.public_joint_names))


def test_reference_cast_preserves_gradient(layer):
    angles, reference = _inputs(layer)
    # A saved CPU float64 reference is normalized to the layer's pose dtype/device.
    reference = reference.cpu().double().requires_grad_()
    out = layer.pose(angles, reference_pose=reference, apply_correctives=False)
    (gradient,) = torch.autograd.grad(out["vertices"].sum(), (reference,))
    assert gradient.dtype == torch.float64
    assert gradient.device.type == "cpu"
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 1e-4


def _lookup_catalog(layer):
    data = _catalog_data(layer)
    metadata = json.loads(data["reference_pose_history_metadata"].item())
    for selector, value, reference_id in (
        ("soma_version", "0.1.1", "t_pose_world__v0.1.1"),
        ("asset_revision", _ASSET_REVISION, _ASSET_ID),
    ):
        metadata["references"].append(
            {
                "id": reference_id,
                "data_key": "t_pose_world",
                selector: value,
            }
        )
        for suffix in ("", "__joint_names", "__parent_ids"):
            data[reference_id + suffix] = data[_REFERENCE_ID + suffix].copy()
    data["reference_pose_history_metadata"] = np.array(json.dumps(metadata))
    return data


def _versioned_catalog(layer, versions):
    data = _catalog_data(layer)
    base = {
        suffix: data.pop(_REFERENCE_ID + suffix) for suffix in ("", "__joint_names", "__parent_ids")
    }
    metadata = json.loads(data["reference_pose_history_metadata"].item())
    metadata["references"] = []
    for index, version in enumerate(versions):
        key = f"t_pose_world__v{version}"
        metadata["references"].append(
            {"id": key, "soma_version": version, "data_key": "t_pose_world"}
        )
        for suffix, array in base.items():
            data[key + suffix] = array.copy()
        # Give every revision a distinct valid payload, so incorrect selection is visible.
        angle = 0.1 * (index + 1)
        cosine, sine = np.cos(angle), np.sin(angle)
        data[key][1] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    metadata["default_reference_id"] = metadata["references"][0]["id"]
    data["reference_pose_history_metadata"] = np.array(json.dumps(metadata))
    return data


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("0.1.0", "0.1.0"),
        ("v0.3.0", "0.2.0"),
        ("0.9.9", "0.9.0"),
        ("0.10.0", "0.10.0"),
        ("0.11.0", "0.10.0"),
        ("1.0.0", "1.0.0"),
        ("9.9.9", "1.0.0"),
    ],
)
def test_version_floor_uses_highest_numeric_revision_not_catalog_order(requested, expected):
    layer = _tiny_layer()
    data = _versioned_catalog(layer, ["0.10.0", "0.1.0", "1.0.0", "0.9.0", "0.2.0"])
    layer._reference_pose_history = ReferencePoseHistory(data)
    torch.testing.assert_close(
        layer.get_reference_pose(version=requested),
        layer.get_reference_pose(f"t_pose_world__v{expected}"),
        rtol=0,
        atol=0,
    )
    assert len(layer.list_reference_poses()) == 5


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("1.0.0-alpha.2", "1.0.0-alpha.1"),
        ("1.0.0-alpha.10", "1.0.0-alpha.10"),
        ("1.0.0-alpha.11", "1.0.0-alpha.10"),
        ("1.0.0-alpha.beta", "1.0.0-alpha.beta"),
        ("1.0.0-beta", "1.0.0-alpha.beta"),
        ("1.0.0-rc.1", "1.0.0-rc.1"),
        ("1.0.0-rc.2+build.7", "1.0.0-rc.1"),
        ("1.0.0", "1.0.0+build.1"),
        ("v1.0.0+build.2", "1.0.0+build.1"),
    ],
)
def test_version_floor_obeys_prerelease_and_build_precedence(requested, expected):
    layer = _tiny_layer()
    versions = [
        "1.0.0+build.1",
        "1.0.0-alpha.10",
        "1.0.0-rc.1",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
    ]
    layer._reference_pose_history = ReferencePoseHistory(_versioned_catalog(layer, versions))
    torch.testing.assert_close(
        layer.get_reference_pose(version=requested),
        layer.get_reference_pose(f"t_pose_world__v{expected}"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("versions", [["1.0.0", "1.0.0+build.1"], ["1.0.0+one", "1.0.0+two"]])
def test_catalog_rejects_ambiguous_equal_semantic_precedence(versions):
    with pytest.raises(ValueError):
        ReferencePoseHistory(_versioned_catalog(_tiny_layer(), versions))


def test_version_floor_is_scoped_to_data_key():
    layer = _tiny_layer()
    data = _versioned_catalog(layer, ["0.1.0", "0.2.0"])
    metadata = json.loads(data["reference_pose_history_metadata"].item())
    newer = metadata["references"][1]
    old_key = newer["id"]
    newer["data_key"] = "other_pose_world"
    newer["id"] = "other_pose_world__v0.2.0"
    for suffix in ("", "__joint_names", "__parent_ids"):
        data[newer["id"] + suffix] = data.pop(old_key + suffix)
    data["reference_pose_history_metadata"] = np.array(json.dumps(metadata))
    layer._reference_pose_history = ReferencePoseHistory(data)
    torch.testing.assert_close(
        layer.get_reference_pose(version="0.3.0", data_key="t_pose_world"),
        layer.get_reference_pose(_REFERENCE_ID),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layer.get_reference_pose(version="0.3.0", data_key="other_pose_world"),
        layer.get_reference_pose(newer["id"]),
        rtol=0,
        atol=0,
    )


def test_asset_only_catalog_is_not_a_version_fallback():
    layer = _tiny_layer()
    data = _lookup_catalog(layer)
    metadata = json.loads(data["reference_pose_history_metadata"].item())
    for record in metadata["references"]:
        if "soma_version" in record:
            for suffix in ("", "__joint_names", "__parent_ids"):
                del data[record["id"] + suffix]
    metadata["references"] = [
        record for record in metadata["references"] if "asset_revision" in record
    ]
    metadata["default_reference_id"] = _ASSET_ID
    data["reference_pose_history_metadata"] = np.array(json.dumps(metadata))
    layer._reference_pose_history = ReferencePoseHistory(data)
    with pytest.raises(KeyError):
        layer.get_reference_pose(version="9.9.9")
    assert layer.get_reference_pose(asset_revision=_ASSET_REVISION).shape[-2:] == (3, 3)


def test_exact_release_lookup_and_discovery(layer):
    layer._reference_pose_history = ReferencePoseHistory(_lookup_catalog(layer))
    expected = layer.get_reference_pose(_REFERENCE_ID)
    for version in ("0.1.0", "0.1.1"):
        actual = layer.get_reference_pose(version=version, data_key="t_pose_world")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.zero_()
    torch.testing.assert_close(
        layer.get_reference_pose(asset_revision=_ASSET_REVISION), expected, rtol=0, atol=0
    )
    records = layer.list_reference_poses()
    assert len(records) == 3
    assert records[0]["version"] == "0.1.0"
    assert all("soma_version" not in record for record in records)
    assert "lookups" not in records[0]
    records[0]["version"] = "changed"
    assert layer.list_reference_poses()[0]["version"] == "0.1.0"
    layer.bind_pose_world = layer.bind_pose_world.double()
    assert layer.get_reference_pose(version="0.1.0").dtype == torch.float64
    layer.public_joint_parent_ids[3] = 0
    with pytest.raises(ValueError, match="hierarchy"):
        layer.get_reference_pose(version="0.1.0")


@pytest.mark.parametrize(
    "selector",
    [
        {"version": "0.0.9"},
        {"reference_id": "t_pose_world__v0.1.2"},
        {"version": "0.1.0", "data_key": "bind_pose_world"},
        {"asset_revision": "sha256:" + "b" * 64},
    ],
)
def test_unresolvable_reference_selectors_fail(selector):
    layer = _tiny_layer()
    layer._reference_pose_history = ReferencePoseHistory(_lookup_catalog(layer))
    with pytest.raises(KeyError):
        layer.get_reference_pose(**selector)


@pytest.mark.parametrize("entry_point", ["getter", "pose", "forward"])
def test_obsolete_version_keyword_is_rejected(layer, entry_point):
    layer._reference_pose_history = ReferencePoseHistory(_lookup_catalog(layer))
    selector = {"soma_version": "0.1.0"}
    angles, _ = _inputs(layer)
    with pytest.raises(TypeError, match="soma_version"):
        if entry_point == "getter":
            layer.get_reference_pose(**selector)
        elif entry_point == "pose":
            layer.pose(angles, reference_pose=selector)
        else:
            layer(angles, torch.zeros(1, 1, device=layer.device), reference_pose=selector)


@pytest.mark.parametrize(
    "selector",
    [
        {},
        {"reference_id": _REFERENCE_ID, "version": "0.1.0"},
        {"asset_revision": _ASSET_REVISION, "version": "0.1.0"},
        {"version": ""},
        {"version": "0.1"},
        {"version": "vv0.1.0"},
        {"version": "0.1.0-01"},
        {"version": "0.1.x"},
        {"asset_revision": "missing"},
        {"version": 1},
        {"version": "0.1.0", "data_key": ""},
        {"reference_id": _REFERENCE_ID, "data_key": "bind_pose_world"},
    ],
)
def test_ambiguous_or_invalid_lookup_selectors_fail(selector):
    layer = _tiny_layer()
    layer._reference_pose_history = ReferencePoseHistory(_lookup_catalog(layer))
    with pytest.raises(ValueError):
        layer.get_reference_pose(**selector)


@pytest.mark.parametrize(
    "problem",
    [
        "not_list",
        "not_dict",
        "duplicate",
        "mismatched_id",
        "both_selectors",
        "missing_key",
        "empty",
    ],
)
def test_invalid_named_reference_is_rejected(problem):
    data = _lookup_catalog(_tiny_layer())
    metadata = json.loads(data["reference_pose_history_metadata"].item())
    records = metadata["references"]
    if problem == "not_list":
        metadata["references"] = {}
    elif problem == "not_dict":
        records[0] = "invalid"
    elif problem == "duplicate":
        records.append(dict(records[0]))
    elif problem == "mismatched_id":
        records[0]["id"] = "t_pose_world__v9.9.9"
    elif problem == "both_selectors":
        records[0]["asset_revision"] = _ASSET_REVISION
    elif problem == "missing_key":
        del records[0]["data_key"]
    else:
        records[0]["soma_version"] = ""
    data["reference_pose_history_metadata"] = np.array(json.dumps(metadata))
    with pytest.raises(ValueError):
        ReferencePoseHistory(data)


@pytest.mark.parametrize("use_forward", [False, True])
@pytest.mark.parametrize(
    "selector",
    [
        {"version": "0.1.0", "data_key": "t_pose_world"},
        {"version": "v0.3.0", "data_key": "t_pose_world"},
        {"reference_id": _REFERENCE_ID},
        {"asset_revision": _ASSET_REVISION},
    ],
)
def test_dictionary_reference_matches_tensor_and_resolves_once(
    layer, monkeypatch, use_forward, selector
):
    layer._reference_pose_history = ReferencePoseHistory(_lookup_catalog(layer))
    angles, _ = _inputs(layer)
    options = {"apply_correctives": layer.procedural_transforms is not None}
    reference = layer.get_reference_pose(**selector)
    expected = layer.pose(angles, reference_pose=reference, **options)
    original = dict(selector)
    resolve_calls = []
    getter = layer.get_reference_pose

    def get_reference(**kwargs):
        resolve_calls.append(dict(kwargs))
        return getter(**kwargs)

    monkeypatch.setattr(layer, "get_reference_pose", get_reference)
    if use_forward:
        actual = layer(
            angles, torch.zeros(1, 1, device=layer.device), reference_pose=selector, **options
        )
    else:
        actual = layer.pose(angles, reference_pose=selector, **options)
    _assert_output_close(actual, expected)
    assert resolve_calls == [original]
    assert selector == original


@pytest.mark.parametrize(
    "selector,error",
    [
        ({}, ValueError),
        ({"version": "0.1.0", "unknown_key": "value"}, TypeError),
        ({"version": "0.1.0", "asset_revision": _ASSET_REVISION}, ValueError),
        ({"version": "0.0.9"}, KeyError),
        ({"version": "0.1.0", "data_key": "missing"}, KeyError),
    ],
)
def test_invalid_dictionary_fails_before_identity_update(layer, selector, error):
    layer._reference_pose_history = ReferencePoseHistory(_lookup_catalog(layer))
    angles, _ = _inputs(layer)
    cached_shape = layer._cached_rest_shape
    cached_bind = layer._cached_bind_transforms_world
    original = dict(selector)
    with pytest.raises(error):
        layer(
            angles,
            torch.zeros(3, 1, device=layer.device),
            reference_pose=selector,
            apply_correctives=False,
        )
    assert layer._cached_rest_shape is cached_shape
    assert layer._cached_bind_transforms_world is cached_bind
    assert selector == original


@pytest.mark.parametrize("use_forward", [False, True])
def test_dictionary_reference_conflicts_with_absolute_pose(layer, monkeypatch, use_forward):
    angles, _ = _inputs(layer)
    cached_shape = layer._cached_rest_shape
    cached_bind = layer._cached_bind_transforms_world

    def unexpected_lookup(**kwargs):
        raise AssertionError("Conflicting flags must fail before resolving a reference")

    monkeypatch.setattr(layer, "get_reference_pose", unexpected_lookup)
    options = {
        "reference_pose": {"version": "0.1.0"},
        "absolute_pose": True,
        "apply_correctives": False,
    }
    with pytest.raises(ValueError, match="absolute_pose"):
        if use_forward:
            layer(angles, torch.zeros(3, 1, device=layer.device), **options)
        else:
            layer.pose(angles, **options)
    assert layer._cached_rest_shape is cached_shape
    assert layer._cached_bind_transforms_world is cached_bind
