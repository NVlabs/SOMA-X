# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix

from soma.geometry.lbs import batch_rodrigues
from soma.geometry.rig_utils import joint_world_to_local
from soma.geometry.transforms import project_rotations_to_so3
from soma.io import load_lod_rig_from_usd, load_lod_rigs_from_usd
from soma.procedural_transforms import (
    SOMA_ALIGNED_X_SWING_TWIST_MODE,
    SOMA_LOCAL_X_EULER_TWIST_MODE,
    SOMA_LOCAL_X_SWING_TWIST_MODE,
    SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME,
    SOMA_PROCEDURAL_TRANSFORM_MODES,
    SOMAProceduralParameterTransform,
    derive_soma_rig_without_procedural_joints,
    has_soma_twist_joints,
    load_soma_procedural_transform_definition,
    local_x_euler_from_matrix,
    parse_soma_procedural_transform_definition,
)
from tests._optional_assets import body_identity_skip_reason

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
CORE_ASSET = ASSETS_DIR / "SOMA_neutral.npz"
TEMPLATE_RIG = ASSETS_DIR / "SOMA_template_rig.usda"
PROCEDURAL_DEFINITION = ASSETS_DIR / SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME
PROCEDURAL_TRANSFORM_DEFINITION = load_soma_procedural_transform_definition(PROCEDURAL_DEFINITION)
SOMA_TWIST_SEGMENTS = PROCEDURAL_TRANSFORM_DEFINITION.segments
V0025_TWIST_FRACTIONS = (0.05, 1.0 / 3.0, 2.0 / 3.0, 0.95)
MAX_BIND_ROTATION_ERROR_DEGREES = 0.01


def _rotation_errors_degrees(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    # FK round trips accumulate small non-orthogonal float32 drift. Compare the
    # nearest rotations on CPU float64 so this measures orientation rather than
    # scale/shear without introducing a platform-specific CUDA SVD.
    actual_so3 = project_rotations_to_so3(
        actual.detach().to(device="cpu", dtype=torch.float64)
    )
    expected_so3 = project_rotations_to_so3(
        expected.detach().to(device="cpu", dtype=torch.float64)
    )
    relative = actual_so3 @ expected_so3.transpose(-1, -2)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
    sine = 0.5 * torch.linalg.vector_norm(
        torch.stack(
            (
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ),
            dim=-1,
        ),
        dim=-1,
    )
    return torch.rad2deg(torch.atan2(sine, cosine))


def _definition_data() -> dict:
    return json.loads(PROCEDURAL_DEFINITION.read_text(encoding="utf-8"))


def _set_matrix_entry(
    data: dict,
    matrix_name: str,
    row: str,
    column: str,
    value: float,
) -> None:
    for entry in data["parameter_matrices"][matrix_name]["entries"]:
        if entry["row"] == row and entry["column"] == column:
            entry["value"] = value
            return
    raise AssertionError(f"Missing {matrix_name} entry for {row}, {column}")


def _minimal_source_names() -> list[str]:
    return [
        "Root",
        "Hips",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftLeg",
        "LeftShin",
        "LeftFoot",
        "RightLeg",
        "RightShin",
        "RightFoot",
    ]


def _twist_joint_names() -> list[str]:
    return [twist_joint for segment in SOMA_TWIST_SEGMENTS for twist_joint in segment.twist_joints]


def _public_joint_names() -> np.ndarray:
    return np.array(PROCEDURAL_TRANSFORM_DEFINITION.public_joint_names)


def _public_mid_rig() -> dict:
    return derive_soma_rig_without_procedural_joints(
        load_lod_rig_from_usd(TEMPLATE_RIG, "mid"),
        _public_joint_names(),
        segments=SOMA_TWIST_SEGMENTS,
    )


def _rotation_modes(mode: str) -> tuple[str, ...]:
    return (mode,) * len(_twist_joint_names())


def _twist_parent_by_name() -> dict[str, str]:
    return {
        twist_joint: segment.start_joint
        for segment in SOMA_TWIST_SEGMENTS
        for twist_joint in segment.twist_joints
    }


def _target_names(source_names: list[str]) -> list[str]:
    return source_names + _twist_joint_names()


def _target_t_pose_world(source_names: list[str], target_names: list[str]) -> torch.Tensor:
    positions = {
        "Root": (0.0, 0.0, 0.0),
        "Hips": (0.0, 1.0, 0.0),
        "LeftArm": (0.0, 0.0, 0.0),
        "LeftForeArm": (3.0, 0.0, 0.0),
        "LeftHand": (6.0, 0.0, 0.0),
        "RightArm": (0.0, 1.0, 0.0),
        "RightForeArm": (-3.0, 1.0, 0.0),
        "RightHand": (-6.0, 1.0, 0.0),
        "LeftLeg": (0.0, 2.0, 0.0),
        "LeftShin": (0.0, -1.0, 0.0),
        "LeftFoot": (0.0, -4.0, 0.0),
        "RightLeg": (1.0, 2.0, 0.0),
        "RightShin": (1.0, -1.0, 0.0),
        "RightFoot": (1.0, -4.0, 0.0),
    }
    target_by_name = {name: idx for idx, name in enumerate(target_names)}
    transforms = torch.eye(4).reshape(1, 4, 4).repeat(len(target_names), 1, 1)
    for name in source_names:
        transforms[target_by_name[name], :3, 3] = torch.tensor(positions[name])
    for segment in SOMA_TWIST_SEGMENTS:
        start = torch.tensor(positions[segment.start_joint])
        end = torch.tensor(positions[segment.end_joint])
        for twist_name, fraction in zip(
            segment.twist_joints,
            V0025_TWIST_FRACTIONS,
            strict=True,
        ):
            transforms[target_by_name[twist_name], :3, 3] = start + fraction * (end - start)
    return transforms


def _make_parameter_transform(
    source_names: list[str],
    target_names: list[str],
    *,
    rotation_extraction_modes: tuple[str, ...] | list[str] | None = None,
    target_t_pose_world: torch.Tensor | None = None,
    target_joint_parent_ids: torch.Tensor | np.ndarray | None = None,
    target_bind_pose_world: torch.Tensor | None = None,
) -> SOMAProceduralParameterTransform:
    return SOMAProceduralParameterTransform(
        source_names,
        target_names,
        rotation_extraction_modes=(
            PROCEDURAL_TRANSFORM_DEFINITION.rotation_extraction_modes
            if rotation_extraction_modes is None
            else rotation_extraction_modes
        ),
        segments=SOMA_TWIST_SEGMENTS,
        rotation_entries=PROCEDURAL_TRANSFORM_DEFINITION.rotation_entries,
        translation_entries=PROCEDURAL_TRANSFORM_DEFINITION.translation_entries,
        target_t_pose_world=target_t_pose_world,
        target_joint_parent_ids=target_joint_parent_ids,
        target_bind_pose_world=target_bind_pose_world,
    )


def _identity_rotations(batch_size: int, num_joints: int) -> torch.Tensor:
    return torch.eye(3).reshape(1, 1, 3, 3).expand(batch_size, num_joints, 3, 3).contiguous()


def _rx(angle: float) -> torch.Tensor:
    rotvec = torch.tensor([[angle, 0.0, 0.0]], dtype=torch.float32)
    return batch_rodrigues(rotvec)[0]


def _ry(angle: float) -> torch.Tensor:
    rotvec = torch.tensor([[0.0, angle, 0.0]], dtype=torch.float32)
    return batch_rodrigues(rotvec)[0]


def _rz(angle: float) -> torch.Tensor:
    rotvec = torch.tensor([[0.0, 0.0, angle]], dtype=torch.float32)
    return batch_rodrigues(rotvec)[0]


def _local_y_euler_from_matrix(rotations: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        -rotations[..., 2, 0],
        torch.sqrt(rotations[..., 2, 1].square() + rotations[..., 2, 2].square()),
    )


def _link_core_assets(tmp_path: Path) -> Path:
    data_root = tmp_path / "assets"
    data_root.mkdir()
    link_path = data_root / CORE_ASSET.name
    try:
        link_path.symlink_to(CORE_ASSET)
    except OSError:
        shutil.copy2(CORE_ASSET, link_path)
    return data_root


def _synthetic_twist_rig(base_rig: dict) -> dict:
    rig = dict(base_rig)
    base_names = [str(name) for name in base_rig["joint_names"]]
    source_by_name = {name: idx for idx, name in enumerate(base_names)}
    extra_names = _twist_joint_names()
    twist_parent_by_name = _twist_parent_by_name()
    joint_names = np.array(base_names + extra_names)

    parent_ids = list(np.asarray(base_rig["joint_parent_ids"], dtype=np.int32))
    for twist_name in extra_names:
        parent_ids.append(source_by_name[twist_parent_by_name[twist_name]])
    parent_ids = np.asarray(parent_ids, dtype=np.int32)

    bind_world = np.asarray(base_rig["bind_pose_world"], dtype=np.float32)
    t_pose_world = np.asarray(base_rig["t_pose_world"], dtype=np.float32)
    bind_extra = []
    t_pose_extra = []
    for segment in SOMA_TWIST_SEGMENTS:
        start_idx = source_by_name[segment.start_joint]
        end_idx = source_by_name[segment.end_joint]
        for _twist_name, fraction in zip(
            segment.twist_joints,
            V0025_TWIST_FRACTIONS,
            strict=True,
        ):
            bind = bind_world[start_idx].copy()
            bind[:3, 3] = bind_world[start_idx, :3, 3] + fraction * (
                bind_world[end_idx, :3, 3] - bind_world[start_idx, :3, 3]
            )
            t_pose = t_pose_world[start_idx].copy()
            t_pose[:3, 3] = t_pose_world[start_idx, :3, 3] + fraction * (
                t_pose_world[end_idx, :3, 3] - t_pose_world[start_idx, :3, 3]
            )
            bind_extra.append(bind)
            t_pose_extra.append(t_pose)
    bind_world = np.concatenate([bind_world, np.stack(bind_extra)], axis=0)
    t_pose_world = np.concatenate([t_pose_world, np.stack(t_pose_extra)], axis=0)
    bind_local = joint_world_to_local(torch.from_numpy(bind_world), parent_ids).numpy()
    t_pose_local = joint_world_to_local(torch.from_numpy(t_pose_world), parent_ids).numpy()

    W = np.asarray(
        csc_matrix(
            (
                base_rig["skinning_weights_data"],
                base_rig["skinning_weights_indices"],
                base_rig["skinning_weights_indptr"],
            ),
            shape=base_rig["skinning_weights_shape"],
        ).todense(),
        dtype=np.float32,
    )
    extra_weights = np.zeros((W.shape[0], len(extra_names)), dtype=np.float32)
    twists_by_source = defaultdict(list)
    for extra_idx, twist_name in enumerate(extra_names):
        twists_by_source[twist_parent_by_name[twist_name]].append(extra_idx)
    for source_name, extra_ids in twists_by_source.items():
        source_idx = source_by_name[source_name]
        source_weights = W[:, source_idx].copy()
        W[:, source_idx] = source_weights * 0.5
        for extra_idx in extra_ids:
            extra_weights[:, extra_idx] = source_weights * (0.5 / len(extra_ids))
    W = np.concatenate([W, extra_weights], axis=1)
    W_sparse = csc_matrix(W)

    rig.update(
        joint_names=joint_names,
        joint_parent_ids=parent_ids,
        bind_pose_world=bind_world.astype(np.float32),
        bind_pose_local=bind_local.astype(np.float32),
        t_pose_world=t_pose_world.astype(np.float32),
        t_pose_local=t_pose_local.astype(np.float32),
        skinning_weights_data=W_sparse.data.astype(np.float32),
        skinning_weights_indices=W_sparse.indices.astype(np.int32),
        skinning_weights_indptr=W_sparse.indptr.astype(np.int32),
        skinning_weights_shape=np.array(W_sparse.shape, dtype=np.int32),
    )
    return rig


def _dense_weights(rig: dict) -> np.ndarray:
    return np.asarray(
        csc_matrix(
            (
                rig["skinning_weights_data"],
                rig["skinning_weights_indices"],
                rig["skinning_weights_indptr"],
            ),
            shape=rig["skinning_weights_shape"],
        ).todense(),
        dtype=np.float32,
    )


def _make_synthetic_twist_layer(
    monkeypatch,
    tmp_path,
    lod: str = "low",
):
    import soma.body.soma as soma_module
    from soma import SOMALayer
    from soma.io import load_lod_rig_from_usd as original_load_lod_rig_from_usd

    original_load_many = soma_module.load_lod_rigs_from_usd
    public_joint_names = _public_joint_names()

    def fake_load_lod_rig_from_usd(path, rig_lod, skin_mesh_name=None):
        if Path(path) == TEMPLATE_RIG:
            universal = original_load_lod_rig_from_usd(
                TEMPLATE_RIG,
                rig_lod,
                skin_mesh_name=skin_mesh_name,
            )
            base = derive_soma_rig_without_procedural_joints(universal, public_joint_names)
            return _synthetic_twist_rig(base)
        return original_load_lod_rig_from_usd(path, rig_lod, skin_mesh_name=skin_mesh_name)

    def fake_load_lod_rigs_from_usd(path, lods, skin_mesh_names=None):
        if Path(path) != TEMPLATE_RIG:
            return original_load_many(path, lods, skin_mesh_names=skin_mesh_names)
        skin_mesh_names = skin_mesh_names or {}
        return {
            rig_lod.lower(): fake_load_lod_rig_from_usd(
                path,
                rig_lod,
                skin_mesh_name=skin_mesh_names.get(
                    rig_lod.lower(),
                    skin_mesh_names.get(rig_lod),
                ),
            )
            for rig_lod in lods
        }

    monkeypatch.setattr(soma_module, "load_lod_rigs_from_usd", fake_load_lod_rigs_from_usd)
    return SOMALayer(
        data_root=ASSETS_DIR,
        lod=lod,
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=True,
    )


def test_derives_public_rig_and_aggregates_procedural_skin_weights():
    base = _public_mid_rig()
    synthetic = _synthetic_twist_rig(base)

    derived = derive_soma_rig_without_procedural_joints(synthetic, base["joint_names"])

    assert len(derived["joint_names"]) == len(base["joint_names"]) == 78
    assert not any("Twist" in str(name) for name in derived["joint_names"])
    np.testing.assert_array_equal(derived["joint_names"], base["joint_names"])
    np.testing.assert_array_equal(derived["joint_parent_ids"], base["joint_parent_ids"])
    np.testing.assert_allclose(_dense_weights(derived), _dense_weights(base), atol=1e-6)


def test_derivation_without_public_joint_names_requires_explicit_segments():
    base = _public_mid_rig()

    with pytest.raises(ValueError, match="requires explicit procedural twist segments"):
        derive_soma_rig_without_procedural_joints(base)


def test_packaged_definition_is_unsuffixed_and_declares_twist_segments():
    data = _definition_data()
    definition = load_soma_procedural_transform_definition(PROCEDURAL_DEFINITION)
    public_rig = data["public_rig_derivation"]
    raw_segments = data["segments"]

    assert PROCEDURAL_DEFINITION.name == "SOMA_procedural_transforms.json"
    assert ".v" not in PROCEDURAL_DEFINITION.name
    assert "main_joint_names" in public_rig
    assert "kept_joint_names" not in public_rig
    assert definition.path == PROCEDURAL_DEFINITION
    assert definition.modes == SOMA_PROCEDURAL_TRANSFORM_MODES
    assert definition.rotation_extraction_modes == (SOMA_ALIGNED_X_SWING_TWIST_MODE,) * len(
        _twist_joint_names()
    )
    assert len(definition.segments) == len(raw_segments)
    assert all(len(segment.twist_joints) == 4 for segment in definition.segments)
    assert definition.segments[0].start_joint == raw_segments[0]["start_joint"]
    assert definition.segments[0].twist_joints == tuple(raw_segments[0]["twist_joints"])
    assert len(definition.public_joint_names) == 78
    assert definition.main_joint_names == definition.public_joint_names


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_packaged_template_rig_lods_preserve_procedural_contract():
    rigs = load_lod_rigs_from_usd(TEMPLATE_RIG, ("mid", "low", "xlo"))

    for lod, vertex_count in (("mid", 18056), ("low", 4505), ("xlo", 612)):
        rig = rigs[lod]
        derived = derive_soma_rig_without_procedural_joints(
            rig,
            _public_joint_names(),
            segments=SOMA_TWIST_SEGMENTS,
        )

        assert len(rig["joint_names"]) == 110
        assert has_soma_twist_joints(rig["joint_names"], segments=SOMA_TWIST_SEGMENTS)
        assert rig["bind_shape"].shape == (vertex_count, 3)
        assert derived["joint_names"].tolist() == _public_joint_names().tolist()
        assert tuple(derived["skinning_weights_shape"]) == (vertex_count, 78)


@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_aligned_twist_is_zero_at_usd_bind_pose_not_rest_pose():
    rig = load_lod_rig_from_usd(TEMPLATE_RIG, "low")
    source_names = list(PROCEDURAL_TRANSFORM_DEFINITION.public_joint_names)
    target_names = [str(name) for name in rig["joint_names"]]
    transform = _make_parameter_transform(
        source_names,
        target_names,
        target_t_pose_world=torch.from_numpy(rig["t_pose_world"]),
        target_joint_parent_ids=rig["joint_parent_ids"],
        target_bind_pose_world=torch.from_numpy(rig["bind_pose_world"]),
    )
    source_target_ids = torch.tensor([target_names.index(name) for name in source_names])
    source_rotations = _identity_rotations(1, len(source_names))
    bind_world = torch.from_numpy(rig["bind_pose_world"])[source_target_ids].unsqueeze(0)
    rest_world = torch.from_numpy(rig["t_pose_world"])[source_target_ids].unsqueeze(0)

    bind_twist = transform._twist_angles_from_source(source_rotations, bind_world)
    rest_twist = transform._twist_angles_from_source(source_rotations, rest_world)

    torch.testing.assert_close(bind_twist, torch.zeros_like(bind_twist), atol=1e-6, rtol=0.0)
    assert torch.rad2deg(rest_twist.abs()).max().item() > 1.0


def test_definition_parser_reports_invalid_axis():
    data = _definition_data()
    data["segments"][0]["source_axis"] = "roll"

    with pytest.raises(ValueError, match="source_axis"):
        parse_soma_procedural_transform_definition(data)


def test_definition_parser_reports_duplicate_outputs():
    data = _definition_data()
    data["segments"][0]["twist_joints"][1] = data["segments"][0]["twist_joints"][0]

    with pytest.raises(ValueError, match="duplicate"):
        parse_soma_procedural_transform_definition(data)


def test_definition_parser_reports_unknown_matrix_channel():
    data = _definition_data()
    data["parameter_matrices"]["rotation"]["entries"][0]["column"] = "MissingJoint"

    with pytest.raises(ValueError, match="unknown rotation matrix column"):
        parse_soma_procedural_transform_definition(data)


def test_parameter_transform_uses_json_sidecar_matrices_over_template_positions():
    data = _definition_data()
    _set_matrix_entry(data, "rotation", "LeftArmTwist2", "LeftArm", -0.75)
    _set_matrix_entry(data, "rotation", "LeftArmTwist2", "LeftForeArm", 0.25)
    _set_matrix_entry(data, "translation", "LeftArmTwist2", "LeftArm", 0.75)
    _set_matrix_entry(data, "translation", "LeftArmTwist2", "LeftForeArm", 0.25)
    definition = parse_soma_procedural_transform_definition(data)
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    target_t_pose_world = _target_t_pose_world(source_names, target_names)
    transform = SOMAProceduralParameterTransform(
        source_names,
        target_names,
        rotation_extraction_modes=definition.rotation_extraction_modes,
        segments=definition.segments,
        rotation_entries=definition.rotation_entries,
        translation_entries=definition.translation_entries,
        target_t_pose_world=target_t_pose_world,
    )
    twist_row = {name: idx for idx, name in enumerate(transform.twist_joint_names)}
    source_col = {name: idx for idx, name in enumerate(source_names)}
    target_col = {name: idx for idx, name in enumerate(target_names)}
    row = twist_row["LeftArmTwist2"]
    target_row = target_col["LeftArmTwist2"]

    assert transform.rotation_parameter_matrix[row, source_col["LeftArm"]].item() == (
        pytest.approx(-0.75)
    )
    assert transform.rotation_parameter_matrix[row, source_col["LeftForeArm"]].item() == (
        pytest.approx(0.25)
    )
    assert transform.segment_fractions[row].item() == pytest.approx(0.25)
    assert transform.translation_parameter_matrix[target_row, target_col["LeftArm"]].item() == (
        pytest.approx(0.75)
    )
    assert transform.translation_parameter_matrix[target_row, target_col["LeftForeArm"]].item() == (
        pytest.approx(0.25)
    )
    assert transform.translation_parameter_matrix[target_row, target_row].item() == 0.0

    transforms = target_t_pose_world.clone().unsqueeze(0)
    transforms[:, target_row, :3, 3] = torch.tensor([99.0, 0.0, 0.0])
    out = transform(target_world_transforms=transforms).transforms

    torch.testing.assert_close(
        out[0, target_row, :3, 3],
        torch.tensor([0.75, 0.0, 0.0]),
    )


def test_definition_parser_requires_rotation_extraction():
    data = _definition_data()
    del data["rotation_extraction"]

    with pytest.raises(ValueError, match="rotation_extraction is required"):
        parse_soma_procedural_transform_definition(data)


def test_definition_parser_supports_per_procedural_joint_rotation_extraction():
    data = _definition_data()
    data["rotation_extraction"] = {
        "default": SOMA_LOCAL_X_EULER_TWIST_MODE,
        "per_procedural_joint": {
            "LeftForeArmTwist2": SOMA_LOCAL_X_SWING_TWIST_MODE,
            "RightForeArmTwist2": SOMA_LOCAL_X_SWING_TWIST_MODE,
        },
    }

    definition = parse_soma_procedural_transform_definition(data)
    twist_modes = dict(zip(_twist_joint_names(), definition.rotation_extraction_modes, strict=True))

    assert twist_modes["LeftForeArmTwist1"] == SOMA_LOCAL_X_EULER_TWIST_MODE
    assert twist_modes["LeftForeArmTwist2"] == SOMA_LOCAL_X_SWING_TWIST_MODE
    assert twist_modes["RightForeArmTwist2"] == SOMA_LOCAL_X_SWING_TWIST_MODE


def _assert_twist_translations_on_segments(layer, transforms: torch.Tensor) -> None:
    target_by_name = {str(name): idx for idx, name in enumerate(layer.rig_data["joint_names"])}
    row = 0
    for segment in SOMA_TWIST_SEGMENTS:
        start = transforms[:, target_by_name[segment.start_joint], :3, 3]
        end = transforms[:, target_by_name[segment.end_joint], :3, 3]
        for twist_name in segment.twist_joints:
            fraction = layer.procedural_transforms.segment_fractions[row]
            expected = start + (end - start) * fraction
            actual = transforms[:, target_by_name[twist_name], :3, 3]
            torch.testing.assert_close(actual, expected)
            row += 1


def test_transform_requires_explicit_segments():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)

    with pytest.raises(ValueError, match="requires explicit procedural twist segments"):
        SOMAProceduralParameterTransform(source_names, target_names)
    with pytest.raises(ValueError, match="requires explicit rotation extraction modes"):
        SOMAProceduralParameterTransform(source_names, target_names, segments=SOMA_TWIST_SEGMENTS)
    with pytest.raises(ValueError, match="requires JSON sidecar matrix entries"):
        SOMAProceduralParameterTransform(
            source_names,
            target_names,
            rotation_extraction_modes=PROCEDURAL_TRANSFORM_DEFINITION.rotation_extraction_modes,
            segments=SOMA_TWIST_SEGMENTS,
        )


def test_local_x_euler_zero_pose_generates_identity_twists():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    rotations = _identity_rotations(2, len(source_names))

    out = transform(source_rotations=rotations).rotations

    assert out.shape == (2, len(target_names), 3, 3)
    twist_ids = [target_names.index(name) for name in _twist_joint_names()]
    expected = _identity_rotations(2, len(twist_ids))
    torch.testing.assert_close(out[:, twist_ids], expected)


def test_local_x_parameter_matrix_matches_soma_fractional_distribution():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    rotations = _identity_rotations(1, len(source_names))
    rotations[0, source_names.index("LeftArm")] = _rx(0.3)
    rotations[0, source_names.index("LeftForeArm")] = _rx(0.6)
    rotations[0, source_names.index("LeftHand")] = _rx(0.9)
    rotations[0, source_names.index("RightArm")] = _rx(0.3)
    rotations[0, source_names.index("RightForeArm")] = _rx(0.6)
    rotations[0, source_names.index("RightHand")] = _rx(0.9)
    rotations[0, source_names.index("LeftLeg")] = _rx(0.2)
    rotations[0, source_names.index("LeftShin")] = _rx(0.4)
    rotations[0, source_names.index("LeftFoot")] = _rx(0.7)
    rotations[0, source_names.index("RightLeg")] = _rx(0.2)
    rotations[0, source_names.index("RightShin")] = _rx(0.4)
    rotations[0, source_names.index("RightFoot")] = _rx(0.7)

    out = transform(source_rotations=rotations).rotations[0]
    source_angles = {
        "LeftArm": 0.3,
        "LeftForeArm": 0.6,
        "LeftHand": 0.9,
        "RightArm": 0.3,
        "RightForeArm": 0.6,
        "RightHand": 0.9,
        "LeftLeg": 0.2,
        "LeftShin": 0.4,
        "LeftFoot": 0.7,
        "RightLeg": 0.2,
        "RightShin": 0.4,
        "RightFoot": 0.7,
    }
    expected = {}
    for segment in SOMA_TWIST_SEGMENTS:
        start_angle = source_angles[segment.start_joint]
        end_angle = source_angles[segment.end_joint]
        for twist_name, fraction in zip(
            segment.twist_joints,
            V0025_TWIST_FRACTIONS,
            strict=True,
        ):
            if segment.reverse:
                expected[twist_name] = start_angle * (fraction - 1.0) + end_angle * fraction
            else:
                expected[twist_name] = end_angle * fraction
    for twist_name, angle in expected.items():
        row = list(transform.twist_joint_names).index(twist_name)
        axis = int(transform.twist_axis_ids[row].item())
        sign = transform.twist_axis_signs[row].item()
        generated_rot = out[target_names.index(twist_name)]
        if axis == 0:
            generated = local_x_euler_from_matrix(generated_rot) * sign
        else:
            generated = _local_y_euler_from_matrix(generated_rot) * sign
        torch.testing.assert_close(generated, torch.tensor(angle))

    matrix = transform.rotation_parameter_matrix
    twist_row = {name: idx for idx, name in enumerate(transform.twist_joint_names)}
    source_col = {name: idx for idx, name in enumerate(source_names)}
    assert matrix[twist_row["LeftForeArmTwist4"], source_col["LeftHand"]].item() == pytest.approx(
        0.95
    )
    assert matrix[twist_row["LeftForeArmTwist4"], source_col["LeftForeArm"]].item() == 0.0
    assert matrix[twist_row["RightForeArmTwist4"], source_col["RightHand"]].item() == pytest.approx(
        0.95
    )
    assert matrix[twist_row["RightForeArmTwist4"], source_col["RightForeArm"]].item() == 0.0
    assert matrix[twist_row["LeftArmTwist1"], source_col["LeftArm"]].item() == pytest.approx(-0.95)
    assert matrix[twist_row["LeftArmTwist1"], source_col["LeftForeArm"]].item() == pytest.approx(
        0.05
    )
    assert matrix[twist_row["LeftArmTwist4"], source_col["LeftArm"]].item() == pytest.approx(-0.05)
    assert matrix[twist_row["LeftArmTwist4"], source_col["LeftForeArm"]].item() == pytest.approx(
        0.95
    )
    assert transform.source_twist_axis_ids[source_col["LeftLeg"]].item() == 0
    assert transform.source_twist_axis_signs[source_col["LeftLeg"]].item() == 1.0
    assert transform.source_twist_axis_ids[source_col["RightLeg"]].item() == 0
    assert transform.source_twist_axis_signs[source_col["RightLeg"]].item() == 1.0


def test_upper_leg_twist_uses_soma_leg_local_x_channel():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    target_by_name = {name: idx for idx, name in enumerate(target_names)}
    source_by_name = {name: idx for idx, name in enumerate(source_names)}
    transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    rotations = _identity_rotations(1, len(source_names))
    rotations[0, source_by_name["LeftLeg"]] = _rx(0.6)
    rotations[0, source_by_name["LeftShin"]] = _rx(0.9)

    out = transform(source_rotations=rotations).rotations[0]

    expected = {
        twist_name: 0.6 * (fraction - 1.0) + 0.9 * fraction
        for twist_name, fraction in zip(
            [
                "LeftLegTwist1",
                "LeftLegTwist2",
                "LeftLegTwist3",
                "LeftLegTwist4",
            ],
            V0025_TWIST_FRACTIONS,
            strict=True,
        )
    }
    for twist_name, angle in expected.items():
        row = list(transform.twist_joint_names).index(twist_name)
        sign = transform.twist_axis_signs[row].item()
        generated = local_x_euler_from_matrix(out[target_by_name[twist_name]]) * sign
        torch.testing.assert_close(generated, torch.tensor(angle))


def test_swing_twist_mode_matches_euler_for_pure_axis_twist():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    euler_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_EULER_TWIST_MODE),
    )
    swing_twist_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    rotations = _identity_rotations(1, len(source_names))
    rotations[0, source_names.index("LeftHand")] = _rx(0.9)
    rotations[0, source_names.index("LeftFoot")] = _ry(0.6)
    rotations[0, source_names.index("RightFoot")] = _ry(0.6)

    euler_out = euler_transform(source_rotations=rotations).rotations
    swing_twist_out = swing_twist_transform(source_rotations=rotations).rotations

    twist_ids = [target_names.index(name) for name in _twist_joint_names()]
    torch.testing.assert_close(swing_twist_out[:, twist_ids], euler_out[:, twist_ids])


def test_swing_twist_mode_removes_swing_from_twist_channel():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    euler_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_EULER_TWIST_MODE),
    )
    swing_twist_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    rotations = _identity_rotations(1, len(source_names))
    twist_angle = 0.7
    rotations[0, source_names.index("LeftHand")] = _rz(0.5) @ _rx(twist_angle)

    euler_out = euler_transform(source_rotations=rotations).rotations[0]
    swing_twist_out = swing_twist_transform(source_rotations=rotations).rotations[0]

    q_w = torch.cos(torch.tensor(0.5 / 2.0)) * torch.cos(torch.tensor(twist_angle / 2.0))
    q_x = torch.cos(torch.tensor(0.5 / 2.0)) * torch.sin(torch.tensor(twist_angle / 2.0))
    expected_source_twist = 4.0 * torch.atan2(q_x, 1.0 + q_w)
    twist2_id = target_names.index("LeftForeArmTwist2")
    euler_generated = local_x_euler_from_matrix(euler_out[twist2_id])
    swing_twist_generated = local_x_euler_from_matrix(swing_twist_out[twist2_id])
    torch.testing.assert_close(
        swing_twist_generated,
        V0025_TWIST_FRACTIONS[1] * expected_source_twist,
    )
    assert abs(euler_generated.item() - swing_twist_generated.item()) > 0.01


def test_per_joint_rotation_extraction_modes_are_vectorized_by_mode():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    mixed_modes = list(_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE))
    twist_row = {name: idx for idx, name in enumerate(_twist_joint_names())}
    mixed_modes[twist_row["LeftForeArmTwist1"]] = SOMA_LOCAL_X_EULER_TWIST_MODE
    mixed_modes[twist_row["LeftForeArmTwist2"]] = SOMA_LOCAL_X_SWING_TWIST_MODE

    euler_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_EULER_TWIST_MODE),
    )
    swing_twist_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    mixed_transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=mixed_modes,
    )
    rotations = _identity_rotations(1, len(source_names))
    rotations[0, source_names.index("LeftHand")] = _rz(0.5) @ _rx(0.7)

    euler_out = euler_transform(source_rotations=rotations).rotations
    swing_twist_out = swing_twist_transform(source_rotations=rotations).rotations
    mixed_out = mixed_transform(source_rotations=rotations).rotations

    left_forearm_twist1 = target_names.index("LeftForeArmTwist1")
    left_forearm_twist2 = target_names.index("LeftForeArmTwist2")
    torch.testing.assert_close(
        mixed_out[:, left_forearm_twist1],
        euler_out[:, left_forearm_twist1],
    )
    torch.testing.assert_close(
        mixed_out[:, left_forearm_twist2],
        swing_twist_out[:, left_forearm_twist2],
    )
    assert mixed_transform.rotation_parameter_matrices_by_mode.shape == (
        2,
        len(_twist_joint_names()),
        len(source_names),
    )


def test_swing_twist_mode_half_angle_stays_finite_for_perpendicular_pi_swing():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    transform = _make_parameter_transform(
        source_names,
        target_names,
        rotation_extraction_modes=_rotation_modes(SOMA_LOCAL_X_SWING_TWIST_MODE),
    )
    rotations = _identity_rotations(1, len(source_names))
    rotations[0, source_names.index("LeftHand")] = _rz(torch.pi)

    out = transform(source_rotations=rotations).rotations[0]

    twist2_id = target_names.index("LeftForeArmTwist2")
    generated = local_x_euler_from_matrix(out[twist2_id])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(generated, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_twist_translation_matrix_preserves_stretched_public_segments():
    source_names = _minimal_source_names()
    target_names = _target_names(source_names)
    transform = _make_parameter_transform(
        source_names,
        target_names,
        target_t_pose_world=_target_t_pose_world(source_names, target_names),
    )
    target_by_name = {name: idx for idx, name in enumerate(target_names)}
    transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, len(target_names), 1, 1)
    transforms[:, [target_by_name[name] for name in _twist_joint_names()], :3, 3] = 99.0

    transforms[0, target_by_name["LeftForeArm"], :3, 3] = torch.tensor([10.0, 0.0, 0.0])
    transforms[0, target_by_name["LeftHand"], :3, 3] = torch.tensor([16.0, 3.0, 0.0])
    transforms[1, target_by_name["RightForeArm"], :3, 3] = torch.tensor([0.0, 0.0, 0.0])
    transforms[1, target_by_name["RightHand"], :3, 3] = torch.tensor([0.0, -9.0, 0.0])

    out = transform(target_world_transforms=transforms).transforms

    torch.testing.assert_close(
        out[0, target_by_name["LeftForeArmTwist1"], :3, 3],
        torch.tensor([10.3, 0.15, 0.0]),
    )
    torch.testing.assert_close(
        out[0, target_by_name["LeftForeArmTwist4"], :3, 3],
        torch.tensor([15.7, 2.85, 0.0]),
    )
    torch.testing.assert_close(
        out[1, target_by_name["RightForeArmTwist1"], :3, 3],
        torch.tensor([0.0, -0.45, 0.0]),
    )
    torch.testing.assert_close(
        out[1, target_by_name["RightForeArmTwist4"], :3, 3],
        torch.tensor([0.0, -8.55, 0.0]),
    )
    matrix = transform.translation_parameter_matrix
    assert matrix[target_by_name["LeftArm"], target_by_name["LeftArm"]].item() == 1.0
    assert matrix[target_by_name["LeftForeArmTwist4"], target_by_name["LeftForeArm"]].item() == (
        pytest.approx(0.05, abs=1e-6)
    )
    assert matrix[target_by_name["LeftForeArmTwist4"], target_by_name["LeftHand"]].item() == (
        pytest.approx(0.95, abs=1e-6)
    )


def test_default_twist_mode_requires_packaged_definition(tmp_path):
    from soma import SOMALayer

    data_root = _link_core_assets(tmp_path)
    with pytest.raises(FileNotFoundError, match="SOMA procedural transform definition"):
        SOMALayer(
            data_root=data_root,
            device="cpu",
            identity_model_type="soma",
            mode="dense",
        )


def test_twist_mode_reports_malformed_packaged_definition(tmp_path):
    from soma import SOMALayer

    data_root = _link_core_assets(tmp_path)
    (data_root / SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME).write_text(
        "{",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid SOMA procedural transform definition JSON"):
        SOMALayer(
            data_root=data_root,
            device="cpu",
            identity_model_type="soma",
            mode="dense",
        )


def test_twist_mode_rejects_non_twist_rig_asset(monkeypatch, tmp_path):
    import soma.body.soma as soma_module
    from soma import SOMALayer

    no_twist_path = tmp_path / "no_twist.usda"
    no_twist_path.write_text("#usda 1.0\n")
    original_load_many = soma_module.load_lod_rigs_from_usd

    def fake_load_lod_rigs_from_usd(path, lods, skin_mesh_names=None):
        if Path(path) == no_twist_path:
            return {rig_lod.lower(): _public_mid_rig() for rig_lod in lods}
        return original_load_many(path, lods, skin_mesh_names=skin_mesh_names)

    monkeypatch.setattr(soma_module, "load_lod_rigs_from_usd", fake_load_lod_rigs_from_usd)
    with pytest.raises(ValueError, match="require a SOMA template rig with twist joints"):
        SOMALayer(
            data_root=ASSETS_DIR,
            device="cpu",
            identity_model_type="soma",
            mode="dense",
            enable_procedural_transforms=True,
            template_rig_path=no_twist_path,
        )


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_default_layer_uses_procedural_transforms_and_keeps_public_contract():
    from soma import SOMALayer

    layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    out = layer(poses, identity, apply_correctives=False)

    assert layer.procedural_transforms is not None
    assert len(layer.public_joint_names) == 78
    assert len(layer.rig_data["joint_names"]) > len(layer.public_joint_names)
    assert out["joints"].shape == (1, 77, 3)
    assert out["transforms"].shape == (1, 78, 4, 4)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_legacy_layer_opt_out_keeps_78_transform_and_77_joint_contract():
    from soma import SOMALayer

    layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=False,
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    out = layer(poses, identity, apply_correctives=False)

    assert layer.procedural_transforms is None
    assert len(layer.rig_data["joint_names"]) == 78
    torch.testing.assert_close(layer.public_joint_parent_ids, layer.joint_parent_ids)
    assert out["joints"].shape == (1, 77, 3)
    assert out["transforms"].shape == (1, 78, 4, 4)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_public_transforms_match_public_layer_after_identity_fit():
    from soma import SOMALayer

    public_layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=False,
    )
    twist_layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=True,
    )
    identity = torch.zeros(1, public_layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(2, 77, 3)
    poses[0, list(public_layer._public_joint_names).index("LeftForeArm") - 1, 0] = 0.5
    poses[1, list(public_layer._public_joint_names).index("RightLeg") - 1, 1] = -0.4
    transl = torch.tensor([[0.1, -0.2, 0.3], [-0.2, 0.1, 0.0]])

    public_layer.prepare_identity(identity, repose_to_bind_pose=False)
    twist_layer.prepare_identity(identity, repose_to_bind_pose=False)
    public_out = public_layer.pose(
        poses,
        transl=transl,
        apply_correctives=False,
        fk_only=True,
    )
    twist_out = twist_layer.pose(
        poses,
        transl=transl,
        apply_correctives=False,
        fk_only=True,
    )

    torch.testing.assert_close(twist_out["transforms"], public_out["transforms"])
    torch.testing.assert_close(twist_out["joints"], public_out["joints"])


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_reposed_bind_shape_preserves_public_neutral_outputs():
    from soma import SOMALayer

    public_layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=False,
        correctives_model_path=None,
    )
    twist_layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=True,
        correctives_model_path=None,
    )
    identity = torch.zeros(1, public_layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)

    public_layer.prepare_identity(identity, repose_to_bind_pose=True)
    twist_layer.prepare_identity(identity, repose_to_bind_pose=True)
    public_out = public_layer.pose(poses, apply_correctives=False)
    twist_out = twist_layer.pose(poses, apply_correctives=False)

    torch.testing.assert_close(twist_out["transforms"], public_out["transforms"])
    torch.testing.assert_close(twist_out["joints"], public_out["joints"])
    assert twist_out["vertices"].shape == public_out["vertices"].shape
    assert torch.isfinite(twist_out["vertices"]).all()


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_public_rig_view_folds_target_weights():
    from soma import SOMALayer

    layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=True,
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    layer.prepare_identity(identity, repose_to_bind_pose=False)

    view = layer.public_rig_view()

    assert view.joint_names == layer.public_joint_names
    assert view.joint_parent_ids.shape == (len(layer.public_joint_names),)
    assert view.skinning_weights.shape == (
        layer.skinning_weights.shape[0],
        len(layer.public_joint_names),
    )
    torch.testing.assert_close(
        view.bind_transforms_world,
        layer._cached_bind_transforms_world[:, layer.public_transform_joint_indices],
    )
    torch.testing.assert_close(
        view.skinning_weights.sum(dim=1),
        layer.skinning_weights.sum(dim=1),
    )

    target_rotations = torch.eye(3).expand(2, len(layer.target_joint_names), 3, 3).clone()
    public_rotations = layer.to_public_rotations(target_rotations)
    assert public_rotations.shape == (2, len(layer.public_joint_names), 3, 3)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_export_soma_usd_uses_public_rig_view_for_twist_layer(monkeypatch, tmp_path):
    import soma.io as io_module
    from soma import SOMALayer

    layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=True,
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    layer.prepare_identity(identity, repose_to_bind_pose=False)

    captured = {}

    def fake_save_soma_usd(output_path, rotations, root_translation, **kwargs):
        captured["output_path"] = output_path
        captured["rotations"] = rotations
        captured["root_translation"] = root_translation
        captured.update(kwargs)

    monkeypatch.setattr(io_module, "save_soma_usd", fake_save_soma_usd)

    target_rotations = torch.eye(3).expand(1, len(layer.target_joint_names), 3, 3).clone()
    io_module.export_soma_usd(
        tmp_path / "twist_public.usda",
        layer,
        target_rotations,
        torch.zeros(1, 3),
    )

    assert captured["rotations"].shape[-3] == len(layer.public_joint_names)
    assert list(captured["joint_names"]) == list(layer.public_joint_names)
    np.testing.assert_array_equal(
        captured["joint_parent_ids"],
        layer.public_joint_parent_ids.detach().cpu().numpy(),
    )
    assert captured["skinning_weights"].shape == (
        layer.skinning_weights.shape[0],
        len(layer.public_joint_names),
    )


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_loads_packaged_definition(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)

    assert layer.procedural_transform_definition is not None
    assert layer.procedural_transform_definition.path == PROCEDURAL_DEFINITION
    assert layer.procedural_transforms.segments == SOMA_TWIST_SEGMENTS
    assert layer.procedural_transforms.rotation_extraction_modes == (
        layer.procedural_transform_definition.rotation_extraction_modes
    )
    assert layer.procedural_transforms_enabled


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_preserves_public_joints_and_fk_full_consistency(
    monkeypatch,
    tmp_path,
):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, list(layer._public_joint_names).index("LeftHand") - 1, 0] = 0.5
    layer.prepare_identity(identity, repose_to_bind_pose=False)

    full = layer.pose(poses, apply_correctives=False)
    fk_only = layer.pose(poses, apply_correctives=False, fk_only=True)

    assert full["joints"].shape == (1, 77, 3)
    assert full["transforms"].shape == (1, len(layer._public_joint_names), 4, 4)
    assert len(layer.rig_data["joint_names"]) > len(layer._public_joint_names)
    assert layer.public_batched_skinning is not None
    assert layer.public_joint_parent_ids.shape == (len(layer._public_joint_names),)
    assert "vertices" not in fk_only
    torch.testing.assert_close(fk_only["transforms"], full["transforms"])
    torch.testing.assert_close(fk_only["joints"], full["joints"])
    pose_rot = batch_rodrigues(poses.reshape(-1, 3)).reshape(1, 77, 3, 3)
    source_rot = layer.procedural_transforms.apply_source_joint_orient(layer._pad_poses(pose_rot))
    internal_fk = layer.batched_skinning.forward_kinematics(
        local_rotations=layer.procedural_transforms(
            source_rotations=source_rot,
            target_local_rotations=layer.batched_skinning.local_rotations,
        ).rotations,
        global_translation=torch.zeros(1, 3),
        absolute_pose=True,
    )
    torch.testing.assert_close(
        full["transforms"],
        internal_fk[:, layer.public_transform_joint_indices],
    )


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_pose_reuses_skinning_fk_for_aligned_twist(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, list(layer._public_joint_names).index("LeftHand") - 1, 0] = 0.5
    layer.prepare_identity(identity, repose_to_bind_pose=False)

    def fail_internal_source_fk(*_args, **_kwargs):
        raise AssertionError("SOMALayer pose should pass public FK into procedural twist math")

    monkeypatch.setattr(
        layer.procedural_transforms,
        "_source_world_transforms_from_rotations",
        fail_internal_source_fk,
    )

    out = layer.pose(poses, apply_correctives=False)

    assert out["vertices"].shape[0] == 1
    assert out["transforms"].shape == (1, len(layer._public_joint_names), 4, 4)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_pose_uses_explicit_fk_lbs_pipeline(monkeypatch, tmp_path):
    from soma.geometry.batched_skinning import BatchedSkinning

    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, list(layer._public_joint_names).index("LeftHand") - 1, 0] = 0.5
    layer.prepare_identity(identity, repose_to_bind_pose=False)

    def fail_pose(*_args, **_kwargs):
        raise AssertionError("SOMALayer.pose should call FK and LBS explicitly")

    monkeypatch.setattr(BatchedSkinning, "pose", fail_pose)

    out = layer.pose(poses, apply_correctives=False)
    fk_only = layer.pose(poses, apply_correctives=False, fk_only=True)

    assert out["vertices"].shape[0] == 1
    assert out["transforms"].shape == (1, len(layer._public_joint_names), 4, 4)
    assert "vertices" not in fk_only


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_single_fk_path_matches_full_fk_reference(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, list(layer._public_joint_names).index("LeftHand") - 1, 0] = 0.5
    poses[0, list(layer._public_joint_names).index("RightFoot") - 1, 1] = -0.4
    transl = torch.tensor([[0.1, -0.2, 0.3]])
    layer.prepare_identity(identity, repose_to_bind_pose=False)
    layer.batched_skinning.rebind(layer._cached_bind_transforms_world, layer._cached_rest_shape)

    pose_rot = batch_rodrigues(poses.reshape(-1, 3)).reshape(1, 77, 3, 3)
    public_absolute_rotations = layer.procedural_transforms.apply_source_joint_orient(
        layer._pad_poses(pose_rot)
    )

    public_world = layer.batched_skinning.forward_source_kinematics(
        local_rotations=public_absolute_rotations,
        global_translation=transl,
        absolute_pose=True,
    )
    single_fk_world = layer.batched_skinning.expand_source_world_transforms(
        source_rotations=public_absolute_rotations,
        source_world_transforms=public_world,
        transform_expander=layer.procedural_transforms.expand_world_transforms_from_source_fk,
    )
    full_fk_rotations = layer.procedural_transforms(
        source_rotations=public_absolute_rotations,
        target_local_rotations=layer.batched_skinning.local_rotations,
    ).rotations
    full_fk_world = layer.batched_skinning.forward_kinematics(
        local_rotations=full_fk_rotations,
        global_translation=transl,
        absolute_pose=True,
    )
    single_fk_vertices = layer.batched_skinning.linear_blend_skinning(single_fk_world)
    full_fk_vertices = layer.batched_skinning.linear_blend_skinning(full_fk_world)

    torch.testing.assert_close(single_fk_world, full_fk_world, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(single_fk_vertices, full_fk_vertices, atol=1e-5, rtol=1e-5)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.xlo
@pytest.mark.asset_heavy
def test_twist_layer_real_template_expands_all_target_joints():
    from soma import SOMALayer

    layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="xlo",
        device="cpu",
        identity_model_type="soma",
        mode="dense",
        enable_procedural_transforms=True,
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, list(layer._public_joint_names).index("Chest") - 1, 2] = 0.35
    poses[0, list(layer._public_joint_names).index("LeftForeArm") - 1, 0] = 0.6
    transl = torch.tensor([[0.1, -0.2, 0.3]])
    layer.prepare_identity(identity, repose_to_bind_pose=False)
    layer.batched_skinning.rebind(layer._cached_bind_transforms_world, layer._cached_rest_shape)

    pose_rot = batch_rodrigues(poses.reshape(-1, 3)).reshape(1, 77, 3, 3)
    public_absolute_rotations = layer.procedural_transforms.apply_source_joint_orient(
        layer._pad_poses(pose_rot)
    )

    public_world = layer.batched_skinning.forward_source_kinematics(
        local_rotations=public_absolute_rotations,
        global_translation=transl,
        absolute_pose=True,
    )
    expanded_world = layer.batched_skinning.expand_source_world_transforms(
        source_rotations=public_absolute_rotations,
        source_world_transforms=public_world,
        transform_expander=layer.procedural_transforms.expand_world_transforms_from_source_fk,
    )
    full_fk_rotations = layer.procedural_transforms(
        source_rotations=public_absolute_rotations,
        target_local_rotations=layer.batched_skinning.local_rotations,
    ).rotations
    full_fk_world = layer.batched_skinning.forward_kinematics(
        local_rotations=full_fk_rotations,
        global_translation=transl,
        absolute_pose=True,
    )

    target_joint_count = len(layer.rig_data["joint_names"])
    driven = torch.zeros(target_joint_count, dtype=torch.bool)
    driven[layer.public_transform_joint_indices.cpu()] = True
    driven[layer.procedural_transforms.twist_target_ids.cpu()] = True

    assert driven.all()
    torch.testing.assert_close(expanded_world, full_fk_world, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        layer.batched_skinning.linear_blend_skinning(expanded_world),
        layer.batched_skinning.linear_blend_skinning(full_fk_world),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_layer_applies_cached_bind_translation_parameters(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)

    layer.prepare_identity(identity, repose_to_bind_pose=False)
    _assert_twist_translations_on_segments(layer, layer._cached_bind_transforms_world)

    layer.prepare_identity(identity, repose_to_bind_pose=True)
    _assert_twist_translations_on_segments(layer, layer._cached_bind_transforms_world)


@pytest.mark.parametrize("identity_model_type", ["soma", "smpl", "mhr"])
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.asset_heavy
def test_procedural_identity_reposes_to_soma_bind_pose(identity_model_type):
    from soma import SOMALayer

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Warp SOMALayer regression.")
    skip_reason = body_identity_skip_reason(ASSETS_DIR, identity_model_type, lod="low")
    if skip_reason is not None:
        pytest.skip(skip_reason)

    layer = SOMALayer(
        data_root=ASSETS_DIR,
        lod="low",
        device="cuda",
        identity_model_type=identity_model_type,
        mode="warp",
        enable_procedural_transforms=True,
        correctives_model_path=None,
    )
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs, device="cuda")
    scale_params = None
    if identity_model_type == "mhr":
        scale_params = torch.zeros(1, layer.identity_model.num_scale_params, device="cuda")
    poses = (
        torch.eye(3, device="cuda")
        .reshape(1, 1, 3, 3)
        .expand(1, 77, 3, 3)
        .contiguous()
    )
    layer(
        poses,
        identity,
        scale_params=scale_params,
        pose2rot=False,
        apply_correctives=False,
    )

    bind_local_rotations = joint_world_to_local(
        layer._cached_bind_transforms_world,
        layer.joint_parent_ids,
    )[..., :3, :3]
    public_ids = layer.public_transform_joint_indices
    twist_ids = layer.procedural_transforms.twist_target_ids
    for joint_group, joint_ids in (("twist", twist_ids), ("public", public_ids)):
        errors_degrees = _rotation_errors_degrees(
            bind_local_rotations[:, joint_ids],
            layer.bind_pose_local[joint_ids, :3, :3].unsqueeze(0),
        )
        max_error_degrees = errors_degrees.max().item()
        assert max_error_degrees <= MAX_BIND_ROTATION_ERROR_DEGREES, (
            f"{identity_model_type} {joint_group} bind rotation error "
            f"{max_error_degrees:.6f} degrees exceeds "
            f"{MAX_BIND_ROTATION_ERROR_DEGREES:.6f} degrees"
        )

    public_bind_world = layer.public_bind_transforms_world()
    public_bind_local_rotations = joint_world_to_local(
        public_bind_world,
        layer.public_joint_parent_ids,
    )[..., :3, :3]
    generated_twist_rotations = layer.procedural_transforms.twist_rotations_from_source(
        public_bind_local_rotations,
        public_bind_world,
    )
    identity_twist_rotations = torch.eye(
        3,
        dtype=generated_twist_rotations.dtype,
        device=generated_twist_rotations.device,
    ).expand_as(generated_twist_rotations)
    twist_errors_degrees = _rotation_errors_degrees(
        generated_twist_rotations,
        identity_twist_rotations,
    )
    max_twist_error_degrees = twist_errors_degrees.max().item()
    assert max_twist_error_degrees <= MAX_BIND_ROTATION_ERROR_DEGREES, (
        f"{identity_model_type} generated bind-pose twist error "
        f"{max_twist_error_degrees:.6f} degrees exceeds "
        f"{MAX_BIND_ROTATION_ERROR_DEGREES:.6f} degrees"
    )


@pytest.mark.parametrize(
    ("lod", "num_vertices"),
    [
        pytest.param("low", 4505, id="low", marks=(pytest.mark.cpu, pytest.mark.asset_heavy)),
        pytest.param(
            "xlo",
            612,
            id="xlo",
            marks=(pytest.mark.slow, pytest.mark.cpu, pytest.mark.xlo, pytest.mark.asset_heavy),
        ),
    ],
)
def test_twist_layer_lod_plumbing(monkeypatch, tmp_path, lod, num_vertices):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path, lod=lod)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(2, 77, 3)
    poses[1, list(layer._public_joint_names).index("LeftLeg") - 1, 1] = 0.35

    layer.prepare_identity(identity, repose_to_bind_pose=False)
    full = layer.pose(poses, apply_correctives=False)
    fk_only = layer.pose(poses, apply_correctives=False, fk_only=True)

    assert full["vertices"].shape == (2, num_vertices, 3)
    assert full["joints"].shape == (2, 77, 3)
    assert full["transforms"].shape == (2, len(layer._public_joint_names), 4, 4)
    assert len(layer.rig_data["joint_names"]) > len(layer._public_joint_names)
    assert fk_only["transforms"].shape == full["transforms"].shape
    _assert_twist_translations_on_segments(layer, layer._cached_bind_transforms_world)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_rotations_affect_lbs_vertices(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    poses[0, list(layer._public_joint_names).index("LeftHand") - 1, 0] = 0.5
    transl = torch.zeros(1, 3)
    layer.prepare_identity(identity, repose_to_bind_pose=False)

    with_twist = layer.pose(poses, transl=transl, apply_correctives=False)

    control_rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(1, 77, 3, 3)
    no_twist_rotations = layer.procedural_transforms(
        source_rotations=layer._pad_poses(control_rotations)
    ).rotations
    eye = (
        torch.eye(3)
        .reshape(1, 1, 3, 3)
        .expand(
            1,
            len(layer.procedural_transforms.twist_joint_indices),
            3,
            3,
        )
    )
    no_twist_rotations[:, list(layer.procedural_transforms.twist_joint_indices)] = eye
    layer.batched_skinning.rebind(layer._cached_bind_transforms_world, layer._cached_rest_shape)
    without_twist, _ = layer.batched_skinning.pose(
        no_twist_rotations,
        global_translation=transl,
        return_transforms=True,
        absolute_pose=False,
    )

    assert (with_twist["vertices"] - without_twist).abs().max().item() > 1e-6


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_mode_correctives_use_public_joint_rotations(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)

    class FakeCorrectives(torch.nn.Module):
        def __init__(self, num_vertices: int) -> None:
            super().__init__()
            self.num_vertices = num_vertices
            self.seen_shapes = []

        def forward(self, rotations: torch.Tensor) -> dict[str, torch.Tensor]:
            self.seen_shapes.append(tuple(rotations.shape))
            return {
                "out": torch.zeros(
                    rotations.shape[0],
                    self.num_vertices,
                    3,
                    dtype=rotations.dtype,
                    device=rotations.device,
                )
            }

    fake_correctives = FakeCorrectives(layer.bind_shape.shape[0])
    layer.correctives_model = fake_correctives
    layer.prepare_identity(identity, repose_to_bind_pose=True)

    out = layer.pose(poses, apply_correctives=True)
    assert out["transforms"].shape == (1, len(layer._public_joint_names), 4, 4)

    out = layer(poses, identity, apply_correctives=True)
    assert out["transforms"].shape == (1, len(layer._public_joint_names), 4, 4)
    assert fake_correctives.seen_shapes == [
        (1, len(layer._public_joint_names), 3, 3),
        (1, len(layer._public_joint_names), 3, 3),
    ]


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_pose_inversion_uses_public_skeleton_for_twist_layer(monkeypatch, tmp_path):
    from soma.fitting.pose_inversion import PoseInversion

    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)

    inv = PoseInversion(layer, low_lod=False)
    inv.prepare_identity(identity)

    assert inv.soma is layer
    assert inv.soma.procedural_transforms is not None
    assert inv._autograd_soma is layer
    assert len(inv.joint_names) == len(layer._public_joint_names)
    assert inv._cache["parent_ids"].shape[0] == len(layer._public_joint_names)
    assert inv._cache["skinning_weights"].shape[1] == len(layer._public_joint_names)
    assert inv._cache["bone_indices"].max().item() < len(layer._public_joint_names)
    assert layer._cached_bind_transforms_world.shape[1] > len(layer._public_joint_names)


@pytest.mark.slow
@pytest.mark.cpu
@pytest.mark.asset_heavy
def test_twist_mode_device_movement(monkeypatch, tmp_path):
    layer = _make_synthetic_twist_layer(monkeypatch, tmp_path)
    layer.to("cpu")
    assert layer.public_joint_indices.device.type == "cpu"
    assert layer.public_transform_joint_indices.device.type == "cpu"
    assert layer.public_joint_parent_ids.device.type == "cpu"
    assert layer.procedural_transforms.twist_target_ids.device.type == "cpu"
    assert layer.procedural_transforms.rotation_parameter_matrix.device.type == "cpu"
    assert layer.procedural_transforms.segment_fractions.device.type == "cpu"
    assert layer.procedural_transforms.translation_parameter_matrix.device.type == "cpu"
    assert layer.procedural_transforms.source_twist_axis_ids.device.type == "cpu"
    assert layer.procedural_transforms.source_twist_axis_signs.device.type == "cpu"
    assert layer.procedural_transforms.twist_axis_ids.device.type == "cpu"
    assert layer.procedural_transforms.twist_axis_signs.device.type == "cpu"

    identity = torch.zeros(1, layer.identity_model.num_identity_coeffs)
    poses = torch.zeros(1, 77, 3)
    out = layer(poses, identity, apply_correctives=False)
    assert out["transforms"].device.type == "cpu"

    if torch.cuda.is_available():
        layer.to("cuda")
        identity = identity.to("cuda")
        poses = poses.to("cuda")
        out = layer(poses, identity, apply_correctives=False)
        assert out["transforms"].device.type == "cuda"
        assert layer.public_transform_joint_indices.device.type == "cuda"
        assert layer.public_joint_parent_ids.device.type == "cuda"
        assert layer.procedural_transforms.twist_target_ids.device.type == "cuda"
        assert layer.procedural_transforms.rotation_parameter_matrix.device.type == "cuda"
        assert layer.procedural_transforms.segment_fractions.device.type == "cuda"
        assert layer.procedural_transforms.translation_parameter_matrix.device.type == "cuda"
        assert layer.procedural_transforms.source_twist_axis_ids.device.type == "cuda"
        assert layer.procedural_transforms.source_twist_axis_signs.device.type == "cuda"
        assert layer.procedural_transforms.twist_axis_ids.device.type == "cuda"
        assert layer.procedural_transforms.twist_axis_signs.device.type == "cuda"
