# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Procedural parameter transforms for SOMA template rigs."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csc_matrix

from .geometry.lbs import batch_rodrigues
from .geometry.rig_utils import (
    apply_joint_orient_local,
    compute_skeleton_levels,
    joint_local_to_world_levelorder,
    joint_world_to_local,
    precompute_joint_orient,
)
from .geometry.transforms import (
    SE3_from_Rt,
    matrix_to_euler_xyz,
    matrix_to_quaternion_xyzw,
    matrix_to_quaternion_xyzw_stable,
    quaternion_conjugate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_twist_angle_xyzw,
    single_axis_rotation_matrices,
)

SOMA_LOCAL_X_EULER_TWIST_MODE = "local_x_euler"
SOMA_LOCAL_X_SWING_TWIST_MODE = "local_x_swing_twist"
SOMA_ALIGNED_X_SWING_TWIST_MODE = "aligned_x_swing_twist"
SOMA_PROCEDURAL_TRANSFORM_MODES = (
    SOMA_LOCAL_X_EULER_TWIST_MODE,
    SOMA_LOCAL_X_SWING_TWIST_MODE,
    SOMA_ALIGNED_X_SWING_TWIST_MODE,
)
SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME = "SOMA_procedural_transforms.json"
SOMA_PROCEDURAL_AXIS_TO_ID = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SOMATwistSegmentSpec:
    """Generated twist helpers for one public SOMA limb segment."""

    start_joint: str
    end_joint: str
    twist_joints: tuple[str, ...]
    reverse: bool = False
    parent_joint: str | None = None
    source_axis: int = 0
    source_sign: float = 1.0


@dataclass(frozen=True)
class SOMANamedMatrixEntry:
    """One named sparse procedural matrix entry from the JSON sidecar."""

    row: str
    column: str
    value: float


@dataclass(frozen=True)
class SOMAProceduralParameterMatrices:
    """Compiled SOMA procedural parameter matrices."""

    rotation: torch.Tensor
    translation: torch.Tensor
    segment_fractions: torch.Tensor


@dataclass(frozen=True)
class SOMAProceduralTransformOutput:
    """Optional outputs from one procedural parameter transform call."""

    rotations: torch.Tensor | None = None
    transforms: torch.Tensor | None = None


@dataclass(frozen=True)
class SOMAProceduralTransformDefinition:
    """Portable SOMA procedural-control rig definition loaded from JSON."""

    schema_version: str
    modes: tuple[str, ...]
    rotation_extraction_modes: tuple[str, ...]
    public_joint_names: tuple[str, ...]
    segments: tuple[SOMATwistSegmentSpec, ...]
    rotation_entries: tuple[SOMANamedMatrixEntry, ...]
    translation_entries: tuple[SOMANamedMatrixEntry, ...]
    path: Path | None = None

    @property
    def main_joint_names(self) -> tuple[str, ...]:
        """Main non-procedural joint names from the portable JSON schema."""
        return self.public_joint_names


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_mode(value: Any, field: str, modes: Sequence[str]) -> str:
    mode = _require_string(value, field)
    if mode not in modes:
        raise ValueError(f"{field} must be one of {tuple(modes)}, got {mode!r}")
    return mode


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    strings = tuple(_require_string(item, f"{field}[]") for item in _require_sequence(value, field))
    duplicates = sorted({item for item in strings if strings.count(item) > 1})
    if duplicates:
        raise ValueError(f"{field} contains duplicate names: {duplicates}")
    return strings


def _axis_id(value: Any, field: str) -> int:
    if isinstance(value, str):
        axis = SOMA_PROCEDURAL_AXIS_TO_ID.get(value.lower())
        if axis is not None:
            return axis
    elif isinstance(value, int) and value in (0, 1, 2):
        return value
    raise ValueError(f"{field} must be one of 'x', 'y', 'z', 0, 1, or 2")


def _sign(value: Any, field: str) -> float:
    if value in (-1, -1.0):
        return -1.0
    if value in (1, 1.0):
        return 1.0
    raise ValueError(f"{field} must be -1.0 or 1.0")


def _require_segments(
    segments: Sequence[SOMATwistSegmentSpec] | None,
    context: str,
) -> tuple[SOMATwistSegmentSpec, ...]:
    if segments is None:
        raise ValueError(
            f"{context} requires explicit procedural twist segments. Load "
            f"{SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME} with "
            "load_soma_procedural_transform_definition() and pass definition.segments."
        )
    return tuple(segments)


def _require_rotation_extraction_modes(
    rotation_extraction_modes: Sequence[str] | None,
    twist_joint_names: Sequence[str],
    context: str,
) -> tuple[str, ...]:
    if rotation_extraction_modes is None:
        raise ValueError(
            f"{context} requires explicit rotation extraction modes. Load "
            f"{SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME} with "
            "load_soma_procedural_transform_definition() and pass "
            "definition.rotation_extraction_modes."
        )
    modes = tuple(rotation_extraction_modes)
    if len(modes) != len(twist_joint_names):
        raise ValueError(
            f"{context} expected {len(twist_joint_names)} rotation extraction modes, "
            f"got {len(modes)}"
        )
    for mode in modes:
        _require_mode(
            mode, f"{context} rotation_extraction_modes[]", SOMA_PROCEDURAL_TRANSFORM_MODES
        )
    return modes


def _parse_rotation_extraction_modes(
    root: Mapping[str, Any],
    modes: Sequence[str],
    twist_joint_names: Sequence[str],
) -> tuple[str, ...]:
    raw = root.get("rotation_extraction")
    if raw is None:
        raise ValueError("rotation_extraction is required")
    if isinstance(raw, str):
        return (_require_mode(raw, "rotation_extraction", modes),) * len(twist_joint_names)

    config = _require_mapping(raw, "rotation_extraction")
    per_joint = _require_mapping(
        config.get("per_procedural_joint", {}),
        "rotation_extraction.per_procedural_joint",
    )
    unknown_joints = sorted(set(per_joint) - set(twist_joint_names))
    if unknown_joints:
        raise ValueError(
            "rotation_extraction.per_procedural_joint references unknown procedural joints: "
            f"{unknown_joints}"
        )

    default = config.get("default")
    if default is None:
        missing = [name for name in twist_joint_names if name not in per_joint]
        if missing:
            raise ValueError(
                "rotation_extraction.default is required unless every procedural joint "
                f"has an override; missing: {missing}"
            )
        default_mode = None
    else:
        default_mode = _require_mode(default, "rotation_extraction.default", modes)

    parsed_modes = []
    for joint_name in twist_joint_names:
        raw_mode = per_joint.get(joint_name, default_mode)
        parsed_modes.append(
            _require_mode(
                raw_mode,
                f"rotation_extraction.per_procedural_joint[{joint_name}]",
                modes,
            )
        )
    return tuple(parsed_modes)


def _parse_named_sparse_matrix(
    matrix_data: Any,
    matrix_name: str,
    valid_rows: set[str],
    valid_columns: set[str],
    require_entries: bool = True,
) -> tuple[SOMANamedMatrixEntry, ...]:
    matrix = _require_mapping(matrix_data, f"parameter_matrices.{matrix_name}")
    if matrix.get("format") not in (None, "sparse_coo_named"):
        raise ValueError(f"parameter_matrices.{matrix_name}.format must be 'sparse_coo_named'")
    if matrix.get("dtype") not in (None, "float32"):
        raise ValueError(f"parameter_matrices.{matrix_name}.dtype must be 'float32'")
    entries = _require_sequence(
        matrix.get("entries", []),
        f"parameter_matrices.{matrix_name}.entries",
    )
    if require_entries and not entries:
        raise ValueError(f"parameter_matrices.{matrix_name}.entries must not be empty")
    seen = set()
    parsed_entries = []
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"parameter_matrices.{matrix_name}.entries[{index}]")
        row = _require_string(entry.get("row"), f"parameter_matrices.{matrix_name}.entries[].row")
        column = _require_string(
            entry.get("column"),
            f"parameter_matrices.{matrix_name}.entries[].column",
        )
        if row not in valid_rows:
            raise ValueError(f"unknown {matrix_name} matrix row: {row!r}")
        if column not in valid_columns:
            raise ValueError(f"unknown {matrix_name} matrix column: {column!r}")
        try:
            float(entry["value"])
        except KeyError as e:
            raise ValueError(f"parameter_matrices.{matrix_name}.entries[].value is required") from e
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"parameter_matrices.{matrix_name}.entries[].value must be numeric"
            ) from e
        key = (row, column)
        if key in seen:
            raise ValueError(f"duplicate {matrix_name} matrix entry for {row!r}, {column!r}")
        seen.add(key)
        parsed_entries.append(
            SOMANamedMatrixEntry(row=row, column=column, value=float(entry["value"]))
        )
    return tuple(parsed_entries)


def parse_soma_procedural_transform_definition(
    data: Mapping[str, Any],
    path: str | Path | None = None,
) -> SOMAProceduralTransformDefinition:
    """Validate and parse a portable SOMA procedural-control rig definition."""
    root = _require_mapping(data, "definition")
    schema_version = _require_string(root.get("schema_version"), "schema_version")
    modes = _string_tuple(root.get("modes"), "modes")
    unknown_modes = sorted(set(modes) - set(SOMA_PROCEDURAL_TRANSFORM_MODES))
    if unknown_modes:
        raise ValueError(f"unknown procedural transform modes: {unknown_modes}")

    channel_extractors = _require_mapping(root.get("channel_extractors"), "channel_extractors")
    unknown_extractors = sorted(set(channel_extractors) - set(SOMA_PROCEDURAL_TRANSFORM_MODES))
    if unknown_extractors:
        raise ValueError(f"unknown channel extractors: {unknown_extractors}")
    missing_extractors = [mode for mode in modes if mode not in channel_extractors]
    if missing_extractors:
        raise ValueError(f"missing channel extractors for modes: {missing_extractors}")

    public_rig = _require_mapping(root.get("public_rig_derivation"), "public_rig_derivation")
    public_joint_names = _string_tuple(
        public_rig.get("main_joint_names"),
        "public_rig_derivation.main_joint_names",
    )
    public_joint_set = set(public_joint_names)

    segments = []
    for index, raw_segment in enumerate(_require_sequence(root.get("segments"), "segments")):
        segment_data = _require_mapping(raw_segment, f"segments[{index}]")
        start_joint = _require_string(segment_data.get("start_joint"), "segments[].start_joint")
        end_joint = _require_string(segment_data.get("end_joint"), "segments[].end_joint")
        parent_joint_raw = segment_data.get("parent_joint")
        parent_joint = (
            _require_string(parent_joint_raw, "segments[].parent_joint")
            if parent_joint_raw is not None
            else None
        )
        twist_joints = _string_tuple(segment_data.get("twist_joints"), "segments[].twist_joints")
        control_names = (start_joint, end_joint)
        if parent_joint is not None:
            control_names = (*control_names, parent_joint)
        missing_controls = [name for name in control_names if name not in public_joint_set]
        if missing_controls:
            raise ValueError(
                f"segments[{index}] references joints outside the public rig: {missing_controls}"
            )
        segments.append(
            SOMATwistSegmentSpec(
                start_joint=start_joint,
                end_joint=end_joint,
                twist_joints=twist_joints,
                reverse=bool(segment_data.get("reverse", False)),
                parent_joint=parent_joint,
                source_axis=_axis_id(
                    segment_data.get("source_axis", "x"), "segments[].source_axis"
                ),
                source_sign=_sign(segment_data.get("source_sign", 1.0), "segments[].source_sign"),
            )
        )

    parsed_segments = tuple(segments)
    twist_joint_names = tuple(_twist_joint_names(parsed_segments))
    duplicate_outputs = sorted(
        {name for name in twist_joint_names if twist_joint_names.count(name) > 1}
    )
    if duplicate_outputs:
        raise ValueError(f"duplicate procedural outputs: {duplicate_outputs}")
    rotation_extraction_modes = _parse_rotation_extraction_modes(root, modes, twist_joint_names)

    parameter_matrices = _require_mapping(root.get("parameter_matrices"), "parameter_matrices")
    rotation_entries = _parse_named_sparse_matrix(
        parameter_matrices.get("rotation"),
        "rotation",
        set(twist_joint_names),
        public_joint_set,
    )
    translation_entries = _parse_named_sparse_matrix(
        parameter_matrices.get("translation"),
        "translation",
        set(twist_joint_names),
        public_joint_set | set(twist_joint_names),
    )

    return SOMAProceduralTransformDefinition(
        schema_version=schema_version,
        modes=modes,
        rotation_extraction_modes=rotation_extraction_modes,
        public_joint_names=public_joint_names,
        segments=parsed_segments,
        rotation_entries=rotation_entries,
        translation_entries=translation_entries,
        path=Path(path) if path is not None else None,
    )


def load_soma_procedural_transform_definition(
    path: str | Path,
) -> SOMAProceduralTransformDefinition:
    """Load a portable SOMA procedural-control rig definition JSON file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid SOMA procedural transform definition JSON at {path}: {e}") from e
    try:
        return parse_soma_procedural_transform_definition(data, path=path)
    except ValueError as e:
        raise ValueError(f"Invalid SOMA procedural transform definition at {path}: {e}") from e


def _name_list(joint_names: Sequence[str]) -> list[str]:
    return [str(name) for name in joint_names]


def _twist_joint_names(segments: Sequence[SOMATwistSegmentSpec]) -> list[str]:
    return [twist_joint for segment in segments for twist_joint in segment.twist_joints]


def _dense_named_matrix(
    entries: Sequence[SOMANamedMatrixEntry],
    row_names: Sequence[str],
    column_names: Sequence[str],
    matrix_name: str,
    *,
    base_identity: bool = False,
) -> torch.Tensor:
    if base_identity:
        if len(row_names) != len(column_names) or tuple(row_names) != tuple(column_names):
            raise ValueError(f"{matrix_name} identity base requires matching row/column names")
        matrix = torch.eye(len(row_names), dtype=torch.float32)
    else:
        matrix = torch.zeros((len(row_names), len(column_names)), dtype=torch.float32)
    rows = {name: index for index, name in enumerate(row_names)}
    columns = {name: index for index, name in enumerate(column_names)}
    cleared_rows = set()
    for entry in entries:
        try:
            row = rows[entry.row]
            column = columns[entry.column]
        except KeyError as e:
            raise ValueError(
                f"{matrix_name} matrix references unknown row/column: "
                f"{entry.row!r}, {entry.column!r}"
            ) from e
        if base_identity and row not in cleared_rows:
            matrix[row] = 0.0
            cleared_rows.add(row)
        matrix[row, column] = float(entry.value)
    return matrix


def _build_parameter_matrices(
    source_by_name: Mapping[str, int],
    target_by_name: Mapping[str, int],
    segments: Sequence[SOMATwistSegmentSpec],
    rotation_entries: Sequence[SOMANamedMatrixEntry],
    translation_entries: Sequence[SOMANamedMatrixEntry],
) -> SOMAProceduralParameterMatrices:
    """Compile SOMA-owned procedural rotation and translation matrices."""
    source_names = tuple(source_by_name)
    target_names = tuple(target_by_name)
    twist_names = tuple(_twist_joint_names(segments))
    rotation_matrix = _dense_named_matrix(
        rotation_entries,
        twist_names,
        source_names,
        "rotation",
    )
    translation_matrix = _dense_named_matrix(
        translation_entries,
        target_names,
        target_names,
        "translation",
        base_identity=True,
    )
    fractions = []
    for segment in segments:
        end_target_idx = target_by_name[segment.end_joint]
        for twist_joint in segment.twist_joints:
            fractions.append(translation_matrix[target_by_name[twist_joint], end_target_idx])
    segment_fractions = (
        torch.stack(fractions).to(dtype=torch.float32)
        if fractions
        else torch.empty(0, dtype=torch.float32)
    )
    return SOMAProceduralParameterMatrices(
        rotation=rotation_matrix,
        translation=translation_matrix,
        segment_fractions=segment_fractions,
    )


def _build_rotation_parameter_matrices_by_mode(
    rotation_parameter_matrix: torch.Tensor,
    mode_names: Sequence[str],
    rotation_extraction_modes: Sequence[str],
) -> torch.Tensor:
    mode_to_idx = {mode: index for index, mode in enumerate(mode_names)}
    matrices = torch.zeros(
        (len(mode_names), *rotation_parameter_matrix.shape),
        dtype=rotation_parameter_matrix.dtype,
    )
    for row, mode in enumerate(rotation_extraction_modes):
        matrices[mode_to_idx[mode], row] = rotation_parameter_matrix[row]
    return matrices


def has_soma_twist_joints(
    joint_names: Sequence[str],
    segments: Sequence[SOMATwistSegmentSpec] | None = None,
) -> bool:
    """Return whether all SOMA procedural twist joints are present."""
    segments = _require_segments(segments, "has_soma_twist_joints()")
    names = set(_name_list(joint_names))
    return all(name in names for name in _twist_joint_names(segments))


def derive_soma_rig_without_procedural_joints(
    rig_data: Mapping[str, Any],
    public_joint_names: Sequence[str] | None = None,
    segments: Sequence[SOMATwistSegmentSpec] | None = None,
) -> dict[str, Any]:
    """Derive the public SOMA rig and aggregate removed joint weights to parents.

    The v0027 template with twist joints is the universal source rig. This helper
    derives the legacy public rig on the fly by pruning generated procedural and
    auxiliary joints, remapping hierarchy indices, and moving each pruned joint's
    skin weights to its nearest kept parent.
    """
    joint_names = _name_list(rig_data["joint_names"])
    if public_joint_names is None:
        segments = _require_segments(
            segments,
            "derive_soma_rig_without_procedural_joints() without public_joint_names",
        )
        twist_names = set(_twist_joint_names(segments))
        keep_ids = np.array(
            [idx for idx, name in enumerate(joint_names) if name not in twist_names],
            dtype=np.int64,
        )
    else:
        public_names = _name_list(public_joint_names)
        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        missing_public = [name for name in public_names if name not in name_to_idx]
        if missing_public:
            raise ValueError(
                f"Template rig is missing public SOMA joints: {sorted(set(missing_public))}"
            )
        keep_ids = np.asarray([name_to_idx[name] for name in public_names], dtype=np.int64)
    keep_id_set = {int(idx) for idx in keep_ids}
    remove_ids = {idx for idx in range(len(joint_names)) if idx not in keep_id_set}
    if not remove_ids:
        return dict(rig_data)

    parent_ids = np.asarray(rig_data["joint_parent_ids"], dtype=np.int64)
    old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(keep_ids)}

    def nearest_kept_parent(old_idx: int) -> int:
        parent = int(parent_ids[old_idx])
        while parent in remove_ids and parent != int(parent_ids[parent]):
            parent = int(parent_ids[parent])
        return parent

    new_parent_ids = np.zeros((len(keep_ids),), dtype=np.int32)
    for new_idx, old_idx_np in enumerate(keep_ids):
        old_idx = int(old_idx_np)
        parent = int(parent_ids[old_idx])
        if parent == old_idx:
            new_parent_ids[new_idx] = new_idx
            continue
        while parent in remove_ids and parent != int(parent_ids[parent]):
            parent = int(parent_ids[parent])
        new_parent_ids[new_idx] = old_to_new[parent]

    weights = np.asarray(
        csc_matrix(
            (
                rig_data["skinning_weights_data"],
                rig_data["skinning_weights_indices"],
                rig_data["skinning_weights_indptr"],
            ),
            shape=rig_data["skinning_weights_shape"],
        ).todense(),
        dtype=np.float32,
    )
    for removed_idx in sorted(remove_ids):
        weights[:, nearest_kept_parent(removed_idx)] += weights[:, removed_idx]
    weights = weights[:, keep_ids]
    weights_sparse = csc_matrix(weights)

    bind_pose_world = np.asarray(rig_data["bind_pose_world"], dtype=np.float32)[keep_ids]
    t_pose_world = np.asarray(rig_data["t_pose_world"], dtype=np.float32)[keep_ids]
    bind_pose_local = joint_world_to_local(
        torch.from_numpy(bind_pose_world),
        new_parent_ids,
    ).numpy()
    t_pose_local = joint_world_to_local(
        torch.from_numpy(t_pose_world),
        new_parent_ids,
    ).numpy()

    out = dict(rig_data)
    out.update(
        joint_names=np.asarray([joint_names[int(idx)] for idx in keep_ids]),
        joint_parent_ids=new_parent_ids,
        bind_pose_world=bind_pose_world.astype(np.float32),
        bind_pose_local=bind_pose_local.astype(np.float32),
        t_pose_world=t_pose_world.astype(np.float32),
        t_pose_local=t_pose_local.astype(np.float32),
        skinning_weights_data=weights_sparse.data.astype(np.float32),
        skinning_weights_indices=weights_sparse.indices.astype(np.int32),
        skinning_weights_indptr=weights_sparse.indptr.astype(np.int32),
        skinning_weights_shape=np.array(weights_sparse.shape, dtype=np.int32),
    )
    return out


def _build_source_twist_channels(
    source_by_name: Mapping[str, int],
    segments: Sequence[SOMATwistSegmentSpec],
) -> tuple[torch.Tensor, torch.Tensor]:
    axis_ids = torch.zeros((len(source_by_name),), dtype=torch.long)
    signs = torch.ones((len(source_by_name),), dtype=torch.float32)
    assigned = {}
    for segment in segments:
        if segment.source_axis not in (0, 1, 2):
            raise ValueError(f"source_axis must be 0, 1, or 2, got {segment.source_axis}")
        for name in (segment.start_joint, segment.end_joint):
            spec = (segment.source_axis, float(segment.source_sign))
            if name in assigned and assigned[name] != spec:
                raise ValueError(f"Conflicting SOMA twist source channel for joint {name!r}")
            assigned[name] = spec
    for name, (axis, sign) in assigned.items():
        idx = source_by_name[name]
        axis_ids[idx] = axis
        signs[idx] = sign
    return axis_ids, signs


def _build_twist_output_channels(
    segments: Sequence[SOMATwistSegmentSpec],
) -> tuple[torch.Tensor, torch.Tensor]:
    axis_ids = []
    signs = []
    for segment in segments:
        axis_ids.extend([segment.source_axis] * len(segment.twist_joints))
        signs.extend([float(segment.source_sign)] * len(segment.twist_joints))
    return torch.tensor(axis_ids, dtype=torch.long), torch.tensor(signs, dtype=torch.float32)


def local_x_euler_from_matrix(rotations: torch.Tensor) -> torch.Tensor:
    """Return the local X Euler angle for matrices that may include swing.

    This is the source extraction used by the ``local_x_euler`` twist mode.
    ``local_x_swing_twist`` uses quaternion projection instead.
    """
    return torch.atan2(rotations[..., 2, 1], rotations[..., 1, 1])


def _local_euler_xyz_from_matrix(rotations: torch.Tensor) -> torch.Tensor:
    euler = matrix_to_euler_xyz(rotations)
    return torch.stack((local_x_euler_from_matrix(rotations), euler[..., 1], euler[..., 2]), dim=-1)


def _axis_rotations(
    angles: torch.Tensor,
    axis_ids: torch.Tensor,
    axis_signs: torch.Tensor,
) -> torch.Tensor:
    rotvecs = torch.zeros((*angles.shape, 3), dtype=angles.dtype, device=angles.device)
    signed_angles = angles * axis_signs.to(dtype=angles.dtype, device=angles.device)[None]
    scatter_ids = (
        axis_ids.to(device=angles.device)
        .reshape(1, -1, 1)
        .expand(
            angles.shape[0],
            -1,
            1,
        )
    )
    rotvecs.scatter_(-1, scatter_ids, signed_angles.unsqueeze(-1))
    rotvecs = rotvecs.reshape(-1, 3)
    return batch_rodrigues(rotvecs, dtype=angles.dtype).reshape(*angles.shape, 3, 3)


def _swing_twist_channels_from_matrix(
    rotations: torch.Tensor,
    axis_ids: torch.Tensor,
    axis_signs: torch.Tensor,
) -> torch.Tensor:
    quaternions = matrix_to_quaternion_xyzw(rotations)
    axis_signs = axis_signs.to(dtype=rotations.dtype, device=rotations.device)
    twist_angles = quaternion_twist_angle_xyzw(quaternions, axis_ids)
    return twist_angles * axis_signs[None]


def _normalize_vectors(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)


def _project_to_plane(
    vectors: torch.Tensor,
    normals: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    projected = vectors - (vectors * normals).sum(dim=-1, keepdim=True) * normals
    return _normalize_vectors(projected, eps=eps)


def _bind_alignment_quaternions(
    bind_world_transforms: torch.Tensor,
    start_ids: torch.Tensor,
    end_ids: torch.Tensor,
) -> torch.Tensor:
    bind_world_transforms = bind_world_transforms.to(dtype=torch.float32)
    start = bind_world_transforms[start_ids]
    end = bind_world_transforms[end_ids]
    start_rot = start[..., :3, :3]
    span = end[..., :3, 3] - start[..., :3, 3]
    span = _normalize_vectors(span)

    local_x = torch.tensor([1.0, 0.0, 0.0], dtype=start.dtype, device=start.device)
    local_y = torch.tensor([0.0, 1.0, 0.0], dtype=start.dtype, device=start.device)
    local_z = torch.tensor([0.0, 0.0, 1.0], dtype=start.dtype, device=start.device)
    up_x = start_rot @ local_x
    x_sign = torch.where((up_x * span).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0)
    x_axis = span * x_sign

    y_candidate = start_rot @ local_y
    z_candidate = start_rot @ local_z
    world_y = local_y.reshape(1, 3).expand_as(x_axis)
    world_z = local_z.reshape(1, 3).expand_as(x_axis)

    y_axis = _project_to_plane(y_candidate, x_axis)
    y_norm = torch.linalg.norm(
        y_candidate - (y_candidate * x_axis).sum(-1, keepdim=True) * x_axis, dim=-1
    )
    z_proj = _project_to_plane(z_candidate, x_axis)
    z_norm = torch.linalg.norm(
        z_candidate - (z_candidate * x_axis).sum(-1, keepdim=True) * x_axis, dim=-1
    )
    wy_proj = _project_to_plane(world_y, x_axis)
    wy_norm = torch.linalg.norm(world_y - (world_y * x_axis).sum(-1, keepdim=True) * x_axis, dim=-1)
    wz_proj = _project_to_plane(world_z, x_axis)

    y_axis = torch.where((y_norm > 1e-8).reshape(-1, 1), y_axis, z_proj)
    y_axis = torch.where(((y_norm > 1e-8) | (z_norm > 1e-8)).reshape(-1, 1), y_axis, wy_proj)
    y_axis = torch.where(
        ((y_norm > 1e-8) | (z_norm > 1e-8) | (wy_norm > 1e-8)).reshape(-1, 1),
        y_axis,
        wz_proj,
    )

    z_axis = _normalize_vectors(torch.cross(x_axis, y_axis, dim=-1))
    y_axis = _normalize_vectors(torch.cross(z_axis, x_axis, dim=-1))
    align_rot = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    return matrix_to_quaternion_xyzw(align_rot)


def _aligned_virtual_quaternions(
    world_rotations: torch.Tensor,
    bind_quaternions: torch.Tensor,
    align_quaternions: torch.Tensor,
    segment_ids: torch.Tensor,
    joint_ids: torch.Tensor,
) -> torch.Tensor:
    q_current = matrix_to_quaternion_xyzw_stable(world_rotations[:, joint_ids])
    q_bind_inv = quaternion_conjugate_xyzw(bind_quaternions[joint_ids])
    q_align = align_quaternions[segment_ids].unsqueeze(0)
    q = quaternion_multiply_xyzw(
        quaternion_multiply_xyzw(q_current, q_bind_inv.unsqueeze(0)),
        q_align,
    )
    return quaternion_normalize_xyzw(q)


class SOMAProceduralParameterTransform(nn.Module):
    """Expand the public SOMA pose with procedural parameter matrices."""

    def __init__(
        self,
        source_joint_names: Sequence[str],
        target_joint_names: Sequence[str],
        rotation_extraction_modes: Sequence[str] | None = None,
        segments: Sequence[SOMATwistSegmentSpec] | None = None,
        rotation_entries: Sequence[SOMANamedMatrixEntry] | None = None,
        translation_entries: Sequence[SOMANamedMatrixEntry] | None = None,
        target_t_pose_world: torch.Tensor | np.ndarray | None = None,
        target_joint_parent_ids: Sequence[int] | torch.Tensor | np.ndarray | None = None,
        target_bind_pose_world: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        source_names = _name_list(source_joint_names)
        target_names = _name_list(target_joint_names)
        segments = _require_segments(segments, "SOMAProceduralParameterTransform")
        source_by_name = {name: idx for idx, name in enumerate(source_names)}
        target_by_name = {name: idx for idx, name in enumerate(target_names)}
        twist_names = _twist_joint_names(segments)
        rotation_extraction_modes = _require_rotation_extraction_modes(
            rotation_extraction_modes,
            twist_names,
            "SOMAProceduralParameterTransform",
        )
        if rotation_entries is None or translation_entries is None:
            raise ValueError(
                "SOMAProceduralParameterTransform requires JSON sidecar matrix entries. "
                f"Load {SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME} with "
                "load_soma_procedural_transform_definition() and pass "
                "definition.rotation_entries and definition.translation_entries."
            )
        mode_names = tuple(
            mode_name
            for mode_name in SOMA_PROCEDURAL_TRANSFORM_MODES
            if mode_name in rotation_extraction_modes
        )

        missing_source = []
        for segment in segments:
            for name in (segment.start_joint, segment.end_joint):
                if name not in source_by_name:
                    missing_source.append(name)
        missing_target = [name for name in twist_names if name not in target_by_name]
        missing_control_targets = [name for name in source_names if name not in target_by_name]
        duplicate_targets = sorted({name for name in twist_names if twist_names.count(name) > 1})
        if missing_source or missing_target or missing_control_targets or duplicate_targets:
            parts = []
            if missing_source:
                parts.append(f"missing source joints: {sorted(set(missing_source))}")
            if missing_target:
                parts.append(f"missing twist joints: {sorted(set(missing_target))}")
            if missing_control_targets:
                parts.append(
                    f"missing public SOMA joints in twist rig: {sorted(missing_control_targets)}"
                )
            if duplicate_targets:
                parts.append(f"duplicate twist joints: {duplicate_targets}")
            raise ValueError("Invalid SOMA procedural twist rig mapping; " + "; ".join(parts))

        control_source_ids = []
        control_target_ids = []
        for name in source_names:
            control_source_ids.append(source_by_name[name])
            control_target_ids.append(target_by_name[name])
        twist_target_ids = [target_by_name[name] for name in twist_names]
        source_axis_ids, source_axis_signs = _build_source_twist_channels(
            source_by_name,
            segments,
        )
        twist_axis_ids, twist_axis_signs = _build_twist_output_channels(segments)

        self.mode = (
            rotation_extraction_modes[0] if len(set(rotation_extraction_modes)) == 1 else None
        )
        self.rotation_extraction_modes = rotation_extraction_modes
        self.rotation_extraction_mode_names = mode_names
        self.source_joint_names = tuple(source_names)
        self.target_joint_names = tuple(target_names)
        self.segments = segments
        self.twist_joint_names = tuple(twist_names)
        self.register_buffer(
            "control_source_ids",
            torch.tensor(control_source_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "control_target_ids",
            torch.tensor(control_target_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "twist_target_ids",
            torch.tensor(twist_target_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("source_twist_axis_ids", source_axis_ids, persistent=False)
        self.register_buffer("source_twist_axis_signs", source_axis_signs, persistent=False)
        self.register_buffer("twist_axis_ids", twist_axis_ids, persistent=False)
        self.register_buffer(
            "twist_axis_signs",
            twist_axis_signs,
            persistent=False,
        )
        if target_t_pose_world is not None:
            if isinstance(target_t_pose_world, np.ndarray):
                target_t_pose_world = torch.from_numpy(target_t_pose_world)
            if target_t_pose_world.shape[-2:] != (4, 4):
                raise ValueError(
                    "target_t_pose_world must have shape (J, 4, 4), "
                    f"got {target_t_pose_world.shape}"
                )
            if target_t_pose_world.shape[0] != len(target_names):
                raise ValueError(
                    "target_t_pose_world must have the same joint count as target_joint_names"
                )
        if target_bind_pose_world is not None:
            if isinstance(target_bind_pose_world, np.ndarray):
                target_bind_pose_world = torch.from_numpy(target_bind_pose_world)
            if target_bind_pose_world.shape[-2:] != (4, 4):
                raise ValueError(
                    "target_bind_pose_world must have shape (J, 4, 4), "
                    f"got {target_bind_pose_world.shape}"
                )
            if target_bind_pose_world.shape[0] != len(target_names):
                raise ValueError(
                    "target_bind_pose_world must have the same joint count as target_joint_names"
                )
        if target_joint_parent_ids is not None:
            target_joint_parent_ids = torch.as_tensor(target_joint_parent_ids, dtype=torch.long)
            if target_joint_parent_ids.shape != (len(target_names),):
                raise ValueError(
                    "target_joint_parent_ids must have shape "
                    f"({len(target_names)},), got {tuple(target_joint_parent_ids.shape)}"
                )
        target_t_pose_local = None
        if target_t_pose_world is not None and target_joint_parent_ids is not None:
            target_t_pose_local = joint_world_to_local(target_t_pose_world, target_joint_parent_ids)
        self.register_buffer(
            "target_joint_parent_ids",
            target_joint_parent_ids
            if target_joint_parent_ids is not None
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "target_t_pose_local_rotations",
            target_t_pose_local[..., :3, :3]
            if target_t_pose_local is not None
            else torch.empty(0, 3, 3, dtype=torch.float32),
            persistent=False,
        )
        parameter_matrices = _build_parameter_matrices(
            source_by_name,
            target_by_name,
            segments,
            rotation_entries,
            translation_entries,
        )
        source_parent_ids = None
        source_t_pose_world = None
        source_t_pose_local = None
        source_joint_orient = None
        source_joint_orient_parent_t = None
        bind_quaternions = None
        bind_align_quaternions = None
        segment_start_ids = []
        segment_end_ids = []
        segment_parent_ids = []
        segment_reverse_mask = []
        aligned_virtual_segment_ids = []
        aligned_virtual_joint_ids = []
        twist_parent_target_ids = []
        if target_joint_parent_ids is not None:
            twist_parent_target_ids = [
                int(target_joint_parent_ids[target_by_name[name]].item()) for name in twist_names
            ]
        if target_joint_parent_ids is not None:
            source_target_ids = torch.tensor(
                [target_by_name[name] for name in source_names],
                dtype=torch.long,
            )
            source_target_id_to_source_id = {
                int(target_idx): source_idx
                for source_idx, target_idx in enumerate(source_target_ids.tolist())
            }
            source_parent_list = []
            for target_idx in source_target_ids.tolist():
                parent_idx = int(target_joint_parent_ids[int(target_idx)].item())
                while parent_idx not in source_target_id_to_source_id:
                    next_parent_idx = int(target_joint_parent_ids[parent_idx].item())
                    if next_parent_idx == parent_idx:
                        break
                    parent_idx = next_parent_idx
                source_parent_list.append(source_target_id_to_source_id.get(parent_idx, 0))
            source_parent_ids = torch.tensor(source_parent_list, dtype=torch.long)
            if target_t_pose_world is not None:
                source_t_pose_world = target_t_pose_world[source_target_ids].to(dtype=torch.float32)
            if target_bind_pose_world is not None:
                source_bind_pose_world = target_bind_pose_world[source_target_ids].to(
                    dtype=torch.float32
                )
            else:
                source_bind_pose_world = None
            if source_t_pose_world is not None:
                source_t_pose_local = joint_world_to_local(source_t_pose_world, source_parent_ids)
                source_joint_orient, source_joint_orient_parent_t = precompute_joint_orient(
                    source_t_pose_world,
                    source_parent_ids,
                )
            if source_bind_pose_world is not None:
                bind_quaternions = matrix_to_quaternion_xyzw(source_bind_pose_world[..., :3, :3])
            for segment in segments:
                start_idx = source_by_name[segment.start_joint]
                end_idx = source_by_name[segment.end_joint]
                if segment.parent_joint is not None:
                    parent_idx = source_by_name[segment.parent_joint]
                else:
                    parent_idx = int(source_parent_ids[start_idx].item())
                segment_start_ids.append(start_idx)
                segment_end_ids.append(end_idx)
                segment_parent_ids.append(parent_idx)
                segment_reverse_mask.append(bool(segment.reverse))
            segment_start_ids_t = torch.tensor(segment_start_ids, dtype=torch.long)
            segment_end_ids_t = torch.tensor(segment_end_ids, dtype=torch.long)
            if source_bind_pose_world is not None:
                bind_align_quaternions = _bind_alignment_quaternions(
                    source_bind_pose_world,
                    segment_start_ids_t,
                    segment_end_ids_t,
                )
            segment_ids = list(range(len(segment_start_ids)))
            aligned_virtual_segment_ids = segment_ids + segment_ids + segment_ids
            aligned_virtual_joint_ids = segment_end_ids + segment_start_ids + segment_parent_ids
        single_twist_axis = None
        if twist_axis_ids.numel() > 0 and bool(torch.all(twist_axis_ids == twist_axis_ids[0])):
            single_twist_axis = int(twist_axis_ids[0])
        self.register_buffer(
            "segment_fractions", parameter_matrices.segment_fractions, persistent=False
        )
        self.register_buffer(
            "rotation_parameter_matrix",
            parameter_matrices.rotation,
            persistent=False,
        )
        self.register_buffer(
            "rotation_parameter_matrices_by_mode",
            _build_rotation_parameter_matrices_by_mode(
                parameter_matrices.rotation,
                mode_names,
                rotation_extraction_modes,
            ),
            persistent=False,
        )
        self.register_buffer(
            "translation_parameter_matrix",
            parameter_matrices.translation,
            persistent=False,
        )
        self.register_buffer(
            "source_parent_ids",
            source_parent_ids
            if source_parent_ids is not None
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "source_t_pose_local",
            source_t_pose_local
            if source_t_pose_local is not None
            else torch.empty(0, 4, 4, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "source_joint_orient",
            source_joint_orient
            if source_joint_orient is not None
            else torch.empty(0, 3, 3, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "source_joint_orient_parent_t",
            source_joint_orient_parent_t
            if source_joint_orient_parent_t is not None
            else torch.empty(0, 3, 3, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "source_bind_quaternions",
            bind_quaternions
            if bind_quaternions is not None
            else torch.empty(0, 4, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "segment_bind_align_quaternions",
            bind_align_quaternions
            if bind_align_quaternions is not None
            else torch.empty(0, 4, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "segment_start_source_ids",
            torch.tensor(segment_start_ids, dtype=torch.long)
            if segment_start_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "segment_end_source_ids",
            torch.tensor(segment_end_ids, dtype=torch.long)
            if segment_end_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "segment_parent_source_ids",
            torch.tensor(segment_parent_ids, dtype=torch.long)
            if segment_parent_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "segment_source_ids",
            torch.arange(len(segment_start_ids), dtype=torch.long)
            if segment_start_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "segment_reverse_mask",
            torch.tensor(segment_reverse_mask, dtype=torch.bool)
            if segment_reverse_mask
            else torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "aligned_virtual_segment_ids",
            torch.tensor(aligned_virtual_segment_ids, dtype=torch.long)
            if aligned_virtual_segment_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "aligned_virtual_joint_ids",
            torch.tensor(aligned_virtual_joint_ids, dtype=torch.long)
            if aligned_virtual_joint_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "twist_parent_target_ids",
            torch.tensor(twist_parent_target_ids, dtype=torch.long)
            if twist_parent_target_ids
            else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._single_twist_axis = single_twist_axis

    @property
    def twist_joint_indices(self) -> tuple[int, ...]:
        return tuple(int(idx) for idx in self.twist_target_ids.detach().cpu().tolist())

    def apply_source_joint_orient(self, source_rotations: torch.Tensor) -> torch.Tensor:
        """Convert source rotations from T-pose-relative to absolute local rotations."""
        if self.source_joint_orient.numel() == 0:
            return source_rotations
        return apply_joint_orient_local(
            source_rotations,
            self.source_joint_orient.to(
                dtype=source_rotations.dtype, device=source_rotations.device
            ),
            self.source_joint_orient_parent_t.to(
                dtype=source_rotations.dtype,
                device=source_rotations.device,
            ),
        )

    def _target_local_rotations(
        self,
        batch_size: int,
        target_joint_count: int,
        dtype: torch.dtype,
        device: torch.device,
        target_local_rotations: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_local_rotations is None:
            if self.target_t_pose_local_rotations.shape[:1] == (target_joint_count,):
                target_local_rotations = self.target_t_pose_local_rotations
            else:
                return (
                    torch.eye(3, dtype=dtype, device=device)
                    .reshape(1, 1, 3, 3)
                    .expand(batch_size, target_joint_count, 3, 3)
                )
        if target_local_rotations.ndim == 3:
            target_local_rotations = target_local_rotations.unsqueeze(0)
        if target_local_rotations.shape[-3:] != (target_joint_count, 3, 3):
            raise ValueError(
                "target_local_rotations must have shape (B, target_joint_count, 3, 3), "
                f"got {target_local_rotations.shape}"
            )
        target_local_rotations = target_local_rotations.to(dtype=dtype, device=device)
        if target_local_rotations.shape[0] == 1 and batch_size > 1:
            return target_local_rotations.expand(batch_size, -1, -1, -1)
        if target_local_rotations.shape[0] != batch_size:
            raise ValueError(
                "target_local_rotations batch must match source rotations; "
                f"got {target_local_rotations.shape[0]} and {batch_size}"
            )
        return target_local_rotations

    def expand_source_rotations_with_identity_twists(
        self,
        source_rotations: torch.Tensor,
        target_local_rotations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Copy public source rotations into target order and keep target bind rotations elsewhere."""
        if source_rotations.ndim != 4 or source_rotations.shape[-2:] != (3, 3):
            raise ValueError(
                f"source_rotations must have shape (B, J, 3, 3), got {source_rotations.shape}"
            )
        expected = len(self.source_joint_names)
        if source_rotations.shape[1] != expected:
            raise ValueError(f"Expected {expected} source joints, got {source_rotations.shape[1]}")

        batch_size = source_rotations.shape[0]
        num_target_joints = len(self.target_joint_names)
        target_rotations = self._target_local_rotations(
            batch_size,
            num_target_joints,
            dtype=source_rotations.dtype,
            device=source_rotations.device,
            target_local_rotations=target_local_rotations,
        ).clone()
        target_rotations[:, self.control_target_ids] = source_rotations[:, self.control_source_ids]
        return target_rotations

    def _source_world_transforms_from_rotations(
        self,
        source_rotations: torch.Tensor,
    ) -> torch.Tensor:
        if self.source_parent_ids.numel() == 0 or self.source_t_pose_local.numel() == 0:
            raise RuntimeError(
                "aligned_x_swing_twist requires source_world_transforms, or construction "
                "with target_t_pose_world and target_joint_parent_ids"
            )
        local_translations = self.source_t_pose_local.to(
            dtype=source_rotations.dtype,
            device=source_rotations.device,
        )[..., :3, 3]
        if source_rotations.ndim == 3:
            source_rotations = source_rotations.unsqueeze(0)
        local_t = local_translations.unsqueeze(0).expand(
            source_rotations.shape[0],
            -1,
            -1,
        )
        local_transforms = SE3_from_Rt(source_rotations, local_t)
        levels = compute_skeleton_levels(
            self.source_parent_ids.to(device=source_rotations.device),
            device=source_rotations.device,
        )
        return joint_local_to_world_levelorder(local_transforms, levels)

    def _aligned_twist_channels_from_world(
        self,
        source_world_transforms: torch.Tensor,
    ) -> torch.Tensor:
        if self.source_bind_quaternions.numel() == 0:
            raise RuntimeError(
                "aligned_x_swing_twist requires bind data from target_bind_pose_world "
                "and target_joint_parent_ids"
            )
        device = source_world_transforms.device
        dtype = source_world_transforms.dtype
        bind_quaternions = self.source_bind_quaternions.to(dtype=dtype, device=device)
        align_quaternions = self.segment_bind_align_quaternions.to(dtype=dtype, device=device)
        start_ids = self.segment_start_source_ids.to(device=device)
        end_ids = self.segment_end_source_ids.to(device=device)
        reverse_mask = self.segment_reverse_mask.to(device=device)
        segment_count = self.segment_source_ids.numel()

        virtual_quaternions = _aligned_virtual_quaternions(
            source_world_transforms[..., :3, :3],
            bind_quaternions,
            align_quaternions,
            self.aligned_virtual_segment_ids.to(device=device),
            self.aligned_virtual_joint_ids.to(device=device),
        )
        q_end = virtual_quaternions[:, :segment_count]
        q_start = virtual_quaternions[:, segment_count : 2 * segment_count]
        q_parent = virtual_quaternions[:, 2 * segment_count :]
        segment_local_twist = quaternion_twist_angle_xyzw(
            quaternion_normalize_xyzw(
                quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_start), q_end)
            ),
            0,
        )
        segment_inherited_twist = quaternion_twist_angle_xyzw(
            quaternion_normalize_xyzw(
                quaternion_multiply_xyzw(quaternion_conjugate_xyzw(q_parent), q_start)
            ),
            0,
        )
        twist_values = torch.zeros(
            (source_world_transforms.shape[0], len(self.source_joint_names)),
            dtype=dtype,
            device=device,
        )
        twist_values[:, end_ids] = segment_local_twist
        if reverse_mask.any():
            twist_values[:, start_ids[reverse_mask]] = segment_inherited_twist[:, reverse_mask]
        return twist_values

    def _apply_translation_parameters(
        self,
        target_world_transforms: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the compiled translation parameter matrix to fitted transforms.

        Rotation generation follows the segment driver topology, but the template
        helper translations must be owned by SOMA. The translation matrix keeps
        non-procedural joints unchanged and places twist helpers on the fitted
        public segment so identity or body-part stretch remains coherent.
        """
        added_batch = False
        if target_world_transforms.ndim == 3:
            target_world_transforms = target_world_transforms.unsqueeze(0)
            added_batch = True
        elif target_world_transforms.ndim != 4:
            raise ValueError(
                "target_world_transforms must have shape (J, 4, 4) or (B, J, 4, 4), "
                f"got {target_world_transforms.shape}"
            )
        if target_world_transforms.shape[-2:] != (4, 4):
            raise ValueError(
                "target_world_transforms must have shape (..., J, 4, 4), "
                f"got {target_world_transforms.shape}"
            )
        expected = len(self.target_joint_names)
        if target_world_transforms.shape[-3] != expected:
            raise ValueError(
                f"Expected {expected} target joints, got {target_world_transforms.shape[-3]}"
            )

        out = target_world_transforms.clone()
        matrix = self.translation_parameter_matrix.to(
            dtype=target_world_transforms.dtype, device=target_world_transforms.device
        )
        out[..., :3, 3] = torch.matmul(matrix.unsqueeze(0), target_world_transforms[..., :3, 3])
        return out[0] if added_batch else out

    def _apply_rotation_parameters(
        self,
        source_rotations: torch.Tensor,
        source_world_transforms: torch.Tensor | None = None,
        target_local_rotations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_rotations = self.expand_source_rotations_with_identity_twists(
            source_rotations,
            target_local_rotations=target_local_rotations,
        )
        twist_rotations = self.twist_rotations_from_source(
            source_rotations,
            source_world_transforms,
        )
        target_rotations[:, self.twist_target_ids] = (
            target_rotations[:, self.twist_target_ids] @ twist_rotations
        )
        return target_rotations

    def _twist_angles_from_source(
        self,
        source_rotations: torch.Tensor,
        source_world_transforms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_rotations.ndim != 4 or source_rotations.shape[-2:] != (3, 3):
            raise ValueError(
                f"source_rotations must have shape (B, J, 3, 3), got {source_rotations.shape}"
            )
        expected = len(self.source_joint_names)
        if source_rotations.shape[1] != expected:
            raise ValueError(f"Expected {expected} source joints, got {source_rotations.shape[1]}")

        batch_size = source_rotations.shape[0]
        if source_world_transforms is None and SOMA_ALIGNED_X_SWING_TWIST_MODE in (
            self.rotation_extraction_mode_names
        ):
            source_world_transforms = self._source_world_transforms_from_rotations(source_rotations)
        elif source_world_transforms is not None:
            if source_world_transforms.ndim == 3:
                source_world_transforms = source_world_transforms.unsqueeze(0)
            if source_world_transforms.shape[:2] != source_rotations.shape[:2]:
                raise ValueError(
                    "source_world_transforms must match source_rotations batch/joint shape; "
                    f"got {source_world_transforms.shape[:2]} and {source_rotations.shape[:2]}"
                )
            if source_world_transforms.shape[-2:] != (4, 4):
                raise ValueError(
                    "source_world_transforms must have shape (B, J, 4, 4), "
                    f"got {source_world_transforms.shape}"
                )

        source_axis_ids = self.source_twist_axis_ids.to(device=source_rotations.device)
        source_axis_signs = self.source_twist_axis_signs.to(
            dtype=source_rotations.dtype, device=source_rotations.device
        )
        matrices_by_mode = self.rotation_parameter_matrices_by_mode.to(
            dtype=source_rotations.dtype, device=source_rotations.device
        )
        twist_angles = torch.zeros(
            (batch_size, len(self.twist_joint_names)),
            dtype=source_rotations.dtype,
            device=source_rotations.device,
        )
        for mode_index, mode in enumerate(self.rotation_extraction_mode_names):
            if mode == SOMA_LOCAL_X_EULER_TWIST_MODE:
                euler_channels = _local_euler_xyz_from_matrix(source_rotations)
                gather_ids = source_axis_ids.reshape(1, -1, 1).expand(
                    source_rotations.shape[0], -1, 1
                )
                twist_values = (
                    euler_channels.gather(-1, gather_ids).squeeze(-1) * source_axis_signs[None]
                )
            elif mode == SOMA_LOCAL_X_SWING_TWIST_MODE:
                twist_values = _swing_twist_channels_from_matrix(
                    source_rotations,
                    source_axis_ids,
                    source_axis_signs,
                )
            elif mode == SOMA_ALIGNED_X_SWING_TWIST_MODE:
                if source_world_transforms is None:
                    raise RuntimeError("source_world_transforms is required for aligned twist")
                twist_values = self._aligned_twist_channels_from_world(source_world_transforms)
            else:
                raise RuntimeError(f"Unsupported SOMA procedural twist mode: {mode!r}")
            twist_angles = twist_angles + torch.matmul(twist_values, matrices_by_mode[mode_index].T)
        return twist_angles

    def twist_rotations_from_source(
        self,
        source_rotations: torch.Tensor,
        source_world_transforms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate only procedural twist helper rotations from public source data."""
        twist_angles = self._twist_angles_from_source(source_rotations, source_world_transforms)
        if self._single_twist_axis is not None:
            return single_axis_rotation_matrices(
                twist_angles,
                self._single_twist_axis,
                self.twist_axis_signs,
            )
        return _axis_rotations(twist_angles, self.twist_axis_ids, self.twist_axis_signs)

    def expand_world_transforms_from_source_fk(
        self,
        source_rotations: torch.Tensor,
        source_world_transforms: torch.Tensor,
        target_local_rotations: torch.Tensor | None,
        target_local_translations: torch.Tensor,
        target_joint_count: int,
    ) -> torch.Tensor:
        """Expand public FK world transforms to the full procedural skinning rig."""
        if source_rotations.ndim != 4 or source_rotations.shape[-2:] != (3, 3):
            raise ValueError(
                f"source_rotations must have shape (B, J, 3, 3), got {source_rotations.shape}"
            )
        if source_rotations.shape[1] != len(self.source_joint_names):
            raise ValueError(
                f"Expected {len(self.source_joint_names)} source joints, "
                f"got {source_rotations.shape[1]}"
            )
        if target_joint_count != len(self.target_joint_names):
            raise ValueError(
                f"Expected target_joint_count={len(self.target_joint_names)}, "
                f"got {target_joint_count}"
            )

        if source_world_transforms.ndim == 3:
            source_world_transforms = source_world_transforms.unsqueeze(0)
        if source_world_transforms.shape[:2] != source_rotations.shape[:2]:
            raise ValueError(
                "source_world_transforms must match source_rotations batch/joint shape; "
                f"got {source_world_transforms.shape[:2]} and {source_rotations.shape[:2]}"
            )
        if source_world_transforms.shape[-2:] != (4, 4):
            raise ValueError(
                "source_world_transforms must have shape (B, J, 4, 4), "
                f"got {source_world_transforms.shape}"
            )

        batch_size = source_world_transforms.shape[0]
        if target_local_translations.ndim == 2:
            target_local_translations = target_local_translations.unsqueeze(0)
        if target_local_translations.shape[-2:] != (target_joint_count, 3):
            raise ValueError(
                "target_local_translations must have shape (B, target_joint_count, 3), "
                f"got {target_local_translations.shape}"
            )
        if target_local_translations.shape[0] == 1 and batch_size > 1:
            target_local_translations = target_local_translations.expand(batch_size, -1, -1)
        elif target_local_translations.shape[0] != batch_size:
            raise ValueError(
                "target_local_translations batch must match source_world_transforms; "
                f"got {target_local_translations.shape[0]} and {batch_size}"
            )
        target_local_translations = target_local_translations.to(
            dtype=source_world_transforms.dtype,
            device=source_world_transforms.device,
        )
        target_base_rotations = self._target_local_rotations(
            batch_size,
            target_joint_count,
            dtype=source_world_transforms.dtype,
            device=source_world_transforms.device,
            target_local_rotations=target_local_rotations,
        )

        eye4 = torch.eye(
            4,
            dtype=source_world_transforms.dtype,
            device=source_world_transforms.device,
        ).reshape(1, 1, 4, 4)
        target_world_transforms = eye4.expand(
            batch_size,
            target_joint_count,
            4,
            4,
        ).clone()
        target_world_transforms[:, self.control_target_ids] = source_world_transforms[
            :, self.control_source_ids
        ]
        assigned = torch.zeros(
            target_joint_count,
            dtype=torch.bool,
            device=source_world_transforms.device,
        )
        assigned[self.control_target_ids.to(device=source_world_transforms.device)] = True

        twist_target_ids = self.twist_target_ids.to(device=source_world_transforms.device)
        if twist_target_ids.numel() > 0 and self.twist_parent_target_ids.numel() != (
            twist_target_ids.numel()
        ):
            raise RuntimeError(
                "expand_world_transforms_from_source_fk requires target_joint_parent_ids "
                "at construction time."
            )

        if twist_target_ids.numel() > 0:
            twist_rotations = self.twist_rotations_from_source(
                source_rotations=source_rotations,
                source_world_transforms=source_world_transforms,
            ).to(dtype=source_world_transforms.dtype, device=source_world_transforms.device)
            twist_parent_ids = self.twist_parent_target_ids.to(
                device=source_world_transforms.device
            )
            twist_local_transforms = SE3_from_Rt(
                target_base_rotations[:, twist_target_ids] @ twist_rotations,
                target_local_translations[:, twist_target_ids],
            )
            target_world_transforms[:, twist_target_ids] = (
                target_world_transforms[:, twist_parent_ids] @ twist_local_transforms
            )
            assigned[twist_target_ids] = True

        if (~assigned).any():
            if self.target_joint_parent_ids.numel() != target_joint_count:
                raise RuntimeError(
                    "expand_world_transforms_from_source_fk requires target_joint_parent_ids "
                    "to fill non-public, non-procedural target joints."
                )
            target_parent_ids = self.target_joint_parent_ids.to(
                device=source_world_transforms.device
            )
            target_local_transforms = SE3_from_Rt(
                target_base_rotations,
                target_local_translations,
            )
            target_levels = compute_skeleton_levels(
                target_parent_ids,
                device=source_world_transforms.device,
            )
            for joint_ids, _parent_ids in target_levels:
                fill_ids = joint_ids[~assigned[joint_ids]]
                if fill_ids.numel() == 0:
                    continue
                parent_ids = target_parent_ids[fill_ids]
                target_world_transforms[:, fill_ids] = (
                    target_world_transforms[:, parent_ids] @ target_local_transforms[:, fill_ids]
                )
                assigned[fill_ids] = True
        return target_world_transforms

    def forward(
        self,
        source_rotations: torch.Tensor | None = None,
        source_world_transforms: torch.Tensor | None = None,
        target_world_transforms: torch.Tensor | None = None,
        target_local_rotations: torch.Tensor | None = None,
    ) -> SOMAProceduralTransformOutput:
        """Apply the compiled SOMA procedural parameter transform."""
        if source_rotations is None and target_world_transforms is None:
            raise ValueError("Provide source_rotations, target_world_transforms, or both")
        rotations = (
            self._apply_rotation_parameters(
                source_rotations,
                source_world_transforms,
                target_local_rotations=target_local_rotations,
            )
            if source_rotations is not None
            else None
        )
        transforms = (
            self._apply_translation_parameters(target_world_transforms)
            if target_world_transforms is not None
            else None
        )
        return SOMAProceduralTransformOutput(rotations=rotations, transforms=transforms)
