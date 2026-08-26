# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable RTS smoothing for SOMA pose sequences.

The rotation smoother is an error-state Rauch-Tung-Striebel smoother on SO(3).
It keeps the filter state as a unit quaternion plus angular velocity and uses
shortest-arc quaternion log residuals instead of smoothing raw quaternion
components or extracting axes from matrix logs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import torch

from soma.geometry.rig_utils import (
    apply_joint_orient_local,
    get_joint_descendents,
    precompute_joint_orient,
    remove_joint_orient_local,
)
from soma.geometry.transforms import (
    matrix_to_quaternion_xyzw,
    project_rotations_to_so3,
    quaternion_conjugate_xyzw,
    quaternion_exp_xyzw,
    quaternion_log_xyzw,
    quaternion_multiply_xyzw,
    quaternion_normalize_xyzw,
    quaternion_xyzw_to_matrix,
)

RotationConvention = Literal["absolute", "relative"]


@dataclass(frozen=True)
class RTSSmoothingGains:
    """Scalar gains for a constant-velocity RTS smoother."""

    r_obs: float = 0.06
    q_pos: float = 0.1
    q_vel: float = 4.5


@dataclass(frozen=True)
class RTSSmoothingConfig:
    """Configuration for :func:`smooth_pose`."""

    fps: float = 30.0
    rotation: RTSSmoothingGains = field(default_factory=RTSSmoothingGains)
    hand: RTSSmoothingGains | None = field(
        default_factory=lambda: RTSSmoothingGains(r_obs=0.005, q_pos=0.1, q_vel=35.0)
    )
    translation: RTSSmoothingGains | None = field(default_factory=RTSSmoothingGains)
    smooth_root_translation: bool = True


@dataclass(frozen=True)
class RTSSmoothingGroups:
    """Joint-name-derived smoothing groups."""

    hand_indices: frozenset[int] = frozenset()
    limb_indices: frozenset[int] = frozenset()

    def fast_indices(self, include_limbs: bool = False) -> frozenset[int]:
        if include_limbs:
            return self.hand_indices | self.limb_indices
        return self.hand_indices


DEFAULT_RTS_SMOOTHING_CONFIG = RTSSmoothingConfig()
STRONG_RTS_SMOOTHING_CONFIG = RTSSmoothingConfig(
    rotation=RTSSmoothingGains(r_obs=0.16, q_pos=0.05, q_vel=2.0),
    hand=RTSSmoothingGains(r_obs=0.024, q_pos=0.05, q_vel=18.0),
    translation=RTSSmoothingGains(r_obs=0.16, q_pos=0.05, q_vel=2.0),
)
RTS_SMOOTHING_PRESETS: dict[str, RTSSmoothingConfig] = {
    "default": DEFAULT_RTS_SMOOTHING_CONFIG,
    "strong": STRONG_RTS_SMOOTHING_CONFIG,
}


def _right_compose_delta(quaternions: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return quaternion_normalize_xyzw(
        quaternion_multiply_xyzw(quaternions, quaternion_exp_xyzw(delta))
    )


def _quaternion_residual(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    relative = quaternion_multiply_xyzw(quaternion_conjugate_xyzw(predicted), observed)
    return quaternion_log_xyzw(relative)


def _validate_fps(fps: float) -> float:
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    return fps


def _gain_index_groups(
    num_joints: int,
    default_gains: RTSSmoothingGains,
    joint_gains: Mapping[int, RTSSmoothingGains] | None,
    *,
    device: torch.device,
) -> list[tuple[torch.Tensor, RTSSmoothingGains]]:
    grouped: dict[RTSSmoothingGains, list[int]] = {}
    joint_gains = joint_gains or {}
    for joint_idx in range(num_joints):
        gains = joint_gains.get(joint_idx, default_gains)
        grouped.setdefault(gains, []).append(joint_idx)
    return [
        (torch.tensor(indices, dtype=torch.long, device=device), gains)
        for gains, indices in grouped.items()
        if indices
    ]


def _rts_smooth_channels(
    measurements: torch.Tensor,
    *,
    fps: float,
    gains: RTSSmoothingGains,
) -> torch.Tensor:
    """Smooth ``(T, C)`` independent scalar channels with one RTS recursion."""
    num_frames, num_channels = measurements.shape
    if num_frames < 2 or num_channels == 0:
        return measurements.clone()

    dt = 1.0 / _validate_fps(fps)
    device = measurements.device
    f_mat = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64, device=device)
    q_mat = torch.diag(
        torch.tensor(
            [gains.q_pos * dt, gains.q_vel * dt],
            dtype=torch.float64,
            device=device,
        )
    )
    r_obs = torch.tensor(float(gains.r_obs), dtype=torch.float64, device=device)

    state_fwd = torch.zeros(num_frames, num_channels, 2, dtype=torch.float64, device=device)
    cov_fwd = torch.zeros(num_frames, num_channels, 2, 2, dtype=torch.float64, device=device)
    state_fwd[0, :, 0] = measurements[0]
    cov_fwd[0, :, 0, 0] = r_obs
    cov_fwd[0, :, 1, 1] = max(float(gains.q_vel) * dt, 1e-2)

    for frame_idx in range(1, num_frames):
        state_pred = state_fwd[frame_idx - 1] @ f_mat.T
        cov_pred = f_mat @ cov_fwd[frame_idx - 1] @ f_mat.T + q_mat
        innovation_cov = cov_pred[:, 0, 0] + r_obs
        kalman_gain = cov_pred[:, :, 0] / innovation_cov[:, None]
        residual = measurements[frame_idx] - state_pred[:, 0]
        state_fwd[frame_idx] = state_pred + kalman_gain * residual[:, None]
        cov_fwd[frame_idx] = cov_pred - kalman_gain[:, :, None] * cov_pred[:, 0:1, :]
        cov_fwd[frame_idx] = 0.5 * (cov_fwd[frame_idx] + cov_fwd[frame_idx].transpose(-2, -1))

    state_smooth = torch.zeros_like(state_fwd)
    state_smooth[-1] = state_fwd[-1]
    for frame_idx in range(num_frames - 2, -1, -1):
        next_pred_cov = f_mat @ cov_fwd[frame_idx] @ f_mat.T + q_mat
        gain = cov_fwd[frame_idx] @ f_mat.T @ torch.linalg.inv(next_pred_cov)
        next_pred_state = state_fwd[frame_idx] @ f_mat.T
        residual = state_smooth[frame_idx + 1] - next_pred_state
        state_smooth[frame_idx] = state_fwd[frame_idx] + (gain @ residual[..., None]).squeeze(-1)

    return state_smooth[:, :, 0]


def rts_smooth_euclidean(
    values: torch.Tensor,
    *,
    fps: float,
    gains: RTSSmoothingGains,
    joint_gains: Mapping[int, RTSSmoothingGains] | None = None,
) -> torch.Tensor:
    """Smooth ``(T, J, D)`` Euclidean values with a constant-velocity RTS model."""
    if values.ndim != 3:
        raise ValueError(f"Expected values with shape (T, J, D), got {values.shape}.")
    if not torch.is_floating_point(values):
        raise TypeError("values must be a floating-point tensor.")

    fps = _validate_fps(fps)
    num_frames, num_joints, dims = values.shape
    if num_frames < 2:
        return values.clone()

    values64 = values.to(dtype=torch.float64)
    result = torch.empty_like(values64)
    for indices, group_gains in _gain_index_groups(
        num_joints,
        gains,
        joint_gains,
        device=values.device,
    ):
        grouped_values = values64[:, indices].reshape(num_frames, -1)
        smoothed = _rts_smooth_channels(grouped_values, fps=fps, gains=group_gains)
        result[:, indices] = smoothed.reshape(num_frames, indices.numel(), dims)
    return result.to(dtype=values.dtype)


def _smooth_rotation_group(
    quaternions: torch.Tensor,
    *,
    fps: float,
    gains: RTSSmoothingGains,
) -> torch.Tensor:
    num_frames, num_channels = quaternions.shape[:2]
    if num_frames < 2 or num_channels == 0:
        return quaternions.clone()

    dt = 1.0 / _validate_fps(fps)
    device = quaternions.device
    i3 = torch.eye(3, dtype=torch.float64, device=device)
    i6 = torch.eye(6, dtype=torch.float64, device=device)
    f_mat = torch.eye(6, dtype=torch.float64, device=device)
    f_mat[:3, 3:] = dt * i3
    h_mat = torch.zeros(3, 6, dtype=torch.float64, device=device)
    h_mat[:, :3] = i3
    q_diag = torch.tensor(
        [gains.q_pos * dt] * 3 + [gains.q_vel * dt] * 3,
        dtype=torch.float64,
        device=device,
    )
    q_mat = torch.diag(q_diag)
    r_mat = i3 * gains.r_obs

    quat_fwd = torch.empty_like(quaternions)
    vel_fwd = torch.zeros(num_frames, num_channels, 3, dtype=torch.float64, device=device)
    cov_fwd = torch.zeros(num_frames, num_channels, 6, 6, dtype=torch.float64, device=device)
    quat_pred = torch.empty_like(quaternions)
    vel_pred = torch.zeros_like(vel_fwd)
    cov_pred = torch.zeros_like(cov_fwd)

    quat_fwd[0] = quaternions[0]
    quat_pred[0] = quaternions[0]
    cov_fwd[0, :, :3, :3] = i3 * gains.r_obs
    cov_fwd[0, :, 3:, 3:] = i3 * max(float(gains.q_vel) * dt, 1e-2)

    for frame_idx in range(1, num_frames):
        quat_pred[frame_idx] = _right_compose_delta(
            quat_fwd[frame_idx - 1],
            vel_fwd[frame_idx - 1] * dt,
        )
        vel_pred[frame_idx] = vel_fwd[frame_idx - 1]
        cov_pred[frame_idx] = f_mat @ cov_fwd[frame_idx - 1] @ f_mat.T + q_mat

        residual = _quaternion_residual(quat_pred[frame_idx], quaternions[frame_idx])
        innovation_cov = h_mat @ cov_pred[frame_idx] @ h_mat.T + r_mat
        kalman_gain = cov_pred[frame_idx] @ h_mat.T @ torch.linalg.inv(innovation_cov)
        delta = (kalman_gain @ residual[..., None]).squeeze(-1)

        quat_fwd[frame_idx] = _right_compose_delta(quat_pred[frame_idx], delta[:, :3])
        vel_fwd[frame_idx] = vel_pred[frame_idx] + delta[:, 3:]
        cov_fwd[frame_idx] = (i6 - kalman_gain @ h_mat) @ cov_pred[frame_idx]
        cov_fwd[frame_idx] = 0.5 * (cov_fwd[frame_idx] + cov_fwd[frame_idx].transpose(-2, -1))

    quat_smooth = torch.empty_like(quaternions)
    vel_smooth = torch.zeros_like(vel_fwd)
    quat_smooth[-1] = quat_fwd[-1]
    vel_smooth[-1] = vel_fwd[-1]

    for frame_idx in range(num_frames - 2, -1, -1):
        next_pred_cov = f_mat @ cov_fwd[frame_idx] @ f_mat.T + q_mat
        gain = cov_fwd[frame_idx] @ f_mat.T @ torch.linalg.inv(next_pred_cov)
        rot_residual = _quaternion_residual(quat_pred[frame_idx + 1], quat_smooth[frame_idx + 1])
        vel_residual = vel_smooth[frame_idx + 1] - vel_pred[frame_idx + 1]
        residual = torch.cat([rot_residual, vel_residual], dim=-1)
        delta = (gain @ residual[..., None]).squeeze(-1)
        quat_smooth[frame_idx] = _right_compose_delta(quat_fwd[frame_idx], delta[:, :3])
        vel_smooth[frame_idx] = vel_fwd[frame_idx] + delta[:, 3:]

    return quaternion_normalize_xyzw(quat_smooth)


def rts_smooth_rotations(
    rotations: torch.Tensor,
    *,
    fps: float,
    gains: RTSSmoothingGains,
    joint_gains: Mapping[int, RTSSmoothingGains] | None = None,
) -> torch.Tensor:
    """Smooth ``(T, J, 3, 3)`` rotations with an SO(3) error-state RTS model."""
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape (T, J, 3, 3), got {rotations.shape}.")
    if not torch.is_floating_point(rotations):
        raise TypeError("rotations must be a floating-point tensor.")

    fps = _validate_fps(fps)
    num_frames, num_joints = rotations.shape[:2]
    if num_frames < 2:
        return rotations.clone()

    rotations64 = project_rotations_to_so3(rotations.to(dtype=torch.float64))
    quaternions = matrix_to_quaternion_xyzw(rotations64)
    result = torch.empty_like(quaternions)
    for indices, group_gains in _gain_index_groups(
        num_joints,
        gains,
        joint_gains,
        device=rotations.device,
    ):
        result[:, indices] = _smooth_rotation_group(
            quaternions[:, indices],
            fps=fps,
            gains=group_gains,
        )
    return quaternion_xyzw_to_matrix(result).to(dtype=rotations.dtype)


def _joint_names_from_layer(soma_layer) -> list[str] | None:
    if soma_layer is None:
        return None
    if hasattr(soma_layer, "public_joint_names"):
        return list(soma_layer.public_joint_names)
    if hasattr(soma_layer, "rig_data") and "joint_names" in soma_layer.rig_data:
        return [str(name) for name in soma_layer.rig_data["joint_names"]]
    return None


def _parent_ids_from_layer(soma_layer) -> Sequence[int] | torch.Tensor | None:
    if soma_layer is None:
        return None
    if hasattr(soma_layer, "output_joint_parent_ids"):
        return soma_layer.output_joint_parent_ids
    if hasattr(soma_layer, "public_joint_parent_ids"):
        return soma_layer.public_joint_parent_ids
    if hasattr(soma_layer, "joint_parent_ids"):
        return soma_layer.joint_parent_ids
    if hasattr(soma_layer, "rig_data") and "joint_parent_ids" in soma_layer.rig_data:
        return soma_layer.rig_data["joint_parent_ids"]
    return None


def _matching_names_and_parents(
    num_joints: int,
    joint_names: Sequence[str] | None,
    joint_parent_ids: Sequence[int] | torch.Tensor | None,
) -> tuple[list[str] | None, Sequence[int] | torch.Tensor | None]:
    if joint_names is None:
        return None, None

    names = [str(name) for name in joint_names]
    if len(names) == num_joints:
        parents = (
            joint_parent_ids
            if joint_parent_ids is not None and len(joint_parent_ids) == num_joints
            else None
        )
        return names, parents
    if names and names[0] == "Root" and len(names) - 1 == num_joints:
        return names[1:], None
    raise ValueError(f"Got {num_joints} rotations but {len(names)} joint names.")


def _name_matches_any(name: str, tokens: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(token.lower() in lowered for token in tokens)


def derive_smoothing_groups(
    joint_names: Sequence[str],
    joint_parent_ids: Sequence[int] | torch.Tensor | None = None,
) -> RTSSmoothingGroups:
    """Derive hand and limb smoothing groups from joint names and topology."""
    names = [str(name) for name in joint_names]
    parents = None
    if joint_parent_ids is not None:
        parents = (
            joint_parent_ids.tolist()
            if hasattr(joint_parent_ids, "tolist")
            else list(joint_parent_ids)
        )

    hand_indices: set[int] = set()
    if parents is not None:
        for idx, name in enumerate(names):
            lowered = name.lower()
            if lowered.endswith("hand") or lowered.endswith("wrist"):
                hand_indices.add(idx)
                hand_indices.update(get_joint_descendents(parents, idx))

    hand_tokens = ("hand", "thumb", "index", "middle", "ring", "pinky")
    for idx, name in enumerate(names):
        if _name_matches_any(name, hand_tokens):
            hand_indices.add(idx)

    limb_tokens = ("shoulder", "arm", "forearm", "leg", "shin", "foot", "toe")
    limb_indices = {
        idx
        for idx, name in enumerate(names)
        if idx not in hand_indices and _name_matches_any(name, limb_tokens)
    }
    return RTSSmoothingGroups(
        hand_indices=frozenset(hand_indices),
        limb_indices=frozenset(limb_indices),
    )


def _joint_gain_map(
    groups: RTSSmoothingGroups,
    config: RTSSmoothingConfig,
    *,
    use_hand_gains: bool,
    include_limb_gains: bool,
) -> dict[int, RTSSmoothingGains]:
    if not use_hand_gains or config.hand is None:
        return {}
    return {idx: config.hand for idx in groups.fast_indices(include_limb_gains)}


def _public_orient_from_layer(
    soma_layer,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not all(
        hasattr(soma_layer, attr)
        for attr in ("public_transform_joint_indices", "public_joint_parent_ids", "t_pose_world")
    ):
        return None
    public_indices = soma_layer.public_transform_joint_indices.to(
        device=soma_layer.t_pose_world.device
    )
    t_pose_world = soma_layer.t_pose_world[public_indices].to(dtype=dtype, device=device)
    parent_ids = soma_layer.public_joint_parent_ids.to(device=device)
    return precompute_joint_orient(t_pose_world, parent_ids)


def _resolve_joint_orient(
    *,
    expected_joints: int,
    dtype: torch.dtype,
    device: torch.device,
    soma_layer=None,
    t_pose_orient: torch.Tensor | None = None,
    t_pose_orient_parent_T: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if t_pose_orient is not None or t_pose_orient_parent_T is not None:
        if t_pose_orient is None or t_pose_orient_parent_T is None:
            raise ValueError("Pass both t_pose_orient and t_pose_orient_parent_T, or neither.")
        orient = t_pose_orient.to(dtype=dtype, device=device)
        orient_parent_t = t_pose_orient_parent_T.to(dtype=dtype, device=device)
    elif soma_layer is not None:
        public_orient = _public_orient_from_layer(soma_layer, dtype=dtype, device=device)
        if public_orient is not None:
            orient, orient_parent_t = public_orient
        else:
            orient = getattr(soma_layer, "_t_pose_orient", None)
            orient_parent_t = getattr(soma_layer, "_t_pose_orient_parent_T", None)
            if orient is None or orient_parent_t is None:
                raise ValueError("SOMA convention conversion requires joint orient tensors.")
            orient = orient.to(dtype=dtype, device=device)
            orient_parent_t = orient_parent_t.to(dtype=dtype, device=device)
    else:
        raise ValueError(
            "rotation_convention='absolute' requires soma_layer or explicit joint orient tensors."
        )

    if orient.shape[0] != expected_joints or orient_parent_t.shape[0] != expected_joints:
        raise ValueError(
            "Joint orient tensors must match the rotation joint count. "
            f"Got orient={orient.shape[0]}, parent={orient_parent_t.shape[0]}, "
            f"rotations={expected_joints}."
        )
    return orient, orient_parent_t


def _config_from_preset(
    preset: str,
    config: RTSSmoothingConfig | None,
    fps: float | None,
) -> RTSSmoothingConfig:
    if config is None:
        if preset not in RTS_SMOOTHING_PRESETS:
            raise ValueError(f"Unknown RTS smoothing preset: {preset!r}.")
        config = RTS_SMOOTHING_PRESETS[preset]
    if fps is not None:
        config = replace(config, fps=float(fps))
    _validate_fps(config.fps)
    return config


def smooth_pose(
    rotations: torch.Tensor,
    root_translation: torch.Tensor | None = None,
    *,
    soma_layer=None,
    t_pose_orient: torch.Tensor | None = None,
    t_pose_orient_parent_T: torch.Tensor | None = None,
    joint_names: Sequence[str] | None = None,
    joint_parent_ids: Sequence[int] | torch.Tensor | None = None,
    fps: float | None = None,
    preset: str = "default",
    config: RTSSmoothingConfig | None = None,
    rotation_convention: RotationConvention = "absolute",
    output_rotation_convention: RotationConvention | None = None,
    use_hand_gains: bool = True,
    include_limb_gains: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Smooth SOMA rotations and root translation.

    Args:
        rotations: ``(T, J, 3, 3)`` local rotation matrices.
        root_translation: Optional ``(T, 3)`` root translation.
        soma_layer: Optional SOMA layer used for public joint names, parents,
            and T-pose joint orient conversion.
        rotation_convention: ``"absolute"`` for PoseInversion-style rotations
            with joint orient baked in, or ``"relative"`` for T-pose-relative
            rotations. The default output convention matches the input.
    """
    if rotation_convention not in ("absolute", "relative"):
        raise ValueError(f"Unsupported rotation_convention: {rotation_convention!r}.")
    if output_rotation_convention is None:
        output_rotation_convention = rotation_convention
    if output_rotation_convention not in ("absolute", "relative"):
        raise ValueError(f"Unsupported output_rotation_convention: {output_rotation_convention!r}.")
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape (T, J, 3, 3), got {rotations.shape}.")
    if root_translation is not None and (
        root_translation.ndim != 2 or root_translation.shape != (rotations.shape[0], 3)
    ):
        raise ValueError(
            "root_translation must have shape (T, 3) matching rotations; "
            f"got {root_translation.shape} for rotations {rotations.shape}."
        )

    config = _config_from_preset(preset, config, fps)
    joint_names = joint_names or _joint_names_from_layer(soma_layer)
    joint_parent_ids = (
        joint_parent_ids if joint_parent_ids is not None else _parent_ids_from_layer(soma_layer)
    )

    if rotation_convention == "absolute":
        orient, orient_parent_t = _resolve_joint_orient(
            expected_joints=rotations.shape[1],
            dtype=rotations.dtype,
            device=rotations.device,
            soma_layer=soma_layer,
            t_pose_orient=t_pose_orient,
            t_pose_orient_parent_T=t_pose_orient_parent_T,
        )
        working_rotations = remove_joint_orient_local(rotations, orient, orient_parent_t)
    else:
        orient = None
        orient_parent_t = None
        working_rotations = rotations

    group_names, group_parents = _matching_names_and_parents(
        working_rotations.shape[1],
        joint_names,
        joint_parent_ids,
    )
    groups = (
        derive_smoothing_groups(group_names, group_parents)
        if group_names is not None
        else RTSSmoothingGroups()
    )

    smoothed_relative = rts_smooth_rotations(
        working_rotations,
        fps=config.fps,
        gains=config.rotation,
        joint_gains=_joint_gain_map(
            groups,
            config,
            use_hand_gains=use_hand_gains,
            include_limb_gains=include_limb_gains,
        ),
    )

    smoothed_root = root_translation
    if (
        root_translation is not None
        and config.smooth_root_translation
        and config.translation is not None
    ):
        smoothed_root = rts_smooth_euclidean(
            root_translation[:, None, :],
            fps=config.fps,
            gains=config.translation,
        )[:, 0]

    if output_rotation_convention == "relative":
        return smoothed_relative, smoothed_root

    if orient is None or orient_parent_t is None:
        orient, orient_parent_t = _resolve_joint_orient(
            expected_joints=smoothed_relative.shape[1],
            dtype=smoothed_relative.dtype,
            device=smoothed_relative.device,
            soma_layer=soma_layer,
            t_pose_orient=t_pose_orient,
            t_pose_orient_parent_T=t_pose_orient_parent_T,
        )
    return apply_joint_orient_local(smoothed_relative, orient, orient_parent_t), smoothed_root


def so3_angular_velocity(rotations: torch.Tensor) -> torch.Tensor:
    """Return per-frame SO(3) step vectors with shape ``(T - 1, J, 3)``."""
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotations with shape (T, J, 3, 3), got {rotations.shape}.")
    if rotations.shape[0] < 2:
        return rotations.new_zeros((0, rotations.shape[1], 3))
    quaternions = matrix_to_quaternion_xyzw(project_rotations_to_so3(rotations))
    relative = quaternion_multiply_xyzw(
        quaternion_conjugate_xyzw(quaternions[:-1]),
        quaternions[1:],
    )
    return quaternion_log_xyzw(relative.reshape(-1, 4)).reshape(rotations.shape[0] - 1, -1, 3)


def so3_angular_acceleration(rotations: torch.Tensor) -> torch.Tensor:
    """Return finite-difference angular acceleration vectors."""
    velocity = so3_angular_velocity(rotations)
    if velocity.shape[0] < 2:
        return velocity.new_zeros((0, velocity.shape[1], 3))
    return velocity[1:] - velocity[:-1]


def euclidean_acceleration(values: torch.Tensor) -> torch.Tensor:
    """Return second differences for ``(T, D)`` or ``(T, J, D)`` values."""
    if values.shape[0] < 3:
        return values.new_zeros((0, *values.shape[1:]))
    return values[2:] - 2.0 * values[1:-1] + values[:-2]


__all__ = [
    "DEFAULT_RTS_SMOOTHING_CONFIG",
    "RTSSmoothingConfig",
    "RTSSmoothingGains",
    "RTSSmoothingGroups",
    "RTS_SMOOTHING_PRESETS",
    "STRONG_RTS_SMOOTHING_CONFIG",
    "derive_smoothing_groups",
    "euclidean_acceleration",
    "rts_smooth_euclidean",
    "rts_smooth_rotations",
    "smooth_pose",
    "so3_angular_acceleration",
    "so3_angular_velocity",
]
