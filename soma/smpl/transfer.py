# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SMPL-family pose transfer helpers."""

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from ..fitting.pose_inversion import PoseInversion
from ..geometry.barycentric_interp import BarycentricInterpolator
from ..units import Unit


@dataclass
class SMPLFamilyPoseTransferResult:
    """Result of fitting target pose parameters to a source rig animation."""

    rotations: torch.Tensor
    root_translation: torch.Tensor
    per_vertex_error: torch.Tensor
    source_vertices: torch.Tensor
    fit_vertices: torch.Tensor
    reconstructed_vertices: torch.Tensor


def _layer_unit(layer: Any) -> Unit:
    unit = getattr(layer, "output_unit", Unit.METERS)
    if isinstance(unit, Unit):
        return unit
    return Unit.from_name(unit)


def _unit_scale(source_layer: Any, target_layer: Any) -> float:
    source_unit = _layer_unit(source_layer)
    target_unit = _layer_unit(target_layer)
    return source_unit.meters_per_unit / target_unit.meters_per_unit


def _load_mesh(path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = trimesh.load(Path(path), maintain_order=True, process=False)
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
    return vertices, faces


def _num_identity_coeffs_for_layer(layer: Any) -> int:
    value = getattr(layer, "num_identity_coeffs", None)
    if value is not None:
        return int(value)
    identity_model = getattr(layer, "identity_model", None)
    if identity_model is not None:
        value = getattr(identity_model, "num_identity_coeffs", None)
        if value is not None:
            return int(value)
    value = getattr(layer, "num_shape_components", None)
    if value is not None:
        return int(value)
    return 0


def _layer_identity_coeffs(
    layer: Any,
    values: torch.Tensor | None,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    num_coeffs = _num_identity_coeffs_for_layer(layer)
    dtype = getattr(layer, "bind_shape", torch.empty((), dtype=torch.float32)).dtype
    device = getattr(layer, "device", torch.device("cpu"))
    if values is None:
        rows = 1 if batch_size is None else batch_size
        return torch.zeros(rows, num_coeffs, dtype=dtype, device=device)

    coeffs = values.to(dtype=dtype, device=device)
    if coeffs.ndim == 1:
        coeffs = coeffs.unsqueeze(0)
    if coeffs.ndim != 2:
        raise ValueError(f"Expected identity coefficients with shape (B, C), got {coeffs.shape}.")
    if batch_size is not None and coeffs.shape[0] == 1 and batch_size > 1:
        coeffs = coeffs.expand(batch_size, -1)

    if coeffs.shape[1] > num_coeffs:
        return coeffs[:, :num_coeffs]
    if coeffs.shape[1] < num_coeffs:
        pad = torch.zeros(
            coeffs.shape[0],
            num_coeffs - coeffs.shape[1],
            dtype=coeffs.dtype,
            device=coeffs.device,
        )
        coeffs = torch.cat([coeffs, pad], dim=1)
    return coeffs


def _adapt_identity_coeffs(values: torch.Tensor, target_layer: Any) -> torch.Tensor:
    return _layer_identity_coeffs(target_layer, values)


def _pose_batch_size(poses: torch.Tensor, pose2rot: bool) -> int:
    if pose2rot:
        if poses.ndim == 2:
            return 1
        return int(poses.shape[0])
    if poses.ndim == 3 and poses.shape[-2:] == (3, 3):
        return 1
    return int(poses.shape[0])


def _with_pose_batch(poses: torch.Tensor, pose2rot: bool) -> torch.Tensor:
    if pose2rot:
        if poses.ndim == 2:
            return poses.unsqueeze(0)
        return poses
    if poses.ndim == 3 and poses.shape[-2:] == (3, 3):
        return poses.unsqueeze(0)
    return poses


class SMPLFamilyTopologyBridge(torch.nn.Module):
    """Map posed vertices between SOMA and native SMPL-family topologies."""

    def __init__(self, source_layer: Any, target_layer: Any) -> None:
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.scale = _unit_scale(source_layer, target_layer)
        self.direct = self._can_use_direct_topology(source_layer, target_layer)

        if self.direct:
            self.source_to_canonical = None
            self.canonical_to_target = None
            return

        source_family = getattr(source_layer, "topology_family", None)
        target_family = getattr(target_layer, "topology_family", None)
        target_base = getattr(target_layer, "base_mesh_path", None)
        target_wrap = getattr(target_layer, "wrap_mesh_path", None)

        if (
            source_family is None
            and target_family in {"body", "hand"}
            and target_base
            and target_wrap
        ):
            device = getattr(source_layer, "device", torch.device("cpu"))
            target_base_v, _ = _load_mesh(target_base, device)
            target_wrap_v, target_wrap_f = _load_mesh(target_wrap, device)
            source_num_verts = int(source_layer.bind_shape.shape[0])
            if source_num_verts != target_wrap_v.shape[0]:
                raise ValueError(
                    "SOMA-to-SMPL-family topology bridge requires matching SOMA wrap topology. "
                    f"Got {source_num_verts} source vertices, expected {target_wrap_v.shape[0]}."
                )
            self.source_to_canonical = None
            self.canonical_to_target = BarycentricInterpolator(
                target_wrap_v,
                target_wrap_f,
                target_base_v,
            )
            return

        if source_family != target_family:
            raise ValueError(
                f"No registered SMPL-family topology bridge from {source_family!r} "
                f"to {target_family!r}."
            )

        source_base = getattr(source_layer, "base_mesh_path", None)
        source_wrap = getattr(source_layer, "wrap_mesh_path", None)
        if not all((source_base, source_wrap, target_base, target_wrap)):
            raise ValueError(
                "Both SMPL-family layers must define base_mesh_path and wrap_mesh_path."
            )

        device = getattr(source_layer, "device", torch.device("cpu"))
        source_base_v, source_base_f = _load_mesh(source_base, device)
        source_wrap_v, _ = _load_mesh(source_wrap, device)
        target_base_v, _ = _load_mesh(target_base, device)
        target_wrap_v, target_wrap_f = _load_mesh(target_wrap, device)

        self.source_to_canonical = BarycentricInterpolator(
            source_base_v,
            source_base_f,
            source_wrap_v,
        )
        self.canonical_to_target = BarycentricInterpolator(
            target_wrap_v,
            target_wrap_f,
            target_base_v,
        )

    @staticmethod
    def _can_use_direct_topology(source_layer: Any, target_layer: Any) -> bool:
        source_spec = getattr(source_layer, "model_spec", None)
        target_spec = getattr(target_layer, "model_spec", None)
        if source_spec is not None and source_spec == target_spec:
            return True
        return False

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.direct:
            return vertices * self.scale
        if self.source_to_canonical is None:
            canonical_vertices = vertices
        else:
            canonical_vertices = self.source_to_canonical(vertices)
        target_vertices = self.canonical_to_target(canonical_vertices)
        return target_vertices * self.scale


def _prepare_layer_identity(
    layer: Any,
    identity_coeffs: torch.Tensor,
    prepare_kwargs: dict[str, Any] | None,
) -> None:
    kwargs = dict(prepare_kwargs or {})
    signature = inspect.signature(layer.prepare_identity).parameters
    accepted = {key: value for key, value in kwargs.items() if key in signature}
    layer.prepare_identity(identity_coeffs, **accepted)


def _pose_layer(
    layer: Any,
    poses: torch.Tensor,
    root_translation: torch.Tensor,
    *,
    pose2rot: bool,
    absolute_pose: bool,
    extra_kwargs: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    pose_params = inspect.signature(layer.pose).parameters
    kwargs: dict[str, Any] = {
        "pose2rot": pose2rot,
        "absolute_pose": absolute_pose,
    }
    if "global_translation" in pose_params:
        kwargs["global_translation"] = root_translation
    elif "transl" in pose_params:
        kwargs["transl"] = root_translation
    else:
        raise TypeError(
            f"{type(layer).__name__}.pose() must accept either global_translation or transl."
        )
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)
    return layer.pose(poses, **kwargs)


def transfer_smpl_family_pose_parameters(
    source_layer: Any,
    target_layer: Any,
    source_poses: torch.Tensor,
    *,
    source_identity_coeffs: torch.Tensor | None = None,
    target_identity_coeffs: torch.Tensor | None = None,
    source_root_translation: torch.Tensor | None = None,
    source_pose2rot: bool = False,
    source_absolute_pose: bool = True,
    source_prepare_kwargs: dict[str, Any] | None = None,
    target_prepare_kwargs: dict[str, Any] | None = None,
    source_pose_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    topology_bridge: SMPLFamilyTopologyBridge | None = None,
) -> SMPLFamilyPoseTransferResult:
    """Transfer source pose parameters into a target SMPL-family rig layer.

    The source rig is evaluated first, its posed mesh is bridged to the target
    topology if needed, and :class:`PoseInversion` then recovers the target
    layer's absolute local rotations and root translation.
    """
    source_poses = _with_pose_batch(source_poses, source_pose2rot)
    batch_size = _pose_batch_size(source_poses, source_pose2rot)

    source_identity = _layer_identity_coeffs(
        source_layer,
        source_identity_coeffs,
        batch_size=None,
    )
    if target_identity_coeffs is None:
        target_identity = _adapt_identity_coeffs(source_identity, target_layer)
    else:
        target_identity = _layer_identity_coeffs(target_layer, target_identity_coeffs)

    _prepare_layer_identity(source_layer, source_identity, source_prepare_kwargs)
    _prepare_layer_identity(target_layer, target_identity, target_prepare_kwargs)

    device = getattr(source_layer, "device", torch.device("cpu"))
    dtype = source_layer.bind_shape.dtype
    if source_root_translation is None:
        source_root_translation = torch.zeros(batch_size, 3, dtype=dtype, device=device)
    else:
        source_root_translation = source_root_translation.to(dtype=dtype, device=device)
        if source_root_translation.ndim == 1:
            source_root_translation = source_root_translation.unsqueeze(0)
        if source_root_translation.shape[0] == 1 and batch_size > 1:
            source_root_translation = source_root_translation.expand(batch_size, -1)

    with torch.no_grad():
        source_out = _pose_layer(
            source_layer,
            source_poses,
            pose2rot=source_pose2rot,
            absolute_pose=source_absolute_pose,
            root_translation=source_root_translation,
            extra_kwargs=source_pose_kwargs,
        )
        source_vertices = source_out["vertices"]

        if topology_bridge is None:
            topology_bridge = SMPLFamilyTopologyBridge(source_layer, target_layer)
        fit_vertices = topology_bridge(source_vertices)

    inv = PoseInversion(target_layer, low_lod=False)
    with torch.no_grad():
        inv.prepare_identity(target_identity)

    fit_args = {}
    if fit_kwargs is not None:
        fit_args.update(fit_kwargs)
    result = inv.fit(fit_vertices, **fit_args)

    with torch.no_grad():
        recon = target_layer.pose(
            result["rotations"],
            pose2rot=False,
            apply_correctives=False,
            absolute_pose=True,
            global_translation=result["root_translation"],
        )["vertices"]

    return SMPLFamilyPoseTransferResult(
        rotations=result["rotations"],
        root_translation=result["root_translation"],
        per_vertex_error=result["per_vertex_error"],
        source_vertices=source_vertices,
        fit_vertices=fit_vertices,
        reconstructed_vertices=recon,
    )
