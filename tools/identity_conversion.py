# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared optimization and SOMA NPZ helpers for identity backend conversion."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soma.io import SOMANPZData, load_soma_npz, save_soma_npz

logger = logging.getLogger(__name__)

_STANDARD_NPZ_KEYS = {
    "poses",
    "transl",
    "joint_names",
    "identity_model_type",
    "identity_coeffs",
    "rotation_repr",
    "absolute_pose",
    "unit",
    "keep_root",
    "scale_params",
    "joint_orient",
    "global_scale",
    "hand_type",
    # NumPy versions before 2.5 serialize save_soma_npz's allow_pickle=False
    # argument as an array instead of consuming it as a writer option.
    "allow_pickle",
}


@dataclass
class IdentityConversionResult:
    """Optimized target parameters and bind-pose reconstruction diagnostics."""

    identity_coeffs: np.ndarray
    scale_params: np.ndarray | None
    global_scale: float
    vertex_error: np.ndarray
    loss_history: list[float]


def _scale_param_count(layer: Any) -> int | None:
    if layer.identity_model_type == "soma" and hasattr(layer, "hand_type"):
        return len(layer.joint_parent_ids) - 1
    value = getattr(layer, "num_scale_params", None)
    if value is not None:
        return int(value)
    value = getattr(layer.identity_model, "num_scale_params", None)
    return None if value is None else int(value)


def _identity_param_count(layer: Any) -> int:
    return int(layer.identity_model.num_identity_coeffs)


def _layer_device(layer: Any) -> torch.device:
    if hasattr(layer, "device"):
        return torch.device(layer.device)
    return layer.bind_pose_world.device


def _layer_dtype(layer: Any) -> torch.dtype:
    if hasattr(layer, "dtype"):
        return layer.dtype
    return layer.bind_pose_world.dtype


def neutral_scale_params(layer: Any, batch_size: int) -> torch.Tensor | None:
    """Return backend-neutral scale parameters on the layer device."""
    count = _scale_param_count(layer)
    if count is None:
        return None
    fill_value = 1.0 if layer.identity_model_type == "soma" else 0.0
    return torch.full(
        (batch_size, count),
        fill_value,
        dtype=_layer_dtype(layer),
        device=_layer_device(layer),
    )


def _as_parameter_rows(
    values: np.ndarray | None,
    *,
    rows: int,
    width: int | None,
    name: str,
    device: torch.device,
    default: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if width is None:
        if values is not None:
            raise ValueError(f"{name} were provided, but this backend does not use them")
        return None
    if values is None:
        if default is None:
            raise ValueError(f"Missing required {name}")
        return default
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != width:
        raise ValueError(f"Expected {name} with shape (N, {width}); got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1 and rows > 1:
        tensor = tensor.expand(rows, -1)
    elif tensor.shape[0] != rows:
        raise ValueError(f"Expected {name} to have 1 or {rows} rows; got {tensor.shape[0]}")
    return tensor


def _bind_pose_rotations(layer: Any, batch_size: int) -> torch.Tensor:
    local = layer.bind_pose_local
    if hasattr(layer, "hand_type"):
        rotations = local[..., :3, :3]
    else:
        public_indices = layer.public_transform_joint_indices.to(local.device)
        rotations = local[public_indices][1:, :3, :3]
    return rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)


def bind_pose_vertices(
    layer: Any,
    identity_coeffs: torch.Tensor,
    scale_params: torch.Tensor | None,
    global_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate an identity as SOMA-topology vertices in the SOMA bind pose."""
    layer.prepare_identity(
        identity_coeffs,
        scale_params=scale_params,
        repose_to_bind_pose=True,
        global_scale=global_scale,
    )
    if layer.identity_model_type != "soma" or scale_params is None:
        return layer._cached_rest_shape

    rotations = _bind_pose_rotations(layer, identity_coeffs.shape[0])
    scaled_vertices = layer.pose(
        rotations,
        pose2rot=False,
        apply_correctives=False,
        absolute_pose=True,
    )["vertices"]
    layer._cached_scale_params = neutral_scale_params(layer, identity_coeffs.shape[0])
    neutral_vertices = layer.pose(
        rotations,
        pose2rot=False,
        apply_correctives=False,
        absolute_pose=True,
    )["vertices"]
    layer._cached_scale_params = scale_params
    return layer._cached_rest_shape + scaled_vertices - neutral_vertices


def _center_vertices(vertices: torch.Tensor) -> torch.Tensor:
    return vertices - vertices.mean(dim=1, keepdim=True)


def convert_identity_parameters(
    source_layer: Any,
    target_layer: Any,
    source_identity_coeffs: np.ndarray,
    *,
    source_scale_params: np.ndarray | None = None,
    global_scale: float = 1.0,
    optimize_scale_params: bool = True,
    optimize_global_scale: bool = False,
    iterations: int = 200,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
) -> IdentityConversionResult:
    """Fit target-backend parameters to source geometry in the SOMA bind pose."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if global_scale <= 0:
        raise ValueError("global_scale must be greater than zero")
    source_coeffs = torch.as_tensor(
        source_identity_coeffs,
        dtype=torch.float32,
        device=_layer_device(source_layer),
    )
    if source_coeffs.ndim == 1:
        source_coeffs = source_coeffs.unsqueeze(0)
    if source_coeffs.ndim != 2:
        raise ValueError(
            f"source identity_coeffs must have shape (N, C); got {tuple(source_coeffs.shape)}"
        )
    expected_source_width = _identity_param_count(source_layer)
    if source_coeffs.shape[1] != expected_source_width:
        raise ValueError(
            f"Source backend '{source_layer.identity_model_type}' expects "
            f"{expected_source_width} identity coefficients; got {source_coeffs.shape[1]}"
        )

    rows = source_coeffs.shape[0]
    source_neutral_scale = neutral_scale_params(source_layer, rows)
    source_scales = _as_parameter_rows(
        source_scale_params,
        rows=rows,
        width=_scale_param_count(source_layer),
        name="source scale_params",
        device=_layer_device(source_layer),
        default=source_neutral_scale,
    )
    with torch.no_grad():
        source_vertices = bind_pose_vertices(
            source_layer,
            source_coeffs,
            source_scales,
            global_scale,
        ).detach()
        source_vertices = _center_vertices(source_vertices)

    target_coeffs = torch.zeros(
        rows,
        _identity_param_count(target_layer),
        dtype=source_vertices.dtype,
        device=_layer_device(target_layer),
        requires_grad=True,
    )
    target_neutral_scale = neutral_scale_params(target_layer, rows)
    target_scales = None
    parameters: list[torch.Tensor] = [target_coeffs]
    if target_neutral_scale is not None:
        target_scales = target_neutral_scale.clone().detach()
        target_scales.requires_grad_(optimize_scale_params)
        if optimize_scale_params:
            parameters.append(target_scales)

    log_global_scale = torch.tensor(
        np.log(global_scale),
        dtype=source_vertices.dtype,
        device=_layer_device(target_layer),
        requires_grad=optimize_global_scale,
    )
    if optimize_global_scale:
        parameters.append(log_global_scale)

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    loss_history = []
    best_loss = float("inf")
    best_coeffs = target_coeffs.detach().clone()
    best_scales = None if target_scales is None else target_scales.detach().clone()
    best_global_scale = float(global_scale)
    target_neutral_coeffs = torch.zeros_like(target_coeffs)

    for _ in range(iterations):
        optimizer.zero_grad()
        target_global_scale = log_global_scale.exp() if optimize_global_scale else global_scale
        target_vertices = bind_pose_vertices(
            target_layer,
            target_coeffs,
            target_scales,
            target_global_scale,
        )
        target_vertices = _center_vertices(target_vertices)
        if target_vertices.shape != source_vertices.shape:
            raise ValueError(
                "Source and target layers produced different SOMA topology shapes: "
                f"{tuple(source_vertices.shape)} vs {tuple(target_vertices.shape)}"
            )
        data_loss = torch.nn.functional.mse_loss(target_vertices, source_vertices)
        reg_loss = (target_coeffs - target_neutral_coeffs).square().mean()
        if target_scales is not None and optimize_scale_params:
            reg_loss = reg_loss + (target_scales - target_neutral_scale).square().mean()
        if optimize_global_scale:
            reg_loss = reg_loss + (log_global_scale - np.log(global_scale)).square()
        loss = data_loss + regularization * reg_loss
        loss_value = float(loss.detach())
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_coeffs = target_coeffs.detach().clone()
            best_scales = None if target_scales is None else target_scales.detach().clone()
            best_global_scale = (
                float(log_global_scale.detach().exp())
                if optimize_global_scale
                else float(global_scale)
            )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_vertices = bind_pose_vertices(
            target_layer,
            best_coeffs,
            best_scales,
            best_global_scale,
        )
        delta = _center_vertices(final_vertices) - source_vertices
        vertex_error = torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)

    return IdentityConversionResult(
        identity_coeffs=best_coeffs.cpu().numpy(),
        scale_params=None if best_scales is None else best_scales.cpu().numpy(),
        global_scale=best_global_scale,
        vertex_error=vertex_error.cpu().numpy(),
        loss_history=loss_history,
    )


def _poses_for_resave(data: SOMANPZData) -> tuple[np.ndarray, list[str]]:
    poses = data.poses
    joint_names = list(data.joint_names)
    if data.keep_root:
        return poses, joint_names
    root_shape = list(poses.shape)
    root_shape[1] = 1
    if data.rotation_repr == "matrix":
        root_pose = np.broadcast_to(np.eye(3, dtype=poses.dtype), root_shape).copy()
    else:
        root_pose = np.zeros(root_shape, dtype=poses.dtype)
    return np.concatenate([root_pose, poses], axis=1), ["Root", *joint_names]


def convert_soma_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_backend: str,
    layer_factory: Callable[[str, str, str], Any],
    optimize_scale_params: bool = True,
    optimize_global_scale: bool = False,
    iterations: int = 200,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    expected_hand_type: str | None = None,
) -> IdentityConversionResult:
    """Load, convert, and resave a canonical SOMA NPZ animation."""
    data = load_soma_npz(input_path)
    source_backend = data.identity_model_type.lower()
    file_hand_type = data.get("hand_type")
    if expected_hand_type is None and file_hand_type is not None:
        raise ValueError("Full-body conversion does not accept a hand SOMA NPZ")
    if expected_hand_type is not None:
        if file_hand_type is not None and file_hand_type != expected_hand_type:
            raise ValueError(
                f"Input hand_type is '{file_hand_type}', but '{expected_hand_type}' was requested"
            )

    source_layer = layer_factory(source_backend, data.unit, "source")
    target_layer = layer_factory(target_backend, data.unit, "target")
    input_global_scale = float(data.get("global_scale", 1.0))
    result = convert_identity_parameters(
        source_layer,
        target_layer,
        data.identity_coeffs,
        source_scale_params=data.get("scale_params"),
        global_scale=input_global_scale,
        optimize_scale_params=optimize_scale_params,
        optimize_global_scale=optimize_global_scale,
        iterations=iterations,
        learning_rate=learning_rate,
        regularization=regularization,
    )

    extra_arrays: dict[str, Any] = {
        key: value for key, value in data.items() if key not in _STANDARD_NPZ_KEYS
    }
    extra_arrays.update(
        {
            "conversion_source_identity_model_type": np.array(source_backend),
            "conversion_source_identity_coeffs": np.asarray(data.identity_coeffs),
            "conversion_vertex_error": result.vertex_error.astype(np.float32),
            "conversion_iterations": np.int32(iterations),
            "conversion_loss_history": np.asarray(result.loss_history, dtype=np.float32),
        }
    )
    if "scale_params" in data:
        extra_arrays["conversion_source_scale_params"] = np.asarray(data.scale_params)
    if "global_scale" in data:
        extra_arrays["conversion_source_global_scale"] = np.float32(input_global_scale)

    poses, joint_names = _poses_for_resave(data)
    output_global_scale = (
        result.global_scale if optimize_global_scale or "global_scale" in data else None
    )
    save_soma_npz(
        output_path,
        poses,
        data.transl,
        joint_names=joint_names,
        identity_model_type=target_backend,
        identity_coeffs=result.identity_coeffs,
        scale_params=result.scale_params,
        joint_orient=data.get("joint_orient"),
        global_scale=output_global_scale,
        hand_type=expected_hand_type,
        unit=data.unit,
        keep_root=data.keep_root,
        extra_arrays=extra_arrays,
    )
    logger.info(
        "Converted %s -> %s with mean bind-pose vertex error %.6f %s",
        source_backend,
        target_backend,
        float(result.vertex_error.mean()),
        data.unit,
    )
    return result


def model_kwargs(model_path: str | None) -> Mapping[str, Any] | None:
    """Build identity-model constructor kwargs for optional licensed model files."""
    return None if model_path is None else {"model_path": model_path}
