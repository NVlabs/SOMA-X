# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert pose parameters into native SMPL-family pose parameters."""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma.body import SOMALayer  # noqa: E402
from soma.geometry.rig_utils import joint_world_to_local  # noqa: E402
from soma.io import export_soma_usd, load_soma_npz  # noqa: E402
from soma.smpl import create_smpl_family_layer, transfer_smpl_family_pose_parameters  # noqa: E402
from soma.units import Unit  # noqa: E402
from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

BODY_FIT_DEFAULTS = {
    "body_iters": 2,
    "full_iters": 1,
    "lie_iters": 3,
    "lie_lambda": 1e-1,
    "batch_size": 64,
}


def _is_soma_spec(spec: str) -> bool:
    normalized = spec.lower().replace("_", "-")
    return normalized in {
        "soma",
        "somalayer",
        "soma-body",
        "soma-proc",
        "soma-procedural",
        "soma-body-proc",
        "soma-body-procedural",
    }


def _is_procedural_soma_body_spec(spec: str) -> bool:
    normalized = spec.lower().replace("_", "-")
    return normalized in {
        "soma-proc",
        "soma-procedural",
        "soma-body-proc",
        "soma-body-procedural",
    }


def _ensure_smpl_body_spec(spec: str) -> str:
    normalized = spec.lower().replace("_", "-")
    if normalized not in {"smpl", "smplx"}:
        raise ValueError(
            f"Public pose converter supports only SMPL/SMPL-X body specs, got {spec!r}."
        )
    return normalized


def _torch_from_npz(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        if key in data:
            return torch.from_numpy(np.asarray(data[key])).float()
    return None


def _string_from_npz(data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in data:
        return default
    value = data[key]
    if hasattr(value, "shape") and value.shape == ():
        return str(value.item())
    return str(value)


def _load_soma_source(
    spec: str,
    input_path: Path,
    data_root: Path,
    device: torch.device,
    mode: str,
    apply_correctives: bool,
) -> tuple[torch.nn.Module, torch.Tensor, bool, bool, torch.Tensor, torch.Tensor, dict, dict]:
    data = load_soma_npz(input_path)
    output_unit = Unit.from_name(data["unit"])
    identity_model_type = data["identity_model_type"]

    layer = SOMALayer(
        data_root,
        identity_model_type=identity_model_type,
        low_lod=False,
        device=device,
        mode=mode,
        output_unit=output_unit,
        enable_procedural_transforms=_is_procedural_soma_body_spec(spec),
    )
    source_pose_kwargs = {"apply_correctives": apply_correctives}

    poses = torch.from_numpy(np.asarray(data["poses"])).float().to(device)
    pose2rot = data["rotation_repr"] == "rotvec"
    absolute_pose = bool(data["absolute_pose"])
    root_translation = torch.from_numpy(np.asarray(data["transl"])).float().to(device)
    identity = torch.from_numpy(np.asarray(data["identity_coeffs"])).float().to(device)

    prepare_kwargs = {}
    if "scale_params" in data:
        prepare_kwargs["scale_params"] = (
            torch.from_numpy(np.asarray(data["scale_params"])).float().to(device)
        )
    if "global_scale" in data:
        prepare_kwargs["global_scale"] = data["global_scale"]

    identity_kwargs = {}
    if "bone_length_flexibles" in data:
        identity_kwargs["bone_length_flexibles"] = (
            torch.from_numpy(np.asarray(data["bone_length_flexibles"])).float().to(device)
        )
    if identity_kwargs:
        prepare_kwargs["kwargs"] = identity_kwargs

    return (
        layer,
        poses,
        pose2rot,
        absolute_pose,
        root_translation,
        identity,
        prepare_kwargs,
        source_pose_kwargs,
    )


def _amass_pose_array(
    poses: np.ndarray,
    source_num_joints: int,
    source_model_spec: str,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None]
    if poses.ndim == 3 and poses.shape[-2:] == (source_num_joints, 3):
        return poses
    if poses.ndim != 2:
        raise ValueError(f"Expected AMASS-style poses with shape (T, P), got {poses.shape}.")

    target_width = source_num_joints * 3
    if poses.shape[1] == target_width:
        return poses.reshape(poses.shape[0], source_num_joints, 3)

    out = np.zeros((poses.shape[0], target_width), dtype=np.float32)
    if source_model_spec == "smpl" and poses.shape[1] >= 66:
        out[:, :66] = poses[:, :66]
    elif source_model_spec == "smplx" and poses.shape[1] >= 156:
        out[:, :156] = poses[:, :156]
    else:
        width = min(poses.shape[1], target_width)
        out[:, :width] = poses[:, :width]
    return out.reshape(poses.shape[0], source_num_joints, 3)


def _load_amass_style_source(
    spec: str,
    input_path: Path,
    data_root: Path,
    device: torch.device,
    mode: str,
    apply_correctives: bool,
) -> tuple[torch.nn.Module, torch.Tensor, bool, bool, torch.Tensor, torch.Tensor, dict, dict]:
    data = np.load(input_path, allow_pickle=True)
    gender = _string_from_npz(data, "gender", "neutral").lower()
    spec = _ensure_smpl_body_spec(spec)
    layer = create_smpl_family_layer(
        spec,
        data_root,
        device=device,
        mode=mode,
        output_unit=Unit.METERS,
        gender=gender,
    )

    if "poses" not in data:
        raise ValueError("SMPL-family input must contain AMASS-style key 'poses'.")

    poses = (
        torch.from_numpy(_amass_pose_array(data["poses"], layer.num_joints, layer.model_spec))
        .float()
        .to(device)
    )
    num_frames = poses.shape[0]

    trans = data["trans"] if "trans" in data else np.zeros((num_frames, 3), dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim == 1:
        trans = trans[None]
    root_translation = torch.from_numpy(trans).float().to(device)

    betas = (
        data["betas"] if "betas" in data else np.zeros(layer.num_identity_coeffs, dtype=np.float32)
    )
    identity = torch.from_numpy(np.asarray(betas, dtype=np.float32)).to(device)

    return (
        layer,
        poses,
        True,
        True,
        root_translation,
        identity,
        {},
        {"apply_correctives": apply_correctives},
    )


def _load_source(
    spec: str,
    input_path: Path,
    data_root: Path,
    device: torch.device,
    mode: str,
    apply_correctives: bool,
) -> tuple[torch.nn.Module, torch.Tensor, bool, bool, torch.Tensor, torch.Tensor, dict, dict]:
    if _is_soma_spec(spec):
        return _load_soma_source(
            spec,
            input_path,
            data_root,
            device,
            mode,
            apply_correctives,
        )
    return _load_amass_style_source(spec, input_path, data_root, device, mode, apply_correctives)


def _stats(error: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(error.mean().detach().cpu()),
        "median": float(error.median().detach().cpu()),
        "max": float(error.max().detach().cpu()),
    }


def _fit_kwargs_for_target(
    target_layer: torch.nn.Module,
    *,
    body_iters: int | None = None,
    full_iters: int | None = None,
    lie_iters: int | None = None,
    lie_lambda: float | None = None,
    batch_size: int | None = None,
) -> dict[str, int | float]:
    kwargs = dict(BODY_FIT_DEFAULTS)
    overrides = {
        "body_iters": body_iters,
        "full_iters": full_iters,
        "lie_iters": lie_iters,
        "lie_lambda": lie_lambda,
        "batch_size": batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            kwargs[key] = value
    return kwargs


def _root_joint_idx(layer: torch.nn.Module) -> int:
    return int(getattr(layer, "root_joint_idx", 0))


def _layer_unit_name(layer: torch.nn.Module) -> str:
    unit = getattr(layer, "output_unit", None)
    if isinstance(unit, Unit):
        return unit.unit_name
    if isinstance(unit, str):
        return Unit.from_name(unit).unit_name
    native_unit = getattr(layer, "NATIVE_UNIT", Unit.METERS)
    if isinstance(native_unit, Unit):
        return native_unit.unit_name
    return "meters"


def _pose_layer_for_inspection(
    layer: torch.nn.Module,
    poses: torch.Tensor,
    root_translation: torch.Tensor,
    *,
    pose2rot: bool,
    absolute_pose: bool,
    extra_kwargs: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    pose_params = inspect.signature(layer.pose).parameters
    kwargs: dict[str, object] = {
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
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return layer.pose(poses, **kwargs)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _output_parent_ids(layer: torch.nn.Module, layer_out: dict[str, torch.Tensor]) -> torch.Tensor:
    transforms = layer_out.get("transforms")
    output_parent_ids = getattr(layer, "output_joint_parent_ids", None)
    if output_parent_ids is not None:
        if transforms is None or transforms.shape[-3] == len(output_parent_ids):
            return output_parent_ids
    public_parent_ids = getattr(layer, "public_joint_parent_ids", None)
    if (
        transforms is not None
        and public_parent_ids is not None
        and transforms.shape[-3] == len(public_parent_ids)
    ):
        return public_parent_ids
    return layer.joint_parent_ids


def _skeleton_positions_for_render(
    layer_out: dict[str, torch.Tensor],
    parent_ids: np.ndarray,
) -> np.ndarray:
    joints = _to_numpy(layer_out["joints"])
    if joints.shape[1] == len(parent_ids):
        return joints
    transforms = layer_out.get("transforms")
    if transforms is None:
        return joints
    transform_joints = _to_numpy(transforms[..., :3, 3])
    if transform_joints.shape[1] == len(parent_ids):
        return transform_joints
    return joints


def _export_inspection_usds(
    inspect_dir: Path,
    *,
    source_layer: torch.nn.Module,
    target_layer: torch.nn.Module,
    source_out: dict[str, torch.Tensor],
    target_out: dict[str, torch.Tensor],
    source_rotations: torch.Tensor,
    source_root_translation: torch.Tensor,
    target_rotations: torch.Tensor,
    target_root_translation: torch.Tensor,
    fps: float,
) -> dict[str, str]:
    inspect_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_skeleton_usd": inspect_dir / "source_skeleton.usda",
        "target_reconstruction_skeleton_usd": inspect_dir / "target_reconstruction_skeleton.usda",
    }

    export_soma_usd(
        paths["source_skeleton_usd"],
        source_layer,
        source_rotations,
        source_root_translation,
        fps=fps,
        root_joint_idx=_root_joint_idx(source_layer),
        skin_mesh_name=getattr(source_layer, "default_skin_mesh_name", "source_mesh"),
    )
    export_soma_usd(
        paths["target_reconstruction_skeleton_usd"],
        target_layer,
        target_rotations,
        target_root_translation,
        fps=fps,
        root_joint_idx=_root_joint_idx(target_layer),
        skin_mesh_name=getattr(target_layer, "default_skin_mesh_name", "target_mesh"),
    )
    return {key: str(path) for key, path in paths.items()}


def _render_inspection(
    inspect_dir: Path,
    *,
    source_layer: torch.nn.Module,
    target_layer: torch.nn.Module,
    source_out: dict[str, torch.Tensor],
    target_out: dict[str, torch.Tensor],
    target_fit_vertices: torch.Tensor,
    fps: float,
    image_size: int,
    max_render_frames: int | None,
) -> dict[str, str]:
    import imageio.v2 as imageio

    from tools.vis_pyrender import (
        MeshRenderer,
        compute_camera_pose,
        default_pyopengl_platform,
        overlay_skeleton,
        render_mesh_panel,
        save_image,
        set_pyopengl_platform,
    )

    set_pyopengl_platform(default_pyopengl_platform())
    inspect_dir.mkdir(parents=True, exist_ok=True)

    source_vertices = _to_numpy(source_out["vertices"])
    target_vertices = _to_numpy(target_out["vertices"])
    fit_vertices = _to_numpy(target_fit_vertices)
    source_faces = _to_numpy(source_layer.faces).astype(np.int32)
    target_faces = _to_numpy(target_layer.faces).astype(np.int32)
    source_parents = _to_numpy(_output_parent_ids(source_layer, source_out)).astype(np.int32)
    target_parents = _to_numpy(_output_parent_ids(target_layer, target_out)).astype(np.int32)
    source_joints = _skeleton_positions_for_render(source_out, source_parents)
    target_joints = _skeleton_positions_for_render(target_out, target_parents)

    frame_count = source_vertices.shape[0]
    if max_render_frames is not None:
        frame_count = min(frame_count, max_render_frames)

    cam_seed = np.concatenate([source_vertices[0], fit_vertices[0], target_vertices[0]], axis=0)
    source_cam_pose = compute_camera_pose(source_vertices[0], cam_dist_scale=4.5)
    target_cam_pose = compute_camera_pose(
        np.concatenate([fit_vertices[0], target_vertices[0]], axis=0),
        cam_dist_scale=4.5,
    )
    light_dir = np.array([0.0, -0.3, -1.0])
    extent = np.linalg.norm(cam_seed.max(axis=0) - cam_seed.min(axis=0))
    joint_radius = max(float(extent) * 0.008, 0.001)
    bone_radius = joint_radius * 0.35

    comparison_path = inspect_dir / "render_comparison.mp4"
    overlay_path = inspect_dir / "render_overlay.mp4"
    comparison_writer = imageio.get_writer(str(comparison_path), fps=fps)
    overlay_writer = imageio.get_writer(str(overlay_path), fps=fps)
    first_comparison = None
    first_overlay = None
    renderer = MeshRenderer(image_size=image_size, light_intensity=5.0)
    renderer.camera.zfar = 500.0
    try:
        for frame_idx in range(frame_count):
            source_panel = render_mesh_panel(
                renderer,
                source_vertices[frame_idx],
                source_faces,
                mesh_color=(0.65, 0.65, 0.65, 1.0),
                cam_pose=source_cam_pose,
                light_dir=light_dir,
            )
            source_panel = overlay_skeleton(
                renderer,
                source_panel,
                source_joints[frame_idx],
                source_parents,
                color=(0.9, 0.15, 0.12, 1.0),
                cam_pose=source_cam_pose,
                light_dir=light_dir,
                joint_radius=joint_radius,
                bone_radius=bone_radius,
            )

            target_panel = render_mesh_panel(
                renderer,
                target_vertices[frame_idx],
                target_faces,
                mesh_color=(0.25, 0.75, 0.35, 1.0),
                cam_pose=target_cam_pose,
                light_dir=light_dir,
            )
            target_panel = overlay_skeleton(
                renderer,
                target_panel,
                target_joints[frame_idx],
                target_parents,
                color=(0.0, 0.35, 0.12, 1.0),
                cam_pose=target_cam_pose,
                light_dir=light_dir,
                joint_radius=joint_radius,
                bone_radius=bone_radius,
            )

            fit_panel = render_mesh_panel(
                renderer,
                fit_vertices[frame_idx],
                target_faces,
                mesh_color=(0.65, 0.65, 0.65, 1.0),
                cam_pose=target_cam_pose,
                light_dir=light_dir,
            )
            recon_panel = render_mesh_panel(
                renderer,
                target_vertices[frame_idx],
                target_faces,
                mesh_color=(0.15, 0.65, 0.95, 1.0),
                cam_pose=target_cam_pose,
                light_dir=light_dir,
            )
            overlay_panel = np.clip(0.55 * fit_panel + 0.45 * recon_panel, 0, 255).astype(np.uint8)
            overlay_panel = overlay_skeleton(
                renderer,
                overlay_panel,
                target_joints[frame_idx],
                target_parents,
                color=(0.0, 0.25, 0.7, 1.0),
                cam_pose=target_cam_pose,
                light_dir=light_dir,
                joint_radius=joint_radius,
                bone_radius=bone_radius,
            )

            comparison = np.concatenate([source_panel, target_panel, overlay_panel], axis=1)
            if first_comparison is None:
                first_comparison = comparison
                first_overlay = overlay_panel
            comparison_writer.append_data(comparison[..., ::-1])
            overlay_writer.append_data(overlay_panel[..., ::-1])
    finally:
        comparison_writer.close()
        overlay_writer.close()
        renderer.delete()

    frame_path = inspect_dir / "render_comparison_frame0.png"
    overlay_frame_path = inspect_dir / "render_overlay_frame0.png"
    if first_comparison is not None:
        save_image(str(frame_path), first_comparison)
    if first_overlay is not None:
        save_image(str(overlay_frame_path), first_overlay)

    return {
        "comparison_video": str(comparison_path),
        "overlay_video": str(overlay_path),
        "comparison_frame": str(frame_path),
        "overlay_frame": str(overlay_frame_path),
    }


def _write_inspection_outputs(
    inspect_dir: Path,
    *,
    source_layer: torch.nn.Module,
    target_layer: torch.nn.Module,
    source_poses: torch.Tensor,
    source_root: torch.Tensor,
    source_pose2rot: bool,
    source_absolute_pose: bool,
    source_pose_kwargs: dict[str, object],
    result,
    export_usd: bool,
    render: bool,
    fps: float,
    image_size: int,
    max_render_frames: int | None,
) -> dict[str, str]:
    with torch.no_grad():
        source_out = _pose_layer_for_inspection(
            source_layer,
            source_poses,
            source_root,
            pose2rot=source_pose2rot,
            absolute_pose=source_absolute_pose,
            extra_kwargs=source_pose_kwargs,
        )
        target_out = target_layer.pose(
            result.rotations,
            pose2rot=False,
            apply_correctives=False,
            absolute_pose=True,
            global_translation=result.root_translation,
        )
        source_parent_ids = _output_parent_ids(source_layer, source_out)
        source_local = joint_world_to_local(source_out["transforms"], source_parent_ids)
        source_rotations = source_local[..., :3, :3]
        source_root_translation = source_local[:, _root_joint_idx(source_layer), :3, 3]

    outputs = {}
    if export_usd:
        outputs.update(
            _export_inspection_usds(
                inspect_dir,
                source_layer=source_layer,
                target_layer=target_layer,
                source_out=source_out,
                target_out=target_out,
                source_rotations=source_rotations,
                source_root_translation=source_root_translation,
                target_rotations=result.rotations,
                target_root_translation=result.root_translation,
                fps=fps,
            )
        )
    if render:
        outputs.update(
            _render_inspection(
                inspect_dir,
                source_layer=source_layer,
                target_layer=target_layer,
                source_out=source_out,
                target_out=target_out,
                target_fit_vertices=result.fit_vertices,
                fps=fps,
                image_size=image_size,
                max_render_frames=max_render_frames,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("assets"))
    parser.add_argument(
        "--source",
        required=True,
        help="soma, soma-procedural, smpl, or smplx",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="SMPL-family body target: smpl or smplx",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .npz path")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="warp", choices=("warp",))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--body-iters",
        type=int,
        default=None,
        help="Override model-specific PoseInversion body_iters.",
    )
    parser.add_argument(
        "--full-iters",
        type=int,
        default=None,
        help="Override model-specific PoseInversion full_iters.",
    )
    parser.add_argument(
        "--lie-iters",
        type=int,
        default=None,
        help="Override model-specific PoseInversion lie_iters.",
    )
    parser.add_argument(
        "--lie-lambda",
        type=float,
        default=None,
        help="Override model-specific PoseInversion Lie-GN regularization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override model-specific pose-inversion chunk size.",
    )
    parser.add_argument(
        "--apply-correctives",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply SOMA source pose correctives before fitting target pose parameters.",
    )
    parser.add_argument(
        "--inspect-dir",
        type=Path,
        default=None,
        help="Directory for optional USD and render inspection outputs.",
    )
    parser.add_argument(
        "--export-usd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export source/fit/reconstruction mesh USDs and source/target skeletal USDs.",
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render source, reconstruction, and overlay videos with skeletons.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Inspection output FPS.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Rendered panel size in pixels.",
    )
    parser.add_argument(
        "--max-render-frames",
        type=int,
        default=60,
        help="Maximum frames to render.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    (
        source_layer,
        source_poses,
        source_pose2rot,
        source_absolute_pose,
        source_root,
        source_id,
        source_prepare_kwargs,
        source_pose_kwargs,
    ) = _load_source(
        args.source,
        args.input,
        args.data_root,
        device,
        args.mode,
        args.apply_correctives,
    )
    target_layer = create_smpl_family_layer(
        _ensure_smpl_body_spec(args.target),
        args.data_root,
        device=device,
        mode=args.mode,
        output_unit=Unit.METERS,
    )

    if args.max_frames is not None:
        source_poses = source_poses[: args.max_frames]
        source_root = source_root[: args.max_frames]
        if source_id.ndim > 1 and source_id.shape[0] > 1:
            source_id = source_id[: args.max_frames]
        if "scale_params" in source_prepare_kwargs:
            scale_params = source_prepare_kwargs["scale_params"]
            if scale_params.ndim > 1 and scale_params.shape[0] > 1:
                source_prepare_kwargs["scale_params"] = scale_params[: args.max_frames]
        if "kwargs" in source_prepare_kwargs:
            for key, value in list(source_prepare_kwargs["kwargs"].items()):
                if hasattr(value, "ndim") and value.ndim > 1 and value.shape[0] > 1:
                    source_prepare_kwargs["kwargs"][key] = value[: args.max_frames]

    fit_kwargs = _fit_kwargs_for_target(
        target_layer,
        body_iters=args.body_iters,
        full_iters=args.full_iters,
        lie_iters=args.lie_iters,
        lie_lambda=args.lie_lambda,
        batch_size=args.batch_size,
    )

    result = transfer_smpl_family_pose_parameters(
        source_layer,
        target_layer,
        source_poses,
        source_identity_coeffs=source_id,
        source_root_translation=source_root,
        source_pose2rot=source_pose2rot,
        source_absolute_pose=source_absolute_pose,
        source_prepare_kwargs=source_prepare_kwargs,
        source_pose_kwargs=source_pose_kwargs,
        fit_kwargs=fit_kwargs,
    )

    inspection_outputs = {}
    if args.export_usd or args.render:
        inspect_dir = args.inspect_dir
        if inspect_dir is None:
            inspect_dir = args.output.with_suffix("").parent / f"{args.output.stem}_inspect"
        inspection_outputs = _write_inspection_outputs(
            inspect_dir,
            source_layer=source_layer,
            target_layer=target_layer,
            source_poses=source_poses,
            source_root=source_root,
            source_pose2rot=source_pose2rot,
            source_absolute_pose=source_absolute_pose,
            source_pose_kwargs=source_pose_kwargs,
            result=result,
            export_usd=args.export_usd,
            render=args.render,
            fps=args.fps,
            image_size=args.image_size,
            max_render_frames=args.max_render_frames,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target_rotations=result.rotations.detach().cpu().numpy(),
        target_root_translation=result.root_translation.detach().cpu().numpy(),
        per_vertex_error=result.per_vertex_error.detach().cpu().numpy(),
        source_vertices=result.source_vertices.detach().cpu().numpy(),
        fit_vertices=result.fit_vertices.detach().cpu().numpy(),
        reconstructed_vertices=result.reconstructed_vertices.detach().cpu().numpy(),
        source=np.asarray(args.source),
        target=np.asarray(args.target),
        fit_kwargs_json=np.asarray(json.dumps(fit_kwargs)),
        inspection_outputs_json=np.asarray(json.dumps(inspection_outputs)),
    )

    print(
        json.dumps(
            {
                "fit_kwargs": fit_kwargs,
                "inspection_outputs": inspection_outputs,
                "per_vertex_error": _stats(result.per_vertex_error),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
