# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prototype SOMAHand-to-MANO round trip.

This tool exercises:
  MANO test data -> SOMAHand pose inversion -> SOMAHand mesh ->
  MANO topology transfer -> MANO rig pose inversion.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma._smpl_family_loader import ensure_chumpy_compat  # noqa: E402
from soma.fitting.pose_inversion import PoseInversion  # noqa: E402
from soma.geometry.barycentric_interp import BarycentricInterpolator  # noqa: E402
from soma.hand import SOMAHandLayer  # noqa: E402
from soma.hand.mano import MANOLayer  # noqa: E402
from soma.io import export_soma_usd, save_vertex_animation_usd  # noqa: E402
from tools.conversion_utils import add_hand_inversion_args  # noqa: E402
from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_mano_data(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {path}")
    else:
        raise ValueError(f"Expected .npz file or directory, got {path}")
    return [{"path": f, "data": np.load(f, allow_pickle=True)} for f in files]


def build_soma_to_mano_interpolator(
    data_root: str | Path,
    hand_type: str,
    device: torch.device,
) -> BarycentricInterpolator:
    mano_dir = Path(data_root) / "MANO"
    mesh_soma = trimesh.load(
        mano_dir / f"SOMA_wrap_{hand_type}.obj",
        maintain_order=True,
        process=False,
    )
    mesh_mano = trimesh.load(
        mano_dir / f"base_hand_{hand_type}.obj",
        maintain_order=True,
        process=False,
    )
    v_soma = torch.from_numpy(np.asarray(mesh_soma.vertices, dtype=np.float32)).to(device)
    f_soma = torch.from_numpy(np.asarray(mesh_soma.faces, dtype=np.int64)).to(device)
    v_mano = torch.from_numpy(np.asarray(mesh_mano.vertices, dtype=np.float32)).to(device)
    return BarycentricInterpolator(v_soma, f_soma, v_mano)


def error_stats(error_m: torch.Tensor) -> dict[str, float]:
    error_mm = error_m.detach() * 1000.0
    per_frame = error_mm.mean(dim=1)
    return {
        "mean_mm": float(error_mm.mean().cpu()),
        "median_mm": float(error_mm.median().cpu()),
        "max_mm": float(error_mm.max().cpu()),
        "per_frame_min_mm": float(per_frame.min().cpu()),
        "per_frame_max_mm": float(per_frame.max().cpu()),
    }


def rotation_error_stats(recovered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return angular error stats between two rotation-matrix animation tensors."""
    rel = recovered.detach() @ target.detach().transpose(-1, -2)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    error_deg = torch.rad2deg(torch.acos(cos_angle))
    per_frame = error_deg.mean(dim=1)
    return {
        "mean_deg": float(error_deg.mean().cpu()),
        "median_deg": float(error_deg.median().cpu()),
        "max_deg": float(error_deg.max().cpu()),
        "per_frame_min_deg": float(per_frame.min().cpu()),
        "per_frame_max_deg": float(per_frame.max().cpu()),
    }


def evaluate_source_mano_params(
    data_root: str | Path,
    hand_type: str,
    mano_params: dict[str, np.ndarray],
    root_translation: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate stored MANO rotation-matrix params with the official SMPL-X layer."""
    ensure_chumpy_compat()
    from smplx import MANOLayer

    model_path = Path(data_root) / "MANO" / f"MANO_{hand_type.upper()}.pkl"
    mano_layer = MANOLayer(
        model_path=str(model_path),
        use_pca=False,
        is_rhand=hand_type == "right",
        num_betas=10,
    ).to(device)

    frame_count = root_translation.shape[0]
    betas = torch.from_numpy(mano_params["betas"][:frame_count]).float().to(device)
    global_orient = (
        torch.from_numpy(mano_params["global_orient"][:frame_count])
        .float()
        .to(device)
        .reshape(frame_count, 1, 3, 3)
    )
    hand_pose = (
        torch.from_numpy(mano_params["hand_pose"][:frame_count])
        .float()
        .to(device)
        .reshape(frame_count, 15, 3, 3)
    )

    with torch.no_grad():
        out = mano_layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            transl=root_translation,
            return_verts=True,
        )
    return out.vertices, out.joints


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _export_mano_usd(
    out_path: str | Path,
    layer: MANOLayer,
    identity_coeffs: torch.Tensor,
    rotations: torch.Tensor,
    root_translation: torch.Tensor,
    fps: float,
) -> None:
    layer.prepare_identity(identity_coeffs)
    export_soma_usd(
        out_path,
        layer,
        rotations,
        root_translation,
        fps=fps,
        unit="meters",
        root_joint_idx=0,
        skin_mesh_name=layer.default_skin_mesh_name,
    )


def _export_soma_hand_usd(
    out_path: str | Path,
    layer: SOMAHandLayer,
    identity_coeffs: torch.Tensor,
    rotations: torch.Tensor,
    root_translation: torch.Tensor,
    fps: float,
) -> None:
    layer.prepare_identity(identity_coeffs)
    export_soma_usd(
        out_path,
        layer,
        rotations,
        root_translation,
        fps=fps,
        root_joint_idx=0,
        skin_mesh_name=layer.default_skin_mesh_name,
    )


def render_mesh_comparison(
    out_prefix: str,
    original: torch.Tensor,
    soma_to_mano: torch.Tensor,
    mano_recon: torch.Tensor,
    faces: torch.Tensor,
    image_size: int,
    fps: int,
    max_frames: int | None,
) -> None:
    import imageio.v2 as imageio

    from tools.vis_pyrender import (
        MeshRenderer,
        compute_camera_pose,
        default_pyopengl_platform,
        save_image,
        set_pyopengl_platform,
    )

    set_pyopengl_platform(default_pyopengl_platform())
    frame_count = original.shape[0] if max_frames is None else min(original.shape[0], max_frames)
    faces_np = _to_numpy(faces).astype(np.int32)
    panels = [
        (_to_numpy(original[:frame_count]), (0.65, 0.65, 0.65, 1.0)),
        (_to_numpy(soma_to_mano[:frame_count]), (0.2, 0.65, 0.95, 1.0)),
        (_to_numpy(mano_recon[:frame_count]), (0.3, 0.78, 0.2, 1.0)),
    ]

    cam_seed = np.concatenate([panel[0][0] for panel in panels], axis=0)
    cam_pose = compute_camera_pose(cam_seed, cam_dist_scale=4.5)
    light_dir = np.array([0.0, -0.3, -1.0])

    renderer = MeshRenderer(image_size=image_size, light_intensity=5.0)
    renderer.camera.zfar = 500.0
    comparison_writer = imageio.get_writer(f"{out_prefix}_comparison.mp4", fps=fps)
    soma_overlay_writer = imageio.get_writer(f"{out_prefix}_overlay_soma_to_mano.mp4", fps=fps)
    mano_overlay_writer = imageio.get_writer(f"{out_prefix}_overlay_mano_recon.mp4", fps=fps)
    first_comparison = None
    first_soma_overlay = None
    first_mano_overlay = None
    for frame_idx in range(frame_count):
        images = []
        for verts, color in panels:
            renderer.setup_mesh(
                faces=faces_np,
                mesh_color=color,
                metallic=0.0,
                roughness=0.5,
                cam_pose=cam_pose,
                light_dir=light_dir,
                base_color_factor=[0.9, 0.9, 0.9, 1.0],
            )
            images.append(renderer.render_frame(verts[frame_idx]))
        frame = np.concatenate(images, axis=1)
        soma_overlay = np.clip(0.55 * images[0] + 0.45 * images[1], 0, 255).astype(np.uint8)
        mano_overlay = np.clip(0.55 * images[0] + 0.45 * images[2], 0, 255).astype(np.uint8)

        if first_comparison is None:
            first_comparison = frame
            first_soma_overlay = soma_overlay
            first_mano_overlay = mano_overlay
        comparison_writer.append_data(frame[..., ::-1])
        soma_overlay_writer.append_data(soma_overlay[..., ::-1])
        mano_overlay_writer.append_data(mano_overlay[..., ::-1])

    comparison_writer.close()
    soma_overlay_writer.close()
    mano_overlay_writer.close()
    renderer.delete()
    if first_comparison is not None:
        save_image(f"{out_prefix}_comparison_frame0.png", first_comparison)
        save_image(f"{out_prefix}_overlay_soma_to_mano_frame0.png", first_soma_overlay)
        save_image(f"{out_prefix}_overlay_mano_recon_frame0.png", first_mano_overlay)


def process_sequence(
    *,
    seq: dict[str, Any],
    seq_idx: int,
    hand_type: str,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    mode: str,
    bcd_iters: int,
    lie_iters: int,
    lie_lambda: float,
    batch_size: int,
    max_frames: int | None,
    export_usd: bool,
    render: bool,
    image_size: int,
    fps: int,
    max_render_frames: int | None,
) -> dict[str, Any]:
    side = f"{hand_type[0]}h"
    data = seq["data"]
    mano_params = data[f"mano_{side}"].item()
    source_transl = torch.from_numpy(data[f"cam_t_{side}"]).float().to(device)
    source_global_orient = torch.from_numpy(mano_params["global_orient"]).float().to(device)
    source_hand_pose = torch.from_numpy(mano_params["hand_pose"]).float().to(device)
    betas = torch.from_numpy(mano_params["betas"]).float().to(device)

    if max_frames is not None:
        source_transl = source_transl[:max_frames]
        source_global_orient = source_global_orient[:max_frames]
        source_hand_pose = source_hand_pose[:max_frames]
        betas = betas[:max_frames]

    frame_count = source_transl.shape[0]
    betas_single = betas[:1]
    source_rotations = torch.cat(
        [
            source_global_orient.reshape(frame_count, 1, 3, 3),
            source_hand_pose.reshape(frame_count, 15, 3, 3),
        ],
        dim=1,
    )

    mano_layer = MANOLayer(data_root, hand_type=hand_type, device=device, mode=mode)
    with torch.no_grad():
        mano_layer.prepare_identity(betas_single.expand(frame_count, -1))
        source_mano = mano_layer.pose(
            source_rotations,
            pose2rot=False,
            absolute_pose=True,
            global_translation=source_transl,
        )
        source_mano_verts = source_mano["vertices"]
        source_wrist = source_mano["joints"][:, 0]
        mano_centered = source_mano_verts - source_wrist[:, None, :]
        source_mano_centered = mano_layer.pose(
            source_rotations,
            pose2rot=False,
            absolute_pose=True,
            global_translation=torch.zeros_like(source_transl),
        )["vertices"]

    hand_layer = SOMAHandLayer(
        data_root=str(data_root),
        hand_type=hand_type,
        device=str(device),
        identity_model_type="mano",
        mode=mode,
    ).to(device)
    soma_inv = PoseInversion(hand_layer, low_lod=False)
    with torch.no_grad():
        soma_inv.prepare_identity(betas_single)
        soma_target_m = soma_inv.transfer_to_soma(mano_centered)

    t0 = time.perf_counter()
    soma_result = soma_inv.fit(
        soma_target_m,
        body_iters=0,
        finger_iters=0,
        full_iters=bcd_iters,
        lie_iters=lie_iters,
        lie_lambda=lie_lambda,
        batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    soma_time = time.perf_counter() - t0

    with torch.no_grad():
        hand_layer.prepare_identity(betas_single.expand(frame_count, -1))
        soma_recon = hand_layer.pose(
            soma_result["rotations"],
            pose2rot=False,
            absolute_pose=True,
            global_translation=soma_result["root_translation"],
        )["vertices"]

    soma_to_mano = build_soma_to_mano_interpolator(data_root, hand_type, device)
    with torch.no_grad():
        mano_from_soma = soma_to_mano(soma_recon)

    mano_inv = PoseInversion(mano_layer, low_lod=False)
    with torch.no_grad():
        mano_inv.prepare_identity(betas_single)

    t_direct = time.perf_counter()
    direct_mano_result = mano_inv.fit(
        source_mano_centered,
        body_iters=0,
        finger_iters=0,
        full_iters=bcd_iters,
        lie_iters=lie_iters,
        lie_lambda=lie_lambda,
        batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    direct_mano_time = time.perf_counter() - t_direct

    t1 = time.perf_counter()
    mano_result = mano_inv.fit(
        mano_from_soma,
        body_iters=0,
        finger_iters=0,
        full_iters=bcd_iters,
        lie_iters=lie_iters,
        lie_lambda=lie_lambda,
        batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    mano_time = time.perf_counter() - t1

    with torch.no_grad():
        mano_layer.prepare_identity(betas_single.expand(frame_count, -1))
        mano_recon = mano_layer.pose(
            mano_result["rotations"],
            pose2rot=False,
            absolute_pose=True,
            global_translation=mano_result["root_translation"],
        )["vertices"]
        direct_mano_recon = mano_layer.pose(
            direct_mano_result["rotations"],
            pose2rot=False,
            absolute_pose=True,
            global_translation=direct_mano_result["root_translation"],
        )["vertices"]

    soma_error = torch.linalg.norm(soma_recon - soma_target_m, dim=-1)
    direct_mano_error = torch.linalg.norm(direct_mano_recon - source_mano_centered, dim=-1)
    mano_error = torch.linalg.norm(mano_recon - mano_from_soma, dim=-1)
    full_error = torch.linalg.norm(mano_recon - mano_centered, dim=-1)
    source_mesh_roundtrip_error = torch.linalg.norm(mano_recon - source_mano_centered, dim=-1)

    transl = source_wrist + mano_result["root_translation"]
    rotations = mano_result["rotations"]
    translation_error = torch.linalg.norm(transl - source_transl, dim=-1)[:, None]
    direct_translation_error = torch.linalg.norm(direct_mano_result["root_translation"], dim=-1)[
        :, None
    ]
    out_prefix = output_dir / f"{hand_type}_{seq_idx}_{seq['path'].stem[:32]}"
    stats = {
        "hand": hand_type,
        "sequence": str(seq["path"]),
        "frames": int(frame_count),
        "soma_fit": error_stats(soma_error),
        "direct_mano_fit": error_stats(direct_mano_error),
        "mano_fit": error_stats(mano_error),
        "full_roundtrip": error_stats(full_error),
        "source_mesh_roundtrip": error_stats(source_mesh_roundtrip_error),
        "direct_mano_rotation_roundtrip": rotation_error_stats(
            direct_mano_result["rotations"], source_rotations
        ),
        "direct_mano_translation_roundtrip": error_stats(direct_translation_error),
        "source_mano_rotation_roundtrip": rotation_error_stats(rotations, source_rotations),
        "source_mano_translation_roundtrip": error_stats(translation_error),
        "soma_time_s": soma_time,
        "direct_mano_time_s": direct_mano_time,
        "mano_time_s": mano_time,
        "soma_ms_per_frame": soma_time / frame_count * 1000.0,
        "direct_mano_ms_per_frame": direct_mano_time / frame_count * 1000.0,
        "mano_ms_per_frame": mano_time / frame_count * 1000.0,
    }

    np.savez_compressed(
        f"{out_prefix}_mano_params.npz",
        betas=_to_numpy(betas_single.expand(frame_count, -1)),
        global_orient=_to_numpy(rotations[:, 0:1]),
        hand_pose=_to_numpy(rotations[:, 1:]),
        transl=_to_numpy(transl),
        root_translation=_to_numpy(mano_result["root_translation"]),
        source_global_orient=_to_numpy(source_rotations[:, 0:1]),
        source_hand_pose=_to_numpy(source_rotations[:, 1:]),
        source_transl=_to_numpy(source_transl),
        direct_global_orient=_to_numpy(direct_mano_result["rotations"][:, 0:1]),
        direct_hand_pose=_to_numpy(direct_mano_result["rotations"][:, 1:]),
        direct_root_translation=_to_numpy(direct_mano_result["root_translation"]),
        stats_json=np.array(json.dumps(stats)),
    )

    with open(f"{out_prefix}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if export_usd:
        save_vertex_animation_usd(
            f"{out_prefix}_mano_source.usda",
            source_mano_verts,
            mano_layer.faces,
            fps=float(fps),
            unit="meters",
            prim_path="/MANO_Source",
        )
        _export_mano_usd(
            f"{out_prefix}_mano_source_skel.usda",
            mano_layer,
            betas_single,
            source_rotations,
            source_transl,
            fps=float(fps),
        )
        _export_mano_usd(
            f"{out_prefix}_mano_direct.usda",
            mano_layer,
            betas_single,
            direct_mano_result["rotations"],
            source_wrist + direct_mano_result["root_translation"],
            fps=float(fps),
        )
        _export_soma_hand_usd(
            f"{out_prefix}_soma_recon.usda",
            hand_layer,
            betas_single,
            soma_result["rotations"],
            soma_result["root_translation"] + source_wrist,
            fps=float(fps),
        )
        _export_mano_usd(
            f"{out_prefix}_mano_recon.usda",
            mano_layer,
            betas_single,
            rotations,
            transl,
            fps=float(fps),
        )

    if render:
        render_mesh_comparison(
            str(out_prefix),
            mano_centered,
            mano_from_soma,
            mano_recon,
            mano_layer.faces,
            image_size,
            fps,
            max_render_frames,
        )

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prototype SOMAHand -> MANO roundtrip.")
    parser.add_argument("--input", required=True, help="MANO test .npz file or directory.")
    parser.add_argument("--output-dir", default="out/soma2mano")
    parser.add_argument("--hand-type", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--mode",
        choices=["warp"],
        default="warp",
        help="Skinning mode for pose inversion. Only warp is supported by this prototype.",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-usd", action="store_true")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-render-frames", type=int, default=60)
    add_hand_inversion_args(parser, bcd_iters=3, lie_iters=10, batch_size=32)
    add_logging_args(parser)
    args = parser.parse_args(argv)
    configure_logging(args)

    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_mano_data(args.input)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    hand_types = ["left", "right"] if args.hand_type == "both" else [args.hand_type]

    all_stats = []
    for hand_type in hand_types:
        for seq_idx, seq in enumerate(sequences):
            stats = process_sequence(
                seq=seq,
                seq_idx=seq_idx,
                hand_type=hand_type,
                data_root=data_root,
                output_dir=output_dir,
                device=device,
                mode=args.mode,
                bcd_iters=args.bcd_iters,
                lie_iters=args.lie_iters,
                lie_lambda=args.lie_lambda,
                batch_size=args.batch_size,
                max_frames=args.max_frames,
                export_usd=not args.no_usd,
                render=not args.no_render,
                image_size=args.image_size,
                fps=args.fps,
                max_render_frames=args.max_render_frames,
            )
            all_stats.append(stats)
            logger.info(
                f"{hand_type} seq{seq_idx}: "
                f"SOMA mean={stats['soma_fit']['mean_mm']:.3f}mm, "
                f"direct MANO mean={stats['direct_mano_fit']['mean_mm']:.3f}mm, "
                f"MANO mean={stats['mano_fit']['mean_mm']:.3f}mm, "
                f"full mean={stats['full_roundtrip']['mean_mm']:.3f}mm, "
                f"rot mean={stats['source_mano_rotation_roundtrip']['mean_deg']:.3f}deg"
            )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
