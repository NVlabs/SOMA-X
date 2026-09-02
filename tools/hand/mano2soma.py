# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MANO to SOMA hand pose converter.

Loads MANO test data (betas + posed vertices + ground-truth joints),
transfers posed MANO meshes to SOMAHand topology, runs PoseInversion
to recover SOMAHand pose parameters, and validates the round trip
numerically and visually.

Pipeline:
  1. Load MANO test data -> posed MANO verts (778v)
  2. Transfer to SOMAHand topology (2859v) via BarycentricInterpolator
  3. PoseInversion.fit() -> recovered rotations + root translation
  4. Forward pass SOMAHandLayer -> reconstructed verts
  5. Numerical comparison + renders with skeleton overlays

Usage:
    python -m tools.hand.mano2soma --input assets/MANO/test_data
    python -m tools.hand.mano2soma --input assets/MANO/test_data --no-render
    python -m tools.hand.mano2soma --input assets/MANO/test_data --hand-type left --lie-iters 5
    python -m tools.hand.mano2soma --input assets/MANO/test_data/seq_0.npz --output-dir out/mano2soma
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma.fitting.pose_inversion import PoseInversion  # noqa: E402
from soma.geometry.rig_utils import joint_world_to_local  # noqa: E402
from soma.hand import SOMAHandLayer  # noqa: E402
from soma.hand.mano import MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS  # noqa: E402
from soma.io import export_soma_usd  # noqa: E402
from tools.conversion_utils import add_hand_inversion_args  # noqa: E402
from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402
from tools.vis_pyrender import overlay_skeleton, render_mesh_panel  # noqa: E402

logger = logging.getLogger(__name__)


def render_overlay_frame(
    renderer,
    mano_verts,
    mano_faces,
    mano_joints,
    mano_parent_ids,
    soma_verts,
    soma_faces,
    soma_joints,
    soma_parent_ids,
    cam_pose,
    light_dir,
    joint_radius=0.15,
    bone_radius=0.06,
):
    """Render 3-panel: MANO mesh+skel | SOMA-Hand mesh+skel | overlay+both skels."""
    # Panel 1: MANO target (gray) + MANO skeleton (red)
    img_mano = render_mesh_panel(
        renderer,
        mano_verts,
        mano_faces,
        mesh_color=(0.65, 0.65, 0.65, 1.0),
        cam_pose=cam_pose,
        light_dir=light_dir,
    )
    panel1 = overlay_skeleton(
        renderer,
        img_mano,
        mano_joints,
        mano_parent_ids,
        color=(0.75, 0.2, 0.15, 1.0),
        cam_pose=cam_pose,
        light_dir=light_dir,
        joint_radius=joint_radius,
        bone_radius=bone_radius,
        metallic=0.3,
        roughness=0.3,
    )

    # Panel 2: SOMA-Hand recon (green) + SOMA-Hand skeleton (dark green)
    img_soma = render_mesh_panel(
        renderer,
        soma_verts,
        soma_faces,
        mesh_color=(0.3, 0.78, 0.2, 1.0),
        cam_pose=cam_pose,
        light_dir=light_dir,
    )
    panel2 = overlay_skeleton(
        renderer,
        img_soma,
        soma_joints,
        soma_parent_ids,
        color=(0.15, 0.55, 0.15, 1.0),
        cam_pose=cam_pose,
        light_dir=light_dir,
        joint_radius=joint_radius,
        bone_radius=bone_radius,
        metallic=0.3,
        roughness=0.3,
    )

    # Panel 3: Blended overlay + both skeletons
    panel3 = (0.5 * img_mano.astype(np.float32) + 0.5 * img_soma.astype(np.float32)).astype(
        np.uint8
    )
    panel3 = overlay_skeleton(
        renderer,
        panel3,
        mano_joints,
        mano_parent_ids,
        color=(0.75, 0.2, 0.15, 1.0),
        cam_pose=cam_pose,
        light_dir=light_dir,
        joint_radius=joint_radius,
        bone_radius=bone_radius,
        metallic=0.3,
        roughness=0.3,
    )
    panel3 = overlay_skeleton(
        renderer,
        panel3,
        soma_joints,
        soma_parent_ids,
        color=(0.15, 0.55, 0.15, 1.0),
        cam_pose=cam_pose,
        light_dir=light_dir,
        joint_radius=joint_radius,
        bone_radius=bone_radius,
        metallic=0.3,
        roughness=0.3,
    )

    return np.concatenate([panel1, panel2, panel3], axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mano_data(path):
    """Load MANO test sequences from a .npz file or directory of .npz files.

    Returns:
        list of dicts with 'path' and 'data' keys.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {path}")
    else:
        raise ValueError(f"Expected .npz file or directory, got: {path}")

    sequences = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        sequences.append({"path": f, "data": d})
    return sequences


# ---------------------------------------------------------------------------
# USD export helper
# ---------------------------------------------------------------------------


def _export_usd(out_path, hand_layer, rotations, root_translation):
    """Export skeletal USD from SOMAHandLayer results."""
    export_soma_usd(out_path, hand_layer, rotations, root_translation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MANO to SOMA hand pose converter.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to MANO .npz file or directory of .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="out/mano2soma",
        help="Output directory for renders, USDs, and stats (default: out/mano2soma).",
    )
    parser.add_argument(
        "--hand-type",
        default="both",
        choices=["left", "right", "both"],
        help="Which hand(s) to process (default: both).",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")

    add_hand_inversion_args(
        parser,
        bcd_iters=1,
        lie_iters=3,
        batch_size=32,
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    # Rendering
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30).")
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF FPS (default: 15).")
    parser.add_argument("--pyopengl-platform", default=None)
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    sequences = load_mano_data(args.input)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    logger.info(f"Loaded {len(sequences)} test sequence(s) from {args.input}")

    # --- Set up rendering ---
    if not args.no_render:
        from tools.vis_pyrender import (
            MeshRenderer,
            compute_camera_pose,
            default_pyopengl_platform,
            save_image,
            set_pyopengl_platform,
        )

        platform = args.pyopengl_platform or default_pyopengl_platform()
        set_pyopengl_platform(platform)

    # --- Process ---
    hand_types = ["left", "right"] if args.hand_type == "both" else [args.hand_type]
    all_stats = []

    for hand_type in hand_types:
        mano_key = f"mano_{hand_type[0]}h"
        verts_key = f"verts_{hand_type[0]}h"
        joints_key = f"joints_{hand_type[0]}h"

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  {hand_type.upper()} HAND")
        logger.info(f"{'=' * 60}")

        hand_layer = SOMAHandLayer(
            data_root=str(data_root),
            hand_type=hand_type,
            device=str(device),
            identity_model_type="mano",
        ).to(device)

        pose_inv = PoseInversion(hand_layer, low_lod=False)

        faces_np = hand_layer.faces.detach().cpu().numpy().astype(np.int32)
        parent_ids = hand_layer.joint_parent_ids.cpu().tolist()

        for seq_idx, seq in enumerate(sequences):
            seq_name = seq["path"].stem[:40]
            d = seq["data"]

            mano_params = d[mano_key].item()
            gt_mano_verts = torch.from_numpy(d[verts_key]).float().to(device)
            gt_joints = torch.from_numpy(d[joints_key]).float().to(device)
            betas = torch.from_numpy(mano_params["betas"]).float().to(device)

            T = gt_mano_verts.shape[0]
            if args.max_frames and T > args.max_frames:
                T = args.max_frames
                gt_mano_verts = gt_mano_verts[:T]
                gt_joints = gt_joints[:T]
                betas = betas[:T]

            logger.info(f"\n--- Seq {seq_idx}: {seq_name} ({T} frames) ---")

            # ============================================================
            # Step 1: Transfer posed MANO verts to SOMAHand topology
            # ============================================================
            wrist_pos = gt_joints[:, 0:1, :]  # (T, 1, 3)
            gt_mano_centered = gt_mano_verts - wrist_pos  # meters

            with torch.no_grad():
                soma_target_verts = hand_layer.identity_model._to_soma_interp(
                    gt_mano_centered
                )  # (T, 2859, 3) meters

            logger.info(f"  MANO verts: {gt_mano_verts.shape}")
            logger.info(
                f"  SOMA target: {soma_target_verts.shape} ({hand_layer.output_unit.unit_name})"
            )

            # ============================================================
            # Step 2: Prepare identity and run pose inversion
            # ============================================================
            betas_single = betas[:1]
            out_prefix = f"{args.output_dir}/{hand_type}_{seq_idx}"

            with torch.no_grad():
                pose_inv.prepare_identity(betas_single)

            # Skeleton transfer only USD
            with torch.no_grad():
                skel_world = pose_inv._skel_transfer.fit(soma_target_verts)
                skel_local = joint_world_to_local(skel_world, hand_layer.joint_parent_ids)
                st_rotations = skel_local[:, :, :3, :3]
                st_root_t = skel_local[:, 0, :3, 3]

            _export_usd(f"{out_prefix}_skeltransfer.usda", hand_layer, st_rotations, st_root_t)
            logger.info(f"  Skeleton transfer USD: {out_prefix}_skeltransfer.usda")

            # BCD + Lie-GN
            t0 = time.perf_counter()
            result = pose_inv.fit(
                soma_target_verts,
                body_iters=0,
                finger_iters=0,
                full_iters=args.bcd_iters,
                lie_iters=args.lie_iters,
                lie_lambda=args.lie_lambda,
                batch_size=args.batch_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            lie_err = result["per_vertex_error"].mean().item() * 1000  # mm

            logger.info(f"  Inversion: {dt:.2f}s ({T / dt:.0f} FPS)")
            logger.info(f"  Pose inversion error (mean per-vert): {lie_err:.3f} mm")

            recovered_rotations = result["rotations"]  # (T, J, 3, 3)
            root_translation = result["root_translation"]  # (T, 3)

            # ============================================================
            # Step 3: Forward pass with recovered params
            # ============================================================
            with torch.no_grad():
                hand_layer.prepare_identity(betas_single.expand(T, -1))
                recon = hand_layer.pose(
                    recovered_rotations,
                    pose2rot=False,
                    absolute_pose=True,
                    global_translation=root_translation,
                )

            recon_verts_m = recon["vertices"]  # meters
            recon_joints_m = recon["joints"]  # meters

            # ============================================================
            # Step 4: Numerical comparison
            # ============================================================
            per_vert_err = (soma_target_verts - recon_verts_m).norm(dim=-1)

            excl = hand_layer.excluded_vert_ids
            valid_mask = torch.ones(per_vert_err.shape[1], dtype=torch.bool, device=device)
            if excl is not None and len(excl) > 0:
                valid_mask[excl] = False
            per_vert_err_valid = per_vert_err[:, valid_mask]

            mean_err = per_vert_err_valid.mean().item() * 1000
            max_err = per_vert_err_valid.max().item() * 1000
            median_err = per_vert_err_valid.median().item() * 1000
            per_frame_mean = per_vert_err_valid.mean(dim=1) * 1000

            logger.info(
                f"\n  Round-trip vertex error (mm) [excl {(~valid_mask).sum().item()} boundary verts]:"
            )
            logger.info(f"    mean:   {mean_err:.3f}")
            logger.info(f"    median: {median_err:.3f}")
            logger.info(f"    max:    {max_err:.3f}")
            logger.info(
                f"    per-frame mean: min={per_frame_mean.min():.3f}, max={per_frame_mean.max():.3f}"
            )

            all_stats.append(
                {
                    "hand": hand_type,
                    "seq": seq_idx,
                    "frames": T,
                    "mean_mm": mean_err,
                    "median_mm": median_err,
                    "max_mm": max_err,
                    "per_frame_min_mm": per_frame_mean.min().item(),
                    "per_frame_max_mm": per_frame_mean.max().item(),
                    "inversion_time_s": dt,
                    "ms_per_frame": dt / T * 1000,
                    "fps": T / dt,
                }
            )

            # ============================================================
            # Step 5: Export reconstruction USD
            # ============================================================
            _export_usd(
                f"{out_prefix}_recon.usda", hand_layer, recovered_rotations, root_translation
            )
            logger.info(f"  Reconstruction USD: {out_prefix}_recon.usda")

            # ============================================================
            # Step 6: Render comparison video + GIF with skeleton overlays
            # ============================================================
            if args.no_render:
                continue

            import imageio.v2 as imageio
            from tqdm import tqdm

            mano_verts = soma_target_verts.cpu().numpy()  # (T, 2859, 3)
            mano_joints = (gt_joints - gt_joints[:, 0:1, :]).cpu().numpy()

            soma_verts = recon_verts_m.cpu().numpy()  # (T, 2859, 3)
            soma_joints = recon_joints_m.cpu().numpy()  # (T, 25, 3)

            all_v = np.concatenate([mano_verts[0], soma_verts[0]], axis=0)
            cam_pose = compute_camera_pose(all_v, cam_dist_scale=4.5)
            light_dir = np.array([0.0, -0.3, -1.0])

            renderer = MeshRenderer(image_size=args.image_size, light_intensity=5.0)
            renderer.camera.zfar = 500.0

            mp4_path = f"{out_prefix}_comparison.mp4"
            gif_path = f"{out_prefix}_comparison.gif"
            writer = imageio.get_writer(mp4_path, fps=args.fps)
            gif_frames = []

            for t in tqdm(range(T), desc=f"Rendering {hand_type} seq {seq_idx}"):
                frame = render_overlay_frame(
                    renderer,
                    mano_verts[t],
                    faces_np,
                    mano_joints[t],
                    MANO_JOINT_PARENT_IDS_WITH_FINGERTIPS,
                    soma_verts[t],
                    faces_np,
                    soma_joints[t],
                    parent_ids,
                    cam_pose,
                    light_dir,
                    joint_radius=0.0012,
                    bone_radius=0.0005,
                )
                writer.append_data(frame[..., ::-1])  # RGB -> BGR for mp4
                gif_frames.append(frame)

            writer.close()
            renderer.delete()

            # Save GIF (subsample for file size)
            imageio.mimsave(gif_path, gif_frames, fps=args.gif_fps, loop=0)

            # Save representative still frame
            still_path = f"{out_prefix}_frame0.png"
            save_image(still_path, gif_frames[0])

            logger.info(f"  Video: {mp4_path} ({T} frames)")
            logger.info(f"  GIF:   {gif_path} ({len(gif_frames)} frames @ {args.gif_fps} fps)")
            logger.info(f"  Still: {still_path}")

    # ============================================================
    # Summary stats
    # ============================================================
    logger.info(f"\n{'=' * 60}")
    logger.info("  SUMMARY")
    logger.info(f"{'=' * 60}")
    for s in all_stats:
        logger.info(
            f"  {s['hand']:5s} seq{s['seq']}: {s['frames']:3d}f  "
            f"mean={s['mean_mm']:.2f}mm  median={s['median_mm']:.2f}mm  "
            f"max={s['max_mm']:.2f}mm  "
            f"per-frame=[{s['per_frame_min_mm']:.2f}, {s['per_frame_max_mm']:.2f}]mm  "
            f"{s['ms_per_frame']:.1f}ms/f ({s['fps']:.0f} FPS)"
        )

    np.savez(f"{args.output_dir}/stats.npz", stats=all_stats)
    logger.info(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
