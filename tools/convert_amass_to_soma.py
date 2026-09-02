# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert AMASS dataset to SOMA format.

AMASS dataset (https://amass.is.tue.mpg.de/) contains motion capture data
in SMPL format. This script converts AMASS sequences to SOMA pose parameters
using PoseInversion.fit() — the same analytical inverse-LBS pipeline as
smpl2soma.py.

Typical AMASS npz file structure:
- poses: (T, 156) - Full body pose in axis-angle (52 joints x 3)
- trans: (T, 3) - Root translation
- betas: (10,) or (16,) - Shape parameters
- gender: 'male', 'female', or 'neutral'
- mocap_framerate: Frame rate of original capture
- dmpls: (T, 8) - Optional DMPLs parameters (for soft tissue dynamics)

Usage:
    # Single file
    python -m tools.convert_amass_to_soma --input <amass.npz> --output-npz out/test.npz
    python -m tools.convert_amass_to_soma --input <amass.npz> --output-npz out/test.npz --no-render

    # Batch convert entire dataset (mirrors folder structure)
    python -m tools.convert_amass_to_soma --input-dir /path/to/amass/ --output-dir out/amass_soma/
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import smplx
import torch
from tqdm import tqdm

from soma.body import SOMALayer
from soma.fitting.pose_inversion import PoseInversion
from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec
from soma.io import add_npz_args, save_soma_npz
from soma.units import Unit
from tools.logging_utils import add_logging_args, configure_logging
from tools.vis_pyrender import (
    default_pyopengl_platform,
    look_at,
    render_comparison_video,
    set_pyopengl_platform,
)

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

set_pyopengl_platform(default_pyopengl_platform())
logger = logging.getLogger(__name__)


def load_amass_npz(npz_path):
    """
    Load AMASS npz file and extract relevant data.

    Args:
        npz_path: Path to AMASS .npz file

    Returns:
        dict with:
            - poses: (T, 72) full SMPL pose (24 joints x 3 axis-angle)
            - betas: (10,) shape parameters
            - trans: (T, 3) root translation
            - gender: string
            - fps: framerate
    """
    data = np.load(npz_path, allow_pickle=True)

    logger.info(f"\nLoading AMASS data from {npz_path}")
    keys = list(data.keys())
    logger.info(f"  Available keys: {keys}")

    for key in keys:
        if hasattr(data[key], "shape"):
            logger.info(f"    {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            logger.info(f"    {key}: {type(data[key])} = {data[key]}")

    # AMASS poses[:, :66] = 22 joints x 3 (root + 21 body joints)
    # Pad to 72 = 24 joints x 3 to match SMPL's expected shape
    poses_full = data["poses"]
    poses_amass = poses_full[:, :66]
    T = poses_amass.shape[0]
    poses_smpl = np.zeros((T, 72))
    poses_smpl[:, :66] = poses_amass
    trans = data["trans"] if "trans" in data else np.zeros((T, 3))
    betas = data["betas"][:10] if "betas" in data else np.zeros(10)
    gender = str(data["gender"]) if "gender" in data else "neutral"
    fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 30.0

    logger.info(f"  Sequence length: {T} frames")
    logger.info(f"  Shape (betas): {betas.shape}")
    logger.info(f"  Gender: {gender}")
    logger.info(f"  FPS: {fps}")

    return {"poses": poses_smpl, "betas": betas, "trans": trans, "gender": gender, "fps": fps}


def convert_amass_sequence(amass_data, args, smpl_model, inv, device="cuda"):
    """
    Convert AMASS sequence to SOMA format using PoseInversion.

    Args:
        amass_data: dict from load_amass_npz
        args: parsed command-line arguments
        smpl_model: pre-created SMPL model
        inv: pre-created PoseInversion instance
        device: torch device

    Returns:
        dict with:
            - rotations: (T, J, 3, 3) absolute rotation matrices
            - root_transl: (T, 3) root translation
            - betas: (T, 10) shape parameters on device
            - body_pose: (T, 69) body pose axis-angle on device
            - global_orient: (T, 3) global orient axis-angle on device
            - transl: (T, 3) translation on device
    """
    poses_smpl = torch.from_numpy(amass_data["poses"]).float().to(device)
    betas = torch.from_numpy(amass_data["betas"]).float().to(device)
    trans = torch.from_numpy(amass_data["trans"]).float().to(device)

    T = poses_smpl.shape[0]
    global_orient = poses_smpl[:, :3]  # (T, 3) axis-angle
    body_pose = poses_smpl[:, 3:]  # (T, 69) axis-angle
    betas_expanded = betas.unsqueeze(0).expand(T, -1)  # (T, 10)

    batch_size = args.batch_size or T

    # Prepare identity for this sequence's betas
    inv.prepare_identity(betas_expanded[:1])

    # Warmup
    with torch.no_grad():
        warmup_out = smpl_model(
            body_pose=body_pose[:1],
            global_orient=global_orient[:1],
            betas=betas_expanded[:1],
            transl=trans[:1],
        )
    inv.fit(
        warmup_out.vertices,
        body_iters=args.body_iters,
        finger_iters=args.finger_iters,
        full_iters=args.full_iters,
        lie_iters=args.lie_iters,
        lie_lambda=args.lie_lambda,
        autograd_iters=args.autograd_iters,
        autograd_lr=args.autograd_lr,
    )

    # Inversion
    parts = [
        f"body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters}",
    ]
    if args.lie_iters > 0:
        parts.append(f"lie-gn={args.lie_iters}, lambda={args.lie_lambda}")
    if args.autograd_iters > 0:
        parts.append(f"autograd={args.autograd_iters}, lr={args.autograd_lr}")
    if args.batch_size:
        parts.append(f"batch_size={batch_size}")
    logger.info(f"\nInverting {T} frames ({', '.join(parts)})...")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_rotations = []
    all_root_transl = []
    all_errors = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                betas=betas_expanded[start:end],
                transl=trans[start:end],
            )
        result = inv.fit(
            smpl_out.vertices,
            body_iters=args.body_iters,
            finger_iters=args.finger_iters,
            full_iters=args.full_iters,
            lie_iters=args.lie_iters,
            lie_lambda=args.lie_lambda,
            autograd_iters=args.autograd_iters,
            autograd_lr=args.autograd_lr,
        )
        all_rotations.append(result["rotations"].cpu())
        all_root_transl.append(result["root_translation"].cpu())
        all_errors.append(result["per_vertex_error"].cpu())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(all_rotations, dim=0)
    root_transl = torch.cat(all_root_transl, dim=0)
    err = torch.cat(all_errors, dim=0)

    logger.info(f"  Time: {dt:.3f}s ({T / dt:.0f} fps)")
    logger.info(f"  Mean vertex error: {err.mean():.6f} m")
    logger.info(f"  Max vertex error:  {err.max():.6f} m")

    return {
        "rotations": rotations,
        "root_transl": root_transl,
        "betas": betas_expanded,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": trans,
        "per_vertex_error": err,
    }


def _save_conversion(conv, inv, args, output_path):
    """Save conversion result to a SOMA .npz file."""
    _soma = inv.soma
    orient_device = _soma._t_pose_orient.device
    rotations = conv["rotations"].to(orient_device)
    root_transl = conv["root_transl"]
    betas = conv["betas"]

    # Convert absolute -> relative, matrix -> rotvec
    rel_rotations = remove_joint_orient_local(
        rotations, _soma._t_pose_orient, _soma._t_pose_orient_parent_T
    )
    poses_rotvec = matrix_to_rotvec(rel_rotations.reshape(-1, 3, 3)).reshape(
        rotations.shape[0], rotations.shape[1], 3
    )

    save_transl = root_transl.clone()
    target_unit = Unit.from_name(args.output_unit)
    unit_scale = _soma.output_unit.meters_per_unit / target_unit.meters_per_unit
    if unit_scale != 1.0:
        save_transl = save_transl * unit_scale

    save_soma_npz(
        output_path,
        poses_rotvec,
        save_transl,
        joint_names=list(_soma.rig_data["joint_names"]),
        identity_model_type=_soma.identity_model_type,
        identity_coeffs=betas[:1],
        joint_orient=_soma._t_pose_orient,
        unit=args.output_unit,
        keep_root=args.keep_root,
        extra_arrays={"per_vertex_error": conv["per_vertex_error"]},
    )


def main():
    parser = argparse.ArgumentParser(description="Convert AMASS dataset to SOMA format.")

    # Input: single file or directory (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", help="Path to a single AMASS .npz file (e.g., 01_01_poses.npz)"
    )
    input_group.add_argument(
        "--input-dir", help="Path to AMASS root directory for batch conversion"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory for batch mode (mirrors input folder structure).",
    )
    parser.add_argument("--data-root", default=None, help="Path to SOMA assets (default: ./assets)")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process frames in chunks of this size (default: all at once).",
    )
    parser.add_argument(
        "--body-iters", type=int, default=2, help="Analytical body iterations (default: 2)."
    )
    parser.add_argument(
        "--finger-iters", type=int, default=0, help="Analytical finger iterations (default: 0)."
    )
    parser.add_argument(
        "--full-iters", type=int, default=1, help="Analytical full iterations (default: 1)."
    )
    parser.add_argument(
        "--lie-iters",
        type=int,
        default=3,
        help="Lie algebra Gauss-Newton iterations (default: 3).",
    )
    parser.add_argument(
        "--lie-lambda",
        type=float,
        default=1e-1,
        help="Tikhonov regularisation for Lie-GN (default: 1e-1).",
    )
    parser.add_argument(
        "--autograd-iters",
        type=int,
        default=0,
        help="Autograd FK refinement iterations after analytical solve (default: 0 = off).",
    )
    parser.add_argument(
        "--autograd-lr",
        type=float,
        default=5e-3,
        help="Learning rate for autograd FK (default: 5e-3).",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle file order in batch mode (useful for multi-worker parallelism).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files whose output already exists (default: on). Use --no-skip-existing to force.",
    )
    add_logging_args(parser)
    add_npz_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    # Validate argument combinations
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")

    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create models once
    smpl_model = smplx.create(
        model_type="smpl",
        model_path=data_root / "SMPL" / "SMPL_NEUTRAL.pkl",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)

    soma = SOMALayer(
        data_root,
        identity_model_type="smpl",
        device=device,
        mode="warp",
    )
    inv = PoseInversion(soma, low_lod=True)

    if args.input_dir:
        # --- Batch mode ---
        input_root = Path(args.input_dir)
        output_root = Path(args.output_dir)
        npz_files = sorted(input_root.rglob("*.npz"))

        if not npz_files:
            logger.warning(f"No .npz files found under {input_root}")
            return

        # Pre-filter for skip-existing so the progress bar reflects real work
        num_skipped = 0
        if args.skip_existing:
            work_items = []
            for npz_path in npz_files:
                rel_path = npz_path.relative_to(input_root)
                out_path = output_root / rel_path
                if out_path.exists():
                    num_skipped += 1
                else:
                    work_items.append((npz_path, rel_path, out_path))
        else:
            work_items = [
                (p, p.relative_to(input_root), output_root / p.relative_to(input_root))
                for p in npz_files
            ]

        if args.shuffle:
            random.shuffle(work_items)

        logger.info(
            f"Found {len(npz_files)} .npz files under {input_root}"
            f" ({num_skipped} already exist, {len(work_items)} to convert)"
        )

        num_failed = 0
        num_converted = 0

        pbar = tqdm(work_items, unit="file", dynamic_ncols=True)
        for npz_path, rel_path, out_path in pbar:
            pbar.set_postfix_str(str(rel_path), refresh=False)
            # Re-check at processing time (another worker may have finished it)
            if args.skip_existing and out_path.exists():
                num_skipped += 1
                continue
            try:
                amass_data = load_amass_npz(npz_path)

                conv = convert_amass_sequence(amass_data, args, smpl_model, inv, device=device)
                _save_conversion(conv, inv, args, str(out_path))
                num_converted += 1
            except Exception as e:
                logger.warning(f"FAILED {rel_path}: {e}")
                num_failed += 1

        logger.info(
            f"\nBatch complete: {num_converted} converted, {num_skipped} skipped, "
            f"{num_failed} failed"
        )

    else:
        # --- Single file mode ---
        amass_data = load_amass_npz(args.input)
        conv = convert_amass_sequence(amass_data, args, smpl_model, inv, device=device)

        if args.output_npz:
            _save_conversion(conv, inv, args, args.output_npz)

        if args.no_render:
            return

        # --- Render comparison video ---
        rotations = conv["rotations"]
        root_transl = conv["root_transl"]
        betas = conv["betas"]
        body_pose = conv["body_pose"]
        global_orient = conv["global_orient"]
        transl = conv["transl"]

        _soma = inv.soma
        bs = _soma.batched_skinning
        bind_transforms = _soma._cached_bind_transforms_world
        rest_shape = _soma._cached_rest_shape

        num_frames = rotations.shape[0]
        batch_size = args.batch_size or num_frames

        smpl_verts_all = []
        soma_verts_all = []

        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            with torch.no_grad():
                smpl_out = smpl_model(
                    body_pose=body_pose[start:end],
                    global_orient=global_orient[start:end],
                    betas=betas[start:end],
                    transl=transl[start:end],
                )
            smpl_verts_all.append(smpl_out.vertices.cpu().numpy())

            chunk_bind = bind_transforms.expand(end - start, -1, -1, -1)
            chunk_rest = rest_shape.expand(end - start, -1, -1)
            bs.rebind(chunk_bind, chunk_rest)
            with torch.no_grad():
                sv, _ = bs.pose(
                    rotations[start:end].to(device),
                    root_transl[start:end].to(device),
                    absolute_pose=True,
                    return_transforms=True,
                )
            soma_verts_all.append(sv.detach().cpu().numpy())

        output_video = (
            args.output_npz.replace(".npz", ".mp4") if args.output_npz else "out/amass2soma.mp4"
        )
        logger.info("\nRendering comparison video...")
        # Per-frame centroid from source mesh; apply to both so overlay stays aligned
        smpl_verts_all = np.concatenate(smpl_verts_all, axis=0)
        soma_verts_all = np.concatenate(soma_verts_all, axis=0)
        centroids = smpl_verts_all.mean(axis=1, keepdims=True)  # (N, 1, 3)
        smpl_verts_all = smpl_verts_all - centroids
        soma_verts_all = soma_verts_all - centroids

        vmin = smpl_verts_all[0].min(axis=0)
        vmax = smpl_verts_all[0].max(axis=0)
        center = (vmin + vmax) * 0.5
        extent = np.linalg.norm(vmax - vmin) + 1e-6
        eye = center - np.array([0.0, max(extent * 3.0, 1.0), 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cam_pose = look_at(eye, center, up)
        light_dir = np.array([0.0, 0.5, -0.5])

        render_comparison_video(
            output_video,
            smpl_verts_all,
            smpl_model.faces,
            soma_verts_all,
            _soma.faces.cpu().numpy(),
            label_source="SMPL (AMASS)",
            cam_pose=cam_pose,
            light_dir=light_dir,
            fps=int(amass_data["fps"]),
        )


if __name__ == "__main__":
    main()
