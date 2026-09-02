# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SMPL to SOMA pose converter.

Converts SMPL posed meshes to SOMA skeleton parameters using
PoseInversion.fit() — analytical iterative inverse-LBS Newton-Schulz
refinement, optionally followed by autograd FK optimization.

Usage:
    python -m tools.smpl2soma
    python -m tools.smpl2soma --body-iters 3 --full-iters 1 --batch-size 64
    python -m tools.smpl2soma --autograd-iters 10  # analytical + autograd
    python -m tools.smpl2soma --no-render
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import smplx
import torch

from soma.body import SOMALayer
from soma.fitting.pose_inversion import PoseInversion
from soma.geometry.transforms import matrix_to_rotvec, rotation_6d_to_matrix
from soma.io import add_npz_args, export_soma_usd
from tools.conversion_utils import add_inversion_args, export_soma_npz
from tools.logging_utils import add_logging_args, configure_logging
from tools.vis_pyrender import (
    default_pyopengl_platform,
    render_comparison_video,
    set_pyopengl_platform,
)

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

set_pyopengl_platform(default_pyopengl_platform())
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SMPL to SOMA pose converter.")
    add_inversion_args(parser, batch_size=None, autograd=True)
    parser.add_argument(
        "--subsample", type=int, default=4, help="Frame subsampling factor (default: 4)."
    )
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument("--output-usd", default=None, help="Output .usd/.usda/.usdc with UsdSkel.")
    parser.add_argument("--fps", type=int, default=30, help="Output video/USD FPS (default: 30).")
    add_logging_args(parser)
    add_npz_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    device = args.device
    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"

    # --- Load SMPL animation ---
    data_path = data_root / "SMPL" / "smpl_anim.npy"
    smpl_rot_mats = np.load(data_path, allow_pickle=True).item()

    body_pose_6d = torch.from_numpy(smpl_rot_mats["body_pose_6d"]).float().to(device)
    transl = torch.from_numpy(smpl_rot_mats["transl"]).float().to(device)
    global_orient_6d = torch.from_numpy(smpl_rot_mats["global_orient_6d"]).float().to(device)
    betas = torch.from_numpy(smpl_rot_mats["betas"]).float().to(device)

    body_pose = matrix_to_rotvec(rotation_6d_to_matrix(body_pose_6d))
    global_orient = matrix_to_rotvec(rotation_6d_to_matrix(global_orient_6d))

    seq_len = body_pose.shape[0]
    idx = np.arange(0, seq_len, args.subsample)
    body_pose = body_pose[idx]
    global_orient = global_orient[idx]
    betas = betas[idx]
    transl = transl[idx]
    num_frames = len(idx)
    logger.info(f"Loaded {num_frames} frames (subsampled {args.subsample}x from {seq_len})")

    # --- Set up SMPL model ---
    smpl_model = smplx.create(
        model_type="smpl",
        model_path=data_root / "SMPL" / "SMPL_NEUTRAL.pkl",
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)
    smpl_faces = smpl_model.faces

    # --- Set up SOMA + PoseInversion ---
    soma = SOMALayer(
        data_root,
        identity_model_type="smpl",
        device=device,
        mode="warp",
    )
    inv = PoseInversion(soma, low_lod=True)
    inv.prepare_identity(betas[:1])

    # --- Fused SMPL forward + inversion (chunked to bound memory) ---
    batch_size = args.batch_size or num_frames
    parts = [
        f"body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters}",
    ]
    if args.lie_iters > 0:
        parts.append(f"lie-gn={args.lie_iters}, lambda={args.lie_lambda}")
    if args.autograd_iters > 0:
        parts.append(f"autograd={args.autograd_iters}, lr={args.autograd_lr}")
    if args.batch_size:
        parts.append(f"batch_size={batch_size}")
    logger.info(f"\nInverting ({', '.join(parts)})...")

    # Warmup
    with torch.no_grad():
        warmup_out = smpl_model(
            body_pose=body_pose[:1],
            global_orient=global_orient[:1],
            betas=betas[:1],
            transl=transl[:1],
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

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_rotations = []
    all_root_transl = []
    all_errors = []

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                betas=betas[start:end],
                transl=transl[start:end],
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

    logger.info(f"  Time: {dt:.3f}s ({num_frames / dt:.0f} fps)")
    logger.info(f"  Mean vertex error: {err.mean():.6f} m")
    logger.info(f"  Max vertex error:  {err.max():.6f} m")

    # --- Save NPZ if requested ---
    if args.output_npz:
        export_soma_npz(
            args.output_npz,
            rotations,
            root_transl,
            inv.soma,
            output_unit=args.output_unit,
            keep_root=args.keep_root,
            identity_coeffs=betas[:1],
        )

    if args.output_usd:
        # Prepare full-res soma with fitted identity for USD export
        soma.prepare_identity(betas[:1])
        export_soma_usd(
            args.output_usd,
            soma,
            rotations,
            root_transl,
            fps=float(args.fps),
        )

    if args.no_render:
        return

    # --- Render: re-run SMPL forward + SOMA reconstruct in chunks ---
    _soma = inv.soma
    bs = _soma.batched_skinning
    bind_transforms = _soma._cached_bind_transforms_world
    rest_shape = _soma._cached_rest_shape

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

    tag_parts = [f"analytical_b{args.body_iters}f{args.full_iters}"]
    if args.lie_iters > 0:
        tag_parts.append(f"lie{args.lie_iters}")
    if args.autograd_iters > 0:
        tag_parts.append(f"ag{args.autograd_iters}")
    out_name = "out/smpl2soma_" + "_".join(tag_parts) + ".mp4"
    logger.info(f"\nRendering comparison video -> {out_name}")
    render_comparison_video(
        out_name,
        np.concatenate(smpl_verts_all, axis=0),
        smpl_faces,
        np.concatenate(soma_verts_all, axis=0),
        _soma.faces.cpu().numpy(),
        label_source="SMPL",
        cam_dist_scale=3.0,
    )


if __name__ == "__main__":
    main()
