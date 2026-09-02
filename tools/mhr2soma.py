# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MHR to SOMA pose converter.

Reads SAM 3D Body parquet files containing MHR parameters
(shape_params, model_params) and converts them to SOMA skeleton
parameters using PoseInversion.

SAM 3D Body model_params layout (204 floats):
  [0:3]   = global translation (cm)
  [3:136] = pose parameters (axis-angle)
  [136:204] = body-part scale parameters (68)

Usage:
    python -m tools.mhr2soma --input /path/to/sam_3d_body/coco_train
    python -m tools.mhr2soma --input /path/to/sam_3d_body/coco_train --output-npz out/coco_soma.npz
    python -m tools.mhr2soma --input /path/to/sam_3d_body/coco_train/000000.parquet --max-samples 100
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from soma.body import SOMALayer  # noqa: E402
from soma.fitting.pose_inversion import (  # noqa: E402
    PoseInversion,
    _bind_joint_positions_from_cache,
    _heel_vertex_ids,
)
from soma.fitting.rts_smoothing import RTS_SMOOTHING_PRESETS, smooth_pose  # noqa: E402
from soma.geometry.rig_utils import get_body_part_vertex_ids  # noqa: E402
from soma.io import add_npz_args, export_soma_usd  # noqa: E402
from soma.units import Unit  # noqa: E402
from tools.conversion_utils import add_inversion_args, export_soma_npz  # noqa: E402
from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)


def load_sam_parquet(path, max_samples=None):
    """Load MHR parameters from SAM 3D Body parquet file(s).

    Args:
        path: Path to a single .parquet file or a directory containing them.
        max_samples: Maximum number of samples to load (None = all).

    Returns:
        dict with shape_params (N, 45), model_params (N, 204), and metadata.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {path}")
        dfs = []
        n = 0
        for f in files:
            df = pd.read_parquet(f)
            df = df[df["mhr_valid"]]
            dfs.append(df)
            n += len(df)
            if max_samples is not None and n >= max_samples:
                break
        df = pd.concat(dfs, ignore_index=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df[df["mhr_valid"]]
    else:
        raise ValueError(f"Expected .parquet file or directory, got: {path}")

    if max_samples is not None:
        df = df.iloc[:max_samples]

    shape_params = np.stack(df["shape_params"].values).astype(np.float32)
    model_params = np.stack(df["model_params"].values).astype(np.float32)

    logger.info(f"Loaded {len(df)} MHR samples from {path}")
    return {
        "shape_params": shape_params,
        "model_params": model_params,
        "datasets": df["dataset"].values if "dataset" in df.columns else None,
        "images": df["image"].values if "image" in df.columns else None,
    }


def parse_mhr_model_params(model_params):
    """Parse model_params (N, 204) into translation, pose, and scale.

    Returns:
        translation: (N, 3) in centimeters
        pose_params: (N, 133) axis-angle body pose (excluding global translation)
        scale_params: (N, 68) body-part scales
    """
    translation = model_params[:, :3]
    pose_params = model_params[:, 3:136]
    scale_params = model_params[:, 136:]
    return translation, pose_params, scale_params


def _vertex_ids_for_roots(cache, root_names):
    name_to_idx = {name: idx for idx, name in enumerate(cache["joint_names"])}
    ids = []
    for root_name in root_names:
        root_idx = name_to_idx.get(root_name)
        if root_idx is None:
            continue
        root_vids = get_body_part_vertex_ids(
            cache["skinning_weights"],
            cache["parent_ids"],
            root_idx,
            include_root=True,
        )
        ids.append(
            torch.as_tensor(
                root_vids,
                dtype=torch.long,
                device=cache["skinning_weights"].device,
            )
        )
    if not ids:
        return None
    return torch.unique(torch.cat(ids))


def compute_region_error_metrics(err, inv):
    """Compute coarse body-region error summaries for fit diagnostics."""
    cache = inv._cache
    if cache is None:
        return {}

    feet = _vertex_ids_for_roots(cache, ["LeftFoot", "RightFoot"])
    bind_shape = inv.soma._cached_rest_shape.detach()
    bind_joint_positions = _bind_joint_positions_from_cache(
        cache,
        dtype=bind_shape.dtype,
        device=bind_shape.device,
    )
    heel_vids = _heel_vertex_ids(
        cache["joint_names"],
        cache["parent_ids"],
        cache["skinning_weights"],
        bind_shape,
        bind_joint_positions,
    )
    heels = (
        torch.as_tensor(
            heel_vids,
            dtype=torch.long,
            device=cache["skinning_weights"].device,
        )
        if heel_vids
        else None
    )
    hands = _vertex_ids_for_roots(cache, ["LeftHand", "RightHand"])
    head = _vertex_ids_for_roots(cache, ["Head"])

    regions = {
        "all": torch.arange(err.shape[1], device=err.device),
        "heels": heels,
        "feet": feet,
        "hands": hands,
        "head": head,
    }

    metrics = {}
    for name, vids in regions.items():
        if vids is None or len(vids) == 0:
            continue
        vals = err[:, vids.to(err.device)].reshape(-1)
        metrics[name] = {
            "n": len(vids),
            "mean": float(vals.mean()),
            "p95": float(torch.quantile(vals, 0.95)),
            "max": float(vals.max()),
        }
    return metrics


def print_region_error_summary(metrics, unit_label):
    """Log contact-focused region errors for fit diagnostics."""
    if "feet" in metrics and "hands" in metrics:
        contact_mean = 0.5 * (metrics["feet"]["mean"] + metrics["hands"]["mean"])
        logger.info(f"  Feet/hands equal-region mean: {contact_mean:.6f} {unit_label}")

    logger.info("  Region vertex error:")
    for name in ("heels", "feet", "hands", "head", "all"):
        if name not in metrics:
            continue
        vals = metrics[name]
        logger.info(
            f"    {name:6s} n={vals['n']:5d} "
            f"mean={vals['mean']:.6f} {unit_label}, "
            f"p95={vals['p95']:.6f} {unit_label}, "
            f"max={vals['max']:.6f} {unit_label}"
        )


def print_pose_drift_summary(local_rotation_drift, root_translation_drift, joint_names, unit_label):
    """Log drift from the warm-start pose used by regularized refinement."""
    if local_rotation_drift is None:
        return

    drift_deg = torch.rad2deg(local_rotation_drift)
    if len(joint_names) + 1 == drift_deg.shape[1]:
        drift_eval = drift_deg[:, 1:]
        eval_names = joint_names
    elif joint_names and joint_names[0] == "Root" and drift_deg.shape[1] > 1:
        drift_eval = drift_deg[:, 1:]
        eval_names = joint_names[1:]
    else:
        drift_eval = drift_deg
        eval_names = joint_names

    vals = drift_eval.reshape(-1)
    logger.info("  Local-rotation drift from refinement warm start:")
    logger.info(
        f"    mean={vals.mean():.4f} deg, "
        f"p95={torch.quantile(vals, 0.95):.4f} deg, "
        f"max={vals.max():.4f} deg"
    )

    joint_means = drift_eval.mean(dim=0)
    topk = min(5, joint_means.numel())
    if topk:
        top_vals, top_ids = torch.topk(joint_means, topk)
        top_parts = [
            f"{eval_names[int(idx)]}={float(val):.3f} deg"
            for val, idx in zip(top_vals, top_ids, strict=True)
        ]
        logger.info(f"    largest mean joint drift: {', '.join(top_parts)}")

    if root_translation_drift is not None:
        logger.info(
            f"    root translation drift mean={root_translation_drift.mean():.6f} {unit_label}, "
            f"p95={torch.quantile(root_translation_drift, 0.95):.6f} {unit_label}, "
            f"max={root_translation_drift.max():.6f} {unit_label}"
        )


def parse_pose_prior_weights(spec):
    """Parse named autograd pose-prior profiles or comma-separated weights."""
    if spec is None or spec == "uniform":
        return None
    if spec == "heel_contact":
        weights = {}
        for joint in ("Hips", "Spine1", "Spine2", "Chest"):
            weights[joint] = 0.35
        for side in ("Left", "Right"):
            weights[f"{side}Leg"] = 0.35
            weights[f"{side}Shin"] = 6.0
            weights[f"{side}Foot"] = 8.0
            weights[f"{side}ToeBase"] = 10.0
            weights[f"{side}ToeEnd"] = 10.0
        return weights

    weights = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(
                "Expected --autograd-pose-prior-weights entries as "
                "'JointName=value' or the named profile 'heel_contact'."
            )
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    return weights or None


def get_mhr_posed_vertices(mhr_jit, identity_coeffs, model_params, device, batch_size=64):
    """Run MHR forward pass to get posed vertices.

    Args:
        mhr_jit: TorchScript MHR model.
        identity_coeffs: (N, 45) shape parameters.
        model_params: (N, 204) full model parameters.
        device: torch device.
        batch_size: process in chunks.

    Returns:
        (N, V, 3) posed vertices in centimeters.
    """
    N = identity_coeffs.shape[0]
    face_expr = torch.zeros(1, 72, device=device)
    all_verts = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ic = identity_coeffs[start:end].to(device)
        mp = model_params[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts, _ = mhr_jit(ic, mp, fe)
        all_verts.append(verts.cpu())

    return torch.cat(all_verts, dim=0)


def convert_mhr_to_soma(
    posed_vertices,
    inv,
    body_iters=2,
    finger_iters=0,
    full_iters=1,
    lie_iters=3,
    lie_lambda=1e-1,
    autograd_iters=0,
    autograd_lr=5e-3,
    autograd_translation_lr_scale=1.0,
    autograd_pose_prior=0.0,
    autograd_pose_prior_weights=None,
    autograd_leaf_weight=None,
    leaf_weight=1.0,
    batch_size=64,
):
    """Invert MHR posed vertices to SOMA rotations.

    Args:
        posed_vertices: (N, V, 3) MHR vertices in the same unit as
            the PoseInversion's SOMALayer output_unit.
        inv: PoseInversion instance (already prepared).
        body_iters: analytical body chain iterations.
        finger_iters: analytical finger chain iterations.
        full_iters: analytical full chain iterations.
        lie_iters: Lie algebra Gauss-Newton iterations (0 = disabled).
        lie_lambda: Tikhonov regularisation for Lie-GN (default: 1e-1).
        autograd_iters: Adam optimization steps through FK + LBS.
        autograd_lr: learning rate for autograd Adam.
        autograd_translation_lr_scale: root-translation learning-rate
            multiplier for autograd Adam.
        autograd_pose_prior: local-rotation prior weight for autograd FK.
        autograd_pose_prior_weights: optional per-joint pose-prior multipliers.
        autograd_leaf_weight: optional vertex weighting used only by autograd FK.
        leaf_weight: extremity vertex weight.  Float or dict, e.g.
            ``{"head": 2, "hands": 2, "feet": 5, "heels": 10}``.
        batch_size: process in chunks.

    Returns:
        dict with rotations (N, J, 3, 3), root_translation (N, 3),
        per_vertex_error (N, V).
    """
    return inv.fit(
        posed_vertices.to(inv.soma.device),
        body_iters=body_iters,
        finger_iters=finger_iters,
        full_iters=full_iters,
        lie_iters=lie_iters,
        lie_lambda=lie_lambda,
        autograd_iters=autograd_iters,
        autograd_lr=autograd_lr,
        autograd_translation_lr_scale=autograd_translation_lr_scale,
        autograd_pose_prior=autograd_pose_prior,
        autograd_pose_prior_weights=autograd_pose_prior_weights,
        autograd_leaf_weight=autograd_leaf_weight,
        leaf_weight=leaf_weight,
        batch_size=batch_size,
    )


def main():
    parser = argparse.ArgumentParser(description="MHR to SOMA pose converter.")
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to SAM 3D Body .parquet file or directory of parquet files. "
            "Download parquet data from "
            "https://huggingface.co/datasets/facebook/sam-3d-body-dataset"
        ),
    )
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering.")
    parser.add_argument("--output-usd", default=None, help="Output .usd/.usda/.usdc with UsdSkel.")
    add_npz_args(parser)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    add_inversion_args(parser, autograd=True)
    parser.add_argument(
        "--leaf-weight",
        type=float,
        default=1.0,
        help="Uniform extremity vertex weight (default: 1.0 = no upweight).",
    )
    parser.add_argument(
        "--hand-weight",
        type=float,
        default=None,
        help="Override whole-hand vertex weight (default: same as --leaf-weight).",
    )
    parser.add_argument(
        "--foot-weight",
        type=float,
        default=None,
        help="Override foot vertex weight (default: same as --leaf-weight).",
    )
    parser.add_argument(
        "--heel-weight",
        type=float,
        default=None,
        help="Override rear heel vertex weight without weighting the whole foot.",
    )
    parser.add_argument(
        "--autograd-pose-prior",
        type=float,
        default=0.0,
        help="Local-rotation prior weight for autograd FK refinement.",
    )
    parser.add_argument(
        "--autograd-pose-prior-weights",
        default=None,
        help=(
            "Optional autograd pose-prior joint weights. Use 'heel_contact' "
            "or comma-separated JointName=value entries. Values >1 stiffen a "
            "joint; values <1 let it move more."
        ),
    )
    parser.add_argument(
        "--autograd-hand-weight",
        type=float,
        default=None,
        help="Whole-hand vertex weight used only by autograd FK.",
    )
    parser.add_argument(
        "--autograd-foot-weight",
        type=float,
        default=None,
        help="Foot vertex weight used only by autograd FK.",
    )
    parser.add_argument(
        "--autograd-heel-weight",
        type=float,
        default=None,
        help="Rear heel vertex weight used only by autograd FK.",
    )
    parser.add_argument("--fps", type=int, default=4, help="Video frame rate (default: 4).")
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply reusable SO(3) RTS smoothing to SOMA rotations and root translation.",
    )
    parser.add_argument(
        "--smooth-preset",
        choices=tuple(RTS_SMOOTHING_PRESETS),
        default="default",
        help="RTS smoothing preset (default: default).",
    )
    parser.add_argument(
        "--smooth-fps",
        type=float,
        default=None,
        help="Source frame rate for RTS smoothing dynamics (default: --fps).",
    )
    parser.add_argument(
        "--smooth-include-limbs",
        action="store_true",
        help="Apply faster hand gains to limb joints as well as hand/finger joints.",
    )
    parser.add_argument(
        "--smooth-no-hand-gains",
        action="store_true",
        help="Use body RTS gains for every joint.",
    )
    parser.add_argument(
        "--no-smooth-root",
        action="store_true",
        help="Keep root translation unchanged when --smooth is enabled.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    data_root = Path(args.data_root) if args.data_root else repo_root / "assets"
    device = args.device

    # --- Load data ---
    sam_data = load_sam_parquet(args.input, max_samples=args.max_samples)
    shape_params = sam_data["shape_params"]
    model_params_raw = sam_data["model_params"]
    N = shape_params.shape[0]

    # Extract MHR's 6 flexible bone-length parameters (body indices 124-129,
    # model_params indices 130-135). These modify skeleton proportions (spine,
    # neck, shoulder, arm, hip, leg length) and are identity-like, but SAM 3D
    # Body's fitter lets them vary per frame. We pass them through to SOMA's
    # identity model via kwargs so the rest shape matches.
    bone_length_flexibles = model_params_raw[:, 130:136].copy()

    translation_cm, pose_params, scale_params = parse_mhr_model_params(model_params_raw)
    logger.info(f"  Samples: {N}")
    logger.info(
        f"  Translation range (cm): [{translation_cm.min():.1f}, {translation_cm.max():.1f}]"
    )
    logger.info(f"  Scale params range: [{scale_params.min():.3f}, {scale_params.max():.3f}]")

    # --- Set up MHR model ---
    import trimesh

    mhr_faces = trimesh.load(
        data_root / "MHR" / "base_body_lod1.obj", maintain_order=True, process=False
    ).faces
    mhr_jit = torch.jit.load(data_root / "MHR" / "mhr_model_lod1.pt", map_location=device)
    identity_coeffs_t = torch.from_numpy(shape_params).float()
    model_params_t = torch.from_numpy(model_params_raw).float()
    face_expr = torch.zeros(1, 72, device=device)

    # --- Set up SOMA + PoseInversion ---
    # Use output_unit=CENTIMETERS so the internal rest shape matches MHR's
    # native unit — no ad-hoc unit conversion needed.
    logger.info("\nInitializing SOMA layer...")
    soma = SOMALayer(
        data_root,
        identity_model_type="mhr",
        device=device,
        mode="warp",
        output_unit=Unit.CENTIMETERS,
    )

    all_ic = torch.from_numpy(shape_params).float().to(device)
    all_sp = torch.from_numpy(scale_params).float().to(device)
    all_bl = torch.from_numpy(bone_length_flexibles).float().to(device)

    # Use low LOD for inversion (faster, negligible accuracy loss),
    # high LOD (soma) for rendering/evaluation.
    inv = PoseInversion(soma, low_lod=True)

    # Build leaf_weight: uniform or per-group if an override is set.
    if any(w is not None for w in (args.hand_weight, args.foot_weight, args.heel_weight)):
        leaf_weight = {
            "head": args.leaf_weight,
            "hands": args.leaf_weight if args.hand_weight is None else args.hand_weight,
            "feet": args.leaf_weight if args.foot_weight is None else args.foot_weight,
        }
        if args.heel_weight is not None:
            leaf_weight["heels"] = args.heel_weight
    else:
        leaf_weight = args.leaf_weight

    autograd_leaf_weight = None
    if any(
        w is not None
        for w in (args.autograd_hand_weight, args.autograd_foot_weight, args.autograd_heel_weight)
    ):
        autograd_leaf_weight = {
            "head": 1.0,
            "hands": 1.0 if args.autograd_hand_weight is None else args.autograd_hand_weight,
            "feet": 1.0 if args.autograd_foot_weight is None else args.autograd_foot_weight,
        }
        if args.autograd_heel_weight is not None:
            autograd_leaf_weight["heels"] = args.autograd_heel_weight

    autograd_pose_prior_weights = parse_pose_prior_weights(args.autograd_pose_prior_weights)

    # --- Fused MHR forward + inversion (chunked to bound memory) ---
    import time

    parts = []
    if args.body_iters > 0 or args.finger_iters > 0 or args.full_iters > 0:
        parts.append(
            f"analytical (body={args.body_iters}, finger={args.finger_iters}, full={args.full_iters})"
        )
    if args.lie_iters > 0:
        parts.append(f"lie-gn ({args.lie_iters} iters, lambda={args.lie_lambda})")
    if args.autograd_iters > 0:
        parts.append(
            f"autograd FK ({args.autograd_iters} iters, lr={args.autograd_lr}, "
            f"translation_lr_scale={args.autograd_translation_lr_scale}, "
            f"pose_prior={args.autograd_pose_prior})"
        )
    method_desc = " + ".join(parts) if parts else "none"
    if leaf_weight != 1.0:
        method_desc += f", leaf_weight={leaf_weight}"
    if autograd_leaf_weight is not None:
        method_desc += f", autograd_leaf_weight={autograd_leaf_weight}"
    if autograd_pose_prior_weights is not None:
        method_desc += f", autograd_pose_prior_weights={autograd_pose_prior_weights}"

    batch_size = args.batch_size
    logger.info(f"\nInverting {N} samples with {method_desc}...")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_rotations = []
    all_root_transl = []
    all_errors = []
    all_rotation_drift = []
    all_root_translation_drift = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Prepare identity for this chunk (includes per-frame bone-length flexibles)
        chunk_kwargs = {"bone_length_flexibles": all_bl[start:end]}
        inv.prepare_identity(all_ic[start:end], all_sp[start:end], kwargs=chunk_kwargs)

        # MHR forward pass for this chunk
        ic = identity_coeffs_t[start:end].to(device)
        mp = model_params_t[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts_cm, _ = mhr_jit(ic, mp, fe)

        # Invert this chunk (no further chunking — already chunk-sized)
        result = convert_mhr_to_soma(
            verts_cm,
            inv,
            body_iters=args.body_iters,
            finger_iters=args.finger_iters,
            full_iters=args.full_iters,
            lie_iters=args.lie_iters,
            lie_lambda=args.lie_lambda,
            autograd_iters=args.autograd_iters,
            autograd_lr=args.autograd_lr,
            autograd_translation_lr_scale=args.autograd_translation_lr_scale,
            autograd_pose_prior=args.autograd_pose_prior,
            autograd_pose_prior_weights=autograd_pose_prior_weights,
            autograd_leaf_weight=autograd_leaf_weight,
            leaf_weight=leaf_weight,
            batch_size=None,
        )

        all_rotations.append(result["rotations"].cpu())
        all_root_transl.append(result["root_translation"].cpu())
        all_errors.append(result["per_vertex_error"].cpu())
        if "local_rotation_drift" in result:
            all_rotation_drift.append(result["local_rotation_drift"].cpu())
        if "root_translation_drift" in result:
            all_root_translation_drift.append(result["root_translation_drift"].cpu())

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rotations = torch.cat(all_rotations, dim=0)
    root_transl = torch.cat(all_root_transl, dim=0)
    err = torch.cat(all_errors, dim=0)
    rotation_drift = torch.cat(all_rotation_drift, dim=0) if all_rotation_drift else None
    root_translation_drift = (
        torch.cat(all_root_translation_drift, dim=0) if all_root_translation_drift else None
    )

    unit_label = "cm" if soma.output_unit == Unit.CENTIMETERS else "m"
    logger.info(f"  Inversion time: {dt:.2f}s ({N / dt:.0f} FPS)")
    logger.info(f"  Mean vertex error: {err.mean():.6f} {unit_label}")
    logger.info(f"  Max vertex error:  {err.max():.6f} {unit_label}")
    logger.info(f"  Median vertex error: {err.median():.6f} {unit_label}")
    region_metrics = compute_region_error_metrics(err, inv)
    print_region_error_summary(region_metrics, unit_label)
    print_pose_drift_summary(
        rotation_drift,
        root_translation_drift,
        list(soma.rig_data["joint_names"]),
        unit_label,
    )

    if args.smooth:
        smooth_config = replace(
            RTS_SMOOTHING_PRESETS[args.smooth_preset],
            fps=float(args.smooth_fps if args.smooth_fps is not None else args.fps),
            smooth_root_translation=not args.no_smooth_root,
        )
        logger.info(
            "\nApplying SO(3) RTS smoothing "
            f"(preset={args.smooth_preset}, fps={smooth_config.fps:g})..."
        )
        rotations, root_transl = smooth_pose(
            rotations,
            root_transl,
            soma_layer=soma,
            config=smooth_config,
            rotation_convention="absolute",
            output_rotation_convention="absolute",
            use_hand_gains=not args.smooth_no_hand_gains,
            include_limb_gains=args.smooth_include_limbs,
        )

    # --- Save output ---
    if args.output_npz:
        extra_arrays = {"bone_length_flexibles": bone_length_flexibles}
        if args.smooth:
            extra_arrays.update(
                {
                    "rts_smoothing": np.array("so3_error_state"),
                    "rts_smoothing_preset": np.array(args.smooth_preset),
                    "rts_smoothing_fps": np.float32(smooth_config.fps),
                    "rts_smoothing_root_enabled": np.bool_(not args.no_smooth_root),
                }
            )
        if rotation_drift is not None:
            extra_arrays["local_rotation_drift_deg"] = torch.rad2deg(rotation_drift).numpy()
        export_soma_npz(
            args.output_npz,
            rotations,
            root_transl,
            soma,
            output_unit=args.output_unit,
            keep_root=args.keep_root,
            identity_coeffs=shape_params,
            scale_params=scale_params,
            extra_arrays=extra_arrays,
        )

    if args.output_usd:
        # USD uses a single bind pose, but SAM 3D Body has per-sample identity
        # (shape + scale + bone-length params vary per frame). Warn if they differ.
        ic_np = all_ic.cpu().numpy()
        sp_np = all_sp.cpu().numpy()
        bl_np = all_bl.cpu().numpy()
        identity_varies = (
            np.std(ic_np, axis=0).max() > 1e-4
            or np.std(sp_np, axis=0).max() > 1e-4
            or np.std(bl_np, axis=0).max() > 1e-4
        )
        if identity_varies:
            import warnings

            warnings.warn(
                "Identity parameters vary across samples. USD export uses the "
                "first sample's shape as a fixed bind pose; skinning will be "
                "approximate for samples with different body shapes.",
                stacklevel=1,
            )

        first_kwargs = {"bone_length_flexibles": all_bl[:1]}
        soma.prepare_identity(all_ic[:1], all_sp[:1], kwargs=first_kwargs)
        export_soma_usd(
            args.output_usd,
            soma,
            rotations,
            root_transl,
            fps=float(args.fps),
            unit="centimeters",
        )

    if args.no_render:
        return

    # --- Render ---
    from tools.vis_pyrender import (  # noqa: E402
        default_pyopengl_platform,
        render_comparison_video,
        set_pyopengl_platform,
    )

    set_pyopengl_platform(default_pyopengl_platform())

    cm_to_m = Unit.CENTIMETERS.meters_per_unit
    soma_faces = soma.faces.cpu().numpy()

    # Re-run MHR forward + SOMA reconstruct in chunks for rendering
    # (avoids materializing all vertices at once).
    mhr_verts_all = []
    eval_verts_all = []
    bs = soma.batched_skinning

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Prepare identity for this chunk (high LOD, with bone-length flexibles)
        chunk_kwargs = {"bone_length_flexibles": all_bl[start:end]}
        soma.prepare_identity(all_ic[start:end], all_sp[start:end], kwargs=chunk_kwargs)
        chunk_bind = soma._cached_bind_transforms_world
        chunk_rest = soma._cached_rest_shape
        bs.rebind(chunk_bind, chunk_rest)

        # Re-run MHR forward for this chunk
        ic = identity_coeffs_t[start:end].to(device)
        mp = model_params_t[start:end].to(device)
        fe = face_expr.expand(end - start, -1)
        with torch.no_grad():
            verts_cm, _ = mhr_jit(ic, mp, fe)
        mhr_verts_all.append((verts_cm * cm_to_m).cpu().numpy())

        # Reconstruct via batched_skinning.pose() at high LOD
        chunk_rot = rotations[start:end].to(device)
        chunk_transl = root_transl[start:end].to(device)
        with torch.no_grad():
            eval_v, _ = bs.pose(chunk_rot, chunk_transl, absolute_pose=True, return_transforms=True)
        eval_verts_all.append((eval_v.detach() * cm_to_m).cpu().numpy())

    mhr_verts_m = np.concatenate(mhr_verts_all, axis=0)
    eval_verts = np.concatenate(eval_verts_all, axis=0)

    parts_tag = []
    if args.body_iters > 0 or args.finger_iters > 0 or args.full_iters > 0:
        parts_tag.append("analytical")
    if args.lie_iters > 0:
        parts_tag.append(f"lie{args.lie_iters}")
    if args.autograd_iters > 0:
        parts_tag.append(f"autograd{args.autograd_iters}")
    method_tag = "_".join(parts_tag) if parts_tag else "none"
    out_name = f"out/mhr2soma_eval_{method_tag}.mp4"
    logger.info(f"Rendering MHR (with correctives) vs SOMA (no correctives) -> {out_name}")
    render_comparison_video(
        out_name,
        mhr_verts_m,
        mhr_faces,
        eval_verts,
        soma_faces,
        center=True,
        cam_dist_scale=5.0,
        fps=args.fps,
        label_source="MHR",
        label_soma=f"SOMA {method_tag}",
    )


if __name__ == "__main__":
    main()
