# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for *2soma conversion tools.

Used by hand/mano2soma.py, mhr2soma.py, and smpl2soma.py to avoid duplicating
argument parsing, NPZ export prep, and USD export logic.
"""

from soma.geometry.rig_utils import remove_joint_orient_local
from soma.geometry.transforms import matrix_to_rotvec
from soma.io import save_soma_npz
from soma.units import Unit

# ---------------------------------------------------------------------------
# Shared argparse helpers
# ---------------------------------------------------------------------------


def add_inversion_args(
    parser,
    *,
    body_iters=2,
    finger_iters=0,
    full_iters=1,
    lie_iters=3,
    lie_lambda=1e-1,
    batch_size=64,
    autograd=False,
):
    """Add standard PoseInversion parameters to an ArgumentParser.

    Defaults are model-specific -- callers override via keyword arguments.
    ``autograd=True`` adds --autograd-iters and --autograd-lr.
    """
    parser.add_argument(
        "--body-iters",
        type=int,
        default=body_iters,
        help=f"Analytical body iterations (default: {body_iters}).",
    )
    parser.add_argument(
        "--finger-iters",
        type=int,
        default=finger_iters,
        help=f"Analytical finger iterations (default: {finger_iters}).",
    )
    parser.add_argument(
        "--full-iters",
        type=int,
        default=full_iters,
        help=f"Analytical full iterations (default: {full_iters}).",
    )
    parser.add_argument(
        "--lie-iters",
        type=int,
        default=lie_iters,
        help=f"Lie algebra Gauss-Newton iterations (default: {lie_iters}).",
    )
    parser.add_argument(
        "--lie-lambda",
        type=float,
        default=lie_lambda,
        help=f"Tikhonov regularisation for Lie-GN (default: {lie_lambda}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help=f"Batch size for processing (default: {batch_size}).",
    )
    if autograd:
        parser.add_argument(
            "--autograd-iters",
            type=int,
            default=0,
            help="Autograd FK optimization steps after analytical solve (default: 0 = off).",
        )
        parser.add_argument(
            "--autograd-lr",
            type=float,
            default=5e-3,
            help="Autograd learning rate (default: 5e-3).",
        )
        parser.add_argument(
            "--autograd-translation-lr-scale",
            type=float,
            default=1.0,
            help=("Multiplier for the autograd root-translation learning rate (default: 1)."),
        )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda).")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to SOMA assets (default: <repo>/assets).",
    )


def add_hand_inversion_args(
    parser,
    *,
    bcd_iters=1,
    lie_iters=3,
    lie_lambda=1e-1,
    batch_size=64,
):
    """Add hand-specific PoseInversion parameters to an ArgumentParser.

    For hand-only pose inversion, the body/finger/full BCD split is
    unnecessary -- a single ``--bcd-iters`` controls the analytical solve
    (mapped to ``full_iters`` internally, with ``body_iters=0`` and
    ``finger_iters=0``).
    """
    parser.add_argument(
        "--bcd-iters",
        type=int,
        default=bcd_iters,
        help=f"BCD analytical iterations (default: {bcd_iters}).",
    )
    parser.add_argument(
        "--lie-iters",
        type=int,
        default=lie_iters,
        help=f"Lie algebra Gauss-Newton iterations (default: {lie_iters}).",
    )
    parser.add_argument(
        "--lie-lambda",
        type=float,
        default=lie_lambda,
        help=f"Tikhonov regularisation for Lie-GN (default: {lie_lambda}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help=f"Batch size for processing (default: {batch_size}).",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda).")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to SOMA assets (default: <repo>/assets).",
    )


# ---------------------------------------------------------------------------
# NPZ export
# ---------------------------------------------------------------------------


def resolve_joint_orient(soma, num_joints):
    """Joint-orient tensors matching ``num_joints`` rotations from ``soma``.

    Procedural layers skin an expanded target skeleton (110 joints) while
    ``PoseInversion.fit`` and ``SOMALayer.pose`` operate on the public skeleton
    (78 joints).  ``soma._t_pose_orient`` is sized to the target skeleton, so
    applying it to public rotations raises a size mismatch (110 vs 78).

    Returns:
        ``(orient, orient_parent_T, joint_names)`` sized to ``num_joints``.
    """
    orient = getattr(soma, "_t_pose_orient", None)
    if orient is not None and orient.shape[0] == num_joints:
        return orient, soma._t_pose_orient_parent_T, list(soma.rig_data["joint_names"])
    public_names = list(getattr(soma, "public_joint_names", ()))
    if len(public_names) == num_joints and hasattr(soma, "public_joint_orient"):
        orient, orient_parent_T = soma.public_joint_orient()
        return orient, orient_parent_T, public_names
    raise ValueError(
        "Cannot resolve joint orient for rotations with "
        f"{num_joints} joints: layer target skeleton has "
        f"{None if orient is None else orient.shape[0]} joints and public skeleton has "
        f"{len(public_names)}."
    )


def export_soma_npz(
    output_path,
    rotations,
    root_transl,
    soma,
    *,
    output_unit,
    keep_root=False,
    identity_coeffs=None,
    scale_params=None,
    extra_arrays=None,
):
    """Layer-oriented NPZ export, parallel to :func:`~soma.io.export_soma_usd`.

    Converts absolute rotation matrices -> relative (joint orient removed)
    -> axis-angle rotvec, applies unit conversion, and delegates to
    :func:`~soma.io.save_soma_npz` (the data-oriented primitive).

    Args:
        output_path: destination .npz path.
        rotations: (N, J, 3, 3) absolute rotation matrices.
        root_transl: (N, 3) root translation in soma.output_unit.
        soma: SOMALayer instance (provides joint orient, names, etc.).
        output_unit: target unit name string (e.g. "meters", "centimeters").
        keep_root: include virtual root joint in output.
        identity_coeffs: (N, K) or (1, K) identity coefficients to store.
        scale_params: (N, S) scale parameters to store (MHR only).
        extra_arrays: dict of additional arrays to store in the NPZ.
    """
    orient, orient_parent_T, joint_names = resolve_joint_orient(soma, rotations.shape[1])
    orient = orient.to(device=rotations.device, dtype=rotations.dtype)
    orient_parent_T = orient_parent_T.to(device=rotations.device, dtype=rotations.dtype)
    rel_rotations = remove_joint_orient_local(rotations, orient, orient_parent_T)
    poses_rotvec = matrix_to_rotvec(rel_rotations.reshape(-1, 3, 3)).reshape(
        rotations.shape[0], rotations.shape[1], 3
    )

    save_transl = root_transl.clone()
    target_unit = Unit.from_name(output_unit)
    unit_scale = soma.output_unit.meters_per_unit / target_unit.meters_per_unit
    if unit_scale != 1.0:
        save_transl = save_transl * unit_scale

    save_soma_npz(
        output_path,
        poses_rotvec,
        save_transl,
        joint_names=joint_names,
        identity_model_type=soma.identity_model_type,
        identity_coeffs=identity_coeffs,
        scale_params=scale_params,
        joint_orient=orient,
        unit=output_unit,
        keep_root=keep_root,
        extra_arrays=extra_arrays,
    )
