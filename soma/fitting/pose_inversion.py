# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose inversion for SOMA-compatible skeleton layers.

``fit()`` is the unified API for recovering SOMA skeleton rotations
from posed mesh vertices.  Input vertices are interpreted in the wrapped
layer's ``output_unit``; returned ``root_translation`` and
``per_vertex_error`` use the same unit.  It supports multiple modes:

- **Analytical + Lie-GN** (default): iterative inverse-LBS refinement
  followed by FK-aware Lie algebra Gauss-Newton refinement.
- **Analytical only**: iterative inverse-LBS refinement with
  ``lie_iters=0``.
- **Autograd FK only**: 6D rotation optimization by backpropagating
  through FK + LBS. Slow but controllable (e.g extra weights on
  extremities).
- **Analytical + Lie-GN + autograd FK**: the default solve warm-starts
  autograd refinement.

Accepts vertices in any supported topology (SOMA, MHR, SMPL, ...) —
non-SOMA meshes are automatically transferred to SOMA topology using
the identity model's barycentric interpolator.

Usage::

    from soma.body import SOMALayer
    from soma.pose_inversion import PoseInversion

    soma = SOMALayer("assets", identity_model_type="mhr", device="cuda")
    inv = PoseInversion(soma)
    inv.prepare_identity(identity_coeffs, scale_params)

    # Default: analytical warm start + Lie-GN refinement
    result = inv.fit(posed_vertices)

    # Analytical only
    result = inv.fit(posed_vertices, lie_iters=0)

    # Autograd FK only
    result = inv.fit(posed_vertices, body_iters=0, full_iters=0,
                     lie_iters=0, autograd_iters=100)

    # Analytical + Lie-GN + autograd FK
    result = inv.fit(posed_vertices, autograd_iters=10)

    # result["rotations"]        (B, J, 3, 3) absolute local rotations
    # result["root_translation"] (B, 3) in soma.output_unit
    # result["per_vertex_error"] (B, V) in soma.output_unit
"""

from collections.abc import Mapping
from typing import Any

import torch

from ..geometry.batched_skinning import topk_skinning
from ..geometry.lbs import batch_rodrigues
from ..geometry.lbs_warp import linear_blend_skinning
from ..geometry.rig_utils import (
    compute_skeleton_levels,
    get_body_part_vertex_ids,
    get_joint_descendents,
    joint_local_to_world_levelorder,
    joint_world_to_local,
)
from ..geometry.skeleton_transfer import SkeletonTransfer
from ..geometry.transforms import (
    SE3_from_Rt,
    SE3_inverse,
    align_vectors,
    compute_covariance,
    kabsch,
    newton_schulz,
    regularize_covariance_with_reference,
    rotation_matrices_are_valid,
)

try:
    from ..geometry.fused_refit_warp import fused_refit_level as _fused_refit_level
except ImportError:
    _fused_refit_level = None

# Joints constrained to Z-only rotation in t-pose-relative frame.
_1DOF_Z_JOINTS = frozenset({"LeftForeArm", "RightForeArm", "LeftShin", "RightShin"})

_HIPS_IDX = 1  # SOMA Hips joint (child of virtual Root at 0)
_ROTATION_METHODS = frozenset({"auto", "kabsch", "newton-schulz"})
_REFIT_ROTATION_METHODS = _ROTATION_METHODS
_LIE_GN_DAMPING_FACTORS = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
_AUTO_REFIT_PRIOR_STRENGTH = 0.05


class PoseInversionResult(dict[str, torch.Tensor]):
    """Structured result returned by :obj:`~soma.pose_inversion.PoseInversion.fit`.

    Behaves like a ``dict`` for backwards compatibility
    (``result["rotations"]``) while also supporting attribute access
    (``result.rotations``).
    """

    rotations: torch.Tensor
    root_translation: torch.Tensor
    per_vertex_error: torch.Tensor

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _validate_rotation_method(
    method: str,
    parameter_name: str,
    choices: frozenset[str] = _ROTATION_METHODS,
) -> str:
    if method not in choices:
        choices_str = "', '".join(sorted(choices))
        raise ValueError(f"Unknown {parameter_name}: {method!r}. Use '{choices_str}'.")
    return method


def _solve_lie_gn_normal_equations(JtJ: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve batched Lie-GN normal equations with deterministic damping fallback."""
    B, N, _ = JtJ.shape
    dtype = JtJ.dtype
    device = JtJ.device
    eps = torch.finfo(dtype).eps

    eye = torch.eye(N, dtype=dtype, device=device).expand(B, N, N)
    diag_scale_raw = JtJ.diagonal(dim1=-2, dim2=-1).abs().mean(dim=-1)
    scaled_system = diag_scale_raw > eps
    diag_scale = diag_scale_raw.clamp_min(eps)

    solution = torch.zeros_like(rhs)
    solved = torch.zeros(B, dtype=torch.bool, device=device)

    for damping in _LIE_GN_DAMPING_FACTORS:
        if damping == 0.0:
            system = JtJ
        else:
            system = JtJ + eye * (diag_scale * damping)[:, None, None]

        candidate, info = torch.linalg.solve_ex(
            system,
            rhs.unsqueeze(-1),
            check_errors=False,
        )
        candidate = candidate.squeeze(-1)
        ok = (info == 0) & torch.isfinite(candidate).all(dim=-1)
        if damping != 0.0:
            ok &= scaled_system
        update = ok & ~solved
        if torch.any(update):
            solution[update] = candidate[update]
            solved[update] = True
        if torch.all(solved):
            return solution

    remaining = ~solved
    if torch.any(remaining):
        diag = JtJ.diagonal(dim1=-2, dim2=-1)
        fallback_den = torch.where(
            diag.abs() > eps,
            diag,
            diag_scale[:, None],
        )
        fallback = rhs / fallback_den
        ok = torch.isfinite(fallback).all(dim=-1) & remaining & scaled_system
        if torch.any(ok):
            solution[ok] = fallback[ok]

    return solution


def _active_lie_joint_indices(descendant_weights: torch.Tensor, root_idx: int) -> torch.Tensor:
    """Return joints with geometry influence, excluding the full-body virtual root."""
    active_mask = descendant_weights.any(dim=1)
    if root_idx > 0:
        active_mask[0] = False
    return active_mask.nonzero(as_tuple=True)[0]


def _align_vectors_auto(
    target: torch.Tensor,
    source: torch.Tensor,
    reference_rotation: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """NS-first Procrustes solve with a weak reference gauge for ambiguous twist."""
    covariance = compute_covariance(target, source, virtual_normal=True, eps=eps)
    regularized_covariance = regularize_covariance_with_reference(
        covariance,
        reference_rotation=reference_rotation,
        prior_strength=_AUTO_REFIT_PRIOR_STRENGTH,
        eps=eps,
    )

    regularized_rotation = newton_schulz(regularized_covariance)

    regularized_valid = rotation_matrices_are_valid(
        regularized_rotation,
        det_tol=1e-3,
        orthogonality_tol=1e-3,
    )

    needs_kabsch = ~regularized_valid | (torch.linalg.det(regularized_covariance) < 0)

    fallback_rotation = regularized_rotation
    if torch.any(needs_kabsch):
        fallback_rotation = fallback_rotation.clone()
        fallback_rotation[needs_kabsch] = kabsch(regularized_covariance[needs_kabsch])

    return fallback_rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bexpand(t, B):
    """Add batch dim and expand to B if unbatched, else return as-is."""
    if t.ndim == 2:  # (J, 3) or (V, 3) -> (B, J, 3)
        return t.unsqueeze(0).expand(B, -1, -1)
    if t.ndim == 3 and t.shape[0] == 1 and B > 1:  # (1, J, 3) -> (B, J, 3)
        return t.expand(B, -1, -1)
    return t


def _bexpand4(t, B):
    """Add batch dim and expand to B for 4x4 matrices."""
    if t.ndim == 3:  # (J, 4, 4) -> (B, J, 4, 4)
        return t.unsqueeze(0).expand(B, -1, -1, -1)
    if t.ndim == 4 and t.shape[0] == 1 and B > 1:
        return t.expand(B, -1, -1, -1)
    return t


def _to_sparse_weights(dense_weights, K):
    """Convert (V, J) dense weights to top-K sparse (weights, indices)."""
    V, J = dense_weights.shape
    device = dense_weights.device
    actual_K = min(K, J)
    topk_vals, topk_idx = torch.topk(dense_weights, actual_K, dim=1)
    if actual_K < K:
        pad = K - actual_K
        topk_vals = torch.cat([topk_vals, torch.zeros(V, pad, device=device)], dim=1)
        topk_idx = torch.cat(
            [topk_idx, torch.zeros(V, pad, device=device, dtype=torch.long)],
            dim=1,
        )
    return topk_vals.float(), topk_idx.int()


def _bind_joint_positions_from_cache(cache, *, dtype: torch.dtype, device: torch.device):
    """Return rest-pose joint positions from cached inverse bind transforms."""
    bind_world = SE3_inverse(cache["W_bind_inv"])
    if bind_world.ndim == 4:
        bind_world = bind_world[0]
    bind_world = bind_world.to(device=device, dtype=dtype)
    return bind_world[:, :3, 3]


def _heel_vertex_ids(
    joint_names,
    joint_parent_ids,
    skinning_weights,
    bind_shape,
    bind_joint_positions,
):
    """Return rear foot vertices behind each Foot joint toward the heel."""
    if bind_shape is None or bind_joint_positions is None:
        return []
    if bind_shape.ndim == 3:
        bind_shape = bind_shape[0]
    if bind_joint_positions.ndim == 3:
        bind_joint_positions = bind_joint_positions[0]

    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    heel_ids = []
    for side in ("Left", "Right"):
        foot_idx = name_to_idx.get(f"{side}Foot")
        toe_idx = name_to_idx.get(f"{side}ToeBase")
        if foot_idx is None or toe_idx is None:
            continue
        foot_vids = get_body_part_vertex_ids(
            skinning_weights,
            joint_parent_ids,
            foot_idx,
            include_root=True,
        )
        if not foot_vids:
            continue
        foot_vids_t = torch.as_tensor(
            foot_vids,
            dtype=torch.long,
            device=bind_shape.device,
        )
        foot_to_toe = bind_joint_positions[toe_idx] - bind_joint_positions[foot_idx]
        foot_to_toe = foot_to_toe / foot_to_toe.norm().clamp_min(torch.finfo(bind_shape.dtype).eps)
        proj = (bind_shape[foot_vids_t] - bind_joint_positions[foot_idx]) @ foot_to_toe
        heel_cutoff = torch.quantile(proj, 0.35)
        heel_ids.append(foot_vids_t[proj <= heel_cutoff])

    if not heel_ids:
        return []
    return torch.unique(torch.cat(heel_ids)).tolist()


def _compute_vertex_weights(
    joint_names,
    joint_parent_ids,
    skinning_weights,
    leaf_weight,
    *,
    bind_shape=None,
    bind_joint_positions=None,
):
    """Compute per-vertex importance weights using body-part vertex grouping.

    Uses ``get_body_part_vertex_ids`` to find vertices belonging to head
    (and descendents), hands (and descendents), and feet (and descendents).
    Those vertices are upweighted by the specified weights.

    Args:
        joint_names: list of joint name strings.
        joint_parent_ids: (J,) parent indices.
        skinning_weights: (V, J) dense skinning weights.
        leaf_weight: either a scalar (uniform for all extremities) or a
            dict mapping group names to weights, e.g.
            ``{"head": 2.0, "hands": 2.0, "feet": 5.0}``.
            Supported keys: ``"head"``, ``"hands"``, ``"feet"``, and
            ``"heels"``.
            1.0 = uniform (no upweighting).
        bind_shape: optional rest vertices used by geometric groups such as
            ``"heels"``.
        bind_joint_positions: optional rest joint positions used by geometric
            groups such as ``"heels"``.

    Returns:
        (V,) float32 tensor of per-vertex weights, or None if all weights <= 1.
    """
    # Normalize to per-group dict
    if isinstance(leaf_weight, Mapping):
        group_weights = leaf_weight
    else:
        if leaf_weight <= 1.0:
            return None
        group_weights = {"head": leaf_weight, "hands": leaf_weight, "feet": leaf_weight}

    # Map group names to root joint names
    _GROUP_ROOTS = {
        "head": ["Head"],
        "hands": ["LeftHand", "RightHand"],
        "feet": ["LeftFoot", "RightFoot"],
    }

    V = skinning_weights.shape[0]
    device = skinning_weights.device
    weights = torch.ones(V, device=device)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    any_upweight = False
    for group_name, w in group_weights.items():
        if w <= 1.0:
            continue
        if group_name == "heels":
            vids = _heel_vertex_ids(
                joint_names,
                joint_parent_ids,
                skinning_weights,
                bind_shape,
                bind_joint_positions,
            )
            if vids:
                weights[vids] = w
                any_upweight = True
            continue
        roots = _GROUP_ROOTS.get(group_name, [])
        for root_name in roots:
            j_idx = name_to_idx.get(root_name)
            if j_idx is None:
                continue
            vids = get_body_part_vertex_ids(
                skinning_weights, joint_parent_ids, j_idx, include_root=True
            )
            weights[vids] = w
            any_upweight = True

    return weights if any_upweight else None


def _normalized_vertex_weights(
    joint_names,
    joint_parent_ids,
    skinning_weights,
    leaf_weight,
    *,
    dtype: torch.dtype,
    bind_shape=None,
    bind_joint_positions=None,
):
    """Return mean-one vertex weights, or None for uniform weighting."""
    weights = _compute_vertex_weights(
        joint_names,
        joint_parent_ids,
        skinning_weights,
        leaf_weight,
        bind_shape=bind_shape,
        bind_joint_positions=bind_joint_positions,
    )
    if weights is None:
        return None
    weights = weights.to(dtype=dtype)
    return weights / weights.mean().clamp_min(torch.finfo(dtype).eps)


def _joint_pose_prior_weights(
    joint_names,
    joint_weights: Mapping[str, float] | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """Return per-joint pose-prior weights, or None for uniform prior."""
    if not joint_weights:
        return None

    weights = torch.ones(len(joint_names), dtype=dtype, device=device)
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for joint_name, weight in joint_weights.items():
        if joint_name not in name_to_idx:
            raise ValueError(f"Unknown joint in pose-prior weights: {joint_name!r}")
        weights[name_to_idx[joint_name]] = float(weight)
    return weights


def _classify_joints(joint_names, parent_ids_list):
    """Split joints into body and finger index sets."""
    hand_indices = {i for i, name in enumerate(joint_names) if name.endswith("Hand")}
    finger_set = set()
    for hand_idx in hand_indices:
        finger_set.update(get_joint_descendents(parent_ids_list, hand_idx))
    body_set = set(range(len(joint_names))) - finger_set
    return body_set, finger_set


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------

_MAX_LBS_K = 5  # Cap sparse LBS K to reduce kernel work (K=5 loses < 0.01% weight)


def _precompute_refit_cache(
    joint_names,
    joint_parent_ids,
    bind_world,
    bind_shape,
    skinning_weights,
    t_pose_world,
    root_idx=1,
):
    """Precompute sparse LBS cache for iterative refinement.

    Follows the MHR solver pattern: per-joint sparse subtree/non-subtree
    weight decomposition for inverse-LBS Kabsch.
    """
    device = bind_shape.device
    J = len(joint_parent_ids)
    parent_ids_list = (
        joint_parent_ids.tolist() if hasattr(joint_parent_ids, "tolist") else list(joint_parent_ids)
    )

    bind_local = joint_world_to_local(bind_world, joint_parent_ids)
    bind_local_t = bind_local[:, :3, 3]  # (J, 3)
    W_bind_inv = SE3_inverse(bind_world)  # (J, 4, 4)
    levels = compute_skeleton_levels(joint_parent_ids, device=device)
    bone_indices, bone_weights = topk_skinning(skinning_weights, K=8, pad_index=0)

    # Identify end joints (no children)
    children_count = [0] * J
    for j in range(J):
        p = parent_ids_list[j]
        if p != j:
            children_count[p] += 1
    end_joints = {j for j in range(J) if children_count[j] == 0}

    body_set, finger_set = _classify_joints(joint_names, parent_ids_list)

    # Per-joint subtree info.
    # Skip virtual root at j=0 for full-body (root_idx=1); include j=0 for
    # hand models (root_idx=0) where joint 0 is the wrist with real geometry.
    first_joint = 0 if root_idx == 0 else 1
    sw_cpu = skinning_weights.cpu()
    joint_infos = []
    max_K = 1
    for j_idx in range(first_joint, J):
        if j_idx in end_joints:
            continue
        subtree = [j_idx] + get_joint_descendents(parent_ids_list, j_idx)
        subtree_cols = torch.tensor(subtree, dtype=torch.long)
        arm_mask = sw_cpu[:, subtree_cols].sum(dim=1) > 0.01
        arm_vids = torch.where(arm_mask)[0].to(device)
        if len(arm_vids) == 0:
            continue

        subtree_mask = torch.zeros(J, device=device, dtype=torch.bool)
        subtree_mask[torch.tensor(subtree, device=device, dtype=torch.long)] = True

        sw_arm = skinning_weights[arm_vids]
        sw_arm = sw_arm * (sw_arm > 1e-6)
        sw_sub = sw_arm * subtree_mask.float()
        sw_non = sw_arm * (~subtree_mask).float()

        k = max(
            int((sw_sub > 0).sum(dim=1).max().item()),
            int((sw_non > 0).sum(dim=1).max().item()),
        )
        max_K = max(max_K, k)
        joint_infos.append((j_idx, arm_vids, sw_sub, sw_non))

    # Cap K to limit LBS kernel work
    max_K = min(max_K, _MAX_LBS_K)

    # Build sparse cache per joint
    joint_cache = {}
    for j_idx, arm_vids, sw_sub, sw_non in joint_infos:
        sub_bw, sub_bi = _to_sparse_weights(sw_sub, K=max_K)
        non_bw, non_bi = _to_sparse_weights(sw_non, K=max_K)
        joint_cache[j_idx] = {
            "arm_vids": arm_vids,
            "bind_verts_arm": bind_shape[arm_vids],
            "sub_bone_weights": sub_bw,
            "sub_bone_indices": sub_bi,
            "non_bone_weights": non_bw,
            "non_bone_indices": non_bi,
            "sub_weight_sum": sw_sub.sum(dim=1),
        }

    # Build level-order groups, split body / finger
    body_groups, finger_groups = [], []
    for joint_ids, _parent_ids in levels:
        jlist = joint_ids.tolist()
        bg = [j for j in jlist if j in joint_cache and j in body_set]
        fg = [j for j in jlist if j in joint_cache and j in finger_set]
        if bg:
            body_groups.append(bg)
        if fg:
            finger_groups.append(fg)

    # 1-DOF constraint data (vectorised for all constrained joints)
    t_orient = t_pose_world[:, :3, :3]  # (J, 3, 3)
    constrained_indices = []
    orient_j_list, orient_p_list = [], []
    for j_idx, name in enumerate(joint_names):
        if name in _1DOF_Z_JOINTS:
            constrained_indices.append(j_idx)
            orient_j_list.append(t_orient[j_idx])
            orient_p_list.append(t_orient[parent_ids_list[j_idx]])
    constrained_data = None
    constrained_set = set()
    if constrained_indices:
        constrained_data = {
            "indices": torch.tensor(constrained_indices, device=device, dtype=torch.long),
            "orient_j": torch.stack(orient_j_list),  # (C, 3, 3)
            "orient_p": torch.stack(orient_p_list),  # (C, 3, 3)
        }
        constrained_set = set(constrained_indices)

    # Precompute per-level batched data for fused Warp kernel
    if _fused_refit_level is not None:
        all_groups = body_groups + finger_groups
        body_level_data = _precompute_level_batch_data(
            body_groups, joint_cache, parent_ids_list, device
        )
        finger_level_data = _precompute_level_batch_data(
            finger_groups, joint_cache, parent_ids_list, device
        )
        all_level_data = _precompute_level_batch_data(
            all_groups, joint_cache, parent_ids_list, device
        )
    else:
        body_level_data = None
        finger_level_data = None
        all_level_data = None

    return {
        "joint_names": joint_names,
        "parent_ids": joint_parent_ids,
        "parent_ids_list": parent_ids_list,
        "bind_local_t": bind_local_t,
        "W_bind_inv": W_bind_inv,
        "levels": levels,
        "joint_cache": joint_cache,
        "body_groups": body_groups,
        "finger_groups": finger_groups,
        "constrained_data": constrained_data,
        "constrained_set": constrained_set,
        "t_pose_orient": t_orient,
        "skinning_weights": skinning_weights,
        "bone_weights": bone_weights.to(dtype=bind_shape.dtype, device=device),
        "bone_indices": bone_indices.to(device=device),
        "body_level_data": body_level_data,
        "finger_level_data": finger_level_data,
        "all_level_data": all_level_data,
    }


# ---------------------------------------------------------------------------
# LBS & world transforms
# ---------------------------------------------------------------------------


def _build_world_transforms(pose_local, cache):
    """Build world transforms from local rotations + bind translations."""
    root_idx = cache.get("root_idx", _HIPS_IDX)
    B = pose_local.shape[0]
    bind_t = cache["bind_local_t"]  # (J, 3) or (B, J, 3)
    if bind_t.ndim == 2:
        local_t = bind_t.unsqueeze(0).expand(B, -1, -1).clone()
    else:
        local_t = bind_t.expand(B, -1, -1).clone()
    local_t[:, root_idx, :] = pose_local[:, root_idx, :3, 3]
    T_local = SE3_from_Rt(pose_local[:, :, :3, :3], local_t)
    return joint_local_to_world_levelorder(T_local, cache["levels"])


# ---------------------------------------------------------------------------
# Per-joint inverse-LBS Procrustes refit
# ---------------------------------------------------------------------------


def _refit_joint(
    pose_local,
    target,
    j_idx,
    W,
    D,
    cache,
    jcache,
    vert_weights=None,
    rotation_method="auto",
):
    """Re-fit one joint via inverse-LBS Procrustes alignment using sparse LBS."""
    arm_vids = jcache["arm_vids"]
    bind_verts = jcache["bind_verts_arm"]
    sub_bw = jcache["sub_bone_weights"]
    sub_bi = jcache["sub_bone_indices"]
    non_bw = jcache["non_bone_weights"]
    non_bi = jcache["non_bone_indices"]
    sub_w_sum = jcache["sub_weight_sum"]

    B = pose_local.shape[0]
    bv = _bexpand(bind_verts, B)

    # Sparse LBS: subtree and non-subtree contributions
    q_world = linear_blend_skinning(bv, sub_bw, sub_bi, D)
    c_xyz = linear_blend_skinning(bv, non_bw, non_bi, D)

    # Transform subtree into parent frame
    W_p_inv = SE3_inverse(W[:, j_idx])
    R_inv = W_p_inv[:, :3, :3]
    t_inv = W_p_inv[:, :3, 3]

    sw = sub_w_sum.view(1, -1, 1)
    src = q_world @ R_inv.transpose(-2, -1) + t_inv.unsqueeze(1) * sw

    p_parent = W[:, j_idx, :3, 3]
    tgt = target[:, arm_vids, :] - c_xyz - p_parent.unsqueeze(1) * sw

    # Weighted alignment: multiply both sides by sqrt(w), so
    # H = (sqrt(w) * tgt)^T @ (sqrt(w) * src) = tgt^T @ diag(w) @ src.
    if vert_weights is not None:
        sqrt_w = vert_weights[arm_vids].sqrt().unsqueeze(0).unsqueeze(-1)  # (1, n, 1)
        target_for_alignment = tgt * sqrt_w
        source_for_alignment = src * sqrt_w
    else:
        target_for_alignment = tgt
        source_for_alignment = src

    if rotation_method == "auto":
        R_ref = W[:, j_idx, :3, :3]
        R_new = _align_vectors_auto(target_for_alignment, source_for_alignment, R_ref)
    else:
        R_new = align_vectors(target_for_alignment, source_for_alignment, method=rotation_method)

    # Write back as local rotation
    grandparent_idx = cache["parent_ids_list"][j_idx]
    if grandparent_idx == j_idx:
        # Root joint (self-parent): world rotation IS local rotation
        pose_local[:, j_idx, :3, :3] = R_new
    else:
        R_gp_world = W[:, grandparent_idx, :3, :3]
        pose_local[:, j_idx, :3, :3] = R_gp_world.transpose(-2, -1) @ R_new


# ---------------------------------------------------------------------------
# DOF constraints
# ---------------------------------------------------------------------------


def _constrain_1dof_z(pose_local, cache):
    """Constrain elbow/knee joints to Z-only rotation (vectorised)."""
    cd = cache["constrained_data"]
    if cd is None:
        return
    indices = cd["indices"]  # (C,)
    orient_j = cd["orient_j"]  # (C, 3, 3)
    orient_p = cd["orient_p"]  # (C, 3, 3)

    B = pose_local.shape[0]
    R_abs = pose_local[:, indices, :3, :3]  # (B, C, 3, 3)

    # To t-pose relative: R_tpose = orient_parent @ R_abs @ orient_j^T
    R_tpose = orient_p.unsqueeze(0) @ R_abs @ orient_j.transpose(-2, -1).unsqueeze(0)

    # Extract Z angle
    rz = torch.atan2(R_tpose[:, :, 1, 0], R_tpose[:, :, 0, 0])  # (B, C)

    # Reconstruct Z-only rotation
    cos_rz = torch.cos(rz)
    sin_rz = torch.sin(rz)
    R_z = torch.zeros(B, len(indices), 3, 3, device=pose_local.device, dtype=pose_local.dtype)
    R_z[:, :, 0, 0] = cos_rz
    R_z[:, :, 0, 1] = -sin_rz
    R_z[:, :, 1, 0] = sin_rz
    R_z[:, :, 1, 1] = cos_rz
    R_z[:, :, 2, 2] = 1.0

    # Back to absolute: R_abs = orient_parent^T @ R_z @ orient_j
    R_constrained = orient_p.transpose(-2, -1).unsqueeze(0) @ R_z @ orient_j.unsqueeze(0)
    pose_local[:, indices, :3, :3] = R_constrained


# ---------------------------------------------------------------------------
# Root translation
# ---------------------------------------------------------------------------


def _update_root_translation(pose_local, target, cache, vert_weights=None):
    """Shift root joint translation to minimise mean vertex residual."""
    root_idx = cache.get("root_idx", _HIPS_IDX)
    jcache = cache["joint_cache"].get(root_idx)
    if jcache is None:
        return
    B = pose_local.shape[0]
    W = _build_world_transforms(pose_local, cache)
    D = W @ _bexpand4(cache["W_bind_inv"], B)

    arm_vids = jcache["arm_vids"]
    bv = _bexpand(jcache["bind_verts_arm"], B)
    current = linear_blend_skinning(
        bv,
        jcache["sub_bone_weights"],
        jcache["sub_bone_indices"],
        D,
    )
    residual = target[:, arm_vids, :] - current
    if vert_weights is not None:
        w = vert_weights[arm_vids].to(device=residual.device, dtype=residual.dtype)
        delta_t = (residual * w.view(1, -1, 1)).sum(dim=1) / w.sum().clamp_min(
            torch.finfo(residual.dtype).eps
        )
    else:
        delta_t = residual.mean(dim=1)
    pose_local[:, root_idx, :3, 3] += delta_t


# ---------------------------------------------------------------------------
# Refit passes (MHR-style: per-group FK rebuild + per-group constraint)
# ---------------------------------------------------------------------------


def _run_refit_passes(
    pose_local,
    target,
    cache,
    groups,
    constrain_1dof=True,
    vert_weights=None,
    rotation_method="auto",
):
    """One round of top-down refit following the MHR solver pattern.

    For each group (skeleton level):
    1. Rebuild FK once (joints in the same level are independent).
    2. Refit each joint via inverse-LBS Procrustes alignment.
    3. Apply 1-DOF constraints if any constrained joint is in this group.

    This ensures child joints see properly constrained parent rotations.
    """
    joint_cache = cache["joint_cache"]
    constrained_set = cache["constrained_set"]
    B = pose_local.shape[0]

    for group in groups:
        # Rebuild W once per group — same-level joints don't affect each other's parents
        W = _build_world_transforms(pose_local, cache)
        D = W @ _bexpand4(cache["W_bind_inv"], B)

        for j_idx in group:
            jcache = joint_cache.get(j_idx)
            if jcache is None:
                continue
            _refit_joint(
                pose_local,
                target,
                j_idx,
                W,
                D,
                cache,
                jcache,
                vert_weights,
                rotation_method,
            )

        # Apply 1-DOF constraint after each group (like MHR's per-group DOF constraint)
        if constrain_1dof and constrained_set.intersection(group):
            _constrain_1dof_z(pose_local, cache)


# ---------------------------------------------------------------------------
# Fused Warp kernel refit (2 kernel launches per skeleton level)
# ---------------------------------------------------------------------------


def _precompute_level_batch_data(groups, joint_cache, parent_ids_list, device):
    """Precompute concatenated LBS data per skeleton level for fused refit.

    Flattens per-joint sparse weights/indices into per-level tensors so the
    fused Warp kernel can process an entire skeleton level in 2 launches
    (LBS+covariance + SVD) instead of per-joint Python loops.
    """
    level_data = []
    for group in groups:
        active = [j for j in group if j in joint_cache]
        if not active:
            level_data.append(None)
            continue

        bind_parts, sub_bw_parts, sub_bi_parts = [], [], []
        non_bw_parts, non_bi_parts, sw_sum_parts, vids_parts = [], [], [], []
        counts_list, parents = [], []

        for j in active:
            jc = joint_cache[j]
            counts_list.append(len(jc["arm_vids"]))
            bind_parts.append(jc["bind_verts_arm"])
            sub_bw_parts.append(jc["sub_bone_weights"])
            sub_bi_parts.append(jc["sub_bone_indices"])
            non_bw_parts.append(jc["non_bone_weights"])
            non_bi_parts.append(jc["non_bone_indices"])
            sw_sum_parts.append(jc["sub_weight_sum"])
            vids_parts.append(jc["arm_vids"])
            parents.append(parent_ids_list[j])

        counts = torch.tensor(counts_list, dtype=torch.int32, device=device)
        offsets = torch.zeros(len(active), dtype=torch.int32, device=device)
        if len(counts) > 1:
            offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        V_total = int(counts.sum().item())

        # Per-vertex joint index (local, 0..J_level-1) for the fused kernel
        joint_id_per_vert = torch.empty(V_total, dtype=torch.long, device=device)
        for k_idx in range(len(active)):
            s = offsets[k_idx].item()
            e = s + counts[k_idx].item()
            joint_id_per_vert[s:e] = k_idx

        level_data.append(
            {
                "joint_indices": torch.tensor(active, dtype=torch.long, device=device),
                "parent_indices": torch.tensor(parents, dtype=torch.long, device=device),
                "joint_list": active,
                "bind_verts_cat": torch.cat(bind_parts),
                "sub_bw_cat": torch.cat(sub_bw_parts),
                "sub_bi_cat": torch.cat(sub_bi_parts),
                "non_bw_cat": torch.cat(non_bw_parts),
                "non_bi_cat": torch.cat(non_bi_parts),
                "sub_w_sum_cat": torch.cat(sw_sum_parts),
                "arm_vids_cat": torch.cat(vids_parts),
                "counts": counts,
                "offsets": offsets,
                "V_total": V_total,
                "joint_id_per_vert": joint_id_per_vert,
            }
        )
    return level_data


def _run_refit_passes_fused(
    pose_local,
    target,
    cache,
    level_data_list,
    batched_identity=False,
    rotation_method="auto",
):
    """Refit using fused Warp kernel — 2 kernel launches per skeleton level.

    Replaces per-level: 2 LBS + SE3_inv + src/tgt + covariance + rotation projection (~10 ops)
    With: 1 fused LBS+covariance launch + 1 rotation launch = 2 launches per level.
    """
    root_idx = cache.get("root_idx", _HIPS_IDX)
    bind_local_t = cache["bind_local_t"]
    skel_levels = cache["levels"]
    B = pose_local.shape[0]

    if bind_local_t.ndim == 2:
        local_t = bind_local_t.unsqueeze(0).expand(B, -1, -1).clone()
    else:
        local_t = bind_local_t.expand(B, -1, -1).clone()
    local_t[:, root_idx, :] = pose_local[:, root_idx, :3, 3]
    T_local = SE3_from_Rt(pose_local[:, :, :3, :3], local_t)

    joint_cache = cache["joint_cache"]

    for ld in level_data_list:
        if ld is None:
            continue

        # Full FK rebuild per level (same-level joints are independent)
        W = joint_local_to_world_levelorder(T_local, skel_levels)
        D = W @ _bexpand4(cache["W_bind_inv"], B)

        ji = ld["joint_indices"]
        pi = ld["parent_indices"]
        J_level = len(ji)

        # For batched identity, rebuild bind_verts_cat from current joint_cache
        # (which has been sliced to the current chunk by _slice_bind_cache)
        if batched_identity:
            parts = [joint_cache[j]["bind_verts_arm"] for j in ld["joint_list"]]
            bind_verts_cat = torch.cat(parts, dim=1)  # (B, V_total, 3)
        else:
            bind_verts_cat = ld["bind_verts_cat"]  # (V_total, 3)

        if rotation_method == "kabsch":
            fused_rotation_method = "svd"
        else:
            fused_rotation_method = rotation_method

        reference_rotations = W[:, ji, :3, :3] if rotation_method == "auto" else None
        R_all = _fused_refit_level(
            bind_verts_cat,
            ld["sub_bw_cat"],
            ld["sub_bi_cat"],
            ld["non_bw_cat"],
            ld["non_bi_cat"],
            ld["sub_w_sum_cat"],
            ld["arm_vids_cat"],
            ld["offsets"],
            ld["counts"],
            ld["joint_id_per_vert"],
            ji,
            D,
            W,
            target,
            J_level,
            rotation_method=fused_rotation_method,
            reference_rotations=reference_rotations,
            auto_prior_strength=_AUTO_REFIT_PRIOR_STRENGTH,
        )

        # Write back local rotations: R_local = R_parent^T @ R_world_new
        # For root joints (self-parent: pi == ji), world == local, so R_local = R_world_new.
        R_gp = W[:, pi, :3, :3]
        R_local_new = R_gp.transpose(-2, -1) @ R_all
        # Fix self-parent root joints: use R_world directly
        self_parent_mask = pi == ji
        if self_parent_mask.any():
            R_local_new[:, self_parent_mask] = R_all[:, self_parent_mask]
        pose_local[:, ji, :3, :3] = R_local_new
        # Update T_local in-place so next level sees updated parents
        T_local[:, ji, :3, :3] = pose_local[:, ji, :3, :3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PoseInversion:
    """Invert posed vertices to a layer's skeleton rotations.

    Accepts vertices in the wrapped layer topology. For :class:`SOMALayer`,
    it also accepts supported identity-model native topologies (MHR, SMPL,
    etc.) and transfers them through the identity model's barycentric
    interpolator. All translational inputs and outputs are in the wrapped
    layer's ``output_unit``.

    Args:
        soma_layer: A :obj:`~soma.body.SOMALayer` instance (any LOD).
        low_lod: Use low-LOD SOMA topology (4505 verts) for the iterative
            refit.  This is ~4x fewer vertices than mid-LOD (18056) with
            negligible accuracy loss (~0.006 cm).  When *True* and the
            given *soma_layer* is mid-LOD, a second low-LOD SOMALayer is
            created internally for the refit.  XLO layers keep their own
            topology because there is no direct xlo-to-low vertex transfer.
            Default ``True``.
        skeleton_transfer_rotation_method: rotation extraction method used
            by the initial :class:`SkeletonTransfer` pose estimate. ``"auto"``
            uses a Newton-Schulz-first reference-gauge policy.
        refit_rotation_method: rotation extraction method used by the
            analytical inverse-LBS refit. ``"auto"`` applies
            a weak reference-rotation gauge to the local Procrustes covariance,
            then uses SVD only as a final validity projection if Newton-Schulz
            does not produce a rotation. ``"kabsch"`` and ``"newton-schulz"``
            remain available for diagnostics.

    Usage::

        inv = PoseInversion(soma_layer)
        inv.prepare_identity(identity_coeffs, scale_params)
        result = inv.fit(posed_vertices)  # any supported topology, in soma_layer.output_unit
    """

    def __init__(
        self,
        soma_layer: Any,
        low_lod: bool = True,
        skeleton_transfer_rotation_method: str = "auto",
        refit_rotation_method: str = "auto",
    ) -> None:
        self._soma_orig = soma_layer
        self._cache = None
        self._skel_transfer = None
        self._pose_transfer_interp = None
        self._batched_identity = False
        self._root_joint_idx = getattr(soma_layer, "root_joint_idx", 1)
        self.output_unit = getattr(soma_layer, "output_unit", None)
        procedural_transforms_enabled = bool(
            getattr(soma_layer, "procedural_transforms_enabled", False)
        )
        if getattr(soma_layer, "lod", None) == "xlo":
            low_lod = False
        self.skeleton_transfer_rotation_method = skeleton_transfer_rotation_method
        self.refit_rotation_method = refit_rotation_method

        # Decide which SOMALayer to use for the refit.
        # Procedural twist helpers are dependent skinning joints, so all
        # analytical state is built from the layer's public rig view instead
        # of constructing a separate non-procedural public layer.
        self._autograd_soma = None
        needs_low_lod_refit_layer = low_lod and not getattr(soma_layer, "low_lod", False)
        if needs_low_lod_refit_layer:
            # Create an internal low-LOD layer for the refit.  Procedural mode
            # remains enabled here so autograd refinement can evaluate twist
            # LBS through the same public-pose API as the original layer.
            from .body import SOMALayer

            template_rig_path = getattr(soma_layer, "procedural_template_rig_path", None)
            self.soma = SOMALayer(
                soma_layer.data_root,
                lod="low",
                device=soma_layer.device,
                identity_model_type=soma_layer.identity_model_type,
                mode=soma_layer.mode,
                output_unit=soma_layer.output_unit,
                identity_model_kwargs=soma_layer.identity_model_kwargs,
                template_rig_path=template_rig_path,
                enable_procedural_transforms=procedural_transforms_enabled,
            )
        else:
            self.soma = soma_layer
        if procedural_transforms_enabled:
            self._autograd_soma = self.soma

        self._soma_num_verts = self.soma.bind_shape.shape[0]
        # Full-res SOMA vertex count (before any low-LOD subsampling).
        # When low_lod=True, inputs at this size are downsampled via nv_lod_mid_to_low
        # instead of being routed through the identity-model topology transfer.
        nv_mid_to_low = self.soma.nv_lod_mid_to_low
        if (
            nv_mid_to_low is not None
            and nv_mid_to_low.max().item() < self.soma.rig_data["bind_shape"].shape[0]
        ):
            self._soma_full_num_verts = self.soma.rig_data["bind_shape"].shape[0]
        else:
            self._soma_full_num_verts = None

        # For low-LOD MHR: the identity model's interpolator is built for
        # the low-res MHR mesh (lod6), but pose inversion receives full-res
        # (lod1, 18439-vert) MHR vertices.  Set up a direct interpolator
        # from the full-res MHR source mesh to low-LOD SOMA target.
        if (
            self.soma.low_lod
            and self.soma.identity_model_type == "mhr"
            and self.soma.nv_lod_mid_to_low is not None
        ):
            self._setup_pose_transfer()

    @property
    def skeleton_transfer_rotation_method(self) -> str:
        return self._skeleton_transfer_rotation_method

    @skeleton_transfer_rotation_method.setter
    def skeleton_transfer_rotation_method(self, method: str) -> None:
        method = _validate_rotation_method(method, "skeleton_transfer_rotation_method")
        self._skeleton_transfer_rotation_method = method
        if self._skel_transfer is not None:
            self._skel_transfer.rotation_method = method

    @property
    def refit_rotation_method(self) -> str:
        return self._refit_rotation_method

    @refit_rotation_method.setter
    def refit_rotation_method(self, method: str) -> None:
        self._refit_rotation_method = _validate_rotation_method(
            method,
            "refit_rotation_method",
            _REFIT_ROTATION_METHODS,
        )

    def _setup_pose_transfer(self) -> None:
        """Build barycentric interpolator: full-res MHR -> low-LOD SOMA."""
        import trimesh

        from ..geometry.barycentric_interp import BarycentricInterpolator

        soma = self.soma
        data_root = soma.data_root
        device = soma.device

        mesh_mhr = trimesh.load(
            data_root / "MHR" / "base_body_lod1.obj",
            maintain_order=True,
            process=False,
        )
        V_mhr = torch.from_numpy(mesh_mhr.vertices).float().to(device)
        F_mhr = torch.from_numpy(mesh_mhr.faces).to(device)

        mesh_soma = trimesh.load(
            data_root / "MHR" / "SOMA_wrap_lod1.obj",
            maintain_order=True,
            process=False,
        )
        V_soma = torch.from_numpy(mesh_soma.vertices).float().to(device)
        V_soma_low = V_soma[soma.nv_lod_mid_to_low]

        self._pose_transfer_interp = BarycentricInterpolator(V_mhr, F_mhr, V_soma_low)

    @property
    def joint_names(self) -> list[str]:
        if hasattr(self.soma, "public_joint_names"):
            return list(self.soma.public_joint_names)
        return list(self.soma.rig_data["joint_names"])

    def transfer_to_soma(self, vertices: torch.Tensor) -> torch.Tensor:
        """Transfer vertices from identity-model topology to SOMA topology.

        If vertices are already on SOMA topology, returns them unchanged.

        Args:
            vertices: (B, V, 3) or (V, 3) in any supported topology.

        Returns:
            (B, V_soma, 3) vertices on SOMA topology.
        """
        squeezed = vertices.ndim == 2
        if squeezed:
            vertices = vertices.unsqueeze(0)

        V = vertices.shape[-2]
        if V == self._soma_num_verts:
            return vertices if not squeezed else vertices

        # Full-res SOMA input with low-LOD inversion: subsample directly.
        if self._soma_full_num_verts is not None and V == self._soma_full_num_verts:
            soma_verts = vertices[:, self.soma.nv_lod_mid_to_low, :]
            if squeezed:
                soma_verts = soma_verts.squeeze(0)
            return soma_verts

        # Use dedicated pose-transfer interpolator when available
        if self._pose_transfer_interp is not None:
            soma_verts = self._pose_transfer_interp(vertices)
        else:
            identity_model = self.soma.identity_model
            if not hasattr(identity_model, "_to_soma_interp"):
                raise ValueError(
                    f"Vertex count {V} does not match SOMA ({self._soma_num_verts}) "
                    f"and no topology transfer is available for identity model "
                    f"'{self.soma.identity_model_type}'."
                )
            soma_verts = identity_model._to_soma_interp(vertices)

        if squeezed:
            soma_verts = soma_verts.squeeze(0)
        return soma_verts

    def prepare_identity(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        repose_to_bind_pose: bool = True,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Set up rig from identity parameters.

        Supports both single identity ``(1, C)`` and batched identities
        ``(B, C)``.  Structural caches (sparse weights, groups, levels)
        are built once on the first call and reused on subsequent calls.
        Per-identity bind data (``bind_local_t``, ``W_bind_inv``,
        ``bind_verts``) is updated every call.

        Args:
            identity_coeffs: (1, C) or (B, C) identity coefficients.
            scale_params: (1, S) or (B, S) optional scale parameters.
            repose_to_bind_pose: if True (default), transform the rest
                shape into SOMA's bind pose.  Set to False when the
                target vertices are posed relative to the identity
                model's native rest pose (e.g. MHR).
            kwargs: optional dict passed to the identity model's ``get_rest_shape``.
        """
        soma = self.soma
        soma.prepare_identity(
            identity_coeffs,
            scale_params,
            repose_to_bind_pose=repose_to_bind_pose,
            kwargs=kwargs,
        )
        if self._autograd_soma is not None and self._autograd_soma is not soma:
            self._autograd_soma.prepare_identity(
                identity_coeffs,
                scale_params,
                repose_to_bind_pose=repose_to_bind_pose,
                kwargs=kwargs,
            )

        public_view = soma.public_rig_view() if hasattr(soma, "public_rig_view") else None
        bind_world = (
            public_view.bind_transforms_world
            if public_view is not None
            else soma._cached_bind_transforms_world
        )  # (B, J, 4, 4)
        bind_shape = soma._cached_rest_shape  # (B, V, 3)
        joint_parent_ids = (
            public_view.joint_parent_ids if public_view is not None else soma.joint_parent_ids
        )
        skinning_weights = (
            public_view.skinning_weights if public_view is not None else soma.skinning_weights
        )
        t_pose_world = public_view.t_pose_world if public_view is not None else soma.t_pose_world

        # SkeletonTransfer and structural caches use first identity
        bind_world_0 = bind_world[0]  # (J, 4, 4)
        bind_shape_0 = bind_shape[0]  # (V, 3)

        # Build structural cache once (sparse weights, groups, levels, Warp batch data).
        # These depend only on skinning_weights and joint_parent_ids, which are constant.
        if self._cache is None:
            self._skel_transfer = SkeletonTransfer(
                joint_parent_ids,
                bind_world_0,
                bind_shape_0,
                skinning_weights,
                rotation_method=self.skeleton_transfer_rotation_method,
                vertex_ids_to_exclude=soma.excluded_vert_ids,
                root_joint_idx=self._root_joint_idx,
            )

            self._cache = _precompute_refit_cache(
                self.joint_names,
                joint_parent_ids,
                bind_world_0,
                bind_shape_0,
                skinning_weights,
                t_pose_world,
                root_idx=self._root_joint_idx,
            )
            self._cache["root_idx"] = self._root_joint_idx
        else:
            # Update identity-dependent bind and refit data. The joint group
            # topology is stable, but cached bind vertices and fused level data
            # are identity-specific.
            self._skel_transfer.update_bind(bind_world_0, bind_shape_0)
            self._cache = _precompute_refit_cache(
                self.joint_names,
                joint_parent_ids,
                bind_world_0,
                bind_shape_0,
                skinning_weights,
                t_pose_world,
                root_idx=self._root_joint_idx,
            )
            self._cache["root_idx"] = self._root_joint_idx

        # Replace bind-dependent entries with batched versions when B > 1
        B_id = bind_world.shape[0]
        self._batched_identity = B_id > 1
        if self._batched_identity:
            self._full_bind_world = bind_world
            self._full_rest_shape = bind_shape

            bind_local = joint_world_to_local(bind_world, joint_parent_ids)
            self._cache["bind_local_t"] = bind_local[:, :, :3, 3]  # (B, J, 3)
            self._cache["W_bind_inv"] = SE3_inverse(bind_world)  # (B, J, 4, 4)

            for jc in self._cache["joint_cache"].values():
                jc["bind_verts_arm"] = bind_shape[:, jc["arm_vids"]]  # (B, n, 3)

    def _save_bind_cache(self) -> None:
        """Save identity-dependent cache entries before chunked slicing."""
        cache = self._cache
        self._saved_bind = {
            "bind_local_t": cache["bind_local_t"],
            "W_bind_inv": cache["W_bind_inv"],
            "joint_bind_verts": {j: jc["bind_verts_arm"] for j, jc in cache["joint_cache"].items()},
        }

    def _slice_bind_cache(self, start: int, end: int) -> None:
        """Slice batched bind cache entries for a chunk [start:end]."""
        cache = self._cache
        cache["bind_local_t"] = self._saved_bind["bind_local_t"][start:end]
        cache["W_bind_inv"] = self._saved_bind["W_bind_inv"][start:end]

        for j, jc in cache["joint_cache"].items():
            jc["bind_verts_arm"] = self._saved_bind["joint_bind_verts"][j][start:end]

    def _restore_bind_cache(self) -> None:
        """Restore batched bind cache entries after chunked processing."""
        cache = self._cache
        cache["bind_local_t"] = self._saved_bind["bind_local_t"]
        cache["W_bind_inv"] = self._saved_bind["W_bind_inv"]

        for j, jc in cache["joint_cache"].items():
            jc["bind_verts_arm"] = self._saved_bind["joint_bind_verts"][j]

    def _chunked_call(
        self,
        method,
        posed_vertices: torch.Tensor,
        batch_size: int,
        **kwargs: Any,
    ) -> PoseInversionResult:
        """Process posed_vertices in chunks, handling per-identity bind data."""
        B = posed_vertices.shape[0]
        soma = self.soma
        saved_rest = soma._cached_rest_shape

        if self._batched_identity:
            self._save_bind_cache()

        chunks = []
        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            chunk = posed_vertices[start:end]

            if self._batched_identity:
                soma._cached_rest_shape = self._full_rest_shape[start:end]
                self._slice_bind_cache(start, end)

            # Call without batch_size to avoid infinite recursion
            chunks.append(method(chunk, **kwargs))

        if self._batched_identity:
            soma._cached_rest_shape = saved_rest
            self._restore_bind_cache()

        return PoseInversionResult(
            {key: torch.cat([c[key] for c in chunks], dim=0) for key in chunks[0]}
        )

    def fit(
        self,
        posed_vertices: torch.Tensor,
        body_iters: int = 2,
        finger_iters: int = 0,
        full_iters: int = 1,
        lie_iters: int = 3,
        lie_lambda: float = 1e-1,
        autograd_iters: int = 0,
        autograd_lr: float = 5e-3,
        autograd_translation_lr_scale: float = 1.0,
        autograd_pose_prior: float = 0.0,
        autograd_leaf_weight: float | Mapping[str, float] | None = None,
        autograd_pose_prior_weights: Mapping[str, float] | None = None,
        constrain_1dof: bool = False,
        leaf_weight: float | Mapping[str, float] = 1.0,
        batch_size: int | None = None,
    ) -> PoseInversionResult:
        """Fit SOMA skeleton rotations to posed vertices.

        Supports several modes depending on the iteration arguments:

        - **Analytical + Lie algebra Gauss-Newton** (default):
          ``body_iters=2, full_iters=1, lie_iters=3``.  The analytical
          solve gives a fast warm start, then Lie-GN solves all active joint
          rotations simultaneously via a dense FK-coupled normal equation.
        - **Analytical only**: ``lie_iters=0``.
        - **Lie algebra Gauss-Newton only**:
          ``body_iters=0, full_iters=0, lie_iters=5``.
        - **Autograd FK only**:
          ``body_iters=0, full_iters=0, lie_iters=0, autograd_iters=10``.
          Slow but controllable (e.g. extra weights on extremities).
        - **Default + autograd FK**: ``autograd_iters=10``.  The default
          analytical + Lie-GN solve warm-starts autograd refinement.

        Args:
            posed_vertices: (B, V, 3) vertices on any supported topology.
            body_iters: analytical iterations for body chain (default: 2).
            finger_iters: analytical iterations for finger chain (default: 0).
            full_iters: analytical iterations for all joints (default: 1).
            lie_iters: Lie algebra Gauss-Newton iterations (default: 3).
                When > 0, runs a dense batched Gauss-Newton solve in SO(3)
                after the analytical solver.  Each iter solves a (3J x 3J)
                system for all joint twists simultaneously.
            lie_lambda: Tikhonov regularisation for the Lie-GN normal
                equations (default: 1e-1).
            autograd_iters: Adam optimization steps through FK + LBS (default: 0).
                When > 0, runs autograd refinement after the analytical solve.
            autograd_lr: learning rate for autograd Adam (default: 5e-3).
            autograd_translation_lr_scale: multiplier for the root-translation
                Adam learning rate.  Useful because translations are optimized
                in layer output units while rotations use unitless 6D
                parameters.
            autograd_pose_prior: local-rotation prior weight for autograd FK.
                Penalizes deviation from the initial local rotation matrices.
                0.0 disables it.
            autograd_leaf_weight: optional vertex weighting used only by the
                autograd FK stage.  This lets the analytical/Lie warm start
                remain unweighted while the refinement emphasizes contact
                regions.
            autograd_pose_prior_weights: optional per-joint multipliers for
                the autograd pose prior.  Values > 1 stiffen a joint relative
                to the warm start; values < 1 let it move more.
            constrain_1dof: apply 1-DOF constraints on elbows/knees
                (analytical only).
            leaf_weight: importance multiplier for extremity vertices.
                Float for uniform (e.g. 3.0), or dict for per-group
                (e.g. ``{"head": 2, "hands": 2, "feet": 5, "heels": 10}``).
                1.0 = uniform (default).
            batch_size: process in chunks of this size.

        Returns:
            dict with ``rotations`` (B, J, 3, 3),
            ``root_translation`` (B, 3), and
            ``per_vertex_error`` (B, V) L2 error per vertex.
        """
        if self._cache is None:
            raise RuntimeError("Call prepare_identity() first.")

        B = posed_vertices.shape[0]
        if batch_size is not None and B > batch_size:
            return self._chunked_call(
                self.fit,
                posed_vertices,
                batch_size,
                body_iters=body_iters,
                finger_iters=finger_iters,
                full_iters=full_iters,
                lie_iters=lie_iters,
                lie_lambda=lie_lambda,
                autograd_iters=autograd_iters,
                autograd_lr=autograd_lr,
                autograd_translation_lr_scale=autograd_translation_lr_scale,
                autograd_pose_prior=autograd_pose_prior,
                autograd_leaf_weight=autograd_leaf_weight,
                autograd_pose_prior_weights=autograd_pose_prior_weights,
                constrain_1dof=constrain_1dof,
                leaf_weight=leaf_weight,
            )

        cache = self._cache

        # Auto-transfer to SOMA topology if needed
        with torch.no_grad():
            posed_vertices = self.transfer_to_soma(posed_vertices)

        has_analytical = body_iters > 0 or finger_iters > 0 or full_iters > 0

        if has_analytical:
            with torch.no_grad():
                result = self._fit_analytical(
                    posed_vertices,
                    cache,
                    body_iters,
                    finger_iters,
                    full_iters,
                    constrain_1dof,
                    leaf_weight,
                )
        else:
            result = None

        if lie_iters > 0:
            with torch.no_grad():
                result = self._fit_lie_algebra_gn(
                    posed_vertices,
                    cache,
                    lie_iters,
                    lie_lambda,
                    leaf_weight,
                    init_result=result,
                )

        if autograd_iters > 0:
            result = self._fit_autograd_fk(
                posed_vertices,
                cache,
                autograd_iters,
                autograd_lr,
                autograd_translation_lr_scale,
                leaf_weight if autograd_leaf_weight is None else autograd_leaf_weight,
                autograd_pose_prior,
                autograd_pose_prior_weights,
                init_result=result,
            )

        if result is None:
            raise ValueError(
                "At least one of body_iters, finger_iters, full_iters, "
                "lie_iters, or autograd_iters must be > 0."
            )

        return result

    def _fit_analytical(
        self,
        posed_vertices: torch.Tensor,
        cache: Mapping[str, Any],
        body_iters: int,
        finger_iters: int,
        full_iters: int,
        constrain_1dof: bool,
        leaf_weight: float | Mapping[str, float],
    ) -> PoseInversionResult:
        """Analytical iterative inverse-LBS refinement."""
        body_groups = cache["body_groups"]
        finger_groups = cache["finger_groups"]
        all_groups = body_groups + finger_groups

        # Compute per-vertex importance weights for leaf joints
        bind_shape = self.soma._cached_rest_shape.detach()
        bind_joint_positions = _bind_joint_positions_from_cache(
            cache,
            dtype=bind_shape.dtype,
            device=bind_shape.device,
        )
        vert_weights = _compute_vertex_weights(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            leaf_weight,
            bind_shape=bind_shape,
            bind_joint_positions=bind_joint_positions,
        )

        # --- Pass 1: unconstrained skeleton fit ---
        pose_world = self._skel_transfer.fit(posed_vertices)
        pose_local = joint_world_to_local(pose_world, cache["parent_ids"])

        if constrain_1dof:
            _constrain_1dof_z(pose_local, cache)

        # Select refit path: fused Warp kernel or fallback PyTorch
        use_fused = (
            _fused_refit_level is not None
            and cache["body_level_data"] is not None
            # PyTorch Kabsch stays on the SVD path; NS and auto use Warp.
            and self.refit_rotation_method in {"newton-schulz", "auto"}
            and vert_weights is None  # fused kernel doesn't support vertex weighting
            and not constrain_1dof  # fused kernel doesn't support per-level DOF constraints
        )

        if use_fused:
            body_ld = cache["body_level_data"]
            finger_ld = cache["finger_level_data"]
            all_ld = cache["all_level_data"]
            bi = self._batched_identity

            for _ in range(body_iters):
                _run_refit_passes_fused(
                    pose_local,
                    posed_vertices,
                    cache,
                    body_ld,
                    bi,
                    self.refit_rotation_method,
                )
                _update_root_translation(pose_local, posed_vertices, cache, vert_weights)

            for _ in range(finger_iters):
                _run_refit_passes_fused(
                    pose_local,
                    posed_vertices,
                    cache,
                    finger_ld,
                    bi,
                    self.refit_rotation_method,
                )

            for _ in range(full_iters):
                _run_refit_passes_fused(
                    pose_local,
                    posed_vertices,
                    cache,
                    all_ld,
                    bi,
                    self.refit_rotation_method,
                )
                _update_root_translation(pose_local, posed_vertices, cache, vert_weights)
        else:
            # --- Body iterations ---
            for _ in range(body_iters):
                _run_refit_passes(
                    pose_local,
                    posed_vertices,
                    cache,
                    body_groups,
                    constrain_1dof,
                    vert_weights,
                    self.refit_rotation_method,
                )
                _update_root_translation(pose_local, posed_vertices, cache, vert_weights)

            # --- Finger iterations ---
            for _ in range(finger_iters):
                _run_refit_passes(
                    pose_local,
                    posed_vertices,
                    cache,
                    finger_groups,
                    constrain_1dof,
                    vert_weights,
                    self.refit_rotation_method,
                )

            # --- Full iterations ---
            for _ in range(full_iters):
                _run_refit_passes(
                    pose_local,
                    posed_vertices,
                    cache,
                    all_groups,
                    constrain_1dof,
                    vert_weights,
                    self.refit_rotation_method,
                )
                _update_root_translation(pose_local, posed_vertices, cache, vert_weights)

        # --- Extract output ---
        root_idx = self._root_joint_idx
        rotations = pose_local[:, :, :3, :3].clone()  # (B, J, 3, 3)
        root_translation = pose_local[:, root_idx, :3, 3].clone()

        # --- Compute per-vertex error via internal LBS ---
        B = pose_local.shape[0]
        W = _build_world_transforms(pose_local, cache)
        D = W @ _bexpand4(cache["W_bind_inv"], B)
        bind_shape = self.soma._cached_rest_shape.expand(B, -1, -1)
        recon = linear_blend_skinning(
            bind_shape,
            cache["bone_weights"],
            cache["bone_indices"],
            D,
        )
        per_vertex_error = torch.norm(recon - posed_vertices, dim=-1)  # (B, V)

        return PoseInversionResult(
            rotations=rotations,
            root_translation=root_translation,
            per_vertex_error=per_vertex_error,
        )

    def _fit_lie_algebra_gn(
        self,
        target: torch.Tensor,
        cache: Mapping[str, Any],
        n_iters: int,
        lambda_reg: float = 1e-1,
        leaf_weight: float | Mapping[str, float] = 1.0,
        init_result: PoseInversionResult | None = None,
    ) -> PoseInversionResult:
        """FK-aware dense Lie algebra Gauss-Newton pose refinement.

        Solves for all joint rotations simultaneously via a (3J x 3J) linear
        system. Uses the Kinematic Lever Arm to construct the exact FK-coupled
        Jacobian:

            q_{i,j} = sum_{k in D(j)} w_{i,k} * (p_world_{i,k} - c_j)

        where D(j) is the subtree of joint j and c_j is its world pivot.
        The Jacobian block is J_{i,j} = -[q_{i,j}]_x, which correctly captures
        that rotating joint j moves ALL descendant bone positions — not just the
        direct attachment as in the independent-joint approximation.

        Args:
            target: (B, V, 3) SOMA-topology vertices (already transferred).
            cache: precomputed refit cache.
            n_iters: number of Gauss-Newton iterations.
            lambda_reg: Marquardt damping factor applied to active joints.
                The virtual root and joints with zero descendant skinning
                influence are factored out of the solve entirely. Each active
                diagonal entry is scaled by (1 + lambda_reg). Unit-invariant;
                0.1 gives light damping.
            leaf_weight: optional vertex importance weights for the weighted
                least-squares objective.
            init_result: if provided, warm-start from this result; otherwise
                warm-start from skeleton transfer.
        """
        soma = self.soma
        B = target.shape[0]
        device = target.device
        dtype = target.dtype

        W_weights = cache["skinning_weights"]  # (V, J) full dense weights
        W_bind_inv = cache["W_bind_inv"]  # (J, 4, 4) or (B, J, 4, 4)
        bind_shape = soma._cached_rest_shape  # (1, V, 3) or (B, V, 3)
        if bind_shape.shape[0] == 1 and B > 1:
            bind_shape = bind_shape.expand(B, -1, -1)
        bind_joint_positions = _bind_joint_positions_from_cache(
            cache,
            dtype=target.dtype,
            device=target.device,
        )
        vert_weights = _normalized_vertex_weights(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            leaf_weight,
            dtype=target.dtype,
            bind_shape=bind_shape,
            bind_joint_positions=bind_joint_positions,
        )

        sp_bw = cache["bone_weights"]  # (V, K_sparse) sparse weights
        sp_bi = cache["bone_indices"]  # (V, K_sparse) sparse indices

        parent_ids_list = cache["parent_ids_list"]
        parent_ids_tensor = cache["parent_ids"]  # (J,) tensor
        J = len(parent_ids_list)
        W_bind_inv_b = _bexpand4(W_bind_inv, B)

        # Initialise pose_local
        root_idx = self._root_joint_idx
        has_virtual_root = root_idx > 0  # full-body: joint 0 is virtual root
        eye3 = torch.eye(3, device=device, dtype=dtype)
        if init_result is not None:
            pose_local = torch.zeros(B, J, 4, 4, device=device, dtype=dtype)
            pose_local[:, :, :3, :3] = init_result["rotations"]
            pose_local[:, root_idx, :3, 3] = init_result["root_translation"]
        else:
            W_init = self._skel_transfer.fit(target)
            pose_local = joint_world_to_local(W_init, cache["parent_ids"])
        if has_virtual_root:
            pose_local[:, 0, :3, :3] = eye3

        # ---- Precompute ancestor mask (constant across iterations) ----
        # A[j, k] = 1.0 if k is in the subtree of j (j is ancestor-or-equal of k)
        # Equivalently: rotating joint j affects all bones k where A[j, k] = 1.
        A = torch.zeros(J, J, device=device, dtype=dtype)
        for k in range(J):
            cur = k
            while True:
                A[cur, k] = 1.0
                par = parent_ids_list[cur]
                if par == cur:  # virtual root self-loops
                    break
                cur = par

        # AW[j, v] = sum_k A[j, k] * W_weights[v, k]  — total descendant weight
        # Shape: (J, J) @ (J, V) -> (J, V)
        AW = A @ W_weights.T  # (J, V)

        # Active joints: only those whose subtree actually influences vertices.
        # Virtual roots are ancestors of the whole skeleton, so their AW rows are
        # nonzero even though they are structural and must stay at identity. Leaf
        # end-joints have zero AW rows. Factor both groups out and solve only the
        # (3*K_act x 3*K_act) system.
        active_idx = _active_lie_joint_indices(AW, root_idx)  # (K_act,) long
        K_act = active_idx.shape[0]

        for _ in range(n_iters):
            # ---- FK -> world transforms + D matrices ----
            W_world = _build_world_transforms(pose_local, cache)
            D = W_world @ W_bind_inv_b  # (B, J, 4, 4)
            R_D = D[:, :, :3, :3]  # (B, J, 3, 3)
            t_D = D[:, :, :3, 3]  # (B, J, 3)
            c = W_world[:, :, :3, 3]  # (B, J, 3) — joint world pivot positions

            # p_world[b, k, v] = R_D[b,k] @ v_bind[v] + t_D[b,k]
            # World position of vertex v if 100% attached to bone k
            p_world = torch.einsum("bjmn,bvn->bjvm", R_D, bind_shape) + t_D[:, :, None, :]

            # Current vertices via LBS
            v_curr = torch.einsum("vj,bjvm->bvm", W_weights, p_world)
            residual = target - v_curr  # (B, V, 3)

            # ---- Kinematic Lever Arm q[b, j, v] (active joints only) ----
            weighted_p = p_world * W_weights.T[None, :, :, None]  # (B, J, V, 3)
            V_verts = bind_shape.shape[-2]
            AWP = (A @ weighted_p.reshape(B, J, V_verts * 3)).reshape(B, J, V_verts, 3)
            q_full = AWP - AW[None, :, :, None] * c[:, :, None, :]  # (B, J, V, 3)
            q = q_full[:, active_idx]  # (B, K_act, V, 3)

            # ---- J^T e and J^T J for active joints only ----
            e_exp = residual[:, None, :, :].expand(-1, K_act, -1, -1)
            if vert_weights is not None:
                wv = vert_weights.view(1, 1, V_verts, 1)
                Jte_act = (torch.linalg.cross(q, e_exp) * wv).sum(dim=2)
                q_weighted = q * wv
            else:
                Jte_act = torch.linalg.cross(q, e_exp).sum(dim=2)  # (B, K_act, 3)
                q_weighted = q

            dot_sum = torch.einsum("bjvm,bkvm->bjk", q_weighted, q)
            outer_sum = torch.einsum("bjvn,bkvm->bjkmn", q_weighted, q)
            JtJ_blocks = dot_sum[..., None, None] * eye3 - outer_sum  # (B, K_act, K_act, 3, 3)

            JtJ = JtJ_blocks.permute(0, 1, 3, 2, 4).reshape(B, K_act * 3, K_act * 3)
            JtJ.diagonal(dim1=-2, dim2=-1)[:] *= 1.0 + lambda_reg

            delta_act = _solve_lie_gn_normal_equations(
                JtJ,
                Jte_act.reshape(B, K_act * 3),
            ).reshape(B, K_act, 3)

            # Scatter back into full (B, J, 3) delta; inactive joints get zero twist
            delta_omega = torch.zeros(B, J, 3, device=device, dtype=dtype)
            delta_omega[:, active_idx] = delta_act

            # ---- Per-frame backtracking line search ----
            # Even with active-joint factoring, the linearization breaks down near
            # the model-mismatch floor.  The line search accepts each frame's step
            # only if it reduces error, avoiding oscillatory divergence.
            residual_norm = residual.norm(dim=-1)
            if vert_weights is not None:
                pre_err = (residual_norm * vert_weights.view(1, -1)).sum(dim=-1) / (
                    vert_weights.sum().clamp_min(torch.finfo(target.dtype).eps)
                )
            else:
                pre_err = residual_norm.mean(dim=-1)  # (B,)
            R_world = W_world[:, :, :3, :3]  # (B, J, 3, 3)
            R_world_parents = R_world[:, parent_ids_tensor, :, :]

            pose_local_accepted = pose_local.clone()
            accepted_err = pre_err.clone()

            # Identify root joints (self-parent) for correct local rotation writeback
            is_self_parent = parent_ids_tensor == torch.arange(J, device=device)

            for alpha in (1.0, 0.5, 0.25, 0.125):
                dR = batch_rodrigues((alpha * delta_omega).reshape(B * J, 3), dtype=dtype).reshape(
                    B, J, 3, 3
                )
                R_world_new = dR @ R_world
                R_local_try = R_world_parents.transpose(-2, -1) @ R_world_new
                # Root joints (self-parent): world rotation IS local rotation
                if is_self_parent.any():
                    R_local_try[:, is_self_parent] = R_world_new[:, is_self_parent]
                pose_local_try = pose_local.clone()
                pose_local_try[:, :, :3, :3] = R_local_try
                W_try = _build_world_transforms(pose_local_try, cache)
                D_try = W_try @ W_bind_inv_b
                recon_try = linear_blend_skinning(bind_shape, sp_bw, sp_bi, D_try)
                err_try_per_vertex = (target - recon_try).norm(dim=-1)
                if vert_weights is not None:
                    err_try = (err_try_per_vertex * vert_weights.view(1, -1)).sum(dim=-1) / (
                        vert_weights.sum().clamp_min(torch.finfo(target.dtype).eps)
                    )
                else:
                    err_try = err_try_per_vertex.mean(dim=-1)
                improved = err_try < accepted_err
                pose_local_accepted[improved] = pose_local_try[improved]
                accepted_err[improved] = err_try[improved]

            pose_local = pose_local_accepted

            # Fine-tune root translation every iteration
            _update_root_translation(pose_local, target, cache, vert_weights)

        # --- Extract result ---
        rotations = pose_local[:, :, :3, :3].clone()
        root_translation = pose_local[:, root_idx, :3, 3].clone()

        # Per-vertex error
        W_final = _build_world_transforms(pose_local, cache)
        D_final = W_final @ _bexpand4(W_bind_inv, B)
        recon = linear_blend_skinning(
            bind_shape,
            cache["bone_weights"],
            cache["bone_indices"],
            D_final,
        )
        per_vertex_error = torch.norm(recon - target, dim=-1)

        return PoseInversionResult(
            rotations=rotations,
            root_translation=root_translation,
            per_vertex_error=per_vertex_error,
        )

    def _fit_autograd_fk(
        self,
        target: torch.Tensor,
        cache: Mapping[str, Any],
        n_iters: int,
        lr: float,
        translation_lr_scale: float,
        leaf_weight: float | Mapping[str, float],
        pose_prior: float = 0.0,
        pose_prior_weights: Mapping[str, float] | None = None,
        init_result: PoseInversionResult | None = None,
    ) -> PoseInversionResult:
        """FK-based gradient optimization of local 6D rotations.

        Args:
            target: (B, V, 3) SOMA-topology vertices (already transferred).
            cache: precomputed refit cache.
            n_iters: number of Adam steps.
            lr: learning rate.
            translation_lr_scale: root-translation LR multiplier.
            leaf_weight: vertex importance weights.
            pose_prior: local-rotation prior weight against the initial fit.
            pose_prior_weights: optional per-joint pose-prior multipliers.
            init_result: if provided, warm-start from this result's rotations
                and root_translation.  Otherwise, warm-start from skeleton
                transfer.
        """
        from ..geometry.transforms import rotation_6d_to_matrix

        if self._autograd_soma is not None:
            return self._fit_autograd_public_layer(
                target,
                cache,
                n_iters,
                lr,
                translation_lr_scale,
                leaf_weight,
                pose_prior,
                pose_prior_weights,
                init_result,
            )

        soma = self.soma
        target = target.detach()
        B = target.shape[0]

        # Fixed LBS data (detached — no grad through identity)
        bone_weights = cache["bone_weights"].detach()
        bone_indices = cache["bone_indices"].detach()
        W_bind_inv = cache["W_bind_inv"].detach()  # (J, 4, 4) or (B, J, 4, 4)
        bind_shape = soma._cached_rest_shape.detach()  # (1, V, 3)
        if bind_shape.shape[0] == 1 and B > 1:
            bind_shape = bind_shape.expand(B, -1, -1)
        bind_joint_positions = _bind_joint_positions_from_cache(
            cache,
            dtype=target.dtype,
            device=target.device,
        )

        bind_local_t = cache["bind_local_t"].detach()
        levels = cache["levels"]

        root_idx = self._root_joint_idx
        has_virtual_root = root_idx > 0  # full-body: joint 0 is virtual root

        with torch.no_grad():
            if init_result is not None:
                R_local_init = init_result["rotations"].clone()
                root_t_init = init_result["root_translation"].clone()
            else:
                W_init = self._skel_transfer.fit(target)  # (B, J, 4, 4)
                T_local_init = joint_world_to_local(W_init, cache["parent_ids"])
                R_local_init = T_local_init[:, :, :3, :3].clone()
                root_t_init = T_local_init[:, root_idx, :3, 3]
            if has_virtual_root:
                # Virtual root must stay identity
                R_local_init[:, 0] = torch.eye(3, device=target.device, dtype=target.dtype)

        J = R_local_init.shape[1]

        if has_virtual_root:
            # Optimize joints 1..J-1 (virtual root frozen to identity)
            rot6d_body = R_local_init[:, 1:, :2, :].reshape(B, -1, 6)
            rot6d_opt = rot6d_body.clone().detach().requires_grad_(True)
            eye3 = torch.eye(3, device=target.device, dtype=target.dtype)
            root_6d = eye3[:2, :].reshape(1, 1, 6).expand(B, 1, 6)
        else:
            # Optimize all joints (no virtual root)
            rot6d_body = R_local_init[:, :, :2, :].reshape(B, -1, 6)
            rot6d_opt = rot6d_body.clone().detach().requires_grad_(True)

        transl_opt = root_t_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                {"params": [rot6d_opt], "lr": lr},
                {"params": [transl_opt], "lr": lr * translation_lr_scale},
            ]
        )

        vert_weights = _normalized_vertex_weights(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            leaf_weight,
            dtype=target.dtype,
            bind_shape=bind_shape,
            bind_joint_positions=bind_joint_positions,
        )
        prior_joint_slice = slice(1, None) if has_virtual_root else slice(None)
        joint_prior_weights = _joint_pose_prior_weights(
            cache["joint_names"],
            pose_prior_weights,
            dtype=target.dtype,
            device=target.device,
        )
        R_local_ref = R_local_init.detach()

        for _ in range(n_iters):
            optimizer.zero_grad()

            if has_virtual_root:
                all_6d = torch.cat([root_6d, rot6d_opt.reshape(B, J - 1, 6)], dim=1)
            else:
                all_6d = rot6d_opt.reshape(B, J, 6)
            R_local = rotation_6d_to_matrix(all_6d.reshape(B * J, 6)).reshape(B, J, 3, 3)

            # FK: local rotations + bind translations -> world transforms
            if bind_local_t.ndim == 2:
                local_t = bind_local_t.unsqueeze(0).expand(B, -1, -1).clone()
            else:
                local_t = bind_local_t.expand(B, -1, -1).clone()
            local_t[:, root_idx] = transl_opt

            T_local = SE3_from_Rt(R_local, local_t)
            W = joint_local_to_world_levelorder(T_local, levels)
            D = W @ _bexpand4(W_bind_inv, B)
            verts = linear_blend_skinning(bind_shape, bone_weights, bone_indices, D)

            if vert_weights is not None:
                # Weighted MSE: upweight leaf-joint vertices
                w = vert_weights.unsqueeze(0).unsqueeze(-1)  # (1, V, 1)
                loss = (w * (verts - target) ** 2).mean()
            else:
                loss = torch.nn.functional.mse_loss(verts, target)
            if pose_prior > 0.0:
                R_delta = R_local[:, prior_joint_slice] - R_local_ref[:, prior_joint_slice]
                if joint_prior_weights is not None:
                    w = joint_prior_weights[prior_joint_slice].view(1, -1, 1, 1)
                    loss = loss + pose_prior * (w * R_delta.square()).mean()
                else:
                    loss = loss + pose_prior * R_delta.square().mean()
            loss.backward()
            optimizer.step()

        # Extract result
        with torch.no_grad():
            if has_virtual_root:
                all_6d = torch.cat([root_6d, rot6d_opt.reshape(B, J - 1, 6)], dim=1)
            else:
                all_6d = rot6d_opt.reshape(B, J, 6)
            R_local = rotation_6d_to_matrix(all_6d.reshape(B * J, 6)).reshape(B, J, 3, 3)

            if bind_local_t.ndim == 2:
                local_t = bind_local_t.unsqueeze(0).expand(B, -1, -1).clone()
            else:
                local_t = bind_local_t.expand(B, -1, -1).clone()
            local_t[:, root_idx] = transl_opt.detach()

            T_local = SE3_from_Rt(R_local, local_t)
            W = joint_local_to_world_levelorder(T_local, levels)
            D = W @ _bexpand4(W_bind_inv, B)
            verts = linear_blend_skinning(bind_shape, bone_weights, bone_indices, D)
            per_vertex_error = torch.norm(verts - target, dim=-1)

            R_ref = R_local_ref[:, prior_joint_slice]
            R_fit = R_local[:, prior_joint_slice]
            R_rel = R_fit @ R_ref.transpose(-1, -2)
            cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            local_rotation_drift = torch.zeros(B, J, device=target.device, dtype=target.dtype)
            local_rotation_drift[:, prior_joint_slice] = torch.acos(cos_angle.clamp(-1.0, 1.0))
            root_translation_drift = torch.norm(transl_opt.detach() - root_t_init, dim=-1)

        return PoseInversionResult(
            rotations=R_local,
            root_translation=transl_opt.detach(),
            per_vertex_error=per_vertex_error,
            local_rotation_drift=local_rotation_drift,
            root_translation_drift=root_translation_drift,
        )

    def _fit_autograd_public_layer(
        self,
        target: torch.Tensor,
        cache: Mapping[str, Any],
        n_iters: int,
        lr: float,
        translation_lr_scale: float,
        leaf_weight: float | Mapping[str, float],
        pose_prior: float = 0.0,
        pose_prior_weights: Mapping[str, float] | None = None,
        init_result: PoseInversionResult | None = None,
    ) -> PoseInversionResult:
        """Autograd refinement through a public-pose layer.

        This path is used when the original layer has procedural twist joints.
        The optimized variables remain the 78 public SOMA joint rotations, but
        the forward pass runs through the procedural layer so LBS evaluates the
        twist-joint skinning implied by those public rotations.
        """
        from ..geometry.transforms import rotation_6d_to_matrix

        layer = self._autograd_soma
        if layer is None:
            raise RuntimeError("Internal error: no autograd layer is available.")

        target = target.detach()
        B = target.shape[0]
        root_idx = self._root_joint_idx

        with torch.no_grad():
            if init_result is not None:
                R_local_init = init_result["rotations"].clone()
                root_t_init = init_result["root_translation"].clone()
            else:
                W_init = self._skel_transfer.fit(target)
                T_local_init = joint_world_to_local(W_init, cache["parent_ids"])
                R_local_init = T_local_init[:, :, :3, :3].clone()
                root_t_init = T_local_init[:, root_idx, :3, 3]
            R_local_init[:, 0] = torch.eye(3, device=target.device, dtype=target.dtype)

        J = R_local_init.shape[1]
        rot6d_body = R_local_init[:, 1:, :2, :].reshape(B, J - 1, 6)
        rot6d_opt = rot6d_body.clone().detach().requires_grad_(True)
        eye3 = torch.eye(3, device=target.device, dtype=target.dtype)
        root_6d = eye3[:2, :].reshape(1, 1, 6).expand(B, 1, 6)
        transl_opt = root_t_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                {"params": [rot6d_opt], "lr": lr},
                {"params": [transl_opt], "lr": lr * translation_lr_scale},
            ]
        )

        bind_shape = self.soma._cached_rest_shape.detach()
        if bind_shape.shape[0] == 1 and B > 1:
            bind_shape = bind_shape.expand(B, -1, -1)
        bind_joint_positions = _bind_joint_positions_from_cache(
            cache,
            dtype=target.dtype,
            device=target.device,
        )
        vert_weights = _normalized_vertex_weights(
            cache["joint_names"],
            cache["parent_ids"],
            cache["skinning_weights"],
            leaf_weight,
            dtype=target.dtype,
            bind_shape=bind_shape,
            bind_joint_positions=bind_joint_positions,
        )
        joint_prior_weights = _joint_pose_prior_weights(
            cache["joint_names"],
            pose_prior_weights,
            dtype=target.dtype,
            device=target.device,
        )
        R_local_ref = R_local_init.detach()

        skinning = layer.batched_skinning
        target_bind_shape = layer._cached_rest_shape.detach()
        target_W_bind_inv = skinning.inverse_bind_transform.detach()
        target_bone_indices, target_bone_weights = topk_skinning(layer.skinning_weights.detach())
        target_bone_indices = target_bone_indices.to(device=target.device)
        target_bone_weights = target_bone_weights.to(dtype=target.dtype, device=target.device)
        bind_batch = target_bind_shape.shape[0] if target_bind_shape.ndim == 3 else 1
        effective_batch = bind_batch if B == 1 and bind_batch > 1 else B
        bone_scales = layer._pose_batch_bone_scales(
            effective_batch,
            skinning.local_translations.dtype,
            skinning.local_translations.device,
        )
        target_local_t_override = None
        public_local_t_override = None
        if bone_scales is not None:
            target_local_t_override = layer._apply_target_bone_scales(bone_scales).detach()
            public_local_t_override = layer._apply_public_bone_scales(
                bone_scales,
                skinning.source_local_translations,
            ).detach()

        def procedural_vertices(public_rotations, root_translation):
            public_world = skinning.forward_source_kinematics(
                local_rotations=public_rotations,
                global_translation=root_translation,
                absolute_pose=True,
                local_translations=public_local_t_override,
            )
            target_world = skinning.expand_source_world_transforms(
                source_rotations=public_rotations,
                source_world_transforms=public_world,
                transform_expander=(
                    layer.procedural_transforms.expand_world_transforms_from_source_fk
                ),
                target_local_translations=target_local_t_override,
            )
            batch_size = target_world.shape[0]
            D = target_world @ _bexpand4(target_W_bind_inv, batch_size)
            vertices = target_bind_shape
            if vertices.shape[0] == 1 and batch_size > 1:
                vertices = vertices.expand(batch_size, -1, -1)
            return linear_blend_skinning(
                vertices,
                target_bone_weights,
                target_bone_indices,
                D,
            )

        for _ in range(n_iters):
            optimizer.zero_grad()
            all_6d = torch.cat([root_6d, rot6d_opt.reshape(B, J - 1, 6)], dim=1)
            R_local = rotation_6d_to_matrix(all_6d.reshape(B * J, 6)).reshape(B, J, 3, 3)
            verts = procedural_vertices(R_local, transl_opt)

            if vert_weights is not None:
                loss = (vert_weights.unsqueeze(0).unsqueeze(-1) * (verts - target).square()).mean()
            else:
                loss = torch.nn.functional.mse_loss(verts, target)
            if pose_prior > 0.0:
                R_delta = R_local[:, 1:] - R_local_ref[:, 1:]
                if joint_prior_weights is not None:
                    w = joint_prior_weights[1:].view(1, -1, 1, 1)
                    loss = loss + pose_prior * (w * R_delta.square()).mean()
                else:
                    loss = loss + pose_prior * R_delta.square().mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            all_6d = torch.cat([root_6d, rot6d_opt.reshape(B, J - 1, 6)], dim=1)
            R_local = rotation_6d_to_matrix(all_6d.reshape(B * J, 6)).reshape(B, J, 3, 3)
            verts = procedural_vertices(R_local, transl_opt.detach())
            per_vertex_error = torch.norm(verts - target, dim=-1)

            R_rel = R_local[:, 1:] @ R_local_ref[:, 1:].transpose(-1, -2)
            cos_angle = (R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            local_rotation_drift = torch.zeros(B, J, device=target.device, dtype=target.dtype)
            local_rotation_drift[:, 1:] = torch.acos(cos_angle.clamp(-1.0, 1.0))
            root_translation_drift = torch.norm(transl_opt.detach() - root_t_init, dim=-1)

        return PoseInversionResult(
            rotations=R_local,
            root_translation=transl_opt.detach(),
            per_vertex_error=per_vertex_error,
            local_rotation_drift=local_rotation_drift,
            root_translation_drift=root_translation_drift,
        )

    def roundtrip(
        self, posed_vertices: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, PoseInversionResult]:
        """Invert and forward for verification.

        Returns:
            (soma_vertices, result) where soma_vertices is (B, V_soma, 3).
        """
        result = self.fit(posed_vertices, **kwargs)
        cache = self._cache
        B = result["rotations"].shape[0]
        pose_local = torch.zeros(
            B,
            len(cache["parent_ids_list"]),
            4,
            4,
            device=result["rotations"].device,
            dtype=result["rotations"].dtype,
        )
        pose_local[:, :, :3, :3] = result["rotations"]
        pose_local[:, self._root_joint_idx, :3, 3] = result["root_translation"]
        W = _build_world_transforms(pose_local, cache)
        D = W @ _bexpand4(cache["W_bind_inv"], B)
        bind_shape = self.soma._cached_rest_shape.expand(B, -1, -1)
        vertices = linear_blend_skinning(
            bind_shape,
            cache["bone_weights"],
            cache["bone_indices"],
            D,
        )
        return vertices, result
