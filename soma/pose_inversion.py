# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose inversion for SOMA skeleton.

``fit()`` is the unified API for recovering SOMA skeleton rotations
from posed mesh vertices.  It supports three modes:

- **Analytical only** (default): iterative inverse-LBS Newton-Schulz
  refinement.  Extremely fast (~1200 FPS) with comparable accuracy.
- **Autograd FK only**: 6D rotation optimization by backpropagating
  through FK + LBS. Slow but controllable (e.g extra weights on
  extremities).
- **Analytical + autograd FK**: analytical solve warm-starts autograd
  refinement. Best of the both worlds.

Accepts vertices in any supported topology (SOMA, MHR, SMPL, ...) —
non-SOMA meshes are automatically transferred to SOMA topology using
the identity model's barycentric interpolator.

Usage::

    from soma.soma import SOMALayer
    from soma.pose_inversion import PoseInversion

    soma = SOMALayer("assets", identity_model_type="mhr", device="cuda")
    inv = PoseInversion(soma)
    inv.prepare_identity(identity_coeffs, scale_params)

    # Analytical only (fast, default)
    result = inv.fit(posed_vertices)

    # Autograd FK only
    result = inv.fit(posed_vertices, body_iters=0, full_iters=0,
                     autograd_iters=100)

    # Analytical + autograd FK (best accuracy)
    result = inv.fit(posed_vertices, autograd_iters=10)

    # result["rotations"]        (B, J, 3, 3) absolute local rotations
    # result["root_translation"] (B, 3)
    # result["per_vertex_error"] (B, V)
    # result["joint_positions"]  (B, J, 3) world-space joint positions
"""

import torch

from .geometry.lbs_warp import linear_blend_skinning
from .geometry.rig_utils import (
    compute_skeleton_levels,
    get_body_part_vertex_ids,
    get_joint_descendents,
    joint_local_to_world_levelorder,
    joint_world_to_local,
)
from .geometry.skeleton_transfer import SkeletonTransfer
from .geometry.transforms import SE3_from_Rt, SE3_inverse, align_vectors

try:
    from .geometry.fused_refit_warp import fused_refit_level as _fused_refit_level
except ImportError:
    _fused_refit_level = None

# Joints constrained to Z-only rotation in t-pose-relative frame.
_1DOF_Z_JOINTS = frozenset({"LeftForeArm", "RightForeArm", "LeftShin", "RightShin"})

_HIPS_IDX = 1  # SOMA Hips joint (child of virtual Root at 0)
HIPS_IDX = _HIPS_IDX  #: Public alias — index of the Hips joint (Root's child).


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


def _compute_vertex_weights(joint_names, joint_parent_ids, skinning_weights, leaf_weight):
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
            Supported keys: ``"head"``, ``"hands"``, ``"feet"``.
            1.0 = uniform (no upweighting).

    Returns:
        (V,) float32 tensor of per-vertex weights, or None if all weights <= 1.
    """
    # Normalise to per-group dict
    if isinstance(leaf_weight, dict):
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

    # Identify end joints (no children)
    children_count = [0] * J
    for j in range(J):
        p = parent_ids_list[j]
        if p != j:
            children_count[p] += 1
    end_joints = {j for j in range(J) if children_count[j] == 0}

    body_set, finger_set = _classify_joints(joint_names, parent_ids_list)

    # Per-joint subtree info (skip virtual Root at j=0)
    sw_cpu = skinning_weights.cpu()
    joint_infos = []
    max_K = 1
    for j_idx in range(1, J):
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
        "body_level_data": body_level_data,
        "finger_level_data": finger_level_data,
        "all_level_data": all_level_data,
    }


# ---------------------------------------------------------------------------
# LBS & world transforms
# ---------------------------------------------------------------------------


def _build_world_transforms(pose_local, cache):
    """Build world transforms from local rotations + bind translations."""
    B = pose_local.shape[0]
    bind_t = cache["bind_local_t"]  # (J, 3) or (B, J, 3)
    if bind_t.ndim == 2:
        local_t = bind_t.unsqueeze(0).expand(B, -1, -1).clone()
    else:
        local_t = bind_t.expand(B, -1, -1).clone()
    local_t[:, _HIPS_IDX, :] = pose_local[:, _HIPS_IDX, :3, 3]
    T_local = SE3_from_Rt(pose_local[:, :, :3, :3], local_t)
    return joint_local_to_world_levelorder(T_local, cache["levels"])


def build_world_transforms(pose_local, cache):
    """Build world SE3 transforms from local rotations and bind translations.

    Public wrapper around the internal FK computation.

    Args:
        pose_local: (B, J, 4, 4) local SE3 transforms.  Rotation in the
            upper-left 3x3; hips translation in ``[:, HIPS_IDX, :3, 3]``.
        cache: dict returned by ``PoseInversion._cache`` after
            ``prepare_identity()`` has been called.  Required keys:
            ``"bind_local_t"`` and ``"levels"``.

    Returns:
        (B, J, 4, 4) world SE3 transforms via level-order FK.
    """
    return _build_world_transforms(pose_local, cache)


# ---------------------------------------------------------------------------
# Per-joint inverse LBS Kabsch refit
# ---------------------------------------------------------------------------


def _refit_joint(pose_local, target, j_idx, W, D, cache, jcache, vert_weights=None):
    """Re-fit one joint via inverse-LBS Kabsch using sparse LBS."""
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

    # Weighted alignment: weight tgt so H = tgt^T @ diag(w) @ src
    if vert_weights is not None:
        w = vert_weights[arm_vids].unsqueeze(0).unsqueeze(-1)  # (1, n, 1)
        R_new = align_vectors(tgt * w, src, method="newton-schulz")
    else:
        R_new = align_vectors(tgt, src, method="newton-schulz")

    # Write back as local rotation
    grandparent_idx = cache["parent_ids_list"][j_idx]
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


def _update_root_translation(pose_local, target, cache):
    """Shift hips translation to minimise mean vertex residual."""
    jcache = cache["joint_cache"].get(_HIPS_IDX)
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
    delta_t = (target[:, arm_vids, :] - current).mean(dim=1)
    pose_local[:, _HIPS_IDX, :3, 3] += delta_t


# ---------------------------------------------------------------------------
# Refit passes (MHR-style: per-group FK rebuild + per-group constraint)
# ---------------------------------------------------------------------------


def _run_refit_passes(pose_local, target, cache, groups, constrain_1dof=True, vert_weights=None):
    """One round of top-down refit following the MHR solver pattern.

    For each group (skeleton level):
    1. Rebuild FK once (joints in the same level are independent).
    2. Refit each joint via inverse-LBS Kabsch.
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
            _refit_joint(pose_local, target, j_idx, W, D, cache, jcache, vert_weights)

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


def _run_refit_passes_fused(pose_local, target, cache, level_data_list, batched_identity=False):
    """Refit using fused Warp kernel — 2 kernel launches per skeleton level.

    Replaces per-level: 2 LBS + SE3_inv + src/tgt + covariance + SVD (~10 ops)
    With: 1 fused LBS+covariance launch + 1 SVD launch = 2 launches per level.
    """
    bind_local_t = cache["bind_local_t"]
    skel_levels = cache["levels"]
    B = pose_local.shape[0]

    if bind_local_t.ndim == 2:
        local_t = bind_local_t.unsqueeze(0).expand(B, -1, -1).clone()
    else:
        local_t = bind_local_t.expand(B, -1, -1).clone()
    local_t[:, _HIPS_IDX, :] = pose_local[:, _HIPS_IDX, :3, 3]
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

        R_all = _fused_refit_level(
            bind_verts_cat,
            ld["sub_bw_cat"],
            ld["sub_bi_cat"],
            ld["non_bw_cat"],
            ld["non_bi_cat"],
            ld["sub_w_sum_cat"],
            ld["arm_vids_cat"],
            ld["joint_id_per_vert"],
            ji,
            D,
            W,
            target,
            J_level,
        )

        # Write back local rotations: R_local = R_grandparent^T @ R_world_new
        R_gp = W[:, pi, :3, :3]
        pose_local[:, ji, :3, :3] = R_gp.transpose(-2, -1) @ R_all
        # Update T_local in-place so next level sees updated parents
        T_local[:, ji, :3, :3] = pose_local[:, ji, :3, :3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PoseInversion:
    """Invert posed vertices to SOMA skeleton rotations.

    Accepts vertices in SOMA topology or in the identity model's native
    topology (MHR, SMPL, …).  Non-SOMA vertices are automatically
    transferred via the identity model's barycentric interpolator.

    Args:
        soma_layer: A :class:`SOMALayer` instance (any LOD).
        low_lod: Use low-LOD SOMA topology (4505 verts) for the iterative
            refit.  This is ~4x fewer vertices than mid-LOD (18056) with
            negligible accuracy loss (~0.006 cm).  When *True* and the
            given *soma_layer* is mid-LOD, a second low-LOD SOMALayer is
            created internally for the refit.  Default ``True``.

    Usage::

        inv = PoseInversion(soma_layer)
        inv.prepare_identity(identity_coeffs, scale_params)
        result = inv.fit(posed_vertices)  # any supported topology
    """

    def __init__(self, soma_layer, low_lod=True):
        self._soma_orig = soma_layer
        self._cache = None
        self._skel_transfer = None
        self._pose_transfer_interp = None
        self._batched_identity = False

        # Decide which SOMALayer to use for the refit
        if low_lod and not soma_layer.low_lod:
            # Create an internal low-LOD SOMALayer for the refit
            from .soma import SOMALayer

            self.soma = SOMALayer(
                soma_layer.data_root,
                low_lod=True,
                device=soma_layer.device,
                identity_model_type=soma_layer.identity_model_type,
                mode=soma_layer.mode,
                output_unit=soma_layer.output_unit,
                identity_model_kwargs=soma_layer.identity_model_kwargs,
            )
        else:
            self.soma = soma_layer

        self._soma_num_verts = self.soma.bind_shape.shape[0]

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

    def _setup_pose_transfer(self):
        """Build barycentric interpolator: full-res MHR -> low-LOD SOMA."""
        import trimesh

        from .geometry.barycentric_interp import BarycentricInterpolator

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
    def joint_names(self):
        return list(self.soma.rig_data["joint_names"])

    def transfer_to_soma(self, vertices):
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

    def prepare_identity(self, identity_coeffs, scale_params=None, repose_to_bind_pose=True):
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
        """
        soma = self.soma
        soma.prepare_identity(
            identity_coeffs,
            scale_params,
            repose_to_bind_pose=repose_to_bind_pose,
        )

        bind_world = soma._cached_bind_transforms_world  # (B, J, 4, 4)
        bind_shape = soma._cached_rest_shape  # (B, V, 3)

        # SkeletonTransfer and structural caches use first identity
        bind_world_0 = bind_world[0]  # (J, 4, 4)
        bind_shape_0 = bind_shape[0]  # (V, 3)

        # Build structural cache once (sparse weights, groups, levels, Warp batch data).
        # These depend only on skinning_weights and joint_parent_ids, which are constant.
        if self._cache is None:
            self._skel_transfer = SkeletonTransfer(
                soma.joint_parent_ids,
                bind_world_0,
                bind_shape_0,
                soma.skinning_weights,
                rotation_method="kabsch",
                vertex_ids_to_exclude=soma.facial_inner_geometry,
            )

            self._cache = _precompute_refit_cache(
                self.joint_names,
                soma.joint_parent_ids,
                bind_world_0,
                bind_shape_0,
                soma.skinning_weights,
                soma.t_pose_world,
            )
        else:
            # Update SkeletonTransfer bind data only
            self._skel_transfer.update_bind(bind_world_0, bind_shape_0)

            # Update bind-dependent entries in cache (single-identity)
            bind_local_0 = joint_world_to_local(bind_world_0.unsqueeze(0), soma.joint_parent_ids)
            self._cache["bind_local_t"] = bind_local_0[0, :, :3, 3]  # (J, 3)
            self._cache["W_bind_inv"] = SE3_inverse(bind_world_0.unsqueeze(0))[0]  # (J, 4, 4)
            for jc in self._cache["joint_cache"].values():
                jc["bind_verts_arm"] = bind_shape_0[jc["arm_vids"]]  # (n, 3)

        # Replace bind-dependent entries with batched versions when B > 1
        B_id = bind_world.shape[0]
        self._batched_identity = B_id > 1
        if self._batched_identity:
            self._full_bind_world = bind_world
            self._full_rest_shape = bind_shape

            bind_local = joint_world_to_local(bind_world, soma.joint_parent_ids)
            self._cache["bind_local_t"] = bind_local[:, :, :3, 3]  # (B, J, 3)
            self._cache["W_bind_inv"] = SE3_inverse(bind_world)  # (B, J, 4, 4)

            for jc in self._cache["joint_cache"].values():
                jc["bind_verts_arm"] = bind_shape[:, jc["arm_vids"]]  # (B, n, 3)

    def _save_bind_cache(self):
        """Save identity-dependent cache entries before chunked slicing."""
        cache = self._cache
        self._saved_bind = {
            "bind_local_t": cache["bind_local_t"],
            "W_bind_inv": cache["W_bind_inv"],
            "joint_bind_verts": {j: jc["bind_verts_arm"] for j, jc in cache["joint_cache"].items()},
        }

    def _slice_bind_cache(self, start, end):
        """Slice batched bind cache entries for a chunk [start:end]."""
        cache = self._cache
        cache["bind_local_t"] = self._saved_bind["bind_local_t"][start:end]
        cache["W_bind_inv"] = self._saved_bind["W_bind_inv"][start:end]

        for j, jc in cache["joint_cache"].items():
            jc["bind_verts_arm"] = self._saved_bind["joint_bind_verts"][j][start:end]

    def _restore_bind_cache(self):
        """Restore batched bind cache entries after chunked processing."""
        cache = self._cache
        cache["bind_local_t"] = self._saved_bind["bind_local_t"]
        cache["W_bind_inv"] = self._saved_bind["W_bind_inv"]

        for j, jc in cache["joint_cache"].items():
            jc["bind_verts_arm"] = self._saved_bind["joint_bind_verts"][j]

    def _chunked_call(self, method, posed_vertices, batch_size, **kwargs):
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

        return {key: torch.cat([c[key] for c in chunks], dim=0) for key in chunks[0]}

    def fit(
        self,
        posed_vertices,
        body_iters=2,
        finger_iters=0,
        full_iters=1,
        autograd_iters=0,
        autograd_lr=5e-3,
        constrain_1dof=False,
        leaf_weight=1.0,
        batch_size=None,
    ):
        """Fit SOMA skeleton rotations to posed vertices.

        Supports three modes depending on the iteration arguments:

        - **Analytical only** (default): ``body_iters=2, full_iters=1``.
          Extremely fast (~1200 FPS) with comparable accuracy.
        - **Autograd FK only**: ``body_iters=0, full_iters=0, autograd_iters=10``.
          Slow but controllable (e.g. extra weights on extremities).
        - **Analytical + autograd FK**: ``body_iters=2, full_iters=1, autograd_iters=10``.
          Analytical solve warm-starts autograd refinement.
          Best of both worlds.

        Args:
            posed_vertices: (B, V, 3) vertices on any supported topology.
            body_iters: analytical iterations for body chain (default: 2).
            finger_iters: analytical iterations for finger chain (default: 0).
            full_iters: analytical iterations for all joints (default: 1).
            autograd_iters: Adam optimization steps through FK + LBS (default: 0).
                When > 0, runs autograd refinement after the analytical solve.
            autograd_lr: learning rate for autograd Adam (default: 5e-3).
            constrain_1dof: apply 1-DOF constraints on elbows/knees
                (analytical only).
            leaf_weight: importance multiplier for extremity vertices.
                Float for uniform (e.g. 3.0), or dict for per-group
                (e.g. ``{"head": 2, "hands": 2, "feet": 5}``).
                1.0 = uniform (default).
            batch_size: process in chunks of this size.

        Returns:
            dict with ``rotations`` (B, J, 3, 3),
            ``root_translation`` (B, 3),
            ``per_vertex_error`` (B, V) L2 error per vertex, and
            ``joint_positions`` (B, J, 3) world-space joint positions.
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
                autograd_iters=autograd_iters,
                autograd_lr=autograd_lr,
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

        if autograd_iters > 0:
            result = self._fit_autograd_fk(
                posed_vertices,
                cache,
                autograd_iters,
                autograd_lr,
                leaf_weight,
                init_result=result,
            )

        if result is None:
            raise ValueError(
                "At least one of body_iters, finger_iters, full_iters, "
                "or autograd_iters must be > 0."
            )

        return result

    def _fit_analytical(
        self,
        posed_vertices,
        cache,
        body_iters,
        finger_iters,
        full_iters,
        constrain_1dof,
        leaf_weight,
    ):
        """Analytical iterative inverse-LBS Newton-Schulz refinement."""
        body_groups = cache["body_groups"]
        finger_groups = cache["finger_groups"]
        all_groups = body_groups + finger_groups

        # Compute per-vertex importance weights for leaf joints
        vert_weights = _compute_vertex_weights(
            cache["joint_names"], cache["parent_ids"], cache["skinning_weights"], leaf_weight
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
            and vert_weights is None  # fused kernel doesn't support vertex weighting
            and not constrain_1dof  # fused kernel doesn't support per-level DOF constraints
        )

        if use_fused:
            body_ld = cache["body_level_data"]
            finger_ld = cache["finger_level_data"]
            all_ld = cache["all_level_data"]
            bi = self._batched_identity

            for _ in range(body_iters):
                _run_refit_passes_fused(pose_local, posed_vertices, cache, body_ld, bi)
                _update_root_translation(pose_local, posed_vertices, cache)

            for _ in range(finger_iters):
                _run_refit_passes_fused(pose_local, posed_vertices, cache, finger_ld, bi)

            for _ in range(full_iters):
                _run_refit_passes_fused(pose_local, posed_vertices, cache, all_ld, bi)
                _update_root_translation(pose_local, posed_vertices, cache)
        else:
            # --- Body iterations ---
            for _ in range(body_iters):
                _run_refit_passes(
                    pose_local, posed_vertices, cache, body_groups, constrain_1dof, vert_weights
                )
                _update_root_translation(pose_local, posed_vertices, cache)

            # --- Finger iterations ---
            for _ in range(finger_iters):
                _run_refit_passes(
                    pose_local, posed_vertices, cache, finger_groups, constrain_1dof, vert_weights
                )

            # --- Full iterations ---
            for _ in range(full_iters):
                _run_refit_passes(
                    pose_local, posed_vertices, cache, all_groups, constrain_1dof, vert_weights
                )
                _update_root_translation(pose_local, posed_vertices, cache)

        # --- Extract output ---
        rotations = pose_local[:, :, :3, :3].clone()  # (B, J, 3, 3)
        root_translation = pose_local[:, _HIPS_IDX, :3, 3].clone()

        # --- Compute per-vertex error via internal LBS ---
        B = pose_local.shape[0]
        W = _build_world_transforms(pose_local, cache)
        D = W @ _bexpand4(cache["W_bind_inv"], B)
        bs = self.soma.batched_skinning
        bind_shape = self.soma._cached_rest_shape.expand(B, -1, -1)
        recon = linear_blend_skinning(
            bind_shape,
            bs.get_bone_weights(),
            bs.get_bone_indices(),
            D,
        )
        per_vertex_error = torch.norm(recon - posed_vertices, dim=-1)  # (B, V)

        return {
            "rotations": rotations,
            "root_translation": root_translation,
            "per_vertex_error": per_vertex_error,
            "joint_positions": W[:, :, :3, 3].clone(),  # (B, J, 3)
        }

    def _fit_autograd_fk(
        self,
        target,
        cache,
        n_iters,
        lr,
        leaf_weight,
        init_result=None,
    ):
        """FK-based gradient optimization of local 6D rotations.

        Args:
            target: (B, V, 3) SOMA-topology vertices (already transferred).
            cache: precomputed refit cache.
            n_iters: number of Adam steps.
            lr: learning rate.
            leaf_weight: vertex importance weights.
            init_result: if provided, warm-start from this result's rotations
                and root_translation.  Otherwise, warm-start from skeleton
                transfer.
        """
        from .geometry.transforms import rotation_6d_to_matrix

        soma = self.soma
        target = target.detach()
        B = target.shape[0]

        # Fixed LBS data (detached — no grad through identity)
        bs = soma.batched_skinning
        bone_weights = bs.get_bone_weights().detach()
        bone_indices = bs.get_bone_indices().detach()
        W_bind_inv = cache["W_bind_inv"].detach()  # (J, 4, 4) or (B, J, 4, 4)
        bind_shape = soma._cached_rest_shape.detach()  # (1, V, 3)
        if bind_shape.shape[0] == 1 and B > 1:
            bind_shape = bind_shape.expand(B, -1, -1)

        bind_local_t = cache["bind_local_t"].detach()
        levels = cache["levels"]

        with torch.no_grad():
            if init_result is not None:
                R_local_init = init_result["rotations"].clone()
                hips_t_init = init_result["root_translation"].clone()
            else:
                W_init = self._skel_transfer.fit(target)  # (B, J, 4, 4)
                T_local_init = joint_world_to_local(W_init, cache["parent_ids"])
                R_local_init = T_local_init[:, :, :3, :3].clone()
                hips_t_init = T_local_init[:, _HIPS_IDX, :3, 3]
            # Root must stay identity (soma.pose() prepends identity root)
            R_local_init[:, 0] = torch.eye(3, device=target.device, dtype=target.dtype)

        J = R_local_init.shape[1]

        # Optimize joints 1..J-1 (root is frozen to identity)
        rot6d_body = R_local_init[:, 1:, :2, :].reshape(B, -1, 6)
        rot6d_opt = rot6d_body.clone().detach().requires_grad_(True)
        transl_opt = hips_t_init.clone().detach().requires_grad_(True)

        # Fixed root 6D (identity matrix -> first two rows)
        eye3 = torch.eye(3, device=target.device, dtype=target.dtype)
        root_6d = eye3[:2, :].reshape(1, 1, 6).expand(B, 1, 6)

        optimizer = torch.optim.Adam([rot6d_opt, transl_opt], lr=lr)

        # Per-vertex importance weights for leaf joints
        vert_weights = _compute_vertex_weights(
            cache["joint_names"], cache["parent_ids"], cache["skinning_weights"], leaf_weight
        )

        for _ in range(n_iters):
            optimizer.zero_grad()

            # Prepend frozen root identity to optimized body rotations
            all_6d = torch.cat([root_6d, rot6d_opt.reshape(B, J - 1, 6)], dim=1)
            R_local = rotation_6d_to_matrix(all_6d.reshape(B * J, 6)).reshape(B, J, 3, 3)

            # FK: local rotations + bind translations -> world transforms
            if bind_local_t.ndim == 2:
                local_t = bind_local_t.unsqueeze(0).expand(B, -1, -1).clone()
            else:
                local_t = bind_local_t.expand(B, -1, -1).clone()
            local_t[:, _HIPS_IDX] = transl_opt

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
            loss.backward()
            optimizer.step()

        # Extract result
        with torch.no_grad():
            all_6d = torch.cat([root_6d, rot6d_opt.reshape(B, J - 1, 6)], dim=1)
            R_local = rotation_6d_to_matrix(all_6d.reshape(B * J, 6)).reshape(B, J, 3, 3)

            if bind_local_t.ndim == 2:
                local_t = bind_local_t.unsqueeze(0).expand(B, -1, -1).clone()
            else:
                local_t = bind_local_t.expand(B, -1, -1).clone()
            local_t[:, _HIPS_IDX] = transl_opt.detach()

            T_local = SE3_from_Rt(R_local, local_t)
            W = joint_local_to_world_levelorder(T_local, levels)
            D = W @ _bexpand4(W_bind_inv, B)
            verts = linear_blend_skinning(bind_shape, bone_weights, bone_indices, D)
            per_vertex_error = torch.norm(verts - target, dim=-1)

        return {
            "rotations": R_local,
            "root_translation": transl_opt.detach(),
            "per_vertex_error": per_vertex_error,
            "joint_positions": W[:, :, :3, 3].clone(),  # (B, J, 3)
        }

    def roundtrip(self, posed_vertices, **kwargs):
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
        pose_local[:, _HIPS_IDX, :3, 3] = result["root_translation"]
        W = _build_world_transforms(pose_local, cache)
        D = W @ _bexpand4(cache["W_bind_inv"], B)
        bs = self.soma.batched_skinning
        bind_shape = self.soma._cached_rest_shape.expand(B, -1, -1)
        vertices = linear_blend_skinning(
            bind_shape,
            bs.get_bone_weights(),
            bs.get_bone_indices(),
            D,
        )
        return vertices, result
