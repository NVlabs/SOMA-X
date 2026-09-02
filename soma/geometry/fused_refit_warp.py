# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused Warp kernel for inverse-LBS refit: LBS + SE3_inv + src/tgt + covariance.

Replaces per-level: 2 LBS launches + ~10 PyTorch ops + 2 alignment launches
With: 1 fused launch + 1 Newton-Schulz launch = 2 kernel launches per level.
"""

import torch
import warp as wp

from soma.geometry._warp_utils import cache_warp_kernel, ensure_warp_initialized
from soma.geometry.transforms import rotation_matrices_are_valid


@cache_warp_kernel
def _create_fused_lbs_cov_kernel(K: int):
    """Create fused LBS + covariance kernel for a given sparse K.

    Each thread processes one (batch, vertex) pair:
    1. Subtree LBS: q = sum_k w_k * D_k * v
    2. Non-subtree LBS: c = sum_k w_k * D_k * v
    3. SE3_inverse: R^T, -R^T @ t for the vertex's owning joint
    4. src = R^T @ q + t_inv * sw
    5. tgt = target[vid] - c - p * sw
    6. Atomic accumulate outer(tgt, src) into per-joint covariance
    """
    weights_dtype = wp.types.vector(K, dtype=wp.float32)
    indices_dtype = wp.types.vector(K, dtype=wp.int32)

    @wp.kernel
    def fused_lbs_cov(
        # Per-vertex data (B, V_total) — batched for per-identity bind shapes
        bind_verts: wp.array2d(dtype=wp.vec3),
        sub_weights: wp.array(dtype=weights_dtype),
        sub_indices: wp.array(dtype=indices_dtype),
        non_weights: wp.array(dtype=weights_dtype),
        non_indices: wp.array(dtype=indices_dtype),
        sub_w_sum: wp.array(dtype=wp.float32),
        arm_vids: wp.array(dtype=wp.int32),
        joint_offsets: wp.array(dtype=wp.int32),
        joint_counts: wp.array(dtype=wp.int32),
        joint_id_per_vert: wp.array(dtype=wp.int32),
        # Per-joint data (J_level)
        joint_indices: wp.array(dtype=wp.int32),
        # Global matrices
        D: wp.array2d(dtype=wp.mat44f),  # (B, J) deformation
        W: wp.array2d(dtype=wp.mat44f),  # (B, J) world transforms
        target: wp.array2d(dtype=wp.vec3),  # (B, V_full)
        # Output: covariance accumulators (B * J_level * 9)
        cov_out: wp.array(dtype=wp.float32),
        J_level: wp.int32,
    ):
        batch_id, vert_id = wp.tid()

        v = bind_verts[batch_id, vert_id]

        # --- Subtree LBS ---
        ws = sub_weights[vert_id]
        si = sub_indices[vert_id]
        q = wp.vec3(0.0, 0.0, 0.0)
        for i in range(wp.static(K)):
            q = q + float(ws[i]) * wp.transform_point(D[batch_id, int(si[i])], v)

        # --- Non-subtree LBS ---
        wn = non_weights[vert_id]
        ni = non_indices[vert_id]
        c = wp.vec3(0.0, 0.0, 0.0)
        for i in range(wp.static(K)):
            c = c + float(wn[i]) * wp.transform_point(D[batch_id, int(ni[i])], v)

        # --- Joint world transform -> SE3_inverse ---
        jid_local = int(joint_id_per_vert[vert_id])
        jid_global = int(joint_indices[jid_local])
        W_j = W[batch_id, jid_global]

        # Extract R columns and t
        r00 = W_j[0, 0]
        r01 = W_j[0, 1]
        r02 = W_j[0, 2]
        r10 = W_j[1, 0]
        r11 = W_j[1, 1]
        r12 = W_j[1, 2]
        r20 = W_j[2, 0]
        r21 = W_j[2, 1]
        r22 = W_j[2, 2]
        tx = W_j[0, 3]
        ty = W_j[1, 3]
        tz = W_j[2, 3]

        # t_inv = -R^T @ t
        t_inv_x = -(r00 * tx + r10 * ty + r20 * tz)
        t_inv_y = -(r01 * tx + r11 * ty + r21 * tz)
        t_inv_z = -(r02 * tx + r12 * ty + r22 * tz)

        sw = sub_w_sum[vert_id]

        # src = R^T @ q + t_inv * sw
        src_x = r00 * q[0] + r10 * q[1] + r20 * q[2] + t_inv_x * sw
        src_y = r01 * q[0] + r11 * q[1] + r21 * q[2] + t_inv_y * sw
        src_z = r02 * q[0] + r12 * q[1] + r22 * q[2] + t_inv_z * sw

        # tgt = target[vid] - c - p * sw
        vid = int(arm_vids[vert_id])
        t_v = target[batch_id, vid]
        tgt_x = t_v[0] - c[0] - tx * sw
        tgt_y = t_v[1] - c[1] - ty * sw
        tgt_z = t_v[2] - c[2] - tz * sw

        # --- Atomic accumulate covariance: H[i,j] += tgt[i] * src[j] ---
        base = (batch_id * J_level + jid_local) * 9
        wp.atomic_add(cov_out, base + 0, tgt_x * src_x)
        wp.atomic_add(cov_out, base + 1, tgt_x * src_y)
        wp.atomic_add(cov_out, base + 2, tgt_x * src_z)
        wp.atomic_add(cov_out, base + 3, tgt_y * src_x)
        wp.atomic_add(cov_out, base + 4, tgt_y * src_y)
        wp.atomic_add(cov_out, base + 5, tgt_y * src_z)
        wp.atomic_add(cov_out, base + 6, tgt_z * src_x)
        wp.atomic_add(cov_out, base + 7, tgt_z * src_y)
        wp.atomic_add(cov_out, base + 8, tgt_z * src_z)

        # Match transforms.compute_covariance(..., virtual_normal=True).
        # The first vertex thread for each joint adds a synthetic normal
        # correspondence derived from the joint's first two point pairs.
        joint_start = int(joint_offsets[jid_local])
        joint_count = int(joint_counts[jid_local])
        if vert_id == joint_start and joint_count >= 2:
            vert_id_1 = joint_start + 1
            v1 = bind_verts[batch_id, vert_id_1]

            ws1 = sub_weights[vert_id_1]
            si1 = sub_indices[vert_id_1]
            q1 = wp.vec3(0.0, 0.0, 0.0)
            for i in range(wp.static(K)):
                q1 = q1 + float(ws1[i]) * wp.transform_point(D[batch_id, int(si1[i])], v1)

            wn1 = non_weights[vert_id_1]
            ni1 = non_indices[vert_id_1]
            c1 = wp.vec3(0.0, 0.0, 0.0)
            for i in range(wp.static(K)):
                c1 = c1 + float(wn1[i]) * wp.transform_point(D[batch_id, int(ni1[i])], v1)

            sw1 = sub_w_sum[vert_id_1]

            src1_x = r00 * q1[0] + r10 * q1[1] + r20 * q1[2] + t_inv_x * sw1
            src1_y = r01 * q1[0] + r11 * q1[1] + r21 * q1[2] + t_inv_y * sw1
            src1_z = r02 * q1[0] + r12 * q1[1] + r22 * q1[2] + t_inv_z * sw1

            vid1 = int(arm_vids[vert_id_1])
            t_v1 = target[batch_id, vid1]
            tgt1_x = t_v1[0] - c1[0] - tx * sw1
            tgt1_y = t_v1[1] - c1[1] - ty * sw1
            tgt1_z = t_v1[2] - c1[2] - tz * sw1

            tgt0 = wp.vec3(tgt_x, tgt_y, tgt_z)
            tgt1 = wp.vec3(tgt1_x, tgt1_y, tgt1_z)
            src0 = wp.vec3(src_x, src_y, src_z)
            src1 = wp.vec3(src1_x, src1_y, src1_z)

            n_tgt = wp.cross(tgt0, tgt1)
            n_src = wp.cross(src0, src1)
            len_n_tgt = wp.length(n_tgt)
            len_n_src = wp.length(n_src)

            if len_n_tgt > 1.0e-9 and len_n_src > 1.0e-9:
                v_tgt = n_tgt * (wp.length(tgt0) / (len_n_tgt + 1.0e-8))
                v_src = n_src * (wp.length(src0) / (len_n_src + 1.0e-8))

                wp.atomic_add(cov_out, base + 0, v_tgt[0] * v_src[0])
                wp.atomic_add(cov_out, base + 1, v_tgt[0] * v_src[1])
                wp.atomic_add(cov_out, base + 2, v_tgt[0] * v_src[2])
                wp.atomic_add(cov_out, base + 3, v_tgt[1] * v_src[0])
                wp.atomic_add(cov_out, base + 4, v_tgt[1] * v_src[1])
                wp.atomic_add(cov_out, base + 5, v_tgt[1] * v_src[2])
                wp.atomic_add(cov_out, base + 6, v_tgt[2] * v_src[0])
                wp.atomic_add(cov_out, base + 7, v_tgt[2] * v_src[1])
                wp.atomic_add(cov_out, base + 8, v_tgt[2] * v_src[2])

    return fused_lbs_cov


@wp.kernel
def newton_schulz_from_flat_cov(
    cov_flat: wp.array(dtype=wp.float32),  # (B * J_level * 9,)
    rotations: wp.array(dtype=wp.mat33),  # (B * J_level,)
):
    """Newton-Schulz polar decomposition on flat covariance buffer -> rotation matrix."""
    tid = wp.tid()
    base = tid * 9

    # fmt: off
    H = wp.mat33(
        cov_flat[base + 0], cov_flat[base + 1], cov_flat[base + 2],
        cov_flat[base + 3], cov_flat[base + 4], cov_flat[base + 5],
        cov_flat[base + 6], cov_flat[base + 7], cov_flat[base + 8],
    )
    # fmt: on

    # Scale by 1/infinity-norm for convergence guarantee
    row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
    row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
    row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
    max_sum = wp.max(row0_sum, wp.max(row1_sum, row2_sum))
    scale = 1.0 / wp.max(max_sum, 1.0e-8)

    R = H * scale

    # Order-2 Newton-Schulz: R_{k+1} = R_k * (3I - R_k^T R_k) / 2
    # 30 iterations handles condition ratios up to ~200K, preventing cascade
    # degradation through deep kinematic chains (fingers).
    for _ in range(30):
        RT_R = wp.transpose(R) * R
        # fmt: off
        term = wp.mat33(
            3.0 - RT_R[0, 0], -RT_R[0, 1], -RT_R[0, 2],
            -RT_R[1, 0], 3.0 - RT_R[1, 1], -RT_R[1, 2],
            -RT_R[2, 0], -RT_R[2, 1], 3.0 - RT_R[2, 2],
        )
        # fmt: on
        R = R * term * 0.5

    # Determinant correction (flip last column if det < 0)
    det = wp.determinant(R)
    sign_factor = wp.where(det < 0.0, -1.0, 1.0)
    # fmt: off
    R = wp.mat33(
        R[0, 0], R[0, 1], R[0, 2] * sign_factor,
        R[1, 0], R[1, 1], R[1, 2] * sign_factor,
        R[2, 0], R[2, 1], R[2, 2] * sign_factor,
    )
    # fmt: on

    rotations[tid] = R


@wp.kernel
def newton_schulz_auto_from_flat_cov(
    cov_flat: wp.array(dtype=wp.float32),  # (B * J_level * 9,)
    reference_rotations: wp.array(dtype=wp.mat33),  # (B * J_level,)
    rotations: wp.array(dtype=wp.mat33),  # (B * J_level,)
    prior_strength: wp.float32,
):
    """NS-first polar decomposition on reference-regularized covariance."""
    tid = wp.tid()
    base = tid * 9

    # fmt: off
    H = wp.mat33(
        cov_flat[base + 0], cov_flat[base + 1], cov_flat[base + 2],
        cov_flat[base + 3], cov_flat[base + 4], cov_flat[base + 5],
        cov_flat[base + 6], cov_flat[base + 7], cov_flat[base + 8],
    )
    # fmt: on

    row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
    row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
    row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
    prior_scale = wp.max(wp.max(row0_sum, wp.max(row1_sum, row2_sum)), 1.0e-8)
    volume_score = wp.abs(wp.determinant(H)) / (prior_scale * prior_scale * prior_scale)
    rank_weight = wp.clamp((0.02 - volume_score) / 0.02, 0.0, 1.0)
    R_ref = reference_rotations[tid]
    H = H + R_ref * (prior_strength * rank_weight * prior_scale)

    # A reflected covariance needs Kabsch's singular-vector correction.
    # Flipping a fixed Newton-Schulz output column can select the wrong rotation.
    if wp.determinant(H) < 0.0:
        U = wp.mat33()
        S = wp.vec3()
        V = wp.mat33()
        wp.svd3(H, U, S, V)
        # fmt: off
        correction = wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
        )
        # fmt: on
        rotations[tid] = U * correction * wp.transpose(V)
        return

    # Scale by 1/infinity-norm for convergence guarantee.
    row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
    row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
    row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
    max_sum = wp.max(row0_sum, wp.max(row1_sum, row2_sum))
    scale = 1.0 / wp.max(max_sum, 1.0e-8)

    R = H * scale

    for _ in range(30):
        RT_R = wp.transpose(R) * R
        # fmt: off
        term = wp.mat33(
            3.0 - RT_R[0, 0], -RT_R[0, 1], -RT_R[0, 2],
            -RT_R[1, 0], 3.0 - RT_R[1, 1], -RT_R[1, 2],
            -RT_R[2, 0], -RT_R[2, 1], 3.0 - RT_R[2, 2],
        )
        # fmt: on
        R = R * term * 0.5

    det = wp.determinant(R)
    sign_factor = wp.where(det < 0.0, -1.0, 1.0)
    # fmt: off
    R = wp.mat33(
        R[0, 0], R[0, 1], R[0, 2] * sign_factor,
        R[1, 0], R[1, 1], R[1, 2] * sign_factor,
        R[2, 0], R[2, 1], R[2, 2] * sign_factor,
    )
    # fmt: on

    rotations[tid] = R


@wp.kernel
def svd_from_flat_cov(
    cov_flat: wp.array(dtype=wp.float32),  # (B * J_level * 9,)
    rotations: wp.array(dtype=wp.mat33),  # (B * J_level,)
):
    """SVD (Kabsch) polar decomposition on flat covariance buffer -> rotation matrix."""
    tid = wp.tid()
    base = tid * 9

    # fmt: off
    H = wp.mat33(
        cov_flat[base + 0], cov_flat[base + 1], cov_flat[base + 2],
        cov_flat[base + 3], cov_flat[base + 4], cov_flat[base + 5],
        cov_flat[base + 6], cov_flat[base + 7], cov_flat[base + 8],
    )
    # fmt: on

    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(H, U, S, V)

    R = U * wp.transpose(V)

    if wp.determinant(R) < 0.0:
        # fmt: off
        correction = wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
        )
        # fmt: on
        R = U * correction * wp.transpose(V)

    rotations[tid] = R


@wp.kernel
def svd_auto_from_flat_cov(
    cov_flat: wp.array(dtype=wp.float32),  # (B * J_level * 9,)
    reference_rotations: wp.array(dtype=wp.mat33),  # (B * J_level,)
    rotations: wp.array(dtype=wp.mat33),  # (B * J_level,)
    prior_strength: wp.float32,
):
    """SVD projection on the same reference-regularized covariance as auto NS."""
    tid = wp.tid()
    base = tid * 9

    # fmt: off
    H = wp.mat33(
        cov_flat[base + 0], cov_flat[base + 1], cov_flat[base + 2],
        cov_flat[base + 3], cov_flat[base + 4], cov_flat[base + 5],
        cov_flat[base + 6], cov_flat[base + 7], cov_flat[base + 8],
    )
    # fmt: on

    row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
    row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
    row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
    prior_scale = wp.max(wp.max(row0_sum, wp.max(row1_sum, row2_sum)), 1.0e-8)
    volume_score = wp.abs(wp.determinant(H)) / (prior_scale * prior_scale * prior_scale)
    rank_weight = wp.clamp((0.02 - volume_score) / 0.02, 0.0, 1.0)
    H = H + reference_rotations[tid] * (prior_strength * rank_weight * prior_scale)

    U = wp.mat33()
    S = wp.vec3()
    V = wp.mat33()
    wp.svd3(H, U, S, V)

    R = U * wp.transpose(V)

    if wp.determinant(R) < 0.0:
        # fmt: off
        correction = wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
        )
        # fmt: on
        R = U * correction * wp.transpose(V)

    rotations[tid] = R


def fused_refit_level(
    bind_verts_cat,  # (V_total, 3) or (B, V_total, 3) float32
    sub_bw_cat,  # (V_total, K) float32
    sub_bi_cat,  # (V_total, K) int32
    non_bw_cat,  # (V_total, K) float32/int32
    non_bi_cat,  # (V_total, K) int32
    sub_w_sum_cat,  # (V_total,) float32
    arm_vids_cat,  # (V_total,) int32/long
    joint_offsets,  # (J_level,) int32/long
    joint_counts,  # (J_level,) int32/long
    joint_id_per_vert,  # (V_total,) long
    joint_indices,  # (J_level,) long
    D,  # (B, J, 4, 4) float32
    W,  # (B, J, 4, 4) float32
    target,  # (B, V_full, 3) float32
    J_level,  # int
    rotation_method="auto",  # "auto", "newton-schulz", or "svd"
    reference_rotations=None,  # (B, J_level, 3, 3), required for auto
    auto_prior_strength=0.05,
):
    """Run fused LBS + covariance + rotation projection for one skeleton level.

    ``bind_verts_cat`` can be unbatched ``(V_total, 3)`` (single identity)
    or batched ``(B, V_total, 3)`` (per-sample identity).  Unbatched is
    expanded to ``(B, V_total, 3)`` for the kernel.

    Returns: R_all (B, J_level, 3, 3) world-space rotations.
    """
    ensure_warp_initialized()

    B = D.shape[0]
    K = sub_bw_cat.shape[1]
    device = wp.device_from_torch(D.device)
    if rotation_method == "auto" and reference_rotations is None:
        raise ValueError("reference_rotations is required for rotation_method='auto'.")

    # Ensure bind_verts is (B, V_total, 3) for the 2D kernel
    if bind_verts_cat.ndim == 2:
        bind_verts_cat = bind_verts_cat.unsqueeze(0).expand(B, -1, -1)
    V_total = bind_verts_cat.shape[1]

    fused_kernel = _create_fused_lbs_cov_kernel(K)

    # Convert to warp arrays (zero-copy from contiguous torch tensors)
    wp_bind = wp.from_torch(bind_verts_cat.contiguous(), dtype=wp.vec3)
    weights_dtype = wp.types.vector(K, dtype=wp.float32)
    indices_dtype = wp.types.vector(K, dtype=wp.int32)
    wp_sub_w = wp.from_torch(sub_bw_cat.contiguous(), dtype=weights_dtype)
    wp_sub_i = wp.from_torch(sub_bi_cat.to(torch.int32).contiguous(), dtype=indices_dtype)
    wp_non_w = wp.from_torch(non_bw_cat.contiguous(), dtype=weights_dtype)
    wp_non_i = wp.from_torch(non_bi_cat.to(torch.int32).contiguous(), dtype=indices_dtype)
    wp_sw_sum = wp.from_torch(sub_w_sum_cat.contiguous(), dtype=wp.float32)
    wp_vids = wp.from_torch(arm_vids_cat.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_offsets = wp.from_torch(joint_offsets.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_counts = wp.from_torch(joint_counts.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_jid = wp.from_torch(joint_id_per_vert.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_ji = wp.from_torch(joint_indices.to(torch.int32).contiguous(), dtype=wp.int32)
    wp_D = wp.from_torch(D.contiguous(), dtype=wp.mat44f)
    wp_W = wp.from_torch(W.contiguous(), dtype=wp.mat44f)
    wp_target = wp.from_torch(target.contiguous(), dtype=wp.vec3)
    wp_reference = None
    if reference_rotations is not None:
        wp_reference = wp.from_torch(
            reference_rotations.contiguous().view(B * J_level, 3, 3),
            dtype=wp.mat33,
        )

    # Allocate covariance buffer: B * J_level * 9 floats, zeroed
    cov_size = B * J_level * 9
    wp_cov = wp.zeros(cov_size, dtype=wp.float32, device=device)

    # Allocate rotation output
    wp_rot = wp.zeros(B * J_level, dtype=wp.mat33, device=device)

    # Launch fused kernel
    wp.launch(
        fused_kernel,
        dim=(B, V_total),
        inputs=[
            wp_bind,
            wp_sub_w,
            wp_sub_i,
            wp_non_w,
            wp_non_i,
            wp_sw_sum,
            wp_vids,
            wp_offsets,
            wp_counts,
            wp_jid,
            wp_ji,
            wp_D,
            wp_W,
            wp_target,
            wp_cov,
            J_level,
        ],
        device=device,
    )

    # Launch rotation kernel
    if rotation_method == "svd":
        wp.launch(
            svd_from_flat_cov,
            dim=B * J_level,
            inputs=[wp_cov, wp_rot],
            device=device,
        )
    elif rotation_method == "auto":
        wp.launch(
            newton_schulz_auto_from_flat_cov,
            dim=B * J_level,
            inputs=[wp_cov, wp_reference, wp_rot, float(auto_prior_strength)],
            device=device,
        )
    else:
        wp.launch(
            newton_schulz_from_flat_cov,
            dim=B * J_level,
            inputs=[wp_cov, wp_rot],
            device=device,
        )

    # Convert back to torch
    R_all = wp.to_torch(wp_rot).view(B, J_level, 3, 3)

    if rotation_method != "svd":
        valid = rotation_matrices_are_valid(R_all)
        if not torch.all(valid):
            wp_rot_svd = wp.zeros(B * J_level, dtype=wp.mat33, device=device)
            if rotation_method == "auto":
                wp.launch(
                    svd_auto_from_flat_cov,
                    dim=B * J_level,
                    inputs=[wp_cov, wp_reference, wp_rot_svd, float(auto_prior_strength)],
                    device=device,
                )
            else:
                wp.launch(
                    svd_from_flat_cov,
                    dim=B * J_level,
                    inputs=[wp_cov, wp_rot_svd],
                    device=device,
                )
            R_svd = wp.to_torch(wp_rot_svd).view(B, J_level, 3, 3)
            R_all = torch.where(valid[..., None, None], R_all, R_svd)

    return R_all
