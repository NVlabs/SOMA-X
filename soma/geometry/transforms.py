# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotation and rigid-transform helpers used across SOMA geometry modules."""

from typing import Literal

import torch
import torch.nn.functional as F

# ============================================================================
# Modular Rotation Estimation Functions
# ============================================================================

AlignmentMethod = Literal["kabsch", "newton-schulz", "auto"]
NEWTON_SCHULZ_ITERS = 30
AUTO_ROTATION_PRIOR_STRENGTH = 0.05
AUTO_ROTATION_RANK_THRESHOLD = 2e-2
AUTO_ROTATION_DEGENERATE_THRESHOLD = 1e-6


def compute_covariance(
    A: torch.Tensor,
    B: torch.Tensor,
    virtual_normal: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute covariance matrix H = A^T @ B for rotation estimation.

    Args:
        A: Target vectors (..., N, 3)
        B: Source vectors (..., N, 3)
        virtual_normal: If True, add synthetic normal correspondence for conditioning
        eps: Small constant for numerical stability

    Returns:
        H: Covariance matrix (..., 3, 3)
    """
    # Basic covariance: H = A^T @ B
    H = torch.einsum("...ni,...nj->...ij", A, B)

    # Virtual normal fix: add synthetic correspondence from cross product
    if virtual_normal and A.shape[-2] >= 2:
        p0, p1 = A[..., 0, :], A[..., 1, :]
        q0, q1 = B[..., 0, :], B[..., 1, :]

        # Compute normal direction (cross product)
        n_src = torch.cross(p0, p1, dim=-1)
        n_dst = torch.cross(q0, q1, dim=-1)

        # Normalize and scale by point cloud radius
        len_n_src = torch.linalg.norm(n_src, dim=-1, keepdim=True)
        len_n_dst = torch.linalg.norm(n_dst, dim=-1, keepdim=True)
        scale_src = torch.linalg.norm(p0, dim=-1, keepdim=True) / (len_n_src + eps)
        scale_dst = torch.linalg.norm(q0, dim=-1, keepdim=True) / (len_n_dst + eps)

        # Check for collinearity
        valid_normal = (len_n_src[..., 0] > 1e-9) & (len_n_dst[..., 0] > 1e-9)

        # Virtual normal vectors
        v_src = n_src * scale_src
        v_dst = n_dst * scale_dst

        # Add virtual correspondence (only for valid normals)
        if torch.any(valid_normal):
            virtual_contrib = torch.einsum("...i,...j->...ij", v_src, v_dst)
            mask = valid_normal[..., None, None].expand(H.shape)
            virtual_contrib = torch.where(mask, virtual_contrib, 0.0)
            H = H + virtual_contrib

    return H


def kabsch(H: torch.Tensor) -> torch.Tensor:
    """Compute rotation matrix from covariance using Kabsch algorithm (SVD).

    Args:
        H: Covariance matrix (..., 3, 3)

    Returns:
        R: Rotation matrix (..., 3, 3) with det(R) = 1
    """
    U, S, Vh = torch.linalg.svd(H)
    I3 = torch.eye(3, dtype=H.dtype, device=H.device)

    # Compute correction for determinant
    UVt = U @ Vh.swapaxes(-2, -1)
    det_sign = torch.where(torch.linalg.det(UVt) < 0, -1.0, 1.0)

    # Apply correction
    Dcorr = I3.expand(H.shape).clone()
    Dcorr[..., -1, -1] = det_sign
    R = U @ Dcorr @ Vh

    return R


def newton_schulz(H: torch.Tensor, num_iters: int = 30, eps: float = 1e-8) -> torch.Tensor:
    """Compute rotation matrix from covariance using Newton-Schulz iteration.

    This is primarily a reference implementation for testing and comparing against
    the Warp-accelerated Newton-Schulz kernel. Production callers should pair
    Newton-Schulz with validity checks or a reference-gauge policy when the
    covariance is rank deficient.

    Args:
        H: Covariance matrix (..., 3, 3)
        num_iters: Number of iterations (default 30)
        eps: Small constant for numerical stability

    Returns:
        R: Rotation matrix (..., 3, 3) with det(R) = 1

    Note:
        Convergence depends on conditioning of H. Ill-conditioned matrices may
        require more iterations or may not converge to high precision.
    """
    # Scale by infinity norm (max absolute row sum) for guaranteed convergence
    row_sums = torch.abs(H).sum(dim=-1)
    max_row_sum = row_sums.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    R = H / (max_row_sum + eps)

    I3 = torch.eye(3, dtype=H.dtype, device=H.device)
    I3_batch = I3.expand(H.shape)

    # Newton-Schulz iteration: R_{k+1} = R_k * (3*I - R_k^T * R_k) / 2
    for _ in range(num_iters):
        RT_R = R.swapaxes(-2, -1) @ R
        term = 3.0 * I3_batch - RT_R
        R = R @ term * 0.5

    # Differentiable determinant correction
    det_R = torch.linalg.det(R)
    sign_factor = torch.where(det_R < 0, -1.0, 1.0)

    # Apply sign correction to last column
    R_corrected = R.clone()
    R_corrected[..., :, 2] = R[..., :, 2] * sign_factor[..., None]

    return R_corrected


def regularize_covariance_with_reference(
    H: torch.Tensor,
    reference_rotation: torch.Tensor | None = None,
    prior_strength: float = AUTO_ROTATION_PRIOR_STRENGTH,
    rank_threshold: float = AUTO_ROTATION_RANK_THRESHOLD,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Add a weak reference-gauge prior to a Procrustes covariance matrix."""
    prior_scale = torch.abs(H).sum(dim=-1).amax(dim=-1).clamp_min(eps)
    volume_score = torch.linalg.det(H).abs() / prior_scale.pow(3)
    rank_weight = ((rank_threshold - volume_score) / rank_threshold).clamp(0.0, 1.0)
    if reference_rotation is None:
        reference_rotation = torch.eye(3, dtype=H.dtype, device=H.device).expand(H.shape)
    return H + (prior_strength * rank_weight * prior_scale)[..., None, None] * reference_rotation


def rotation_matrices_are_valid(
    R: torch.Tensor,
    det_tol: float = 1e-2,
    orthogonality_tol: float = 1e-2,
) -> torch.Tensor:
    """Return a boolean mask for finite right-handed orthonormal rotations."""
    finite = torch.isfinite(R).all(dim=(-2, -1))
    det_R = torch.linalg.det(R)
    det_valid = torch.isfinite(det_R) & (det_R > 0.0) & ((det_R - 1.0).abs() <= det_tol)

    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    ortho_err = (R.swapaxes(-2, -1) @ R - eye).abs().amax(dim=(-2, -1))
    ortho_valid = torch.isfinite(ortho_err) & (ortho_err <= orthogonality_tol)
    return finite & det_valid & ortho_valid


def project_rotations_to_so3(rotations: torch.Tensor) -> torch.Tensor:
    """Project matrices to the nearest proper rotations."""
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {rotations.shape}")

    u, _, vh = torch.linalg.svd(rotations)
    projected = u @ vh
    det = torch.linalg.det(projected)
    det_mask = det < 0
    if torch.any(det_mask).item():
        u = u.clone()
        u[..., :, -1] = torch.where(det_mask[..., None], -u[..., :, -1], u[..., :, -1])
        projected = u @ vh
    return projected


def rodrigues_rotation(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute rotation matrix that aligns vector b to vector a.

    Uses the shortest arc rotation approach similar to SciPy's align_vectors.

    Args:
        a: Target vector (..., 3)
        b: Source vector (..., 3)
        eps: Small constant for numerical stability

    Returns:
        R: Rotation matrix (..., 3, 3) such that R @ b ≈ a
    """
    dtype, device = a.dtype, a.device

    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True)

    a_u = a / torch.clamp(a_norm, min=eps)
    b_u = b / torch.clamp(b_norm, min=eps)

    dot = torch.clamp((a_u * b_u).sum(dim=-1, keepdim=True), -1.0, 1.0)
    v = torch.cross(b_u, a_u, dim=-1)

    zeros = torch.zeros_like(v[..., 0])
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]

    skew_v = torch.stack(
        [
            torch.stack([zeros, -vz, vy], dim=-1),
            torch.stack([vz, zeros, -vx], dim=-1),
            torch.stack([-vy, vx, zeros], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, dtype=dtype, device=device).expand(a.shape[:-1] + (3, 3))

    factor = 1.0 / (1.0 + dot[..., None])
    R = eye + skew_v + factor * (skew_v @ skew_v)

    # Handle Antiparallel Case (180 degree rotation)
    antiparallel_mask = dot[..., 0] < -1.0 + 1e-6

    if torch.any(antiparallel_mask):
        b_anti = b_u[antiparallel_mask]

        basis_shape = b_anti.shape[:-1] + (3,)
        y_vec = torch.zeros(basis_shape, dtype=dtype, device=device)
        y_vec[..., 1] = 1.0
        x_vec = torch.zeros(basis_shape, dtype=dtype, device=device)
        x_vec[..., 0] = 1.0

        w = torch.where((torch.abs(b_anti[..., 0]) > 0.6)[..., None], y_vec, x_vec)

        axis_180 = torch.cross(b_anti, w, dim=-1)
        axis_180 = axis_180 / torch.linalg.norm(axis_180, dim=-1, keepdim=True)

        u_mat = axis_180[..., :, None] * axis_180[..., None, :]
        eye_3 = torch.eye(3, dtype=dtype, device=device)
        R_180 = 2.0 * u_mat - eye_3

        R[antiparallel_mask] = R_180

    return R


# ============================================================================
# High-Level Alignment Function
# ============================================================================


def align_vectors(
    A: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-8,
    method: AlignmentMethod = "auto",
) -> torch.Tensor:
    """
    SciPy-compatible: return rotation C such that C @ b ≈ a.
    Supports broadcasting across leading batch dims. Inputs: (..., N, 3).

    Args:
        A: Target vectors (..., N, 3)
        B: Source vectors (..., N, 3)
        eps: Small constant for numerical stability
        method: 'auto', 'kabsch' (SVD-based), or 'newton-schulz' (iterative)
    """
    if A.shape[-1] != 3 or B.shape[-1] != 3:
        raise NotImplementedError("Only 3D vectors are supported (last dim must be 3).")
    if A.shape[-2] != B.shape[-2]:
        raise ValueError(f"N must match, got {A.shape[-2]} vs {B.shape[-2]}.")

    N = A.shape[-2]

    if N == 1:
        return rodrigues_rotation(A[..., 0, :], B[..., 0, :], eps=eps)

    H = compute_covariance(A, B, virtual_normal=True, eps=eps)

    if method == "newton-schulz":
        R = newton_schulz(H, num_iters=NEWTON_SCHULZ_ITERS, eps=eps)
        valid = rotation_matrices_are_valid(R)
        if torch.all(valid):
            return R
        return torch.where(valid[..., None, None], R, kabsch(H))
    elif method == "auto":
        H_auto = regularize_covariance_with_reference(
            H,
            rank_threshold=AUTO_ROTATION_DEGENERATE_THRESHOLD,
            eps=eps,
        )
        R = newton_schulz(H_auto, num_iters=NEWTON_SCHULZ_ITERS, eps=eps)
        valid = rotation_matrices_are_valid(R)
        needs_kabsch = ~valid | (torch.linalg.det(H_auto) < 0)
        if not torch.any(needs_kabsch):
            return R
        return torch.where(needs_kabsch[..., None, None], kabsch(H_auto), R)
    elif method == "kabsch":
        return kabsch(H)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'kabsch', or 'newton-schulz'.")


def SE3_from_Rt(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    autograd-safe SE(3) transform construction from rotation R and translation t.
    R: (..., 3, 3)
    t: (..., 3)
    Returns: T (..., 4, 4)
    """
    dtype, device = R.dtype, R.device
    upper = torch.cat([R, t[..., None]], dim=-1)  # (..., 3, 4)
    last_row = torch.cat(
        [
            torch.zeros((*upper.shape[:-2], 1, 3), dtype=dtype, device=device),
            torch.ones((*upper.shape[:-2], 1, 1), dtype=dtype, device=device),
        ],
        dim=-1,
    )  # (..., 1, 4)
    return torch.cat([upper, last_row], dim=-2)  # (..., 4, 4)


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    """
    Invert SE(3) transform(s) in homogeneous coordinates.

    Args:
        T: (..., 4, 4) torch.Tensor
    Returns:
        Tinv: (..., 4, 4)
    """
    R = T[..., :3, :3]  # (..., 3, 3)
    t = T[..., :3, 3:4]  # (..., 3, 1)
    R_T = R.swapaxes(-2, -1)  # (..., 3, 3)
    t_new = -(R_T @ t)  # (..., 3, 1)

    Tinv = SE3_from_Rt(R_T, t_new[..., 0])  # (..., 4, 4)
    return Tinv


# --- SO(3) conversions --------------------------------------------------------


def euler_xyz_to_matrix(euler_xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to rotation matrices.

    Args:
        euler_xyz: (..., 3) angles in radians, ordered as X, Y, Z.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    if euler_xyz.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {euler_xyz.shape}")

    cos_angles = torch.cos(euler_xyz)
    sin_angles = torch.sin(euler_xyz)
    cx, cy, cz = cos_angles[..., 0], cos_angles[..., 1], cos_angles[..., 2]
    sx, sy, sz = sin_angles[..., 0], sin_angles[..., 1], sin_angles[..., 2]
    return torch.stack(
        [
            cy * cz,
            -cx * sz + sx * sy * cz,
            sx * sz + cx * sy * cz,
            cy * sz,
            cx * cz + sx * sy * sz,
            -sx * cz + cx * sy * sz,
            -sy,
            sx * cy,
            cx * cy,
        ],
        dim=-1,
    ).reshape(euler_xyz.shape[:-1] + (3, 3))


def matrix_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to XYZ Euler angles.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) angles in radians, ordered as X, Y, Z.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {R.shape}")

    sy = -R[..., 2, 0].clamp(-1.0, 1.0)
    y = torch.asin(sy)
    x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    z = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return torch.stack([x, y, z], dim=-1)


def quaternion_normalize_xyzw(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize XYZW quaternions.

    Args:
        quaternion: (..., 4) quaternions ordered as x, y, z, w.
        eps: Small constant used when normalizing quaternions.

    Returns:
        (..., 4) unit quaternions ordered as x, y, z, w.
    """
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected (...,4), got {quaternion.shape}")
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_standardize_xyzw(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize XYZW quaternions and choose the non-negative-w representative."""
    q = quaternion_normalize_xyzw(quaternion, eps=eps)
    return torch.where(q[..., 3:] < 0.0, -q, q)


def quaternion_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of XYZW quaternions."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected (...,4), got {quaternion.shape}")
    out = quaternion.clone()
    out[..., :3] = -out[..., :3]
    return out


def quaternion_multiply_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of XYZW quaternions.

    Args:
        a: (..., 4) left quaternions ordered as x, y, z, w.
        b: (..., 4) right quaternions ordered as x, y, z, w.

    Returns:
        (..., 4) product quaternions ordered as x, y, z, w.
    """
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) operands, got {a.shape} and {b.shape}")
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def quaternion_log_xyzw(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Map XYZW unit quaternions to rotation vectors using the shortest arc."""
    q = quaternion_standardize_xyzw(quaternion, eps=eps)
    vec = q[..., :3]
    w = q[..., 3].clamp(-1.0, 1.0)
    vec_norm = torch.linalg.norm(vec, dim=-1)
    angle = 2.0 * torch.atan2(vec_norm, w)

    small = vec_norm < eps
    factor = torch.where(
        small,
        2.0 / w.clamp_min(eps),
        angle / vec_norm.clamp_min(eps),
    )
    return vec * factor[..., None]


def quaternion_exp_xyzw(rotvec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Map rotation vectors to XYZW unit quaternions."""
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {rotvec.shape}")

    theta = torch.linalg.norm(rotvec, dim=-1)
    half_theta = 0.5 * theta
    theta2 = theta * theta
    theta4 = theta2 * theta2
    small = theta < 1e-6
    imag_scale = torch.where(
        small,
        0.5 - theta2 / 48.0 + theta4 / 3840.0,
        torch.sin(half_theta) / theta.clamp_min(eps),
    )
    quaternion = torch.cat(
        (rotvec * imag_scale[..., None], torch.cos(half_theta)[..., None]),
        dim=-1,
    )
    return quaternion_standardize_xyzw(quaternion, eps=eps)


def quaternion_xyzw_to_matrix(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert XYZW quaternions to rotation matrices.

    Args:
        quaternion: (..., 4) quaternions ordered as x, y, z, w.
        eps: Small constant used when normalizing quaternions.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = quaternion_normalize_xyzw(quaternion, eps=eps)
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion_xyzw(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert rotation matrices to XYZW unit quaternions.

    Args:
        R: (..., 3, 3) rotation matrices.
        eps: Small constant used when normalizing quaternions.

    Returns:
        (..., 4) quaternions ordered as x, y, z, w with non-negative w.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {R.shape}")

    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(0.0))
    qx = 0.5 * torch.copysign(
        torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(0.0)),
        m21 - m12,
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(0.0)),
        m02 - m20,
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(0.0)),
        m10 - m01,
    )
    return quaternion_normalize_xyzw(torch.stack((qx, qy, qz, qw), dim=-1), eps=eps)


def matrix_to_quaternion_xyzw_stable(
    R: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert rotation matrices to XYZW quaternions with finite sqrt gradients.

    This follows the same signed-branch convention as ``matrix_to_quaternion_xyzw``
    but adds a tiny value inside each square root.  It is intended for code paths
    that need gradients through matrix-to-quaternion conversion near zero-valued
    branches.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {R.shape}")

    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(0.0) + eps)
    qx = 0.5 * torch.copysign(
        torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(0.0) + eps),
        m21 - m12,
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(0.0) + eps),
        m02 - m20,
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(0.0) + eps),
        m10 - m01,
    )
    return quaternion_normalize_xyzw(torch.stack((qx, qy, qz, qw), dim=-1), eps=eps)


def single_axis_rotation_matrices(
    angles: torch.Tensor,
    axis: int,
    axis_signs: torch.Tensor,
) -> torch.Tensor:
    """Create rotation matrices for angles around one shared local axis."""
    signed_angles = angles * axis_signs.to(dtype=angles.dtype, device=angles.device)[None]
    c = torch.cos(signed_angles)
    s = torch.sin(signed_angles)
    z = torch.zeros_like(signed_angles)
    o = torch.ones_like(signed_angles)
    if axis == 0:
        values = (o, z, z, z, c, -s, z, s, c)
    elif axis == 1:
        values = (c, z, s, z, o, z, -s, z, c)
    elif axis == 2:
        values = (c, -s, z, s, c, z, z, z, o)
    else:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    return torch.stack(values, dim=-1).reshape(*angles.shape, 3, 3)


def quaternion_half_angle_xyzw(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the principal half-angle quaternion for XYZW rotations.

    ``q`` and ``-q`` represent the same rotation. This helper first chooses the
    representation with non-negative ``w`` and then computes the normalized
    square-root quaternion ``sqrt(q)`` using ``[v, w + 1]``. This is useful when
    extracting twist angles near 180 degrees without an unstable full-angle
    projection.
    """
    q = quaternion_standardize_xyzw(quaternion, eps=eps)
    return quaternion_normalize_xyzw(
        torch.cat((q[..., :3], q[..., 3:] + 1.0), dim=-1),
        eps=eps,
    )


def quaternion_twist_angle_xyzw(
    quaternion: torch.Tensor,
    axis_ids: int | torch.Tensor = 0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Extract signed twist angles around local axes from XYZW quaternions.

    Args:
        quaternion: (..., 4) quaternions ordered as x, y, z, w.
        axis_ids: scalar axis id or tensor broadcastable to ``quaternion.shape[:-1]``.
            Axis ids use ``0=x``, ``1=y``, ``2=z``.
        eps: Small constant used when normalizing quaternions.

    Returns:
        ``quaternion.shape[:-1]`` twist angles in radians.
    """
    q_half = quaternion_half_angle_xyzw(quaternion, eps=eps)
    if isinstance(axis_ids, torch.Tensor):
        axis_ids_t = axis_ids.to(device=quaternion.device, dtype=torch.long)
    else:
        axis_ids_t = torch.tensor(axis_ids, dtype=torch.long, device=quaternion.device)
    if torch.any((axis_ids_t < 0) | (axis_ids_t > 2)):
        raise ValueError("axis_ids must contain only 0, 1, or 2")
    if axis_ids_t.ndim == 0:
        twist_imag = q_half[..., int(axis_ids_t.item())]
    else:
        while axis_ids_t.ndim < q_half.ndim - 1:
            axis_ids_t = axis_ids_t.unsqueeze(0)
        try:
            gather_ids = axis_ids_t.expand(q_half.shape[:-1]).unsqueeze(-1)
        except RuntimeError as e:
            raise ValueError(
                "axis_ids must be broadcastable to quaternion.shape[:-1], "
                f"got {tuple(axis_ids_t.shape)} for {tuple(q_half.shape[:-1])}"
            ) from e
        twist_imag = q_half[..., :3].gather(-1, gather_ids).squeeze(-1)
    return 4.0 * torch.atan2(twist_imag, q_half[..., 3])


def matrix_to_rotvec(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    (...,3,3) rotation matrices -> (...,3) rotation vectors (axis * angle).
    Robust for small angles and near-pi.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {R.shape}")

    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    S = R - R.swapaxes(-2, -1)
    v = torch.stack(
        [
            S[..., 2, 1] - S[..., 1, 2],
            S[..., 0, 2] - S[..., 2, 0],
            S[..., 1, 0] - S[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = 0.5 * torch.linalg.norm(v, dim=-1)

    # Regions
    small = theta <= 1e-3
    near_pi = theta >= (torch.pi - 1e-3)

    # Small-angle series
    theta2_approx = torch.clamp(3.0 - tr, min=0.0)
    factor_small = 0.5 + theta2_approx / 12.0
    w_small = v * factor_small[..., None]

    # Generic
    denom = torch.where(sin_theta < eps, eps, 2.0 * sin_theta)
    factor_gen = theta / denom
    w_gen = v * factor_gen[..., None]

    # Near-pi: axis from diagonals + sign from v
    R00, R11, R22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    u0 = torch.sqrt(torch.clamp((R00 - R11 - R22 + 1.0) * 0.5, min=0.0))
    u1 = torch.sqrt(torch.clamp((-R00 + R11 - R22 + 1.0) * 0.5, min=0.0))
    u2 = torch.sqrt(torch.clamp((-R00 - R11 + R22 + 1.0) * 0.5, min=0.0))
    u = torch.stack([u0, u1, u2], dim=-1)

    sx, sy, sz = torch.sign(v[..., 0]), torch.sign(v[..., 1]), torch.sign(v[..., 2])
    sx = torch.where(sx == 0, 1, sx)
    sy = torch.where(sy == 0, 1, sy)
    sz = torch.where(sz == 0, 1, sz)
    u = torch.stack([u[..., 0] * sx, u[..., 1] * sy, u[..., 2] * sz], dim=-1)

    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)
    u_norm = torch.where(u_norm < eps, eps, u_norm)
    axis_pi = u / u_norm
    w_pi = axis_pi * theta[..., None]

    return torch.where(near_pi[..., None], w_pi, torch.where(small[..., None], w_small, w_gen))


def rotvec_to_matrix(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    (...,3) rotation vectors -> (...,3,3) rotation matrices.
    Robust near zero.
    """
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected (...,3), got {rotvec.shape}")

    theta = torch.linalg.norm(rotvec, dim=-1)
    denom = torch.where(theta < eps, eps, theta)[..., None]
    axis = rotvec / denom

    K = torch.zeros(rotvec.shape[:-1] + (3, 3), dtype=rotvec.dtype, device=rotvec.device)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    A = sin_t / torch.where(theta < eps, 1.0, theta)
    B = (1.0 - cos_t) / torch.where(theta < eps, 1.0, theta * theta)

    R = eye + A[..., None, None] * K + B[..., None, None] * (K @ K)

    small = theta < 1e-6
    return torch.where(small[..., None, None], eye + K, R)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation by Zhou et al. [1] to rotation matrix.

    Uses Gram-Schmidt orthogonalization per Section B of [1].

    Args:
        d6: 6D rotation representation, of size ``(*, 6)``.

    Returns:
        Batch of rotation matrices of size ``(*, 3, 3)``.

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
