# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Warp-based implementations of geometric transformations for improved GPU performance.
"""

from typing import Literal

import torch
import warp as wp

from soma.geometry._warp_utils import cache_warp_kernel, ensure_warp_initialized

AlignmentMethod = Literal["auto", "kabsch", "newton-schulz"]


def _get_warp_types(torch_dtype):
    """Map PyTorch dtype to corresponding Warp scalar, vector, and matrix types."""
    if torch_dtype == torch.float16:
        return wp.float16, wp.vec3h, wp.mat33h
    elif torch_dtype == torch.float64:
        return wp.float64, wp.vec3d, wp.mat33d
    else:  # float32 or default
        return wp.float32, wp.vec3, wp.mat33


@cache_warp_kernel
def _create_covariance_kernel(dtype_scalar, dtype_vec, dtype_mat):
    """Factory to create covariance kernel for specific dtypes."""

    @wp.kernel
    def compute_covariance_matrices(
        flat_a: wp.array(dtype=dtype_vec),
        flat_b: wp.array(dtype=dtype_vec),
        offsets: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        cov_matrices: wp.array(dtype=dtype_mat),
    ):
        """
        Compute covariance matrix A^T @ B for each batch.
        This kernel accumulates the covariance matrix elements.
        Separate from SVD computation to enable autodiff.
        """
        tid = wp.tid()

        start_idx = offsets[tid]
        count = counts[tid]

        if count == 0:
            # fmt: off
            cov_matrices[tid] = dtype_mat(
                dtype_scalar(0.0), dtype_scalar(0.0), dtype_scalar(0.0),
                dtype_scalar(0.0), dtype_scalar(0.0), dtype_scalar(0.0),
                dtype_scalar(0.0), dtype_scalar(0.0), dtype_scalar(0.0)
            )
            # fmt: on
            return

        # Accumulate into scalar variables (not matrix)
        # This is necessary for Warp's autodiff to work properly
        c00 = dtype_scalar(0.0)
        c01 = dtype_scalar(0.0)
        c02 = dtype_scalar(0.0)
        c10 = dtype_scalar(0.0)
        c11 = dtype_scalar(0.0)
        c12 = dtype_scalar(0.0)
        c20 = dtype_scalar(0.0)
        c21 = dtype_scalar(0.0)
        c22 = dtype_scalar(0.0)

        for i in range(count):
            idx = start_idx + i
            vec_a = flat_a[idx]
            vec_b = flat_b[idx]

            # Compute A^T @ B to get R such that R @ B ≈ A (scipy convention)
            c00 += vec_a[0] * vec_b[0]
            c01 += vec_a[0] * vec_b[1]
            c02 += vec_a[0] * vec_b[2]

            c10 += vec_a[1] * vec_b[0]
            c11 += vec_a[1] * vec_b[1]
            c12 += vec_a[1] * vec_b[2]

            c20 += vec_a[2] * vec_b[0]
            c21 += vec_a[2] * vec_b[1]
            c22 += vec_a[2] * vec_b[2]

        if count >= 2:
            a0 = flat_a[start_idx]
            a1 = flat_a[start_idx + 1]
            b0 = flat_b[start_idx]
            b1 = flat_b[start_idx + 1]

            n_a = wp.cross(a0, a1)
            n_b = wp.cross(b0, b1)
            len_n_a = wp.length(n_a)
            len_n_b = wp.length(n_b)

            if len_n_a > dtype_scalar(1.0e-9) and len_n_b > dtype_scalar(1.0e-9):
                v_a = n_a * (wp.length(a0) / (len_n_a + dtype_scalar(1.0e-8)))
                v_b = n_b * (wp.length(b0) / (len_n_b + dtype_scalar(1.0e-8)))

                c00 += v_a[0] * v_b[0]
                c01 += v_a[0] * v_b[1]
                c02 += v_a[0] * v_b[2]

                c10 += v_a[1] * v_b[0]
                c11 += v_a[1] * v_b[1]
                c12 += v_a[1] * v_b[2]

                c20 += v_a[2] * v_b[0]
                c21 += v_a[2] * v_b[1]
                c22 += v_a[2] * v_b[2]

        # Build matrix from accumulated scalars
        # fmt: off
        cov_matrices[tid] = dtype_mat(
            c00, c01, c02,
            c10, c11, c12,
            c20, c21, c22
        )
        # fmt: on

    return compute_covariance_matrices


# Instantiate kernels for each precision
_compute_covariance_fp16 = _create_covariance_kernel(wp.float16, wp.vec3h, wp.mat33h)
_compute_covariance_fp32 = _create_covariance_kernel(wp.float32, wp.vec3, wp.mat33)
_compute_covariance_fp64 = _create_covariance_kernel(wp.float64, wp.vec3d, wp.mat33d)


@cache_warp_kernel
def _create_rodrigues_kernel(dtype_scalar, dtype_vec, dtype_mat):
    """Factory to create Rodrigues rotation kernel for specific dtypes."""

    @wp.kernel
    def compute_rodrigues_rotation(
        tgt_vecs: wp.array(dtype=dtype_vec),
        src_vecs: wp.array(dtype=dtype_vec),
        out_rotations: wp.array(dtype=dtype_mat),
    ):
        tid = wp.tid()

        # 1. Load Vectors
        # We want to rotate 'src' (a) to match 'tgt' (b)
        a = src_vecs[tid]
        b = tgt_vecs[tid]

        eps = dtype_scalar(1e-8)

        # 2. Precise Normalization (Match PyTorch: clamp denominator)
        # Avoids shrinking the vector like (norm + eps) does
        a_len = wp.length(a)
        b_len = wp.length(b)
        a_unit = a / wp.max(a_len, eps)
        b_unit = b / wp.max(b_len, eps)

        # 3. Compute Dot and Cross (Un-normalized axis)
        # dot = cos(theta)
        dot = wp.dot(a_unit, b_unit)
        dot = wp.clamp(dot, dtype_scalar(-1.0), dtype_scalar(1.0))

        # axis_vec has length sin(theta)
        axis_vec = wp.cross(a_unit, b_unit)

        # 4. Matrix Construction
        # We use the algebraic formula: R = I + K + (1/(1+c)) * K^2
        # where K is the skew-symmetric matrix of the *un-normalized* axis.
        # This avoids dividing by sin(theta) entirely.

        # Handle Antiparallel case (dot near -1)
        if dot < dtype_scalar(-1.0) + dtype_scalar(1e-6):
            # 180-degree rotation logic
            # Find orthogonal vector to a_unit
            # If abs(x) > 0.6, cross with Y, else cross with X

            # Note: Warp doesn't support vector literals easily in conditional assignment
            # so we branch the logic
            perp_axis = dtype_vec(dtype_scalar(0.0), dtype_scalar(0.0), dtype_scalar(0.0))

            if wp.abs(a_unit[0]) > dtype_scalar(0.6):
                # Cross with Y (0, 1, 0)
                perp_axis = wp.cross(
                    a_unit,
                    dtype_vec(dtype_scalar(0.0), dtype_scalar(1.0), dtype_scalar(0.0)),
                )
            else:
                # Cross with X (1, 0, 0)
                perp_axis = wp.cross(
                    a_unit,
                    dtype_vec(dtype_scalar(1.0), dtype_scalar(0.0), dtype_scalar(0.0)),
                )

            # Normalize the rotation axis
            u = perp_axis / wp.length(perp_axis)

            # R = 2*u*u^T - I
            # Manually construct outer product 2*u*u^T
            xx = dtype_scalar(2.0) * u[0] * u[0]
            yy = dtype_scalar(2.0) * u[1] * u[1]
            zz = dtype_scalar(2.0) * u[2] * u[2]
            xy = dtype_scalar(2.0) * u[0] * u[1]
            xz = dtype_scalar(2.0) * u[0] * u[2]
            yz = dtype_scalar(2.0) * u[1] * u[2]

            # fmt: off
            out_rotations[tid] = dtype_mat(
                xx - dtype_scalar(1.0), xy, xz,
                xy, yy - dtype_scalar(1.0), yz,
                xz, yz, zz - dtype_scalar(1.0)
            )
            # fmt: on

        else:
            # General Case (includes parallel)
            # Factor = 1 / (1 + cos_theta)
            factor = dtype_scalar(1.0) / (dtype_scalar(1.0) + dot)

            # Construct Skew-Symmetric Matrix K from un-normalized axis
            kx = axis_vec[0]
            ky = axis_vec[1]
            kz = axis_vec[2]

            # R = I + K + factor * (K @ K)

            # K matrix
            # 0  -z   y
            # z   0  -x
            # -y  x   0

            # We can compute (I + K + factor*K^2) element-wise directly
            # to save matrix multiplications, but explicit matrix math is clearer
            # and Warp optimizes it well.

            # fmt: off
            K = dtype_mat(
                dtype_scalar(0.0), -kz, ky,
                kz, dtype_scalar(0.0), -kx,
                -ky, kx, dtype_scalar(0.0)
            )
            # fmt: on

            K2 = K * K
            # fmt: off
            eye = dtype_mat(
                dtype_scalar(1.0), dtype_scalar(0.0), dtype_scalar(0.0),
                dtype_scalar(0.0), dtype_scalar(1.0), dtype_scalar(0.0),
                dtype_scalar(0.0), dtype_scalar(0.0), dtype_scalar(1.0)
            )
            # fmt: on

            out_rotations[tid] = eye + K + factor * K2

    return compute_rodrigues_rotation


# Instantiate kernels for each precision
_compute_rodrigues_fp16 = _create_rodrigues_kernel(wp.float16, wp.vec3h, wp.mat33h)
_compute_rodrigues_fp32 = _create_rodrigues_kernel(wp.float32, wp.vec3, wp.mat33)
_compute_rodrigues_fp64 = _create_rodrigues_kernel(wp.float64, wp.vec3d, wp.mat33d)


@cache_warp_kernel
def _create_svd_kernel(dtype_scalar, dtype_vec, dtype_mat):
    """Factory to create SVD/Kabsch kernel for specific dtypes."""

    # Workaround: wp.svd3() is broken for fp16, so compute in fp32 and convert
    if dtype_scalar == wp.float16:

        @wp.kernel
        def compute_rotations_from_covariance(
            cov_matrices: wp.array(dtype=dtype_mat),
            out_rotations: wp.array(dtype=dtype_mat),
        ):
            """
            Compute rotation matrices from covariance using SVD (fp16 version).
            Uses fp32 internally to workaround broken fp16 wp.svd3().
            """
            tid = wp.tid()

            # Convert fp16 input to fp32 for SVD computation
            cov_in = cov_matrices[tid]
            # fmt: off
            cov = wp.mat33(
                wp.float32(cov_in[0, 0]), wp.float32(cov_in[0, 1]), wp.float32(cov_in[0, 2]),
                wp.float32(cov_in[1, 0]), wp.float32(cov_in[1, 1]), wp.float32(cov_in[1, 2]),
                wp.float32(cov_in[2, 0]), wp.float32(cov_in[2, 1]), wp.float32(cov_in[2, 2])
            )
            # fmt: on

            U = wp.mat33()
            S = wp.vec3()
            V = wp.mat33()
            wp.svd3(cov, U, S, V)

            # Kabsch: R = U @ V^T
            R = U * wp.transpose(V)

            # Handle reflection case: ensure det(R) = 1
            if wp.determinant(R) < wp.float32(0.0):
                # fmt: off
                correction = wp.mat33(
                    wp.float32(1.0), wp.float32(0.0), wp.float32(0.0),
                    wp.float32(0.0), wp.float32(1.0), wp.float32(0.0),
                    wp.float32(0.0), wp.float32(0.0), wp.float32(-1.0)
                )
                # fmt: on
                R = U * correction * wp.transpose(V)

            # Convert fp32 result back to fp16
            # fmt: off
            out_rotations[tid] = dtype_mat(
                dtype_scalar(R[0, 0]), dtype_scalar(R[0, 1]), dtype_scalar(R[0, 2]),
                dtype_scalar(R[1, 0]), dtype_scalar(R[1, 1]), dtype_scalar(R[1, 2]),
                dtype_scalar(R[2, 0]), dtype_scalar(R[2, 1]), dtype_scalar(R[2, 2])
            )
            # fmt: on
    else:

        @wp.kernel
        def compute_rotations_from_covariance(
            cov_matrices: wp.array(dtype=dtype_mat),
            out_rotations: wp.array(dtype=dtype_mat),
        ):
            """
            Compute rotation matrices from precomputed covariance matrices using SVD.
            Uses Kabsch algorithm for N >= 2 point sets.
            Separating this from covariance computation allows autodiff to work properly.
            """
            tid = wp.tid()

            # Use Kabsch algorithm with SVD
            cov = cov_matrices[tid]

            U = dtype_mat()
            S = dtype_vec()
            V = dtype_mat()
            wp.svd3(cov, U, S, V)

            # Kabsch: R = U @ V^T
            R = U * wp.transpose(V)

            # Handle reflection case: ensure det(R) = 1
            if wp.determinant(R) < dtype_scalar(0.0):
                # Flip sign of last column of U
                # fmt: off
                correction = dtype_mat(
                    dtype_scalar(1.0), dtype_scalar(0.0), dtype_scalar(0.0),
                    dtype_scalar(0.0), dtype_scalar(1.0), dtype_scalar(0.0),
                    dtype_scalar(0.0), dtype_scalar(0.0), dtype_scalar(-1.0)
                )
                # fmt: on
                R = U * correction * wp.transpose(V)

            out_rotations[tid] = R

    return compute_rotations_from_covariance


# Instantiate kernels for each precision
_compute_svd_fp16 = _create_svd_kernel(wp.float16, wp.vec3h, wp.mat33h)
_compute_svd_fp32 = _create_svd_kernel(wp.float32, wp.vec3, wp.mat33)
_compute_svd_fp64 = _create_svd_kernel(wp.float64, wp.vec3d, wp.mat33d)


@cache_warp_kernel
def _create_newton_schulz_kernel(dtype_scalar, dtype_vec, dtype_mat):
    """Factory to create Newton-Schulz kernel for specific dtypes."""

    @wp.kernel
    def compute_rotations_newton_schulz(
        cov_matrices: wp.array(dtype=dtype_mat),
        out_rotations: wp.array(dtype=dtype_mat),
    ):
        """
        Compute rotation matrices from covariance using Newton-Schulz iteration.
        Performs polar decomposition via iterative orthogonalization.
        Uses scaled Newton-Schulz iteration for improved convergence.
        """
        tid = wp.tid()

        H = cov_matrices[tid]

        # Scale by 1/infinity-norm (max absolute row sum) or 1-norm (max col sum)
        # This guarantees convergence without calculating square roots.
        row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
        row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
        row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
        max_sum = wp.max(row0_sum, wp.max(row1_sum, row2_sum))

        scale = dtype_scalar(1.0) / max_sum  # Safe, no sqrt needed

        R = H * scale

        # Order-2 Newton-Schulz iteration
        # R_{k+1} = R_k * (3*I - R_k^T * R_k) / 2
        # 30 iterations handles condition ratios up to ~200K, preventing cascade
        # degradation through deep kinematic chains (fingers).
        for _ in range(30):
            RT_R = wp.transpose(R) * R

            # Compute 3*I - R^T*R
            # fmt: off
            term = dtype_mat(
                dtype_scalar(3.0) - RT_R[0, 0], -RT_R[0, 1], -RT_R[0, 2],
                -RT_R[1, 0], dtype_scalar(3.0) - RT_R[1, 1], -RT_R[1, 2],
                -RT_R[2, 0], -RT_R[2, 1], dtype_scalar(3.0) - RT_R[2, 2]
            )
            # fmt: on

            # R_{k+1} = R_k * (3*I - R^T*R) / 2
            R = R * term * dtype_scalar(0.5)

        # Differentiable determinant correction
        det = wp.determinant(R)

        # Differentiable replacement for: if det < 0: factor = -1 else: factor = 1
        # wp.where(condition, value_if_true, value_if_false)
        # If det < 0 (True) -> returns -1.0
        # If det >= 0 (False) -> returns 1.0
        sign_factor = wp.where(det < dtype_scalar(0.0), dtype_scalar(-1.0), dtype_scalar(1.0))

        # Apply the correction to the last column (Z-axis)
        # We reconstruct the matrix to ensure SSA (Single Static Assignment) compliance
        c02 = R[0, 2] * sign_factor
        c12 = R[1, 2] * sign_factor
        c22 = R[2, 2] * sign_factor

        # Reassemble R with proper dtype
        # fmt: off
        R = dtype_mat(
            R[0,0], R[0,1], c02,
            R[1,0], R[1,1], c12,
            R[2,0], R[2,1], c22
        )
        # fmt: on

        out_rotations[tid] = R

    return compute_rotations_newton_schulz


# Instantiate kernels for each precision
_compute_newton_schulz_fp16 = _create_newton_schulz_kernel(wp.float16, wp.vec3h, wp.mat33h)
_compute_newton_schulz_fp32 = _create_newton_schulz_kernel(wp.float32, wp.vec3, wp.mat33)
_compute_newton_schulz_fp64 = _create_newton_schulz_kernel(wp.float64, wp.vec3d, wp.mat33d)


@cache_warp_kernel
def _create_newton_schulz_auto_kernel(dtype_scalar, dtype_vec, dtype_mat):
    """Factory to create degenerate-safe Newton-Schulz auto kernels."""

    # wp.svd3() does not support fp16 reliably, so reflected fp16 covariances
    # use the same fp32 promotion as the dedicated Kabsch kernel.
    svd_scalar = wp.float32 if dtype_scalar == wp.float16 else dtype_scalar
    svd_vec = wp.vec3 if dtype_scalar == wp.float16 else dtype_vec
    svd_mat = wp.mat33 if dtype_scalar == wp.float16 else dtype_mat

    @wp.kernel
    def compute_rotations_auto(
        cov_matrices: wp.array(dtype=dtype_mat),
        out_rotations: wp.array(dtype=dtype_mat),
    ):
        tid = wp.tid()
        H = cov_matrices[tid]

        row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
        row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
        row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
        prior_scale = wp.max(wp.max(row0_sum, wp.max(row1_sum, row2_sum)), dtype_scalar(1.0e-8))
        volume_score = wp.abs(wp.determinant(H)) / (prior_scale * prior_scale * prior_scale)
        rank_weight = wp.clamp(
            (dtype_scalar(1.0e-6) - volume_score) / dtype_scalar(1.0e-6),
            dtype_scalar(0.0),
            dtype_scalar(1.0),
        )
        prior = dtype_scalar(0.05) * rank_weight * prior_scale

        # Identity-gauge only for truly degenerate covariance. This keeps raw NS
        # behavior for normal SkeletonTransfer cases while avoiding singular
        # gradients for rank-0/rank-1 inputs.
        # fmt: off
        H = dtype_mat(
            H[0, 0] + prior, H[0, 1], H[0, 2],
            H[1, 0], H[1, 1] + prior, H[1, 2],
            H[2, 0], H[2, 1], H[2, 2] + prior,
        )
        # fmt: on

        # A reflected covariance needs Kabsch's singular-vector correction.
        # Flipping a fixed Newton-Schulz output column is not the nearest SO(3)
        # projection and can select a substantially different rotation.
        # Compute the sign in the promoted SVD precision as fp16 determinant
        # evaluation can select the wrong branch for reflected inputs.
        # fmt: off
        H_svd = svd_mat(
            svd_scalar(H[0, 0]), svd_scalar(H[0, 1]), svd_scalar(H[0, 2]),
            svd_scalar(H[1, 0]), svd_scalar(H[1, 1]), svd_scalar(H[1, 2]),
            svd_scalar(H[2, 0]), svd_scalar(H[2, 1]), svd_scalar(H[2, 2]),
        )
        # fmt: on
        if wp.determinant(H_svd) < svd_scalar(0.0):
            U = svd_mat()
            S = svd_vec()
            V = svd_mat()
            wp.svd3(H_svd, U, S, V)
            # fmt: off
            correction = svd_mat(
                svd_scalar(1.0), svd_scalar(0.0), svd_scalar(0.0),
                svd_scalar(0.0), svd_scalar(1.0), svd_scalar(0.0),
                svd_scalar(0.0), svd_scalar(0.0), svd_scalar(-1.0),
            )
            # fmt: on
            R_svd = U * correction * wp.transpose(V)
            # fmt: off
            out_rotations[tid] = dtype_mat(
                dtype_scalar(R_svd[0, 0]), dtype_scalar(R_svd[0, 1]), dtype_scalar(R_svd[0, 2]),
                dtype_scalar(R_svd[1, 0]), dtype_scalar(R_svd[1, 1]), dtype_scalar(R_svd[1, 2]),
                dtype_scalar(R_svd[2, 0]), dtype_scalar(R_svd[2, 1]), dtype_scalar(R_svd[2, 2]),
            )
            # fmt: on
            return

        row0_sum = wp.abs(H[0, 0]) + wp.abs(H[0, 1]) + wp.abs(H[0, 2])
        row1_sum = wp.abs(H[1, 0]) + wp.abs(H[1, 1]) + wp.abs(H[1, 2])
        row2_sum = wp.abs(H[2, 0]) + wp.abs(H[2, 1]) + wp.abs(H[2, 2])
        max_sum = wp.max(wp.max(row0_sum, wp.max(row1_sum, row2_sum)), dtype_scalar(1.0e-8))
        R = H * (dtype_scalar(1.0) / max_sum)

        for _ in range(30):
            RT_R = wp.transpose(R) * R
            # fmt: off
            term = dtype_mat(
                dtype_scalar(3.0) - RT_R[0, 0], -RT_R[0, 1], -RT_R[0, 2],
                -RT_R[1, 0], dtype_scalar(3.0) - RT_R[1, 1], -RT_R[1, 2],
                -RT_R[2, 0], -RT_R[2, 1], dtype_scalar(3.0) - RT_R[2, 2],
            )
            # fmt: on
            R = R * term * dtype_scalar(0.5)

        det = wp.determinant(R)
        sign_factor = wp.where(det < dtype_scalar(0.0), dtype_scalar(-1.0), dtype_scalar(1.0))
        # fmt: off
        R = dtype_mat(
            R[0, 0], R[0, 1], R[0, 2] * sign_factor,
            R[1, 0], R[1, 1], R[1, 2] * sign_factor,
            R[2, 0], R[2, 1], R[2, 2] * sign_factor,
        )
        # fmt: on
        out_rotations[tid] = R

    return compute_rotations_auto


_compute_auto_fp16 = _create_newton_schulz_auto_kernel(wp.float16, wp.vec3h, wp.mat33h)
_compute_auto_fp32 = _create_newton_schulz_auto_kernel(wp.float32, wp.vec3, wp.mat33)
_compute_auto_fp64 = _create_newton_schulz_auto_kernel(wp.float64, wp.vec3d, wp.mat33d)


def _select_rotation_kernel(dtype: torch.dtype, method: AlignmentMethod):
    if dtype == torch.float16:
        kernels = {
            "auto": _compute_auto_fp16,
            "kabsch": _compute_svd_fp16,
            "newton-schulz": _compute_newton_schulz_fp16,
        }
    elif dtype == torch.float64:
        kernels = {
            "auto": _compute_auto_fp64,
            "kabsch": _compute_svd_fp64,
            "newton-schulz": _compute_newton_schulz_fp64,
        }
    else:
        kernels = {
            "auto": _compute_auto_fp32,
            "kabsch": _compute_svd_fp32,
            "newton-schulz": _compute_newton_schulz_fp32,
        }
    if method not in kernels:
        raise ValueError(f"Unknown rotation method: {method!r}.")
    return kernels[method]


class RodriguesRotationWarp(torch.autograd.Function):
    """
    Differentiable Rodrigues rotation using Warp kernel.

    Computes rotation matrices from pairs of vectors using Rodrigues formula.
    SciPy-compatible: Returns R such that R @ B ≈ A.
    Fully differentiable with respect to input vectors.
    """

    @staticmethod
    def forward(ctx, A, B):
        """
        Forward pass using Rodrigues formula.

        Args:
            A: (N, 3) - target vectors
            B: (N, 3) - source vectors

        Returns:
            R: (N, 3, 3) - rotation matrices such that R @ B ≈ A
        """
        ensure_warp_initialized()
        N = len(A)
        device = A.device
        device_str = wp.device_from_torch(device)

        # Get Warp types based on input precision
        dtype_scalar, dtype_vec, dtype_mat = _get_warp_types(A.dtype)

        # Select appropriate kernel
        if A.dtype == torch.float16:
            kernel = _compute_rodrigues_fp16
        elif A.dtype == torch.float64:
            kernel = _compute_rodrigues_fp64
        else:
            kernel = _compute_rodrigues_fp32

        # Convert to Warp arrays with gradient tracking (A is target, B is source)
        # Use detach().contiguous() to avoid non-leaf tensor warning
        wp_tgt = wp.from_torch(A.detach().contiguous(), dtype=dtype_vec, requires_grad=True)
        wp_src = wp.from_torch(B.detach().contiguous(), dtype=dtype_vec, requires_grad=True)
        wp_out = wp.zeros(N, dtype=dtype_mat, device=device_str, requires_grad=True)

        # Create tape for recording
        tape = wp.Tape()

        with tape:
            wp.launch(
                kernel=kernel,
                dim=N,
                inputs=[wp_tgt, wp_src, wp_out],
                device=device_str,
            )

        R_flat = wp.to_torch(wp_out)
        R = R_flat.reshape(N, 3, 3)

        # Save for backward
        ctx.tape = tape
        ctx.wp_src = wp_src
        ctx.wp_tgt = wp_tgt
        ctx.wp_out = wp_out

        return R

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Warp's tape.
        """
        # Zero tape before setting gradients
        ctx.tape.zero()

        # Set output gradient
        _, _, dtype_mat = _get_warp_types(grad_output.dtype)
        wp_grad_out = wp.from_torch(grad_output.contiguous(), dtype=dtype_mat)
        ctx.wp_out.grad = wp_grad_out

        # Run backward pass
        ctx.tape.backward()

        # Extract gradients (A is target, B is source - return in that order)
        grad_A = None
        grad_B = None

        if ctx.needs_input_grad[0] and ctx.wp_tgt.grad is not None:
            grad_A = wp.to_torch(ctx.wp_tgt.grad).clone()

        if ctx.needs_input_grad[1] and ctx.wp_src.grad is not None:
            grad_B = wp.to_torch(ctx.wp_src.grad).clone()

        return grad_A, grad_B


class AlignVectorsWarp(torch.autograd.Function):
    """
    Differentiable Warp-based align_vectors using tape for gradients.

    For N>=2 point sets using Kabsch algorithm (covariance + SVD).
    For N=1, use rodrigues_rotation_warp() instead.

    Forward and backward passes both use Warp for full GPU acceleration.
    """

    @staticmethod
    def forward(ctx, A_flat, B_flat, offsets, counts, method="auto"):
        """
        Forward pass using Warp kernel with tape recording.

        Args:
            A_flat: (total_vectors, 3) - flattened target vectors
            B_flat: (total_vectors, 3) - flattened source vectors
            offsets: (batch_size,) int32 - start index for each batch
            counts: (batch_size,) int32 - number of vectors per batch (must be >= 2)
            method: str - rotation extraction method ('auto', 'kabsch', or 'newton-schulz')

        Returns:
            R: (batch_size, 3, 3) - rotation matrices
        """
        ensure_warp_initialized()
        batch_size = len(offsets)
        device = A_flat.device
        device_str = wp.device_from_torch(device)

        # Get Warp types based on input precision
        dtype_scalar, dtype_vec, dtype_mat = _get_warp_types(A_flat.dtype)

        # Select appropriate kernels
        if A_flat.dtype == torch.float16:
            cov_kernel = _compute_covariance_fp16
        elif A_flat.dtype == torch.float64:
            cov_kernel = _compute_covariance_fp64
        else:
            cov_kernel = _compute_covariance_fp32
        rotation_kernel = _select_rotation_kernel(A_flat.dtype, method)

        # Convert to Warp arrays with gradient tracking
        # Use detach().contiguous() to avoid non-leaf tensor warnings
        wp_a = wp.from_torch(A_flat.detach().contiguous(), dtype=dtype_vec, requires_grad=True)
        wp_b = wp.from_torch(B_flat.detach().contiguous(), dtype=dtype_vec, requires_grad=True)
        wp_off = wp.from_torch(offsets.contiguous(), dtype=wp.int32)
        wp_cnt = wp.from_torch(counts.contiguous(), dtype=wp.int32)
        wp_cov = wp.zeros(batch_size, dtype=dtype_mat, device=device_str, requires_grad=True)
        wp_out = wp.zeros(batch_size, dtype=dtype_mat, device=device_str, requires_grad=True)

        # Create tape for recording
        tape = wp.Tape()

        with tape:
            # Two-kernel approach for autodiff compatibility
            # Kernel 1: Compute covariance matrices
            wp.launch(
                kernel=cov_kernel,
                dim=batch_size,
                inputs=[wp_a, wp_b, wp_off, wp_cnt, wp_cov],
                device=device_str,
            )
            # Kernel 2: Compute rotations from covariance (SVD or Newton-Schulz)
            wp.launch(
                kernel=rotation_kernel,
                dim=batch_size,
                inputs=[wp_cov, wp_out],
                device=device_str,
            )

        R_flat = wp.to_torch(wp_out)
        R = R_flat.reshape(batch_size, 3, 3)

        # Save for backward
        ctx.tape = tape
        ctx.wp_a = wp_a
        ctx.wp_b = wp_b
        ctx.wp_out = wp_out

        return R

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Warp's tape.
        """
        # Zero tape before setting gradients
        ctx.tape.zero()

        # Set output gradient
        _, _, dtype_mat = _get_warp_types(grad_output.dtype)
        wp_grad_out = wp.from_torch(grad_output.contiguous(), dtype=dtype_mat)
        ctx.wp_out.grad = wp_grad_out

        # Run backward pass
        ctx.tape.backward()

        # Extract gradients
        grad_A = None
        grad_B = None

        if ctx.needs_input_grad[0] and ctx.wp_a.grad is not None:
            grad_A = wp.to_torch(ctx.wp_a.grad).clone()

        if ctx.needs_input_grad[1] and ctx.wp_b.grad is not None:
            grad_B = wp.to_torch(ctx.wp_b.grad).clone()

        return grad_A, grad_B, None, None, None  # Added None for method parameter


def rodrigues_rotation_warp(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrices from vector pairs using Rodrigues formula.

    Differentiable Warp-based implementation.
    SciPy-compatible: Returns rotation R such that R @ B ≈ A.

    Args:
        A: (N, 3) torch tensor - target vectors
        B: (N, 3) torch tensor - source vectors

    Returns:
        R: (N, 3, 3) rotation matrices such that R @ B ≈ A

    Example:
        >>> A = torch.randn(100, 3, requires_grad=True)  # target
        >>> B = torch.randn(100, 3, requires_grad=True)  # source
        >>> R = rodrigues_rotation_warp(A, B)  # shape: (100, 3, 3)
        >>> # Verify: R @ B ≈ A
        >>> loss = R.pow(2).sum()
        >>> loss.backward()
    """
    return RodriguesRotationWarp.apply(A, B)


def _fallback_invalid_ragged_rotations(
    rotations: torch.Tensor,
    A_flat: torch.Tensor,
    B_flat: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    method: AlignmentMethod,
) -> torch.Tensor:
    """Replace invalid NS-family ragged-batch rotations with PyTorch projections."""
    if method not in {"auto", "newton-schulz"} or rotations.numel() == 0:
        return rotations

    from soma.geometry.transforms import align_vectors, rotation_matrices_are_valid

    valid = rotation_matrices_are_valid(rotations)
    if torch.all(valid):
        return rotations

    fallback = rotations.clone()
    bad_indices = torch.nonzero(~valid, as_tuple=False).flatten()
    for batch_idx in bad_indices.detach().cpu().tolist():
        start = int(offsets[batch_idx].detach().cpu().item())
        end = start + int(counts[batch_idx].detach().cpu().item())
        fallback_method = "auto" if method == "auto" else "kabsch"
        fallback[batch_idx] = align_vectors(
            A_flat[start:end],
            B_flat[start:end],
            method=fallback_method,
        )
    return torch.where(valid[..., None, None], rotations, fallback)


def align_vectors_warp(
    A_flat: torch.Tensor,
    B_flat: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    method: AlignmentMethod = "auto",
) -> torch.Tensor:
    """
    Differentiable Warp-based batch alignment of vector sets with ragged support.

    Uses auto, Kabsch, or Newton-Schulz rotation extraction for N>=2 point sets.
    For N=1 cases, use rodrigues_rotation_warp() instead.

    Uses Warp kernel with tape for both forward and backward passes.
    Fully GPU-accelerated including gradient computation.

    SciPy-compatible: returns rotation R such that R @ B ≈ A.
    Matches the convention in geometry.transforms.align_vectors().

    Args:
        A_flat: (total_vectors, 3) torch tensor - flattened target vectors
        B_flat: (total_vectors, 3) torch tensor - flattened source vectors
        offsets: (batch_size,) torch tensor (int32) - start index for each batch
        counts: (batch_size,) torch tensor (int32) - number of vectors per batch (must be >= 2)
        method: str - rotation extraction method ('auto', 'kabsch', or 'newton-schulz', default: 'auto')

    Returns:
        R: (batch_size, 3, 3) rotation matrices such that R @ B ≈ A

    Note:
        - Differentiable: gradients flow through both forward and backward passes
        - Uses Warp's tape mechanism for efficient gradient computation
        - Supports arbitrary batch sizes and variable vector counts (ragged batches)
        - For single vector pairs (N=1), use rodrigues_rotation_warp() instead

    Example:
        >>> # Single batch with 10 vectors
        >>> A = torch.randn(10, 3, requires_grad=True)
        >>> B = torch.randn(10, 3, requires_grad=True)
        >>> offsets = torch.tensor([0], dtype=torch.int32)
        >>> counts = torch.tensor([10], dtype=torch.int32)
        >>> R = align_vectors_warp(A, B, offsets, counts)  # shape: (1, 3, 3)
        >>> loss = R.pow(2).sum()
        >>> loss.backward()  # Gradients computed via Warp tape

        >>> # Ragged batch: 5 vectors, 8 vectors, 3 vectors
        >>> A_flat = torch.randn(16, 3, requires_grad=True)  # 5+8+3
        >>> B_flat = torch.randn(16, 3, requires_grad=True)
        >>> offsets = torch.tensor([0, 5, 13], dtype=torch.int32)
        >>> counts = torch.tensor([5, 8, 3], dtype=torch.int32)
        >>> R = align_vectors_warp(A_flat, B_flat, offsets, counts)  # shape: (3, 3, 3)
    """
    rotations = AlignVectorsWarp.apply(A_flat, B_flat, offsets, counts, method)
    return _fallback_invalid_ragged_rotations(
        rotations,
        A_flat,
        B_flat,
        offsets,
        counts,
        method,
    )


class ParallelRodriguesKabschWarp(torch.autograd.Function):
    """
    Parallel execution of Rodrigues (N=1) and Kabsch (N>=2) rotations using two CUDA streams.

    This function launches two independent Warp kernels in parallel:
    - Stream 1: Rodrigues rotation for single-child joints (N=1)
    - Stream 2: Covariance + Kabsch for multi-child joints (N>=2)

    Both streams execute concurrently for improved performance when both types of joints exist.
    """

    @staticmethod
    def forward(
        ctx,
        # Rodrigues inputs
        rodrigues_tgt,
        rodrigues_src,
        # Kabsch inputs
        kabsch_A_flat,
        kabsch_B_flat,
        kabsch_offsets,
        kabsch_counts,
        # Method selection
        method="auto",
    ):
        """
        Forward pass with parallel kernel launches.

        Args:
            rodrigues_tgt: (N1, 3) target vectors for Rodrigues
            rodrigues_src: (N1, 3) source vectors for Rodrigues
            kabsch_A_flat: (N2_total, 3) flattened target vectors for Kabsch
            kabsch_B_flat: (N2_total, 3) flattened source vectors for Kabsch
            kabsch_offsets: (N2_batches,) offsets for Kabsch ragged batch
            kabsch_counts: (N2_batches,) counts for Kabsch ragged batch
            method: str - rotation extraction method ('auto', 'kabsch', or 'newton-schulz', default: 'auto')

        Returns:
            rodrigues_R: (N1, 3, 3) rotation matrices from Rodrigues
            kabsch_R: (N2_batches, 3, 3) rotation matrices from Kabsch
        """
        ensure_warp_initialized()
        device = rodrigues_tgt.device
        device_str = wp.device_from_torch(device)

        N1 = len(rodrigues_tgt) if rodrigues_tgt is not None and rodrigues_tgt.numel() > 0 else 0
        N2_batches = (
            len(kabsch_offsets) if kabsch_offsets is not None and kabsch_offsets.numel() > 0 else 0
        )

        # Determine dtype from inputs
        dtype = (
            rodrigues_tgt.dtype
            if N1 > 0
            else (kabsch_A_flat.dtype if N2_batches > 0 else torch.float32)
        )
        dtype_scalar, dtype_vec, dtype_mat = _get_warp_types(dtype)

        # Select appropriate kernels based on dtype and method
        if dtype == torch.float16:
            rodrigues_kernel = _compute_rodrigues_fp16
            cov_kernel = _compute_covariance_fp16
        elif dtype == torch.float64:
            rodrigues_kernel = _compute_rodrigues_fp64
            cov_kernel = _compute_covariance_fp64
        else:
            rodrigues_kernel = _compute_rodrigues_fp32
            cov_kernel = _compute_covariance_fp32
        rotation_kernel = _select_rotation_kernel(dtype, method)

        # Create tape for recording
        tape = wp.Tape()

        # Prepare Rodrigues inputs/outputs
        wp_rodrigues_tgt = None
        wp_rodrigues_src = None
        wp_rodrigues_out = None

        if N1 > 0:
            wp_rodrigues_tgt = wp.from_torch(
                rodrigues_tgt.detach().contiguous(),
                dtype=dtype_vec,
                requires_grad=True,
            )
            wp_rodrigues_src = wp.from_torch(
                rodrigues_src.detach().contiguous(),
                dtype=dtype_vec,
                requires_grad=True,
            )
            wp_rodrigues_out = wp.zeros(N1, dtype=dtype_mat, device=device_str, requires_grad=True)

        # Prepare Kabsch inputs/outputs
        wp_kabsch_a = None
        wp_kabsch_b = None
        wp_kabsch_off = None
        wp_kabsch_cnt = None
        wp_kabsch_cov = None
        wp_kabsch_out = None

        if N2_batches > 0:
            wp_kabsch_a = wp.from_torch(
                kabsch_A_flat.detach().contiguous(),
                dtype=dtype_vec,
                requires_grad=True,
            )
            wp_kabsch_b = wp.from_torch(
                kabsch_B_flat.detach().contiguous(),
                dtype=dtype_vec,
                requires_grad=True,
            )
            wp_kabsch_off = wp.from_torch(kabsch_offsets.contiguous(), dtype=wp.int32)
            wp_kabsch_cnt = wp.from_torch(kabsch_counts.contiguous(), dtype=wp.int32)
            wp_kabsch_cov = wp.zeros(
                N2_batches,
                dtype=dtype_mat,
                device=device_str,
                requires_grad=True,
            )
            wp_kabsch_out = wp.zeros(
                N2_batches,
                dtype=dtype_mat,
                device=device_str,
                requires_grad=True,
            )

        # Launch kernels in parallel using tape
        # Note: Warp's runtime automatically optimizes kernel launch scheduling.
        # When multiple kernels are launched on CUDA, they may execute concurrently
        # if there are no data dependencies. This provides parallelism without
        # explicit stream management.
        with tape:
            # Launch Rodrigues kernel
            if N1 > 0:
                wp.launch(
                    kernel=rodrigues_kernel,
                    dim=N1,
                    inputs=[wp_rodrigues_tgt, wp_rodrigues_src, wp_rodrigues_out],
                    device=device_str,
                )

            # Launch Kabsch kernels (covariance + SVD)
            if N2_batches > 0:
                wp.launch(
                    kernel=cov_kernel,
                    dim=N2_batches,
                    inputs=[wp_kabsch_a, wp_kabsch_b, wp_kabsch_off, wp_kabsch_cnt, wp_kabsch_cov],
                    device=device_str,
                )
                wp.launch(
                    kernel=rotation_kernel,
                    dim=N2_batches,
                    inputs=[wp_kabsch_cov, wp_kabsch_out],
                    device=device_str,
                )

        # Convert outputs to PyTorch (wp.to_torch() automatically synchronizes)
        # Return empty tensors instead of None for easier handling in calling code
        if N1 > 0:
            rodrigues_R = wp.to_torch(wp_rodrigues_out).reshape(N1, 3, 3)
        else:
            rodrigues_R = torch.empty((0, 3, 3), dtype=dtype, device=device)

        if N2_batches > 0:
            kabsch_R = wp.to_torch(wp_kabsch_out).reshape(N2_batches, 3, 3)
        else:
            kabsch_R = torch.empty((0, 3, 3), dtype=dtype, device=device)

        # Save for backward
        ctx.tape = tape
        ctx.N1 = N1
        ctx.N2_batches = N2_batches
        ctx.wp_rodrigues_tgt = wp_rodrigues_tgt
        ctx.wp_rodrigues_src = wp_rodrigues_src
        ctx.wp_rodrigues_out = wp_rodrigues_out
        ctx.wp_kabsch_a = wp_kabsch_a
        ctx.wp_kabsch_b = wp_kabsch_b
        ctx.wp_kabsch_cov = wp_kabsch_cov
        ctx.wp_kabsch_out = wp_kabsch_out
        ctx.needs_rodrigues = N1 > 0
        ctx.needs_kabsch = N2_batches > 0

        return rodrigues_R, kabsch_R

    @staticmethod
    def backward(ctx, grad_rodrigues_R, grad_kabsch_R):
        """
        Backward pass using Warp's tape.
        """
        # Zero tape before setting gradients
        ctx.tape.zero()

        # Set output gradients
        if ctx.needs_rodrigues and grad_rodrigues_R is not None:
            _, _, dtype_mat = _get_warp_types(grad_rodrigues_R.dtype)
            wp_grad_rodrigues = wp.from_torch(grad_rodrigues_R.contiguous(), dtype=dtype_mat)
            ctx.wp_rodrigues_out.grad = wp_grad_rodrigues

        if ctx.needs_kabsch and grad_kabsch_R is not None:
            _, _, dtype_mat = _get_warp_types(grad_kabsch_R.dtype)
            wp_grad_kabsch = wp.from_torch(grad_kabsch_R.contiguous(), dtype=dtype_mat)
            ctx.wp_kabsch_out.grad = wp_grad_kabsch

        # Run backward pass
        ctx.tape.backward()

        # Extract gradients
        grad_rodrigues_tgt = None
        grad_rodrigues_src = None
        grad_kabsch_A = None
        grad_kabsch_B = None

        if ctx.needs_rodrigues:
            if ctx.needs_input_grad[0] and ctx.wp_rodrigues_tgt.grad is not None:
                grad_rodrigues_tgt = wp.to_torch(ctx.wp_rodrigues_tgt.grad).clone()

            if ctx.needs_input_grad[1] and ctx.wp_rodrigues_src.grad is not None:
                grad_rodrigues_src = wp.to_torch(ctx.wp_rodrigues_src.grad).clone()

        if ctx.needs_kabsch:
            if ctx.needs_input_grad[2] and ctx.wp_kabsch_a.grad is not None:
                grad_kabsch_A = wp.to_torch(ctx.wp_kabsch_a.grad).clone()

            if ctx.needs_input_grad[3] and ctx.wp_kabsch_b.grad is not None:
                grad_kabsch_B = wp.to_torch(ctx.wp_kabsch_b.grad).clone()

        return (
            grad_rodrigues_tgt,
            grad_rodrigues_src,
            grad_kabsch_A,
            grad_kabsch_B,
            None,
            None,
            None,
        )  # Added None for method parameter


def parallel_rodrigues_kabsch_warp(
    rodrigues_tgt: torch.Tensor,
    rodrigues_src: torch.Tensor,
    kabsch_A_flat: torch.Tensor,
    kabsch_B_flat: torch.Tensor,
    kabsch_offsets: torch.Tensor,
    kabsch_counts: torch.Tensor,
    method: AlignmentMethod = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute Rodrigues and Kabsch rotations in parallel using two CUDA streams.

    Args:
        rodrigues_tgt: (N1, 3) target vectors for Rodrigues (can be None if no N=1 joints)
        rodrigues_src: (N1, 3) source vectors for Rodrigues (can be None if no N=1 joints)
        kabsch_A_flat: (N2_total, 3) flattened target vectors for Kabsch (can be None if no N>=2 joints)
        kabsch_B_flat: (N2_total, 3) flattened source vectors for Kabsch (can be None if no N>=2 joints)
        kabsch_offsets: (N2_batches,) offsets for Kabsch ragged batch
        kabsch_counts: (N2_batches,) counts for Kabsch ragged batch
        method: str - rotation extraction method ('auto', 'kabsch', or 'newton-schulz', default: 'auto')

    Returns:
        rodrigues_R: (N1, 3, 3) rotation matrices from Rodrigues (empty tensor if no N=1 joints)
        kabsch_R: (N2_batches, 3, 3) rotation matrices from Kabsch (empty tensor if no N>=2 joints)

    Example:
        >>> # Joint with single child (Rodrigues)
        >>> tgt1 = torch.randn(10, 3, device='cuda', requires_grad=True)
        >>> src1 = torch.randn(10, 3, device='cuda', requires_grad=True)
        >>>
        >>> # Joints with multiple children (Kabsch)
        >>> A_flat = torch.randn(30, 3, device='cuda', requires_grad=True)  # 10+15+5
        >>> B_flat = torch.randn(30, 3, device='cuda', requires_grad=True)
        >>> offsets = torch.tensor([0, 10, 25], dtype=torch.int32, device='cuda')
        >>> counts = torch.tensor([10, 15, 5], dtype=torch.int32, device='cuda')
        >>>
        >>> # Execute in parallel
        >>> R1, R2 = parallel_rodrigues_kabsch_warp(tgt1, src1, A_flat, B_flat, offsets, counts)
        >>> # R1.shape: (10, 3, 3), R2.shape: (3, 3, 3)
    """
    rodrigues_R, kabsch_R = ParallelRodriguesKabschWarp.apply(
        rodrigues_tgt,
        rodrigues_src,
        kabsch_A_flat,
        kabsch_B_flat,
        kabsch_offsets,
        kabsch_counts,
        method,
    )
    kabsch_R = _fallback_invalid_ragged_rotations(
        kabsch_R,
        kabsch_A_flat,
        kabsch_B_flat,
        kabsch_offsets,
        kabsch_counts,
        method,
    )
    return rodrigues_R, kabsch_R
