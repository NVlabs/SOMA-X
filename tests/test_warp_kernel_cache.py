# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from soma.geometry._warp_utils import cache_warp_kernel
from soma.geometry.align_vectors_warp import _create_newton_schulz_auto_kernel
from soma.geometry.fused_refit_warp import _create_fused_lbs_cov_kernel
from soma.geometry.lbs_warp import get_kernel


def _lbs_kernel(max_bones_count, vertices_scalar_dtype, vec3_dtype, mat44_dtype):
    weights_dtype = wp.types.vector(max_bones_count, dtype=vertices_scalar_dtype)
    indices_dtype = wp.types.vector(max_bones_count, dtype=wp.int32)
    return get_kernel(
        max_bones_count=max_bones_count,
        vertices_scalar_dtype=vertices_scalar_dtype,
        weights_dtype=weights_dtype,
        indices_dtype=indices_dtype,
        vec3_dtype=vec3_dtype,
        mat44_dtype=mat44_dtype,
    )


def test_cache_warp_kernel_normalizes_simple_containers():
    calls = 0

    @cache_warp_kernel
    def factory(values):
        nonlocal calls
        calls += 1
        return object()

    first = factory([1, [2, 3]])
    second = factory((1, (2, 3)))

    assert first is second
    assert calls == 1


def test_lbs_get_kernel_reuses_same_specialization():
    first = _lbs_kernel(4, wp.float32, wp.vec3f, wp.mat44f)
    second = _lbs_kernel(4, wp.float32, wp.vec3f, wp.mat44f)

    assert first is second


def test_lbs_get_kernel_separates_different_specializations():
    fp32_k4 = _lbs_kernel(4, wp.float32, wp.vec3f, wp.mat44f)
    fp32_k8 = _lbs_kernel(8, wp.float32, wp.vec3f, wp.mat44f)
    fp64_k4 = _lbs_kernel(4, wp.float64, wp.vec3d, wp.mat44d)

    assert fp32_k4 is not fp32_k8
    assert fp32_k4 is not fp64_k4


def test_fused_refit_factory_uses_shared_cache():
    first = _create_fused_lbs_cov_kernel(4)
    second = _create_fused_lbs_cov_kernel(K=4)
    third = _create_fused_lbs_cov_kernel(8)

    assert first is second
    assert first is not third


def test_align_vectors_auto_factory_uses_shared_cache():
    first = _create_newton_schulz_auto_kernel(wp.float32, wp.vec3, wp.mat33)
    second = _create_newton_schulz_auto_kernel(
        dtype_scalar=wp.float32,
        dtype_vec=wp.vec3,
        dtype_mat=wp.mat33,
    )
    third = _create_newton_schulz_auto_kernel(wp.float64, wp.vec3d, wp.mat33d)

    assert first is second
    assert first is not third
