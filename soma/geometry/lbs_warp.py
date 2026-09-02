# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Warp-based Linear Blend Skinning with torch.export support.

Changes from original:
- Added @torch.library.custom_op decorators for torch.export compatibility
- Registered fake implementations for shape inference during export
- Wrapped forward/backward logic in custom operator functions
- Public API (linear_blend_skinning) unchanged, fully backward compatible
- Supports dynamic K (any number of bone influences)

Original implementation used torch.autograd.Function (not exportable).
Current implementation uses torch.library.custom_op (exportable via torch.export).

Example usage:
    from soma.geometry.lbs_warp import linear_blend_skinning

    # Prepare inputs
    vertices = torch.rand(4, 1000, 3, device='cuda')  # (batch, num_verts, 3)
    bone_weights = torch.rand(1000, 8, device='cuda')  # (num_verts, K) - sparse
    bone_indices = torch.randint(0, 20, (1000, 8), device='cuda')  # (num_verts, K)
    bone_transforms = torch.eye(4, device='cuda').unsqueeze(0).unsqueeze(0).expand(4, 20, 4, 4)

    # Apply skinning
    posed_verts = linear_blend_skinning(vertices, bone_weights, bone_indices, bone_transforms)

    # With gradients
    bone_transforms.requires_grad_(True)
    posed_verts = linear_blend_skinning(vertices, bone_weights, bone_indices, bone_transforms)
    posed_verts.sum().backward()  # Compute gradients

Export example:
    class LBSModule(torch.nn.Module):
        def __init__(self, bone_weights, bone_indices):
            super().__init__()
            self.register_buffer('bone_weights', bone_weights)
            self.register_buffer('bone_indices', bone_indices)

        def forward(self, vertices, bone_transforms):
            return linear_blend_skinning(vertices, self.bone_weights,
                                        self.bone_indices, bone_transforms)

    module = LBSModule(bone_weights, bone_indices)
    exported = torch.export.export(module, (vertices, bone_transforms))
    torch.export.save(exported, "lbs.pt2")
"""

from typing import Any

import torch
import warp as wp

from soma.geometry._warp_utils import cache_warp_kernel, ensure_warp_initialized


@cache_warp_kernel
def get_kernel(
    max_bones_count: int,
    vertices_scalar_dtype: Any,
    weights_dtype: Any,
    indices_dtype: Any,
    vec3_dtype: Any = wp.vec3,
    mat44_dtype: Any = wp.mat44,
) -> Any:
    """
    Generate the kernel for linear blend skinning.
    """

    @wp.kernel
    def kernel(
        vertices: wp.array2d(dtype=vec3_dtype),
        bone_weights: wp.array(dtype=weights_dtype),
        bone_indices: wp.array(dtype=indices_dtype),
        bone_transforms: wp.array2d(dtype=mat44_dtype),
        output: wp.array2d(dtype=vec3_dtype),
    ):
        batch_id, vertex_id = wp.tid()
        vertex = vertices[batch_id, vertex_id]
        ws = bone_weights[vertex_id]
        ids = bone_indices[vertex_id]
        result = vec3_dtype(
            vertices_scalar_dtype(0.0), vertices_scalar_dtype(0.0), vertices_scalar_dtype(0.0)
        )
        for i in range(wp.static(max_bones_count)):
            result += ws[i] * wp.transform_point(bone_transforms[batch_id, ids[i]], vertex)
        output[batch_id, vertex_id] = result

    return kernel


# Custom operator for torch.export support
@torch.library.custom_op("warp_lbs::lbs_forward", mutates_args=())
def _lbs_custom_op_forward(
    vertices: torch.Tensor,
    bone_weights: torch.Tensor,
    bone_indices: torch.Tensor,
    bone_transforms: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with warp kernel"""
    ensure_warp_initialized()
    B1, num_vertices, _ = vertices.shape
    max_bones_per_vertex = bone_indices.shape[-1]
    batch_size = max(B1, bone_transforms.shape[0])

    if vertices.dtype == torch.float32:
        vec3_dtype = wp.vec3f
        mat44_dtype = wp.mat44f
    elif vertices.dtype == torch.float64:
        vec3_dtype = wp.vec3d
        mat44_dtype = wp.mat44d
    elif vertices.dtype == torch.float16:
        vec3_dtype = wp.vec3h
        mat44_dtype = wp.mat44h
    else:
        raise NotImplementedError(f"Unsupported dtype {vertices.dtype}")

    vertices_scalar_dtype = wp.dtype_from_torch(vertices.dtype)
    device = wp.device_from_torch(vertices.device)

    wp_vertices = wp.from_torch(
        vertices.detach().contiguous(), dtype=vec3_dtype, requires_grad=False
    )
    wp_bone_weights = wp.from_torch(
        bone_weights.detach().contiguous(),
        dtype=wp.types.vector(max_bones_per_vertex, dtype=wp.dtype_from_torch(bone_weights.dtype)),
        requires_grad=False,
    )
    wp_bone_indices = wp.from_torch(
        bone_indices.detach().contiguous(),
        dtype=wp.types.vector(max_bones_per_vertex, dtype=wp.dtype_from_torch(bone_indices.dtype)),
        requires_grad=False,
    )
    wp_bone_transforms = wp.from_torch(
        bone_transforms.detach().contiguous(), dtype=mat44_dtype, requires_grad=False
    )
    wp_output = wp.zeros_like(wp_vertices, requires_grad=False)

    kernel = get_kernel(
        max_bones_per_vertex,
        vertices_scalar_dtype=vertices_scalar_dtype,
        weights_dtype=wp_bone_weights.dtype,
        indices_dtype=wp_bone_indices.dtype,
        vec3_dtype=vec3_dtype,
        mat44_dtype=mat44_dtype,
    )

    wp.launch(
        kernel,
        dim=(batch_size, num_vertices),
        inputs=[wp_vertices, wp_bone_weights, wp_bone_indices, wp_bone_transforms],
        outputs=[wp_output],
        device=device,
    )

    return wp.to_torch(wp_output)


@_lbs_custom_op_forward.register_fake
def _(vertices, bone_weights, bone_indices, bone_transforms):
    """Fake implementation for shape inference during torch.export"""
    return torch.empty_like(vertices)


@torch.library.custom_op("warp_lbs::lbs_backward", mutates_args=())
def _lbs_custom_op_backward(
    vertices: torch.Tensor,
    bone_weights: torch.Tensor,
    bone_indices: torch.Tensor,
    bone_transforms: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass for custom operator"""
    B1, num_vertices, _ = vertices.shape
    max_bones_per_vertex = bone_indices.shape[-1]
    batch_size = max(B1, bone_transforms.shape[0])

    if vertices.dtype == torch.float32:
        vec3_dtype = wp.vec3f
        mat44_dtype = wp.mat44f
    elif vertices.dtype == torch.float64:
        vec3_dtype = wp.vec3d
        mat44_dtype = wp.mat44d
    elif vertices.dtype == torch.float16:
        vec3_dtype = wp.vec3h
        mat44_dtype = wp.mat44h
    else:
        raise NotImplementedError(f"Unsupported dtype {vertices.dtype}")

    vertices_scalar_dtype = wp.dtype_from_torch(vertices.dtype)
    device = wp.device_from_torch(vertices.device)

    wp_vertices = wp.from_torch(
        vertices.detach().contiguous(), dtype=vec3_dtype, requires_grad=True
    )
    wp_bone_weights = wp.from_torch(
        bone_weights.detach().contiguous(),
        dtype=wp.types.vector(max_bones_per_vertex, dtype=wp.dtype_from_torch(bone_weights.dtype)),
        requires_grad=True,
    )
    wp_bone_indices = wp.from_torch(
        bone_indices.detach().contiguous(),
        dtype=wp.types.vector(max_bones_per_vertex, dtype=wp.dtype_from_torch(bone_indices.dtype)),
        requires_grad=False,
    )
    wp_bone_transforms = wp.from_torch(
        bone_transforms.detach().contiguous(), dtype=mat44_dtype, requires_grad=True
    )
    wp_output = wp.from_torch(output.detach().contiguous(), dtype=vec3_dtype, requires_grad=False)
    wp_grad_output = wp.from_torch(grad_output.contiguous(), dtype=vec3_dtype, requires_grad=False)

    kernel = get_kernel(
        max_bones_per_vertex,
        vertices_scalar_dtype=vertices_scalar_dtype,
        weights_dtype=wp_bone_weights.dtype,
        indices_dtype=wp_bone_indices.dtype,
        vec3_dtype=vec3_dtype,
        mat44_dtype=mat44_dtype,
    )

    wp.launch(
        kernel,
        dim=(batch_size, num_vertices),
        inputs=[wp_vertices, wp_bone_weights, wp_bone_indices, wp_bone_transforms],
        outputs=[wp_output],
        adj_inputs=[wp_vertices.grad, wp_bone_weights.grad, None, wp_bone_transforms.grad],
        adj_outputs=[wp_grad_output],
        adjoint=True,
        device=device,
    )

    grad_vertices = (
        wp.to_torch(wp_vertices.grad)
        if wp_vertices.grad is not None
        else torch.zeros_like(vertices)
    )
    grad_bone_weights = (
        wp.to_torch(wp_bone_weights.grad)
        if wp_bone_weights.grad is not None
        else torch.zeros_like(bone_weights)
    )
    grad_bone_transforms = (
        wp.to_torch(wp_bone_transforms.grad)
        if wp_bone_transforms.grad is not None
        else torch.zeros_like(bone_transforms)
    )

    return grad_vertices, grad_bone_weights, grad_bone_transforms


@_lbs_custom_op_backward.register_fake
def _(vertices, bone_weights, bone_indices, bone_transforms, output, grad_output):
    """Fake implementation for backward"""
    return (
        torch.empty_like(vertices),
        torch.empty_like(bone_weights),
        torch.empty_like(bone_transforms),
    )


def _custom_op_backward_wrapper(ctx, grad_output):
    """Backward wrapper for custom operator"""
    vertices, bone_weights, bone_indices, bone_transforms, output = ctx.saved_tensors
    grad_vertices, grad_bone_weights, grad_bone_transforms = _lbs_custom_op_backward(
        vertices, bone_weights, bone_indices, bone_transforms, output, grad_output.contiguous()
    )
    return grad_vertices, grad_bone_weights, None, grad_bone_transforms


def _custom_op_setup_context(ctx, inputs, output):
    """Setup context for backward pass"""
    vertices, bone_weights, bone_indices, bone_transforms = inputs
    ctx.save_for_backward(vertices, bone_weights, bone_indices, bone_transforms, output)


# Register autograd for custom operator
_lbs_custom_op_forward.register_autograd(
    _custom_op_backward_wrapper, setup_context=_custom_op_setup_context
)


def linear_blend_skinning(
    vertices: torch.Tensor,
    bone_weights: torch.Tensor,
    bone_indices: torch.Tensor,
    bone_transforms: torch.Tensor,
) -> torch.Tensor:
    """
    Apply linear blend skinning to vertices using bone weights and transforms.

    This function is compatible with both regular PyTorch usage and torch.export.

    Args:
        vertices: (batch_size, num_vertices, 3) vertex positions
        bone_weights: (num_vertices, max_bones_per_vertex) bone weights
        bone_indices: (num_vertices, max_bones_per_vertex) bone indices
        bone_transforms: (batch_size, num_bones, 4, 4) bone transformation matrices

    Returns:
        Transformed vertices of shape (batch_size, num_vertices, 3)
    """
    return _lbs_custom_op_forward(vertices, bone_weights, bone_indices, bone_transforms)
