# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared NVIDIA Warp helpers for SOMA.

Defers wp.init() until first actual kernel use, allowing DDP workers to properly
set up their CUDA context before Warp initializes.
"""

import os
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any

import warp as wp

wp.config.enable_mempools_at_init = False  # set config before any init

_initialized = False
_fork_hook_registered = False
_warp_kernel_cache: dict[tuple[Callable[..., Any], tuple[tuple[str, object], ...]], Any] = {}


def _disable_cuda_context_in_child():
    """Called in child process after os.fork().

    After fork, CUDA contexts inherited from the parent are invalid. Warp's context
    manager calls is_cuda_driver_initialized() around every kernel launch (even CPU
    kernels) to save/restore the current CUDA context. Patching it to return False
    prevents CUDA error 3 from appearing in worker stderr.
    """
    import warp._src.context as _wc

    _wc.is_cuda_driver_initialized = lambda: False


def ensure_warp_initialized():
    global _initialized, _fork_hook_registered
    if not _initialized:
        wp.init()
        _initialized = True
    if not _fork_hook_registered:
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_disable_cuda_context_in_child)
        _fork_hook_registered = True


def _normalize_cache_arg(value: Any) -> object:
    if isinstance(value, list | tuple):
        value = tuple(_normalize_cache_arg(item) for item in value)
    elif isinstance(value, dict):
        value = tuple((key, _normalize_cache_arg(item)) for key, item in sorted(value.items()))
    elif isinstance(value, set | frozenset):
        value = frozenset(_normalize_cache_arg(item) for item in value)

    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            f"Cannot cache Warp kernel factory argument {value!r}; use hashable arguments."
        ) from exc

    return value


def cache_warp_kernel(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Cache Warp closure-factory kernels by specialization."""
    factory_signature = signature(factory)

    @wraps(factory)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_args = factory_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        key = (
            factory,
            tuple(
                (name, _normalize_cache_arg(value)) for name, value in bound_args.arguments.items()
            ),
        )
        if key not in _warp_kernel_cache:
            # Temporary workaround for Warp <= 1.13 closure factories retaining
            # Adjoint/Var objects. Remove once SOMA requires a Warp release with
            # the upstream closure-factory cache fix.
            _warp_kernel_cache[key] = factory(*args, **kwargs)
        return _warp_kernel_cache[key]

    return wrapper
