# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for Warp initialization helpers."""

from soma.geometry._warp_utils import ensure_warp_initialized

__all__ = ["ensure_warp_initialized"]
