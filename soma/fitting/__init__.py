# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fitting algorithms: pose inversion from posed vertices and post-fit smoothing.

These invert the SOMA body and hand layers (recover skeleton rotations from
vertices) and regularize the resulting pose trajectories. They are shared by
the body and hand packages and by the conversion tools.
"""

from .pose_inversion import PoseInversion, PoseInversionResult
from .pose_inversion_mhr import MHRPoseInversion, MHRPoseInversionResult
from .rts_smoothing import (
    DEFAULT_RTS_SMOOTHING_CONFIG,
    RTS_SMOOTHING_PRESETS,
    STRONG_RTS_SMOOTHING_CONFIG,
    RTSSmoothingConfig,
    RTSSmoothingGains,
    RTSSmoothingGroups,
    smooth_pose,
)

__all__ = [
    "PoseInversion",
    "PoseInversionResult",
    "MHRPoseInversion",
    "MHRPoseInversionResult",
    "DEFAULT_RTS_SMOOTHING_CONFIG",
    "RTS_SMOOTHING_PRESETS",
    "STRONG_RTS_SMOOTHING_CONFIG",
    "RTSSmoothingConfig",
    "RTSSmoothingGains",
    "RTSSmoothingGroups",
    "smooth_pose",
]
