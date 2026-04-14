# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__version__ = "0.1.0"

from .assets import get_assets_dir
from .geometry.rig_utils import remove_joint_orient_local
from .identity_model import BaseIdentityModel, create_identity_model
from .io import add_npz_args, save_soma_npz
from .pose_inversion import HIPS_IDX, PoseInversion, build_world_transforms
from .soma import SOMALayer
from .units import Unit

# Backward compatibility: prefer SOMALayer
SomaLayer = SOMALayer


def setup_warp_for_ddp():
    """
    Call this at the start of each DDP worker process, before creating SOMALayer.

    Example::

        def ddp_worker(rank, world_size):
            soma.setup_warp_for_ddp()  # sets PYTORCH_NO_CUDA_MEMORY_CACHING internally
            import torch
            torch.cuda.set_device(rank)
            ...
    """
    import os

    os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    from soma.geometry._warp_init import ensure_warp_initialized

    ensure_warp_initialized()


__all__ = [
    "__version__",
    "get_assets_dir",
    "build_world_transforms",
    "HIPS_IDX",
    "PoseInversion",
    "SOMALayer",
    "SomaLayer",
    "Unit",
    "BaseIdentityModel",
    "remove_joint_orient_local",
    "add_npz_args",
    "create_identity_model",
    "save_soma_npz",
    "setup_warp_for_ddp",
]
