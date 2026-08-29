# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SOMA-X package exports."""

__version__ = "0.2.4"

from pathlib import Path

from .assets import get_assets_dir
from .geometry.rig_utils import remove_joint_orient_local
from .identity_model import BaseIdentityModel, create_identity_model
from .io import (
    SOMA_TEMPLATE_RIG_FILENAME,
    SOMA_XLO_TEMPLATE_RIG_FILENAME,
    add_npz_args,
    fan_triangulate,
    find_lod_skin_mesh_name,
    list_usd_meshes,
    load_lod_rig_from_usd,
    load_lod_rigs_from_usd,
    load_rig_from_usd,
    load_usd_animation,
    load_usd_mesh,
    load_usd_skeleton,
    load_usd_skinning,
    save_soma_npz,
    save_soma_usd,
    write_usd_mesh,
)
from .smpl import (
    SMPLFamilyPoseTransferResult,
    SMPLFamilyTopologyBridge,
    SMPLLayer,
    SMPLXLayer,
    create_smpl_family_layer,
    transfer_smpl_family_pose_parameters,
)
from .soma import SOMALayer
from .units import Unit

# Backward compatibility: prefer SOMALayer
SomaLayer = SOMALayer

_OPTIONAL_EXPORTS = {}
if (Path(__file__).with_name("ha" "nd") / "__init__.py").is_file():
    from importlib import import_module

    _OPTIONAL_EXPORTS.update(
        {
            "SOMA" "HandLayer": ("soma." "hand", "SOMA" "HandLayer"),
            "MA" "NOLayer": ("soma." "hand.mano", "MA" "NOLayer"),
        }
    )


def __getattr__(name: str):
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def setup_warp_for_ddp() -> None:
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
    "SOMALayer",
    "SMPLLayer",
    "SMPLXLayer",
    "SomaLayer",
    "Unit",
    "BaseIdentityModel",
    "SMPLFamilyPoseTransferResult",
    "SMPLFamilyTopologyBridge",
    "remove_joint_orient_local",
    "SOMA_TEMPLATE_RIG_FILENAME",
    "SOMA_XLO_TEMPLATE_RIG_FILENAME",
    "add_npz_args",
    "create_identity_model",
    "create_smpl_family_layer",
    "transfer_smpl_family_pose_parameters",
    "fan_triangulate",
    "find_lod_skin_mesh_name",
    "list_usd_meshes",
    "load_lod_rig_from_usd",
    "load_lod_rigs_from_usd",
    "load_rig_from_usd",
    "load_usd_animation",
    "load_usd_mesh",
    "load_usd_skeleton",
    "load_usd_skinning",
    "save_soma_npz",
    "save_soma_usd",
    "write_usd_mesh",
    "setup_warp_for_ddp",
]

__all__.extend(_OPTIONAL_EXPORTS)
