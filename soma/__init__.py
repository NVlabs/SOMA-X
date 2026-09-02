# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public SOMA-X package exports."""

__version__ = "0.3.0"

import sys as _sys

from . import body as _body
from . import fitting as _fitting
from .assets import get_assets_dir
from .body import SOMALayer, SOMAPoseOutput, create_identity_model
from .geometry.rig_utils import remove_joint_orient_local
from .hand import MANOLayer, SOMAHandLayer, SOMAHandPoseOutput
from .identity_model import BaseIdentityModel
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
from .units import Unit

# Backward compatibility: prefer SOMALayer
SomaLayer = SOMALayer

# Legacy module paths. Registering the implementation modules under their
# pre-0.3 names makes ``import soma.soma`` / ``from soma.pose_inversion import X``
# and pickled class references resolve without keeping shim files around.
_LEGACY_MODULES = {
    "soma.soma": _body.soma,
    "soma.pose_inversion": _fitting.pose_inversion,
    "soma.pose_inversion_mhr": _fitting.pose_inversion_mhr,
    "soma.rts_smoothing": _fitting.rts_smoothing,
}
for _name, _module in _LEGACY_MODULES.items():
    _sys.modules.setdefault(_name, _module)
    setattr(_sys.modules[__name__], _name.rsplit(".", 1)[1], _module)


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
    "SOMAPoseOutput",
    "SOMAHandLayer",
    "SOMAHandPoseOutput",
    "MANOLayer",
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
