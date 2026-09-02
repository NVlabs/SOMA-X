# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-body SOMA layer and identity-aware posing pipeline.

This module defines :class:`SOMALayer`, the full-body entry point of the
library. SOMA maps every supported identity backend to a single canonical
mesh topology and 78-joint skeleton, enabling a shared LBS pipeline across
backends.

The sections below describe the SOMA skeleton joint layout and the shapes /
semantics of the tensors that ``SOMALayer`` consumes. Shapes use ``B`` for
batch size, ``V`` for the LOD-dependent body vertex count, and ``J = 78``
for the full-body joint count.

LOD vertex counts
-----------------

The posed vertex tensor has shape ``(B, V, 3)``, where ``V`` depends on
the selected ``lod``:

.. list-table::
   :header-rows: 1
   :widths: 20 20

   * - ``lod``
     - Vertices
   * - ``"mid"``
     - 18,056
   * - ``"low"``
     - 4,505
   * - ``"xlo"``
     - 612

``low_lod=True`` is the legacy alias for ``lod="low"``.

SOMA skeleton (78 joints)
-------------------------

The full-body SOMA skeleton has ``J = 78`` joints indexed 0-77. Joint 0 is
a virtual ``Root`` that is always identity (padded internally, not
user-controllable). Joint 1 is ``Hips``.

User-facing ``poses`` and ``joints`` tensors have ``J - 1 = 77`` entries.
``poses[k]`` maps to internal joint ``k + 1``; so ``poses[0]`` is the
``Hips`` rotation, which is effectively the **global body rotation** in
the caller's frame.

.. list-table::
   :header-rows: 1
   :widths: 12 15 10 63

   * - User idx
     - Internal idx
     - Count
     - Region
   * - --
     - 0
     - 1
     - ``Root`` (virtual, always identity)
   * - 0-3
     - 1-4
     - 4
     - Hips, Spine1, Spine2, Chest
   * - 4-10
     - 5-11
     - 7
     - Neck1, Neck2, Head, HeadEnd, Jaw, LeftEye, RightEye
   * - 11-38
     - 12-39
     - 28
     - Left arm + hand (shoulder, arm, forearm, hand, 5 fingers)
   * - 39-66
     - 40-67
     - 28
     - Right arm + hand (mirror)
   * - 67-71
     - 68-72
     - 5
     - Left leg + foot (leg, shin, foot, toe-base, toe-end)
   * - 72-76
     - 73-77
     - 5
     - Right leg + foot (mirror)

Public joint names are available at runtime via ``layer._public_joint_names``
(78-element array including the virtual Root at index 0). In the legacy
78-joint rig this is the same as ``layer.rig_data["joint_names"]``; in
procedural mode ``rig_data`` is the expanded internal skinning skeleton.

Procedural transforms
---------------------

By default, SOMALayer keeps the expanded v0027 SOMA twist-joint skeleton from
the universal ``SOMA_template_rig.usda`` template, or from an explicit
``template_rig_path`` override. Passing ``enable_procedural_transforms=False``
opts out to the legacy 78-joint rig derived from that same template by pruning
procedural/auxiliary joints and aggregating their skin weights to the nearest
kept parent. The public ``poses`` input and all returned public outputs keep the
78-joint SOMA contract: ``joints`` has the 77 user-facing non-Root joints and
``transforms`` has the 78 public joints including Root. The expanded twist
skeleton is used internally for FK/LBS and cached bind data, but is not returned
from ``pose()`` / ``forward()``.

Procedural transforms require ``SOMA_procedural_transforms.json`` in
``data_root``. That sidecar is the runtime source of truth for twist topology and
parameter metadata, including the rotation extraction policy and sparse
parameter matrices. The template USD supplies the concrete skeleton and
bind/T-pose data.

Procedural mode uses a single SOMA-owned procedural parameter transform with
compiled rotation and translation matrices matching the segment driver
topology: forearm and shin twist come from hand and foot twist, while upper arm
and thigh twist use reverse start/end compensation. The SOMA implementation is
a sidecar-driven parameter-matrix path with SOMA-owned translations.
The JSON ``rotation_extraction`` policy selects ``"local_x_euler"``,
``"local_x_swing_twist"``, ``"aligned_x_swing_twist"``, or
per-procedural-joint mixes of those extractors. ``"local_x_euler"`` reads the
configured SOMA local-X twist channel from local Euler angles.
``"local_x_swing_twist"`` extracts the same configured channel by projecting a
half-angle stabilized source quaternion onto its SOMA twist axis.
``"aligned_x_swing_twist"`` extracts segment-aligned virtual twist from source
world transforms and bind alignment. The current pose-channel convention maps
arm local-X twist to axis X, left leg local-X twist to negative axis Y, and
right leg local-X twist to positive axis Y. The JSON translation matrix places
twist helpers along the fitted public segment, preserving identity and
body-part stretch instead of trusting independently fitted twist translations.
Pose correctives are supported only in this procedural-transform mode.

Pose tensor (``poses``)
-----------------------

Shape ``(B, 77, 3)`` axis-angle, or ``(B, 77, 3, 3)`` rotation matrices
when ``pose2rot=False``.

- ``poses[0]`` = Hips rotation = global body rotation in the caller's frame.
- Remaining entries are joint-local rotations, interpreted **relative to
  the T-pose joint orient** by default. This T-pose-relative convention
  matches SMPL-style pose parameters. Pass ``absolute_pose=True`` to
  treat them as already-oriented absolute local rotations instead.

Root translation (``transl``, optional)
---------------------------------------

Shape ``(B, 3)``. Hips translation in ``output_unit``. ``None`` keeps the
Hips at the origin.

Identity coefficients (``identity_coeffs``)
-------------------------------------------

Shape ``(B, K)``. ``K = num_shape_components`` is backend-dependent,
selected via the ``identity_model_type`` constructor argument.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - ``identity_model_type``
     - K
     - Source
   * - ``"mhr"`` (default)
     - 45
     - MHR body-shape identity
   * - ``"soma"``
     - 128
     - PCA in ``SOMA_neutral.npz``
   * - ``"smpl"`` / ``"smplh"`` / ``"smplx"``
     - varies (typ. 10)
     - SMPL / SMPL-H / SMPL-X betas
   * - ``"anny"``
     - data-driven
     - Anny phenotype dims
   * - ``"garment"``
     - data-driven
     - Garment-measurement PCA

Body-part scales (``scale_params``, optional)
---------------------------------------------

Backend-dependent. Pass ``None`` to skip.

.. list-table::
   :header-rows: 1
   :widths: 10 20 25 45

   * - Backend
     - Shape
     - When applied
     - Semantics
   * - ``mhr``
     - ``(B, 68)``
     - ``prepare_identity()``
     - **Required** for MHR. Native MHR body-part scale vector.
   * - ``soma``
     - ``(B, layer.num_scale_params)``
     - ``pose()``
     - Active limb-segment and finger bone-length scale ratios only.
       ``layer.scale_param_names`` gives the ordered child public-joint names
       and ``layer.scale_param_segments`` gives matching ``(parent, child)``
       local-translation edges. Each value multiplies that edge's local
       parent-to-child translation; 1.0 = no change. Hip/shoulder-width and
       other inactive joints are not exposed. Legacy 56-value tensors remain
       accepted and receive unit scales for the appended left/right Foot and
       ToeBase controls.
   * - ``anny``
     - data-driven
     - ``prepare_identity()``
     - Optional Anny local-change adjustments (per-body-part shape tweaks).
   * - Others
     - unused
     - --
     - Pass ``None``.

Correctives
-----------

Pose-dependent corrective vertex offsets are applied by default
(``apply_correctives=True`` on ``pose()`` / ``forward()``) when a corrective
checkpoint is loaded. Body correctives require procedural transforms. Pass
``apply_correctives=False`` for a pure LBS output, or construct with
``correctives_model_path=None`` to skip loading the checkpoint entirely.

Units
-----

Native unit of the SOMA rig is **centimeters**. Output unit is configurable
via ``output_unit`` in ``__init__`` (default ``Unit.METERS``). All
translational quantities returned by ``pose()`` / ``forward()`` --
vertices, joints, transforms -- are in ``output_unit``. Up axis ``+Y``,
forward axis ``+Z``, right-handed.
"""

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree

from ..correctives_model import (
    _DEFAULT_CORRECTIVES_MODEL_PATH,
    CorrectivesMLP,
    _resolve_correctives_model_path,
)
from ..geometry._warp_init import ensure_warp_initialized
from ..geometry.barycentric_interp import BarycentricInterpolator
from ..geometry.batched_skinning import BatchedSkinning, FKTopology
from ..geometry.lbs import batch_rodrigues
from ..geometry.rig_utils import (
    apply_joint_orient_local,
    joint_world_to_local,
    precompute_joint_orient,
)
from ..geometry.skeleton_transfer import SkeletonTransfer
from ..io import (
    SOMA_TEMPLATE_RIG_FILENAME,
    SOMA_XLO_TEMPLATE_RIG_FILENAME,
    fan_triangulate,
    load_lod_rigs_from_usd,
    missing_soma_neutral_rig_keys,
)
from ..procedural_transforms import (
    SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME,
    SOMAProceduralParameterTransform,
    derive_soma_rig_without_procedural_joints,
    has_soma_twist_joints,
    load_soma_procedural_transform_definition,
)
from ..units import Unit
from .identity_model import create_identity_model

logger = logging.getLogger(__name__)

BODY_LODS = ("mid", "low", "xlo")


def _resolve_body_lod(low_lod: bool, lod: str | None) -> str:
    """Resolve legacy ``low_lod`` and the explicit body LOD selector."""
    if lod is None:
        return "low" if low_lod else "mid"
    lod = lod.lower()
    if lod not in BODY_LODS:
        raise ValueError(f"lod must be one of {BODY_LODS}, got {lod!r}")
    if low_lod and lod != "low":
        raise ValueError("low_lod=True is only compatible with lod='low'")
    return lod


def _nearest_lod_vertex_ids(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_vertex_ids: np.ndarray,
) -> np.ndarray:
    """Map source vertex IDs to nearest target vertex IDs and drop duplicates."""
    if source_vertex_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    tree = cKDTree(target_vertices)
    _, nearest = tree.query(source_vertices[source_vertex_ids])
    return np.unique(nearest).astype(np.int64)


def _dense_skinning_weights(rig_data: Mapping[str, Any]) -> np.ndarray:
    """Return a dense ``(V, J)`` skinning-weight matrix from rig arrays."""
    return np.asarray(
        csc_matrix(
            (
                rig_data["skinning_weights_data"],
                rig_data["skinning_weights_indices"],
                rig_data["skinning_weights_indptr"],
            ),
            shape=rig_data["skinning_weights_shape"],
        ).todense()
    )


def _resolve_template_rig_path(data_root: Path, rig_path: str | Path | None) -> Path:
    if rig_path is not None:
        path = Path(rig_path)
        if not path.exists():
            raise FileNotFoundError(f"Template rig asset not found: {path}")
        return path
    return data_root / SOMA_TEMPLATE_RIG_FILENAME


def _public_joint_names_from_assets(
    rig_data: Mapping[str, Any],
    procedural_definition,
    *,
    core_asset: Path,
    definition_path: Path,
) -> np.ndarray:
    if "joint_names" in rig_data:
        return np.array(rig_data["joint_names"]).copy()
    if procedural_definition is not None:
        return np.array(procedural_definition.public_joint_names)
    raise FileNotFoundError(
        f"Core asset '{core_asset}' does not contain joint_names. "
        f"Install '{definition_path.name}' next to it so the public SOMA joint contract "
        "can be derived from the procedural definition."
    )


def _raise_missing_template_for_slim_npz(
    missing_keys: Sequence[str],
    *,
    core_asset: Path,
    template_rig_path: Path,
) -> None:
    if missing_keys:
        raise FileNotFoundError(
            f"Template rig asset not found: {template_rig_path}. "
            f"Core asset '{core_asset}' is a slim SOMA_neutral.npz and no longer contains "
            f"rig fields: {', '.join(missing_keys)}. Install "
            f"'{SOMA_TEMPLATE_RIG_FILENAME}' next to the core asset."
        )


class SOMAPoseOutput(dict[str, torch.Tensor]):
    """Structured output returned by :obj:`~soma.body.SOMALayer.pose` and :obj:`~soma.body.SOMALayer.forward`.

    Behaves like a `dict` for backwards compatibility (`out["vertices"]`)
    while also supporting attribute access (`out.vertices`). `vertices` is
    absent when `fk_only=True`; `joints` and `transforms` are always
    populated.
    """

    vertices: torch.Tensor
    joints: torch.Tensor
    transforms: torch.Tensor

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


@dataclass(frozen=True)
class SOMAPublicRigView:
    """Public-joint view of a SOMA target skinning rig.

    Procedural SOMA layers keep an expanded target skeleton for LBS, but their
    public API is still the 78-joint SOMA body skeleton.  This view packages
    the public hierarchy, fitted bind transforms, and target-weight folding
    needed by downstream tools that should operate on public joints only.
    """

    joint_names: tuple[str, ...]
    joint_parent_ids: torch.Tensor
    target_joint_indices: torch.Tensor
    target_to_public_joint_indices: torch.Tensor
    bind_transforms_world: torch.Tensor
    bind_transforms_local: torch.Tensor
    t_pose_world: torch.Tensor
    skinning_weights: torch.Tensor


class SOMALayer(nn.Module):
    """Full-body parametric human model with a 78-joint SOMA skeleton.

    Combines a pluggable identity backend, identity-dependent skeleton
    fitting, LBS skinning (Warp-accelerated or dense), and optional
    pose-dependent corrective vertex offsets.

    Two-phase API:

    1. ``prepare_identity(identity_coeffs, scale_params=None)`` -- cache
       rest shape + fitted skeleton for an identity.
    2. ``pose(poses, transl=None)`` -- apply articulation to the cached
       identity.

    ``forward()`` is a convenience wrapper that calls both.

    See the :mod:`soma.body.soma` module docstring for the SOMA skeleton joint
    layout, pose tensor conventions, per-backend identity dimensions, and
    ``scale_params`` semantics.
    """

    LEGACY_NUM_BONE_SCALE_PARAMS = 56
    NUM_BONE_SCALE_PARAMS = 60  # Active public-joint local translation scales.
    BODY_BONE_SCALE_JOINT_NAMES = (
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftShin",
        "RightShin",
    )
    FINGER_BONE_SCALE_JOINT_PREFIXES = (
        "LeftHandThumb",
        "LeftHandIndex",
        "LeftHandMiddle",
        "LeftHandRing",
        "LeftHandPinky",
        "RightHandThumb",
        "RightHandIndex",
        "RightHandMiddle",
        "RightHandRing",
        "RightHandPinky",
    )
    FOOT_BONE_SCALE_JOINT_NAMES = (
        "LeftFoot",
        "RightFoot",
        "LeftToeBase",
        "RightToeBase",
    )

    def __init__(
        self,
        data_root: str | Path | None = None,
        low_lod: bool = False,
        device: str | torch.device = "cuda",
        identity_model_type: str = "mhr",
        mode: str = "warp",
        output_unit: Unit = Unit.METERS,
        identity_model_kwargs: Mapping[str, Any] | None = None,
        lod: str | None = None,
        template_rig_path: str | Path | None = None,
        enable_procedural_transforms: bool = True,
        load_correctives_model: bool | None = None,
        correctives_model_path: str | Path | None = _DEFAULT_CORRECTIVES_MODEL_PATH,
    ) -> None:
        """Build a SOMALayer with the selected identity backend.

        Args:
            data_root: Directory containing ``SOMA_neutral.npz``,
                ``SOMA_template_rig.usda``, and the per-backend model
                folders. If ``None`` or missing, assets are downloaded
                from HuggingFace automatically.
            low_lod: If ``True``, use the low-LOD mesh (4,505 vertices).
                Faster inference at the cost of detail. Legacy alias for
                ``lod="low"``.
            device: Torch device for all buffers and intermediate
                tensors (e.g. ``"cuda"``, ``"cpu"``).
            identity_model_type: Identity backend. One of ``"mhr"``
                (default), ``"soma"``, ``"smpl"``, ``"smplh"``,
                ``"smplx"``, ``"anny"``, ``"garment"``. See
                :mod:`soma.body.soma` for per-backend identity dimensions
                and ``scale_params`` semantics.
            mode: Skinning backend. ``"warp"`` uses the NVIDIA Warp
                accelerated LBS kernel; other values fall back to the
                dense PyTorch implementation.
            output_unit: Unit for all translational outputs of
                ``pose()`` / ``forward()`` (vertices, joints,
                transforms). Default ``Unit.METERS``.
            identity_model_kwargs: Extra keyword arguments forwarded to
                the identity-model constructor. Used e.g. by SMPL/SMPL-X
                to pass ``model_path``.
            lod: Body mesh level of detail: ``"mid"`` (18,056 vertices),
                ``"low"`` (4,505 vertices), or ``"xlo"`` (612 vertices).
                ``"xlo"`` loads mesh topology, bind shape, skinning
                weights, and UVs from
                the xlo mesh in ``SOMA_template_rig.usda`` in ``data_root``.
            template_rig_path: Optional override path to a v0027 SOMA
                twist-joint template. If omitted,
                ``data_root/SOMA_template_rig.usda`` is used as the universal
                template.
            enable_procedural_transforms: If ``True`` (default), keep the expanded
                v0027 twist-joint rig and drive SOMA-owned twist joints from the
                JSON procedural definition. ``False`` opts out to the legacy
                78-joint public rig.
            load_correctives_model: Deprecated compatibility alias. Use
                ``correctives_model_path=None`` instead of ``False``.
            correctives_model_path: Path to a pose-corrective checkpoint. Defaults
                to ``data_root/correctives_model.pt`` when procedural transforms
                are enabled. Pass ``None`` to skip loading correctives.
        """
        super().__init__()

        self.lod = _resolve_body_lod(low_lod, lod)
        self.identity_model_kwargs = dict(identity_model_kwargs or {})

        if data_root is None or not Path(data_root).exists():
            if data_root is not None:
                logger.info(
                    "data_root '%s' not found, downloading assets from HuggingFace...",
                    data_root,
                )
            else:
                logger.info("No data_root provided, downloading assets from HuggingFace...")
            from .assets import get_assets_dir

            data_root = get_assets_dir()

        data_root = Path(data_root)
        self.correctives_model_path = _resolve_correctives_model_path(
            data_root=data_root,
            correctives_model_path=correctives_model_path,
            load_correctives_model=load_correctives_model,
            default_enabled=enable_procedural_transforms,
        )

        # Check for core asset file
        core_asset = data_root / "SOMA_neutral.npz"
        if not core_asset.exists():
            raise FileNotFoundError(
                f"Core asset 'SOMA_neutral.npz' not found in '{data_root}'.\n"
                "Please ensure the assets are correctly downloaded and extracted."
            )

        self.identity_model_type = identity_model_type
        try:
            self.rig_data = dict(np.load(core_asset, allow_pickle=False))
        except Exception as e:
            raise RuntimeError(
                f"Error loading core asset 'SOMA_neutral.npz': {e}\n"
                "Please ensure the assets are correctly downloaded with 'git lfs pull'."
            ) from e

        self.procedural_transforms_enabled = enable_procedural_transforms
        self.procedural_template_rig_path = None
        self.procedural_transform_definition = None
        procedural_transform_segments = None
        procedural_rotation_extraction_modes = None
        procedural_rotation_entries = None
        procedural_translation_entries = None

        # Merge rig tensors from the canonical template USD.
        # Shape PCA data and mesh topology remain sourced from SOMA_neutral.npz.
        template_rig_path = _resolve_template_rig_path(data_root, template_rig_path)
        definition_path = data_root / SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME
        if definition_path.exists():
            self.procedural_transform_definition = load_soma_procedural_transform_definition(
                definition_path
            )
            procedural_transform_segments = self.procedural_transform_definition.segments
            procedural_rotation_extraction_modes = (
                self.procedural_transform_definition.rotation_extraction_modes
            )
            procedural_rotation_entries = self.procedural_transform_definition.rotation_entries
            procedural_translation_entries = (
                self.procedural_transform_definition.translation_entries
            )
            if "joint_names" in self.rig_data and tuple(
                map(str, self.rig_data["joint_names"])
            ) != tuple(self.procedural_transform_definition.public_joint_names):
                raise ValueError(
                    "SOMA procedural transform definition public joints do not match "
                    "SOMA_neutral.npz joint_names."
                )
        elif self.procedural_transforms_enabled:
            raise FileNotFoundError(
                "Procedural transforms require the SOMA procedural transform definition "
                f"at '{definition_path}'."
            )
        template_mid_rig_data = None
        public_joint_names = _public_joint_names_from_assets(
            self.rig_data,
            self.procedural_transform_definition,
            core_asset=core_asset,
            definition_path=definition_path,
        )
        template_lod_rigs = {}
        if template_rig_path.exists():
            self.procedural_template_rig_path = template_rig_path
            template_lods = ("mid", "low", "xlo") if self.lod == "xlo" else ("mid",)
            template_lod_rigs = load_lod_rigs_from_usd(template_rig_path, template_lods)
            template_mid_rig_data = template_lod_rigs["mid"]
            public_mid_rig_data = derive_soma_rig_without_procedural_joints(
                template_mid_rig_data,
                public_joint_names,
                segments=procedural_transform_segments,
            )
            self.rig_data.update(public_mid_rig_data)
        elif self.procedural_transforms_enabled:
            raise FileNotFoundError(
                "Procedural transforms require the universal SOMA template rig with "
                f"twist joints at '{template_rig_path}'."
            )
        else:
            _raise_missing_template_for_slim_npz(
                missing_soma_neutral_rig_keys(self.rig_data),
                core_asset=core_asset,
                template_rig_path=template_rig_path,
            )
        self._public_joint_names = np.array(self.rig_data["joint_names"]).copy()
        self._public_mid_rig_data = dict(self.rig_data)
        if self.procedural_transforms_enabled:
            if template_mid_rig_data is None or not has_soma_twist_joints(
                template_mid_rig_data["joint_names"],
                segments=procedural_transform_segments,
            ):
                raise ValueError(
                    "Procedural transforms require a SOMA template rig with twist joints."
                )
            self.rig_data.update(template_mid_rig_data)
        self._mid_rig_data = dict(self.rig_data)
        xlo_skeleton_lod_rig_data = None
        xlo_skeleton_transfer_rig_data = None
        if self.lod == "xlo":
            xlo_usd_rig = template_rig_path
            if not xlo_usd_rig.exists():
                raise FileNotFoundError(
                    f"XLO LOD requested, but '{SOMA_XLO_TEMPLATE_RIG_FILENAME}' "
                    f"was not found in '{data_root}'. Copy the versioned SOMA "
                    "v0027 template USD into the assets directory under the "
                    "canonical SOMA template filename."
                )
            xlo_skeleton_lod_rig_data = template_lod_rigs["low"]
            xlo_rig_data = template_lod_rigs["xlo"]
            xlo_skeleton_transfer_rig_data = xlo_skeleton_lod_rig_data
            if not self.procedural_transforms_enabled:
                xlo_skeleton_lod_rig_data = derive_soma_rig_without_procedural_joints(
                    xlo_skeleton_lod_rig_data,
                    public_joint_names,
                    segments=procedural_transform_segments,
                )
                xlo_rig_data = derive_soma_rig_without_procedural_joints(
                    xlo_rig_data,
                    public_joint_names,
                    segments=procedural_transform_segments,
                )
                xlo_skeleton_transfer_rig_data = xlo_skeleton_lod_rig_data
            else:
                xlo_skeleton_transfer_rig_data = derive_soma_rig_without_procedural_joints(
                    xlo_skeleton_lod_rig_data,
                    public_joint_names,
                    segments=procedural_transform_segments,
                )
            self.rig_data.update(xlo_rig_data)
        self.device = device
        self.data_root = data_root
        self.low_lod = self.lod == "low"
        self.mode = mode
        self.output_unit = output_unit
        self.root_joint_idx = 1  # Hips (child of virtual root at 0)

        # Pre-initialize Warp in the main process so DataLoader forked workers
        # inherit _initialized=True and skip wp.init() (avoids CUDA error 3 in workers).
        ensure_warp_initialized()

        shape_mean = torch.from_numpy(self.rig_data["mean"]).to(device)
        self.register_buffer("shape_mean", shape_mean, persistent=False)
        self.register_buffer(
            "shape_pca", torch.from_numpy(self.rig_data["shapedirs"]).to(device), persistent=False
        )
        self.register_buffer(
            "shape_eigenvalues",
            torch.from_numpy(self.rig_data["eigenvalues"]).to(device),
            persistent=False,
        )
        self.num_shape_components = self.shape_pca.shape[0]
        self.parents = [i - 1 for i in self.rig_data["joint_parent_ids"]][1:]

        bind_shape = torch.from_numpy(self.rig_data["bind_shape"]).to(device)
        skinning_weights_np = _dense_skinning_weights(self.rig_data)
        skinning_weights = skinning_weights_np
        skeleton_rig_data = (
            self._public_mid_rig_data if self.procedural_transforms_enabled else self.rig_data
        )
        skeleton_bind_shape = torch.from_numpy(skeleton_rig_data["bind_shape"]).to(device)
        skeleton_skinning_weights = _dense_skinning_weights(skeleton_rig_data)
        skeleton_transfer_bind_shape = skeleton_bind_shape
        skeleton_transfer_skinning_weights = skeleton_skinning_weights
        self.identity_lod_transfer = None
        self.xlo_skeleton_transfer = None
        identity_uses_low_lod = self.lod == "low"
        identity_lod_mid_to_low = None
        identity_lod_faces = None
        identity_vertex_ids_to_exclude = None
        correctives_vertex_index_map = None
        xlo_skeleton_joint_parent_ids = None
        xlo_skeleton_bind_pose_world = None
        xlo_skeleton_excluded_vert_ids = None
        if self.lod == "low":
            nv_lod_mid_to_low = self._mid_rig_data["lod_mid_to_low"]
            self.register_buffer(
                "nv_lod_mid_to_low",
                torch.from_numpy(nv_lod_mid_to_low).long().to(device),
                persistent=False,
            )
            self.register_buffer(
                "faces",
                torch.from_numpy(self.rig_data["triangles_low"]).to(device),
                persistent=False,
            )
            self.register_buffer("bind_shape", bind_shape[nv_lod_mid_to_low], persistent=False)
            skinning_weights = skinning_weights[nv_lod_mid_to_low]
            skeleton_transfer_bind_shape = skeleton_bind_shape[nv_lod_mid_to_low]
            skeleton_transfer_skinning_weights = skeleton_skinning_weights[nv_lod_mid_to_low]
            self.register_buffer("xlo_skeleton_mid_to_low", None, persistent=False)
            identity_uses_low_lod = True
            identity_lod_mid_to_low = self.nv_lod_mid_to_low
            identity_lod_faces = self.faces
            correctives_vertex_index_map = self.nv_lod_mid_to_low
        elif self.lod == "xlo":
            nv_lod_mid_to_low = self._mid_rig_data["lod_mid_to_low"]
            self.register_buffer(
                "xlo_skeleton_mid_to_low",
                torch.from_numpy(nv_lod_mid_to_low).long().to(device),
                persistent=False,
            )
            if "face_vert_indices" not in self.rig_data or "face_vert_counts" not in self.rig_data:
                raise RuntimeError(
                    f"XLO LOD asset '{SOMA_XLO_TEMPLATE_RIG_FILENAME}' does not contain "
                    "mesh face topology."
                )
            faces_xlo = fan_triangulate(
                self.rig_data["face_vert_indices"],
                self.rig_data["face_vert_counts"],
            )
            self.register_buffer("faces", torch.from_numpy(faces_xlo).to(device), persistent=False)
            self.register_buffer("bind_shape", bind_shape.to(device), persistent=False)
            self.nv_lod_mid_to_low = None

            xlo_transfer_skinning_weights = _dense_skinning_weights(xlo_skeleton_transfer_rig_data)
            xlo_skeleton_transfer_bind_shape = torch.from_numpy(
                xlo_skeleton_transfer_rig_data["bind_shape"]
            ).to(device)
            xlo_skeleton_transfer_skinning_weights = torch.from_numpy(
                xlo_transfer_skinning_weights
            ).to(device)
            xlo_skeleton_joint_parent_ids = torch.from_numpy(
                xlo_skeleton_transfer_rig_data["joint_parent_ids"]
            ).to(device)
            xlo_skeleton_bind_pose_world = torch.from_numpy(
                xlo_skeleton_transfer_rig_data["bind_pose_world"]
            ).to(device)
            if (
                "face_vert_indices" not in xlo_skeleton_transfer_rig_data
                or "face_vert_counts" not in xlo_skeleton_transfer_rig_data
            ):
                raise RuntimeError(
                    f"Low LOD asset in '{SOMA_XLO_TEMPLATE_RIG_FILENAME}' does not contain "
                    "mesh face topology required for XLO identity transfer."
                )
            faces_low = fan_triangulate(
                xlo_skeleton_transfer_rig_data["face_vert_indices"],
                xlo_skeleton_transfer_rig_data["face_vert_counts"],
            )
            low_faces = torch.from_numpy(faces_low).long().to(device)
            self.identity_lod_transfer = BarycentricInterpolator(
                xlo_skeleton_transfer_bind_shape.float(),
                low_faces,
                bind_shape.float(),
            )
            identity_uses_low_lod = True
            identity_lod_mid_to_low = self.xlo_skeleton_mid_to_low
            identity_lod_faces = low_faces
            correctives_vertex_index_map = self.xlo_skeleton_mid_to_low
        else:
            self.register_buffer(
                "faces", torch.from_numpy(self.rig_data["triangles"]).to(device), persistent=False
            )
            self.register_buffer("bind_shape", bind_shape.to(device), persistent=False)
            self.nv_lod_mid_to_low = None
            self.register_buffer("xlo_skeleton_mid_to_low", None, persistent=False)

        facial_inner_geometry_np = np.concatenate(
            [
                self._mid_rig_data["segment_eye_bags"],
                self._mid_rig_data["segment_mouth_bag"],
            ]
        )
        identity_vertex_ids_to_exclude = torch.from_numpy(facial_inner_geometry_np).to(device)

        if self.lod == "low":
            facial_inner_geometry = torch.from_numpy(facial_inner_geometry_np).to(device)
            num_high_verts = self._mid_rig_data["bind_shape"].shape[0]
            inverse_lod_map = torch.full((num_high_verts,), -1, dtype=torch.long, device=device)
            inverse_lod_map[self.nv_lod_mid_to_low] = torch.arange(
                self.nv_lod_mid_to_low.shape[0], device=device
            )
            facial_low = inverse_lod_map[facial_inner_geometry.long()]
            facial_inner_geometry = facial_low[facial_low >= 0]
            identity_vertex_ids_to_exclude = facial_inner_geometry
        elif self.lod == "xlo":
            facial_inner_geometry = torch.from_numpy(
                _nearest_lod_vertex_ids(
                    self._mid_rig_data["bind_shape"],
                    self.rig_data["bind_shape"],
                    facial_inner_geometry_np,
                )
            ).to(device)
            num_mid_verts = self._mid_rig_data["bind_shape"].shape[0]
            inverse_lod_map = torch.full((num_mid_verts,), -1, dtype=torch.long, device=device)
            inverse_lod_map[self.xlo_skeleton_mid_to_low] = torch.arange(
                self.xlo_skeleton_mid_to_low.shape[0], device=device
            )
            facial_low = inverse_lod_map[
                torch.from_numpy(facial_inner_geometry_np).long().to(device)
            ]
            xlo_skeleton_excluded_vert_ids = facial_low[facial_low >= 0]
            identity_vertex_ids_to_exclude = xlo_skeleton_excluded_vert_ids
        else:
            facial_inner_geometry = torch.from_numpy(facial_inner_geometry_np).to(device)

        self.register_buffer(
            "skinning_weights", torch.from_numpy(skinning_weights).to(device), persistent=False
        )
        self.register_buffer(
            "joint_parent_ids",
            torch.from_numpy(self.rig_data["joint_parent_ids"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "bind_pose_world",
            torch.from_numpy(self.rig_data["bind_pose_world"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "bind_pose_local",
            torch.from_numpy(self.rig_data["bind_pose_local"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "t_pose_world",
            torch.from_numpy(self.rig_data["t_pose_world"]).to(device),
            persistent=False,
        )
        self.register_buffer(
            "t_pose_local",
            torch.from_numpy(self.rig_data["t_pose_local"]).to(device),
            persistent=False,
        )
        target_name_to_idx = {
            str(name): idx for idx, name in enumerate(self.rig_data["joint_names"])
        }
        missing_public = [
            str(name) for name in self._public_joint_names if str(name) not in target_name_to_idx
        ]
        if missing_public:
            raise ValueError(f"Twist rig is missing public SOMA joints: {sorted(missing_public)}")
        self.register_buffer(
            "public_joint_indices",
            torch.tensor(
                [target_name_to_idx[str(name)] for name in self._public_joint_names[1:]],
                dtype=torch.long,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "public_transform_joint_indices",
            torch.tensor(
                [target_name_to_idx[str(name)] for name in self._public_joint_names],
                dtype=torch.long,
                device=device,
            ),
            persistent=False,
        )
        public_name_to_idx = {str(name): idx for idx, name in enumerate(self._public_joint_names)}
        target_joint_names = [str(name) for name in self.rig_data["joint_names"]]
        target_parent_ids = np.asarray(self.rig_data["joint_parent_ids"], dtype=np.int64)
        public_parent_ids = []
        for name in self._public_joint_names:
            target_idx = target_name_to_idx[str(name)]
            parent_idx = int(target_parent_ids[target_idx])
            while target_joint_names[parent_idx] not in public_name_to_idx:
                next_parent_idx = int(target_parent_ids[parent_idx])
                if next_parent_idx == parent_idx:
                    break
                parent_idx = next_parent_idx
            public_parent_ids.append(public_name_to_idx.get(target_joint_names[parent_idx], 0))
        self.register_buffer(
            "public_joint_parent_ids",
            torch.tensor(public_parent_ids, dtype=self.joint_parent_ids.dtype, device=device),
            persistent=False,
        )
        legacy_bone_scale_public_joint_indices = [
            idx
            for idx, name in enumerate(self._public_joint_names)
            if self._is_body_bone_scale_joint(str(name))
        ]
        if len(legacy_bone_scale_public_joint_indices) != self.LEGACY_NUM_BONE_SCALE_PARAMS:
            raise RuntimeError(
                "Unexpected legacy SOMA bone-scale parameter layout: "
                f"expected {self.LEGACY_NUM_BONE_SCALE_PARAMS}, "
                f"got {len(legacy_bone_scale_public_joint_indices)}"
            )
        missing_lower_limb_scale_joints = [
            name
            for name in self.FOOT_BONE_SCALE_JOINT_NAMES
            if name not in public_name_to_idx
        ]
        if missing_lower_limb_scale_joints:
            raise RuntimeError(
                "SOMA public rig is missing foot-scaling joints: "
                f"{missing_lower_limb_scale_joints}"
            )
        bone_scale_public_joint_indices = [
            *legacy_bone_scale_public_joint_indices,
            *(public_name_to_idx[name] for name in self.FOOT_BONE_SCALE_JOINT_NAMES),
        ]
        self.soma_bone_scale_param_names = tuple(
            str(self._public_joint_names[idx]) for idx in bone_scale_public_joint_indices
        )
        self.soma_bone_scale_param_segments = tuple(
            (
                str(self._public_joint_names[public_parent_ids[idx]]),
                str(self._public_joint_names[idx]),
            )
            for idx in bone_scale_public_joint_indices
        )
        if len(self.soma_bone_scale_param_names) != self.NUM_BONE_SCALE_PARAMS:
            raise RuntimeError(
                "Unexpected SOMA bone-scale parameter layout: "
                f"expected {self.NUM_BONE_SCALE_PARAMS}, "
                f"got {len(self.soma_bone_scale_param_names)}"
            )
        self.register_buffer(
            "bone_scale_public_joint_indices",
            torch.tensor(bone_scale_public_joint_indices, dtype=torch.long, device=device),
            persistent=False,
        )
        target_to_public = []
        for target_idx, _target_name in enumerate(target_joint_names):
            parent_idx = target_idx
            while target_joint_names[parent_idx] not in public_name_to_idx:
                next_parent_idx = int(target_parent_ids[parent_idx])
                if next_parent_idx == parent_idx:
                    break
                parent_idx = next_parent_idx
            target_to_public.append(public_name_to_idx.get(target_joint_names[parent_idx], 0))
        self.register_buffer(
            "target_to_public_joint_indices",
            torch.tensor(target_to_public, dtype=torch.long, device=device),
            persistent=False,
        )
        self.procedural_transforms = None
        if self.procedural_transforms_enabled:
            self.procedural_transforms = SOMAProceduralParameterTransform(
                self._public_joint_names,
                self.rig_data["joint_names"],
                rotation_extraction_modes=procedural_rotation_extraction_modes,
                segments=procedural_transform_segments,
                rotation_entries=procedural_rotation_entries,
                translation_entries=procedural_translation_entries,
                target_t_pose_world=self.rig_data["t_pose_world"],
                target_joint_parent_ids=self.rig_data["joint_parent_ids"],
                target_bind_pose_world=self.rig_data["bind_pose_world"],
            ).to(device)
        self.register_buffer("excluded_vert_ids", facial_inner_geometry, persistent=False)
        # Backward-compatible alias
        self.facial_inner_geometry = self.excluded_vert_ids

        self.skeleton_transfer = None
        if self.lod != "xlo":
            self.skeleton_transfer = SkeletonTransfer(
                torch.from_numpy(skeleton_rig_data["joint_parent_ids"]).to(device),
                torch.from_numpy(skeleton_rig_data["bind_pose_world"]).to(device),
                skeleton_transfer_bind_shape,
                torch.from_numpy(skeleton_transfer_skinning_weights).to(device),
                rotation_method="auto",
                vertex_ids_to_exclude=self.facial_inner_geometry,
            )
        if self.lod == "xlo":
            self.xlo_skeleton_transfer = SkeletonTransfer(
                xlo_skeleton_joint_parent_ids,
                xlo_skeleton_bind_pose_world,
                xlo_skeleton_transfer_bind_shape,
                xlo_skeleton_transfer_skinning_weights,
                rotation_method="auto",
                vertex_ids_to_exclude=xlo_skeleton_excluded_vert_ids,
            )
            self.skeleton_transfer = self.xlo_skeleton_transfer

        source_fk = None
        if self.procedural_transforms is not None:
            source_fk = FKTopology(
                parent_ids=self.public_joint_parent_ids,
                target_joint_indices=self.public_transform_joint_indices,
                global_translation_joint_idx=self.root_joint_idx,
            )
        self.batched_skinning = BatchedSkinning(
            self.joint_parent_ids,
            self.skinning_weights,
            self.bind_pose_world,
            self.bind_shape,
            joint_orient=self.t_pose_world,
            mode=self.mode,
            source_fk=source_fk,
        )
        self.public_batched_skinning = self._make_public_batched_skinning()

        self.identity_model = create_identity_model(
            identity_model_type,
            data_root,
            identity_uses_low_lod,
            device,
            output_unit=output_unit,
            nv_lod_mid_to_low=identity_lod_mid_to_low,
            soma_low_lod_faces=identity_lod_faces,
            vertex_ids_to_exclude=identity_vertex_ids_to_exclude,
            **self.identity_model_kwargs,
        )
        if self.identity_model_type == "soma":
            self.scale_param_names = self.soma_bone_scale_param_names
            self.scale_param_segments = self.soma_bone_scale_param_segments
            self.num_scale_params = len(self.scale_param_names)
        else:
            identity_scale_names = getattr(self.identity_model, "scale_param_names", None)
            self.scale_param_names = (
                tuple(identity_scale_names) if identity_scale_names is not None else ()
            )
            self.scale_param_segments = ()
            self.num_scale_params = self.identity_model.num_scale_params

        self.correctives_model = None
        if self.correctives_model_path is not None:
            self.correctives_model = CorrectivesMLP.load_checkpoint(
                self.correctives_model_path,
                map_location=device,
                v_index_map=correctives_vertex_index_map,
                output_unit=output_unit,
            )
        self._corrective_config = {"first_joint_index": 0, "input_type": "tfm"}

        if self.t_pose_world is not None:
            self._t_pose_orient, self._t_pose_orient_parent_T = precompute_joint_orient(
                self.t_pose_world, self.joint_parent_ids
            )
        else:
            self._t_pose_orient = None
            self._t_pose_orient_parent_T = None

        self._cached_identity_rest_shape = None
        self._cached_rest_shape = None
        self._cached_bind_transforms_world = None
        self._cached_scale_params = None
        self._cached_global_scale = 1.0

    @property
    def default_skin_mesh_name(self) -> str:
        """Default USD skin-mesh prim name for this layer's topology.

        Consumed by :obj:`~soma.io.export_soma_usd` when the caller does
        not pass an explicit `skin_mesh_name`. Uses the `c_skin_<lod>`
        convention (`mid` for mid-LOD, `lo` for low-LOD, `xlo` for extra-low LOD).
        """
        return {"mid": "c_skin_mid", "low": "c_skin_lo", "xlo": "c_skin_xlo"}[self.lod]

    @property
    def public_joint_names(self) -> tuple[str, ...]:
        """Names of the public SOMA joints returned by :meth:`pose`."""
        return tuple(str(name) for name in self._public_joint_names)

    @property
    def target_joint_names(self) -> tuple[str, ...]:
        """Names of the internal target skinning joints."""
        return tuple(str(name) for name in self.rig_data["joint_names"])

    @property
    def output_joint_parent_ids(self) -> torch.Tensor:
        """Parent ids for the public transforms returned by :meth:`pose`."""
        return self.public_joint_parent_ids

    def public_skinning_weights(self) -> torch.Tensor:
        """Return target skinning weights folded onto the public SOMA hierarchy."""
        weights = self.skinning_weights
        public_count = len(self.public_joint_names)
        if weights.shape[1] == public_count:
            return weights
        folded = torch.zeros(
            weights.shape[0],
            public_count,
            dtype=weights.dtype,
            device=weights.device,
        )
        return folded.index_add(1, self.target_to_public_joint_indices.to(weights.device), weights)

    def public_bind_transforms_world(
        self,
        bind_transforms_world: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select public-joint world bind transforms from target bind transforms."""
        if bind_transforms_world is None:
            if self._cached_bind_transforms_world is None:
                raise RuntimeError("No cached identity. Call prepare_identity() first.")
            bind_transforms_world = self._cached_bind_transforms_world
        public_indices = self.public_transform_joint_indices.to(bind_transforms_world.device)
        return bind_transforms_world[..., public_indices, :, :]

    def public_rig_view(
        self,
        bind_transforms_world: torch.Tensor | None = None,
    ) -> SOMAPublicRigView:
        """Return the public-joint view of this layer's current fitted rig."""
        public_bind_world = self.public_bind_transforms_world(bind_transforms_world)
        public_parent_ids = self.public_joint_parent_ids.to(public_bind_world.device)
        public_indices = self.public_transform_joint_indices.to(public_bind_world.device)
        return SOMAPublicRigView(
            joint_names=self.public_joint_names,
            joint_parent_ids=public_parent_ids,
            target_joint_indices=public_indices,
            target_to_public_joint_indices=self.target_to_public_joint_indices.to(
                self.skinning_weights.device
            ),
            bind_transforms_world=public_bind_world,
            bind_transforms_local=joint_world_to_local(public_bind_world, public_parent_ids),
            t_pose_world=self.t_pose_world[
                self.public_transform_joint_indices.to(self.t_pose_world.device)
            ],
            skinning_weights=self.public_skinning_weights(),
        )

    def _make_public_batched_skinning(self) -> BatchedSkinning | None:
        if self.procedural_transforms is None:
            return None
        public_indices = self.public_transform_joint_indices.to(self.bind_pose_world.device)
        return BatchedSkinning(
            self.public_joint_parent_ids,
            self.public_skinning_weights(),
            self.bind_pose_world[public_indices],
            self.bind_shape,
            joint_orient=self.t_pose_world[public_indices],
            mode=self.mode,
            global_translation_joint_idx=self.root_joint_idx,
        )

    def to_public_rotations(
        self, rotations: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Reduce target-joint rotations to the public SOMA joint order."""
        if not isinstance(rotations, torch.Tensor):
            rotations = np.asarray(rotations)
        public_count = len(self.public_joint_names)
        target_count = len(self.target_joint_names)
        rotation_count = rotations.shape[-3]
        if rotation_count == public_count:
            return rotations
        if rotation_count == target_count:
            if isinstance(rotations, torch.Tensor):
                return rotations[
                    ...,
                    self.public_transform_joint_indices.to(rotations.device),
                    :,
                    :,
                ]
            return rotations[
                ...,
                self.public_transform_joint_indices.detach().cpu().numpy(),
                :,
                :,
            ]
        raise ValueError(
            f"Expected rotations for {public_count} public joints or "
            f"{target_count} target joints, got {rotation_count}."
        )

    def _apply(self, fn) -> "SOMALayer":
        super()._apply(fn)
        self.device = self.bind_pose_world.device
        self.dtype = self.bind_pose_world.dtype
        # BatchedSkinning is not an nn.Module, so its internal tensors (bone weights,
        # joint orient, skeleton levels, etc.) are not moved by the default _apply.
        # Reinitialize it from the registered buffers, which are now on the new device.
        source_fk = None
        if self.procedural_transforms is not None:
            source_fk = FKTopology(
                parent_ids=self.public_joint_parent_ids,
                target_joint_indices=self.public_transform_joint_indices,
                global_translation_joint_idx=self.root_joint_idx,
            )
        self.batched_skinning = BatchedSkinning(
            self.joint_parent_ids,
            self.skinning_weights,
            self.bind_pose_world,
            self.bind_shape,
            joint_orient=self.t_pose_world,
            mode=self.mode,
            source_fk=source_fk,
        )
        self.public_batched_skinning = self._make_public_batched_skinning()
        # _t_pose_orient / _t_pose_orient_parent_T are plain attributes (not buffers).
        # Recompute them on the new device.
        if self.t_pose_world is not None:
            self._t_pose_orient, self._t_pose_orient_parent_T = precompute_joint_orient(
                self.t_pose_world, self.joint_parent_ids
            )
        return self

    def _pad_poses(self, poses_rot: torch.Tensor) -> torch.Tensor:
        ident = (
            torch.eye(3, device=poses_rot.device).unsqueeze(0).repeat(poses_rot.shape[0], 1, 1, 1)
        )
        poses_rot = torch.cat([ident, poses_rot], dim=1)
        return poses_rot

    def _apply_joint_orient(self, poses_rot_relative: torch.Tensor) -> torch.Tensor:
        """Convert relative-to-T-pose rotations to absolute skinning space.
        Matches BatchedSkinning.pose() when joint_orient is set."""
        if self._t_pose_orient is None:
            return poses_rot_relative
        return apply_joint_orient_local(
            poses_rot_relative, self._t_pose_orient, self._t_pose_orient_parent_T
        )

    def _pin_virtual_root_to_origin(self, transforms: torch.Tensor) -> torch.Tensor:
        """Keep the dummy Root transform from carrying identity-fit offsets."""
        if transforms.shape[1] == 0 or self.root_joint_idx != 1:
            return transforms
        root_eye = torch.eye(4, dtype=transforms.dtype, device=transforms.device).view(1, 1, 4, 4)
        mask = torch.zeros(
            1,
            transforms.shape[1],
            1,
            1,
            dtype=transforms.dtype,
            device=transforms.device,
        )
        mask[:, 0] = 1
        return transforms * (1 - mask) + root_eye * mask

    def _expand_public_bind_transforms(self, public_bind_transforms: torch.Tensor) -> torch.Tensor:
        """Expand identity-fitted public bind transforms to the full target topology."""
        if self.procedural_transforms is None:
            return public_bind_transforms
        target_bind = (
            self.bind_pose_world.unsqueeze(0)
            .expand(
                public_bind_transforms.shape[0],
                -1,
                -1,
                -1,
            )
            .clone()
        )
        target_bind[:, self.public_transform_joint_indices] = public_bind_transforms
        return self.procedural_transforms(target_world_transforms=target_bind).transforms

    def _repose_public_bind_pose(
        self,
        public_bind_transforms: torch.Tensor,
        rest_shape: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repose through the public rig before expanding procedural twist helpers."""
        if self.public_batched_skinning is None:
            raise RuntimeError("Public bind-pose reposing requires procedural transforms.")
        public_indices = self.public_transform_joint_indices.to(self.bind_pose_local.device)
        public_bind_pose_local = self.bind_pose_local[public_indices]
        self.public_batched_skinning.rebind(public_bind_transforms, rest_shape)
        return self.public_batched_skinning.pose(
            local_rotations=public_bind_pose_local[..., :3, :3],
            global_translation=public_bind_pose_local[..., self.root_joint_idx, :3, 3],
            align_translation=torch.tensor([0, 0, 0], device=self.device),
            return_transforms=True,
            absolute_pose=True,
        )

    @classmethod
    def _is_body_bone_scale_joint(cls, name: str) -> bool:
        return name in cls.BODY_BONE_SCALE_JOINT_NAMES or any(
            name.startswith(prefix) for prefix in cls.FINGER_BONE_SCALE_JOINT_PREFIXES
        )

    def _normalize_soma_bone_scales(
        self,
        bone_scales: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.identity_model_type != "soma" or bone_scales is None:
            return bone_scales
        expected = self.num_scale_params
        legacy = self.LEGACY_NUM_BONE_SCALE_PARAMS
        if bone_scales.ndim != 2 or bone_scales.shape[1] not in (expected, legacy):
            raise ValueError(
                "SOMA scale_params must have shape "
                f"(B, {expected}) or legacy shape (B, {legacy}); "
                f"got {tuple(bone_scales.shape)}. "
                "Use layer.scale_param_names for the active control order."
            )
        if bone_scales.shape[1] == legacy:
            appended_scales = bone_scales.new_ones(
                bone_scales.shape[0],
                expected - legacy,
            )
            return torch.cat((bone_scales, appended_scales), dim=1)
        return bone_scales

    def _pose_batch_bone_scales(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.identity_model_type != "soma" or self._cached_scale_params is None:
            return None
        bone_scales = self._normalize_soma_bone_scales(self._cached_scale_params)
        assert bone_scales is not None
        bone_scales = bone_scales.to(dtype=dtype, device=device)
        if bone_scales.shape[0] == 1 and batch_size > 1:
            return bone_scales.expand(batch_size, -1)
        if bone_scales.shape[0] != batch_size:
            raise ValueError(
                "SOMA scale_params batch must be 1 or match the effective pose batch; "
                f"got {bone_scales.shape[0]} and {batch_size}"
            )
        return bone_scales

    def _base_local_translations_for_scale(
        self,
        base_translations: torch.Tensor,
        batch_size: int,
        joint_count: int,
    ) -> torch.Tensor:
        if base_translations.ndim == 2:
            base_translations = base_translations.unsqueeze(0)
        if base_translations.shape[-2:] != (joint_count, 3):
            raise ValueError(
                f"Expected local translations for {joint_count} joints, "
                f"got {tuple(base_translations.shape)}"
            )
        if base_translations.shape[0] == 1 and batch_size > 1:
            return base_translations.expand(batch_size, -1, -1)
        if base_translations.shape[0] != batch_size:
            raise ValueError(
                "Local translation batch must be 1 or match the effective pose batch; "
                f"got {base_translations.shape[0]} and {batch_size}"
            )
        return base_translations

    def _full_public_bone_scales(self, bone_scales: torch.Tensor) -> torch.Tensor:
        full_scales = torch.ones(
            bone_scales.shape[0],
            len(self._public_joint_names),
            dtype=bone_scales.dtype,
            device=bone_scales.device,
        )
        scale_joint_indices = self.bone_scale_public_joint_indices.to(device=bone_scales.device)
        if scale_joint_indices.numel() > 0:
            full_scales[:, scale_joint_indices] = bone_scales
        return full_scales

    def _apply_public_bone_scales(
        self,
        bone_scales: torch.Tensor,
        base_translations: torch.Tensor,
    ) -> torch.Tensor:
        full_scales = self._full_public_bone_scales(bone_scales)
        base_t = self._base_local_translations_for_scale(
            base_translations,
            bone_scales.shape[0],
            len(self._public_joint_names),
        )
        return base_t * full_scales.unsqueeze(-1)

    def _apply_target_bone_scales(self, bone_scales: torch.Tensor) -> torch.Tensor:
        full_public_scales = self._full_public_bone_scales(bone_scales)
        target_to_public = self.target_to_public_joint_indices.to(device=bone_scales.device)
        target_scales = full_public_scales[:, target_to_public]

        if self.procedural_transforms is not None:
            public_by_name = {str(name): idx for idx, name in enumerate(self._public_joint_names)}
            target_by_name = {
                str(name): idx for idx, name in enumerate(self.rig_data["joint_names"])
            }
            for segment in self.procedural_transforms.segments:
                end_scale = full_public_scales[:, public_by_name[segment.end_joint]]
                for twist_name in segment.twist_joints:
                    target_scales[:, target_by_name[twist_name]] = end_scale

        base_t = self._base_local_translations_for_scale(
            self.batched_skinning.local_translations,
            bone_scales.shape[0],
            len(self.target_joint_names),
        )
        return base_t * target_scales.unsqueeze(-1)

    def prepare_identity(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        repose_to_bind_pose: bool = True,
        global_scale: float | torch.Tensor = 1.0,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Cache rest shape and fitted skeleton for the given identity.

        Call once per identity, then call `pose()` for each new frame.
        This avoids recomputing the identity model and skeleton transfer
        per frame.

        Args:
            identity_coeffs: (B, K) identity coefficients.
            scale_params: backend-dependent per-identity scale tensor
                (SOMA: (B, layer.num_scale_params), active bone-length
                scale ratios ordered by ``layer.scale_param_names`` and
                described by ``layer.scale_param_segments``; legacy SOMA
                (B, 56) tensors are accepted with unit appended Foot/ToeBase
                scales;
                MHR: (B, 68), required; Anny: optional local-change
                adjustments; other backends: unused). See class docstring.
            repose_to_bind_pose: if True, rebind skinning to the bind pose
                after fitting. Keep enabled when `apply_correctives` is used.
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            kwargs: optional dict forwarded to the identity model's
                `get_rest_shape`.
        """
        scale_params = self._normalize_soma_bone_scales(scale_params)
        identity_scale_params = None if self.identity_model_type == "soma" else scale_params
        self._cached_identity_rest_shape = self.identity_model(
            identity_coeffs, identity_scale_params, kwargs=kwargs, global_scale=global_scale
        )
        self._cached_rest_shape = self._cached_identity_rest_shape
        if self.identity_lod_transfer is not None:
            self._cached_rest_shape = self.identity_lod_transfer(self._cached_identity_rest_shape)
        skeleton_rest_shape = self._cached_rest_shape
        skeleton_transfer = self.skeleton_transfer
        if self.lod == "xlo":
            skeleton_rest_shape = self._cached_identity_rest_shape
            skeleton_transfer = self.xlo_skeleton_transfer
        self._cached_bind_transforms_world = self._expand_public_bind_transforms(
            skeleton_transfer.fit(skeleton_rest_shape)
        )
        if repose_to_bind_pose:
            if self.procedural_transforms is not None:
                public_bind_transforms = self.public_bind_transforms_world(
                    self._cached_bind_transforms_world
                )
                self._cached_rest_shape, public_bind_transforms = self._repose_public_bind_pose(
                    public_bind_transforms,
                    self._cached_rest_shape,
                )
                public_bind_transforms = self._pin_virtual_root_to_origin(public_bind_transforms)
                self._cached_bind_transforms_world = self._expand_public_bind_transforms(
                    public_bind_transforms
                )
            else:
                self.batched_skinning.rebind(
                    self._cached_bind_transforms_world,
                    self._cached_rest_shape,
                )
                self._cached_rest_shape, self._cached_bind_transforms_world = (
                    self.batched_skinning.pose(
                        local_rotations=self.bind_pose_local[..., :3, :3],
                        global_translation=self.bind_pose_local[..., 1, :3, 3],
                        align_translation=torch.tensor([0, 0, 0], device=self.device),
                        return_transforms=True,
                        absolute_pose=True,
                    )
                )
                self._cached_bind_transforms_world = self._pin_virtual_root_to_origin(
                    self._cached_bind_transforms_world
                )
        self.batched_skinning.rebind(self._cached_bind_transforms_world, self._cached_rest_shape)
        self._cached_scale_params = scale_params if self.identity_model_type == "soma" else None
        self._cached_global_scale = global_scale

    def pose(
        self,
        poses: torch.Tensor,
        transl: torch.Tensor | None = None,
        pose2rot: bool = True,
        apply_correctives: bool = True,
        absolute_pose: bool = False,
        fk_only: bool = False,
        return_transforms: bool | None = None,
    ) -> SOMAPoseOutput:
        """Pose the cached identity. Call `prepare_identity()` first.

        Args:
            poses: (B, 77, 3) axis-angle, or (B, 77, 3, 3) rot matrices.
                `poses[0]` = Hips rotation (global body rotation);
                remaining entries = joint-local articulation.
            transl: (B, 3) Hips translation in `output_unit`.
                If None, Hips stays at origin.
            pose2rot: convert axis-angle to rot matrices if True.
            apply_correctives: if True, apply pose-dependent corrective offsets.
            absolute_pose: if True, rotations are absolute (not relative to
                T-pose joint orient).
            fk_only: if True, run forward kinematics only and skip LBS.
            return_transforms: **deprecated** -- `"transforms"` is always
                included in the result. Emits `DeprecationWarning` and
                has no effect.

        Returns:
            SOMAPoseOutput (all translations in `output_unit`):

            - `vertices`: (B, V, 3). Omitted if `fk_only=True`.
            - `joints`: (B, 77, 3).
            - `transforms`: (B, 78, 4, 4). Public skeleton (includes Root).
              In procedural mode, internal twist-joint FK/LBS transforms are
              intentionally not returned.
        """
        if return_transforms is not None:
            warnings.warn(
                "return_transforms is deprecated; 'transforms' is always included in the result.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self._cached_rest_shape is None or self._cached_bind_transforms_world is None:
            raise RuntimeError("No cached identity. Call prepare_identity() before pose().")
        rest_shape = self._cached_rest_shape
        rest_bind_transforms_world = self._cached_bind_transforms_world

        batch_size, num_joints = poses.shape[:2]
        expected_pose_joints = len(self._public_joint_names) - 1
        if num_joints != expected_pose_joints:
            raise ValueError(
                f"Expected poses to contain {expected_pose_joints} public SOMA joints; "
                f"got {num_joints}"
            )
        if transl is None:
            transl = torch.zeros(batch_size, 3, device=poses.device)
        if pose2rot:
            poses_rot = batch_rodrigues(poses.view(-1, 3)).view(batch_size, num_joints, 3, 3)
        else:
            poses_rot = poses.view(batch_size, num_joints, 3, 3)

        poses_rot = self._pad_poses(poses_rot)
        public_poses_rot = poses_rot
        public_absolute_rotations = (
            public_poses_rot
            if absolute_pose
            else (
                self.procedural_transforms.apply_source_joint_orient(public_poses_rot)
                if self.procedural_transforms is not None
                else self._apply_joint_orient(public_poses_rot)
            )
        )

        # Correctives are per-vertex offsets to rest_shape -- only needed when LBS runs.
        if apply_correctives and not fk_only:
            if self.procedural_transforms is None:
                raise RuntimeError(
                    "SOMALayer correctives require procedural transforms; construct with "
                    "enable_procedural_transforms=True or pass apply_correctives=False."
                )
            if self.correctives_model is None:
                raise RuntimeError(
                    "apply_correctives=True but no corrective model is loaded. Construct with "
                    "a valid correctives_model_path or pass apply_correctives=False."
                )
            correctives_input = public_absolute_rotations
            out_correctives = self.correctives_model(correctives_input)["out"]
            gs = self._cached_global_scale
            if isinstance(gs, torch.Tensor):
                out_correctives = out_correctives * gs.reshape(-1, 1, 1)
            elif gs != 1.0:
                out_correctives = out_correctives * gs
            if self.identity_lod_transfer is not None:
                out_correctives = (
                    self.identity_lod_transfer(self._cached_identity_rest_shape + out_correctives)
                    - rest_shape
                )
            rest_shape = rest_shape + out_correctives

        if rest_bind_transforms_world.shape[0] == 1 and batch_size > 1:
            bind_transforms = rest_bind_transforms_world.expand(batch_size, -1, -1, -1)
            rest_shape = rest_shape.expand(batch_size, -1, -1)
        else:
            bind_transforms = rest_bind_transforms_world
        self.batched_skinning.rebind(bind_transforms, rest_shape)

        bind_batch = bind_transforms.shape[0]
        effective_batch = bind_batch if batch_size == 1 and bind_batch > 1 else batch_size
        bone_scales = self._pose_batch_bone_scales(
            effective_batch,
            self.batched_skinning.local_translations.dtype,
            self.batched_skinning.local_translations.device,
        )
        target_local_t_override = None
        public_local_t_override = None
        if bone_scales is not None:
            target_local_t_override = self._apply_target_bone_scales(bone_scales)
            if self.procedural_transforms is not None:
                public_local_t_override = self._apply_public_bone_scales(
                    bone_scales,
                    self.batched_skinning.source_local_translations,
                )
            else:
                public_local_t_override = target_local_t_override

        if self.procedural_transforms is not None:
            public_world_transforms = self.batched_skinning.forward_source_kinematics(
                local_rotations=public_absolute_rotations,
                global_translation=transl,
                absolute_pose=True,
                local_translations=public_local_t_override,
            )
            T_world = self.batched_skinning.expand_source_world_transforms(
                source_rotations=public_absolute_rotations,
                source_world_transforms=public_world_transforms,
                transform_expander=(
                    self.procedural_transforms.expand_world_transforms_from_source_fk
                ),
                target_local_translations=target_local_t_override,
            )
            output_transforms = public_world_transforms
        else:
            public_world_transforms = self.batched_skinning.forward_kinematics(
                local_rotations=public_absolute_rotations,
                global_translation=transl,
                absolute_pose=True,
                local_translations=target_local_t_override,
            )
            T_world = public_world_transforms
            output_transforms = T_world[:, self.public_transform_joint_indices]

        joints = output_transforms[..., :3, 3][:, 1:, :]
        if fk_only:
            return SOMAPoseOutput(joints=joints, transforms=output_transforms)

        vertices = self.batched_skinning.linear_blend_skinning(T_world)

        return SOMAPoseOutput(
            vertices=vertices,
            joints=joints,
            transforms=output_transforms,
        )

    def forward(
        self,
        poses: torch.Tensor,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        pose2rot: bool = True,
        apply_correctives: bool = True,
        absolute_pose: bool = False,
        global_scale: float | torch.Tensor = 1.0,
        kwargs: Mapping[str, Any] | None = None,
        return_transforms: bool | None = None,
    ) -> SOMAPoseOutput:
        """Combined prepare_identity + pose (convenience).

        Args:
            poses: (B, 77, 3) axis-angle, or (B, 77, 3, 3) rot matrices.
                `poses[0]` = Hips rotation (global body rotation);
                remaining entries = joint-local articulation.
            identity_coeffs: (B, K) identity coefficients.
            scale_params: backend-dependent per-identity scale tensor
                (SOMA: (B, layer.num_scale_params), active bone-length
                scale ratios ordered by ``layer.scale_param_names`` and
                described by ``layer.scale_param_segments``; legacy SOMA
                (B, 56) tensors are accepted with unit appended Foot/ToeBase
                scales;
                MHR: (B, 68), required; Anny: optional local-change
                adjustments; other backends: unused). See class docstring.
            transl: (B, 3) Hips translation in `output_unit`.
                If None, Hips stays at origin.
            pose2rot: convert axis-angle to rot matrices if True.
            apply_correctives: if True, apply pose-dependent corrective offsets.
            absolute_pose: if True, rotations are absolute (not relative to
                T-pose joint orient).
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            kwargs: optional dict forwarded to the identity model's
                `get_rest_shape`.
            return_transforms: **deprecated** -- `"transforms"` is always
                included in the result. Emits `DeprecationWarning` and
                has no effect.

        Returns:
            SOMAPoseOutput (all translations in `output_unit`):

            - `vertices`: (B, V, 3).
            - `joints`: (B, 77, 3).
            - `transforms`: (B, 78, 4, 4). Public skeleton (includes Root).
              In procedural mode, internal twist-joint FK/LBS transforms are
              intentionally not returned.
        """
        if return_transforms is not None:
            warnings.warn(
                "return_transforms is deprecated; 'transforms' is always included in the result.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.prepare_identity(
            identity_coeffs,
            scale_params,
            repose_to_bind_pose=apply_correctives or self.procedural_transforms is not None,
            global_scale=global_scale,
            kwargs=kwargs,
        )
        return self.pose(
            poses,
            transl=transl,
            pose2rot=pose2rot,
            apply_correctives=apply_correctives,
            absolute_pose=absolute_pose,
        )
