"""Hand-only SOMA layer.

This module defines :class:`SOMAHandLayer`, a hand-only parametric model
operating in wrist-local coordinate space. The hand mesh and its 25 joints
are a **strict subset of the full-body SOMA topology and skeleton** (the
78-joint ``SOMALayer`` skeleton). ``SOMAHand.npz`` stores the hand mapping
(vertex IDs, remapped faces, and hand-joint indices), the SOMA hand identity
PCA, and a bind-relative articulation-pose PCA. Actual rig vertex data and
skinning weights are sliced from ``SOMA_template_rig.usda`` at init time.

The sections below describe the SOMAHand joint layout and the shapes /
semantics of the tensors that ``SOMAHandLayer`` consumes. Shapes use ``B``
for batch size, ``Vh`` for the LOD-dependent hand vertex count, and
``Jh = 25`` for the hand joint count.

LOD vertex counts
-----------------

The posed vertex tensor has shape ``(B, Vh, 3)``, where ``Vh`` depends on
the selected ``lod``:

.. list-table::
   :header-rows: 1
   :widths: 20 30

   * - ``lod``
     - Vertices
   * - ``"mid"``
     - 2,859 per hand
   * - ``"low"``
     - 718 per hand
   * - ``"xlo"``
     - 134 per hand

``low_lod=True`` is the legacy alias for ``lod="low"``.

SOMAHand skeleton (25 joints)
-----------------------------

The hand root is the wrist (index 0) at the origin of the output frame.

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Index
     - Joint(s)
   * - 0
     - Wrist (root)
   * - 1-4
     - Thumb1, Thumb2, Thumb3, ThumbEnd
   * - 5-9
     - Index1, Index2, Index3, Index4, IndexEnd
   * - 10-14
     - Middle1, Middle2, Middle3, Middle4, MiddleEnd
   * - 15-19
     - Ring1, Ring2, Ring3, Ring4, RingEnd
   * - 20-24
     - Pinky1, Pinky2, Pinky3, Pinky4, PinkyEnd

Pose tensor (``poses``)
-----------------------

Shape ``(B, 25, 3)`` axis-angle, or ``(B, 25, 3, 3)`` rotation matrices
when ``pose2rot=False``.

- Joint 0 = global wrist rotation. Since the output frame IS wrist-local,
  this rotates the whole hand in the caller's frame.
- Joints 1-24 = finger articulation, interpreted **relative to the T-pose
  joint orient** by default. Pass ``absolute_pose=True`` to treat them as
  absolute local rotations instead.

Global translation (``global_translation``, optional)
-----------------------------------------------------

Shape ``(B, 3)`` or ``(3,)``. Wrist translation in ``output_unit``.
``None`` keeps the wrist at the origin (pure wrist-local output).

Identity coefficients (``identity_coeffs``)
-------------------------------------------

Shape ``(B, K)``. ``K = num_shape_components`` is backend-dependent,
selected via the ``identity_model_type`` constructor argument.

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - ``identity_model_type``
     - K
     - Source
   * - ``"soma"`` (default)
     - 20
     - PCA in ``SOMAHand.npz``
   * - ``"mano"``
     - 10
     - MANO shape betas
   * - ``"mhr"``
     - 5
     - MHR hand-shape dims (40-44 of 45)

Bone-length scales (``scale_params``, optional)
-----------------------------------------------

Backend-dependent. Pass ``None`` to skip.
SOMA backend bone-length scaling uses the checked-in rig weights directly.

.. list-table::
   :header-rows: 1
   :widths: 10 15 25 50

   * - Backend
     - Shape
     - When applied
     - Semantics
   * - ``soma``
     - ``(B, 24)``
     - ``pose()``
     - Per-joint bone-length scales for joints 1-24. 1.0 = no change.
       Override ``local_translations`` in skinning.
   * - ``mano``
     - unused
     - --
     - Pass ``None``.
   * - ``mhr``
     - ``(B, 26)``
     - ``prepare_identity()``
     - MHR native hand-scale vector (overall hand scale + 25 per-finger
       segment lengths / offsets / null transforms). Baked into the rest
       shape.

Units
-----

Native unit of the SOMAHand rig is **centimeters**
(``NATIVE_UNIT = Unit.CENTIMETERS``). Output unit is configurable via
``output_unit`` in ``__init__`` (default ``Unit.METERS``). All
translational quantities returned by ``pose()`` / ``forward()`` --
vertices, joints, transforms -- are in ``output_unit``. Prepared
identity caches and pose-inversion data are also kept in ``output_unit``.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csc_matrix

from ..correctives_model import (
    _DEFAULT_CORRECTIVES_MODEL_PATH,
    CorrectivesMLP,
    _resolve_correctives_model_path,
)
from ..geometry._warp_init import ensure_warp_initialized
from ..geometry.barycentric_interp import BarycentricInterpolator
from ..geometry.batched_skinning import BatchedSkinning
from ..geometry.lbs import batch_rodrigues
from ..geometry.rig_utils import (
    apply_joint_orient_local,
    joint_world_to_local,
    precompute_joint_orient,
)
from ..geometry.skeleton_transfer import SkeletonTransfer
from ..io import (
    SOMA_TEMPLATE_RIG_FILENAME,
    fan_triangulate,
    load_lod_rigs_from_usd,
    missing_soma_neutral_rig_keys,
)
from ..procedural_transforms import (
    SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME,
    derive_soma_rig_without_procedural_joints,
    load_soma_procedural_transform_definition,
)
from ..units import Unit
from .identity_model import SOMAHandIdentityModel

logger = logging.getLogger(__name__)


class SOMAHandPoseOutput(dict[str, torch.Tensor]):
    """Structured output returned by :obj:`~soma.hand.SOMAHandLayer.pose` and :obj:`~soma.hand.SOMAHandLayer.forward`.

    Behaves like a `dict` for backwards compatibility (`out["vertices"]`)
    while also supporting attribute access (`out.vertices`). `vertices` is
    absent when `fk_only=True`.
    """

    vertices: torch.Tensor
    joints: torch.Tensor
    transforms: torch.Tensor

    def __getattr__(self, name: str) -> torch.Tensor:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


_VALID_HAND_LODS = ("mid", "low", "xlo")


def _resolve_hand_lod(low_lod: bool, lod: str | None) -> str:
    if lod is None:
        return "low" if low_lod else "mid"
    lod = lod.lower()
    if lod not in _VALID_HAND_LODS:
        raise ValueError(f"Unsupported hand LOD {lod!r}; expected one of {_VALID_HAND_LODS}")
    if low_lod and lod != "low":
        raise ValueError("low_lod=True is only compatible with lod='low'")
    return lod


def _dense_skinning_weights(rig_data: Mapping[str, Any]) -> np.ndarray:
    return (
        csc_matrix(
            (
                rig_data["skinning_weights_data"],
                rig_data["skinning_weights_indices"],
                rig_data["skinning_weights_indptr"],
            ),
            shape=rig_data["skinning_weights_shape"],
        )
        .toarray()
        .astype(np.float32)
    )


def _public_joint_names_from_assets(
    rig_data: Mapping[str, Any],
    *,
    core_asset: Path,
    definition_path: Path,
) -> np.ndarray:
    if "joint_names" in rig_data:
        return np.array(rig_data["joint_names"]).copy()
    if definition_path.exists():
        definition = load_soma_procedural_transform_definition(definition_path)
        return np.array(definition.public_joint_names)
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


def _localize_points(points: np.ndarray, wrist_inv: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return (wrist_inv @ np.hstack([points, ones]).T).T[:, :3]


def _remap_faces(faces: np.ndarray, selected_vert_ids: np.ndarray, num_verts: int) -> np.ndarray:
    inverse = np.full((num_verts,), -1, dtype=np.int64)
    inverse[selected_vert_ids] = np.arange(selected_vert_ids.shape[0], dtype=np.int64)
    remapped = inverse[faces]
    keep = (remapped >= 0).all(axis=1)
    return remapped[keep].astype(np.int32)


def _boundary_vertices_from_faces(faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.int32)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(boundary_edges).astype(np.int32)


def _hand_weights(
    rig_data: Mapping[str, Any],
    body_vert_ids: np.ndarray,
    hand_joint_ids: np.ndarray,
    boundary_loop: np.ndarray,
) -> np.ndarray:
    weights = _dense_skinning_weights(rig_data)[body_vert_ids][:, hand_joint_ids].copy()
    if boundary_loop.size:
        weights[boundary_loop] = 0.0
        weights[boundary_loop, 0] = 1.0
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-6] = 1.0
    return weights / row_sums


def _derive_low_hand_lod(
    rig_data: Mapping[str, Any],
    low_rig_data: Mapping[str, Any],
    mid_vert_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid_to_hand_local = np.full((rig_data["bind_shape"].shape[0],), -1, dtype=np.int64)
    mid_to_hand_local[mid_vert_ids] = np.arange(mid_vert_ids.shape[0], dtype=np.int64)

    mid_to_low = rig_data["lod_mid_to_low"]
    keep = mid_to_hand_local[mid_to_low] >= 0
    body_vert_ids = np.flatnonzero(keep).astype(np.int32)
    source_mid_local = mid_to_hand_local[mid_to_low[body_vert_ids]].astype(np.int64)

    body_faces = fan_triangulate(
        low_rig_data["face_vert_indices"],
        low_rig_data["face_vert_counts"],
    )
    faces = _remap_faces(body_faces, body_vert_ids, low_rig_data["bind_shape"].shape[0])
    boundary_loop = _boundary_vertices_from_faces(faces)
    return body_vert_ids, source_mid_local, faces, boundary_loop


def _derive_xlo_hand_lod(
    rig_data: Mapping[str, Any],
    xlo_rig_data: Mapping[str, Any],
    mid_vert_ids: np.ndarray,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid_bind = torch.from_numpy(rig_data["bind_shape"]).float().to(device)
    mid_faces = torch.from_numpy(rig_data["triangles"]).long().to(device)
    xlo_bind = torch.from_numpy(xlo_rig_data["bind_shape"]).float().to(device)
    interp = BarycentricInterpolator(mid_bind, mid_faces, xlo_bind)

    hand_mid_set = set(int(v) for v in mid_vert_ids.tolist())
    source_faces = rig_data["triangles"][interp.face_ids.detach().cpu().numpy()]
    keep = np.array(
        [any(int(v) in hand_mid_set for v in face) for face in source_faces],
        dtype=bool,
    )
    body_vert_ids = np.flatnonzero(keep).astype(np.int32)

    body_faces = fan_triangulate(
        xlo_rig_data["face_vert_indices"],
        xlo_rig_data["face_vert_counts"],
    )
    faces = _remap_faces(body_faces, body_vert_ids, xlo_rig_data["bind_shape"].shape[0])
    boundary_loop = _boundary_vertices_from_faces(faces)
    return body_vert_ids, faces, boundary_loop


# ---------------------------------------------------------------------------
# SOMAHandLayer
# ---------------------------------------------------------------------------


class SOMAHandLayer(nn.Module):
    """Hand-only parametric model operating in wrist-local coordinate space.

    Two-phase API (matching ``SOMALayer``):

    1. ``prepare_identity(identity_coeffs, scale_params=None)`` -- cache
       rest shape + fitted skeleton for an identity.
    2. ``pose(poses, global_translation=None)`` -- apply articulation to
       the cached identity.

    ``forward()`` is a convenience wrapper that calls both.

    See the :mod:`soma.hand` module docstring for the SOMAHand joint
    layout (25 joints, strict subset of the full-body SOMA skeleton), pose
    tensor conventions, per-backend identity dimensions, and
    ``scale_params`` semantics.
    """

    NATIVE_UNIT = Unit.CENTIMETERS

    NUM_BONE_SCALE_PARAMS = 24  # joints 1-24 (SOMA backend scale_params dim)

    def __init__(
        self,
        data_root: str | Path | None = None,
        hand_type: str = "left",
        device: str | torch.device = "cuda",
        identity_model_type: str = "soma",
        mode: str = "warp",
        output_unit: Unit = Unit.METERS,
        identity_model_kwargs: Mapping[str, Any] | None = None,
        lod: str | None = None,
        low_lod: bool = False,
        load_correctives_model: bool | None = None,
        correctives_model_path: str | Path | None = _DEFAULT_CORRECTIVES_MODEL_PATH,
    ) -> None:
        """Build a SOMAHandLayer with the selected identity backend.

        Args:
            data_root: Directory containing ``SOMAHand.npz``,
                ``SOMA_neutral.npz``, and the per-backend model folders.
                If ``None`` or missing, assets are downloaded from
                HuggingFace automatically.
            hand_type: ``"left"`` or ``"right"``.
            device: Torch device for all buffers and intermediate
                tensors (e.g. ``"cuda"``, ``"cpu"``).
            identity_model_type: Identity backend. One of ``"soma"``
                (default, hand PCA from ``SOMAHand.npz``), ``"mano"``,
                or ``"mhr"``. See :mod:`soma.hand` for per-backend
                identity dimensions and ``scale_params`` semantics.
            mode: Skinning backend. ``"warp"`` uses the NVIDIA Warp
                accelerated LBS kernel; other values fall back to the
                dense PyTorch implementation.
            output_unit: Unit for all translational outputs of
                ``pose()`` / ``forward()`` (vertices, joints,
                transforms). Default ``Unit.METERS``.
            identity_model_kwargs: Extra keyword arguments forwarded to
                the identity-model constructor. Used e.g. by MANO to pass
                ``model_path``.
            lod: Hand mesh level of detail: ``"mid"`` (2,859 vertices per
                hand), ``"low"`` (718 vertices per hand), or ``"xlo"``
                (134 vertices per hand). Defaults to ``"mid"``, or
                ``"low"`` when ``low_lod=True``.
            low_lod: Legacy alias for ``lod="low"``.
            load_correctives_model: Deprecated compatibility alias. Use
                ``correctives_model_path=None`` instead of ``False``.
            correctives_model_path: Path to a pose-corrective checkpoint. Defaults
                to ``data_root/correctives_model.pt``. Pass ``None`` to skip
                loading correctives.
        """
        super().__init__()
        if hand_type not in ("left", "right"):
            raise ValueError(f"hand_type must be 'left' or 'right', got '{hand_type}'")
        self.lod = _resolve_hand_lod(low_lod, lod)

        # Pre-initialize Warp in the main process so DataLoader forked workers
        # inherit _initialized=True and skip wp.init() (avoids CUDA error 3 in
        # workers). Matches SOMALayer.__init__.
        ensure_warp_initialized()

        if data_root is None or not Path(data_root).exists():
            if data_root is not None:
                logger.info("data_root '%s' not found, downloading assets...", data_root)
            from ..assets import get_assets_dir

            data_root = get_assets_dir()

        data_root = Path(data_root)
        self.correctives_model_path = _resolve_correctives_model_path(
            data_root=data_root,
            correctives_model_path=correctives_model_path,
            load_correctives_model=load_correctives_model,
        )
        self._unit_conversion = self.NATIVE_UNIT.meters_per_unit / output_unit.meters_per_unit

        # -- Load mapping from SOMAHand.npz ---------------------------------
        hand_asset = data_root / "SOMAHand.npz"
        if not hand_asset.exists():
            raise FileNotFoundError(f"Hand asset not found: {hand_asset}\n")
        _map = np.load(hand_asset, allow_pickle=False)
        p = f"{hand_type}_"

        mid_vert_ids = _map[f"{p}vert_ids"]  # (Vh,) int
        hand_joint_ids = _map[f"{p}hand_joint_ids_global"]  # (25,) int
        hand_parent_ids = _map[f"{p}joint_parent_ids"].tolist()  # list[int]
        mid_boundary_loop = _map[f"{p}boundary_loop"]  # (Nb,) int

        self.hand_type = hand_type
        self.identity_model_type = identity_model_type
        self.identity_model_kwargs = dict(identity_model_kwargs or {})
        self.mode = mode
        self.device = device
        self.output_unit = output_unit
        self.data_root = data_root
        self.low_lod = self.lod == "low"
        self.nv_lod_mid_to_low = None
        self.excluded_vert_ids = None
        self.root_joint_idx = 0  # wrist is the root (no virtual root)
        self.hand_joint_ids_global = hand_joint_ids.tolist()
        self.wrist_global_id = int(_map[f"{p}wrist_global_id"])

        # -- Load core data, then merge rig tensors from SOMA_template_rig.usda.
        core_asset = data_root / "SOMA_neutral.npz"
        if not core_asset.exists():
            raise FileNotFoundError(
                f"Core asset not found: {core_asset}\n"
                "Run 'git lfs pull' to fetch LFS-tracked files."
            )
        _rig = dict(np.load(core_asset, allow_pickle=False))
        definition_path = data_root / SOMA_PROCEDURAL_TRANSFORM_DEFINITION_FILENAME
        public_joint_names = _public_joint_names_from_assets(
            _rig,
            core_asset=core_asset,
            definition_path=definition_path,
        )
        usd_rig = data_root / SOMA_TEMPLATE_RIG_FILENAME
        template_lod_rigs = {}
        if usd_rig.exists():
            if self.lod == "mid":
                template_lods = ("mid",)
            elif self.lod == "low":
                template_lods = ("mid", "low")
            else:
                template_lods = ("mid", "low", "xlo")
            template_lod_rigs = load_lod_rigs_from_usd(usd_rig, template_lods)
            template_mid_rig_data = template_lod_rigs["mid"]
            _rig.update(
                derive_soma_rig_without_procedural_joints(
                    template_mid_rig_data,
                    public_joint_names,
                )
            )
        else:
            _raise_missing_template_for_slim_npz(
                missing_soma_neutral_rig_keys(_rig),
                core_asset=core_asset,
                template_rig_path=usd_rig,
            )

        hj = hand_joint_ids  # (25,) global joint indices

        bind_pose = _rig["bind_pose_world"].astype(np.float64)  # (78,4,4) cm
        t_pose = _rig["t_pose_world"].astype(np.float64)  # (78,4,4) cm

        # -- Slice to hand verts/joints -------------------------------------
        # Transform from body-world to hand-world (wrist at origin). Because
        # the hand-only model has the wrist as root, wrist-local IS this
        # model's world frame -- hence the _world suffix below.
        wrist_inv = np.linalg.inv(bind_pose[self.wrist_global_id])
        hand_bind_pose_world = wrist_inv[None] @ bind_pose[hj]  # (25, 4, 4) cm
        hand_t_pose_world = wrist_inv[None] @ t_pose[hj]  # (25, 4, 4) cm

        mid_triangles = _map[f"{p}triangles"].astype(np.int32)
        body_lod_vert_ids = mid_vert_ids.astype(np.int32)
        identity_lod_mid_ids = None
        identity_lod_transfer = None
        correctives_vertex_index_map = mid_vert_ids
        correctives_lod_transfer = None
        skeleton_lod_mid_ids = None
        skeleton_bind_shape_world = None
        skeleton_W = None

        if self.lod == "mid":
            hand_faces = mid_triangles
            boundary_loop = mid_boundary_loop.astype(np.int32)
            hand_bind_shape_world = _localize_points(
                _rig["bind_shape"][body_lod_vert_ids].astype(np.float64),
                wrist_inv,
            )
            hand_W = _hand_weights(_rig, body_lod_vert_ids, hj, boundary_loop)
        else:
            if not usd_rig.exists():
                raise FileNotFoundError(
                    f"Hand LOD '{self.lod}' requested, but '{SOMA_TEMPLATE_RIG_FILENAME}' "
                    f"was not found in '{data_root}'."
                )
            low_rig_data = derive_soma_rig_without_procedural_joints(
                template_lod_rigs["low"],
                public_joint_names,
            )
            low_body_vert_ids, low_source_mid_local, low_faces, low_boundary_loop = (
                _derive_low_hand_lod(_rig, low_rig_data, mid_vert_ids)
            )
            if self.lod == "low":
                body_lod_vert_ids = low_body_vert_ids
                identity_lod_mid_ids = low_source_mid_local
                correctives_vertex_index_map = mid_vert_ids[low_source_mid_local]
                hand_faces = low_faces
                boundary_loop = low_boundary_loop
                hand_bind_shape_world = _localize_points(
                    low_rig_data["bind_shape"][body_lod_vert_ids].astype(np.float64),
                    wrist_inv,
                )
                hand_W = _hand_weights(low_rig_data, body_lod_vert_ids, hj, boundary_loop)
            else:
                xlo_rig_data = derive_soma_rig_without_procedural_joints(
                    template_lod_rigs["xlo"],
                    public_joint_names,
                )
                body_lod_vert_ids, hand_faces, boundary_loop = _derive_xlo_hand_lod(
                    _rig,
                    xlo_rig_data,
                    mid_vert_ids,
                    device,
                )
                hand_bind_shape_world = _localize_points(
                    xlo_rig_data["bind_shape"][body_lod_vert_ids].astype(np.float64),
                    wrist_inv,
                )
                hand_W = _hand_weights(xlo_rig_data, body_lod_vert_ids, hj, boundary_loop)

                skeleton_lod_mid_ids = low_source_mid_local
                skeleton_bind_shape_world = _localize_points(
                    low_rig_data["bind_shape"][low_body_vert_ids].astype(np.float64),
                    wrist_inv,
                )
                skeleton_W = _hand_weights(
                    low_rig_data,
                    low_body_vert_ids,
                    hj,
                    low_boundary_loop,
                )
                correctives_vertex_index_map = mid_vert_ids[low_source_mid_local]
                correctives_lod_transfer = BarycentricInterpolator(
                    torch.from_numpy(skeleton_bind_shape_world).float().to(device),
                    torch.from_numpy(low_faces).long().to(device),
                    torch.from_numpy(hand_bind_shape_world).float().to(device),
                )

        if self.lod == "xlo":
            identity_lod_transfer = BarycentricInterpolator(
                torch.from_numpy(
                    _localize_points(
                        _rig["bind_shape"][mid_vert_ids].astype(np.float64),
                        wrist_inv,
                    )
                )
                .float()
                .to(device),
                torch.from_numpy(mid_triangles).long().to(device),
                torch.from_numpy(hand_bind_shape_world).float().to(device),
            )

        # -- Convert to torch tensors --------------------------------------
        def to_t(arr, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype, device=device)

        hand_W_t = to_t(hand_W)
        skeleton_W_t = to_t(skeleton_W) if skeleton_W is not None else hand_W_t
        hand_bind_pose_world_t = to_t(hand_bind_pose_world)
        hand_bind_pose_world_t[..., :3, 3] *= self._unit_conversion
        hand_t_pose_world_t = to_t(hand_t_pose_world)
        hand_t_pose_world_t[..., :3, 3] *= self._unit_conversion
        hand_bind_shape_world_t = to_t(hand_bind_shape_world) * self._unit_conversion
        skeleton_bind_shape_world_t = (
            to_t(skeleton_bind_shape_world) * self._unit_conversion
            if skeleton_bind_shape_world is not None
            else hand_bind_shape_world_t
        )

        # SkeletonTransfer (hand-world output_unit)
        self.skeleton_transfer = SkeletonTransfer(
            joint_parent_ids=hand_parent_ids,
            bind_world_transforms=hand_bind_pose_world_t,
            bind_shape=skeleton_bind_shape_world_t,
            skinning_weights=skeleton_W_t,
            rotation_method="auto",
            use_sparse_rbf_matrix=False,
            root_joint_idx=0,
        )
        self.identity_lod_transfer = identity_lod_transfer
        self.correctives_lod_transfer = correctives_lod_transfer

        # BatchedSkinning (hand-world output_unit; rebind() called in prepare_identity)
        self.batched_skinning = BatchedSkinning(
            joint_parent_ids=hand_parent_ids,
            skinning_weights=hand_W_t,
            bind_world_transforms=hand_bind_pose_world_t,
            bind_shapes=hand_bind_shape_world_t,
            joint_orient=hand_t_pose_world_t,
            mode=mode,
            global_translation_joint_idx=0,
        )

        # -- Identity model -------------------------------------------------
        if identity_model_type == "soma":
            self.identity_model = SOMAHandIdentityModel(
                data_root,
                low_lod=False,
                device=device,
                hand_map=_map,
                hand_type=hand_type,
                output_unit=output_unit,
            )
        elif identity_model_type == "mano":
            from .identity_model import MANOHandIdentityModel

            self.identity_model = MANOHandIdentityModel(
                data_root,
                low_lod=False,
                device=device,
                hand_type=hand_type,
                output_unit=output_unit,
                **self.identity_model_kwargs,
            )
            # Exclude Laplacian-blended wrist verts from pose inversion
            self.excluded_vert_ids = self.identity_model.no_correspondence_ids
        elif identity_model_type == "mhr":
            from .identity_model import MHRHandIdentityModel

            self.identity_model = MHRHandIdentityModel(
                data_root,
                low_lod=False,
                device=device,
                hand_type=hand_type,
                output_unit=output_unit,
            )
        else:
            raise ValueError(
                f"Unknown identity_model_type '{identity_model_type}'. "
                "Supported: 'soma', 'mano', 'mhr'"
            )

        self.correctives_model = None
        if self.correctives_model_path is not None:
            self.correctives_model = CorrectivesMLP.load_checkpoint(
                self.correctives_model_path,
                map_location=device,
                v_index_map=correctives_vertex_index_map,
                joint_indices=hand_joint_ids,
                output_unit=output_unit,
            )

        # -- Topology / provenance / rig buffers ----------------------------
        self.register_buffer(
            "hand_vert_ids", to_t(body_lod_vert_ids, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "hand_mid_vert_ids", to_t(mid_vert_ids, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "identity_lod_mid_ids",
            to_t(identity_lod_mid_ids, dtype=torch.long)
            if identity_lod_mid_ids is not None
            else None,
            persistent=False,
        )
        self.register_buffer(
            "xlo_skeleton_mid_to_low",
            to_t(skeleton_lod_mid_ids, dtype=torch.long)
            if skeleton_lod_mid_ids is not None
            else None,
            persistent=False,
        )
        self.register_buffer("faces", to_t(hand_faces, dtype=torch.long), persistent=False)
        self.register_buffer(
            "joint_parent_ids",
            torch.tensor(hand_parent_ids, dtype=torch.long, device=device),
            persistent=False,
        )
        # For the hand-only model, wrist is the root of its own world frame:
        # the wrist-relative transforms we computed above ARE the world-frame
        # transforms. Parent-relative ("*_local") variants are derived here so
        # prepare_identity() can match SOMALayer's repose_to_bind_pose flow.
        self.register_buffer("bind_pose_world", hand_bind_pose_world_t, persistent=False)
        self.register_buffer(
            "bind_pose_local",
            joint_world_to_local(hand_bind_pose_world_t, hand_parent_ids),
            persistent=False,
        )
        self.register_buffer("t_pose_world", hand_t_pose_world_t, persistent=False)
        self.register_buffer(
            "t_pose_local",
            joint_world_to_local(hand_t_pose_world_t, hand_parent_ids),
            persistent=False,
        )
        self.register_buffer("bind_shape", hand_bind_shape_world_t, persistent=False)
        self.register_buffer("skinning_weights", hand_W_t, persistent=False)
        self.register_buffer(
            "_correctives_to_hand_frame",
            to_t(wrist_inv[:3, :3]),
            persistent=False,
        )
        self._t_pose_orient, self._t_pose_orient_parent_T = precompute_joint_orient(
            self.t_pose_world,
            self.joint_parent_ids,
        )

        # Joint names for PoseInversion compatibility
        hand_joint_names = [str(_rig["joint_names"][gid]) for gid in hand_joint_ids]
        self.rig_data = {"joint_names": hand_joint_names}

        self._cached_identity_rest_shape = None
        self._cached_correctives_rest_shape = None
        self._cached_rest_shape = None
        self._cached_bind_transforms_world = None
        self._cached_scale_params = None
        self._cached_global_scale = 1.0
        self._identity_prepared = False

    @property
    def default_skin_mesh_name(self) -> str:
        """Default USD skin-mesh prim name for this hand's topology.

        Consumed by :obj:`~soma.io.export_soma_usd` when the caller does
        not pass an explicit `skin_mesh_name`. Encodes handedness so
        left/right exports don't collide when written into the same stage.
        """
        side = "l" if self.hand_type == "left" else "r"
        suffix = {"mid": "mid", "low": "lo", "xlo": "xlo"}[self.lod]
        return f"{side}_hand_{suffix}"

    def _apply(self, fn) -> "SOMAHandLayer":
        super()._apply(fn)
        self.device = self.bind_pose_world.device
        self.dtype = self.bind_pose_world.dtype
        # BatchedSkinning is the only non-nn.Module in the graph; its derived
        # state (warp bone indices, skeleton levels, joint-orient conjugates)
        # isn't migrated by super()._apply. Rebuild it from the already-migrated
        # buffers -- matches SOMALayer._apply.
        self.batched_skinning = BatchedSkinning(
            joint_parent_ids=self.joint_parent_ids,
            skinning_weights=self.skinning_weights,
            bind_world_transforms=self.bind_pose_world,
            bind_shapes=self.bind_shape,
            joint_orient=self.t_pose_world,
            mode=self.mode,
            global_translation_joint_idx=0,
        )
        self._t_pose_orient, self._t_pose_orient_parent_T = precompute_joint_orient(
            self.t_pose_world,
            self.joint_parent_ids,
        )
        return self

    @property
    def num_shape_components(self) -> int:
        """Number of identity coefficients."""
        return self.identity_model.num_identity_coeffs

    def _apply_lod_transfer(self, mid_rest_shape: torch.Tensor) -> torch.Tensor:
        if self.identity_lod_mid_ids is not None:
            return mid_rest_shape[:, self.identity_lod_mid_ids, :]
        if self.identity_lod_transfer is not None:
            return self.identity_lod_transfer(mid_rest_shape)
        return mid_rest_shape

    def get_rest_shape(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        global_scale: float | torch.Tensor = 1.0,
        kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Compute hand rest shape from identity coefficients.

        Args:
            identity_coeffs: (B, K) identity coefficients.
            scale_params: backend-dependent per-identity scale vector
                (SOMA: (B, 24); MHR: (B, 26); MANO: unused). See class docstring.
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            kwargs: optional dict forwarded to the identity model's
                `get_rest_shape`.

        Returns:
            (B, Vh, 3) wrist-local rest shape in output_unit.
        """
        mid_rest_shape = self.identity_model(
            identity_coeffs,
            scale_params=scale_params,
            kwargs=kwargs,
            global_scale=global_scale,
        )
        return self._apply_lod_transfer(mid_rest_shape)

    def prepare_identity(
        self,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor | None = None,
        repose_to_bind_pose: bool = True,
        global_scale: float | torch.Tensor = 1.0,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Cache rest shape and fitted skeleton for the given identity.

        Args:
            identity_coeffs: (B, K) identity coefficients.
            scale_params: backend-dependent per-identity scale vector
                (SOMA: (B, 24); MHR: (B, 26); MANO: unused). See class docstring.
                MHR consumes this here; SOMA caches it for `pose()`.
            repose_to_bind_pose: if True, rebind skinning to the bind pose
                after fitting. Keep enabled when `apply_correctives` is used.
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            kwargs: optional dict forwarded to the identity model's
                `get_rest_shape`.
        """
        self._cached_identity_rest_shape = self.identity_model(
            identity_coeffs,
            scale_params=scale_params,
            kwargs=kwargs,
            global_scale=global_scale,
        )
        hand_rest_shape = self._apply_lod_transfer(self._cached_identity_rest_shape)
        skeleton_rest_shape = hand_rest_shape
        self._cached_correctives_rest_shape = hand_rest_shape
        if self.xlo_skeleton_mid_to_low is not None:
            skeleton_rest_shape = self._cached_identity_rest_shape[
                :, self.xlo_skeleton_mid_to_low, :
            ]
            self._cached_correctives_rest_shape = skeleton_rest_shape

        # Fit skeleton to rest shape (wrist-local output_unit)
        fitted_transforms = self.skeleton_transfer.fit(skeleton_rest_shape)  # (B, 25, 4, 4)

        # Cache for PoseInversion compatibility
        self._cached_rest_shape = hand_rest_shape
        self._cached_bind_transforms_world = fitted_transforms

        if repose_to_bind_pose:
            self.batched_skinning.rebind(
                self._cached_bind_transforms_world,
                self._cached_rest_shape,
            )
            self._cached_rest_shape, self._cached_bind_transforms_world = (
                self.batched_skinning.pose(
                    local_rotations=self.bind_pose_local[..., :3, :3],
                    global_translation=self.bind_pose_local[..., self.root_joint_idx, :3, 3],
                    return_transforms=True,
                    absolute_pose=True,
                )
            )

        self.batched_skinning.rebind(self._cached_bind_transforms_world, self._cached_rest_shape)
        self._cached_scale_params = scale_params if self.identity_model_type == "soma" else None
        self._cached_global_scale = global_scale
        self._identity_prepared = True

    def _apply_joint_orient(self, poses_rot_relative: torch.Tensor) -> torch.Tensor:
        """Convert T-pose-relative rotations to absolute local skinning rotations."""
        return apply_joint_orient_local(
            poses_rot_relative,
            self._t_pose_orient,
            self._t_pose_orient_parent_T,
        )

    def _apply_bone_scales(self, bone_scales: torch.Tensor) -> torch.Tensor:
        """Compute scaled local translations from bone_scales.

        Args:
            bone_scales: (B, 24) per-joint scale factors for joints 1-24.
                1.0 = no change.  All joints (including end joints) scale.

        Returns:
            (B, 25, 3) scaled local translations.
        """
        B = bone_scales.shape[0]

        # Build full (B, 25) scale vector: joint 0 (wrist) = 1.0
        full_scales = torch.ones(B, 25, device=bone_scales.device, dtype=bone_scales.dtype)
        full_scales[:, 1:] = bone_scales

        base_t = self.batched_skinning.local_translations
        if base_t.ndim == 2:
            base_t = base_t.unsqueeze(0).expand(B, -1, -1)
        return base_t * full_scales.unsqueeze(-1)

    def pose(
        self,
        poses: torch.Tensor,
        pose2rot: bool = True,
        apply_correctives: bool = False,
        absolute_pose: bool = False,
        global_translation: torch.Tensor | None = None,
        fk_only: bool = False,
    ) -> SOMAHandPoseOutput:
        """Pose the cached identity. Call prepare_identity() first.

        For the SOMA backend, `scale_params` cached by `prepare_identity()`
        are applied here as per-joint bone-length scales (override of
        `local_translations`). MHR already baked them into the rest shape.

        Args:
            poses: (B, 25, 3) axis-angle, or (B, 25, 3, 3) rot matrices.
                Joint 0 = global wrist rotation; joints 1-24 = fingers.
            pose2rot: convert axis-angle to rot matrices if True.
            apply_correctives: if True, apply pose-dependent corrective
                offsets from the shared SOMA body correctives checkpoint.
            absolute_pose: if True, rotations are absolute (not relative to
                T-pose joint orient). Matches SOMALayer convention.
            global_translation: (B, 3) or (3,) wrist translation in
                output_unit. If None, wrist stays at origin.
            fk_only: if True, run forward kinematics only and skip LBS.

        Returns:
            SOMAHandPoseOutput (all translations in `output_unit`):

            - `vertices`: (B, Vh, 3). Omitted if `fk_only=True`.
            - `joints`: (B, 25, 3).
            - `transforms`: (B, 25, 4, 4).
        """
        if self._cached_rest_shape is None or self._cached_bind_transforms_world is None:
            raise RuntimeError("No cached identity. Call prepare_identity() before pose().")

        B = poses.shape[0]

        if pose2rot:
            poses_rot = batch_rodrigues(poses.reshape(-1, 3)).reshape(B, 25, 3, 3)
        else:
            poses_rot = poses.reshape(B, 25, 3, 3)

        if global_translation is not None:
            global_translation = global_translation.to(
                dtype=poses_rot.dtype, device=poses_rot.device
            )
        else:
            global_translation = torch.zeros(B, 3, device=poses_rot.device, dtype=poses_rot.dtype)

        # SOMA backend: scale_params from prepare_identity() are per-joint
        # bone-length scales applied at pose time via local_translations
        # override. MHR's scale_params were consumed at identity time.
        local_t_override = None
        if self._cached_scale_params is not None and isinstance(
            self.identity_model, SOMAHandIdentityModel
        ):
            local_t_override = self._apply_bone_scales(self._cached_scale_params)

        rest_shape = self._cached_rest_shape
        bind_transforms = self._cached_bind_transforms_world
        if apply_correctives and not fk_only:
            if self.correctives_model is None:
                raise RuntimeError(
                    "apply_correctives=True but no corrective model is loaded. Construct with "
                    "a valid correctives_model_path or pass apply_correctives=False."
                )
            correctives_input = poses_rot if absolute_pose else self._apply_joint_orient(poses_rot)
            out_correctives = self.correctives_model(correctives_input)["out"]
            frame_rot = self._correctives_to_hand_frame.to(
                dtype=out_correctives.dtype,
                device=out_correctives.device,
            )
            out_correctives = torch.einsum("ij,bvj->bvi", frame_rot, out_correctives)
            gs = self._cached_global_scale
            if isinstance(gs, torch.Tensor):
                out_correctives = out_correctives * gs.reshape(-1, 1, 1)
            elif gs != 1.0:
                out_correctives = out_correctives * gs
            if self.correctives_lod_transfer is not None:
                if self._cached_correctives_rest_shape is None:
                    raise RuntimeError(
                        "No cached corrective source identity. "
                        "Call prepare_identity() before pose()."
                    )
                out_correctives = (
                    self.correctives_lod_transfer(
                        self._cached_correctives_rest_shape + out_correctives
                    )
                    - rest_shape
                )
            rest_shape = rest_shape + out_correctives

        if fk_only:
            T_world = self.batched_skinning.forward_kinematics(
                local_rotations=poses_rot,
                global_translation=global_translation,
                absolute_pose=absolute_pose,
                local_translations=local_t_override,
            )
            return SOMAHandPoseOutput(
                joints=T_world[..., :3, 3],
                transforms=T_world,
            )

        if bind_transforms.shape[0] == 1 and B > 1:
            bind_transforms = bind_transforms.expand(B, -1, -1, -1)
        if rest_shape.shape[0] == 1 and B > 1:
            rest_shape = rest_shape.expand(B, -1, -1)
        self.batched_skinning.rebind(bind_transforms, rest_shape)

        vertices, T_world = self.batched_skinning.pose(
            local_rotations=poses_rot,
            global_translation=global_translation,
            return_transforms=True,
            absolute_pose=absolute_pose,
            local_translations=local_t_override,
        )

        return SOMAHandPoseOutput(
            vertices=vertices,
            joints=T_world[..., :3, 3],
            transforms=T_world,
        )

    def forward(
        self,
        poses: torch.Tensor,
        identity_coeffs: torch.Tensor,
        pose2rot: bool = True,
        apply_correctives: bool = False,
        absolute_pose: bool = False,
        global_translation: torch.Tensor | None = None,
        global_scale: float | torch.Tensor = 1.0,
        scale_params: torch.Tensor | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> SOMAHandPoseOutput:
        """Combined prepare_identity + pose (convenience).

        Args:
            poses: (B, 25, 3) axis-angle, or (B, 25, 3, 3) rot matrices.
                Joint 0 = global wrist rotation; joints 1-24 = fingers.
            identity_coeffs: (B, K) identity coefficients.
            pose2rot: convert axis-angle to rot matrices if True.
            apply_correctives: if True, apply pose-dependent corrective offsets.
            absolute_pose: if True, rotations are absolute (not relative to
                T-pose joint orient). Matches SOMALayer convention.
            global_translation: (B, 3) or (3,) wrist translation in
                output_unit. If None, wrist stays at origin.
            global_scale: uniform scale scalar or (B,) tensor. Default 1.0.
            scale_params: backend-dependent per-identity scale vector
                (SOMA: (B, 24); MHR: (B, 26); MANO: unused). See class docstring.
            kwargs: optional dict forwarded to the identity model's
                `get_rest_shape`.

        Returns:
            SOMAHandPoseOutput (all translations in `output_unit`):

            - `vertices`: (B, Vh, 3).
            - `joints`: (B, 25, 3).
            - `transforms`: (B, 25, 4, 4).
        """
        self.prepare_identity(
            identity_coeffs,
            scale_params=scale_params,
            repose_to_bind_pose=apply_correctives,
            global_scale=global_scale,
            kwargs=kwargs,
        )
        return self.pose(
            poses,
            pose2rot=pose2rot,
            apply_correctives=apply_correctives,
            absolute_pose=absolute_pose,
            global_translation=global_translation,
        )
