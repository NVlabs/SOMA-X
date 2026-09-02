# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""I/O helpers for SOMA: NPZ animation files and USD mesh/skeleton I/O.

An .npz file contains everything needed to replay an animation with
SOMALayer: identity model type, identity coefficients, poses, root
translation, and metadata describing the representation.
"""

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom

from .units import Unit

logger = logging.getLogger(__name__)


class UVPrimvarEntry(dict[str, Any]):
    """Serialized UV primvar data returned by :obj:`~soma.io.load_usd_mesh`.

    Behaves like a ``dict`` for backwards compatibility
    (``entry["coordinates"]``) while also supporting attribute access
    (``entry.coordinates``).
    """

    coordinates: np.ndarray
    indices: np.ndarray | None
    interpolation: str

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class SOMANPZData(dict[str, Any]):
    """Dictionary returned by :obj:`~soma.io.load_soma_npz`.

    Behaves like a ``dict`` for backwards compatibility
    (``data["poses"]``) while also supporting attribute access
    (``data.poses``). Optional fields (``scale_params``, ``joint_orient``,
    ``global_scale``, ``hand_type``) are present only if they were saved.
    """

    poses: np.ndarray
    transl: np.ndarray
    joint_names: list
    identity_model_type: str
    identity_coeffs: np.ndarray
    rotation_repr: str
    absolute_pose: bool
    unit: str
    keep_root: bool
    scale_params: np.ndarray
    joint_orient: np.ndarray
    global_scale: float
    hand_type: str

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class RigUSDData(dict[str, Any]):
    """Dictionary returned by :obj:`~soma.io.load_rig_from_usd`.

    Behaves like a ``dict`` for backwards compatibility
    (``rig["joint_names"]``) while also supporting attribute access
    (``rig.joint_names``). Mesh-related fields (``face_vert_indices``,
    ``face_vert_counts``, ``uv_data``) are present only when the body skin
    mesh carries polygon/UV data.
    """

    joint_names: np.ndarray
    joint_parent_ids: np.ndarray
    bind_pose_world: np.ndarray
    bind_pose_local: np.ndarray
    t_pose_world: np.ndarray
    t_pose_local: np.ndarray
    bind_shape: np.ndarray
    skinning_weights_data: np.ndarray
    skinning_weights_indices: np.ndarray
    skinning_weights_indptr: np.ndarray
    skinning_weights_shape: np.ndarray
    face_vert_indices: np.ndarray
    face_vert_counts: np.ndarray
    uv_data: dict[str, UVPrimvarEntry]

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ---------------------------------------------------------------------------
# USD mesh I/O
# ---------------------------------------------------------------------------


def list_usd_meshes(usd_file_path: str | Path) -> list[str]:
    """Return prim paths of all Mesh prims in a USD file.

    Args:
        usd_file_path: Path to the USD file.

    Returns:
        list of prim path strings.
    """
    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")
    return [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]


def _is_uv_primvar(pv):
    attr = pv.GetAttr()
    type_name = attr.GetTypeName()
    if type_name not in (
        Sdf.ValueTypeNames.TexCoord2fArray,
        Sdf.ValueTypeNames.TexCoord2f,
        Sdf.ValueTypeNames.Float2Array,
        Sdf.ValueTypeNames.Float2,
    ):
        return False
    return pv.GetInterpolation() in ("vertex", "varying", "faceVarying")


def _read_uv_primvars(mesh) -> dict[str, UVPrimvarEntry]:
    """Return UV primvars from *mesh* as ``{name: {coordinates, indices, interpolation}}``."""
    uv_data = {}
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if not _is_uv_primvar(pv):
            continue
        coords = pv.GetAttr().Get()
        if not coords or len(coords) == 0:
            continue
        uvs = np.array(coords, dtype=np.float32)
        if uvs.ndim == 1:
            if uvs.size % 2 != 0:
                continue
            uvs = uvs.reshape(-1, 2)
        uv_indices = None
        if pv.IsIndexed():
            idx = pv.GetIndicesAttr().Get()
            if idx:
                uv_indices = np.array(idx, dtype=np.int32)
        uv_data[pv.GetPrimvarName()] = UVPrimvarEntry(
            coordinates=uvs,
            indices=uv_indices,
            interpolation=pv.GetInterpolation(),
        )
    return uv_data


def _write_uv_primvars(primvars, uv_data: Mapping[str, UVPrimvarEntry]) -> None:
    """Write *uv_data* as ``TexCoord2fArray`` primvars onto *primvars*."""
    _interp_tokens = {
        "vertex": UsdGeom.Tokens.vertex,
        "faceVarying": UsdGeom.Tokens.faceVarying,
        "uniform": UsdGeom.Tokens.uniform,
        "constant": UsdGeom.Tokens.constant,
    }
    for name, info in uv_data.items():
        pv = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.TexCoord2fArray)
        pv.Set([Gf.Vec2f(float(u[0]), float(u[1])) for u in info["coordinates"]])
        pv.SetInterpolation(
            _interp_tokens.get(info.get("interpolation", "faceVarying"), UsdGeom.Tokens.faceVarying)
        )
        if info.get("indices") is not None:
            pv.SetIndices(info["indices"].tolist())


def load_usd_mesh(
    usd_file_path: str | Path,
    mesh_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, UVPrimvarEntry]]:
    """Load a mesh from a USD file.

    Args:
        usd_file_path: Path to the USD file.
        mesh_name: Prim path of the mesh (e.g. ``"/Root/Mesh"``).
            A leading ``/`` is added if missing.

    Returns:
        ``(vertices, face_vert_indices, face_vert_counts, uv_data)`` where

        - *vertices*: ``(V, 3)`` float32
        - *face_vert_indices*: ``(sum(counts),)`` int32 — flattened
        - *face_vert_counts*: ``(F,)`` int32 — verts-per-face
        - *uv_data*: ``{name: {"coordinates": (M,2), "indices": ..., "interpolation": str}}``
    """
    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")

    if not mesh_name.startswith("/"):
        mesh_name = "/" + mesh_name
    prim = stage.GetPrimAtPath(mesh_name)
    if not prim:
        avail = list_usd_meshes(usd_file_path)
        raise RuntimeError(
            f"Mesh '{mesh_name}' not found in '{usd_file_path}'. Available meshes: {avail}"
        )

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        raise ValueError(f"Mesh '{mesh_name}' has no points")
    vertices = np.array(pts, dtype=np.float32)
    _fvi = mesh.GetFaceVertexIndicesAttr().Get()
    _fvc = mesh.GetFaceVertexCountsAttr().Get()
    if _fvi is None or _fvc is None:
        raise ValueError(f"Mesh '{mesh_name}' has no face topology")
    fvi = np.array(_fvi, dtype=np.int32)
    fvc = np.array(_fvc, dtype=np.int32)

    uv_data = {}
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if not _is_uv_primvar(pv):
            continue
        coords = pv.GetAttr().Get()
        if not coords or len(coords) == 0:
            continue
        uvs = np.array(coords, dtype=np.float32)
        if uvs.ndim == 1:
            if uvs.size % 2 != 0:
                continue
            uvs = uvs.reshape(-1, 2)
        uv_indices = None
        if pv.IsIndexed():
            idx = pv.GetIndicesAttr().Get()
            if idx:
                uv_indices = np.array(idx, dtype=np.int32)
        uv_data[pv.GetPrimvarName()] = {
            "coordinates": uvs,
            "indices": uv_indices,
            "interpolation": pv.GetInterpolation(),
        }

    return vertices, fvi, fvc, uv_data


def write_usd_mesh(
    usd_file_path: str | Path,
    mesh_name: str,
    vertices: np.ndarray,
    face_vert_indices: np.ndarray,
    face_vert_counts: np.ndarray,
    uv_data: dict | None = None,
) -> None:
    """Write a mesh to a new USD file.

    Args:
        usd_file_path: Output path.
        mesh_name: Prim path for the mesh.
        vertices: ``(V, 3)`` vertex positions.
        face_vert_indices: Flattened face-vertex indices.
        face_vert_counts: Verts-per-face array.
        uv_data: Optional UV sets (same format as :obj:`~soma.io.load_usd_mesh` output).
    """
    stage = Usd.Stage.CreateNew(str(usd_file_path))
    if not mesh_name.startswith("/"):
        mesh_name = "/" + mesh_name

    mesh = UsdGeom.Mesh(stage.DefinePrim(mesh_name, "Mesh"))
    mesh.CreatePointsAttr().Set([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices])
    mesh.CreateFaceVertexIndicesAttr().Set(face_vert_indices.tolist())
    mesh.CreateFaceVertexCountsAttr().Set(face_vert_counts.tolist())

    if uv_data:
        _interp_tokens = {
            "vertex": UsdGeom.Tokens.vertex,
            "faceVarying": UsdGeom.Tokens.faceVarying,
            "uniform": UsdGeom.Tokens.uniform,
            "constant": UsdGeom.Tokens.constant,
        }
        for name, info in uv_data.items():
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(name, Sdf.ValueTypeNames.TexCoord2fArray)
            pv.Set([Gf.Vec2f(float(u[0]), float(u[1])) for u in info["coordinates"]])
            pv.SetInterpolation(
                _interp_tokens.get(
                    info.get("interpolation", "faceVarying"), UsdGeom.Tokens.faceVarying
                )
            )
            if info.get("indices") is not None:
                pv.SetIndices(info["indices"].tolist())

    stage.GetRootLayer().Save()


def fan_triangulate(
    face_vert_indices: np.ndarray,
    face_vert_counts: np.ndarray,
) -> np.ndarray:
    """Convert a polygon soup to triangles via fan triangulation.

    Args:
        face_vert_indices: Flattened face-vertex indices.
        face_vert_counts: Verts-per-face array.

    Returns:
        ``(F_tri, 3)`` int32 triangle array.
    """
    triangles = []
    offset = 0
    for count in face_vert_counts:
        v0 = face_vert_indices[offset]
        for j in range(1, count - 1):
            triangles.append([v0, face_vert_indices[offset + j], face_vert_indices[offset + j + 1]])
        offset += count
    return np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)


# ---------------------------------------------------------------------------
# USD skeleton I/O
# ---------------------------------------------------------------------------


def load_usd_skeleton(
    usd_file_path: str | Path,
) -> tuple[list[str], np.ndarray, list[int]]:
    """Extract skeleton from a USD file.

    Args:
        usd_file_path: Path to the USD file.

    Returns:
        Tuple ``(joint_paths, bind_transforms, parent_ids)``.
        ``joint_paths`` is the list of J joint path strings (for example
        ``"Root/Hips"``), ``bind_transforms`` is a ``(J, 4, 4)`` float32 array
        of world-space bind transforms in USD row-major convention
        (``point * M``), and ``parent_ids`` contains the J parent indices
        with ``-1`` for the root.
    """
    from pxr import UsdSkel

    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")

    skel_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No Skeleton prim found in '{usd_file_path}'")

    skel = UsdSkel.Skeleton(skel_prim)
    joint_paths = list(skel.GetJointsAttr().Get())
    J = len(joint_paths)

    bind_xforms = skel.GetBindTransformsAttr().Get()
    if bind_xforms is None:
        raise RuntimeError(f"No bindTransforms in '{usd_file_path}'")
    bind_transforms = np.array(bind_xforms, dtype=np.float32).reshape(J, 4, 4)

    path_to_idx = {j: i for i, j in enumerate(joint_paths)}
    parent_ids = [path_to_idx.get(j.rsplit("/", 1)[0] if "/" in j else "", -1) for j in joint_paths]

    return joint_paths, bind_transforms, parent_ids


def load_usd_animation(
    usd_file_path: str | Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract SkelAnimation rotations and translations from a USD file.

    Args:
        usd_file_path: Path to the USD file.

    Returns:
        ``(rot_mats, translations)`` where

        - *rot_mats*: ``(J, 3, 3)`` float32 — local rotation matrices
        - *translations*: ``(J, 3)`` float32 — local translations

        Returns ``None`` if no SkelAnimation is found.
    """
    from pxr import UsdSkel

    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")

    anim_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Animation)), None)
    if anim_prim is None:
        return None

    anim = UsdSkel.Animation(anim_prim)
    quats = anim.GetRotationsAttr().Get()
    trans = anim.GetTranslationsAttr().Get()
    if quats is None:
        return None

    from scipy.spatial.transform import Rotation

    J = len(quats)
    # USD Quatf: real=w, imaginary=(x,y,z); scipy wants (x,y,z,w)
    quats_xyzw = np.array(
        [
            [q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()]
            for q in quats
        ],
        dtype=np.float64,
    )
    rot_mats = Rotation.from_quat(quats_xyzw).as_matrix().astype(np.float32)

    translations = np.zeros((J, 3), dtype=np.float32)
    if trans is not None:
        translations = np.array([[t[0], t[1], t[2]] for t in trans], dtype=np.float32)

    return rot_mats, translations


def load_usd_skinning(
    usd_file_path: str | Path,
    mesh_prim_path: str | None = None,
) -> tuple[np.ndarray, int]:
    """Extract skinning weights from a skinned mesh in a USD file.

    The mesh binding may define its own joint order (a subset of the skeleton
    joints).  The returned weight matrix is indexed by the **skeleton** joint
    order so it aligns with :obj:`~soma.io.load_usd_skeleton`.

    Args:
        usd_file_path: Path to the USD file.
        mesh_prim_path: Prim path of the mesh to read skinning from.  If
            ``None``, the first Mesh prim in the file is used.

    Returns:
        ``(skinning_weights, num_joints)`` where

        - *skinning_weights*: ``(V, J_skel)`` float32 dense weight matrix
        - *num_joints*: number of skeleton joints
    """
    from pxr import UsdSkel

    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")

    if mesh_prim_path is not None:
        if not mesh_prim_path.startswith("/"):
            mesh_prim_path = "/" + mesh_prim_path
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if not mesh_prim:
            raise RuntimeError(f"Mesh '{mesh_prim_path}' not found in '{usd_file_path}'")
    else:
        mesh_prim = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)), None)
        if mesh_prim is None:
            raise RuntimeError(f"No Mesh prim found in '{usd_file_path}'")

    skel_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No Skeleton prim found in '{usd_file_path}'")

    binding = UsdSkel.BindingAPI(mesh_prim)
    ji_pv = binding.GetJointIndicesPrimvar()
    jw_pv = binding.GetJointWeightsPrimvar()
    if not ji_pv or not jw_pv:
        raise RuntimeError(
            f"No skinning primvars (skel:jointIndices / skel:jointWeights) "
            f"found on '{mesh_prim.GetPath()}'"
        )

    _pts = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    if not _pts:
        raise ValueError(f"Mesh '{mesh_prim.GetPath()}' has no points")
    V = len(_pts)
    K = ji_pv.GetElementSize()
    ji = np.array(ji_pv.Get(), dtype=np.int32).reshape(V, K)
    jw = np.array(jw_pv.Get(), dtype=np.float32).reshape(V, K)

    skel = UsdSkel.Skeleton(skel_prim)
    skel_joints = list(skel.GetJointsAttr().Get())
    J = len(skel_joints)
    skel_joint_to_idx = {name: i for i, name in enumerate(skel_joints)}

    # The mesh binding may declare its own joint subset; map to skeleton indices.
    binding_joints = binding.GetJointsAttr().Get()
    if binding_joints and len(binding_joints) > 0:
        binding_to_skel = np.array(
            [skel_joint_to_idx.get(str(j), -1) for j in binding_joints], dtype=np.int32
        )
    else:
        binding_to_skel = np.arange(J, dtype=np.int32)

    v_idx = np.repeat(np.arange(V, dtype=np.int32), K)
    j_idx = binding_to_skel[ji.ravel()]
    w_vals = jw.ravel()
    valid = (w_vals > 0) & (j_idx >= 0)

    W = np.zeros((V, J), dtype=np.float32)
    np.add.at(W, (v_idx[valid], j_idx[valid]), w_vals[valid])
    return W, J


# ---------------------------------------------------------------------------
# SOMA template rig
# ---------------------------------------------------------------------------

# Filename of the SOMA body template rig asset (in data_root).
SOMA_TEMPLATE_RIG_FILENAME = "SOMA_template_rig.usda"
SOMA_XLO_TEMPLATE_RIG_FILENAME = SOMA_TEMPLATE_RIG_FILENAME
SOMA_NEUTRAL_RIG_KEYS = (
    "joint_names",
    "joint_parent_ids",
    "bind_pose_world",
    "bind_pose_local",
    "t_pose_world",
    "t_pose_local",
    "bind_shape",
    "skinning_weights_data",
    "skinning_weights_indices",
    "skinning_weights_indptr",
    "skinning_weights_shape",
)


def missing_soma_neutral_rig_keys(data: Mapping[str, Any]) -> tuple[str, ...]:
    """Return rig keys absent from a loaded ``SOMA_neutral.npz`` mapping."""
    return tuple(key for key in SOMA_NEUTRAL_RIG_KEYS if key not in data)


# Generic fallback mesh prim name for `save_soma_usd` and the name-based
# lookup in `load_rig_from_usd`. Intentionally free of any SOMA naming
# convention (no LOD suffix, no hand/body hint); layers that know their
# topology should expose `default_skin_mesh_name` and let `export_soma_usd`
# forward that into the writer.
# Auto-discovery in `load_rig_from_usd` still works regardless of the name
# chosen at write time.
DEFAULT_SKIN_MESH_NAME = "Mesh"


def _find_mesh_by_name(stage, name):
    """Return the first ``UsdGeomMesh`` prim whose leaf name matches *name*, or None."""
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh) and p.GetPath().name == name:
            return p
    return None


def _find_first_mesh_under_skel_root(stage):
    """Return the first ``UsdGeomMesh`` descendant of any ``UsdSkelRoot``, or None.

    Used as a fallback when the canonical skin-mesh name isn't present.
    """
    from pxr import UsdSkel

    for root_prim in stage.Traverse():
        if not root_prim.IsA(UsdSkel.Root):
            continue
        for desc in Usd.PrimRange(root_prim):
            if desc.IsA(UsdGeom.Mesh):
                return desc
    return None


_LOD_SKIN_MESH_CANDIDATES = {
    "mid": ("c_skin_mid", "c_bodyRig_mid", "c_skin"),
    "low": ("c_skin_lo", "c_bodyRig_lo", "c_skin_low", "c_bodyRig_low"),
    "xlo": (
        "c_skin_xlo",
        "c_bodyRig_xlo",
        "c_skin_extra_low",
        "c_bodyRig_extra_low",
        "c_body_xlo",
        "skin_xlo",
    ),
}

_LOD_NAME_TOKENS = {
    "mid": ("mid",),
    "low": ("_lo", "_low", "low"),
    "xlo": ("xlo", "extra_low", "extralow", "extra-low"),
}


def _mesh_has_skinning(prim) -> bool:
    """Return True when *prim* carries UsdSkel joint-index and weight primvars."""
    from pxr import UsdSkel

    binding = UsdSkel.BindingAPI(prim)
    return bool(binding.GetJointIndicesPrimvar() and binding.GetJointWeightsPrimvar())


def _open_usd_stage(usd_path: str | Path):
    usd_path_str = str(usd_path)
    if not Path(usd_path_str).exists():
        raise FileNotFoundError(f"USD file not found: {usd_path_str}")

    stage = Usd.Stage.Open(usd_path_str)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_path_str}")
    return stage


def _find_lod_skin_mesh_name_on_stage(stage, usd_path: str | Path, lod: str) -> str:
    usd_path_str = str(usd_path)
    lod = lod.lower()
    if lod not in _LOD_SKIN_MESH_CANDIDATES:
        valid_lods = tuple(_LOD_SKIN_MESH_CANDIDATES)
        raise ValueError(f"Unsupported LOD {lod!r}; expected one of {valid_lods}")

    skin_meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and _mesh_has_skinning(p)]
    by_name = {p.GetPath().name: p for p in skin_meshes}
    for name in _LOD_SKIN_MESH_CANDIDATES[lod]:
        if name in by_name:
            return name

    tokens = _LOD_NAME_TOKENS[lod]
    matches = [
        p.GetPath().name for p in skin_meshes if any(t in p.GetPath().name.lower() for t in tokens)
    ]
    if matches:
        matches.sort(
            key=lambda n: (
                0 if "skin" in n.lower() else 1 if "body" in n.lower() else 2,
                len(n),
                n,
            )
        )
        return matches[0]

    available = sorted({p.GetPath().name for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)})
    skinned_available = sorted(by_name)
    raise ValueError(
        f"Could not find a skinned mesh for LOD {lod!r} in '{usd_path_str}'. "
        f"Skinned meshes: {skinned_available}. Available meshes: {available}"
    )


def find_lod_skin_mesh_name(usd_path: str | Path, lod: str) -> str:
    """Find the skinned mesh leaf name for a body LOD in a UsdSkel asset.

    The SOMA templates use naming conventions such as ``c_skin_xlo`` or
    ``c_bodyRig_xlo``.  This helper keeps the runtime tolerant to small naming
    differences while still requiring a skinned mesh, not just any mesh whose
    name contains the LOD token.
    """
    stage = _open_usd_stage(usd_path)
    return _find_lod_skin_mesh_name_on_stage(stage, usd_path, lod)


def load_lod_rig_from_usd(
    usd_path: str | Path,
    lod: str,
    *,
    skin_mesh_name: str | None = None,
) -> RigUSDData:
    """Load rig data for a specific body LOD from a UsdSkel USD file."""
    stage = _open_usd_stage(usd_path)
    if skin_mesh_name is None:
        skin_mesh_name = _find_lod_skin_mesh_name_on_stage(stage, usd_path, lod)
    return _load_rig_from_usd_stage(stage, usd_path, skin_mesh_name=skin_mesh_name)


def load_lod_rigs_from_usd(
    usd_path: str | Path,
    lods: Sequence[str],
    *,
    skin_mesh_names: Mapping[str, str] | None = None,
) -> dict[str, RigUSDData]:
    """Load multiple LOD rigs from one UsdSkel USD file with a shared stage."""
    stage = _open_usd_stage(usd_path)
    skin_mesh_names = skin_mesh_names or {}
    rigs = {}
    for lod in lods:
        lod_key = lod.lower()
        if lod_key in rigs:
            continue
        skin_mesh_name = skin_mesh_names.get(lod_key, skin_mesh_names.get(lod))
        if skin_mesh_name is None:
            skin_mesh_name = _find_lod_skin_mesh_name_on_stage(stage, usd_path, lod_key)
        rigs[lod_key] = _load_rig_from_usd_stage(
            stage,
            usd_path,
            skin_mesh_name=skin_mesh_name,
        )
    return rigs


def load_rig_from_usd(usd_path: str | Path, *, skin_mesh_name: str | None = None) -> RigUSDData:
    """Load SOMA template rig data from a UsdSkel USD file.

    Extracts the joint hierarchy, bind/T-pose transforms, bind-shape vertex
    positions, and skinning weights from the USD skeleton.  Returns the
    rig-related keys that used to be stored in ``SOMA_neutral.npz`` so they
    can be merged with the slim core asset when initializing
    :obj:`~soma.body.SOMALayer`.

    Shape PCA data (``mean``, ``shapedirs``, ``eigenvalues``) and mesh
    topology (``triangles``, ``triangles_low``, LOD maps, facial segments)
    are not included; they must still be loaded from ``SOMA_neutral.npz``.

    Args:
        usd_path: Path to the ``.usda``, ``.usdc``, or ``.usd`` template rig.
        skin_mesh_name: Name of the ``UsdGeomMesh`` prim holding the body
            skin (leaf path component, not full path). If ``None`` (the
            default), look for ``DEFAULT_SKIN_MESH_NAME`` first, and
            fall back to the first ``UsdGeomMesh`` descendant of any
            ``UsdSkelRoot``. Pass an explicit name to pin the search.

    Returns:
        Dictionary with rig keys supplied by ``SOMA_template_rig.usda``:
        ``joint_names``, ``joint_parent_ids``, ``bind_pose_world``,
        ``bind_pose_local``, ``t_pose_world``, ``t_pose_local``,
        ``bind_shape``, ``skinning_weights_data``,
        ``skinning_weights_indices``, ``skinning_weights_indptr``, and
        ``skinning_weights_shape``. When the body skin mesh carries polygon
        or UV data, the result also includes ``face_vert_indices``,
        ``face_vert_counts``, and ``uv_data`` for :obj:`~soma.io.save_soma_usd`.

    Raises:
        FileNotFoundError: if the USD file does not exist.
        RuntimeError: if the file cannot be opened, has no
            ``UsdSkelSkeleton``, has no ``bindTransforms``, has joint/
            transform count mismatches, or has malformed skinning primvars.
        ValueError: if the skin mesh cannot be located under
            *skin_mesh_name* (or the auto-discovery fallback), or has
            no points.
    """
    stage = _open_usd_stage(usd_path)
    return _load_rig_from_usd_stage(stage, usd_path, skin_mesh_name=skin_mesh_name)


def _load_rig_from_usd_stage(
    stage,
    usd_path: str | Path,
    *,
    skin_mesh_name: str | None = None,
) -> RigUSDData:
    """Load SOMA template rig data from an already-open UsdSkel stage."""
    from pxr import UsdSkel
    from scipy.sparse import csc_matrix

    from .geometry.rig_utils import joint_local_to_world, joint_world_to_local

    usd_path_str = str(usd_path)

    skel_prim = next((p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)), None)
    if skel_prim is None:
        raise RuntimeError(f"No UsdSkelSkeleton prim found in '{usd_path_str}'")
    skel = UsdSkel.Skeleton(skel_prim)

    joint_paths_raw = skel.GetJointsAttr().Get()
    if joint_paths_raw is None or len(joint_paths_raw) == 0:
        raise RuntimeError(f"Skeleton '{skel_prim.GetPath()}' in '{usd_path_str}' has no joints")
    joint_paths = list(joint_paths_raw)
    J = len(joint_paths)

    bind_xforms = skel.GetBindTransformsAttr().Get()
    if bind_xforms is None:
        raise RuntimeError(
            f"No bindTransforms on skeleton '{skel_prim.GetPath()}' in '{usd_path_str}'"
        )
    if len(bind_xforms) != J:
        raise RuntimeError(
            f"Skeleton '{skel_prim.GetPath()}' has {J} joints but "
            f"{len(bind_xforms)} bindTransforms in '{usd_path_str}'"
        )
    bind_usd = np.array(bind_xforms, dtype=np.float32).reshape(J, 4, 4)

    rest_xforms = skel.GetRestTransformsAttr().Get()
    if rest_xforms is not None and len(rest_xforms) != J:
        raise RuntimeError(
            f"Skeleton '{skel_prim.GetPath()}' has {J} joints but "
            f"{len(rest_xforms)} restTransforms in '{usd_path_str}'"
        )
    rest_usd = (
        np.array(rest_xforms, dtype=np.float32).reshape(J, 4, 4)
        if rest_xforms is not None
        else None
    )

    path_to_idx = {j: i for i, j in enumerate(joint_paths)}
    parent_ids_list = [
        path_to_idx.get(j.rsplit("/", 1)[0] if "/" in j else "", -1) for j in joint_paths
    ]

    joint_names = np.array([j.split("/")[-1] for j in joint_paths])
    # Root has no parent in USD (-1); SOMA convention: root points to itself (0).
    joint_parent_ids = np.array(parent_ids_list, dtype=np.int32)
    joint_parent_ids[joint_parent_ids < 0] = 0

    # USD stores matrices row-major (point * M); SOMA uses column-major (M * point).
    bind_pose_world = bind_usd.swapaxes(-2, -1)
    t_pose_local = rest_usd.swapaxes(-2, -1) if rest_usd is not None else bind_pose_world.copy()

    parent_ids_t = torch.from_numpy(joint_parent_ids)
    t_pose_world = joint_local_to_world(torch.from_numpy(t_pose_local), parent_ids_t).numpy()
    bind_pose_local = joint_world_to_local(torch.from_numpy(bind_pose_world), parent_ids_t).numpy()

    # --- Locate the skin mesh -------------------------------------------------
    if skin_mesh_name is not None:
        if not isinstance(skin_mesh_name, str) or not skin_mesh_name or "/" in skin_mesh_name:
            raise ValueError(
                f"skin_mesh_name must be a non-empty leaf name (no '/'), got {skin_mesh_name!r}"
            )
        skin_prim = _find_mesh_by_name(stage, skin_mesh_name)
        missing_label = f"named '{skin_mesh_name}'"
    else:
        skin_prim = _find_mesh_by_name(stage, DEFAULT_SKIN_MESH_NAME)
        if skin_prim is None:
            skin_prim = _find_first_mesh_under_skel_root(stage)
        missing_label = f"named '{DEFAULT_SKIN_MESH_NAME}' or any UsdGeomMesh under a UsdSkelRoot"
    if skin_prim is None:
        available = sorted({p.GetPath().name for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)})
        raise ValueError(
            f"Body skin mesh {missing_label} not found in '{usd_path_str}'. "
            f"Available UsdGeomMesh prims: {available}"
        )

    skin_mesh = UsdGeom.Mesh(skin_prim)
    _skin_pts = skin_mesh.GetPointsAttr().Get()
    if not _skin_pts:
        raise ValueError(
            f"Body skin mesh '{skin_prim.GetPath()}' has no points in '{usd_path_str}'"
        )
    bind_shape = np.array(_skin_pts, dtype=np.float32)

    _fvi = skin_mesh.GetFaceVertexIndicesAttr().Get()
    _fvc = skin_mesh.GetFaceVertexCountsAttr().Get()
    face_vert_indices = np.array(_fvi, dtype=np.int32) if _fvi is not None else None
    face_vert_counts = np.array(_fvc, dtype=np.int32) if _fvc is not None else None
    uv_data = _read_uv_primvars(skin_mesh)

    binding = UsdSkel.BindingAPI(skin_prim)
    ji_pv = binding.GetJointIndicesPrimvar()
    jw_pv = binding.GetJointWeightsPrimvar()
    if not ji_pv or not jw_pv:
        raise RuntimeError(
            f"No skinning primvars (skel:jointIndices / skel:jointWeights) "
            f"found on '{skin_prim.GetPath()}' in '{usd_path_str}'"
        )

    V = len(bind_shape)
    K = ji_pv.GetElementSize()
    if K <= 0:
        raise RuntimeError(f"Skinning primvar element size is {K} on '{skin_prim.GetPath()}'")
    ji_raw = np.array(ji_pv.Get(), dtype=np.int32)
    jw_raw = np.array(jw_pv.Get(), dtype=np.float32)
    if ji_raw.size != V * K or jw_raw.size != V * K:
        raise RuntimeError(
            f"Skinning primvars on '{skin_prim.GetPath()}' have inconsistent shape: "
            f"expected V*K = {V}*{K} = {V * K}, got jointIndices={ji_raw.size}, "
            f"jointWeights={jw_raw.size}"
        )
    ji = ji_raw.reshape(V, K)
    jw = jw_raw.reshape(V, K)

    # The mesh binding may declare its own joint subset; map to skeleton indices.
    skel_joint_to_idx = {name: i for i, name in enumerate(joint_paths)}
    binding_joints = binding.GetJointsAttr().Get()
    if binding_joints and len(binding_joints) > 0:
        binding_to_skel = np.array(
            [skel_joint_to_idx.get(str(j), -1) for j in binding_joints], dtype=np.int32
        )
    else:
        binding_to_skel = np.arange(J, dtype=np.int32)

    v_idx = np.repeat(np.arange(V, dtype=np.int32), K)
    j_idx = binding_to_skel[ji.ravel()]
    w_vals = jw.ravel()
    valid = (w_vals > 0) & (j_idx >= 0)

    W = np.zeros((V, J), dtype=np.float32)
    np.add.at(W, (v_idx[valid], j_idx[valid]), w_vals[valid])
    sw = csc_matrix(W)

    out = RigUSDData(
        joint_names=joint_names,
        joint_parent_ids=joint_parent_ids,
        bind_pose_world=bind_pose_world,
        bind_pose_local=bind_pose_local,
        t_pose_world=t_pose_world,
        t_pose_local=t_pose_local,
        bind_shape=bind_shape,
        skinning_weights_data=sw.data.astype(np.float32),
        skinning_weights_indices=sw.indices.astype(np.int32),
        skinning_weights_indptr=sw.indptr.astype(np.int32),
        skinning_weights_shape=np.array(sw.shape, dtype=np.int32),
    )
    if face_vert_indices is not None and face_vert_counts is not None:
        out["face_vert_indices"] = face_vert_indices
        out["face_vert_counts"] = face_vert_counts
        out["uv_data"] = uv_data
    return out


# ---------------------------------------------------------------------------
# NPZ animation I/O
# ---------------------------------------------------------------------------


def add_npz_args(parser: argparse.ArgumentParser) -> None:
    """Add common NPZ output arguments to an argparse parser."""
    parser.add_argument(
        "--output-npz",
        default=None,
        help="Output .npz file with SOMA pose parameters.",
    )
    parser.add_argument(
        "--keep-root",
        action="store_true",
        help="Include the virtual Root joint (J=78). Off by default (J=77) "
        "to match SOMALayer.pose() input convention.",
    )
    parser.add_argument(
        "--output-unit",
        choices=[u.unit_name for u in Unit],
        default=Unit.METERS.unit_name,
        help="Unit for translational quantities in the output .npz. Default: meters.",
    )


def _to_f32(x):
    """Convert tensor or array to float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32).numpy()
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    return arr


def save_soma_npz(
    out_path: str | Path,
    poses: np.ndarray | torch.Tensor,
    transl: np.ndarray | torch.Tensor,
    *,
    joint_names: Sequence[str],
    identity_model_type: str,
    identity_coeffs: np.ndarray | torch.Tensor,
    scale_params: np.ndarray | torch.Tensor | None = None,
    joint_orient: np.ndarray | torch.Tensor | None = None,
    global_scale: float | None = None,
    hand_type: str | None = None,
    unit: str = "meters",
    keep_root: bool = False,
    extra_arrays: Mapping[str, Any] | None = None,
) -> None:
    """Save SOMA animation to an .npz file.

    This function performs no conversion — it saves data exactly as
    provided.  Callers are responsible for unit conversion, rotation
    representation, and absolute-to-relative conversion beforehand.

    The rotation representation is inferred from the shape of *poses*:
    ``(N, J, 3)`` for axis-angle (rotvec), ``(N, J, 3, 3)`` for matrices.

    Whether the poses are absolute or relative to the T-pose is inferred
    from *joint_orient*: if provided, the poses are assumed to be
    T-pose-relative (and the joint orient is stored so a reader can
    convert back if needed).  If omitted, the poses are assumed absolute.

    Args:
        out_path: Output file path.
        poses: (N, J, 3) axis-angle or (N, J, 3, 3) rotation matrices,
            matching ``SOMALayer.pose()``'s *poses* argument.
        transl: (N, 3) root translation, matching ``SOMALayer.pose()``'s
            *transl* argument.
        joint_names: list of J joint name strings.
        identity_model_type: string identifying the identity model
            (e.g. ``"smpl"``, ``"mhr"``, ``"anny"``).
        identity_coeffs: (N, C) or (1, C) identity coefficients.
        scale_params: (N, S) or (1, S) optional per-identity scale
            vector. For SOMA, this stores the active bone-length scale
            ratios. For MHR and Anny, this is the body-part scales the
            identity model consumes at rest-shape time.
        joint_orient: (J, 3, 3) per-joint orientation from
            :obj:`~soma.geometry.rig_utils.precompute_joint_orient`.
            If provided, poses are stored as T-pose-relative.  If None,
            poses are stored as absolute.
        global_scale: scalar uniform scale factor applied on top of
            identity (e.g. fitted hand or body size).
        hand_type: ``"left"`` or ``"right"`` for hand animations; omit
            for body animations.
        unit: Unit label for translational quantities (``"meters"``,
            ``"centimeters"``, or ``"millimeters"``).
        keep_root: Include virtual Root joint (J=78 vs J=77).
        extra_arrays: Optional dict of additional arrays to include.
    """
    joint_names = list(joint_names)

    poses_np = _to_f32(poses)
    ndim = poses_np.ndim
    if ndim == 3 and poses_np.shape[-1] == 3:
        rotation_repr = "rotvec"
    elif ndim == 4 and poses_np.shape[-2:] == (3, 3):
        rotation_repr = "matrix"
    else:
        raise ValueError(
            f"Cannot infer rotation representation from poses shape {poses_np.shape}. "
            "Expected (N, J, 3) for rotvec or (N, J, 3, 3) for matrix."
        )

    absolute_pose = joint_orient is None

    # Strip Root joint (index 0) unless keep_root
    if not keep_root:
        poses_np = poses_np[:, 1:]
        joint_names = joint_names[1:]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "poses": poses_np,
        "transl": _to_f32(transl),
        "joint_names": np.array(joint_names),
        "identity_model_type": np.array(identity_model_type),
        "identity_coeffs": _to_f32(identity_coeffs),
        "keep_root": np.bool_(keep_root),
        "absolute_pose": np.bool_(absolute_pose),
        "rotation_repr": np.array(rotation_repr),
        "unit": np.array(unit),
    }
    if scale_params is not None:
        save_dict["scale_params"] = _to_f32(scale_params)
    if joint_orient is not None:
        save_dict["joint_orient"] = _to_f32(joint_orient)
    if global_scale is not None:
        save_dict["global_scale"] = np.float32(global_scale)
    if hand_type is not None:
        save_dict["hand_type"] = np.array(hand_type)
    if extra_arrays:
        save_dict.update(extra_arrays)

    np.savez_compressed(str(out_path), allow_pickle=False, **save_dict)

    pose_label = "absolute" if absolute_pose else "relative"
    root_label = "with Root (J=78)" if keep_root else "no Root (J=77)"
    summary_lines = [f"Saved: {out_path}"]
    if hand_type is not None:
        summary_lines.append(f"  hand_type: {hand_type}")
    summary_lines.append(f"  identity_model_type: {identity_model_type}")
    summary_lines.append(f"  identity_coeffs: {_to_f32(identity_coeffs).shape}")
    if scale_params is not None:
        summary_lines.append(f"  scale_params: {_to_f32(scale_params).shape}")
    if global_scale is not None:
        summary_lines.append(f"  global_scale: {float(global_scale):.4f}")
    summary_lines.append(f"  poses: {poses_np.shape} ({rotation_repr}, {pose_label}, {root_label})")
    summary_lines.append(f"  transl: {_to_f32(transl).shape} ({unit})")
    summary_lines.append(f"  joint_names: {len(joint_names)} joints")
    logger.info("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------


def load_soma_npz(path: str | Path) -> SOMANPZData:
    """Load a SOMA animation .npz saved by :obj:`~soma.io.save_soma_npz`.

    Returns a dict with the following keys:

    - ``poses``: ``(N, J, 3)`` rotvec or ``(N, J, 3, 3)`` matrices.
    - ``transl``: ``(N, 3)`` root translation.
    - ``joint_names``: list of *J* joint name strings.
    - ``identity_model_type``: e.g. ``"mhr"``, ``"smpl"``.
    - ``identity_coeffs``: ``(N, C)`` or ``(1, C)``.
    - ``rotation_repr``: ``"rotvec"`` or ``"matrix"``.
    - ``absolute_pose``: bool — True if poses are absolute (no joint orient).
    - ``unit``: unit label string (e.g. ``"meters"``).
    - ``keep_root``: bool — whether the Root joint (index 0) is included.

    Optional keys (present only if saved):

    - ``scale_params``: ``(N, S)`` or ``(1, S)``.
    - ``joint_orient``: ``(J, 3, 3)`` per-joint orient (present when
      ``absolute_pose=False``).
    - ``global_scale``: scalar uniform scale factor.
    - ``hand_type``: ``"left"`` or ``"right"`` (absent for body animations).

    Any extra arrays stored via *extra_arrays* are included as-is.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        dict of numpy arrays and Python scalars.
    """
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)

    result = SOMANPZData(
        poses=data["poses"],
        transl=data["transl"],
        joint_names=list(data["joint_names"]),
        identity_model_type=str(data["identity_model_type"]),
        identity_coeffs=data["identity_coeffs"],
        rotation_repr=str(data["rotation_repr"]),
        absolute_pose=bool(data["absolute_pose"]),
        unit=str(data["unit"]),
        keep_root=bool(data.get("keep_root", False)),
    )

    if "scale_params" in data:
        result["scale_params"] = data["scale_params"]
    if "joint_orient" in data:
        result["joint_orient"] = data["joint_orient"]
    if "global_scale" in data:
        result["global_scale"] = float(data["global_scale"])
    if "hand_type" in data:
        result["hand_type"] = str(data["hand_type"])

    _known = {
        "poses",
        "transl",
        "joint_names",
        "identity_model_type",
        "identity_coeffs",
        "rotation_repr",
        "absolute_pose",
        "unit",
        "keep_root",
        "scale_params",
        "joint_orient",
        "global_scale",
        "hand_type",
    }
    for key in data.files:
        if key not in _known:
            result[key] = data[key]

    return result


# ---------------------------------------------------------------------------
# USD skeletal animation export
# ---------------------------------------------------------------------------

_HIPS_IDX = 1  # SOMA Hips joint (child of virtual Root at index 0)


def _build_joint_paths(joint_names, joint_parent_ids):
    """Build UsdSkel path tokens from flat joint names and parent IDs."""
    names = list(joint_names)
    paths = [""] * len(joint_parent_ids)
    paths[0] = names[0]
    for j in range(1, len(joint_parent_ids)):
        paths[j] = paths[int(joint_parent_ids[j])] + "/" + names[j]
    return paths


def _rotmats_to_quats_wxyz(R):
    """Convert (N, J, 3, 3) rotation matrices to (N, J, 4) wxyz quaternions.

    Projects near-degenerate matrices to the nearest proper rotation via SVD,
    and applies inter-frame sign continuity correction.
    """
    from scipy.spatial.transform import Rotation as ScipyR

    N, J = R.shape[:2]
    flat = R.reshape(-1, 3, 3).copy()
    U, _S, Vt = np.linalg.svd(flat)
    R_clean = U @ Vt
    det = np.linalg.det(R_clean)
    neg = det < 0
    if neg.any():
        U[neg, :, -1] *= -1
        R_clean[neg] = U[neg] @ Vt[neg]
    q_xyzw = ScipyR.from_matrix(R_clean).as_quat()
    q_wxyz = np.empty_like(q_xyzw)
    q_wxyz[:, 0] = q_xyzw[:, 3]
    q_wxyz[:, 1:] = q_xyzw[:, :3]
    q_wxyz = q_wxyz.reshape(N, J, 4)
    for n in range(1, N):
        flip = (q_wxyz[n] * q_wxyz[n - 1]).sum(axis=-1) < 0
        q_wxyz[n, flip] *= -1
    return q_wxyz.astype(np.float32)


def save_soma_usd(
    out_path: str | Path,
    rotations: np.ndarray | torch.Tensor | None = None,
    root_translation: np.ndarray | torch.Tensor | None = None,
    *,
    joint_names: Sequence[str],
    joint_parent_ids: Sequence[int] | np.ndarray | torch.Tensor,
    bind_transforms_world: np.ndarray | torch.Tensor,
    bind_transforms_local: np.ndarray | torch.Tensor,
    rest_shape: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor | None = None,
    face_vert_indices: np.ndarray | torch.Tensor | None = None,
    face_vert_counts: np.ndarray | torch.Tensor | None = None,
    uv_data: Mapping[str, UVPrimvarEntry] | None = None,
    skinning_weights: np.ndarray | torch.Tensor,
    unit: str = "meters",
    fps: float = 30.0,
    topk: int = 8,
    root_joint_idx: int | None = None,
    skin_mesh_name: str = DEFAULT_SKIN_MESH_NAME,
) -> None:
    """Save a SOMA skeletal rig (and optionally animation) to a USD file with UsdSkel.

    When *rotations* and *root_translation* are omitted, only the static rig is
    written (skeleton bind pose + skinned mesh) with no SkelAnimation prim.

    The rotations and root_translation should come directly from
    :obj:`~soma.fitting.pose_inversion.PoseInversion.fit` (local-space,
    ``absolute_pose=True`` convention).

    Args:
        out_path: Output ``.usd``, ``.usda``, or ``.usdc`` file path.
        rotations: (N, J, 3, 3) local rotation matrices, or ``None`` for a static rig.
        root_translation: (N, 3) root joint local translation, or ``None`` for a static rig.
        joint_names: list of J joint name strings (including Root).
        joint_parent_ids: (J,) parent index array.
        bind_transforms_world: (J, 4, 4) or (1, J, 4, 4) world-space bind transforms.
        bind_transforms_local: (J, 4, 4) or (1, J, 4, 4) local-space bind transforms.
        rest_shape: (V, 3) bind-pose vertex positions.
        faces: (F, 3) triangle face indices.  Used as fallback when
            *face_vert_indices*/*face_vert_counts* are not provided.
        face_vert_indices: Flattened polygon face-vertex indices (e.g. quads from
            the template USD).  Takes priority over *faces* when provided.
        face_vert_counts: Per-face vertex counts matching *face_vert_indices*.
        uv_data: Optional UV sets in nested-dict format
            ``{name: {"coordinates": (M,2), "indices": ..., "interpolation": str}}``,
            as returned by :obj:`~soma.io.load_usd_mesh` or :obj:`~soma.io.load_rig_from_usd`.
        skinning_weights: (V, J) dense skinning weights.
        unit: ``"meters"``, ``"centimeters"``, or ``"millimeters"``.
        fps: Frames per second (timeCodesPerSecond).
        topk: Max joint influences per vertex for sparse skinning.
        root_joint_idx: Index of the joint that receives root_translation.
            Defaults to 1 (SOMA body Hips); pass 0 for hand models.
        skin_mesh_name: Leaf name for the skinned mesh prim under the
            ``UsdSkelRoot``. Defaults to ``DEFAULT_SKIN_MESH_NAME`` so that
            files written here round-trip cleanly through
            :obj:`~soma.io.load_rig_from_usd`.
    """
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt
    except ImportError:
        raise ImportError(
            "USD export requires the 'usd-core' package. Install it with: pip install usd-core"
        ) from None

    if root_joint_idx is None:
        root_joint_idx = _HIPS_IDX
    if not isinstance(skin_mesh_name, str) or not skin_mesh_name or "/" in skin_mesh_name:
        raise ValueError(
            f"skin_mesh_name must be a non-empty leaf name (no '/'), got {skin_mesh_name!r}"
        )

    has_animation = rotations is not None
    bind_transforms_world = np.asarray(bind_transforms_world, dtype=np.float64)
    bind_transforms_local = np.asarray(bind_transforms_local, dtype=np.float64)
    rest_shape = _to_f32(rest_shape)
    skinning_weights = _to_f32(skinning_weights)
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)
    joint_parent_ids = np.asarray(joint_parent_ids)

    if bind_transforms_world.ndim == 4:
        bind_transforms_world = bind_transforms_world[0]
    if bind_transforms_local.ndim == 4:
        bind_transforms_local = bind_transforms_local[0]
    if rest_shape.ndim == 3:
        rest_shape = rest_shape[0]

    J = len(joint_parent_ids)
    V = rest_shape.shape[0]

    joint_paths = _build_joint_paths(joint_names, joint_parent_ids)
    bind_local_t = bind_transforms_local[:, :3, 3].astype(np.float32)

    from .geometry.batched_skinning import topk_skinning

    idx, wts = topk_skinning(torch.from_numpy(skinning_weights).float(), K=topk, pad_index=0)
    idx_np = idx.numpy().astype(np.int32).reshape(-1)
    wts_np = wts.numpy().astype(np.float32).reshape(-1)
    K = idx.shape[1]

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(out_path)

    meters_per_unit = Unit.from_name(unit).meters_per_unit
    stage.SetMetadata("metersPerUnit", meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    skel_root = UsdSkel.Root.Define(stage, "/Root")
    skel = UsdSkel.Skeleton.Define(stage, skel_root.GetPath().AppendChild("Skeleton"))
    skel.CreateJointsAttr(Vt.TokenArray(joint_paths))
    skel.CreateBindTransformsAttr(
        Vt.Matrix4dArray.FromNumpy(bind_transforms_world.swapaxes(-2, -1))
    )
    skel.CreateRestTransformsAttr(
        Vt.Matrix4dArray.FromNumpy(bind_transforms_local.swapaxes(-2, -1))
    )

    if has_animation:
        rotations = _to_f32(rotations)
        root_translation = _to_f32(root_translation)
        N = rotations.shape[0]
        quats = _rotmats_to_quats_wxyz(rotations)

        stage.SetTimeCodesPerSecond(fps)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(N - 1)

        skel_anim = UsdSkel.Animation.Define(stage, skel.GetPath().AppendChild("Anim"))
        skel_anim.CreateJointsAttr(Vt.TokenArray(joint_paths))
        rot_attr = skel_anim.CreateRotationsAttr()
        transl_attr = skel_anim.CreateTranslationsAttr()
        scales_attr = skel_anim.CreateScalesAttr()

        rot_attr.Set(Vt.QuatfArray([Gf.Quatf(1, 0, 0, 0)] * J))
        transl_attr.Set(Vt.Vec3fArray.FromNumpy(bind_local_t))
        scales_attr.Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * J))

        skel_bind = UsdSkel.BindingAPI.Apply(skel.GetPrim())
        skel_bind.CreateAnimationSourceRel().SetTargets([skel_anim.GetPath()])

        for frame_idx in range(N):
            tc = Usd.TimeCode(float(frame_idx))
            frame_quats = Vt.QuatfArray(
                [
                    Gf.Quatf(
                        float(quats[frame_idx, j, 0]),
                        float(quats[frame_idx, j, 1]),
                        float(quats[frame_idx, j, 2]),
                        float(quats[frame_idx, j, 3]),
                    )
                    for j in range(J)
                ]
            )
            rot_attr.Set(frame_quats, tc)
            frame_t = bind_local_t.copy()
            frame_t[root_joint_idx] = root_translation[frame_idx]
            transl_attr.Set(Vt.Vec3fArray.FromNumpy(frame_t), tc)
            scales_attr.Set(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * J), tc)

    mesh = UsdGeom.Mesh.Define(stage, skel_root.GetPath().AppendChild(skin_mesh_name))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(rest_shape))
    if face_vert_indices is not None and face_vert_counts is not None:
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray(np.asarray(face_vert_counts, dtype=np.int32).tolist())
        )
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray(np.asarray(face_vert_indices, dtype=np.int32).tolist())
        )
        F_count = len(face_vert_counts)
    elif faces is not None:
        F_count = faces.shape[0]
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * F_count))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))
    else:
        raise ValueError("Either faces or (face_vert_indices, face_vert_counts) must be provided.")

    mesh_bind = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    mesh_bind.CreateSkeletonRel().SetTargets([skel.GetPath()])
    mesh_bind.CreateSkinningMethodAttr().Set(UsdSkel.Tokens.classicLinear)
    mesh_bind.CreateGeomBindTransformAttr().Set(Gf.Matrix4d(1.0))

    primvars = UsdGeom.PrimvarsAPI(mesh)
    ji_pv = primvars.CreatePrimvar("skel:jointIndices", Sdf.ValueTypeNames.IntArray, "vertex", K)
    ji_pv.Set(Vt.IntArray.FromNumpy(idx_np))
    jw_pv = primvars.CreatePrimvar("skel:jointWeights", Sdf.ValueTypeNames.FloatArray, "vertex", K)
    jw_pv.Set(Vt.FloatArray.FromNumpy(wts_np))

    if uv_data:
        _write_uv_primvars(primvars, uv_data)

    stage.SetDefaultPrim(skel_root.GetPrim())
    stage.GetRootLayer().Save()

    summary_lines = [f"Saved USD: {out_path}"]
    if has_animation:
        summary_lines.append(f"  joints: {J}, frames: {N} @ {fps} fps")
    else:
        summary_lines.append(f"  joints: {J} (static rig, no animation)")
    summary_lines.append(f"  vertices: {V}, faces: {F_count}")
    summary_lines.append(f"  skinning: top-{K} influences/vertex")
    summary_lines.append(f"  unit: {unit} (metersPerUnit={meters_per_unit})")
    logger.info("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# Vertex animation + high-level SOMA layer export wrappers
# ---------------------------------------------------------------------------


def save_vertex_animation_usd(
    out_path: str | Path,
    vertices: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    *,
    unit: str = "meters",
    fps: float = 30.0,
    prim_path: str = "/Mesh",
) -> None:
    """Save a mesh with per-frame vertex positions to USD.

    No skeleton or skinning -- just animated ``Points`` on a ``Mesh`` prim.
    Useful for exporting target meshes, blendshape sequences, or any
    topology-constant vertex animation.

    Args:
        out_path: Output ``.usd``, ``.usda``, or ``.usdc`` file path.
        vertices: (N, V, 3) per-frame vertex positions.
        faces: (F, 3) triangle face indices (constant across frames).
        unit: ``"meters"``, ``"centimeters"``, or ``"millimeters"``.
        fps: Frames per second.
        prim_path: USD prim path for the mesh (default ``"/Mesh"``).
    """
    try:
        from pxr import Usd, UsdGeom, Vt
    except ImportError:
        raise ImportError(
            "USD export requires the 'usd-core' package. Install it with: pip install usd-core"
        ) from None

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    N, V = vertices.shape[:2]
    F_count = faces.shape[0]

    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(out_path)

    meters_per_unit = Unit.from_name(unit).meters_per_unit
    stage.SetMetadata("metersPerUnit", meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(N - 1)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * F_count))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.flatten().tolist()))

    pts_attr = mesh.CreatePointsAttr()
    for frame_idx in range(N):
        pts_attr.Set(
            Vt.Vec3fArray.FromNumpy(vertices[frame_idx]),
            Usd.TimeCode(float(frame_idx)),
        )

    stage.SetDefaultPrim(mesh.GetPrim())
    stage.GetRootLayer().Save()

    logger.info(
        "Saved vertex animation USD: %s\n"
        "  vertices: %s, faces: %s, frames: %s @ %s fps\n"
        "  unit: %s (metersPerUnit=%s)",
        out_path,
        V,
        F_count,
        N,
        fps,
        unit,
        meters_per_unit,
    )


def export_soma_usd(
    output_path: str | Path,
    soma_layer: Any,
    rotations: np.ndarray | torch.Tensor,
    root_translation: np.ndarray | torch.Tensor,
    *,
    fps: float = 30.0,
    unit: str | None = None,
    root_joint_idx: int | None = None,
    skin_mesh_name: str | None = None,
) -> None:
    """Export skeletal animation to USD from a prepared SOMA layer.

    High-level convenience wrapper around :obj:`~soma.io.save_soma_usd` that
    extracts cached bind transforms, rest shape, skinning weights, and
    joint metadata from the layer. ``prepare_identity`` must have been
    called on the layer before this function. Procedural twist SOMA layers are
    exported as the public 78-joint skeleton for now; internal twist skinning
    weights are folded into the nearest public ancestor joint.

    Args:
        output_path: destination .usd/.usda/.usdc path.
        soma_layer: SOMALayer-compatible layer with cached identity.
        rotations: (N, J, 3, 3) absolute rotation matrices. For procedural
            twist layers, either public or internal joint rotations are
            accepted, and internal rotations are reduced to the public joints.
        root_translation: (N, 3) root translation in ``soma_layer.output_unit``.
        fps: animation frame rate.
        unit: unit string for USD metadata. If None, uses the layer's
            output unit name. If provided for a layer with ``output_unit``,
            it must match ``soma_layer.output_unit``.
        root_joint_idx: override for root joint index. If None, uses
            ``soma_layer.root_joint_idx`` if present, else omitted.
        skin_mesh_name: Leaf name of the skinned mesh prim. When ``None``
            (default), read from ``soma_layer.default_skin_mesh_name``
            if that property exists, else falls back to
            ``DEFAULT_SKIN_MESH_NAME``. Pass an explicit name to override.
    """
    from .geometry.rig_utils import joint_world_to_local

    bw = soma_layer._cached_bind_transforms_world
    if bw.ndim == 4 and bw.shape[0] == 1:
        bw = bw[0]
    rs = soma_layer._cached_rest_shape
    if rs.ndim == 3 and rs.shape[0] == 1:
        rs = rs[0]

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    joint_names = (
        soma_layer.rig_data["joint_names"]
        if isinstance(soma_layer.rig_data["joint_names"], list)
        else list(soma_layer.rig_data["joint_names"])
    )
    joint_parent_ids = soma_layer.joint_parent_ids
    skinning_weights = soma_layer.skinning_weights
    export_rotations = rotations
    if hasattr(soma_layer, "public_rig_view"):
        public_view = soma_layer.public_rig_view(bind_transforms_world=bw)
        bw = public_view.bind_transforms_world
        bl = public_view.bind_transforms_local
        joint_names = list(public_view.joint_names)
        joint_parent_ids = public_view.joint_parent_ids
        skinning_weights = public_view.skinning_weights
        export_rotations = soma_layer.to_public_rotations(export_rotations)
    else:
        bl = joint_world_to_local(bw, joint_parent_ids)

    kwargs = {}
    if root_joint_idx is not None:
        kwargs["root_joint_idx"] = root_joint_idx
    elif hasattr(soma_layer, "root_joint_idx"):
        kwargs["root_joint_idx"] = soma_layer.root_joint_idx

    layer_output_unit = getattr(soma_layer, "output_unit", None)
    if unit is None:
        if layer_output_unit is not None:
            unit = layer_output_unit.unit_name
        elif hasattr(soma_layer, "NATIVE_UNIT"):
            unit = soma_layer.NATIVE_UNIT.unit_name
        else:
            unit = "meters"
    elif layer_output_unit is not None and unit != layer_output_unit.unit_name:
        raise ValueError(
            "export_soma_usd writes the layer's output_unit data. "
            f"Got unit={unit!r}, but soma_layer.output_unit is "
            f"{layer_output_unit.unit_name!r}. Construct the layer with the desired "
            "output_unit before exporting."
        )

    if skin_mesh_name is None:
        skin_mesh_name = getattr(soma_layer, "default_skin_mesh_name", DEFAULT_SKIN_MESH_NAME)

    save_soma_usd(
        output_path,
        _to_np(export_rotations),
        _to_np(root_translation),
        joint_names=joint_names,
        joint_parent_ids=_to_np(joint_parent_ids),
        bind_transforms_world=_to_np(bw),
        bind_transforms_local=_to_np(bl),
        rest_shape=_to_np(rs),
        faces=_to_np(soma_layer.faces),
        skinning_weights=_to_np(skinning_weights),
        unit=unit,
        fps=fps,
        skin_mesh_name=skin_mesh_name,
        **kwargs,
    )
