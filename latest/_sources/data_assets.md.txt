# Data Assets

SOMA ships the following model assets in `assets/`, loaded at layer `__init__` time (callers typically do not touch them directly):

- [`SOMA_template_rig.usda`](#soma_template_rigusda) -- **required rig source** (joint hierarchy, bind/T-pose, bind-shape, skinning weights). This is the v0026 nvHuman template with SOMA procedural twist joints.
- [`SOMA_procedural_transforms.json`](#soma_procedural_transformsjson) -- portable procedural-control definition for the v0026 twist setup. The filename is intentionally unsuffixed; the schema version is inside the file.
- [`SOMA_neutral.npz`](#soma_neutralnpz) -- PCA shape model, mesh topology, UVs, LOD maps, semantic segments, and metadata.
- [Per-backend model folders](#per-backend-model-folders) -- native identity models for `mhr`, `smpl` / `smplx`, `anny`, `garment`, each with OBJ pairs used to compute the mesh correspondence to SOMA topology.

`.npz` files use `np.savez` with `allow_pickle=False`. SOMA native unit is **centimeters**, up axis `+Y`, forward axis `+Z`. Values below use `V` for the mid-LOD body vertex count (`18056`) and `J` for full-body joint count (`78`).

## `SOMA_template_rig.usda`

UsdSkel file holding the canonical body rig. Loaded by {py:func}`~soma.io.load_lod_rig_from_usd` during `SOMALayer.__init__`. This is the **source of truth** for the rig; the slim `SOMA_neutral.npz` no longer stores the rig fallback fields. Procedural-control topology and parameter metadata are loaded from `SOMA_procedural_transforms.json`.

The checked-in template is the v0026 nvHuman `nvHuman_male_skel.usda` publish with SOMA procedural twist joints and updated skin weights. By default, `SOMALayer` keeps the expanded template skeleton and the public pose input remains the current 77 controllable SOMA joints. Passing `enable_procedural_transforms=False` derives the legacy 78-joint public rig in memory by pruning procedural/auxiliary joints and aggregating each removed joint's skinning weights to its nearest kept parent. `SOMA_procedural_transforms.json` defines procedural topology, rotation extraction, and sparse parameter matrices.

Keys supplied by the USD:

- `joint_names`, `joint_parent_ids`
- `bind_pose_world`, `bind_pose_local`, `t_pose_world`, `t_pose_local`
- `bind_shape`
- `skinning_weights_{data,indices,indptr,shape}`
- Optionally `face_vert_indices`, `face_vert_counts`, `uv_data` (when the skin mesh carries polygon + UV data)

Keys **not** covered by the USD (still loaded from `SOMA_neutral.npz`):

- Shape PCA (`mean`, `shapedirs`, `eigenvalues`)
- Mesh topology (`triangles`, `triangles_low`, LOD maps)
- Semantic segments (`segment_*`)
- `mirror_vert_indices`, UV primvars stored as npz keys

Structure (top-level `UsdSkelRoot` at `/OUTPUT`):

- `/OUTPUT/c_skeleton_grp/Root` -- `UsdSkelSkeleton` with joint hierarchy, `bindTransforms`, `restTransforms`
- `/OUTPUT/c_skeleton_grp/Root/Animation` -- `UsdSkelAnimation`
- `/OUTPUT/c_geometry_grp/MainMesh/Meshes/c_skin_mid` -- the mid-LOD skin mesh (`default_skin_mesh_name` for `SOMALayer(..., lod="mid")`)
- `/OUTPUT/c_geometry_grp/MainMesh/Meshes/c_skin_lo` -- the low-LOD skin mesh
- `/OUTPUT/c_geometry_grp/MainMesh/Meshes/c_skin_xlo` -- the extra-low-LOD skin mesh

The skin mesh name is resolved by `load_lod_rig_from_usd`, which prefers LOD-specific names such as `c_skin_mid`, `c_skin_lo`, and `c_skin_xlo`.

Procedural mode uses a single SOMA-owned procedural parameter transform with compiled rotation and translation matrices loaded from `SOMA_procedural_transforms.json`: forearm and shin twist come from hand and foot twist, while upper arm and thigh twist use reverse start/end compensation. The SOMA path is sidecar-driven and owns translation generation. `rotation_extraction` in the JSON selects `local_x_euler`, `local_x_swing_twist`, `aligned_x_swing_twist`, or per-procedural-joint mixes of those extractors. The checked-in SOMA sidecar uses `aligned_x_swing_twist` globally. `local_x_euler` reads the configured SOMA local-X twist channel from local Euler angles. `local_x_swing_twist` extracts the same configured channel by projecting a half-angle stabilized source quaternion onto its SOMA twist axis. The current pose-channel convention maps arm local-X twist to axis X, left leg local-X twist to negative axis Y, and right leg local-X twist to positive axis Y. The JSON translation matrix places twist helpers along the fitted public segment, preserving identity and body-part stretch instead of trusting independently fitted twist translations. Body pose correctives are supported only with procedural transforms and can be disabled by constructing `SOMALayer(correctives_model_path=None)`.

## `SOMA_procedural_transforms.json`

Declarative procedural-control definition loaded by `SOMALayer` from
`data_root`. It is the authoritative source for supported extraction modes,
public 78-joint derivation from the 122-joint template, twist segments, sparse
rotation and translation parameter matrices, rotation extraction policy, and
evaluation order. `SOMALayer` requires this sidecar unless
`enable_procedural_transforms=False`; there is no
Python hard-coded SOMA twist topology or extraction-mode fallback. See
[`Procedural Control Format`](procedural_control_format.md) for the schema and
non-Python consumer plan.

## `SOMA_neutral.npz`

Single-package PCA shape model + topology data for the neutral full-body mesh (`gender: neutral`, right-handed). Produced by the asset pipeline; its `metadata` field is a JSON string with model version, provenance, training-source information, and the asset split contract. Runtime rig data comes from `SOMA_template_rig.usda`, which is required for the slim NPZ.

### Shape / PCA

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `mean` | `(V, 3)` | f32 | Mean (neutral) vertex positions in native cm. |
| `shapedirs` | `(K, V*3)` | f32 | PCA shape directions flattened per vertex, `K = 128`. |
| `eigenvalues` | `(K,)` | f32 | PCA eigenvalues; `SOMAIdentityModel` scales coeffs by `sqrt(eigenvalues)` before linear reconstruction. |

### Topology

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `triangles` | `(T, 3)` | i32 | Triangle faces for the mid-LOD mesh; `T = 36108`. |
| `face_vert_indices` | `(sum(counts),)` | i32 | USD-style flat per-face vertex-index stream (ngon-compatible). |
| `face_vert_counts` | `(F,)` | i32 | USD-style per-face vertex count (3 or 4); `F = 18054`. |
| `mirror_vert_indices` | `(V,)` | i32 | Per-vertex left/right mirror index. `mirror_vert_indices[i]` is the vertex symmetric to `i` across the midline. |

### Removed rig fields

The slim asset no longer stores `joint_names`, `joint_parent_ids`, `bind_pose_world`, `bind_pose_local`, `t_pose_world`, `t_pose_local`, `bind_shape`, or `skinning_weights_{data,indices,indptr,shape}`. These arrays are loaded from `SOMA_template_rig.usda`.

### UV primvars (three sets)

Three UV sets (`st`, `st1`, `st2`) preserved from the source USD. Each uses `faceVarying` interpolation: indices are flat per-face-corner lookups into `uv_coord_*`.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `uv_coord_st` / `uv_coord_st1` / `uv_coord_st2` | `(U, 2)` | f32 | UV coordinates; `U` varies per set (~19.6k). |
| `uv_indices_st` / `uv_indices_st1` / `uv_indices_st2` | `(sum(counts),)` | i32 | Per-face-corner UV index lookup. |
| `uv_interp_st` / `uv_interp_st1` / `uv_interp_st2` | scalar | `<U11` | Always `faceVarying`. |

### Low-LOD subset

A ~1:4 vertex subset for faster inference (`SOMALayer(..., lod="low")`; legacy alias `low_lod=True`). The low-LOD mesh is a strict vertex-index subset of the mid-LOD mesh.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `lod_mid_to_low` | `(V_lo,)` | i32 | Mid-LOD vertex indices that survive in the low-LOD mesh. `V_lo = 4505`. |
| `triangles_low` | `(T_lo, 3)` | i32 | Low-LOD triangles in low-LOD vertex index space; `T_lo = 9006`. |
| `face_vert_indices_low` | `(sum,)` | i32 | Flat per-face vertex-index stream for low-LOD. |
| `face_vert_counts_low` | `(F_lo,)` | i32 | Per-face vertex count for low-LOD; `F_lo = 4524`. |

### Extra-low LOD USD asset

`SOMALayer(..., lod="xlo")` returns 612 body vertices with the same SOMA skeleton and identity backends, but loads the extra-low mesh topology, bind vertices, skinning weights, and UVs from the xlo mesh in `assets/SOMA_template_rig.usda`. Unlike `lod="low"`, xlo is not a strict vertex-index subset of the mid mesh. Runtime identity shapes and pose correctives are transferred from the mid SOMA bind geometry to xlo with barycentric interpolation. Identity-dependent skeleton fitting still uses the low-LOD mesh from the same v0026 USD internally, because direct xlo fitting is too sparse around limbs for stable joint placement across random body shapes.

The expected asset is the versioned nvHuman v0026 USD publish, stored under the canonical `SOMA_template_rig.usda` filename. The loader auto-detects skinned LOD meshes using names such as `c_skin_mid`, `c_skin_lo`, and `c_skin_xlo`.

### Semantic segments (vertex id lists)

Each entry is a flat `int32` array of vertex IDs into the mid-LOD mesh. Useful for masking/filtering (e.g. excluding inner geometry from PoseInversion).

| Key | Description |
|-----|-------------|
| `segment_head` | Head (face + skull surface). |
| `segment_feet` | Feet. |
| `segment_torso` | Torso. |
| `segment_mouth_bag` | Inner mouth geometry (excluded from pose inversion). |
| `segment_eye_bags` | Inner eye geometry (excluded from pose inversion). |
| `segment_between_toes` | Inter-toe geometry. |
| `segment_hair` | Hair strands. |
| `segment_haircap` | Hair cap. |
| `segment_armpits` | Armpits. |

### Metadata

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `metadata` | scalar | unicode | JSON string: `model_version`, `gender`, `units`, `up_axis`, `forward_axis`, `handedness`, `asset_contract`, and `provenance` (training sources, PCA config, rig source). |

## Per-backend model folders

Each non-SOMA identity backend (`mhr`, `smpl`, `smplx`, `anny`, `garment`) has its own subdirectory under `data_root/` holding a pair of OBJ meshes that define the mesh correspondence to SOMA topology (plus, for some backends, the backend's native identity model).

### Correspondence OBJ pair

Every backend folder contains:

- `base_body.obj` (or `base_body_lod{1,6}.obj` for MHR, or `mean.obj` for Garment) -- the backend's **native-topology** mesh in its own native frame and unit.
- `SOMA_wrap.obj` (or `SOMA_wrap_lod1.obj`) -- the **SOMA-topology** mesh wrapped onto the native mesh's surface.

Together these define a surface-to-surface correspondence: vertex `i` of `SOMA_wrap.obj` sits on (or very near) the native mesh's surface, at the same material point across every pose / shape. At `__init__` time, each identity model calls `_setup_topology_transfer[_with_blending](V_native, F_native, V_soma, F_soma, ...)` to precompute the barycentric transfer that maps a shaped native mesh back into SOMA topology. For backends that leave a head region not covered by the wrap (MHR, SMPL*), a Laplacian blend at the boundary smooths the transition.

The OBJ meshes are loaded with `trimesh.load(..., maintain_order=True, process=False)` so vertex ordering is preserved -- do not reprocess them with any tool that reorders vertices.

### Folder contents

| Folder | Backend class | Checked-in native model | OBJ correspondence pair |
|--------|---------------|-------------------------|-------------------------|
| `MHR/` | `MHRIdentityModel` | `mhr_model_lod{1,6}.pt` (TorchScript) | `base_body_lod{1,6}.obj` + `SOMA_wrap_lod1.obj` |
| `SMPL/` | `SMPLIdentityModel` (type `smpl`) | -- (see below) | `base_body.obj` + `SOMA_wrap.obj` |
| `SMPLX/` | `SMPLIdentityModel` (type `smplx`) | -- (see below) | `base_body.obj` + `SOMA_wrap.obj` |
| `Anny/` | `AnnyIdentityModel` | (loaded from the `anny` Python package at runtime) | `base_body.obj` + `SOMA_wrap.obj` |
| `GarmentMeasurements/` | `GarmentMeasurementIdentityModel` | -- (see below) | `mean.obj` + `SOMA_wrap.obj` |

### Optional user-placed model files

The SMPL / SMPL-X identity models are **not redistributed** with SOMA. To use those backends, download the official models and place them in the corresponding folder:

| Backend | Expected file(s) | Source |
|---------|------------------|--------|
| `smpl`  | `SMPL/SMPL_NEUTRAL.pkl` | [SMPL](https://smpl.is.tue.mpg.de/) |
| `smplx` | `SMPLX/SMPLX_NEUTRAL.pkl` | [SMPL-X](https://smpl-x.is.tue.mpg.de/) |

`SMPLIdentityModel` raises `FileNotFoundError` at `__init__` with a pointer to these filenames if the files are missing. An explicit path can be passed via `identity_model_kwargs={"model_path": ...}`.

The `garment` backend expects `GarmentMeasurements/point.npz`, which must be generated locally from the publicly available `point.pca` binary distributed with the upstream [GarmentMeasurements repo](https://github.com/mbotsch/GarmentMeasurements). Convert with `tools/convert_gm_pca_to_npz.py`:

```bash
git clone https://github.com/mbotsch/GarmentMeasurements
python tools/convert_gm_pca_to_npz.py ./GarmentMeasurements/data/pca/point.pca assets/GarmentMeasurements/point.npz
```
