# Procedural Control Format

SOMA procedural controls are declared in `assets/SOMA_procedural_transforms.json`.
The filename is intentionally not suffixed by the schema version; versioning lives
inside the file as `schema_version`.

This JSON sidecar is the authoritative runtime source for SOMA procedural
topology and parameter metadata. The Python runtime, Maya reference, Blender
reference, and other consumers should load this file alongside
`SOMA_template_rig.usda`; embedding the definition into USD metadata is out of
scope for the current sidecar format.

## Scope

The current schema describes the v0026 nvHuman template twist setup consumed by
`SOMALayer` when procedural transforms are enabled. It is declarative: consumers
resolve stable joint names to indices, validate the definition, compile numeric
buffers, and evaluate the same ordered transform without running arbitrary
Python code.

## Required Top-Level Fields

- `schema_version`: semantic schema version for the JSON structure.
- `modes`: supported extraction modes. SOMA currently supports `local_x_euler`,
  `local_x_swing_twist`, and `aligned_x_swing_twist`.
- `template_asset`: identity of the template rig asset the definition targets.
- `conventions`: units, angle units, scalar dtype, matrix layout, rotation
  handedness, Euler order, quaternion order, and axes.
- `public_rig_derivation`: stable `main_joint_names` plus policies for removed
  joint parent remapping and skin-weight aggregation.
- `channel_extractors`: deterministic extraction semantics for each mode.
- `rotation_extraction`: selected extraction policy. This may be a single mode
  string applied globally, or an object with `default` and
  `per_procedural_joint` overrides keyed by generated twist-joint name.
- `segments`: main start/end joints, generated twist joints, reverse
  compensation flag, source axis, and sign.
- `parameter_matrices`: sparse COO-style named rotation and translation
  matrices with float32 values.
- `evaluation_order`: the required runtime operation order.

## Evaluation Semantics

Public rig derivation keeps the named 78 main SOMA joints from the 122-joint template.
All other template joints are removed by remapping their parent to the nearest
kept ancestor and adding their skinning-weight column into that kept ancestor
before the removed column is dropped.

### Pose Rotation Convention

Procedural channel extraction starts from the same public pose convention used by
`SOMALayer.pose()` and the normal skinning path. Let `p(j)` be the parent of
joint `j`, let `O_j` be joint `j`'s T-pose world orient, and let `R_abs,j` be
the absolute local rotation consumed by FK.

For the default `absolute_pose=False` path, callers provide T-pose-relative local
rotations `R_rel,j`, which are converted before FK:

```{math}
R_{\mathrm{abs},j} =
O_{p(j)}^{\mathsf{T}} R_{\mathrm{rel},j} O_j
```

For `absolute_pose=True`, callers provide absolute local rotations directly:

```{math}
R_{\mathrm{abs},j} = R_{\mathrm{in},j}
```

The inverse conversion, used when writing absolute local rotations back to the
T-pose-relative convention, is:

```{math}
R_{\mathrm{rel},j} =
O_{p(j)} R_{\mathrm{abs},j} O_j^{\mathsf{T}}
```

The T-pose-relative convention is the SMPL-style convention. `absolute_pose=True`
is for already-oriented local rotations, such as DCC joint matrices or
PoseInversion output with joint orient baked in.

### Rotation Extraction Modes

`rotation_extraction` chooses which extractor feeds each generated twist-joint
row. A global string applies to every procedural joint. The object form supports
mixed extractors with `default` and `per_procedural_joint` overrides:

```json
{
  "default": "local_x_euler",
  "per_procedural_joint": {
    "LeftForeArmTwist2": "local_x_swing_twist"
  }
}
```

For each extraction mode `m`, SOMA builds a source-channel vector `c^(m)` from
public joint rotations and applies the mode-specific sparse rotation matrix
`M^(m)`. The generated twist angles are the sum of all selected mode products:

```{math}
\theta = \sum_m M^{(m)} c^{(m)}
```

Each generated twist joint `k` is emitted as a signed axis rotation:

```{math}
R_{\mathrm{twist},k} =
\operatorname{Rot}_{a_k}(s_k \theta_k)
```

The supported extraction modes define `c^(m)` as follows:

- `local_x_euler`: reads the configured SOMA local-X twist channel from XYZ local
  Euler angles on the absolute local source rotation.
- `local_x_swing_twist`: converts the absolute local source rotation to an
  `xyzw` quaternion and extracts the configured twist axis with half-angle
  stabilization.
- `aligned_x_swing_twist`: derives segment-aligned virtual orientations from
  source world transforms and bind alignment, then extracts X-axis swing/twist
  from the relative virtual segment orientation. Reverse-compensation segments
  use inherited start-joint twist where declared by the sidecar.

The checked-in v0026 SOMA sidecar declares all three modes and uses
`aligned_x_swing_twist` globally.

The rotation matrix rows are generated twist joints and columns are public SOMA
joints. The translation matrix keeps identity rows by default and overrides twist
rows with start/end public segment weights. Runtimes may resolve names once and
cache dense, COO, CSR, or device-native buffers as long as they preserve the
declared float32 values and evaluation order. Mixed extraction modes should stay
vectorized by compiling one rotation matrix per extractor and summing the
mode-specific matrix products; do not loop over procedural joints at evaluation
time.

## Validation

The Python loader rejects malformed definitions before constructing the evaluator.
It reports missing source joints, missing twist joints, duplicate outputs,
unknown matrix rows or columns, invalid axes, invalid signs, unsupported modes,
and missing or invalid rotation extraction policies.

## Non-Python Consumer Plan

A Maya or game-engine evaluator should:

1. Load `SOMA_template_rig.usda` and `SOMA_procedural_transforms.json`.
2. Resolve every joint name in `main_joint_names`, `segments`, and sparse matrix
   entries to local runtime indices.
3. Validate duplicate outputs, axis/sign values, matrix names, rotation
   extraction policies, and non-degenerate start/end translation segments.
4. Compile source-channel extraction buffers, sparse matrices, and output emitter
   buffers to the target runtime format.
5. Apply the declared `evaluation_order` during evaluation, then hand the
   expanded skeleton to that runtime's FK/LBS implementation.
