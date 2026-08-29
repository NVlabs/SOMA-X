# Changelog

## v0.2.4

Changes from v0.2.3:

- Keeps the virtual Root fixed during Lie-GN pose inversion, improving
  convergence without introducing an unobservable root transform.

## v0.2.3

Changes from v0.2.2:

- Adds structured Hugging Face model-card metadata for the Apache 2.0 license
  and public discovery tags, and fixes the model-card license link.

## v0.2.2

Changes from v0.2.1:

- Updates `SOMA_template_rig.usda` with a simplified SOMA template that removes
  unused helper joints and reduces the asset from approximately 345 MB to 28 MB
  while preserving the public pose interface.
- Fixes procedural twist evaluation to use the USD skin bind pose and correctly
  reposes SMPL and SMPL-X identities when procedural transforms are enabled.
- Adds reusable RTS pose smoothing through `soma.rts_smoothing.smooth_pose()`
  and the `--smooth` options in `tools/mhr2soma.py`.
- Adds four native SOMA foot-scaling controls while continuing to accept
  legacy 56-value scale tensors.
- Improves automatic alignment and PoseInversion robustness for reflected
  inputs.

## v0.2.1

SOMA-X v0.2.1 is the recommended 0.2 series release. The 0.2 series is a
substantial full-body update focused on deformation quality, procedural rig
controls, body model interoperability, updated rig assets, corrective-model
support, and public documentation. New in v0.2.1, SOMA-X adds a corrective
model path for procedural twist-joint rigs.

### Highlights

- Added procedural transforms that drive twist joints for improved full-body
  deformations, with Blender and Maya reference implementations for DCC
  evaluation.
- New in v0.2.1: added corrective-model support that is aware of procedural
  twist joints, including corrected bind reposing for twist-rig workflows.
- Added extra-low body LOD support for lightweight rendering and fitting
  workflows.
- Added pose conversion utilities for interoperability across SOMA, SMPL, and
  SMPL-X workflows.
- Improved PoseInversion, skinning, geometry utilities, and runtime stability
  for full-body use cases.
- Reorganized public SOMA assets around standard 3D rig data and SOMA-specific
  identity/auxiliary-geometry metadata.
- Added public API documentation and package-release automation.

### Full-Body Runtime and Geometry

- Added procedural transform support for driving twist joints, improving
  deformation quality in articulated body motion through `SOMALayer`.
- Added corrective-model execution for twist-joint-aware SOMA body rigs and
  extra-low LOD paths.
- Added native SOMA body bone-scaling support and flexible identity-preparation
  controls.
- Improved PoseInversion with more stable rotation alignment and solve paths.
- Split and clarified batched skinning/FK APIs.
- Added support for an extra-low SOMA body LOD.

### Model Interoperability

- Added `tools/pose_converter.py` for transferring pose data across SOMA, SMPL, and
  SMPL-X workflows, including SOMA procedural pose outputs and native
  SMPL/SMPL-X pose parameter formats.
- Added `soma.smpl.transfer` helpers for programmatic pose-transfer workflows
  across SOMA and SMPL-family model layers.

### Assets

- Added `SOMA_template_rig.usda` as a standard USD representation of the SOMA
  rig, including updated twist-joint and LOD information for 3D/DCC pipelines.
- Reorganized the public asset split so standard rigging data lives in USD,
  while `SOMA_neutral.npz` carries identity shape model information, auxiliary
  geometry information, and associated metadata.
- Added `SOMA_procedural_transforms.json` as a portable sidecar describing the
  procedural twist-joint transforms used by the runtime and DCC references.
- Added the corrective model asset for twist-joint-aware body rigs.
- Updated public asset metadata for the 0.2 series release.

### DCC Tooling

- Added Blender and Maya reference implementations under
  `tools/soma_procedural_blender` and `tools/soma_procedural_maya`.
- These tools demonstrate and validate procedural twist transforms in DCC
  environments; they are not identity-generation or auto-rigging tools.

### Documentation

- Added public API documentation for core SOMA-X modules.
- Added documentation for data assets, procedural controls, and geometry APIs.
- Added versioned GitHub Pages deployment support:
  - `/latest/`
  - `/v0.2/`
  - `/stable/`

### Packaging and Release Process

- Package version is `py-soma-x==0.2.1`.
- Added release automation for docs, package builds, PyPI Trusted Publishing,
  and public mirror validation.

### Upgrade Notes

- Existing `SOMALayer` usage remains source-compatible for common full-body
  workflows.
- Users can continue installing optional third-party identity-model support with
  the documented extras and provider-specific assets.
- Procedural-control workflows require the updated v0.2 assets and sidecar
  files.
- v0.2.1 is the first 0.2 series package with corrective-model support for
  procedural twist-joint rigs.
- The PyPI package should be installed or upgraded with:

  ```bash
  pip install --upgrade py-soma-x==0.2.1
  ```

## v0.2.0

SOMA-X v0.2.0 is a substantial full-body update focused on deformation quality,
procedural rig controls, body model interoperability, updated rig assets, and
public documentation.

### Highlights

- Added procedural transforms that drive twist joints for improved full-body
  deformations, with Blender and Maya reference implementations for DCC
  evaluation.
- Added extra-low body LOD support for lightweight rendering and fitting
  workflows.
- Added pose conversion utilities for interoperability across SOMA, SMPL, and
  SMPL-X workflows.
- Improved PoseInversion, skinning, geometry utilities, and runtime stability
  for full-body use cases.
- Reorganized public SOMA assets around standard 3D rig data and SOMA-specific
  identity/auxiliary-geometry metadata.
- Added public API documentation and package-release automation.

### Full-Body Runtime and Geometry

- Added procedural transform support for driving twist joints, improving
  deformation quality in articulated body motion through `SOMALayer`.
- Added native SOMA body bone-scaling support and flexible identity-preparation
  controls.
- Improved PoseInversion with more stable rotation alignment and solve paths.
- Split and clarified batched skinning/FK APIs.
- Added support for an extra-low SOMA body LOD.

### Model Interoperability

- Added `tools/pose_converter.py` for transferring pose data across SOMA, SMPL, and
  SMPL-X workflows, including SOMA procedural pose outputs and native
  SMPL/SMPL-X pose parameter formats.
- Added `soma.smpl.transfer` helpers for programmatic pose-transfer workflows
  across SOMA and SMPL-family model layers.

### Assets

- Added `SOMA_template_rig.usda` as a standard USD representation of the SOMA
  rig, including updated twist-joint and LOD information for 3D/DCC pipelines.
- Reorganized the public asset split so standard rigging data lives in USD,
  while `SOMA_neutral.npz` carries identity shape model information, auxiliary
  geometry information, and associated metadata.
- Added `SOMA_procedural_transforms.json` as a portable sidecar describing the
  procedural twist-joint transforms used by the runtime and DCC references.
- Updated public asset metadata for the v0.2.0 release.

### DCC Tooling

- Added Blender and Maya reference implementations under
  `tools/soma_procedural_blender` and `tools/soma_procedural_maya`.
- These tools demonstrate and validate procedural twist transforms in DCC
  environments; they are not identity-generation or auto-rigging tools.

### Documentation

- Added public API documentation for core SOMA-X modules.
- Added documentation for data assets, procedural controls, and geometry APIs.
- Added versioned GitHub Pages deployment support:
  - `/latest/`
  - `/v0.2/`
  - `/stable/`

### Packaging and Release Process

- Package version is `py-soma-x==0.2.0`.
- Added release automation for docs, package builds, PyPI Trusted Publishing,
  and public mirror validation.

### Upgrade Notes

- Existing `SOMALayer` usage remains source-compatible for common full-body
  workflows.
- Users can continue installing optional third-party identity-model support with
  the documented extras and provider-specific assets.
- New procedural-control workflows require the updated v0.2 assets and sidecar
  files.
- The PyPI package should be installed or upgraded with:

  ```bash
  pip install --upgrade py-soma-x==0.2.0
  ```

### Known Limitations

- The corrective model remains beta and is not intended for use when procedural
  twist joints are on.

## v0.1.0

Initial public SOMA-X release.
