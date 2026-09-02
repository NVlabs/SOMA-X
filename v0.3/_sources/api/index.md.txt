# API Overview

The current API draft focuses on the parts of `soma` that look stable and user-facing from the repository:

- top-level library entry points exported from `soma`
- the main full-body layer: `SOMALayer`
- the left/right hand layer: `SOMAHandLayer`
- `PoseInversion`, which is used by the conversion tooling
- the `soma.io` module for USD and NPZ interoperability
- the lower-level `soma.geometry` building blocks that back skinning, fitting, and rig transforms

## Recommended public surface

For the first published version of the docs, treat these as the primary supported entry points:

- `soma.SOMALayer`
- `soma.body.SOMALayer` and `soma.body.SOMAPoseOutput`
- `soma.SOMAHandLayer`
- `soma.hand.SOMAHandPoseOutput`
- `soma.setup_warp_for_ddp`
- `soma.Unit`
- `soma.create_identity_model` and the `soma.body.identity_model` backends
- `soma.io.*` USD and NPZ helpers
- `soma.fitting.pose_inversion.PoseInversion`
- `soma.fitting.rts_smoothing.smooth_pose`

The historical `soma.soma`, `soma.identity_model`, `soma.pose_inversion`, and
`soma.rts_smoothing` paths remain available for existing applications and
serialized references. New module-qualified imports should use `soma.body` and
`soma.fitting`.

## Advanced building blocks

The `soma.geometry.*` modules are documented as an advanced API section. They are especially useful when you want to:

- run lower-level FK and LBS utilities directly
- fit or retarget skeletons outside the layer wrappers
- reuse the Warp-accelerated alignment and skinning kernels in custom pipelines

These modules are still closer to implementation building blocks than the main layer APIs, so the geometry section should be read as advanced-user reference material rather than the primary onboarding path.
