# Demos and conversion tools

The repository includes optional command-line tools for visualization,
retargeting, pose sampling, and DCC integration. Install the demo extras before
using tools that render images or video:

```bash
uv pip install -e ".[demo]"
```

## Full-body demo

Render one or more identity backends with the shared SOMA body rig:

```bash
python tools/demo_soma_vis.py \
  --data-root assets \
  --output-dir out/body-demo \
  --identity-model-type soma,mhr \
  --lod mid
```

Use `--random-shape` to animate identities and `--motion-file` to provide a
custom motion. The `mid`, `low`, and `xlo` LODs share the same public pose
contract.

## SOMA Hand demo

Render the left and right wrist-local layers from a body motion:

```bash
python tools/hand/demo_soma_hand_vis.py \
  --data-root assets \
  --hand-type left,right \
  --remove-wrist-translation \
  --output-dir out/hand-demo
```

Use `--shape-only --random-shape` to render native hand identity variation
without a body motion. MANO and MHR identity backends are selected with
`--identity-model-type` and require their corresponding local model assets.
For MANO, pass the separately licensed model explicitly with
`--hand-type right --mano-model-path /path/to/MANO_RIGHT.pkl` (or the
left-hand equivalent).
By default, motion renders apply the wrist's animated world transform. Use
`--remove-wrist-translation` to retain wrist orientation while keeping the
hand centered. The default camera framing leaves room around the motion-wide
hand bounds; increase `--camera-framing-scale` to zoom out further.

Both demos accept `--skeleton-overlay` to draw the public skeleton as
octahedral bones over the mesh, and `--skeleton-style {light,skin}` to pick a
neutral light-gray or a darker skin-toned bone color.

## README teaser

`make_teaser.py` composes the body and hand demo renders into the single
README teaser GIF with one title and label specification, so both rows stay
aligned and typographically consistent:

```bash
python tools/demo_soma_vis.py \
  --identity-model-type soma,mhr,smplx,anny,garment \
  --skeleton-overlay --skeleton-style skin \
  --image-size 1440 --max-frames 576 --output-dir out/teaser/body
for backend in soma mhr mano; do
  python tools/hand/demo_soma_hand_vis.py \
    --hand-type right --identity-model-type "$backend" \
    --remove-wrist-translation --camera-framing-scale 1.0 \
    --skeleton-overlay --skeleton-style skin \
    --image-size 1440 --max-frames 576 --output-dir out/teaser/hand
done
python tools/make_teaser.py \
  --body-videos out/teaser/body/soma_fixed_shape_skel.mp4,out/teaser/body/mhr_fixed_shape_skel.mp4,out/teaser/body/smplx_fixed_shape_skel.mp4,out/teaser/body/anny_fixed_shape_skel.mp4,out/teaser/body/garment_fixed_shape_skel.mp4 \
  --body-labels "SOMA native,MHR,SMPL-X,Anny,GM" \
  --hand-videos out/teaser/hand/hand_right_soma_no_correctives_fixed_shape_skel.mp4,out/teaser/hand/hand_right_mhr_no_correctives_fixed_shape_skel.mp4,out/teaser/hand/hand_right_mano_no_correctives_fixed_shape_skel.mp4 \
  --hand-labels "SOMA native,MHR,MANO" \
  --output assets/images/soma-in-action.gif
```

The MANO backend additionally needs `--mano-model-path`, and the SMPL-X and
GarmentMeasurements backends need their locally licensed model files. The
compositor requires `ffmpeg` on the path.

## Sample the hand articulation prior

`sample_soma_hand_pose_pca.py` draws reproducible coefficients from the
distributed articulation prior, converts the bind-relative exponential maps
to the layer's absolute-local rotation convention, poses the skinned mesh, and
writes MP4/GIF output.

```bash
python tools/hand/sample_soma_hand_pose_pca.py \
  --hand-asset assets/SOMAHand.npz \
  --hand-type right \
  --num-poses 60 \
  --seed 20260829 \
  --output-prefix out/soma-hand-samples
```

Use `--n-components`, `--sample-scale`, and `--lod` to control the sampled
prior and rendered geometry.

## SMPL-family to SOMA

Convert an SMPL animation to SOMA and optionally export the recovered pose:

```bash
python -m tools.smpl2soma --output-npz out/smpl-soma.npz
```

The converter uses `PoseInversion`: analytical fitting is the default, and
`--autograd-iters` adds differentiable FK refinement. SMPL/SMPL-X model files
must be supplied under their own license.

## MHR to SOMA

Convert MHR-format parquet data, including SAM 3D Body outputs:

```bash
python -m tools.mhr2soma \
  --input /path/to/parquet-directory \
  --output-npz out/mhr-soma.npz
```

Use `--max-samples` for bounded local checks. The tool also exposes the
reusable RTS smoothing presets through its `--smooth` options.

## Convert identity backends

The identity conversion tools accept a canonical SOMA animation NPZ written
by `soma.io.save_soma_npz`, preserve its animation, and optimize target
identity parameters against the source geometry in the SOMA bind pose.

Convert a full-body SMPL-X fit to native SOMA identity parameters:

```bash
python -m tools.convert_identity_backend input_smplx.npz output_soma.npz \
  --target-backend soma \
  --source-model-path /path/to/SMPLX_NEUTRAL.pkl
```

The source backend and coefficients are read from the input NPZ. This example
therefore converts its stored SMPL-X betas to native SOMA identity
coefficients and bone scales. Other supported full-body targets include MHR,
Anny, SMPL, SMPL-H, SMPL-X, and GarmentMeasurements.

Use the separate hand entry point for native SOMA Hand, MHR hand, and MANO:

```bash
python -m tools.hand.convert_identity_backend input_mhr_hand.npz output_soma_hand.npz \
  --target-backend soma
```

Native SOMA bone scales are optimized by default for body and hand targets.
Global scale remains fixed to the input value unless
`--optimize-global-scale` is explicitly supplied. The output is another SOMA
NPZ containing the target parameters plus `conversion_*` source metadata,
loss history, and per-identity bind-pose vertex error. Use
`--no-optimize-scale-params` to keep target scale parameters neutral.

## AMASS to SOMA

Convert one AMASS sequence or a directory tree of SMPL motion files:

```bash
python -m tools.convert_amass_to_soma \
  --input /path/to/sequence.npz \
  --output-npz out/soma.npz \
  --no-render
```

For batch conversion, use `--input-dir` and `--output-dir`. The exported NPZ
contains SOMA poses, root translations, joint names, reconstruction errors,
and identity parameters.

## MANO interoperability

The hand tools convert pose and identity parameters using user-supplied MANO
v1.2 files:

- `tools/hand/mano2soma.py`
- `tools/hand/soma2mano.py`

Both commands fail with an actionable model-path error when the licensed MANO
file is unavailable. See [SOMA Hand data assets](hand_data_assets.md) for the
topology correspondence and local asset setup.

## DCC integration

Procedural transform references and setup instructions are maintained with
their integrations:

- Blender: `tools/soma_procedural_blender/README.md`
- Maya: `tools/soma_procedural_maya/README.md`

The [procedural control format](procedural_control_format.md) documents the
shared sidecar schema used by the Python runtime and DCC implementations.
