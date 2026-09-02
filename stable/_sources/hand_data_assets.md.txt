# SOMA Hand data assets

SOMA-X v0.3 adds left- and right-hand layers whose meshes and 25-joint
skeletons are strict subsets of the full-body SOMA template.

## `SOMAHand.npz`

This file contains the hand-to-body topology maps, remapped faces, hand joint
indices, native hand identity PCA, and separate left/right articulation-pose
PCA models. `SOMAHandLayer` slices rig vertices, bind transforms, and skinning
weights from `SOMA_template_rig.usda`.

Supported LODs are:

| LOD | Vertices per hand |
|---|---:|
| `mid` | 2,859 |
| `low` | 718 |
| `xlo` | 134 |

The native identity model has 20 components. The left and right pose priors
each have 32 components over the 24 finger joints; the wrist is excluded so
global hand orientation remains independent of articulation.

Pose samples are bind-relative SO(3) exponential maps. Use
`tools/hand/sample_soma_hand_pose_pca.py` to load the prior, convert samples to
the absolute-local convention expected by the layer, pose the skinned mesh,
and optionally render MP4/GIF output.

```bash
python tools/hand/sample_soma_hand_pose_pca.py \
  --hand-type left --num-poses 60 --fps 2 \
  --output-prefix out/somahand_pose_pca_samples
```

## MANO interoperability

The checked-in `assets/MANO` OBJ pairs provide the public topology
correspondence used by the MANO identity backend and conversion tools. The
MANO model itself is not redistributed. Download MANO v1.2 under its own
license and pass `MANO_LEFT.pkl` or `MANO_RIGHT.pkl` explicitly, or place the
files in the documented local asset directory.

The public conversion entry points are:

- `tools/hand/mano2soma.py`
- `tools/hand/soma2mano.py`

Both fail with an actionable file-path error when the user-supplied MANO model
is absent.

## Usage limitations

The pose PCA is an articulation prior, not a gesture classifier, and may not
represent extreme or out-of-distribution hand poses. The wrist remains outside
the prior so applications can control global hand orientation independently.
