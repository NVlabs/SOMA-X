<p align="center">
  <img src="./assets/images/banner.png" alt="SOMA-X" width="100%">
</p>

[![PyPI version](https://badge.fury.io/py/py-soma-x.svg)](https://badge.fury.io/py/py-soma-x)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Technical Report](https://img.shields.io/badge/arXiv-2603.16858-b31b1b.svg)](https://arxiv.org/abs/2603.16858)

**[Documentation](https://nvlabs.github.io/SOMA-X/stable/)** ·
**[PyPI](https://pypi.org/project/py-soma-x/)** ·
**[Hugging Face](https://huggingface.co/nvidia/SOMA-X)** ·
**[Changelog](CHANGELOG.md)**

## Overview

SOMA-X provides a canonical, differentiable representation for parametric
human bodies and hands. It maps supported identity models to shared SOMA
topology and rig conventions so applications can reuse one animation,
retargeting, and geometry pipeline across model families. Runtime skinning and
fitting are accelerated with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The package includes:

- `SOMALayer` for full-body models at mid, low, and extra-low LODs.
- `SOMAHandLayer` for wrist-local left and right hands at the same three LODs.
- Native SOMA body and hand identity models, plus optional MHR, Anny,
  SMPL-family, GarmentMeasurements, and MANO interoperability.
- Pose inversion, procedural rig controls, smoothing, conversion, USD/NPZ I/O,
  and reusable geometry utilities.

All identity models below are driven by SOMA's unified body and hand
skeletons.

<p align="center">
  <img src="assets/images/soma-in-action.gif" alt="SOMA Body and SOMA Hand identity backends animated by the unified skeleton, with the skeleton overlaid" width="720">
</p>

## Installation

```bash
pip install py-soma-x
```

Assets are downloaded from
[nvidia/SOMA-X](https://huggingface.co/nvidia/SOMA-X) on first use and cached by
`huggingface_hub`. Optional identity backends have additional dependencies and
some require separately licensed model files. See the
[installation guide](docs/installation.md) for source installation, extras,
and model-file setup.

## Quick start

### Full body

```python
import torch

from soma import SOMALayer

body = SOMALayer(identity_model_type="mhr", device="cpu")
poses = torch.zeros(1, 77, 3)
identity = torch.zeros(1, body.num_shape_components)

output = body(poses, identity)
vertices = output.vertices
joints = output.joints
```

### SOMA Hand

```python
import torch

from soma import SOMAHandLayer

hand = SOMAHandLayer(hand_type="right", lod="mid", device="cpu")
poses = torch.zeros(1, 25, 3)
identity = torch.zeros(1, hand.num_shape_components)

output = hand(poses, identity)
vertices = output.vertices
joints = output.joints
```

`SOMAHand.npz` includes the native hand identity model and bind-relative
articulation prior. MANO interoperability uses user-supplied MANO v1.2 files;
licensed MANO models are not redistributed.

## Supported identity models

| Scope | Backend | Notes |
|---|---|---|
| Body | MHR | Default high-fidelity body backend |
| Body | SOMA | Native PCA identity and body-part scaling |
| Body | Anny | Anthropometric controls with broad age coverage |
| Body | SMPL / SMPL-H / SMPL-X | User-supplied licensed model files |
| Body | GarmentMeasurements | User-generated local PCA asset |
| Hand | SOMA | Native 20-component hand identity model |
| Hand | MANO | User-supplied MANO v1.2 model files |
| Hand | MHR | Hand identity sliced from the MHR body model |

## Documentation

- [Installation and optional backends](docs/installation.md)
- [Demos, conversion tools, and pose sampling](docs/tools.md)
- [Full-body API](docs/api/somalayer.rst)
- [SOMA Hand API](docs/api/somahandlayer.rst)
- [Body data assets](docs/data_assets.md)
- [SOMA Hand data assets](docs/hand_data_assets.md)
- [Pose inversion](docs/api/pose_inversion.rst)
- [Procedural control format](docs/procedural_control_format.md)

## Related projects

- [GEM-X](https://github.com/NVlabs/GEM-X) — SOMA-based video pose estimation.
- [Kimodo](https://github.com/nv-tlabs/kimodo) — controllable text-to-motion generation. 
- [ARDY](https://research.nvidia.com/labs/sil/projects/ardy/) — an autoregressive diffusion model designed for interactive motion generation. 
- [MotionBricks](https://nvlabs.github.io/motionbricks/) — a real-time motion in-betweener. 
- [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) — human and humanoid motion dataset in SOMA format.
- [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) — SOMA-to-Humanoid retargeting.
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) — physically simulated character learning.
- [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl) — a state-of-the-art whole-body controller.

## Citation

If you use SOMA-X in your work, please cite:

```bibtex
@article{soma2026,
  title={SOMA: Unifying Parametric Human Body Models},
  author={Jun Saito and Jiefeng Li and Michael de Ruyter and Miguel Guerrero and Edy Lim and Ehsan Hassani and Roger Blanco Ribera and Hyejin Moon and Magdalena Dadela and Marco Di Lucca and Qiao Wang and Xueting Li and Sam Wu and Chaeyeon Chung and Yeongho Seol and Jan Kautz and Simon Yuen and Umar Iqbal},
  eprint={2603.16858},
  archivePrefix={arXiv},
  year={2026},
  url={https://arxiv.org/abs/2603.16858},
}
```

## Acknowledgements

- [SMPL-Body](https://smpl.is.tue.mpg.de/bodylicense.html) was used to create
  the SMPL-to-SOMA topology correspondence, courtesy of the Max Planck
  Institute for Intelligent Systems.
- [MHR](https://github.com/facebookresearch/MHR) was used to learn the pose
  corrective model.
- [Anny](https://github.com/naver/anny) provided the basis for Warp-accelerated
  sparse linear blend skinning.
- [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements) was
  used to augment the native body shape model.

## License

SOMA-X is licensed under [Apache-2.0](LICENSE). Optional third-party models and
dependencies retain their own license terms.
