<p align="center">
  <img src="./assets/images/banner.png" alt="Banner" width="100%">
</p>

[![PyPI version](https://badge.fury.io/py/py-soma-x.svg)](https://badge.fury.io/py/py-soma-x)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Technical Report](https://img.shields.io/badge/arXiv-2603.16858-b31b1b.svg)](https://arxiv.org/abs/2603.16858)

**[Documentation](https://nvlabs.github.io/SOMA-X/stable/)** | [PyPI](https://pypi.org/project/py-soma-x/) | [Changelog](CHANGELOG.md)

## Overview

Parametric human body models, including SMPL, SMPL-X, MHR, Anny, and GarmentMeasurements, are central to a wide range of tasks in human reconstruction, animation, and simulation. However, these models are inherently incompatible: each defines its own mesh topology, joint hierarchy, and parameterization, precluding seamless integration. As a result, leveraging complementary strengths across models (such as combining Anny’s age-range control with SMPL-based motion data) necessitates bespoke adapters for every model pair, hindering interoperability and limiting practical applications.

We present **SOMA**—a canonical body topology and rig that acts as a universal pivot for all supported parametric human body models. Instead of replacing existing models, **SOMA unifies them** by mapping their diverse rest shapes onto a single, shared representation. This approach allows any supported identity model to be animated with a unified animation pipeline, eliminating the need for custom adapters or model-specific retargeting. With SOMA, you can mix and match identity sources and pose data at inference time without additional engineering. The entire pipeline remains end-to-end differentiable and GPU-accelerated via NVIDIA Warp. 


See SOMA in action: 

<p align="center">
  <img src="assets/images/soma-in-action.gif" alt="SOMA in Action" width="1000"/>
</p>

## Supported Identity Models

SOMA currently supports five distinct identity models, each offering unique capabilities:

1. [MHR](https://github.com/facebookresearch/MHR): The default identity model in SOMA, providing high-fidelity body shape representation.
2. [Anny](https://github.com/naver/anny): Particularly well-suited for modeling children, broadening applicability to younger subjects.
3. [SMPL-Family](https://smpl.is.tue.mpg.de/): Supports both SMPL and SMPL-X models, enabling interoperability with established standards in the field.
4. **SOMA-shape**: A proprietary PCA-based model developed as part of this project, designed to offer SMPL-like functionality with 128 PCA coefficients for identity representation. 
5. [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements): A PCA-based identity model trained on the CAESAR dataset, suitable for specialized use cases involving garment fitting and measurement.

We welcome community contributions to extend support for additional identity models.

## Unified Pose Correctives (Beta)
Thanks to SOMA's unified framework, pose-dependent corrective deformations that mitigate LBS artifacts are seamlessly available for all supported identity models, including those that do not provide correctives themselves (e.g., Anny and GarmentMeasurements).
<p align="center">
  <img src="assets/images/soma_correctives.gif" alt="SOMA Pose Correctives" width="800"/>
</p>

## Related projects that already support SOMA
SOMA is part of a larger effort to enable human animation, robotics, physical AI, and other applications. We also provide the following works with SOMA support: 

* [GEM](https://github.com/NVlabs/GEM-X) - SOMA-based video pose estimation. 
* [Kimodo](https://github.com/nv-tlabs/kimodo) - SOMA-based controllable text-to-motion generation method for **human(oid)s**.  
* [BONES-SEED Dataset](https://huggingface.co/datasets/bones-studio/seed) - a large scale human(oid) motion capture dataset in SOMA format. Also provides retargeted G1 data.  
* [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) - for SOMA to G1 retargeting. 
* [ProtoMotion](https://github.com/NVlabs/ProtoMotions) - simulation and learning framework for training physically simulated digital human(oid)s
* [GEAR SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) - a humanoid behavior foundation model. (coming soon)

## Installation

### Install from PyPI

The easiest way to install SOMA-X is from [PyPI](https://pypi.org/project/py-soma-x/):

```bash
pip install py-soma-x
```

With optional extras:
```bash
pip install "py-soma-x[smpl]"   # SMPL/SMPL-X support
pip install "py-soma-x[anny]"   # Anny support
```

Assets are automatically downloaded from HuggingFace on first use (cached in `~/.cache/huggingface/hub/`).

> **Note:** SMPL/SMPL-X requires `chumpy`, which must be installed separately:
> ```bash
> pip install --no-build-isolation chumpy
> ```
> If that fails, install from source:
> ```bash
> pip install --no-build-isolation git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa8
> ```
>
> SMPL/SMPL-X model files (`SMPL_NEUTRAL.pkl`, `SMPLX_NEUTRAL.npz`) require a separate license and must be downloaded from [SMPL](https://smpl.is.tue.mpg.de/) / [SMPL-X](https://smpl-x.is.tue.mpg.de/). Pass the model path explicitly:
> ```python
> soma = SOMALayer(
>     identity_model_type="smpl",
>     identity_model_kwargs={"model_path": "/path/to/SMPL_NEUTRAL.pkl"},
> )
> ```

<details>

<summary>Developer installation (clone with Git LFS)</summary>

### Clone with Git LFS (Required for Assets)
This repository uses Git LFS for large asset files (e.g., assets/Nova_neutral.npz). You must install Git LFS to download the actual data; otherwise, you will encounter file loading errors.

1. Install Git LFS (if not installed):
````bash
git lfs install
````

2. Clone and Pull Data:
```bash
git clone https://github.com/NVlabs/SOMA-X.git
cd SOMA-X
git lfs pull
```
_(If you already cloned the repo, just run `git lfs pull` to fetch the missing assets.)_

### Prepare Python environment

**Linux:**
```bash
pip install uv
uv venv .venv
source .venv/bin/activate   # or: . .venv/bin/activate
# Install PyTorch with CUDA — adjust the version (cu124, cu126, cu130, …)
# to match your GPU and driver. See https://pytorch.org/get-started/locally/
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install ".[dev]"
```

**Windows (PowerShell):**
```powershell
pip install uv
uv venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   # one-time setup
.\.venv\Scripts\activate
# Install PyTorch with CUDA — adjust the version (cu124, cu126, cu130, …)
# to match your GPU and driver. See https://pytorch.org/get-started/locally/
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install ".[dev]"
```
Then run tests: `pytest tests/ -v`.  

### Optional Dependencies

**For SMPL and SMPLX support:**

```bash
uv pip install ".[smpl]"
pip install --no-build-isolation chumpy
```
_NOTE: `chumpy` (required by `smplx` at runtime) has a broken PyPI build and must be installed with `--no-build-isolation`. If that fails, install from source: `pip install --no-build-isolation git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa8`_

You also need to download `SMPL_NEUTRAL.pkl` or `SMPLX_NEUTRAL.npz` separately:
1. Visit the [SMPL](https://smpl.is.tue.mpg.de/) or [SMPLX](https://smpl-x.is.tue.mpg.de/) website. 
2. Register and download the SMPL (v1.1.0 for Python) or [SMPL-X](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip) (with removed head bun) model files. 
3. Extract and copy `SMPL_NEUTRAL.pkl` to `./assets/SMPL/SMPL_NEUTRAL.pkl` and `SMPLX_NEUTRAL.npz` to `./assets/SMPLX/SMPLX_NEUTRAL.npz`. 

**Note:** The SMPL models are subject to a separate license and cannot be redistributed with this repository.

**For [Anny](https://github.com/naver/anny) support:**
```bash
uv pip install ".[anny]"
```

**For [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements) support:**
```bash
git clone https://github.com/mbotsch/GarmentMeasurements 
python tools/convert_gm_pca_to_npz.py ./GarmentMeasurements/data/pca/point.pca assets/GarmentMeasurements/point.npz
rm -rf GarmentMeasurements 
```
</details>


## Usage

### Full-body model

```python
import torch
from soma import SOMALayer

# Initialize the layer — assets are auto-downloaded from HuggingFace
soma = SOMALayer(
    identity_model_type="mhr",  # or "soma" "smpl", "smplx", "anny", "garment"
    device="cuda"
)

# Or use a local assets directory
# soma = SOMALayer(data_root="./assets", identity_model_type="mhr", device="cuda")

# Forward pass
# poses: (B, num_joints, 3)
# identity: (B, num_coeffs)
# scale_params: (B, soma.num_scale_params) - Optional, depending on model type.
# For native SOMA bone scales, soma.scale_param_names defines active child joints,
# and soma.scale_param_segments defines the matching local parent-to-child edges.
# The final four native SOMA controls scale the left/right lower legs and feet.
# Legacy 56-value native SOMA tensors remain accepted with unit foot scales.
output = soma(poses, identity, scale_params=scale_params)
vertices = output["vertices"]
```

## Running the Demo

Install the demo environment (includes pyrender, tqdm, imageio with ffmpeg for video output):

```bash
uv pip install ".[demo]"
```

If you want to run all identity models (soma, mhr, anny, smpl, smplx, garment), install the full set and use the same build steps as for tests:

```bash
uv pip install -e ".[all,demo]"
pip install --no-build-isolation chumpy
```

Then run the demo script:

```bash
# Run all models (default: soma, mhr, anny, smpl, smplx, garment)
python tools/demo_soma_vis.py --data-root ./assets --output-dir ./out

# Run specific models only
python tools/demo_soma_vis.py --data-root ./assets --output-dir ./out --identity-model-type soma, mhr, smplx

# Run a single model
python tools/demo_soma_vis.py --data-root ./assets --output-dir ./out --identity-model-type anny

# Run MHR with random shapes
python tools/demo_soma_vis.py --data-root ./assets --output-dir ./out --identity-model-type mhr --random-shape

# Render the extra-low LOD
python tools/demo_soma_vis.py --data-root ./assets --output-dir ./out --identity-model-type soma,mhr --lod xlo
```

This will generate example animation videos for the selected models in the `out/` directory.

**Full-body demo options:**
- `--identity-model-type`: Comma-separated list of models (options: `soma`, `mhr`, `anny`, `smplx`, `smpl`, `garment`, default: all)
- `--lod`: Body mesh LOD to render (`mid`, `low`, or `xlo`; default: `mid`). The `xlo` mesh uses its own USD topology/skinning and uses the v0027 low-LOD mesh for identity skeleton fitting.
- `--random-shape`: Smoothly animate random body shapes instead of neutral shape
- `--motion-file`: Path to custom motion file (default: `assets/ROM5.npy`)
- `--image-size`: Render resolution (default: 1920)
- `--device`: Device to use (default: `cuda:0`)

## Conversion of pose parameters from other models to SOMA
We provide conversion tools for converting from SMPL and MHR pose parameters to SOMA.
Both tools use `PoseInversion.fit()`, which supports two complementary solvers — both initialized by a single-pass skeleton transfer fit for fast convergence:

- **Analytical** (default): iterative inverse-LBS with Newton-Schulz refinement. Extremely fast (~1200 FPS) with comparable accuracy.
- **Autograd FK**: 6D rotation optimization by backpropagating FK + LBS. Slow but controllable (e.g. extra weights on extremities).

The two can be combined: the analytical solve warm-starts autograd refinement — best of both worlds.

### SMPL to SOMA

<img src="assets/images/smpl2soma.gif" alt="SMPL to SOMA conversion" width="400"/>

```bash
# Convert SMPL animation to SOMA (renders comparison video)
python -m tools.smpl2soma

# Export SOMA poses as .npz
python -m tools.smpl2soma --output-npz out/smpl_soma.npz

# Tune analytical iterations (defaults: --body-iters 2 --full-iters 1)
python -m tools.smpl2soma --body-iters 3 --full-iters 1 --batch-size 64

# Analytical + autograd FK refinement (best accuracy)
python -m tools.smpl2soma --body-iters 2 --full-iters 1 --autograd-iters 10
```

**Benchmark** (402 SMPL frames, RTX 5000 Ada):

| Method | Speed | Mean | Median | Max |
|---|---|---|---|---|
| Analytical (body=2, full=1) — **default** | **1279 FPS** | 0.65 cm | 0.52 cm | 17.8 cm |
| Autograd FK (10 iters, lr=5e-3) | 199 FPS | 1.04 cm | 0.97 cm | 18.1 cm |
| Autograd FK (100 iters) | 18 FPS | 0.49 cm | 0.39 cm | 16.8 cm |

### MHR to SOMA

<img src="assets/images/mhr2soma.gif" alt="MHR to SOMA conversion" width="400"/>

For [SAM 3D Body](https://huggingface.co/datasets/facebook/sam-3d-body-dataset) or similar MHR-format data.

```bash
# Convert a directory of SAM 3D Body parquet files
python -m tools.mhr2soma --input path/to/sam_3d_body/data/coco_train

# Convert and export as .npz
python -m tools.mhr2soma --input path/to/parquet_dir --output-npz out/mhr_soma.npz

# Tune analytical iterations (defaults: --body-iters 2 --full-iters 1)
python -m tools.mhr2soma --input path/to/parquet_dir --max-samples 100 --body-iters 3

# Analytical + autograd FK refinement (best accuracy)
python -m tools.mhr2soma --input path/to/parquet_dir --autograd-iters 10
```

**Benchmark** (200 SAM 3D Body samples, RTX 5000 Ada):

| Method | Speed | Mean | Median | Max |
|---|---|---|---|---|
| Analytical (body=2, full=1) — **default** | **342 FPS** | 0.61 cm | 0.34 cm | 14.8 cm |
| Autograd FK (10 iters, lr=5e-3) | 161 FPS | 1.05 cm | 0.76 cm | 13.5 cm |
| Autograd FK (100 iters) | 16 FPS | 0.48 cm | 0.22 cm | 13.3 cm |

> **Note:** The `mhr2soma` tool's end-to-end throughput (~50 samp/s) is dominated by MHR identity model evaluation, not SOMA inversion. The MHR TorchScript model is called twice per sample (once to produce the rest shape, once for posed vertices). The SOMA inversion itself runs at 342 FPS.

### AMASS dataset to SOMA

Convert [AMASS](https://amass.is.tue.mpg.de/) motion sequences (SMPL format `.npz` files) to SOMA.

> **Prerequisites:** Download the AMASS dataset from [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/) and place `SMPL_NEUTRAL.pkl` in `assets/SMPL/` (see SMPL installation above).

```bash
# Single file — converts and renders a comparison video
python -m tools.convert_amass_to_soma --input path/to/amass_sequence.npz

# Single file — export .npz only (skip rendering)
python -m tools.convert_amass_to_soma --input path/to/amass_sequence.npz --output-npz out/soma.npz --no-render

# Batch convert entire dataset (mirrors folder structure)
python -m tools.convert_amass_to_soma --input-dir /data/amass/ --output-dir out/amass_soma/

# Shuffle file order (useful when running multiple workers in parallel)
python -m tools.convert_amass_to_soma --input-dir /data/amass/ --output-dir out/amass_soma/ --shuffle

# Tune analytical iterations
python -m tools.convert_amass_to_soma --input path/to/seq.npz --body-iters 3 --full-iters 1

# Analytical + autograd FK refinement (best accuracy)
python -m tools.convert_amass_to_soma --input path/to/seq.npz --autograd-iters 10
```

The output `.npz` files contain:
- `poses`: `(N, J, 3)` rotation vectors per joint
- `root_translation`: `(N, 3)` root position in meters
- `joint_names`: list of SOMA joint names
- `per_vertex_error`: `(N, V)` reconstruction error per vertex
- `identity_coeffs` / `scale_params`: identity parameters used


**Benchmark** (A100):

| Method | Speed | Mean | Median | Max |
|---|---|---|---|---|
| Analytical (body=2, full=1) — **default** | **17393 FPS** | 0.53 cm | 0.32 cm | 8.8 cm |
| Autograd FK (10 iters, lr=5e-3) | 435 FPS | 0.78 cm | 0.64 cm | 8.8 cm |





## Citation
If you use this code in your work, please cite:

```bibtex
@article{soma2026,
  title={SOMA: Unifying Parametric Human Body Models},
  author={Jun Saito and Jiefeng Li and Michael de Ruyter and Miguel Guerrero and Edy Lim and Ehsan Hassani and Roger Blanco Ribera and Hyejin Moon and Magdalena Dadela and Marco Di Lucca and Qiao Wang and Xueting Li and Jan Kautz and Simon Yuen and Umar Iqbal},
  eprint={2603.16858},
  archivePrefix={arXiv},
  year={2026},
  url={https://arxiv.org/abs/2603.16858},
}
```

## Acknowledgements 
- [SMPL-Body](https://smpl.is.tue.mpg.de/bodylicense.html) was used to create an interpolator between SMPL and SOMA mesh topologies, courtesy of the Max Planck Institute for Intelligent Systems. 
- [MHR](https://github.com/facebookresearch/MHR) was used to learn the pose corrective model. 
- [Anny](https://github.com/naver/anny) for [WARP](https://github.com/NVIDIA/warp)-based sparse linear blend skinning.  
- [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements) was used to augment the data in our shape model.


## License

This codebase is licensed under [Apache-2.0](LICENSE). 

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
