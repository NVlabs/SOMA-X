# Installation

## Install from PyPI

```bash
pip install py-soma-x
```

SOMA-X downloads its public runtime assets from
[`nvidia/SOMA-X`](https://huggingface.co/nvidia/SOMA-X) on first use. The
download is cached by `huggingface_hub`; pass `data_root` to a layer when an
explicit local asset directory is preferred.

Install optional identity backends with extras:

```bash
pip install "py-soma-x[smpl]"
pip install "py-soma-x[anny]"
pip install "py-soma-x[demo]"
```

## Separately licensed model files

SOMA-X does not redistribute SMPL, SMPL-X, or MANO model files. Download the
models from their official sources under the applicable license and pass the
local path explicitly.

```python
from soma import SOMALayer

body = SOMALayer(
    identity_model_type="smpl",
    identity_model_kwargs={"model_path": "/path/to/SMPL_NEUTRAL.pkl"},
)
```

The hand conversion tools accept `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` from
MANO v1.2. See [SOMA Hand data assets](hand_data_assets.md) for the expected
paths and interoperability contract.

The `smplx` dependency requires `chumpy`, whose PyPI package needs build
isolation disabled:

```bash
pip install --no-build-isolation chumpy
```

If that package build is unavailable, install the compatible source revision:

```bash
pip install --no-build-isolation \
  git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa8
```

## Developer installation

The source repository uses Git LFS for runtime and test assets.

```bash
git lfs install
git clone https://github.com/NVlabs/SOMA-X.git
cd SOMA-X
git lfs pull
```

Create an environment and install the development dependencies:

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\activate` after
`uv venv .venv`. Select the PyTorch wheel index that matches the installed GPU
driver and CUDA runtime.

Run the test suite with:

```bash
pytest tests -v
```

## Optional body identity assets

### GarmentMeasurements

Clone the upstream
[GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements)
repository and convert its PCA file into the local SOMA-X asset layout:

```bash
python tools/convert_gm_pca_to_npz.py \
  /path/to/GarmentMeasurements/data/pca/point.pca \
  assets/GarmentMeasurements/point.npz
```

### Anny

```bash
uv pip install -e ".[anny]"
```

### SMPL family

```bash
uv pip install -e ".[smpl]"
pip install --no-build-isolation chumpy
```

Keep licensed model files outside version control. Pass their paths through
the identity backend configuration rather than copying them into a release or
Hugging Face payload.
