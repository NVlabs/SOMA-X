# SOMA-X Documentation

This site is API-first and centered on the `soma` library. The main pages cover:

- `SOMALayer` for the full-body model
- `PoseInversion` for inverse pose fitting
- `soma.io` helpers for NPZ and USD workflows
- `soma.geometry` advanced building blocks (FK, LBS, skeleton fitting, Warp kernels)

The tracked project documents under `docs/` are also included in the navigation so
API docs and project documentation live in one site.

## Local preview

Install the docs dependencies:

```bash
uv pip install -e ".[docs]"
```

Build and serve locally:

```bash
SOMA_DOCS_AUDIENCE=public DOC_VERSION=0.2 sphinx-build -b html docs docs/_build/html
python -m http.server -d docs/_build/html
```

Then open `http://127.0.0.1:8000/`.

```{toctree}
:hidden:
:caption: API

api/index
api/somalayer
api/pose_inversion
api/io
api/geometry
```

```{toctree}
:hidden:
:caption: Data

data_assets
procedural_control_format
```

```{toctree}
:hidden:
:caption: Project Docs

changelog
model_card
BIAS
EXPLAINABILITY
PRIVACY
SAFETY_and_SECURITY
```
