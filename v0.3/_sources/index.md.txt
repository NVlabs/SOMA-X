# SOMA-X Documentation

SOMA-X provides canonical full-body and wrist-local hand layers, model
interoperability, pose inversion, and shared geometry utilities. Start with
the installation and tools guides, then use the model and API references for
the exact runtime contracts.

All identity models below are driven by SOMA's unified body and hand
skeletons.

![SOMA Body and SOMA Hand identity backends animated by the unified skeleton](../assets/images/soma-in-action.gif)

```{toctree}
:hidden:
:caption: Getting Started

installation
tools
```

```{toctree}
:hidden:
:caption: Models and Data

data_assets
hand_data_assets
procedural_control_format
```

```{toctree}
:hidden:
:caption: API Reference

api/index
api/somalayer
api/somahandlayer
api/pose_inversion
api/io
api/geometry
```

```{toctree}
:hidden:
:caption: Project

changelog
model_card
BIAS
EXPLAINABILITY
PRIVACY
SAFETY_and_SECURITY
```

## Local preview

Install the docs dependencies:

```bash
uv pip install -e ".[docs]"
```

Build and serve locally:

```bash
SOMA_DOCS_AUDIENCE=public DOC_VERSION=0.3 sphinx-build -b html docs docs/_build/html
python -m http.server -d docs/_build/html
```

Then open `http://127.0.0.1:8000/`.
