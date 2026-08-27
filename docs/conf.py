# Configuration file for the Sphinx documentation builder.
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys
from pathlib import Path

# Make the soma package importable for autodoc.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# -- Project information ------------------------------------------------------

project = "SOMA-X"
author = "NVIDIA"
copyright = "2026, NVIDIA"
version = "0.2"
release = os.environ.get("SOMA_DOCS_RELEASE", "0.2.3")

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "autodocsumm",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

docs_audience = os.environ.get("SOMA_DOCS_AUDIENCE", "internal").lower()
if docs_audience not in {"internal", "public"}:
    raise ValueError("SOMA_DOCS_AUDIENCE must be 'internal' or 'public'")

if docs_audience == "public":
    root_doc = "index"
    exclude_patterns.extend(["internal", "internal/**", "internal_index.md"])
    extensions.remove("sphinx.ext.viewcode")
else:
    root_doc = "internal_index"

# Render `foo` (single-backticked text) as inline code in RST, so that
# method docstrings using Markdown-style backticks (e.g. `poses`) render
# as monospaced code rather than the default (italic / cross-ref).
default_role = "code"

# -- MyST ---------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

# -- Autodoc ------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": False,
    "exclude-members": "__weakref__",
}
# Render class + __init__ docstrings together: class docstring describes
# the class, __init__ docstring describes constructor parameters.
autoclass_content = "both"
autodoc_typehints = "description"
# Only inject a synthetic Parameters block when the param is actually
# documented. Prevents types-only Parameters dumps for functions with
# no Args:/Returns: body.
autodoc_typehints_description_target = "documented"
autodoc_preserve_defaults = True
autosummary_generate = False

# Napoleon parses Google-style Args:/Returns: blocks in docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False

# -- Intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "nvidia_sphinx_theme"
html_title = "SOMA-X"
html_baseurl = os.environ.get("SOMA_DOCS_BASE_URL", "https://nvlabs.github.io/SOMA-X/")
html_static_path = ["_static"]

# nvidia_sphinx_theme extends pydata_sphinx_theme; most option keys are
# inherited from there. See https://pypi.org/project/nvidia-sphinx-theme/.
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/SOMA-X",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "show_toc_level": 2,
    "navigation_depth": 2,
}

# Enable the version switcher for GitHub Pages builds. Public docs collapse
# patch releases into /vMAJOR.MINOR/ folders, matching NVIDIA Warp.
doc_version = os.environ.get("DOC_VERSION", "")
if doc_version:
    html_theme_options.update(
        {
            "check_switcher": False,
            "switcher": {
                "json_url": f"{html_baseurl.rstrip('/')}/versions.json",
                "version_match": doc_version,
            },
            "show_version_warning_banner": True,
        }
    )

    if re.fullmatch(r"\d+\.\d+", doc_version):
        version = doc_version
        release = doc_version


_PUBLIC_SKIP_MEMBERS = {
    "soma.geometry.batched_skinning.BatchedSkinning",
    "soma.io.export_soma_usd",
    "soma.io.save_soma_npz",
}


def _public_skip_member(app, what, name, obj, skip, options):
    if docs_audience != "public":
        return None
    obj_module = getattr(obj, "__module__", "")
    qualified_name = f"{obj_module}.{name}"
    if qualified_name in _PUBLIC_SKIP_MEMBERS:
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _public_skip_member)
