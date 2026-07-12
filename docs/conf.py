"""
Sphinx configuration for the ct_tfpnp documentation.
"""

import sys
from datetime import date
from pathlib import Path

# Make the package importable without installing it
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── Project ────────────────────────────────────────────────────────────────

project = "ct_tfpnp"
author = "Emilia Zabrzanska"
copyright = f"{date.today().year}, {author}"

try:
    from ct_tfpnp import __version__ as release
except Exception:         
    release = "0.1.0"
version = release

# ── Extensions ─────────────────────────────────────────────────────────────

extensions = [
    "sphinx.ext.autodoc",         # pull docstrings out of the source
    "sphinx.ext.autosummary",     # per-module summary tables
    "sphinx.ext.napoleon",        # understand Google-style Args:/Returns: blocks
    "sphinx.ext.viewcode",        # "[source]" links next to every object
    "sphinx.ext.intersphinx",     # link to torch/numpy docs
    "sphinx.ext.mathjax",         # render the LaTeX in the docstrings
    "myst_parser",                # let the .md notes be included as pages
]

autosummary_generate = True

autodoc_mock_imports = [
    "LION",
    "tomosipo",
    "astra",
    "ts_algorithms",
    "piq",          
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,      # if it has no docstring, do not invent a stub for it
    "show-inheritance": True,
    "member-order": "bysource",  # source order reads better than alphabetical here
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ── HTML output ────────────────────────────────────────────────────────────

html_theme = "furo"
html_title = f"ct_tfpnp {release}"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#1F4E79",
        "color-brand-content": "#1F4E79",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8AB4F8",
        "color-brand-content": "#8AB4F8",
    },
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Markdown support, so FIDELITY.md and friends can be pulled in verbatim
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["deflist", "colon_fence", "dollarmath"]