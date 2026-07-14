"""Sphinx configuration for mioXpektron documentation."""

import importlib.util
import os
import sys
from pathlib import Path

# Add the package root to sys.path so autodoc can find the modules.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_project_metadata():
    metadata_path = ROOT / "mioXpektron" / "_metadata.py"
    spec = importlib.util.spec_from_file_location("_mioxpektron_metadata", metadata_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load project metadata from {metadata_path}")
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)
    return metadata


_metadata = _load_project_metadata()

# -- Project information -----------------------------------------------------

project = _metadata.PROJECT_NAME
copyright = _metadata.COPYRIGHT
author = _metadata.AUTHOR
release = _metadata.VERSION

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# MyST-Parser settings (allows Markdown sources alongside RST)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "mioXpektron Documentation"

# -- Extension configuration -------------------------------------------------

# Napoleon: support Google and NumPy style docstrings
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Autosummary
autosummary_generate = True

# Intersphinx: link to external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
