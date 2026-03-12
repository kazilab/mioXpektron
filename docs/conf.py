"""Sphinx configuration for mioXpektron documentation."""

import os
import sys

# Add the package root to sys.path so autodoc can find the modules.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "mioXpektron"
copyright = "2026, @kazilab.se"
author = "Data Analysis Team @KaziLab.se"
release = "0.0.2"

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
