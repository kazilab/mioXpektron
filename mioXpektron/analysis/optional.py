"""Optional analysis dependency detection."""

from __future__ import annotations

from typing import Dict

try:
    import umap  # noqa: F401

    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

try:
    import xgboost  # noqa: F401

    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

try:
    import shap  # noqa: F401

    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

# t-SNE and cNMF use scikit-learn (core dependency)
HAVE_TSNE = True
HAVE_CNMF = True


def analysis_capabilities() -> Dict[str, bool]:
    """Report which extended analysis features are available."""
    return {
        "pca": True,
        "umap": HAVE_UMAP,
        "tsne": HAVE_TSNE,
        "xgboost": HAVE_XGBOOST,
        "shap": HAVE_SHAP,
        "cnmf": HAVE_CNMF,
    }


def missing_packages() -> Dict[str, str]:
    """Map unavailable features to pip install hints."""
    hints: Dict[str, str] = {}
    if not HAVE_UMAP:
        hints["umap"] = "umap-learn"
    if not HAVE_XGBOOST:
        hints["xgboost"] = "xgboost"
    if not HAVE_SHAP:
        hints["shap"] = "shap"
    return hints