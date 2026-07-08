"""Nonlinear and linear sample embeddings for exploratory analysis."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .optional import HAVE_UMAP, missing_packages

logger = logging.getLogger(__name__)

if HAVE_UMAP:
    import umap as umap_learn

SUPPORTED_EMBEDDINGS = ("pca", "umap", "tsne")


def resolve_embedding_methods(
    *,
    embedding_methods: Optional[Sequence[str]] = None,
    run_umap: bool = False,
    run_tsne: bool = False,
) -> List[str]:
    """Resolve the list of embedding methods to compute."""
    if embedding_methods is not None:
        methods = [m.lower() for m in embedding_methods]
    else:
        methods = ["pca"]
        if run_umap:
            methods.append("umap")
        if run_tsne:
            methods.append("tsne")

    normalized: List[str] = []
    for method in methods:
        if method not in SUPPORTED_EMBEDDINGS:
            raise ValueError(
                f"Unknown embedding method '{method}'. "
                f"Supported: {', '.join(SUPPORTED_EMBEDDINGS)}"
            )
        if method not in normalized:
            normalized.append(method)
    if "pca" not in normalized:
        normalized.insert(0, "pca")
    return normalized


def _scatter_embedding(
    Z: np.ndarray,
    y: pd.Series,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    savepath: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = y.astype(str)
    plt.figure(figsize=(7, 6))
    for lab in sorted(labels.unique()):
        mask = (labels == lab).values
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.8, label=str(lab))
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def compute_pca(
    X_scaled: np.ndarray,
    y: pd.Series,
    savepath: str,
    *,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA embedding with variance ratio metadata."""
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X_scaled)
    _scatter_embedding(
        Z,
        y,
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
        title="PCA (standardized features)",
        savepath=savepath,
    )
    return Z, pca.explained_variance_ratio_


def compute_umap(
    X_scaled: np.ndarray,
    y: pd.Series,
    savepath: str,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> Optional[np.ndarray]:
    """UMAP embedding when umap-learn is installed."""
    if not HAVE_UMAP:
        logger.warning(
            "UMAP requested but umap-learn is not installed. "
            "Install with: pip install mioXpektron[analysis]"
        )
        return None

    reducer = umap_learn.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    Z = reducer.fit_transform(X_scaled)
    _scatter_embedding(
        Z,
        y,
        xlabel="UMAP1",
        ylabel="UMAP2",
        title="UMAP (standardized features)",
        savepath=savepath,
    )
    return Z


def compute_tsne(
    X_scaled: np.ndarray,
    y: pd.Series,
    savepath: str,
    *,
    perplexity: float = 30.0,
    learning_rate: str | float = "auto",
    random_state: int = 0,
) -> np.ndarray:
    """t-SNE embedding via scikit-learn."""
    n_samples = X_scaled.shape[0]
    perp = min(perplexity, max(2.0, (n_samples - 1) / 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=learning_rate,
        random_state=random_state,
        init="pca",
    )
    Z = reducer.fit_transform(X_scaled)
    _scatter_embedding(
        Z,
        y,
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        title="t-SNE (standardized features)",
        savepath=savepath,
    )
    return Z


def run_embeddings(
    X_scaled: np.ndarray,
    y: pd.Series,
    outdir: str,
    *,
    methods: Optional[Sequence[str]] = None,
    run_umap: bool = False,
    run_tsne: bool = False,
    random_state: int = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Compute and save requested embeddings; return coordinate arrays."""
    import os

    os.makedirs(outdir, exist_ok=True)
    resolved = resolve_embedding_methods(
        embedding_methods=methods,
        run_umap=run_umap,
        run_tsne=run_tsne,
    )

    coords: Dict[str, np.ndarray] = {}
    for method in resolved:
        if method == "pca":
            Z, _ = compute_pca(
                X_scaled,
                y,
                os.path.join(outdir, "pca.png"),
                random_state=random_state,
            )
            coords["PCA1"] = Z[:, 0]
            coords["PCA2"] = Z[:, 1]
        elif method == "umap":
            Z = compute_umap(
                X_scaled,
                y,
                os.path.join(outdir, "umap.png"),
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=random_state,
            )
            if Z is not None:
                coords["UMAP1"] = Z[:, 0]
                coords["UMAP2"] = Z[:, 1]
        elif method == "tsne":
            Z = compute_tsne(
                X_scaled,
                y,
                os.path.join(outdir, "tsne.png"),
                perplexity=tsne_perplexity,
                random_state=random_state,
            )
            coords["TSNE1"] = Z[:, 0]
            coords["TSNE2"] = Z[:, 1]

    unavailable = missing_packages()
    for method in resolved:
        if method in unavailable:
            logger.info(
                "Optional package '%s' not installed for %s embedding.",
                unavailable[method],
                method,
            )

    return coords