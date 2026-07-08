"""Visualization helpers for downstream statistical analysis."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .embeddings import compute_pca, compute_tsne, compute_umap
from .optional import HAVE_UMAP

# Backward-compatible alias
_HAVE_UMAP = HAVE_UMAP


def plot_volcano(
    res: pd.DataFrame,
    savepath: str,
    *,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    q_thresh: float = 0.05,
    fc_thresh: float = 1.0,
) -> None:
    """Volcano plot of log2 fold-change versus -log10(p-value)."""
    if group_a is None and "group_a" in res.columns:
        group_a = str(res["group_a"].iloc[0])
    if group_b is None and "group_b" in res.columns:
        group_b = str(res["group_b"].iloc[0])
    xlab = "log2 Fold Change"
    if group_a and group_b:
        xlab = f"log2 Fold Change ({group_a} / {group_b})"

    x = res["log2_FC"].values
    y = -np.log10(res["p_value"].values + 1e-300)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=16, alpha=0.7)
    plt.axvline(fc_thresh, linestyle="--")
    plt.axvline(-fc_thresh, linestyle="--")
    sig = res.loc[res["q_value"] <= q_thresh, "p_value"]
    if not sig.empty:
        p_proxy = sig.max()
        if isinstance(p_proxy, float) and np.isfinite(p_proxy) and p_proxy > 0:
            plt.axhline(-math.log10(p_proxy), linestyle="--")
    plt.xlabel(xlab)
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano plot")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_pca(
    X_scaled: np.ndarray,
    y: pd.Series,
    savepath: str,
    *,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCA scatter plot coloured by group labels."""
    return compute_pca(X_scaled, y, savepath, random_state=random_state)


def plot_umap(
    X_scaled: np.ndarray,
    y: pd.Series,
    savepath: str,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> Optional[np.ndarray]:
    """UMAP embedding plot when umap-learn is installed."""
    return compute_umap(
        X_scaled,
        y,
        savepath,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )


def plot_tsne(
    X_scaled: np.ndarray,
    y: pd.Series,
    savepath: str,
    *,
    perplexity: float = 30.0,
    random_state: int = 0,
) -> np.ndarray:
    """t-SNE scatter plot coloured by group labels."""
    return compute_tsne(
        X_scaled,
        y,
        savepath,
        perplexity=perplexity,
        random_state=random_state,
    )


def plot_heatmap_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    res: pd.DataFrame,
    savepath: str,
    *,
    top_n: int = 25,
    label_col: str = "Group",
) -> None:
    """Heatmap of top differential features (z-scored), samples ordered by group."""
    top_feats = res.sort_values("q_value", ascending=True).head(top_n)["feature"].tolist()
    X_sel = X[top_feats].copy()
    X_z = (X_sel - X_sel.mean(axis=0)) / (X_sel.std(axis=0) + 1e-12)
    labels = y.astype(str)
    order = np.argsort(labels.values)
    X_ord = X_z.values[order, :]
    y_ord = labels.values[order]

    plt.figure(figsize=(max(6, top_n * 0.25), 6))
    plt.imshow(X_ord.T, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(top_feats)), top_feats)
    unique_labels, counts = np.unique(y_ord, return_counts=True)
    boundary = counts[0] if len(counts) > 1 else None
    if boundary is not None and boundary < X_ord.shape[0]:
        plt.axvline(boundary - 0.5)
    plt.xlabel(f"Samples (ordered by {label_col})")
    plt.ylabel("Top features (z-scored)")
    plt.title("Heatmap of top differential features")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()