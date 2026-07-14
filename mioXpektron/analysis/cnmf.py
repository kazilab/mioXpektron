"""Consensus non-negative matrix factorisation for sample clustering."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize


def _cosine_similarity_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = normalize(A, axis=0)
    B_n = normalize(B, axis=0)
    return A_n.T @ B_n


def _align_components(
    W_list: List[np.ndarray],
    H_list: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    W0, H0 = W_list[0], H_list[0]
    aligned_W, aligned_H = [W0], [H0]
    ref = H0.copy()
    for run_idx in range(1, len(H_list)):
        sim = _cosine_similarity_columns(ref.T, H_list[run_idx].T)
        _, col_ind = linear_sum_assignment(1.0 - sim)
        aligned_W.append(W_list[run_idx][:, col_ind])
        aligned_H.append(H_list[run_idx][col_ind, :])
    return aligned_W, aligned_H


def _consensus_matrix(W: np.ndarray) -> np.ndarray:
    labels = np.argmax(W, axis=1)
    return (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(float)


def _pac_score(consensus: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> float:
    vals = consensus[np.triu_indices_from(consensus, k=1)]
    return float(np.mean((vals > lower) & (vals < upper))) if vals.size else 1.0


def run_cnmf(
    X_pos: np.ndarray,
    k_list: List[int],
    *,
    R: int = 30,
    max_iter: int = 1000,
    beta: str = "frobenius",
    random_seeds: Optional[List[int]] = None,
    outdir: Optional[str] = None,
) -> Dict[int, Dict[str, object]]:
    """Run consensus NMF across multiple rank values."""
    if (X_pos < 0).any():
        raise ValueError("cNMF requires non-negative features.")

    n_samples, _ = X_pos.shape
    if random_seeds is None:
        random_seeds = list(range(R))
    elif len(random_seeds) < R:
        repeats = (R + len(random_seeds) - 1) // len(random_seeds)
        random_seeds = (random_seeds * repeats)[:R]

    use_kl = beta.lower() in {"kl", "kullback-leibler"}
    results: Dict[int, Dict[str, object]] = {}

    for k in k_list:
        W_runs: List[np.ndarray] = []
        H_runs: List[np.ndarray] = []
        for seed in random_seeds:
            model = NMF(
                n_components=k,
                init="nndsvda",
                random_state=seed,
                max_iter=max_iter,
                solver="mu" if use_kl else "cd",
                beta_loss="kullback-leibler" if use_kl else "frobenius",
            )
            W_runs.append(model.fit_transform(X_pos))
            H_runs.append(model.components_)

        W_aligned, H_aligned = _align_components(W_runs, H_runs)
        consensus = np.zeros((n_samples, n_samples), dtype=float)
        for W_run in W_aligned:
            consensus += _consensus_matrix(W_run)
        consensus /= len(W_aligned)
        pac = _pac_score(consensus)

        results[k] = {
            "W_mean": np.mean(np.stack(W_aligned, axis=2), axis=2),
            "H_mean": np.mean(np.stack(H_aligned, axis=2), axis=2),
            "consensus": consensus,
            "PAC": pac,
            "W_list": W_aligned,
            "H_list": H_aligned,
        }

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"cnmf_summary_k{k}.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"k={k}\nPAC={pac:.6f}\n")
            np.save(os.path.join(outdir, f"cnmf_consensus_k{k}.npy"), consensus)

    return results


def plot_pac_vs_k(
    results: Dict[int, Dict[str, object]],
    savepath: str,
) -> None:
    """Plot PAC stability scores across candidate rank values."""
    ks = sorted(results.keys())
    pacs = [float(results[k]["PAC"]) for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, pacs, marker="o")
    plt.xlabel("k (number of factors)")
    plt.ylabel("PAC score")
    plt.title("cNMF stability (lower PAC is better)")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def choose_k_by_pac(results: Dict[int, Dict[str, object]]) -> int:
    """Select the rank with the lowest PAC score."""
    return min(results.keys(), key=lambda k: (float(results[k]["PAC"]), k))


def save_consensus_heatmap(
    consensus: np.ndarray,
    labels: pd.Series,
    savepath: str,
    *,
    label_col: str = "Group",
) -> None:
    """Plot a consensus matrix ordered by group labels."""
    order = np.argsort(labels.astype(str).values)
    C_ord = consensus[order][:, order]
    plt.figure(figsize=(6, 5))
    plt.imshow(C_ord, aspect="auto", interpolation="nearest")
    plt.title(f"Consensus matrix (samples ordered by {label_col})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def save_factor_bars(
    H: np.ndarray,
    feature_names: List[str],
    outdir: str,
    *,
    topm: int = 15,
) -> None:
    """Save per-factor top m/z contributors as CSV and bar plots."""
    os.makedirs(outdir, exist_ok=True)
    for j in range(H.shape[0]):
        row = H[j, :]
        idx = np.argsort(row)[::-1][:topm]
        feats = [feature_names[i] for i in idx]
        vals = row[idx]
        pd.DataFrame({"feature": feats, "loading": vals}).to_csv(
            os.path.join(outdir, f"cnmf_factor_{j + 1}_top_features.csv"),
            index=False,
        )
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(idx)), vals)
        plt.xticks(range(len(idx)), feats, rotation=90)
        plt.title(f"cNMF factor {j + 1}: top {topm} features")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cnmf_factor_{j + 1}_bar.png"), dpi=200)
        plt.close()