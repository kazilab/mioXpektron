#!/usr/bin/env python3
"""
Comprehensive analysis for Cancer vs Control ToF-SIMS-like intensity tables.

Input CSV requirements
----------------------
- Columns:
    - 'SampleName'   : sample ID (string)
    - 'Group'        : class label ('Cancer' or 'Control')
    - Remaining cols : numeric features (m/z intensities); header names are m/z values
- File should already be imputed and non-negative if you enable cNMF.

What this script produces
-------------------------
In the output directory (default: ./analysis_outputs), it writes:
- label_counts.csv
- univariate_results.csv  (Welch t-test per feature, log2 fold-change, BH-FDR q-values)
- volcano.png
- pca.png
- (optional) umap.png      -- if --umap flag set and umap-learn is installed
- roc_logistic.png, roc_random_forest.png
- model_performance.csv
- importance_lr_l1.png, importance_rf.png
- heatmap_top{N}.png       -- heatmap of top-N features by FDR
- embeddings.csv           -- PCA (and UMAP if requested)
- If --cnmf is provided:
    - cnmf_summary_k{K}.txt
    - cnmf_consensus_k{K}.npy
    - cnmf_PAC_vs_k.csv
    - cnmf_consensus_best.png
    - cnmf_W_best.npy, cnmf_H_best.npy
    - cnmf_factor_{j}_top_features.csv  (top m/z contributors per factor)
    - cnmf_factor_{j}_bar.png           (bar plot of top contributors)

Usage
-----
python analyze_breast_spectra.py \
    --input aligned_peaks_intensity_breast_new_imputed_rf.csv \
    --outdir analysis_outputs \
    --topn 25 \
    --umap \
    --cnmf --k_list 3 4 5 6 7 --cnmf_reps 30 --cnmf_beta KL

Notes
-----
- Welch t-tests (unequal variances) + Benjamini–Hochberg FDR control.
- PCA on log1p-standardized intensities.
- Classifiers: Logistic Regression (L1, saga) and Random Forest.
- cNMF implements multiple NMF runs per k, aligns factors (Hungarian matching),
  builds a consensus co-clustering matrix, computes PAC stability, and selects k.

References (methods; general, not version-specific)
---------------------------------------------------
- Welch’s t-test: Welch, 1947; BH-FDR: Benjamini & Hochberg, 1995.
- PCA: Pearson, 1901; Hotelling, 1933.
- Logistic regression & L1 regularization: Tibshirani, 1996 (Lasso).
- Random Forest: Breiman, 2001.
- UMAP: McInnes et al., 2018.
- NMF (MU updates): Lee & Seung, 2001; cNMF: Brunet et al., 2004; survey in Berry et al., 2007.

Author’s note
-------------
- Where I recommend KL loss for count-like data, that is a common practice in
  mass-spec intensity modeling, consistent with Poisson-like noise assumptions
  and NMF literature (opinion grounded in cited works above).
"""

import logging
import os
import math
import argparse
import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Optional
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import umap  # optional
    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False

# ----------------------------- I/O --------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------- basic statistics --------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    # Compute adjusted p-values: p[order] * n / rank
    adjusted = p[order] * n / ranks
    # Enforce monotonicity from the bottom up via cumulative minimum on reversed array
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    # Place back into original order
    q = np.empty_like(p, dtype=float)
    q[order] = adjusted
    return np.clip(q, 0, 1)

def compute_univariate_tests(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Welch t-test per feature and log2 fold-change (Cancer / Control)."""
    y = y.astype(str).str.strip().str.capitalize()
    mask_c = (y == "Cancer").values
    mask_n = (y == "Control").values

    means_c = X[mask_c].mean(axis=0).values
    means_n = X[mask_n].mean(axis=0).values

    eps = 1e-12
    log2_fc = np.log2((means_c + eps) / (means_n + eps))

    # Vectorized Welch t-test across all features at once
    Xc = X[mask_c].values  # (n_cancer, n_features)
    Xn = X[mask_n].values  # (n_control, n_features)
    nc, nn = Xc.shape[0], Xn.shape[0]
    mean_c = np.nanmean(Xc, axis=0)
    mean_n = np.nanmean(Xn, axis=0)
    var_c = np.nanvar(Xc, axis=0, ddof=1)
    var_n = np.nanvar(Xn, axis=0, ddof=1)
    se = np.sqrt(var_c / nc + var_n / nn)
    t_stat = np.where(se > 0, (mean_c - mean_n) / se, 0.0)
    # Welch-Satterthwaite degrees of freedom
    num = (var_c / nc + var_n / nn) ** 2
    denom = (var_c / nc) ** 2 / (nc - 1) + (var_n / nn) ** 2 / (nn - 1)
    df_welch = np.where(denom > 0, num / denom, 1.0)
    pvals = 2.0 * stats.t.sf(np.abs(t_stat), df_welch)
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    qvals = bh_fdr(pvals)

    res = pd.DataFrame({
        "feature": X.columns,
        "mean_Cancer": means_c,
        "mean_Control": means_n,
        "log2_FC": log2_fc,
        "p_value": pvals,
        "q_value": qvals
    }).sort_values("q_value", ascending=True).reset_index(drop=True)
    return res

# --------------------------- plots --------------------------------

def plot_volcano(res: pd.DataFrame, outpath: str, q_thresh: float = 0.05, fc_thresh: float = 1.0):
    x = res["log2_FC"].values
    y = -np.log10(res["p_value"].values + 1e-300)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=16, alpha=0.7)
    plt.axvline(fc_thresh, linestyle="--")
    plt.axvline(-fc_thresh, linestyle="--")
    p_proxy = res.loc[res["q_value"] <= q_thresh, "p_value"].max()
    if isinstance(p_proxy, float) and np.isfinite(p_proxy) and p_proxy > 0:
        plt.axhline(-math.log10(p_proxy), linestyle="--")
    plt.xlabel("log2 Fold Change (Cancer / Control)")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano plot")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_pca(X_scaled: np.ndarray, y: pd.Series, outpath: str) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_scaled)
    y = y.astype(str)
    plt.figure(figsize=(7, 6))
    for lab in sorted(y.unique()):
        mask = (y == lab).values
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.8, label=str(lab))
    plt.legend()
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA (standardized features)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return Z, pca.explained_variance_ratio_

def plot_umap(X_scaled: np.ndarray, y: pd.Series, outpath: str, n_neighbors: int = 15, min_dist: float = 0.1):
    if not _HAVE_UMAP:
        return None
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    Z = reducer.fit_transform(X_scaled)
    y = y.astype(str)
    plt.figure(figsize=(7, 6))
    for lab in sorted(y.unique()):
        mask = (y == lab).values
        plt.scatter(Z[mask, 0], Z[mask, 1], s=20, alpha=0.8, label=str(lab))
    plt.legend()
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP (standardized features)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return Z

def plot_heatmap_top_features(X: pd.DataFrame, y: pd.Series, res: pd.DataFrame, outpath: str, top_n: int = 25):
    top_feats = res.sort_values("q_value", ascending=True).head(top_n)["feature"].tolist()
    X_sel = X[top_feats].copy()
    X_z = (X_sel - X_sel.mean(axis=0)) / (X_sel.std(axis=0) + 1e-12)
    X_mat = X_z.values
    order = np.argsort(y.values.astype(str))
    X_ord = X_mat[order, :]
    y_ord = y.values.astype(str)[order]

    plt.figure(figsize=(max(6, top_n * 0.25), 6))
    plt.imshow(X_ord.T, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(top_feats)), top_feats)
    # boundary between groups
    unique_labels, counts = np.unique(y_ord, return_counts=True)
    boundary = counts[0] if len(counts) > 1 else None
    if boundary is not None and boundary < X_ord.shape[0]:
        plt.axvline(boundary - 0.5)
    plt.xlabel("Samples (ordered by Group)")
    plt.ylabel("Top features (z-scored)")
    plt.title("Heatmap of top differential features")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -------------------------- models --------------------------------

def run_models(X_scaled: np.ndarray, y01: np.ndarray, features: List[str], outdir: str, seed: int = 0):
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y01, test_size=0.2, stratify=y01, random_state=seed)

    # Logistic Regression with L1
    lr = LogisticRegression(penalty="l1", solver="saga", max_iter=2000, C=1.0, n_jobs=-1, random_state=seed)
    lr.fit(X_tr, y_tr)
    y_proba_lr = lr.predict_proba(X_te)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_te, y_proba_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_lr, tpr_lr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - Logistic Regression (AUC={auc_lr:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_logistic.png"), dpi=200)
    plt.close()

    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_proba_rf = rf.predict_proba(X_te)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_te, y_proba_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_rf, tpr_rf, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - Random Forest (AUC={auc_rf:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_random_forest.png"), dpi=200)
    plt.close()

    # Cross-validated AUCs
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_cv_lr = cross_val_score(lr, X_scaled, y01, cv=cv, scoring="roc_auc", n_jobs=-1)
    auc_cv_rf = cross_val_score(rf, X_scaled, y01, cv=cv, scoring="roc_auc", n_jobs=-1)

    metrics = pd.DataFrame({
        "model": ["Logistic_L1", "RandomForest"],
        "holdout_AUC": [auc_lr, auc_rf],
        "cv_AUC_mean": [auc_cv_lr.mean(), auc_cv_rf.mean()],
        "cv_AUC_std": [auc_cv_lr.std(), auc_cv_rf.std()],
    })
    metrics.to_csv(os.path.join(outdir, "model_performance.csv"), index=False)

    # Feature importances
    coef = np.abs(lr.coef_.ravel())
    top_idx_lr = np.argsort(coef)[::-1][:15]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(top_idx_lr)), coef[top_idx_lr])
    plt.xticks(range(len(top_idx_lr)), [features[i] for i in top_idx_lr], rotation=90)
    plt.title("Top LR (L1) Feature Coefficients (abs)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "importance_lr_l1.png"), dpi=200)
    plt.close()

    imp = rf.feature_importances_
    top_idx_rf = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(top_idx_rf)), imp[top_idx_rf])
    plt.xticks(range(len(top_idx_rf)), [features[i] for i in top_idx_rf], rotation=90)
    plt.title("Top Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "importance_rf.png"), dpi=200)
    plt.close()

# -------------------------- cNMF ----------------------------------

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

def _cosine_similarity_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = normalize(A, axis=0) + 0.0
    B_n = normalize(B, axis=0) + 0.0
    return A_n.T @ B_n

def _align_components(W_list: List[np.ndarray], H_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    W0, H0 = W_list[0], H_list[0]
    aligned_W, aligned_H = [W0], [H0]
    ref = H0.copy()
    for r in range(1, len(H_list)):
        sim = _cosine_similarity_columns(ref.T, H_list[r].T)  # (k x k)
        row_ind, col_ind = linear_sum_assignment(1.0 - sim)   # maximize similarity
        W_r = W_list[r][:, col_ind]
        H_r = H_list[r][col_ind, :]
        aligned_W.append(W_r)
        aligned_H.append(H_r)
    return aligned_W, aligned_H

def _consensus_matrix(W: np.ndarray) -> np.ndarray:
    labels = np.argmax(W, axis=1)
    # Vectorized: samples with the same label get co-occurrence = 1
    C = (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(float)
    return C

def _pac_score(consensus: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> float:
    vals = consensus[np.triu_indices_from(consensus, k=1)]
    return np.mean((vals > lower) & (vals < upper)) if vals.size else 1.0

def run_cnmf(
    X_pos: np.ndarray,
    k_list: List[int],
    R: int = 30,
    max_iter: int = 1000,
    beta: str = "frobenius",   # "frobenius", "kullback-leibler"
    random_seeds: Optional[List[int]] = None,
    outdir: Optional[str] = None
) -> Dict[int, Dict[str, object]]:
    """
    Consensus NMF across k values.
    Returns dict[k] with W_mean, H_mean, consensus, PAC, W_list, H_list.
    """
    n, p = X_pos.shape
    if random_seeds is None:
        random_seeds = list(range(R))
    elif len(random_seeds) < R:
        random_seeds = (random_seeds * ((R + len(random_seeds) - 1) // len(random_seeds)))[:R]

    results: Dict[int, Dict[str, object]] = {}
    for k in k_list:
        W_runs, H_runs = [], []
        for r in range(R):
            model = NMF(
                n_components=k,
                init="nndsvda",
                random_state=random_seeds[r],
                max_iter=max_iter,
                solver=("mu" if beta != "frobenius" else "cd"),
                beta_loss=("kullback-leibler" if beta == "KL" or beta == "kullback-leibler" else "frobenius"),
            )
            W = model.fit_transform(X_pos)
            H = model.components_
            W_runs.append(W)
            H_runs.append(H)

        W_aligned, H_aligned = _align_components(W_runs, H_runs)

        # consensus
        C_sum = np.zeros((n, n), dtype=float)
        for W_r in W_aligned:
            C_sum += _consensus_matrix(W_r)
        consensus = C_sum / R
        pac = _pac_score(consensus)

        W_mean = np.mean(np.stack(W_aligned, axis=2), axis=2)
        H_mean = np.mean(np.stack(H_aligned, axis=2), axis=2)

        results[k] = dict(
            W_mean=W_mean,
            H_mean=H_mean,
            consensus=consensus,
            PAC=float(pac),
            W_list=W_aligned,
            H_list=H_aligned
        )

        if outdir:
            with open(os.path.join(outdir, f"cnmf_summary_k{k}.txt"), "w") as f:
                f.write(f"k={k}\nPAC={pac:.6f}\n")

            np.save(os.path.join(outdir, f"cnmf_consensus_k{k}.npy"), consensus)

    return results

def choose_k_by_pac(results: Dict[int, Dict[str, object]]) -> int:
    ks = sorted(results.keys())
    pacs = [(k, float(results[k]["PAC"])) for k in ks]
    pacs.sort(key=lambda t: (t[1], t[0]))
    return pacs[0][0]

def save_consensus_heatmap(consensus: np.ndarray, labels: pd.Series, outpath: str):
    order = np.argsort(labels.values.astype(str))
    C_ord = consensus[order][:, order]
    plt.figure(figsize=(6, 5))
    plt.imshow(C_ord, aspect="auto", interpolation="nearest")
    plt.title("Consensus matrix (samples ordered by Group)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_factor_bars(H: np.ndarray, feature_names: List[str], outdir: str, topm: int = 15):
    k = H.shape[0]
    for j in range(k):
        row = H[j, :]
        idx = np.argsort(row)[::-1][:topm]
        feats = [feature_names[i] for i in idx]
        vals = row[idx]
        pd.DataFrame({"feature": feats, "loading": vals}).to_csv(
            os.path.join(outdir, f"cnmf_factor_{j+1}_top_features.csv"), index=False
        )
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(idx)), vals)
        plt.xticks(range(len(idx)), feats, rotation=90)
        plt.title(f"cNMF factor {j+1}: top {topm} features")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cnmf_factor_{j+1}_bar.png"), dpi=200)
        plt.close()

# ---------------------------- main -------------------------------

def main(input_file, outdir, topn=25, umap=False, cnmf=False, k_list=None, cnmf_reps=30, cnmf_beta="frobenius"):

    # Load data
    df = pd.read_csv(input_file)
    # Basic columns
    if "Group" not in df.columns or "SampleName" not in df.columns:
        raise ValueError("Input must contain 'SampleName' and 'Group' columns.")
    meta_cols = ["SampleName", "Group"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    # Coerce numeric features
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_sr = df["Group"].astype(str).str.strip().str.capitalize()
    # Save label counts
    y_sr.value_counts().to_frame("count").to_csv(os.path.join(outdir, "label_counts.csv"))

    # --- Univariate tests ---
    uni = compute_univariate_tests(X_df, y_sr)
    uni.to_csv(os.path.join(outdir, "univariate_results.csv"), index=False)
    plot_volcano(uni, os.path.join(outdir, "volcano.png"))

    # --- PCA / UMAP embeddings ---
    X_log = np.log1p(X_df.values)                 # variance stabilization (keeps non-negativity)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)        # standardized for PCA/UMAP
    Z_pca, var_ratio = plot_pca(X_scaled, y_sr, os.path.join(outdir, "pca.png"))

    coords = pd.DataFrame({
        "SampleName": df["SampleName"],
        "Group": y_sr,
        "PCA1": Z_pca[:, 0],
        "PCA2": Z_pca[:, 1],
    })

    if umap and _HAVE_UMAP:
        Z_umap = plot_umap(X_scaled, y_sr, os.path.join(outdir, "umap.png"))
        if Z_umap is not None:
            coords["UMAP1"] = Z_umap[:, 0]
            coords["UMAP2"] = Z_umap[:, 1]
    coords.to_csv(os.path.join(outdir, "embeddings.csv"), index=False)

    # --- Supervised models ---
    y01 = (y_sr == "Cancer").astype(int).values
    run_models(X_scaled, y01, feature_cols, outdir=outdir, seed=0)

    # --- Heatmap of top features ---
    plot_heatmap_top_features(X_df, y_sr, uni, os.path.join(outdir, f"heatmap_top{topn}.png"), top_n=topn)

    # --- cNMF (optional) ---
    if cnmf:
        # Check non-negativity (NMF requirement)
        if (X_df.values < 0).any():
            raise ValueError("cNMF requires non-negative features. Found negative values in the input.")
        results = run_cnmf(
            X_pos=X_df.values.astype(float, copy=False),
            k_list=k_list,
            R=cnmf_reps,
            max_iter=1000,
            beta=cnmf_beta,
            outdir=outdir
        )
        # Save PAC over k
        pac_table = pd.DataFrame({"k": sorted(results.keys()),
                                  "PAC": [results[k]["PAC"] for k in sorted(results.keys())]})
        pac_table.to_csv(os.path.join(outdir, "cnmf_PAC_vs_k.csv"), index=False)

        best_k = min(results.keys(), key=lambda kk: (results[kk]["PAC"], kk))
        W = results[best_k]["W_mean"]
        H = results[best_k]["H_mean"]
        np.save(os.path.join(outdir, "cnmf_W_best.npy"), W)
        np.save(os.path.join(outdir, "cnmf_H_best.npy"), H)

        # Consensus heatmap (ordered by label)
        save_consensus_heatmap(results[best_k]["consensus"], y_sr, os.path.join(outdir, "cnmf_consensus_best.png"))

        # Save factor bars and top features per factor
        save_factor_bars(H, feature_cols, outdir, topm=15)

        with open(os.path.join(outdir, f"cnmf_summary_k{best_k}.txt"), "a") as f:
            f.write("\nSelected as best k by minimal PAC.\n")

    logger.info(f"Done. Outputs written to: {outdir}")