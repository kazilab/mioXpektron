"""
Normalization method evaluation for ToF-SIMS data.

Evaluates multiple normalization strategies on a set of labelled spectra
using unsupervised, supervised and spectral-quality metrics, then ranks
them with composite scores — following the approach established in
`xpectrass` for FTIR data but adapted to the specifics of ToF-SIMS
(Poisson counting statistics, high dynamic range, ion-yield variation).

Usage
-----
>>> from mioXpektron.normalization import NormalizationEvaluator
>>> evaluator = NormalizationEvaluator(files=["spectra/*.txt"])
>>> summary = evaluator.evaluate()
>>> evaluator.plot()
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    _POLARS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MPL_AVAILABLE = False

from .normalization import normalize, normalization_method_names, tic_normalization
from ..utils.file_management import import_data

OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def spectral_angle(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Spectral Angle Mapper (SAM) in radians; lower => more similar shape."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(cos))


def within_group_mean_sam(X: np.ndarray, groups: np.ndarray) -> float:
    """Mean SAM across all pairs within each group (technical replicates)."""
    groups = np.asarray(groups)
    vals: List[float] = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < 2:
            continue
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                vals.append(spectral_angle(X[idx[i]], X[idx[j]]))
    return float(np.mean(vals)) if vals else np.nan


def _zscore_robust(series: pd.Series) -> pd.Series:
    """Z-score with epsilon to avoid division by zero."""
    return (series - series.mean()) / (series.std(ddof=0) + 1e-12)


def _cv_of_tic(X: np.ndarray) -> float:
    """Coefficient of variation of row-wise TIC — lower => better normalisation."""
    tics = np.nansum(X, axis=1)
    if np.nanmean(tics) == 0:
        return np.nan
    return float(np.nanstd(tics) / np.nanmean(tics))


def _mean_pairwise_correlation(X: np.ndarray, groups: np.ndarray) -> float:
    """Mean within-group Pearson correlation — higher => better consistency."""
    groups = np.asarray(groups)
    vals: List[float] = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < 2:
            continue
        for i in range(idx.size):
            for j in range(i + 1, idx.size):
                r = np.corrcoef(X[idx[i]], X[idx[j]])[0, 1]
                if np.isfinite(r):
                    vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def _dynamic_range(X: np.ndarray) -> float:
    """Median dynamic range (log10(max/min of positive values)) across samples."""
    drs: List[float] = []
    for row in X:
        pos = row[row > 0]
        if pos.size > 1:
            drs.append(np.log10(np.max(pos) / np.min(pos)))
    return float(np.median(drs)) if drs else np.nan


def _negative_fraction(X: np.ndarray) -> float:
    """Fraction of values < 0 across the entire matrix."""
    return float(np.sum(X < 0) / X.size) if X.size > 0 else 0.0


def _has_glob_chars(s: str) -> bool:
    return any(ch in s for ch in "*?[")


# ---------------------------------------------------------------------------
#  Core single-method evaluation
# ---------------------------------------------------------------------------

def evaluate_one_method(
    X_raw: np.ndarray,
    groups: np.ndarray,
    mz_values: np.ndarray,
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    n_clusters: Optional[int] = None,
    cluster_bootstrap_rounds: int = 30,
    cluster_bootstrap_frac: float = 0.8,
    random_state: int = 0,
    compute_supervised: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single normalisation method on the spectra matrix.

    Parameters
    ----------
    X_raw : np.ndarray
        (n_samples, n_channels) raw intensity matrix.
    groups : np.ndarray
        (n_samples,) label per sample.
    mz_values : np.ndarray
        (n_channels,) m/z axis shared by all spectra.
    method : str
        Normalization method name.
    method_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`normalize`.
    n_clusters : int, optional
        Number of clusters.  Defaults to number of unique groups.
    cluster_bootstrap_rounds : int
        Bootstrap rounds for cluster stability.
    cluster_bootstrap_frac : float
        Fraction of samples per bootstrap round.
    random_state : int
        RNG seed.
    compute_supervised : bool
        If True and scikit-learn is available, run supervised CV.

    Returns
    -------
    dict
        Keys include ``method``, all metric values, and ``compute_time_sec``.
    """
    method_kwargs = method_kwargs or {}
    n_samples, n_channels = X_raw.shape
    n_clusters_actual = n_clusters or int(np.unique(groups).size)

    # ---------- normalise ----------
    t0 = time.perf_counter()

    # Methods that need a dataset-level reference
    ref_needed = method in ("pqn", "median_of_ratios", "pareto", "mass_stratified_pqn")
    if ref_needed:
        if method == "pqn":
            reference = np.nanmedian(X_raw, axis=0)
            method_kwargs = {**method_kwargs, "reference": reference}
        elif method == "median_of_ratios":
            # Geometric mean across samples (add pseudo-count for zeros)
            log_means = np.nanmean(np.log(X_raw + 1.0), axis=0)
            reference = np.exp(log_means) - 1.0
            method_kwargs = {**method_kwargs, "reference": reference}
        elif method == "pareto":
            mean = np.nanmean(X_raw, axis=0)
            std = np.nanstd(X_raw, axis=0, ddof=0)
            method_kwargs = {**method_kwargs, "mean": mean, "std": std}
        elif method == "mass_stratified_pqn":
            reference = np.nanmedian(X_raw, axis=0)
            method_kwargs = {
                **method_kwargs,
                "reference": reference,
                "mz_values": mz_values,
            }

    X_norm = np.empty_like(X_raw, dtype=float)
    for i in range(n_samples):
        X_norm[i] = normalize(X_raw[i], method=method, **method_kwargs)

    compute_time = time.perf_counter() - t0

    # ---------- unsupervised metrics ----------
    cv_tic = _cv_of_tic(X_norm)
    within_sam = within_group_mean_sam(X_norm, groups)
    within_corr = _mean_pairwise_correlation(X_norm, groups)
    dyn_range = _dynamic_range(X_norm)
    neg_frac = _negative_fraction(X_norm)

    # Clustering evaluation
    ari_km = nmi_km = cluster_stability = sil = np.nan
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            silhouette_score,
        )

        km = KMeans(n_clusters=n_clusters_actual, n_init="auto",
                     random_state=random_state)
        lab_ref = km.fit_predict(X_norm)
        ari_km = adjusted_rand_score(groups, lab_ref)
        nmi_km = normalized_mutual_info_score(groups, lab_ref)

        if n_samples > n_clusters_actual:
            try:
                sil = float(silhouette_score(X_norm, lab_ref, metric="cosine"))
            except Exception:
                pass

        # Stability
        rng = np.random.default_rng(random_state)
        stab: List[float] = []
        m = min(n_samples, max(2, int(round(cluster_bootstrap_frac * n_samples))))
        for _ in range(cluster_bootstrap_rounds):
            idx = rng.choice(n_samples, size=m, replace=False)
            km_sub = KMeans(n_clusters=n_clusters_actual, n_init="auto",
                            random_state=int(rng.integers(0, 2**31 - 1)))
            lab_sub = km_sub.fit_predict(X_norm[idx])
            stab.append(adjusted_rand_score(lab_ref[idx], lab_sub))
        cluster_stability = float(np.mean(stab))

    except ImportError:
        logger.debug("scikit-learn not available; skipping clustering metrics.")

    # ---------- supervised metrics ----------
    sup_f1 = sup_bal = np.nan
    if compute_supervised:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import f1_score, balanced_accuracy_score

            unique_classes = np.unique(groups)
            if len(unique_classes) >= 2:
                n_splits = min(5, min(Counter(groups).values()))
                if n_splits >= 2:
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                          random_state=random_state)
                    f1s, bals = [], []
                    for tr, te in skf.split(X_norm, groups):
                        clf = LogisticRegression(max_iter=2000, n_jobs=None)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(X_norm[tr], groups[tr])
                        pred = clf.predict(X_norm[te])
                        f1s.append(f1_score(groups[te], pred, average="macro",
                                            zero_division=0))
                        bals.append(balanced_accuracy_score(groups[te], pred))
                    sup_f1 = float(np.mean(f1s))
                    sup_bal = float(np.mean(bals))
        except ImportError:
            logger.debug("scikit-learn not available; skipping supervised metrics.")

    return {
        "method": method,
        "cv_tic": cv_tic,
        "within_group_SAM": within_sam,
        "within_group_corr": within_corr,
        "dynamic_range": dyn_range,
        "negative_fraction": neg_frac,
        "cluster_ARI": float(ari_km),
        "cluster_NMI": float(nmi_km),
        "cluster_stability": float(cluster_stability),
        "silhouette_cosine": float(sil),
        "supervised_macro_f1": sup_f1,
        "supervised_bal_acc": sup_bal,
        "compute_time_sec": compute_time,
    }


# ---------------------------------------------------------------------------
#  Evaluator class (mirrors BaselineMethodEvaluator pattern)
# ---------------------------------------------------------------------------

@dataclass
class NormalizationEvaluator:
    """Evaluate normalization methods on labelled ToF-SIMS spectra.

    Parameters
    ----------
    files : list of str or Path
        Paths or glob patterns expanding to spectrum text files.
    methods : list of str, optional
        Normalization method names.  Defaults to a sensible subset.
    method_kwargs_map : dict, optional
        ``{method_name: {kwarg: value, ...}}`` for method-specific params.
    mz_min, mz_max : float, optional
        m/z range to import.
    n_clusters : int, optional
        Number of clusters for KMeans evaluation.  Auto-detected if omitted.
    cluster_bootstrap_rounds : int
        Bootstrap rounds for stability metric.
    random_state : int
        RNG seed for reproducibility.
    compute_supervised : bool
        Run supervised classification (requires scikit-learn + >=2 groups).
    n_jobs : int
        Parallel workers (joblib).  ``-1`` = all CPUs, ``1`` = sequential.

    Examples
    --------
    >>> evaluator = NormalizationEvaluator(files=["data/*.txt"])
    >>> summary = evaluator.evaluate()
    >>> evaluator.plot()
    """

    files: List[Union[str, Path]] = field(default_factory=list)
    methods: Optional[List[str]] = None
    method_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None
    mz_min: Optional[float] = None
    mz_max: Optional[float] = None
    n_clusters: Optional[int] = None
    cluster_bootstrap_rounds: int = 30
    cluster_bootstrap_frac: float = 0.8
    random_state: int = 0
    compute_supervised: bool = True
    n_jobs: int = -1
    group_patterns: Optional[Dict[str, str]] = None
    group_fn: Optional[Any] = None  # Callable[[str], str]

    # internal state
    _resolved_files: List[Path] = field(default_factory=list, init=False, repr=False)
    _X_raw: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _groups: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _mz: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _results: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._resolved_files = self._expand_files(self.files)
        if not self._resolved_files:
            raise FileNotFoundError(
                "No input files found; verify the provided file paths or "
                "glob patterns."
            )
        if self.methods is None:
            # Sensible defaults for ToF-SIMS
            self.methods = [
                "tic", "median", "rms", "max", "vector",
                "poisson", "sqrt", "log", "vsn", "minmax",
            ]
        self.method_kwargs_map = self.method_kwargs_map or {}
        self.files = self._resolved_files  # type: ignore[assignment]

    # -- file resolution ---------------------------------------------------

    @staticmethod
    def _expand_files(candidates) -> List[Path]:
        paths: List[Path] = []
        for item in candidates:
            s = str(item).strip()
            if not s:
                continue
            if _has_glob_chars(s):
                for hit in sorted(Path().glob(s)):
                    if hit.is_file():
                        paths.append(hit.resolve())
                continue
            p = Path(s)
            if p.is_file():
                paths.append(p.resolve())
            elif p.is_dir():
                for pattern in ("*.txt", "*.csv"):
                    for hit in sorted(p.rglob(pattern)):
                        if hit.is_file():
                            paths.append(hit.resolve())
        return sorted(set(paths))

    # -- data loading ------------------------------------------------------

    def _load_spectra(self):
        """Read all spectrum files into a (samples x channels) matrix."""
        spectra: List[Tuple[str, np.ndarray]] = []
        groups: List[str] = []
        ref_mz: Optional[np.ndarray] = None

        for fp in self._resolved_files:
            try:
                mz, intensity, sample_name, group = import_data(
                    str(fp), self.mz_min, self.mz_max,
                    group_patterns=self.group_patterns,
                    group_fn=self.group_fn,
                )
                if ref_mz is None:
                    ref_mz = mz
                elif len(mz) != len(ref_mz):
                    logger.warning(
                        "Skipping %s: channel count %d != %d",
                        fp.name, len(mz), len(ref_mz),
                    )
                    continue
                elif not np.allclose(mz, ref_mz, atol=1e-6):
                    logger.warning(
                        "Skipping %s: m/z grid mismatch (max delta %.4e)",
                        fp.name, float(np.max(np.abs(mz - ref_mz))),
                    )
                    continue
                spectra.append((sample_name, intensity))
                groups.append(group)
            except Exception as e:
                logger.error("Error loading %s: %s", fp.name, e)

        if not spectra:
            raise RuntimeError("No spectra could be loaded.")

        self._X_raw = np.vstack([s[1] for s in spectra])
        self._groups = np.array(groups)
        self._mz = ref_mz
        self._sample_names = [s[0] for s in spectra]
        logger.info(
            "Loaded %d spectra (%d channels) with groups: %s",
            len(spectra), ref_mz.shape[0],
            dict(Counter(groups)),
        )

    # -- evaluation --------------------------------------------------------

    def evaluate(self) -> pd.DataFrame:
        """Evaluate all methods and return a scored DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per method, sorted by ``score_combined`` (descending).
            Includes raw metrics, z-scored metrics, and four composite scores.
        """
        if self._X_raw is None:
            self._load_spectra()

        methods = self.methods or normalization_method_names()
        rows: List[Dict[str, Any]] = []

        try:
            from joblib import Parallel, delayed

            def _eval(m):
                return evaluate_one_method(
                    self._X_raw, self._groups, self._mz,
                    method=m,
                    method_kwargs=self.method_kwargs_map.get(m, {}),
                    n_clusters=self.n_clusters,
                    cluster_bootstrap_rounds=self.cluster_bootstrap_rounds,
                    cluster_bootstrap_frac=self.cluster_bootstrap_frac,
                    random_state=self.random_state,
                    compute_supervised=self.compute_supervised,
                )

            if self.n_jobs == 1:
                try:
                    from tqdm import tqdm
                    it = tqdm(methods, desc="Evaluating normalization")
                except ImportError:
                    it = methods
                rows = [_eval(m) for m in it]
            else:
                try:
                    from tqdm import tqdm
                    rows = Parallel(n_jobs=self.n_jobs, backend="loky")(
                        delayed(_eval)(m)
                        for m in tqdm(methods, desc="Evaluating normalization")
                    )
                except ImportError:
                    rows = Parallel(n_jobs=self.n_jobs, backend="loky")(
                        delayed(_eval)(m) for m in methods
                    )
        except ImportError:
            # No joblib — sequential fallback
            for m in methods:
                rows.append(evaluate_one_method(
                    self._X_raw, self._groups, self._mz,
                    method=m,
                    method_kwargs=self.method_kwargs_map.get(m, {}),
                    n_clusters=self.n_clusters,
                    cluster_bootstrap_rounds=self.cluster_bootstrap_rounds,
                    cluster_bootstrap_frac=self.cluster_bootstrap_frac,
                    random_state=self.random_state,
                    compute_supervised=self.compute_supervised,
                ))

        res = pd.DataFrame(rows)
        res = self._compute_scores(res)
        self._results = res
        return res

    @staticmethod
    def _compute_scores(res: pd.DataFrame) -> pd.DataFrame:
        """Add z-scored metrics and composite scores to the results table.

        Scoring follows the xpectrass convention: all z-scores are oriented
        so that *higher = better*, then combined into weighted composites.
        """
        # --- Z-scores (higher = better for all) ---

        # Lower is better → negate before z-scoring
        res["z_cv_tic"] = _zscore_robust(-res["cv_tic"])
        res["z_sam"] = _zscore_robust(-res["within_group_SAM"])
        res["z_neg_frac"] = _zscore_robust(-res["negative_fraction"])
        res["z_compute_time"] = _zscore_robust(-res["compute_time_sec"])

        # Higher is better
        res["z_corr"] = _zscore_robust(res["within_group_corr"])
        res["z_dyn_range"] = _zscore_robust(res["dynamic_range"])
        res["z_ari"] = _zscore_robust(res["cluster_ARI"])
        res["z_nmi"] = _zscore_robust(res["cluster_NMI"])
        res["z_stability"] = _zscore_robust(res["cluster_stability"])
        res["z_sil"] = _zscore_robust(res["silhouette_cosine"])
        res["z_f1"] = _zscore_robust(res["supervised_macro_f1"])
        res["z_bal_acc"] = _zscore_robust(res["supervised_bal_acc"])

        # --- Composite scores ---

        # Check whether supervised metrics are available (not all NaN)
        has_supervised = (
            res["supervised_macro_f1"].notna().any()
            and res["supervised_bal_acc"].notna().any()
        )

        # Replace NaN z-scores with 0 so they don't poison weighted sums
        def _z(col: str) -> pd.Series:
            return res[col].fillna(0.0)

        if has_supervised:
            # 1. Combined (balanced)
            #    40% spectral quality, 30% clustering, 20% supervised, 10% practical
            res["score_combined"] = (
                0.15 * _z("z_cv_tic") +
                0.10 * _z("z_sam") +
                0.10 * _z("z_corr") +
                0.05 * _z("z_dyn_range") +
                0.10 * _z("z_ari") +
                0.10 * _z("z_nmi") +
                0.10 * _z("z_stability") +
                0.10 * _z("z_f1") +
                0.10 * _z("z_bal_acc") +
                0.05 * _z("z_neg_frac") +
                0.05 * _z("z_sil")
            )
        else:
            # Redistribute supervised weight (20%) to unsupervised components
            logger.info(
                "Supervised metrics unavailable; using unsupervised-only "
                "weights for score_combined."
            )
            res["score_combined"] = (
                0.20 * _z("z_cv_tic") +
                0.15 * _z("z_sam") +
                0.15 * _z("z_corr") +
                0.10 * _z("z_dyn_range") +
                0.15 * _z("z_ari") +
                0.10 * _z("z_nmi") +
                0.10 * _z("z_stability") +
                0.05 * _z("z_neg_frac")
            )

        # 2. Unsupervised (no labels needed — quality-focused)
        res["score_unsupervised"] = (
            0.25 * _z("z_cv_tic") +
            0.20 * _z("z_sam") +
            0.20 * _z("z_corr") +
            0.15 * _z("z_stability") +
            0.10 * _z("z_sil") +
            0.10 * _z("z_dyn_range")
        )

        # 3. Supervised-focused (NaN if supervised metrics are absent)
        if has_supervised:
            res["score_supervised"] = (
                0.25 * _z("z_f1") +
                0.25 * _z("z_bal_acc") +
                0.15 * _z("z_ari") +
                0.15 * _z("z_nmi") +
                0.10 * _z("z_stability") +
                0.10 * _z("z_cv_tic")
            )
        else:
            res["score_supervised"] = np.nan

        # 4. Efficiency-aware
        res["score_efficient"] = (
            0.85 * res["score_combined"] +
            0.15 * _z("z_compute_time")
        )

        return res.sort_values("score_combined", ascending=False).reset_index(drop=True)

    # -- plotting ----------------------------------------------------------

    def plot(
        self,
        out_dir: Union[str, Path] = "normalization_selection_output",
        save: bool = True,
    ) -> List[Path]:
        """Generate evaluation plots (box plots, bar charts, radar).

        Parameters
        ----------
        out_dir : str or Path
            Sub-folder inside ``OUTPUT_DIR`` for saved figures.
        save : bool
            Persist plots as PNG + PDF.

        Returns
        -------
        list of Path
            Saved file paths.
        """
        if self._results is None:
            raise RuntimeError("Call evaluate() before plotting.")
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for plotting.")

        res = self._results
        out = OUTPUT_DIR / Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        self._pub_style()

        # --- 1. Composite score comparison ---
        fig, ax = plt.subplots(figsize=(10, 5))
        methods = res["method"].tolist()
        x = np.arange(len(methods))
        width = 0.2
        ax.bar(x - 1.5 * width, res["score_combined"], width, label="Combined")
        ax.bar(x - 0.5 * width, res["score_unsupervised"], width, label="Unsupervised")
        ax.bar(x + 0.5 * width, res["score_supervised"], width, label="Supervised")
        ax.bar(x + 1.5 * width, res["score_efficient"], width, label="Efficient")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel("Composite score (higher = better)")
        ax.set_title("Normalization Method Comparison")
        ax.legend(frameon=False)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        fig.tight_layout()
        saved.extend(self._save_fig(fig, out, "composite_scores", save))

        # --- 2. Metric box/bar plots ---
        metric_info = [
            ("cv_tic", "CV of TIC (lower = better)", True),
            ("within_group_SAM", "Within-group SAM (lower = better)", True),
            ("within_group_corr", "Within-group correlation (higher = better)", False),
            ("cluster_ARI", "Adjusted Rand Index (higher = better)", False),
            ("cluster_stability", "Cluster stability (higher = better)", False),
            ("negative_fraction", "Negative fraction (lower = better)", True),
        ]
        for col, title, lower_better in metric_info:
            if col not in res.columns:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.5))
            vals = res[col].values
            x = np.arange(len(methods))
            colors = ["#2ca02c" if not lower_better else "#d62728"] * len(vals)
            best_idx = int(np.nanargmin(vals) if lower_better else np.nanargmax(vals))
            colors[best_idx] = "#1f77b4"
            ax.bar(x, vals, color=colors)
            ax.set_title(title)
            ax.set_ylabel(col)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            fig.tight_layout()
            saved.extend(self._save_fig(fig, out, col, save))

        # --- 3. Overall ranking bar ---
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(methods[::-1], res["score_combined"].values[::-1])
        ax.set_xlabel("Combined score (higher = better)")
        ax.set_title("Overall Normalization Ranking")
        fig.tight_layout()
        saved.extend(self._save_fig(fig, out, "overall_ranking", save))

        # --- 4. Export CSVs ---
        res.to_csv(out / "normalization_eval.csv", index=False)
        summary = {
            "best_method": res.iloc[0]["method"],
            "n_spectra": int(self._X_raw.shape[0]) if self._X_raw is not None else 0,
            "n_channels": int(self._X_raw.shape[1]) if self._X_raw is not None else 0,
            "methods_evaluated": methods,
            "scores": {
                row["method"]: round(row["score_combined"], 4)
                for _, row in res.iterrows()
            },
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved %d files to %s", len(saved) + 2, out)
        return saved

    def print_summary(self, top_n: int = 5) -> None:
        """Print a ranked summary of evaluation results.

        Parameters
        ----------
        top_n : int, default 5
            Number of top methods to display per score variant.
        """
        if self._results is None:
            raise RuntimeError("Call evaluate() before print_summary().")

        res = self._results

        logger.info("=" * 80)
        logger.info("NORMALIZATION EVALUATION SUMMARY (Top %d Methods)", top_n)
        logger.info("=" * 80)

        logger.info("")
        logger.info("--- COMBINED SCORE (Recommended) ---")
        logger.info(
            "Weights: 40%% spectral quality, 30%% clustering, "
            "20%% supervised, 10%% practical"
        )
        for i, row in res.head(top_n).iterrows():
            logger.info(
                "  %d. %-20s | Score: %6.3f | CV-TIC: %.3f | SAM: %.4f | "
                "ARI: %.3f | F1: %.3f",
                i + 1, row["method"], row["score_combined"],
                row["cv_tic"], row["within_group_SAM"],
                row["cluster_ARI"], row["supervised_macro_f1"],
            )

        logger.info("")
        logger.info("--- UNSUPERVISED SCORE ---")
        for i, row in res.nlargest(top_n, "score_unsupervised").iterrows():
            logger.info(
                "  %d. %-20s | Score: %6.3f",
                i + 1, row["method"], row["score_unsupervised"],
            )

        logger.info("")
        logger.info("--- SUPERVISED SCORE ---")
        for i, row in res.nlargest(top_n, "score_supervised").iterrows():
            logger.info(
                "  %d. %-20s | Score: %6.3f | F1: %.3f | Balanced Acc: %.3f",
                i + 1, row["method"], row["score_supervised"],
                row["supervised_macro_f1"], row["supervised_bal_acc"],
            )

        logger.info("")
        logger.info("--- EFFICIENCY-AWARE SCORE ---")
        for i, row in res.nlargest(top_n, "score_efficient").iterrows():
            logger.info(
                "  %d. %-20s | Score: %6.3f | Time: %.4fs",
                i + 1, row["method"], row["score_efficient"],
                row["compute_time_sec"],
            )

        logger.info("=" * 80)

    def preview_overlay(
        self,
        file: Union[str, Path],
        methods: Optional[List[str]] = None,
        max_methods: int = 5,
        mz_min: Optional[float] = None,
        mz_max: Optional[float] = None,
        save_to: Optional[Union[str, Path]] = "normalization_selection_output",
    ) -> None:
        """Plot raw vs normalised overlays for quick visual comparison.

        Parameters
        ----------
        file : str or Path
            Single spectrum file to visualise.
        methods : list of str, optional
            Methods to overlay.  Defaults to top methods from evaluation.
        max_methods : int
            Cap on the number of overlays.
        mz_min, mz_max : float, optional
            m/z window for the plot.
        save_to : str, Path, or None
            Save directory (relative to OUTPUT_DIR).  ``None`` skips saving.
        """
        if not _MPL_AVAILABLE:
            raise ImportError("matplotlib is required for plotting.")

        mz, intensity, name, group = import_data(
            str(file), mz_min or self.mz_min, mz_max or self.mz_max,
            group_patterns=self.group_patterns,
            group_fn=self.group_fn,
        )

        if methods is None:
            if self._results is not None:
                methods = self._results["method"].tolist()[:max_methods]
            else:
                methods = (self.methods or normalization_method_names())[:max_methods]

        self._pub_style()
        fig, axes = plt.subplots(len(methods) + 1, 1,
                                  figsize=(12, 2.5 * (len(methods) + 1)),
                                  sharex=True)

        axes[0].plot(mz, intensity, lw=0.5, color="grey")
        axes[0].set_title(f"Raw: {name}")
        axes[0].set_ylabel("Intensity")

        for i, m in enumerate(methods[:max_methods]):
            kwargs = self.method_kwargs_map.get(m, {}) if self.method_kwargs_map else {}

            def _project_dataset_vector(values: np.ndarray) -> np.ndarray:
                values = np.asarray(values, dtype=float)
                if self._mz is None or values.shape != np.asarray(self._mz).shape:
                    return values
                if len(self._mz) == len(mz) and np.allclose(self._mz, mz, atol=1e-6):
                    return values
                return np.interp(mz, self._mz, values, left=0.0, right=0.0)

            if "reference_idx" in kwargs and self._mz is not None:
                ref_idx = int(kwargs["reference_idx"])
                if 0 <= ref_idx < len(self._mz):
                    ref_mz = float(self._mz[ref_idx])
                    kwargs = {
                        **kwargs,
                        "reference_idx": int(np.argmin(np.abs(mz - ref_mz))),
                    }
            if "reference_indices" in kwargs and self._mz is not None:
                ref_indices = np.asarray(kwargs["reference_indices"], dtype=int)
                mapped = []
                for ref_idx in ref_indices:
                    if 0 <= ref_idx < len(self._mz):
                        ref_mz = float(self._mz[ref_idx])
                        mapped.append(int(np.argmin(np.abs(mz - ref_mz))))
                kwargs = {
                    **kwargs,
                    "reference_indices": mapped,
                }

            # Build reference for dataset-level methods
            if m in ("pqn", "median_of_ratios") and self._X_raw is not None:
                if m == "pqn":
                    kwargs = {
                        **kwargs,
                        "reference": _project_dataset_vector(np.nanmedian(self._X_raw, axis=0)),
                    }
                elif m == "median_of_ratios":
                    log_means = np.nanmean(np.log(self._X_raw + 1.0), axis=0)
                    kwargs = {
                        **kwargs,
                        "reference": _project_dataset_vector(np.exp(log_means) - 1.0),
                    }
            elif m == "pareto" and self._X_raw is not None:
                kwargs = {
                    **kwargs,
                    "mean": _project_dataset_vector(np.nanmean(self._X_raw, axis=0)),
                    "std": _project_dataset_vector(np.nanstd(self._X_raw, axis=0, ddof=0)),
                }
            elif m == "mass_stratified_pqn" and self._X_raw is not None:
                kwargs = {
                    **kwargs,
                    "reference": _project_dataset_vector(np.nanmedian(self._X_raw, axis=0)),
                    "mz_values": mz,
                }

            try:
                norm = normalize(intensity, method=m, **kwargs)
                axes[i + 1].plot(mz, norm, lw=0.5)
                axes[i + 1].set_title(m)
                axes[i + 1].set_ylabel("Norm. Int.")
            except Exception as e:
                axes[i + 1].set_title(f"{m} (failed: {e})")

        axes[-1].set_xlabel("m/z")
        fig.tight_layout()

        if save_to:
            save_dir = OUTPUT_DIR / Path(save_to)
            save_dir.mkdir(parents=True, exist_ok=True)
            for ext in (".png", ".pdf"):
                fig.savefig(save_dir / f"preview_{name}{ext}",
                            bbox_inches="tight", dpi=300)
        plt.show()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _pub_style():
        if not _MPL_AVAILABLE:
            return
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        })

    @staticmethod
    def _save_fig(fig, out_dir: Path, stem: str, save: bool) -> List[Path]:
        saved = []
        if save:
            for ext in (".png", ".pdf"):
                p = out_dir / f"{stem}{ext}"
                fig.savefig(p, bbox_inches="tight")
                saved.append(p)
        plt.close(fig)
        return saved
