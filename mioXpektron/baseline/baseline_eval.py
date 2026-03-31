from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import json
import warnings
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import rankdata
from pybaselines.utils import ParameterWarning
from .baseline_base import (
    baseline_method_names,
    small_param_grid_preset,
    baseline_correction,
    read_spectrum_table,
    _noise_mask_from_quantile,
    compute_metrics,
)

OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------

def _expand_methods(methods: Optional[Iterable[str]], param_grid: Optional[Dict[str, List[Dict]]]):
    """Expand a list of method names with parameter sets from a grid.

    Returns two parallel lists: labels, call_specs, where each element in
    call_specs is (method_name, kwargs_dict).
    """
    if methods is not None:
        all_methods = list(methods)
    elif param_grid:
        # A supplied parameter grid is usually intended to define the candidate set.
        all_methods = list(param_grid.keys())
    else:
        all_methods = baseline_method_names()
    labels, specs = [], []
    for m in all_methods:
        grids = (param_grid or {}).get(m, [dict()])
        for k in grids:
            label = f"{m}" if not k else f"{m}({', '.join(f'{kk}={vv}' for kk, vv in k.items())})"
            labels.append(label)
            specs.append((m, k))
    return labels, specs


def _spec_payload(label: str, spec: Tuple[str, Dict]) -> Dict[str, object]:
    method, kwargs = spec
    return {
        "label": label,
        "method": method,
        "kwargs": dict(kwargs),
    }


def _row_ranks(values: np.ndarray, ascending: bool) -> np.ndarray:
    # Use pandas DataFrame.rank() for vectorized row-wise ranking
    df = pd.DataFrame(values)
    ranked = df.rank(axis=1, method="average", ascending=ascending, na_option="keep")
    return ranked.to_numpy(dtype=float, na_value=np.nan)


def _frame_from_array(sample_names: List[str], labels: List[str], array: np.ndarray) -> pl.DataFrame:
    data = {"sample": sample_names}
    for idx, label in enumerate(labels):
        data[label] = array[:, idx]
    return pl.DataFrame(data)


def _metric_winners(values: np.ndarray, labels: List[str], minimize: bool = True) -> List[Optional[str]]:
    # Vectorized: use np.nanargmin/nanargmax across rows
    labels_arr = np.array(labels)
    winners: List[Optional[str]] = []
    # Mask rows that are entirely NaN
    all_nan = np.all(~np.isfinite(values), axis=1)
    if minimize:
        # Replace NaN with +inf for argmin
        safe = np.where(np.isfinite(values), values, np.inf)
        best_idx = np.argmin(safe, axis=1)
    else:
        # Replace NaN with -inf for argmax
        safe = np.where(np.isfinite(values), values, -np.inf)
        best_idx = np.argmax(safe, axis=1)
    for i in range(values.shape[0]):
        if all_nan[i]:
            winners.append(None)
        else:
            winners.append(labels_arr[best_idx[i]])
    return winners

def _has_glob_chars(s: str) -> bool:
    return any(ch in s for ch in "*?[")


@dataclass
class BaselineMethodEvaluator:
    """Evaluate baseline algorithms on ToF‑SIMS files supplied as paths or globs."""

    files: List[Union[str, Path]] = field(default_factory=list)
    methods: Optional[List[str]] = None
    param_grid: Optional[Dict[str, List[Dict]]] = None
    use_small_param_preset: bool = False
    auto_scale_window_size: bool = True  # Auto-scale window_size based on data size
    # Evaluation-time clipping: keep False so NAR/NBC/BBI remain informative
    eval_clip_negative: bool = False
    topk_for_snr: int = 5
    raw_noise_quantile: float = 0.2  # bottom q region considered 'baseline-only'
    flat_windows: Optional[List[Tuple[float, float]]] = None  # m/z ranges known to be baseline-only
    metrics_for_composite: Tuple[str, ...] = ("rfzn", "nar", "snr", "bbi", "br", "nbc")
    n_jobs: int = -1

    labels: List[str] = field(default_factory=list, init=False)
    specs: List[Tuple[str, Dict]] = field(default_factory=list, init=False)
    _resolved_files: List[Path] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self._resolved_files = self._expand_files(self.files)
        if not self._resolved_files:
            raise FileNotFoundError(
                "No input files found; verify the provided file paths or glob patterns."
            )
        # expose resolved paths back on the public attribute for convenience
        self.files = self._resolved_files
        if self.param_grid is None and self.use_small_param_preset:
            # Auto-scale window_size if enabled
            if self.auto_scale_window_size:
                # Sample first file to get data size for adaptive window_size calculation
                try:
                    df = read_spectrum_table(self._resolved_files[0])
                    n_points = len(df)
                    self.param_grid = small_param_grid_preset(n_points)
                    logger.info(f"ℹ Auto-scaled filter window_size based on {n_points:,} data points")
                except Exception as e:
                    logger.warning(f"⚠ Could not auto-scale window_size ({e}). Using defaults.")
                    self.param_grid = small_param_grid_preset()
            else:
                self.param_grid = small_param_grid_preset()
        self.labels, self.specs = _expand_methods(self.methods, self.param_grid)
        self._label_to_spec = {label: spec for label, spec in zip(self.labels, self.specs)}

    def _expand_files(self, candidates: Iterable[Union[str, Path]]) -> List[Path]:
        paths: List[Path] = []
        for item in candidates:
            s = str(item).strip()
            if not s:
                continue
            if _has_glob_chars(s):
                for hit in Path().glob(s):
                    if hit.is_file():
                        paths.append(hit.resolve())
                continue
            p = Path(s)
            if p.is_file():
                paths.append(p.resolve())
            elif p.is_dir():
                for hit in p.rglob("*"):
                    if hit.is_file():
                        paths.append(hit.resolve())
            else:
                for hit in Path().glob(s):
                    if hit.is_file():
                        paths.append(hit.resolve())

        return sorted(set(paths))

    # -- core ---------------------------------------------------------
    def _score_one(self, file: Path) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        df = read_spectrum_table(file)
        x = np.asarray(df["mz"].to_numpy(), dtype=float)
        y = np.asarray(df["intensity"].to_numpy(), dtype=float)
        # one noise mask per file
        if self.flat_windows:
            noise_mask = np.zeros_like(x, dtype=bool)
            for lo, hi in self.flat_windows:
                noise_mask |= (x >= float(lo)) & (x <= float(hi))
        else:
            noise_mask = _noise_mask_from_quantile(y, self.raw_noise_quantile)

        rfzn_row = np.empty(len(self.specs), dtype=float)
        nar_row  = np.empty_like(rfzn_row)
        snr_row  = np.empty_like(rfzn_row)
        bbi_row  = np.empty_like(rfzn_row)
        br_row   = np.empty_like(rfzn_row)
        nbc_row  = np.empty_like(rfzn_row)

        warn_records: List[dict] = []
        for j, (method, kwargs) in enumerate(self.specs):
            label = self.labels[j]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warnings.simplefilter("always", ParameterWarning)
                try:
                    # Always evaluate with clip_negative=False so NAR/NBC/BBI are meaningful
                    y_corr, bl = baseline_correction(
                        y, method=method, return_baseline=True, clip_negative=self.eval_clip_negative, **kwargs
                    )
                    met = compute_metrics(y_corr, y, bl, x, noise_mask=noise_mask, topk=self.topk_for_snr,
                                          raw_noise_quantile=self.raw_noise_quantile)
                    rfzn_row[j] = met.rfzn
                    nar_row[j]  = met.nar
                    snr_row[j]  = met.snr
                    bbi_row[j]  = met.bbi
                    br_row[j]   = met.br
                    nbc_row[j]  = met.nbc
                except Exception:
                    rfzn_row[j] = np.nan
                    nar_row[j]  = np.nan
                    snr_row[j]  = np.nan
                    bbi_row[j]  = np.nan
                    br_row[j]   = np.nan
                    nbc_row[j]  = np.nan
            if caught:
                for w in caught:
                    warn_records.append({
                        "file": file.name,
                        "label": label,
                        "method": method,
                        "warning_category": getattr(w.category, "__name__", str(w.category)),
                        "message": str(w.message),
                    })

        return file.name, rfzn_row, nar_row, snr_row, bbi_row, br_row, nbc_row, warn_records

    def evaluate(self, noise_quantile: Optional[float] = None, n_jobs: Optional[int] = None):
        if noise_quantile is not None:
            self.raw_noise_quantile = float(noise_quantile)
        if n_jobs is not None:
            self.n_jobs = int(n_jobs)

        worker = delayed(self._score_one)
        it = (worker(f) for f in self.files)
        results = Parallel(n_jobs=self.n_jobs, backend="loky")(tqdm(it, total=len(self.files),
                                                                     desc="baseline eval", ncols=96))

        # assemble
        sample_names = [r[0] for r in results]
        rfzn_arr = np.vstack([r[1] for r in results])
        nar_arr  = np.vstack([r[2] for r in results])
        snr_arr  = np.vstack([r[3] for r in results])
        bbi_arr  = np.vstack([r[4] for r in results])
        br_arr   = np.vstack([r[5] for r in results])
        nbc_arr  = np.vstack([r[6] for r in results])
        warning_lists = [r[7] for r in results]

        warning_log: List[dict] = []
        for warns in warning_lists:
            warning_log.extend(warns)

        rfzn = _frame_from_array(sample_names, self.labels, rfzn_arr)
        nar  = _frame_from_array(sample_names, self.labels, nar_arr)
        snr  = _frame_from_array(sample_names, self.labels, snr_arr)
        bbi  = _frame_from_array(sample_names, self.labels, bbi_arr)
        br   = _frame_from_array(sample_names, self.labels, br_arr)
        nbc  = _frame_from_array(sample_names, self.labels, nbc_arr)

        # rank per metric (average rank, NaNs kept)
        rank_rfzn = _row_ranks(rfzn_arr, ascending=True)
        rank_nar  = _row_ranks(nar_arr,  ascending=True)
        rank_snr  = _row_ranks(snr_arr,  ascending=False)
        rank_bbi  = _row_ranks(np.abs(bbi_arr),  ascending=True)
        rank_br   = _row_ranks(br_arr,   ascending=True)
        rank_nbc  = _row_ranks(nbc_arr,  ascending=True)

        ranks_map = {
            "rfzn": rank_rfzn,
            "nar": rank_nar,
            "snr": rank_snr,
            "bbi": rank_bbi,
            "br": rank_br,
            "nbc": rank_nbc,
        }
        selected_ranks = [ranks_map[m] for m in self.metrics_for_composite if m in ranks_map]
        if not selected_ranks:
            raise ValueError("metrics_for_composite did not match any available metrics")

        comp_arr = np.nanmean(np.stack(selected_ranks, axis=0), axis=0)
        comp = _frame_from_array(sample_names, self.labels, comp_arr)

        medians = np.nanmedian(comp_arr, axis=0)
        ordered_pairs = list(zip(self.labels, medians))
        ordered_pairs.sort(key=lambda kv: (np.inf if np.isnan(kv[1]) else kv[1]))
        overall_order = pl.DataFrame({
            "method": [label for label, _ in ordered_pairs],
            "median_rank": [float(val) if np.isfinite(val) else float("nan") for _, val in ordered_pairs],
        })
        overall_best = next(
            (label for label, val in ordered_pairs if np.isfinite(val)),
            ordered_pairs[0][0] if ordered_pairs else None,
        )
        best_spec_payload = (
            _spec_payload(overall_best, self._label_to_spec[overall_best])
            if overall_best in self._label_to_spec
            else None
        )
        ordered_specs = [
            _spec_payload(label, self._label_to_spec[label])
            for label, _ in ordered_pairs
            if label in self._label_to_spec
        ]

        rfzn_winners = _metric_winners(rfzn_arr, self.labels, minimize=True)
        nar_winners  = _metric_winners(nar_arr,  self.labels, minimize=True)
        snr_winners  = _metric_winners(snr_arr,  self.labels, minimize=False)
        bbi_winners  = _metric_winners(np.abs(bbi_arr),  self.labels, minimize=True)
        br_winners   = _metric_winners(br_arr,   self.labels, minimize=True)
        nbc_winners  = _metric_winners(nbc_arr,  self.labels, minimize=True)

        win_counters = {
            "RFZN": Counter(w for w in rfzn_winners if w),
            "NAR":  Counter(w for w in nar_winners if w),
            "SNR":  Counter(w for w in snr_winners if w),
            "BBI":  Counter(w for w in bbi_winners if w),
            "BR":   Counter(w for w in br_winners if w),
            "NBC":  Counter(w for w in nbc_winners if w),
        }
        win_counts = pl.DataFrame({
            "method": self.labels,
            "RFZN": [int(win_counters["RFZN"].get(label, 0)) for label in self.labels],
            "NAR":  [int(win_counters["NAR"].get(label, 0)) for label in self.labels],
            "SNR":  [int(win_counters["SNR"].get(label, 0)) for label in self.labels],
            "BBI":  [int(win_counters["BBI"].get(label, 0)) for label in self.labels],
            "BR":   [int(win_counters["BR"].get(label, 0)) for label in self.labels],
            "NBC":  [int(win_counters["NBC"].get(label, 0)) for label in self.labels],
        })

        summary = {
            "overall_best_method": overall_best,
            "overall_best_spec": best_spec_payload,
            "overall_order": {label: float(val) if np.isfinite(val) else float("nan") for label, val in ordered_pairs},
            "overall_order_specs": ordered_specs,
            "win_counts": {metric: {label: int(counter.get(label, 0)) for label in self.labels}
                            for metric, counter in win_counters.items()},
            "metrics_for_composite": list(self.metrics_for_composite),
        }

        # expose
        self._rfzn = rfzn
        self._nar = nar
        self._snr = snr
        self._bbi = bbi
        self._br = br
        self._nbc = nbc
        self._comp = comp
        self._overall_order = overall_order
        self._overall_order_methods = [label for label, _ in ordered_pairs]
        self._overall_best = overall_best
        self._overall_best_spec = best_spec_payload
        self._overall_order_specs = ordered_specs
        self._win_counts = win_counts
        self._warnings = warning_log
        return summary

    def warning_log(self) -> pl.DataFrame:
        if not hasattr(self, "_warnings"):
            raise RuntimeError("Call evaluate() before requesting the warning log.")
        if not self._warnings:
            return pl.DataFrame(schema=["file", "label", "method", "warning_category", "message"])
        return pl.DataFrame(self._warnings)

    # -- plotting -----------------------------------------------------
    def _pub_style(self):
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

    def plot(self, out_dir: Union[str, Path] = "baseline_selection_output") -> List[Path]:
        if not hasattr(self, "_rfzn"):
            raise RuntimeError("Call evaluate() before plotting.")
        self._pub_style()
        out_dir = OUTPUT_DIR / Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        def _boxplot(df: pl.DataFrame, title: str, ylabel: str, fname: str):
            numeric_cols = [name for name, dtype in df.schema.items() if name != "sample" and dtype.is_numeric()]
            cleaned = []
            labels = []
            for col in numeric_cols:
                series = df.get_column(col).drop_nulls()
                if series.is_empty():
                    continue
                cleaned.append(series.to_numpy())
                labels.append(col)
            if not cleaned:
                return
            plt.figure(figsize=(9, 4.8))
            plt.boxplot(cleaned, vert=True, patch_artist=True, labels=labels,
                        showfliers=False, widths=0.8,
                        medianprops=dict(color="black", lw=1.2))
            plt.ylabel(ylabel); plt.title(title)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            for ext in (".pdf", ".png"):
                p = out_dir / f"{fname}{ext}"
                plt.savefig(p, bbox_inches="tight")
                saved.append(p)
            plt.close()

        # Six metrics
        _boxplot(self._rfzn, "RFZN across baseline methods", "RFZN (RMS in baseline regions)", "rfzn_box")
        _boxplot(self._nar,  "NAR across baseline methods",  "NAR (negative area ratio)",  "nar_box")
        _boxplot(self._snr,  "SNR across baseline methods",  "SNR (median top‑K peaks)",  "snr_box")
        _boxplot(self._bbi,  "BBI across baseline methods",  "BBI (median bias in baseline regions)", "bbi_box")
        _boxplot(self._br,   "BR across baseline methods",   "BR (RMS of baseline curvature)", "br_box")
        _boxplot(self._nbc,  "NBC across baseline methods",  "NBC (fraction y<0)", "nbc_box")

        # Win counts (all metrics combined)
        totals = (
            self._win_counts
            .with_columns(pl.sum_horizontal(pl.all().exclude("method")).alias("total"))
            .select(["method", "total"])
            .sort("total", descending=True)
        )
        plt.figure(figsize=(9, 4.2))
        plt.bar(totals["method"].to_list(), totals["total"].to_numpy())
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("# metric wins across files")
        plt.title("Total win counts (RFZN, NAR, SNR, BBI, BR, NBC)")
        plt.tight_layout()
        for ext in (".pdf", ".png"):
            p = out_dir / f"win_counts_total{ext}"; plt.savefig(p, bbox_inches="tight"); saved.append(p)
        plt.close()

        # Overall composite ranking bar
        order = self._overall_order
        methods = order["method"].to_list()
        values = order["median_rank"].to_numpy()
        plt.figure(figsize=(9, 4.2))
        plt.bar(methods, values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Median composite rank (lower better)")
        plt.title("Overall baseline method ranking")
        plt.tight_layout()
        for ext in (".pdf", ".png"):
            p = out_dir / f"overall_ranking{ext}"; plt.savefig(p, bbox_inches="tight"); saved.append(p)
        plt.close()

        # Export numeric results
        self._rfzn.write_csv(out_dir / "rfzn_by_file.csv")
        self._nar.write_csv(out_dir / "nar_by_file.csv")
        self._snr.write_csv(out_dir / "snr_by_file.csv")
        self._bbi.write_csv(out_dir / "bbi_by_file.csv")
        self._br.write_csv(out_dir / "br_by_file.csv")
        self._nbc.write_csv(out_dir / "nbc_by_file.csv")
        self._comp.write_csv(out_dir / "composite_rank_by_file.csv")
        self._win_counts.write_csv(out_dir / "win_counts_by_metric.csv")
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "overall_best_method": self._overall_best,
                "overall_best_spec": self._overall_best_spec,
                "overall_order": {m: float(v) for m, v in zip(methods, values)},
                "overall_order_specs": self._overall_order_specs,
                "metrics_for_composite": list(self.metrics_for_composite)
            }, f, indent=2)

        return saved

    # -- helpers ------------------------------------------------------
    def preview_overlay(self, file: Union[str, Path],
                        methods: Optional[List[str]] = None,
                        max_methods: int = 5,
                        save_to: Optional[Union[str, Path]] = "baseline_selection_output",
                        show_errors: bool = True):
        """Plot raw, baseline and corrected overlays for a few methods on a single file.

        Parameters
        ----------
        file : str or Path
            Path to a single spectrum file (not a list!)
        methods : list of str, optional
            Method names to plot. If None, uses top methods from evaluation.
        max_methods : int
            Maximum number of methods to plot (default: 5)
        save_to : str or Path, optional
            Directory to save plots. Set to None to skip saving.
        show_errors : bool
            If True (default), print errors when methods fail instead of silently ignoring them.
        """
        # Handle case where user passes a list instead of single file
        if isinstance(file, (list, tuple)):
            if len(file) == 0:
                raise ValueError("Empty file list provided. Please provide a single file path.")
            logger.warning(f"⚠ Received a list of {len(file)} files. Using the first one: {Path(file[0]).name}")
            file = file[0]

        df = read_spectrum_table(file)
        x = np.asarray(df["mz"].to_numpy(), dtype=float)
        y = np.asarray(df["intensity"].to_numpy(), dtype=float)
        default_order = getattr(self, "_overall_order_methods", baseline_method_names())
        methods = methods or list(default_order)[:max_methods]

        plt.figure(figsize=(9, 4.8))
        plt.plot(x, y, lw=1, label="raw")  # raw

        successful_methods = []
        failed_methods = []

        for m in methods[:max_methods]:
            try:
                method_name, method_kwargs = self._label_to_spec.get(m, (m, {}))
                y_corr, bl = baseline_correction(
                    y,
                    method=method_name,
                    return_baseline=True,
                    clip_negative=False,
                    **method_kwargs,
                )
                plt.plot(x, bl, lw=1, linestyle=":", label=f"baseline: {m}")
                plt.plot(x, y_corr, lw=1, label=f"corrected: {m}")
                successful_methods.append(m)
            except Exception as e:
                failed_methods.append((m, e))
                if show_errors:
                    logger.error(f"✗ Method '{m}' failed: {type(e).__name__}: {str(e)[:100]}")
                continue

        plt.xlabel("m/z"); plt.ylabel("Intensity (a.u.)"); plt.title(Path(file).name)
        plt.legend(ncol=2, frameon=False); plt.tight_layout()
        if save_to:
            save_dir = OUTPUT_DIR / Path(save_to)
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".pdf", ".png"):
                p = save_dir / f"preview_overlay{ext}"
                plt.savefig(p, bbox_inches="tight", dpi=300)
        backend = str(plt.get_backend()).lower()
        if "agg" in backend:
            plt.close()
        else:
            plt.show()

        # Print summary
        if show_errors and (successful_methods or failed_methods):
            summary_lines = [
                f"\n{'='*60}",
                f"Preview Overlay Summary:",
                f"  File: {Path(file).name}",
                f"  Successful: {len(successful_methods)}/{len(methods)} methods",
            ]
            if successful_methods:
                summary_lines.append(f"    ✓ {', '.join(successful_methods)}")
            if failed_methods:
                summary_lines.append(f"  Failed: {len(failed_methods)} methods")
                for method, _ in failed_methods:
                    summary_lines.append(f"    ✗ {method}")
            summary_lines.append(f"{'='*60}\n")
            logger.info("\n".join(summary_lines))
