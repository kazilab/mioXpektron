import os
import re
import glob
from typing import Tuple, List, Dict

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_window(file_path: str, mz_min: float, mz_max: float, norm_tic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one spectrum and return (m/z, intensity) in the requested window.

    Parameters
    ----------
    file_path : str
        Path to a tab-separated spectrum with columns "m/z" and "Intensity".
    mz_min, mz_max : float
        Inclusive m/z window to extract.
    norm_tic : bool, default False
        If True, normalize intensities by total ion count (sum to 1).

    Returns
    -------
    mz : np.ndarray
    inten : np.ndarray
        Intensities scaled by 1e6 (to keep values readable on plots).
    """
    try:
        df = pl.read_csv(file_path, separator="\t", comment_prefix="#")
    except TypeError:
        # For older Polars versions using `comment_char`
        df = pl.read_csv(file_path, separator="\t", comment_char="#")

    df = df.filter((pl.col("m/z") >= mz_min) & (pl.col("m/z") <= mz_max))

    mz = df["m/z"].to_numpy()
    inten = df["Intensity"].to_numpy()
    if norm_tic and inten.size and inten.sum() > 0:
        inten = (inten / inten.sum())*1000000
    return mz, inten


def _infer_group_from_name(path: str) -> str:
    """
    Infer group label from filename:
      - contains "_CC" (case-insensitive) -> "Cancer"
      - contains "_CT" (case-insensitive) -> "Control"
      - otherwise -> "Unknown"
    """
    name = os.path.basename(path).lower()
    if "_cc" in name:
        return "Cancer"
    if "_ct" in name:
        return "Control"
    return "Unknown"


def check_overlapping_peaks2(
    data_dir: str,
    file_pattern: str,
    mz_min: float,
    mz_max: float,
    norm_tic: bool = False,
    alpha: float = 0.18,
    bin_width: float = 0.001,
    show_median: bool = True,
    show_group_cumulative: bool = True,
):
    """
    Overlay spectra with two colors (Cancer vs Control) inferred from file names.

    Parameters
    ----------
    data_dir : str
        Directory containing spectra.
    file_pattern : str
        Glob pattern (e.g., "*.txt").
    mz_min, mz_max : float
        m/z window to visualize.
    norm_tic : bool, default False
        Normalize each spectrum by its TIC prior to plotting.
    alpha : float, default 0.18
        Line transparency for individual spectra.
    bin_width : float, default 0.001
        Common grid step for interpolation (used for medians/cumulative plots).
    show_median : bool, default True
        If True, overlay per-group median curves (thicker lines).
    show_group_cumulative : bool, default True
        If True, plot per-group cumulative intensity curves on a separate figure.

    Notes
    -----
    - Group detection is based on substrings in filenames: "_CC" (Cancer), "_CT" (Control).
    - Files without these markers are labeled "Unknown" and plotted in grey.
    """
    # 1) Gather files
    files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if len(files) == 0:
        raise RuntimeError(f"No files matched: {os.path.join(data_dir, file_pattern)}")

    # 2) Read and group
    grouped_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"Cancer": [], "Control": [], "Unknown": []}
    for fp in files:
        mz, inten = load_window(fp, mz_min, mz_max, norm_tic=norm_tic)
        if mz.size == 0:
            continue
        grouped_curves[_infer_group_from_name(fp)].append((mz, inten))

    n_cancer = len(grouped_curves["Cancer"])
    n_control = len(grouped_curves["Control"])
    n_unknown = len(grouped_curves["Unknown"])

    # 3) Plot overlay: two colors (+ optional grey for Unknown)
    color_map = {"Cancer": "#d62728", "Control": "#1f77b4", "Unknown": "#7f7f7f"}  # red, blue, grey
    plt.figure(figsize=(10, 6))
    for group, curves in grouped_curves.items():
        for mz, inten in curves:
            plt.plot(mz, inten, linewidth=0.7, alpha=alpha, color=color_map[group])
    plt.xlim(mz_min, mz_max)
    plt.xlabel("m/z")
    plt.ylabel("Normalized intensity" if norm_tic else "Intensity (scaled ×1e6)")
    title_counts = f"Cancer n={n_cancer}, Control n={n_control}" + (f", Unknown n={n_unknown}" if n_unknown else "")
    plt.title(f"Spectra overlay by group ({mz_min}–{mz_max} m/z) | {title_counts}")

    # Legend handles (one per group if present)
    handles = []
    labels = []
    for group in ["Cancer", "Control", "Unknown"]:
        if grouped_curves[group]:
            h, = plt.plot([], [], color=color_map[group], lw=3, label=group)
            handles.append(h)
            labels.append(group)
    # 4) Optional: per-group median curves on common grid (thicker lines)
    if show_median:
        mz_common = np.arange(mz_min, mz_max + bin_width, bin_width)
        for group in ["Cancer", "Control"]:
            if not grouped_curves[group]:
                continue
            interp = []
            for mz, inten in grouped_curves[group]:
                interp.append(np.interp(mz_common, mz, inten, left=0.0, right=0.0))
            interp = np.vstack(interp)  # (n, bins)
            median_curve = np.median(interp, axis=0)
            h_med, = plt.plot(mz_common, median_curve, color=color_map[group], lw=2.5, alpha=0.9)
            # Add a legend entry for median if desired (commented to keep legend compact)
            # handles.append(h_med); labels.append(f"{group} median")

    if handles:
        plt.legend(handles=handles, labels=labels, frameon=False)

    plt.tight_layout()
    plt.show()

    # 5) Optional: per-group cumulative intensity curves (sum over spectra)
    if show_group_cumulative:
        mz_common = np.arange(mz_min, mz_max + bin_width, bin_width)

        def group_sum(group: str) -> np.ndarray:
            if not grouped_curves[group]:
                return np.zeros_like(mz_common)
            acc = []
            for mz, inten in grouped_curves[group]:
                acc.append(np.interp(mz_common, mz, inten, left=0.0, right=0.0))
            return np.sum(np.vstack(acc), axis=0)

        cum_cancer = group_sum("Cancer")
        cum_control = group_sum("Control")

        plt.figure(figsize=(10, 4))
        if cum_cancer.sum() > 0:
            plt.plot(mz_common, cum_cancer, label=f"Cancer (n={n_cancer})", color=color_map["Cancer"], lw=1.8)
        if cum_control.sum() > 0:
            plt.plot(mz_common, cum_control, label=f"Control (n={n_control})", color=color_map["Control"], lw=1.8)
        if n_unknown:
            cum_unknown = group_sum("Unknown")
            if cum_unknown.sum() > 0:
                plt.plot(mz_common, cum_unknown, label=f"Unknown (n={n_unknown})", color=color_map["Unknown"], lw=1.4, ls="--")

        plt.xlim(mz_min, mz_max)
        plt.xlabel("m/z")
        plt.ylabel("Cumulative intensity (scaled)")
        plt.title(f"Cumulative intensity by group ({mz_min}–{mz_max} m/z)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
