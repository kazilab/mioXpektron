"""
peak_picker.py
Identify the most intense point within ±0.5 Da of eight reference m/z
values in every calibrated ToF-SIMS spectrum and return the results
as a nested Python dictionary.

Author: Julhash Kazi, Lund University, 2025-06-15
"""
import logging
import os
from glob import glob
from pathlib import Path
from typing import Sequence, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------
# 1) USER SETTINGS
# --------------------------------------------------------------------
TOL_DA_DEFAULT = 0.2          # e.g., ±0.2 Da (fix the comment!)
TOL_PPM_DEFAULT = None        # e.g., 150  (use PPM if not None)
TARGET_MASSES = np.array([
    1.00782503224,    # H+
    15.0229265168,    # CH3+
    22.9897692820,    # 23Na+
    38.9637064864,    # 39K+
    58.065674,        # (example calibrant; ensure it's appropriate)
    86.096974,        # "
    104.107539,       # "
    184.073871,       # "
    224.105171        # "
], dtype=float)

# Optional: integer labels if you need nominal m/z columns as well.
TARGET_NOMINAL = np.rint(TARGET_MASSES).astype(int).tolist()

OUTPUT_DIR = Path("output_files")

# --------------------------------------------------------------------
# 2) CORE PICKER
# --------------------------------------------------------------------
def _ppm_to_da(mz: float, ppm: float) -> float:
    return mz * (ppm * 1e-6)

def _parabolic_vertex(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Fit a quadratic y = ax^2 + bx + c through three points around the apex
    and return the x coordinate of the vertex (-b / (2a)) if stable.
    """
    if len(x) != 3 or len(y) != 3:
        return None
    # Guard against degenerate fits
    if not np.isfinite(y).all() or np.isclose(np.ptp(x), 0.0):
        return None
    # Fit quadratic in x
    coeffs = np.polyfit(x, y, 2)
    a, b, _ = coeffs
    if np.isclose(a, 0.0):
        return None
    xv = -b / (2.0 * a)
    # Only accept if inside the local span
    if xv < x.min() or xv > x.max():
        return None
    return float(xv)


def _local_peak_bounds(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    *,
    min_fraction: float = 0.25,
    min_points: int = 3,
) -> tuple[int, int]:
    """Return a contiguous local support region around the apex."""
    if len(x) == 0:
        return 0, -1

    left = peak_idx
    right = peak_idx
    peak_y = float(y[peak_idx])
    baseline = float(np.nanmin(y))
    rel_height = max(peak_y - baseline, 0.0)
    threshold = baseline + min_fraction * rel_height

    while left > 0 and y[left - 1] >= threshold:
        left -= 1
    while right < len(x) - 1 and y[right + 1] >= threshold:
        right += 1

    while (right - left + 1) < min_points:
        expanded = False
        if left > 0:
            left -= 1
            expanded = True
        if (right - left + 1) >= min_points:
            break
        if right < len(x) - 1:
            right += 1
            expanded = True
        if not expanded:
            break

    return left, right


def _centroid_peak_center(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    *,
    min_points: int = 3,
    subtract_baseline: bool = True,
    apex_fraction: Optional[float] = 0.75,
) -> Optional[float]:
    """Return a local centroid center with optional baseline-aware apex support."""
    if len(x) == 0:
        return None

    left, right = _local_peak_bounds(x, y, peak_idx, min_points=min_points)
    x_local = np.asarray(x[left:right + 1], dtype=np.float64)
    y_local = np.asarray(y[left:right + 1], dtype=np.float64)
    if len(x_local) == 0 or not np.isfinite(y_local).all():
        return None

    if subtract_baseline:
        baseline = float(np.nanmin(y))
        support_profile = np.clip(y_local - baseline, 0.0, None)
    else:
        support_profile = np.clip(y_local, 0.0, None)

    if not np.any(support_profile > 0):
        return None

    peak_local = int(peak_idx - left)
    if apex_fraction is not None and 0.0 < apex_fraction < 1.0:
        threshold = float(np.nanmax(support_profile)) * apex_fraction
        apex_left = peak_local
        apex_right = peak_local
        while apex_left > 0 and support_profile[apex_left - 1] >= threshold:
            apex_left -= 1
        while apex_right < len(support_profile) - 1 and support_profile[apex_right + 1] >= threshold:
            apex_right += 1

        while (apex_right - apex_left + 1) < min_points:
            expanded = False
            if apex_left > 0:
                apex_left -= 1
                expanded = True
            if (apex_right - apex_left + 1) >= min_points:
                break
            if apex_right < len(support_profile) - 1:
                apex_right += 1
                expanded = True
            if not expanded:
                break

        x_local = x_local[apex_left:apex_right+1]

    weights = np.clip(y_local, 0.0, None)
    if apex_fraction is not None and 0.0 < apex_fraction < 1.0:
        weights = weights[apex_left:apex_right+1]

    if not np.any(weights > 0):
        return None
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None
    return float(np.sum(x_local * weights) / weight_sum)


def _interpolate_channel_for_mz(
    mz_values: np.ndarray,
    channel_values: np.ndarray,
    refined_mz: float,
) -> Optional[float]:
    """Map a refined m/z center back to a floating-point channel position."""
    if not np.isfinite(refined_mz):
        return None

    order = np.argsort(mz_values)
    mz_sorted = np.asarray(mz_values, dtype=np.float64)[order]
    ch_sorted = np.asarray(channel_values, dtype=np.float64)[order]

    unique_mz, unique_idx = np.unique(mz_sorted, return_index=True)
    if len(unique_mz) < 2:
        return None

    unique_ch = ch_sorted[unique_idx]
    if refined_mz < unique_mz[0] or refined_mz > unique_mz[-1]:
        return None

    return float(np.interp(refined_mz, unique_mz, unique_ch))

def pick_peaks(
    df: pd.DataFrame,
    targets: Sequence[float] = TARGET_MASSES,
    tol_da: Optional[float] = TOL_DA_DEFAULT,
    tol_ppm: Optional[float] = TOL_PPM_DEFAULT,
    method: str = "centroid",          # "max" | "centroid" | "centroid_raw" | "parabolic"
    min_points: int = 3,
    min_intensity: float = 0.0
) -> List[Dict]:
    """
    Select a representative (m/z, intensity, channel) near each target.
    - 'max':      highest point within window
    - 'centroid': baseline-subtracted, apex-focused centroid within the local peak
    - 'centroid_raw': raw intensity-weighted centroid over the local peak support
    - 'parabolic': pick apex by 'max', then refine m/z by a local quadratic fit
                   using apex±1 bins (falls back to 'max' if not feasible)
    """
    # Use float64 for precision in calibration context
    mz = df["m/z"].astype("float64").to_numpy()
    I  = df["Intensity"].astype("float64").to_numpy()
    ch = df["Channel"].to_numpy()

    # ensure finite
    finite = np.isfinite(mz) & np.isfinite(I)
    mz, I, ch = mz[finite], I[finite], ch[finite]

    out = []
    for xi in targets:
        # decide window
        tol = _ppm_to_da(xi, tol_ppm) if tol_ppm is not None else float(tol_da)
        left, right = xi - tol, xi + tol
        mask = (mz >= left) & (mz <= right) & (I >= min_intensity)

        if not mask.any():
            out.append({
                "Target_m/z": float(xi),
                "Matched_m/z": np.nan,
                "Channel": np.nan,
                "Intensity": 0.0,
                "n_points": 0,
                "edge": False,
                "method_used": "none"
            })
            continue

        idxs = np.flatnonzero(mask)
        mzw  = mz[idxs]
        Iw   = I[idxs]

        # Default pick: max
        k_local = int(np.nanargmax(Iw))
        k_global = int(idxs[k_local])

        if method == "max":
            mz_pick = mz[k_global]
            I_pick  = I[k_global]
            ch_pick = ch[k_global]
            method_used = "max"

        elif method == "centroid":
            mz_c = _centroid_peak_center(
                mzw,
                Iw,
                k_local,
                min_points=3,
                subtract_baseline=True,
                apex_fraction=0.75,
            )
            if mz_c is not None:
                ch_interp = _interpolate_channel_for_mz(mzw, ch[idxs], mz_c)
                if ch_interp is not None:
                    mz_pick, I_pick, ch_pick = mz_c, float(I[k_global]), float(ch_interp)
                    method_used = "centroid"
                else:
                    mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                    method_used = "max_fallback"
            else:
                mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                method_used = "max_fallback"

        elif method == "centroid_raw":
            mz_c = _centroid_peak_center(
                mzw,
                Iw,
                k_local,
                min_points=3,
                subtract_baseline=False,
                apex_fraction=None,
            )
            if mz_c is not None:
                ch_interp = _interpolate_channel_for_mz(mzw, ch[idxs], mz_c)
                if ch_interp is not None:
                    mz_pick, I_pick, ch_pick = mz_c, float(I[k_global]), float(ch_interp)
                    method_used = "centroid_raw"
                else:
                    mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                    method_used = "max_fallback"
            else:
                mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                method_used = "max_fallback"

        elif method == "parabolic":
            # Fit around apex using apex±1 if available
            if len(mzw) >= 3 and 0 < k_local < len(mzw)-1:
                trip_x = mzw[k_local-1:k_local+2]
                trip_y = Iw[k_local-1:k_local+2]
                xv = _parabolic_vertex(trip_x, trip_y)
                if xv is not None and np.isfinite(xv):
                    ch_interp = _interpolate_channel_for_mz(mzw, ch[idxs], xv)
                    if ch_interp is not None:
                        mz_pick, I_pick, ch_pick = float(xv), float(I[k_global]), float(ch_interp)
                        method_used = "parabolic"
                    else:
                        mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                        method_used = "max_fallback"
                else:
                    mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                    method_used = "max_fallback"
            else:
                mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], float(ch[k_global])
                method_used = "max_fallback"
        else:
            raise ValueError(f"Unknown method: {method}")

        edge = np.isclose(mz_pick, left) or np.isclose(mz_pick, right)
        out.append({
            "Target_m/z": float(xi),
            "Matched_m/z": float(mz_pick),
            "Channel": float(ch_pick),
            "Intensity": float(I_pick),
            "n_points": int(mask.sum()),
            "edge": bool(edge),
            "method_used": method_used
        })

    return out

# --------------------------------------------------------------------
# 3) BATCH PROCESSOR
# --------------------------------------------------------------------
def batch_peak_process(
    files: Sequence[str],
    tol_da: Optional[float] = TOL_DA_DEFAULT,
    tol_ppm: Optional[float] = TOL_PPM_DEFAULT,
    method: str = "centroid"
):
    """
    Process multiple spectra and write:
      - peak_summary.tsv  (long format, per spectrum x per target)
      - channel_summary_exact.tsv   (wide, columns = exact target masses)
      - channel_summary_nominal.tsv (wide, columns = nominal integer m/z)
    Returns:
      - calib_channels_dict_exact:  {spectrum: [channels in TARGET_MASSES order]}
      - calib_channels_dict_nominal:{spectrum: [channels in TARGET_NOMINAL order]}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    peak_dict: Dict[str, List[Dict]] = {}

    for path in tqdm(files, desc="Processing spectra"):
        try:
            df = pd.read_csv(
                path, sep="\t", header=0, comment="#",
                usecols=["Channel", "m/z", "Intensity"],
                dtype={"Channel": np.int32, "m/z": np.float64, "Intensity": np.float64}
            )
        except ValueError as e:
            raise RuntimeError(f"{os.path.basename(path)} – bad format: {e}")

        peak_dict[os.path.basename(path)] = pick_peaks(
            df, targets=TARGET_MASSES, tol_da=tol_da, tol_ppm=tol_ppm, method=method
        )

    # 4) Persist long table
    long_df = (
        pd.concat({k: pd.DataFrame(v) for k, v in peak_dict.items()})
          .reset_index(level=1, drop=True)
          .rename_axis("Spectrum")
          .reset_index()
    )
    peak_summary_path = OUTPUT_DIR / "peak_summary.tsv"
    long_df.to_csv(peak_summary_path, sep="\t", index=False)
    logger.info(f"Saved → {peak_summary_path}")

    # 5) Wide tables (exact vs nominal)
    # exact masses as columns
    exact_map = {
        spec: {row["Target_m/z"]: row["Channel"] for row in rows}
        for spec, rows in peak_dict.items()
    }
    summary_exact = (pd.DataFrame.from_dict(exact_map, orient="index")
                       .reindex(columns=list(TARGET_MASSES)))
    summary_exact.index.name = "Spectrum"
    summary_exact_csv = summary_exact.reset_index()
    summary_exact_path = OUTPUT_DIR / "channel_summary_exact.tsv"
    summary_exact_csv.to_csv(summary_exact_path, sep="\t", index=False)
    logger.info(f"Saved → {summary_exact_path}")

    # nominal integers as columns (optional)
    nominal_map = {}
    for spec, rows in peak_dict.items():
        d = {}
        for t, n in zip(TARGET_MASSES, TARGET_NOMINAL):
            # find the row for exact target t
            r = next((r for r in rows if np.isclose(r["Target_m/z"], t)), None)
            d[n] = (None if r is None or pd.isna(r["Channel"]) else int(r["Channel"]))
        nominal_map[spec] = d

    summary_nominal = (pd.DataFrame.from_dict(nominal_map, orient="index")
                         .reindex(columns=TARGET_NOMINAL))
    summary_nominal.index.name = "Spectrum"
    summary_nominal_csv = summary_nominal.reset_index()
    summary_nominal_path = OUTPUT_DIR / "channel_summary_nominal.tsv"
    summary_nominal_csv.to_csv(summary_nominal_path, sep="\t", index=False)
    logger.info(f"Saved → {summary_nominal_path}")

    # 6) Dicts in memory (channel lists in fixed order)
    calib_channels_dict_exact = {
        spectrum: [
            (None if pd.isna(ch) else int(ch))
            for ch in summary_exact.loc[spectrum, list(TARGET_MASSES)]
        ]
        for spectrum in summary_exact.index
    }

    calib_channels_dict_nominal = {
        spectrum: [
            (None if pd.isna(ch) else int(ch))
            for ch in summary_nominal.loc[spectrum, TARGET_NOMINAL]
        ]
        for spectrum in summary_nominal.index
    }

    # Show sample
    logger.info("Dictionary sample (first 2):")
    for k in list(calib_channels_dict_exact)[:2]:
        logger.info("%s → exact: %s | nominal: %s",
                    k, calib_channels_dict_exact[k], calib_channels_dict_nominal[k])

    return calib_channels_dict_exact, calib_channels_dict_nominal


'''
How to call
files = glob("path/to/spectra/*.txt")
calib_exact, calib_nominal = batch_peak_process(
    files,
    tol_da=0.2,           # or use tol_ppm=150, tol_da=None
    tol_ppm=None,
    method="parabolic"    # "max" | "centroid" | "parabolic"
)
'''
