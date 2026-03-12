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

def pick_peaks(
    df: pd.DataFrame,
    targets: Sequence[float] = TARGET_MASSES,
    tol_da: Optional[float] = TOL_DA_DEFAULT,
    tol_ppm: Optional[float] = TOL_PPM_DEFAULT,
    method: str = "centroid",          # "max" | "centroid" | "parabolic"
    min_points: int = 3,
    min_intensity: float = 0.0
) -> List[Dict]:
    """
    Select a representative (m/z, intensity, channel) near each target.
    - 'max':      highest point within window
    - 'centroid': intensity-weighted mean m/z within window
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
            # Intensity-weighted mean m/z
            wsum = Iw.sum()
            if wsum > 0 and len(mzw) >= 1:
                mz_c = float((mzw * Iw).sum() / wsum)
                # Use channel of the closest bin; intensity as apex
                nearest = int(np.argmin(np.abs(mz[idxs] - mz_c)))
                k_global = int(idxs[nearest])
                mz_pick, I_pick, ch_pick = mz_c, float(I[k_global]), int(ch[k_global])
                method_used = "centroid"
            else:
                mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], int(ch[k_global])
                method_used = "max_fallback"

        elif method == "parabolic":
            # Fit around apex using apex±1 if available
            if len(mzw) >= 3 and 0 < k_local < len(mzw)-1:
                trip_x = mzw[k_local-1:k_local+2]
                trip_y = Iw[k_local-1:k_local+2]
                xv = _parabolic_vertex(trip_x, trip_y)
                if xv is not None and np.isfinite(xv):
                    # pick nearest channel for reporting
                    nearest = int(np.argmin(np.abs(mz[idxs] - xv)))
                    k_global = int(idxs[nearest])
                    mz_pick, I_pick, ch_pick = float(xv), float(I[k_global]), int(ch[k_global])
                    method_used = "parabolic"
                else:
                    mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], int(ch[k_global])
                    method_used = "max_fallback"
            else:
                mz_pick, I_pick, ch_pick = mz[k_global], I[k_global], int(ch[k_global])
                method_used = "max_fallback"
        else:
            raise ValueError(f"Unknown method: {method}")

        edge = np.isclose(mz_pick, left) or np.isclose(mz_pick, right)
        out.append({
            "Target_m/z": float(xi),
            "Matched_m/z": float(mz_pick),
            "Channel": int(ch_pick),
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
