"""
Adaptive parameterization helpers for the mioXpektron pipeline.

Each estimator derives a value from data that would otherwise be a fixed
constant.  All functions are pure (no mutation of global state) and return
plain Python / NumPy values that callers feed into the existing config
dataclasses.

Usage is **opt-in**: every config keeps its current defaults; the new
``auto_tune=True`` flag triggers adaptive estimation.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Calibration tolerance estimation
# =====================================================================

def estimate_autodetect_tolerance(
    files: Sequence[str],
    reference_masses: Sequence[float],
    *,
    sample_n: int = 10,
    quantile: float = 0.90,
) -> float:
    """Estimate ``autodetect_tol_da`` from observed peak widths near
    calibrant m/z values.

    Reads a sample of spectra, measures the FWHM of the strongest peak
    within +/-1 Da of each reference mass, and returns a tolerance equal
    to *quantile* of those widths (clamped to [0.05, 2.0] Da).
    """
    from scipy.signal import peak_widths, find_peaks

    chosen = _sample_files(files, sample_n)
    widths: List[float] = []

    for fp in chosen:
        try:
            df = pd.read_csv(fp, sep="\t", header=0, comment="#")
        except Exception:
            continue
        if "m/z" not in df.columns or "Intensity" not in df.columns:
            continue
        mz = df["m/z"].to_numpy(dtype=float)
        intensity = df["Intensity"].to_numpy(dtype=float)

        for ref in reference_masses:
            mask = np.abs(mz - ref) < 1.0
            if mask.sum() < 5:
                continue
            local_y = intensity[mask]
            local_mz = mz[mask]
            pks, _ = find_peaks(local_y, prominence=np.nanmax(local_y) * 0.1)
            if pks.size == 0:
                continue
            best = pks[np.argmax(local_y[pks])]
            try:
                pw, _, _, _ = peak_widths(local_y, [best], rel_height=0.5)
                dx = np.median(np.diff(local_mz))
                if np.isfinite(pw[0]) and np.isfinite(dx) and dx > 0:
                    widths.append(float(pw[0] * dx))
            except Exception:
                continue

    if not widths:
        logger.info("estimate_autodetect_tolerance: no measurable peaks; using 0.5 Da fallback")
        return 0.5

    tol = float(np.clip(np.quantile(widths, quantile), 0.05, 2.0))
    logger.info("estimate_autodetect_tolerance: %.4f Da (from %d measurements)", tol, len(widths))
    return tol


# =====================================================================
# 2. Outlier threshold from residual distribution
# =====================================================================

def estimate_outlier_threshold(
    residuals: npt.NDArray[np.float64],
    *,
    target_false_rejection_rate: float = 0.01,
    bounds: Tuple[float, float] = (2.0, 5.0),
) -> float:
    """Derive ``outlier_threshold`` from observed residual spread.

    Uses the empirical quantile corresponding to
    ``1 - target_false_rejection_rate`` of absolute z-scores (MAD-scaled),
    clamped to *bounds*.
    """
    from scipy.stats import median_abs_deviation

    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 6:
        return 3.0

    mad = median_abs_deviation(r, scale="normal")
    if mad < 1e-12:
        return bounds[1]

    z = np.abs(r - np.median(r)) / mad
    q = float(np.quantile(z, 1.0 - target_false_rejection_rate))
    result = float(np.clip(q, *bounds))
    logger.info("estimate_outlier_threshold: %.2f (MAD=%.4f, n=%d)", result, mad, r.size)
    return result


# =====================================================================
# 3. Screening thresholds from batch residuals
# =====================================================================

def estimate_screening_thresholds(
    stability_df: pd.DataFrame,
    *,
    ppm_quantile: float = 0.85,
    valid_frac_quantile: float = 0.20,
) -> Dict[str, float]:
    """Derive ``screen_max_mean_abs_ppm`` and ``screen_min_valid_fraction``
    from a reference-mass stability table (output of
    ``summarize_reference_mass_stability``).

    Returns a dict with keys ``screen_max_mean_abs_ppm`` and
    ``screen_min_valid_fraction``.
    """
    result: Dict[str, float] = {}

    ppm_col = "mean_abs_ppm"
    if ppm_col in stability_df.columns:
        vals = stability_df[ppm_col].dropna().to_numpy(dtype=float)
        if vals.size > 0:
            result["screen_max_mean_abs_ppm"] = float(
                np.clip(np.quantile(vals, ppm_quantile), 5.0, 200.0)
            )

    frac_col = "valid_fraction"
    if frac_col in stability_df.columns:
        vals = stability_df[frac_col].dropna().to_numpy(dtype=float)
        if vals.size > 0:
            result["screen_min_valid_fraction"] = float(
                np.clip(np.quantile(vals, valid_frac_quantile), 0.3, 1.0)
            )

    logger.info("estimate_screening_thresholds: %s", result)
    return result


# =====================================================================
# 4. Multisegment breakpoints from calibrant distribution
# =====================================================================

def estimate_multisegment_breakpoints(
    reference_masses: Sequence[float],
    n_segments: int = 3,
) -> List[float]:
    """Place segment breakpoints at quantile boundaries of the reference
    mass range so each segment contains roughly equal calibrant counts.
    """
    masses = np.sort([m for m in reference_masses if np.isfinite(m) and m > 0])
    if masses.size < n_segments + 1:
        return [50.0, 200.0, 500.0]

    quantiles = np.linspace(0, 1, n_segments + 1)[1:-1]
    bps = [round(float(np.quantile(masses, q)), 1) for q in quantiles]
    logger.info("estimate_multisegment_breakpoints: %s", bps)
    return bps


# =====================================================================
# 5. Pipeline: batch-derived normalization target
# =====================================================================

def estimate_normalization_target(
    files: Sequence[str],
    *,
    sample_n: int = 20,
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
) -> float:
    """Estimate ``normalization_target`` as the median raw TIC across a
    sample of spectra.  Falls back to ``1e6`` on failure.
    """
    chosen = _sample_files(files, sample_n)
    tics: List[float] = []
    for fp in chosen:
        try:
            df = pd.read_csv(fp, sep="\t", header=0, comment="#")
        except Exception:
            continue
        if "Intensity" not in df.columns:
            continue
        y = df["Intensity"].to_numpy(dtype=float)
        if mz_min is not None and "m/z" in df.columns:
            mz = df["m/z"].to_numpy(dtype=float)
            mask = mz >= mz_min
            if mz_max is not None:
                mask &= mz <= mz_max
            y = y[mask]
        tic = float(np.nansum(y[y > 0]))
        if tic > 0:
            tics.append(tic)

    if not tics:
        logger.info("estimate_normalization_target: no valid spectra; using 1e6 fallback")
        return 1e6

    target = float(np.median(tics))
    logger.info("estimate_normalization_target: %.2e (from %d spectra)", target, len(tics))
    return target


# =====================================================================
# 6. Pipeline: adaptive mz_tolerance from channel spacing
# =====================================================================

def estimate_mz_tolerance(
    files: Sequence[str],
    *,
    sample_n: int = 10,
    multiplier: float = 3.0,
) -> float:
    """Estimate ``mz_tolerance`` from observed median m/z spacing, scaled
    by *multiplier*.  Clamped to [0.01, 1.0].
    """
    chosen = _sample_files(files, sample_n)
    spacings: List[float] = []
    for fp in chosen:
        try:
            df = pd.read_csv(fp, sep="\t", header=0, comment="#")
        except Exception:
            continue
        if "m/z" not in df.columns:
            continue
        mz = df["m/z"].to_numpy(dtype=float)
        mz = mz[np.isfinite(mz)]
        if mz.size < 2:
            continue
        d = np.diff(np.sort(mz))
        d = d[d > 0]
        if d.size > 0:
            spacings.append(float(np.median(d)))

    if not spacings:
        logger.info("estimate_mz_tolerance: no valid spectra; using 0.2 fallback")
        return 0.2

    tol = float(np.clip(np.median(spacings) * multiplier, 0.01, 1.0))
    logger.info("estimate_mz_tolerance: %.4f (from %d spectra)", tol, len(spacings))
    return tol


# =====================================================================
# 7. Baseline: adaptive FlatParams from sampling density
# =====================================================================

def estimate_flat_params(
    files: Sequence[str],
    *,
    sample_n: int = 10,
) -> Dict[str, object]:
    """Estimate ``savgol_window`` and quantile thresholds for
    ``FlatParams`` from the data.

    Returns a dict of keyword overrides suitable for
    ``dataclasses.replace(FlatParams(), **result)``.
    """
    chosen = _sample_files(files, sample_n)
    points_per_da: List[float] = []
    all_y: List[np.ndarray] = []

    for fp in chosen:
        try:
            df = pd.read_csv(fp, sep="\t", header=0, comment="#")
        except Exception:
            continue
        if "m/z" not in df.columns or "Intensity" not in df.columns:
            continue
        mz = df["m/z"].to_numpy(dtype=float)
        y = df["Intensity"].to_numpy(dtype=float)
        rng = np.nanmax(mz) - np.nanmin(mz)
        if rng > 0:
            points_per_da.append(mz.size / rng)
        if y.size > 0:
            all_y.append(y)

    result: Dict[str, object] = {}

    if points_per_da:
        median_ppd = float(np.median(points_per_da))
        window = int(max(5, round(median_ppd * 0.5)))
        if window % 2 == 0:
            window += 1
        result["savgol_window"] = window

    if all_y:
        pooled = np.concatenate(all_y)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size > 100:
            low_frac = float(np.sum(pooled <= np.quantile(pooled, 0.10)) / pooled.size)
            result["y_quantile"] = float(np.clip(low_frac * 2.0, 0.05, 0.40))

    logger.info("estimate_flat_params: %s", result)
    return result


# =====================================================================
# 8. Denoise: adaptive PSD cutoff and peak caps
# =====================================================================

def estimate_denoise_params(
    files: Sequence[str],
    *,
    sample_n: int = 5,
) -> Dict[str, object]:
    """Estimate ``hf_cutoff_frac`` and ``max_peaks`` for the denoise
    selection evaluator from pilot spectra.

    Returns a dict of keyword overrides for ``compare_denoising_methods``.
    """
    from scipy.signal import welch, find_peaks as _find_peaks

    chosen = _sample_files(files, sample_n)
    cutoff_fracs: List[float] = []
    peak_counts: List[int] = []

    for fp in chosen:
        try:
            df = pd.read_csv(fp, sep="\t", header=0, comment="#")
        except Exception:
            continue
        if "m/z" not in df.columns or "Intensity" not in df.columns:
            continue
        mz = df["m/z"].to_numpy(dtype=float)
        y = df["Intensity"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]

        dx = np.median(np.diff(mz[np.isfinite(mz)]))
        if not np.isfinite(dx) or dx <= 0:
            continue

        fs = 1.0 / dx
        nperseg = min(256, y.size // 2)
        if nperseg < 8:
            continue
        f, pxx = welch(y, fs=fs, nperseg=nperseg)
        if f.size < 4:
            continue

        log_pxx = np.log10(np.maximum(pxx, 1e-30))
        noise_floor = np.median(log_pxx[-max(1, len(log_pxx) // 4):])
        signal_above = log_pxx > noise_floor + 0.5
        if signal_above.any():
            last_signal = np.max(np.where(signal_above))
            nyq = 0.5 * fs
            cutoff_fracs.append(float(f[last_signal] / nyq))

        pks, _ = _find_peaks(y, prominence=np.nanstd(y) * 2.0)
        peak_counts.append(int(pks.size))

    result: Dict[str, object] = {}

    if cutoff_fracs:
        result["hf_cutoff_frac"] = float(np.clip(np.median(cutoff_fracs) * 1.2, 0.05, 0.8))

    if peak_counts:
        result["max_peaks"] = int(np.clip(int(np.median(peak_counts) * 1.5), 50, 2000))

    logger.info("estimate_denoise_params: %s", result)
    return result


# =====================================================================
# 9. Bootstrap heuristics from channel statistics
# =====================================================================

def estimate_bootstrap_heuristics(
    files: Sequence[str],
    *,
    sample_n: int = 10,
) -> Dict[str, float]:
    """Derive adaptive bootstrap peak-matching constants from observed
    channel statistics (noise, spacing, range).

    Returns a dict whose keys match the ``_BOOTSTRAP_*`` constant names
    in ``_models.py`` (without the leading underscore).
    """
    from scipy.stats import median_abs_deviation

    chosen = _sample_files(files, sample_n)
    channel_ranges: List[float] = []
    channel_lengths: List[int] = []

    for fp in chosen:
        try:
            df = pd.read_csv(fp, sep="\t", header=0, comment="#")
        except Exception:
            continue
        if "Channel" not in df.columns or "Intensity" not in df.columns:
            continue
        ch = df["Channel"].to_numpy(dtype=float)
        channel_ranges.append(float(np.nanmax(ch) - np.nanmin(ch)))
        channel_lengths.append(len(ch))

    result: Dict[str, float] = {}

    if channel_lengths:
        med_len = float(np.median(channel_lengths))
        result["BOOTSTRAP_PEAK_DISTANCE_DIVISOR"] = int(
            np.clip(med_len / 10.0, 500, 50000)
        )
        result["BOOTSTRAP_MIN_PEAK_DISTANCE_POINTS"] = int(
            np.clip(med_len / 5000.0, 3, 50)
        )

    if channel_ranges:
        med_range = float(np.median(channel_ranges))
        result["BOOTSTRAP_CHANNEL_MARGIN_MIN"] = float(
            np.clip(med_range * 0.01, 500.0, 50000.0)
        )
        result["BOOTSTRAP_K_GUESS_MIN"] = float(
            np.clip(med_range / 100.0, 100.0, 10000.0)
        )
        result["BOOTSTRAP_K_GUESS_MAX"] = float(
            np.clip(med_range * 2.0, 10000.0, 500000.0)
        )

    logger.info("estimate_bootstrap_heuristics: %s", result)
    return result


# =====================================================================
# 10. Unified auto-tune: returns a FlexibleCalibConfig
# =====================================================================

def auto_tune_calib_config(
    files: Sequence[str],
    reference_masses: Sequence[float],
    *,
    base_config=None,
    sample_n: int = 10,
):
    """Build a ``FlexibleCalibConfig`` with data-driven parameters.

    Starts from *base_config* (or the default) and replaces tolerance,
    screening values, and breakpoints with adaptive estimates.  The
    caller can further override any field afterwards.

    Returns a ``FlexibleCalibConfig`` instance.
    """
    from .recalibrate.flexible_calibrator import FlexibleCalibConfig

    if base_config is None:
        base_config = FlexibleCalibConfig(reference_masses=list(reference_masses))

    tol = estimate_autodetect_tolerance(files, reference_masses, sample_n=sample_n)
    bps = estimate_multisegment_breakpoints(reference_masses)

    overrides: Dict[str, object] = {
        "autodetect_tol_da": tol,
        "multisegment_breakpoints": bps,
        "auto_screen_reference_masses": True,
    }

    result = replace(base_config, **overrides)
    logger.info("auto_tune_calib_config: tolerance=%.4f Da, breakpoints=%s", tol, bps)
    return result


# =====================================================================
# Helpers
# =====================================================================

def _sample_files(files: Sequence[str], n: int) -> List[str]:
    """Deterministically sample up to *n* files spread across the list."""
    total = len(files)
    if total <= n:
        return list(files)
    indices = np.round(np.linspace(0, total - 1, n)).astype(int)
    return [files[i] for i in np.unique(indices)]
