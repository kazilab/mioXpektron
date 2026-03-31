"""
ToF-SIMS 1D denoising & smoothing utilities
-------------------------------------------------
This module provides a battery of denoising and smoothing methods for 1D ToF-SIMS spectra
and a *noise-aware* evaluation framework that selects methods based on both peak preservation
and explicit noise reduction criteria. It supports:

• Wavelet shrinkage (universal, SURE, Bayes, etc.) with optional variance-stabilizing transform (Anscombe).
• Classical smoothers (Savitzky–Golay, Gaussian, Median, and a no-op baseline).
• Robust, peak-centric measurements (height, FWHM, area, m/z shift) before/after denoising.
• Explicit noise quantification:
    – Global background σ̂ via MAD on baseline regions (outside expanded FWHM windows)
    – Local σ̂ around each peak (flanking bands) and ΔSNR per peak (in dB)
    – High-frequency residual power via PSD (Welch by default), integrated above a cutoff
• Parallel execution (threads or processes) and optional progress bars.
• Ranking with tunable weights and **hard constraints** to forbid methods that do not denoise.
• Convenience plotting (Pareto: ΔSNR vs |%height|) and multi-window evaluation helpers.

Design notes
------------
• Baseline regions are defined by excluding each reference peak’s FWHM window expanded by `baseline_expand`.
• Local noise is estimated in flanking bands at [1.5W, 3W] on each side of a peak by default.
• PSD is computed on *uniformly resampled* baseline segments to make frequency analysis well-defined.
• Threads are the recommended backend because most heavy NumPy/SciPy routines release the GIL.
"""

from __future__ import annotations

try:
    import polars as pl  # type: ignore
    _POLARS_AVAILABLE = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _POLARS_AVAILABLE = False

import os
import pandas as pd
from typing import Optional, Literal, Tuple, Dict, List, Any
from typing import Iterable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.integrate import trapezoid

from .denoise_main import noise_filtering
from pathlib import Path

OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Parallel/Progress helpers --
def _iter_with_progress(iterable: Iterable, total: int, progress: bool, desc: str):
    """Yield items, optionally wrapped with tqdm progress bar (no hard dep)."""
    if not progress:
        for x in iterable:
            yield x
        return
    try:
        from tqdm.auto import tqdm as _tqdm
        for x in _tqdm(iterable, total=total, desc=desc):
            yield x
    except Exception:
        # Fallback: no tqdm available
        for x in iterable:
            yield x



# --- Robust noise and SNR helpers ---

def _robust_sigma(a: np.ndarray) -> float:
    """Robust noise estimate via MAD (scaled to sigma). Returns NaN if empty."""
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return 1.4826 * mad


def _compute_baseline_mask(x: np.ndarray, ref_meas: List["PeakMeasurement"], expand: float = 2.0) -> NDArray[np.bool_]:
    """Boolean mask of baseline regions: exclude peak windows expanded by `expand`×width.
    True = baseline (non-peak) indices; False = inside expanded peak windows.
    """
    # Collect valid exclusion windows
    bounds = []
    for r in ref_meas:
        w = (r.mz_right - r.mz_left)
        if not np.isfinite(w) or w <= 0:
            continue
        half = 0.5 * expand * w
        bounds.append((r.mz_left - half, r.mz_right + half))

    if not bounds:
        return np.ones(x.size, dtype=bool)

    # Vectorized: stack all bounds and use broadcasting
    bounds_arr = np.array(bounds)  # (n_peaks, 2)
    # Check if x falls inside any exclusion window
    in_any = np.any(
        (x[np.newaxis, :] >= bounds_arr[:, 0:1]) &
        (x[np.newaxis, :] <= bounds_arr[:, 1:2]),
        axis=0
    )
    return ~in_any


def _local_noise_sigma(x: np.ndarray, y: np.ndarray, r: "PeakMeasurement",
                       flank_inner: float = 1.5, flank_outer: float = 3.0) -> float:
    """Estimate local noise using flanking bands outside the peak window.
    Bands: [left - outer*W, left - inner*W] and [right + inner*W, right + outer*W].
    Returns robust sigma (MAD*1.4826) or NaN when insufficient points.
    """
    W = (r.mz_right - r.mz_left)
    if not np.isfinite(W) or W <= 0:
        return np.nan
    l0 = r.mz_left - flank_outer * W
    l1 = r.mz_left - flank_inner * W
    r0 = r.mz_right + flank_inner * W
    r1 = r.mz_right + flank_outer * W
    ml = (x >= l0) & (x <= l1)
    mr = (x >= r0) & (x <= r1)
    band = ml | mr
    if not np.any(band):
        return np.nan
    return _robust_sigma(y[band])


def _ratio_db(num: float, den: float) -> float:
    """Return 20*log10(num/den) with robust guards; NaN if invalid."""
    if not (np.isfinite(num) and np.isfinite(den)):
        return np.nan
    if den <= 0 or num <= 0:
        return np.nan
    return 20.0 * float(np.log10(num / den))


def _resample_uniform(x: np.ndarray, y: np.ndarray, dx: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample (x,y) to a uniform grid using linear interpolation.
    Returns (xu, yu, fs) where fs is the sampling frequency (1/dx).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return x.copy(), y.copy(), np.nan
    x0, x1 = float(x[0]), float(x[-1])
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
        return x.copy(), y.copy(), np.nan
    dx0 = float(np.nanmedian(np.diff(x)))
    if not np.isfinite(dx0) or dx0 <= 0:
        return x.copy(), y.copy(), np.nan
    if dx is None or not np.isfinite(dx) or dx <= 0:
        dx = dx0
    n = int(np.floor((x1 - x0) / dx)) + 1
    xu = x0 + dx * np.arange(n, dtype=np.float64)
    yu = np.interp(xu, x, y)
    fs = 1.0 / dx
    return xu, yu, fs


def _hf_power(
    x: np.ndarray,
    y: np.ndarray,
    baseline_mask: np.ndarray,
    cutoff_hz: Optional[float] = None,
    cutoff_frac: float = 0.3,
    resample_dx: Optional[float] = None,
    psd_method: Literal["welch", "periodogram"] = "welch",
    welch_nperseg: Optional[int] = None,
) -> Tuple[float, float]:
    """High-frequency power and fraction in baseline regions.
    Resamples baseline to a uniform grid, estimates PSD (Welch by default),
    and integrates power above a cutoff.
    """
    if baseline_mask is None or not np.any(baseline_mask):
        return np.nan, np.nan
    xb = np.asarray(x, dtype=np.float64)[baseline_mask]
    yb = np.asarray(y, dtype=np.float64)[baseline_mask]
    if xb.size < 8:
        return np.nan, np.nan
    xu, yu, fs = _resample_uniform(xb, yb, dx=resample_dx)
    if not np.isfinite(fs) or yu.size < 16:
        return np.nan, np.nan
    yu = yu - np.mean(yu)

    # PSD estimation
    if psd_method == "welch":
        # Choose nperseg if not given (next power of two, up to 1024)
        n = yu.size
        if welch_nperseg is None:
            nper = 1
            while nper < max(16, n // 8):
                nper <<= 1
            nper = int(min(nper, 1024))
        else:
            nper = int(welch_nperseg)
        f, Pxx = signal.welch(yu, fs=fs, nperseg=nper, noverlap=nper//2, window="hann", scaling="density", detrend="constant")
    else:
        f, Pxx = signal.periodogram(yu, fs=fs, scaling='density', detrend='constant')

    if f.size == 0:
        return np.nan, np.nan
    nyq = 0.5 * fs
    fc = float(cutoff_hz) if (cutoff_hz is not None and np.isfinite(cutoff_hz)) else float(np.clip(cutoff_frac, 1e-6, 0.999999)) * nyq
    hf_mask = f >= fc
    if not np.any(hf_mask):
        return 0.0, 0.0
    total_power = float(trapezoid(Pxx, f)) if np.all(np.isfinite(Pxx)) else np.nan
    hf_power = float(trapezoid(Pxx[hf_mask], f[hf_mask])) if np.all(np.isfinite(Pxx[hf_mask])) else np.nan
    if not (np.isfinite(total_power) and total_power > 0):
        return hf_power, np.nan
    return hf_power, float(hf_power / total_power)


def _eval_method_task(
    name: str,
    cfg_slim: Dict[str, Any],
    x_arr: NDArray[np.float64],
    y_arr: NDArray[np.float64],
    ref_meas: List["PeakMeasurement"],
    rel_height: float,
    search_ppm: float,
    match_min_prominence_ratio: float,
    match_min_prominence_abs: float,
    match_min_width_pts: float,
    baseline_mask: NDArray[np.bool_],
    sigma_raw_global: float,
    local_sigma_raw: List[float],
    hf_enabled: bool,
    hf_cutoff_hz: Optional[float],
    hf_cutoff_frac: float,
    hf_resample_dx: Optional[float],
    hf_power_raw_global: float,
    flank_inner: float,
    flank_outer: float,
    hf_psd_method: Literal["welch", "periodogram"],
    hf_welch_nperseg: Optional[int],
) -> List[Dict[str, Any]]:
    """Run one denoising method and measure all reference peaks + noise metrics.

    Workflow
    --------
    1) Apply the denoiser (`noise_filtering`) with configuration `cfg_slim`.
    2) Compute a **global** background noise estimate on baseline regions (σ̂ via MAD) and the
       corresponding noise reduction in dB vs. RAW.
    3) Optionally compute **high-frequency residual power** on baseline regions via PSD and its
       reduction in dB (Welch by default).
    4) For each reference peak:
       – Re-measure peak on the denoised signal within a ppm window.
       – Compute % changes (height, FWHM, area), m/z shift, and local ΔSNR in dB using flanking bands.

    Returns
    -------
    list of dict
        One row per reference peak with peak stats and method-level noise metrics.
    """
    # Run denoiser
    y_hat = noise_filtering(y_arr, x=x_arr, **cfg_slim)

    # Global noise estimate (baseline regions only)
    try:
        sigma_new_global = _robust_sigma(y_hat[baseline_mask]) if np.any(baseline_mask) else _robust_sigma(y_hat)
    except Exception:
        sigma_new_global = np.nan
    noise_red_db = _ratio_db(sigma_raw_global, sigma_new_global)

    # High-frequency residual power (baseline regions only)
    if hf_enabled:
        try:
            hf_power_new, hf_frac_new = _hf_power(
                x_arr, y_hat, baseline_mask,
                cutoff_hz=hf_cutoff_hz, cutoff_frac=float(hf_cutoff_frac), resample_dx=hf_resample_dx,
                psd_method=hf_psd_method, welch_nperseg=hf_welch_nperseg,
            )
            hf_power_reduction_db = _ratio_db(hf_power_raw_global, hf_power_new)
        except Exception:
            hf_power_new = np.nan
            hf_frac_new = np.nan
            hf_power_reduction_db = np.nan
    else:
        hf_power_new = np.nan
        hf_frac_new = np.nan
        hf_power_reduction_db = np.nan

    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(ref_meas):
        m_meas = _measure_on_method(
            x_arr,
            y_hat,
            r,
            rel_height=rel_height,
            search_ppm=search_ppm,
            min_prominence_ratio=match_min_prominence_ratio,
            min_prominence_abs=match_min_prominence_abs,
            min_width_pts=match_min_width_pts,
        )
        matched = m_meas is not None

        # Local noise around this peak (post-denoise)
        sigma_loc_raw = local_sigma_raw[i] if i < len(local_sigma_raw) else np.nan
        try:
            sigma_loc_new = _local_noise_sigma(x_arr, y_hat, r, flank_inner=float(flank_inner), flank_outer=float(flank_outer))
        except Exception:
            sigma_loc_new = np.nan

        if matched:
            d_mz = m_meas.mz_center - r.mz_center
            ph = _percent_change(r.height, m_meas.height)
            pf = _percent_change(r.fwhm_pts, m_meas.fwhm_pts)
            pa = _percent_change(r.area, m_meas.area)
            mz_left, mz_right = m_meas.mz_left, m_meas.mz_right
            height_new = m_meas.height
            fwhm_new = m_meas.fwhm_pts
            area_new = m_meas.area
        else:
            d_mz = ph = pf = pa = np.nan
            mz_left = mz_right = np.nan
            height_new = fwhm_new = area_new = np.nan

        # SNRs and ΔSNR (dB)
        snr_raw = (r.height / sigma_loc_raw) if (np.isfinite(r.height) and np.isfinite(sigma_loc_raw) and sigma_loc_raw > 0) else np.nan
        snr_new = (height_new / sigma_loc_new) if (np.isfinite(height_new) and np.isfinite(sigma_loc_new) and sigma_loc_new > 0) else np.nan
        delta_snr_db = _ratio_db(snr_new, snr_raw)

        rows.append(dict(
            method=name,
            mz_ref=r.mz_center,
            mz_shift=d_mz,
            height_ref=r.height,
            height_new=height_new,
            fwhm_ref=r.fwhm_pts,
            fwhm_new=fwhm_new,
            area_ref=r.area,
            area_new=area_new,
            pct_height=ph, pct_fwhm=pf, pct_area=pa,
            mz_left=mz_left, mz_right=mz_right,
            matched=int(matched),
            # noise metrics
            sigma_raw_global=sigma_raw_global,
            sigma_new_global=sigma_new_global,
            noise_reduction_db=noise_red_db,
            sigma_local_raw=sigma_loc_raw,
            sigma_local_new=sigma_loc_new,
            snr_raw=snr_raw,
            snr_new=snr_new,
            delta_snr_db=delta_snr_db,
            hf_power_raw_global=hf_power_raw_global,
            hf_power_new=hf_power_new,
            hf_frac_new=hf_frac_new,
            hf_power_reduction_db=hf_power_reduction_db,
        ))
    return rows


# Top-level, picklable wrapper for process pools
def _task_wrapper(args: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """Picklable top-level wrapper for process pools.

    Unpacks an argument tuple and calls `_eval_method_task`. Using this function ensures
    compatibility with `ProcessPoolExecutor`, which requires picklable callables.
    """
    (
        name,
        cfg_slim,
        x_arr,
        y_arr,
        ref_meas,
        rel_height,
        search_ppm,
        match_min_prominence_ratio,
        match_min_prominence_abs,
        match_min_width_pts,
        baseline_mask,
        sigma_raw_global,
        local_sigma_raw,
        hf_enabled,
        hf_cutoff_hz,
        hf_cutoff_frac,
        hf_resample_dx,
        hf_power_raw_global,
        flank_inner,
        flank_outer,
        hf_psd_method,
        hf_welch_nperseg,
    ) = args
    return _eval_method_task(
        name, cfg_slim, x_arr, y_arr, ref_meas, rel_height, search_ppm,
        match_min_prominence_ratio, match_min_prominence_abs, match_min_width_pts,
        baseline_mask, sigma_raw_global, local_sigma_raw,
        hf_enabled, hf_cutoff_hz, hf_cutoff_frac, hf_resample_dx,
        hf_power_raw_global, flank_inner, flank_outer,
        hf_psd_method, hf_welch_nperseg,
    )


def _build_denoising_method_grid(
    x_arr: NDArray[np.float64],
    *,
    resample_to_uniform: bool,
    target_dx: Optional[float],
    include_derivatives: bool,
) -> Dict[str, Dict[str, Any]]:
    """Construct the candidate denoising-method grid for model selection."""
    methods: Dict[str, Dict[str, Any]] = {}

    for th in ("universal", "bayes", "sure", "sure_opt"):
        for sigstr in ("per_level", "global"):
            for wvlt in ("db4", "db8", "sym5", "sym8", "coif2", "coif3"):
                for cs in (0, 4, 8, 16, 32):
                    for vst in ("none", "anscombe"):
                        label = f"wavelet:{th}:{sigstr}:{wvlt}:{cs}:{vst}"
                        methods[label] = dict(
                            method="wavelet",
                            wavelet=wvlt,
                            level=None,
                            threshold_strategy=th,
                            threshold_mode="soft",
                            sigma=None,
                            sigma_strategy=sigstr,
                            variance_stabilize=vst,
                            cycle_spins=cs,
                            pywt_mode="periodization",
                            clip_nonnegative=True,
                            preserve_tic=False,
                            x=x_arr,
                            resample_to_uniform=resample_to_uniform,
                            target_dx=target_dx,
                        )

    sg_derivs = (0, 1, 2) if include_derivatives else (0,)
    for win_l in (10, 15, 20, 25, 50):
        for poly in (2, 3, 4, 5):
            wl = int(win_l)
            if wl % 2 == 0:
                wl += 1
            if wl <= poly:
                continue
            for deriv in sg_derivs:
                label = f"savitzky_golay:window_{wl}:poly_{poly}:deriv_{deriv}"
                methods[label] = dict(
                    method="savitzky_golay",
                    window_length=wl,
                    polyorder=poly,
                    deriv=deriv,
                    x=x_arr,
                    resample_to_uniform=resample_to_uniform,
                    target_dx=target_dx,
                )

    gaussian_orders = (0, 1, 2) if include_derivatives else (0,)
    for win_lg in (10, 15, 20, 25, 50):
        for gorder in gaussian_orders:
            label = f"gaussian:window_{win_lg}:order_{gorder}"
            methods[label] = dict(
                method="gaussian",
                window_length=win_lg,
                gauss_sigma_pts=max(1.0, win_lg / 6.0),
                gaussian_order=gorder,
                x=x_arr,
                resample_to_uniform=resample_to_uniform,
                target_dx=target_dx,
            )

    for win_lm in (10, 15, 20, 25, 50):
        label = f"median:window_{win_lm}"
        methods[label] = dict(
            method="median",
            window_length=win_lm,
            x=x_arr,
            resample_to_uniform=resample_to_uniform,
            target_dx=target_dx,
        )

    methods["none"] = dict(method="none", x=x_arr, resample_to_uniform=False, target_dx=None)
    return methods


@dataclass
class PeakMeasurement:
    """Container for per-peak measurements.

    Attributes
    ----------
    mz_center : float
        Peak center m/z (from the provided x-axis) at the local maximum index.
    idx_center : int
        Index of the peak center in the array (integer sample index).
    height : float
        Peak height (y at the local maximum) on the measured signal.
    fwhm_pts : float
        Full width at half maximum, measured in *index points* on the measured signal.
    mz_left, mz_right : float
        Left/right m/z boundaries where the peak crosses the chosen relative height (e.g., 50%).
    area : float
        Trapezoidal integral of y over [mz_left, mz_right].
    prominence : float
        Peak prominence measured on the same signal; used to reject weak
        shoulders when re-matching peaks after denoising.
    """
    mz_center: float
    idx_center: int
    height: float
    fwhm_pts: float
    mz_left: float
    mz_right: float
    area: float
    prominence: float = np.nan


def _ensure_axes(y: np.ndarray,
                 x: Optional[np.ndarray]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate inputs and return finite (x, y) arrays of equal shape.

    If `x` is None, a synthetic index axis [0, 1, ..., N-1] is created.
    NaNs/Infs are dropped *synchronously* from both arrays.

    Parameters
    ----------
    y : array-like
        Signal values (intensity). Will be cast to float64 and flattened.
    x : array-like or None
        m/z axis; if None, an index axis is used.

    Returns
    -------
    x_arr, y_arr : np.ndarray
        1D float64 arrays of equal length containing only finite samples.
    """
    # Normalize dtypes/shapes
    y = np.asarray(y, dtype=np.float64).ravel()
    if x is None:
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape != y.shape:
            raise ValueError("x and y must have same shape")
    # Keep only finite pairs
    m = np.isfinite(y) & np.isfinite(x)
    if not np.any(m):
        raise ValueError("No finite data in (x, y).")
    return x[m], y[m]


def _idx_to_mz(idx_f: float, x: np.ndarray) -> float:
    """Map a *floating* sample index to the x-axis (m/z) by linear interpolation.

    Parameters
    ----------
    idx_f : float
        Floating index position (e.g., left/right intercepts from `peak_widths`).
    x : np.ndarray
        Monotonic x-axis array.

    Returns
    -------
    float
        Interpolated m/z value at the requested fractional index.
    """
    n = x.size
    if idx_f <= 0:
        return float(x[0])
    if idx_f >= n - 1:
        return float(x[-1])
    i0 = int(np.floor(idx_f))
    t = idx_f - i0
    return float((1 - t) * x[i0] + t * x[i0 + 1])


def _measure_one_peak(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        peak_idx: int,
        rel_height: float = 0.5,
        prominence: Optional[float] = None,
) -> PeakMeasurement:
    """Measure a single peak at `peak_idx` on signal `y` using SciPy's half-height width.

    Steps
    -----
    1) Use `signal.peak_widths` at the requested `rel_height` to obtain left/right intercepts
       and the FWHM in *index points* on the **provided signal**.
    2) Convert fractional intercept indices back to m/z via linear interpolation on `x`.
    3) Integrate the area within the [left, right] span using `scipy.integrate.trapezoid`.

    Notes
    -----
    • Measurements are performed on the *current* y (raw or denoised), not a global template.
    • Returning FWHM in index points avoids ambiguity when x is nonuniform; boundaries are in m/z.
    """
    # width & boundaries at half height on THIS y
    _pw = signal.peak_widths(y, [peak_idx], rel_height=rel_height)
    widths = _pw[0]
    _ = _pw[1]  # unused: evaluation heights from peak_widths
    left_ips = _pw[2]
    right_ips = _pw[3]
    w = float(widths[0])
    lidx = float(left_ips[0])
    ridx = float(right_ips[0])

    # center & height at integer max location
    idx_center = int(peak_idx)
    mz_center = float(x[idx_center])
    height = float(y[idx_center])

    # boundaries in m/z
    mz_left = _idx_to_mz(lidx, x)
    mz_right = _idx_to_mz(ridx, x)

    # integrate area within [lidx, ridx] using trapezoid (interp to integer grid)
    i0 = max(0, int(np.floor(lidx)))
    i1 = min(x.size - 1, int(np.ceil(ridx)))
    xx = x[i0:i1+1]
    yy = y[i0:i1+1]
    area = float(trapezoid(yy, xx))
    if prominence is None:
        try:
            prominence = float(signal.peak_prominences(y, [peak_idx])[0][0])
        except (ValueError, RuntimeError, FloatingPointError, IndexError):
            prominence = np.nan

    return PeakMeasurement(
        mz_center=mz_center,
        idx_center=idx_center,
        height=height,
        fwhm_pts=w,
        mz_left=mz_left,
        mz_right=mz_right,
        area=area,
        prominence=float(prominence) if prominence is not None else np.nan,
    )


def _build_reference_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 300,
    min_prominence: Optional[float] = None,
    rel_height: float = 0.5,
) -> Tuple[np.ndarray, List[PeakMeasurement]]:
    """
    Detect reference peaks on RAW y (no denoise), then measure their FWHM windows.
    We keep the top 'max_peaks' by prominence.
    """
    # Robust default: let SciPy pick peaks, then keep strongest by prominence
    peaks, props = signal.find_peaks(y, prominence=min_prominence)
    if peaks.size == 0:
        raise ValueError("No peaks detected on raw data; adjust 'min_prominence' or pre-smooth lightly.")

    prom = props.get("prominences", np.zeros_like(peaks, dtype=float))
    order = np.argsort(prom)[::-1]  # descending by prominence
    keep = order[:max_peaks]
    peaks_ref = peaks[keep]

    prom_lookup = {int(pk): float(pm) for pk, pm in zip(peaks.tolist(), prom.tolist())}
    ref_measurements: List[PeakMeasurement] = []
    for p in peaks_ref:
        ref_measurements.append(
            _measure_one_peak(
                x,
                y,
                int(p),
                rel_height=rel_height,
                prominence=prom_lookup.get(int(p), np.nan),
            )
        )
    return peaks_ref, ref_measurements


def _measure_on_method(
    x: NDArray[np.float64],
    y_method: NDArray[np.float64],
    ref_meas: PeakMeasurement,
    rel_height: float = 0.5,
    search_ppm: float = 20.0,
    min_prominence_ratio: float = 0.1,
    min_prominence_abs: float = 0.0,
    min_width_pts: float = 0.25,
) -> Optional[PeakMeasurement]:
    """
    Re-measure a peak near the reference center within a ppm window on ``y_method``.

    Matching is intentionally conservative:
    - candidates must be true local maxima returned by ``find_peaks``
    - candidates must clear a minimum prominence relative to the reference peak
    - widths and enclosed areas must remain positive after re-measurement

    Returns ``None`` if no scientifically plausible match is found.
    """
    mz0 = float(ref_meas.mz_center)
    tol = float(mz0 * search_ppm * 1e-6)
    # ensure python floats to please static type checkers
    x0 = float(x[0])
    xN = float(x[-1])
    left = max(x0, mz0 - tol)
    right = min(xN, mz0 + tol)
    if right <= left:
        return None

    # restrict to indices within the search window
    m: NDArray[np.bool_] = (x >= left) & (x <= right)
    if not np.any(m):
        return None
    idx_window = np.nonzero(m)[0]
    ys = y_method[m]
    if ys.size < 3:
        return None

    # Only consider true local maxima inside the search window; this prevents a
    # monotonic shoulder or boundary sample from being counted as a preserved peak.
    peak_rel, props = signal.find_peaks(ys, prominence=0.0)
    if peak_rel.size == 0:
        return None

    prominences = np.asarray(props.get("prominences", np.full(peak_rel.size, np.nan)), dtype=float)
    ref_prom = float(ref_meas.prominence) if np.isfinite(ref_meas.prominence) else np.nan
    min_prom = float(min_prominence_abs)
    if np.isfinite(ref_prom) and ref_prom > 0:
        min_prom = max(min_prom, float(min_prominence_ratio) * ref_prom)

    valid = np.isfinite(prominences)
    if min_prom > 0:
        valid &= prominences >= min_prom
    if not np.any(valid):
        return None

    peak_rel = peak_rel[valid]
    prominences = prominences[valid]
    abs_idx = idx_window[peak_rel]
    peak_mz = x[abs_idx]

    # Prefer the closest peak to the reference center, then break ties by prominence.
    order = np.lexsort((-prominences, np.abs(peak_mz - mz0)))
    for ord_idx in order:
        idx_center = int(abs_idx[ord_idx])
        prominence = float(prominences[ord_idx])
        try:
            measured = _measure_one_peak(
                x,
                y_method,
                idx_center,
                rel_height=rel_height,
                prominence=prominence,
            )
        except (ValueError, RuntimeError, IndexError, FloatingPointError):
            continue

        if not np.isfinite(measured.fwhm_pts) or measured.fwhm_pts <= float(min_width_pts):
            continue
        if not np.isfinite(measured.area) or measured.area <= 0:
            continue
        if not (np.isfinite(measured.mz_left) and np.isfinite(measured.mz_right)):
            continue
        if measured.mz_right <= measured.mz_left:
            continue
        return measured

    return None


def _percent_change(a: float, b: float) -> float:
    """Percent change from reference `a` to method value `b`.

    Defined as `100 * (b - a) / a`. Returns NaN if `a` is zero or if either input is non-finite.
    Note: This metric is asymmetric; consider log-ratio or symmetric % if needed for analysis.
    """
    if a == 0 or not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    return 100.0 * (b - a) / a


def compare_denoising_methods(
    x: Optional[np.ndarray],
    y: np.ndarray,
    *,
    min_mz: Optional[float] = None,
    max_mz: Optional[float] = None,
    max_peaks: int = 300,
    min_prominence: Optional[float] = None,
    rel_height: float = 0.5,
    search_ppm: float = 20.0,
    match_min_prominence_ratio: float = 0.1,
    match_min_prominence_abs: float = 0.0,
    match_min_width_pts: float = 0.25,
    resample_to_uniform: bool = False,
    include_derivatives: bool = False,
    target_dx: Optional[float] = None,
    return_format: Literal["pandas", "polars"] = "pandas",
    n_jobs: int = -1,
    parallel_backend: Literal["thread", "process"] = "thread",
    progress: bool = True,
    # --- New knobs ---
    baseline_expand: float = 2.0,
    flank_inner: float = 1.5,
    flank_outer: float = 3.0,
    hf_enabled: bool = True,
    hf_cutoff_hz: Optional[float] = None,
    hf_cutoff_frac: float = 0.3,
    hf_resample_dx: Optional[float] = None,
    hf_psd_method: Literal["welch", "periodogram"] = "welch",
    hf_welch_nperseg: Optional[int] = None,
) -> Tuple[Any, Any]:
    """
    Run a battery of denoising/smoothing methods and quantify both **peak preservation** and **noise reduction**.

    Parameters
    ----------
    x : array-like or None
        m/z axis. If None, an index axis [0..N-1] is used.
    y : array-like
        Raw intensities.
    min_mz, max_mz : float, optional
        Optional range restriction on `x` prior to peak detection and evaluation.
    max_peaks : int, default 300
        Maximum number of reference peaks (by prominence) to evaluate in the selected range.
    min_prominence : float, optional
        Prominence threshold for `scipy.signal.find_peaks` during reference detection.
    rel_height : float, default 0.5
        Relative height used for FWHM measurements (e.g., 0.5 = half-height).
    search_ppm : float, default 20.0
        ±ppm window around each reference m/z used to re-detect peaks after denoising.
    match_min_prominence_ratio : float, default 0.1
        Minimum post-denoise prominence required for a matched peak, expressed
        as a fraction of the raw reference prominence.
    match_min_prominence_abs : float, default 0.0
        Absolute lower bound for post-denoise peak prominence.
    match_min_width_pts : float, default 0.25
        Minimum acceptable peak width in index points for a post-denoise match.
    resample_to_uniform : bool, default False
        If True, allow denoisers to resample to a uniform grid internally when beneficial.
    include_derivatives : bool, default False
        If True, include derivative-style Savitzky-Golay (`deriv>0`) and
        Gaussian (`order>0`) operators in the candidate grid. By default the
        search includes only smoothing/denoising variants.
    target_dx : float, optional
        Desired spacing when `resample_to_uniform=True`.
    return_format : {"pandas","polars"}, default "pandas"
        Backend for output DataFrames.
    n_jobs : int, default -1
        Number of workers used to evaluate methods in parallel (1 disables parallelism).
    parallel_backend : {"thread","process"}, default "thread"
        Parallelism backend. Threads are often efficient because NumPy/SciPy/PyWavelets drop the GIL.
    progress : bool, default True
        Show a progress bar if `tqdm` is available.

    baseline_expand : float, default 2.0
        Multiplier to expand each peak's FWHM window when masking baseline regions used for noise/PSD estimates.
    flank_inner, flank_outer : float, defaults 1.5 and 3.0
        Distances (in FWHM multiples) defining flanking bands used for **local** noise estimation.
    hf_enabled : bool, default True
        If True, compute high-frequency (HF) residual power metrics on baseline regions via PSD.
    hf_cutoff_hz : float, optional
        Absolute HF cutoff frequency (cycles per m/z). If None, uses `hf_cutoff_frac * Nyquist`.
    hf_cutoff_frac : float, default 0.3
        Fraction of the Nyquist frequency used as the HF band when `hf_cutoff_hz` is None.
    hf_resample_dx : float, optional
        Δx used to resample baseline segments to a uniform grid for PSD; defaults to median Δx if None.
    hf_psd_method : {"welch","periodogram"}, default "welch"
        PSD estimator for HF power. Welch provides lower-variance estimates on finite windows.
    hf_welch_nperseg : int, optional
        Segment length for Welch PSD. If None, chosen automatically (≈ max(16, N/8), power-of-two, ≤1024).

    Returns
    -------
    summary_df, per_peak_df : DataFrame
        If `return_format="pandas"`, returns pandas.DataFrame; if "polars", returns polars.DataFrame.
        `summary_df` contains method-level medians/IQRs, noise and HF metrics; `per_peak_df` has per-peak rows.
    """
    x_arr, y_arr = _ensure_axes(y, x)

    # Range restriction
    if min_mz is not None or max_mz is not None:
        lo_f = float(-np.inf) if min_mz is None else float(min_mz)
        hi_f = float(np.inf) if max_mz is None else float(max_mz)
        msel: NDArray[np.bool_] = (x_arr >= lo_f) & (x_arr <= hi_f)
        if not np.any(msel):
            raise ValueError("Selected m/z range contains no data.")
        x_arr, y_arr = x_arr[msel], y_arr[msel]

    # Reference peaks on raw
    ref_idx, ref_meas = _build_reference_peaks(
        x_arr, y_arr, max_peaks=max_peaks, min_prominence=min_prominence, rel_height=rel_height
    )
    # --- Noise baselines (computed once on RAW) ---
    baseline_mask = _compute_baseline_mask(x_arr, ref_meas, expand=float(baseline_expand))
    sigma_raw_global = _robust_sigma(y_arr[baseline_mask]) if np.any(baseline_mask) else _robust_sigma(y_arr)
    local_sigma_raw = [ _local_noise_sigma(x_arr, y_arr, r, flank_inner=float(flank_inner), flank_outer=float(flank_outer)) for r in ref_meas ]

    # --- High-frequency power baseline (RAW) ---
    if hf_enabled:
        hf_power_raw_global, hf_frac_raw_global = _hf_power(
            x_arr, y_arr, baseline_mask,
            cutoff_hz=hf_cutoff_hz, cutoff_frac=float(hf_cutoff_frac), resample_dx=hf_resample_dx,
            psd_method=hf_psd_method, welch_nperseg=hf_welch_nperseg,
        )
    else:
        hf_power_raw_global, hf_frac_raw_global = np.nan, np.nan

    methods = _build_denoising_method_grid(
        x_arr,
        resample_to_uniform=resample_to_uniform,
        target_dx=target_dx,
        include_derivatives=include_derivatives,
    )

    # --- Evaluate all methods (optionally in parallel). Process pools need a picklable wrapper.
    rows_peak: List[Dict[str, Any]] = []

    # Build lightweight tasks: drop large arrays from cfg to reduce pickling
    tasks: List[Tuple[str, Dict[str, Any]]] = []
    for name, cfg in methods.items():
        cfg_slim = dict(cfg)
        cfg_slim.pop("x", None)  # will be passed separately
        tasks.append((name, cfg_slim))

    total = len(tasks)
    # Build arg tuples once
    arg_tuples: List[Tuple[Any, ...]] = [
        (
            name, cfg_slim, x_arr, y_arr, ref_meas, rel_height, search_ppm,
            float(match_min_prominence_ratio), float(match_min_prominence_abs), float(match_min_width_pts),
            baseline_mask, sigma_raw_global, local_sigma_raw,
            hf_enabled, hf_cutoff_hz, hf_cutoff_frac, hf_resample_dx,
            hf_power_raw_global, float(flank_inner), float(flank_outer),
            hf_psd_method, hf_welch_nperseg,
        )
        for (name, cfg_slim) in tasks
    ]

    if n_jobs == 1:
        iterable = (_task_wrapper(args) for args in arg_tuples)
        for rows in _iter_with_progress(iterable, total=total, progress=progress, desc="Denoising methods"):
            rows_peak.extend(rows)
    else:
        if parallel_backend == "thread":
            from concurrent.futures import ThreadPoolExecutor as Executor
        elif parallel_backend == "process":
            from concurrent.futures import ProcessPoolExecutor as Executor
        else:
            raise ValueError("parallel_backend must be 'thread' or 'process'")

        _workers = os.cpu_count() or 4 if n_jobs <= 0 else int(n_jobs)
        with Executor(max_workers=_workers) as ex:
            mapped = ex.map(_task_wrapper, arg_tuples, chunksize=1)
            for rows in _iter_with_progress(mapped, total=total, progress=progress, desc=f"Denoising methods ({parallel_backend})"):
                rows_peak.extend(rows)

    if return_format == "pandas":
        import pandas as pd
        per_peak_df = pd.DataFrame(rows_peak)

        def _iqr(s: "pd.Series") -> float:
            """Return the interquartile range for a numeric pandas Series."""
            q = s.quantile([0.25, 0.75])
            return float(q.iloc[1] - q.iloc[0])

        summary = []
        for name, grp in per_peak_df.groupby("method"):
            gmatch = grp[grp["matched"] == 1]
            total = grp.shape[0]
            matched = gmatch.shape[0]
            lost = total - matched
            frac_matched = matched / total if total else np.nan

            # take method-level constants from first row
            gfirst = grp.iloc[0]
            sigma_raw_g = float(gfirst.get("sigma_raw_global", np.nan))
            sigma_new_g = float(gfirst.get("sigma_new_global", np.nan))
            noise_red_db = float(gfirst.get("noise_reduction_db", np.nan))
            hf_red_db = float(gfirst.get("hf_power_reduction_db", np.nan))
            hf_frac_new = float(gfirst.get("hf_frac_new", np.nan))

            summary.append(dict(
                method=name,
                peaks_total=total,
                peaks_matched=matched,
                peaks_lost=lost,
                frac_matched=frac_matched,
                mz_shift_med=float(gmatch["mz_shift"].median()) if matched else np.nan,
                mz_shift_iqr=_iqr(gmatch["mz_shift"]) if matched else np.nan,
                pct_height_med=float(gmatch["pct_height"].median()) if matched else np.nan,
                pct_height_iqr=_iqr(gmatch["pct_height"]) if matched else np.nan,
                pct_fwhm_med=float(gmatch["pct_fwhm"].median()) if matched else np.nan,
                pct_fwhm_iqr=_iqr(gmatch["pct_fwhm"]) if matched else np.nan,
                pct_area_med=float(gmatch["pct_area"].median()) if matched else np.nan,
                pct_area_iqr=_iqr(gmatch["pct_area"]) if matched else np.nan,
                # noise-aware metrics
                sigma_raw_global=sigma_raw_g,
                sigma_new_global=sigma_new_g,
                noise_reduction_db=noise_red_db,
                delta_snr_db_med=float(gmatch["delta_snr_db"].median()) if matched else np.nan,
                delta_snr_db_iqr=_iqr(gmatch["delta_snr_db"]) if matched else np.nan,
                hf_power_reduction_db=hf_red_db,
                hf_frac_new_global=hf_frac_new,
            ))

        summary_df = pd.DataFrame(summary).sort_values(["method"]).reset_index(drop=True)
        return summary_df, per_peak_df

    elif return_format == "polars":
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("polars is not installed. Install it or use return_format='pandas'.") from e

        per_peak_df = pl.DataFrame(rows_peak)
        summary_df = (
            per_peak_df
            .group_by("method")
            .agg([
                pl.count().alias("peaks_total"),
                pl.col("matched").sum().alias("peaks_matched"),
                (pl.count() - pl.col("matched").sum()).alias("peaks_lost"),
                (pl.col("matched").sum() / pl.count()).alias("frac_matched"),
                pl.col("mz_shift").filter((pl.col("matched") == 1) & ~pl.col("mz_shift").is_nan()).median().alias("mz_shift_med"),
                (
                    pl.col("mz_shift").filter((pl.col("matched") == 1) & ~pl.col("mz_shift").is_nan()).quantile(0.75)
                    - pl.col("mz_shift").filter((pl.col("matched") == 1) & ~pl.col("mz_shift").is_nan()).quantile(0.25)
                ).alias("mz_shift_iqr"),
                pl.col("pct_height").filter((pl.col("matched") == 1) & ~pl.col("pct_height").is_nan()).median().alias("pct_height_med"),
                (
                    pl.col("pct_height").filter((pl.col("matched") == 1) & ~pl.col("pct_height").is_nan()).quantile(0.75)
                    - pl.col("pct_height").filter((pl.col("matched") == 1) & ~pl.col("pct_height").is_nan()).quantile(0.25)
                ).alias("pct_height_iqr"),
                pl.col("pct_fwhm").filter((pl.col("matched") == 1) & ~pl.col("pct_fwhm").is_nan()).median().alias("pct_fwhm_med"),
                (
                    pl.col("pct_fwhm").filter((pl.col("matched") == 1) & ~pl.col("pct_fwhm").is_nan()).quantile(0.75)
                    - pl.col("pct_fwhm").filter((pl.col("matched") == 1) & ~pl.col("pct_fwhm").is_nan()).quantile(0.25)
                ).alias("pct_fwhm_iqr"),
                pl.col("pct_area").filter((pl.col("matched") == 1) & ~pl.col("pct_area").is_nan()).median().alias("pct_area_med"),
                (
                    pl.col("pct_area").filter((pl.col("matched") == 1) & ~pl.col("pct_area").is_nan()).quantile(0.75)
                    - pl.col("pct_area").filter((pl.col("matched") == 1) & ~pl.col("pct_area").is_nan()).quantile(0.25)
                ).alias("pct_area_iqr"),
                # noise-aware metrics (take first since constant per method)
                pl.col("sigma_raw_global").first().alias("sigma_raw_global"),
                pl.col("sigma_new_global").first().alias("sigma_new_global"),
                pl.col("noise_reduction_db").first().alias("noise_reduction_db"),
                pl.col("delta_snr_db").filter((pl.col("matched") == 1) & ~pl.col("delta_snr_db").is_nan()).median().alias("delta_snr_db_med"),
                (
                    pl.col("delta_snr_db").filter((pl.col("matched") == 1) & ~pl.col("delta_snr_db").is_nan()).quantile(0.75)
                    - pl.col("delta_snr_db").filter((pl.col("matched") == 1) & ~pl.col("delta_snr_db").is_nan()).quantile(0.25)
                ).alias("delta_snr_db_iqr"),
                pl.col("hf_power_reduction_db").first().alias("hf_power_reduction_db"),
                pl.col("hf_frac_new").first().alias("hf_frac_new_global"),
            ])
            .sort("method")
        )
        return summary_df, per_peak_df

    else:
        raise ValueError("return_format must be 'pandas' or 'polars'")


def aggregate_method_summaries(
    summary,
    *,
    unit_label: str = "windows",
    return_format: Literal["pandas", "polars"] = "pandas",
):
    """Aggregate per-method summaries across windows, spectra, or other units.

    The aggregation intentionally gives each unit equal weight by taking the
    median of unit-level metrics, while match fractions remain peak-weighted.
    This avoids a single busy window or spectrum dominating the ranking.
    """
    if pl is not None and isinstance(summary, pl.DataFrame):
        df = summary.to_pandas()
    else:
        df = summary.copy()

    if "method" not in df.columns:
        raise KeyError("summary must contain a 'method' column")

    metric_cols = (
        "mz_shift_med",
        "mz_shift_iqr",
        "pct_height_med",
        "pct_height_iqr",
        "pct_fwhm_med",
        "pct_fwhm_iqr",
        "pct_area_med",
        "pct_area_iqr",
        "sigma_raw_global",
        "sigma_new_global",
        "noise_reduction_db",
        "delta_snr_db_med",
        "delta_snr_db_iqr",
        "hf_power_reduction_db",
        "hf_frac_new_global",
    )

    def _median_col(group: "pd.DataFrame", col: str) -> float:
        if col not in group.columns:
            return np.nan
        s = pd.to_numeric(group[col], errors="coerce")
        return float(s.median()) if s.notna().any() else np.nan

    def _sum_int(group: "pd.DataFrame", col: str, default: int = 0) -> int:
        if col not in group.columns:
            return int(default)
        s = pd.to_numeric(group[col], errors="coerce")
        return int(s.fillna(0).sum())

    rows = []
    for method, grp in df.groupby("method", sort=True):
        peaks_total = _sum_int(grp, "peaks_total")
        peaks_matched = _sum_int(grp, "peaks_matched")
        peaks_lost = _sum_int(grp, "peaks_lost", default=max(peaks_total - peaks_matched, 0))
        row = {
            "method": method,
            unit_label: int(grp.shape[0]),
            "peaks_total": peaks_total,
            "peaks_matched": peaks_matched,
            "peaks_lost": peaks_lost,
            "frac_matched": (peaks_matched / peaks_total) if peaks_total else np.nan,
        }
        for col in metric_cols:
            row[col] = _median_col(grp, col)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    if return_format == "pandas":
        return out
    if return_format == "polars":
        if pl is None:
            raise ImportError("polars is not installed. Install it or use return_format='pandas'.")
        return pl.DataFrame(out)
    raise ValueError("return_format must be 'pandas' or 'polars'")


# Multi-window helper (stratified evaluation):
# a thin wrapper that loops over windows, tags results, and concatenates:


def compare_methods_in_windows(
    x: np.ndarray,
    y: np.ndarray,
    windows: list[tuple[float, float]],
    *,
    per_window_max_peaks: int = 50,
    min_prominence: Optional[float] = None,
    rel_height: float = 0.5,
    search_ppm: float = 20.0,
    match_min_prominence_ratio: float = 0.1,
    match_min_prominence_abs: float = 0.0,
    match_min_width_pts: float = 0.25,
    resample_to_uniform: bool = False,
    include_derivatives: bool = False,
    target_dx: Optional[float] = None,
    return_format: Literal["pandas", "polars"] = "pandas",
    n_jobs: int = -1,
    parallel_backend: Literal["thread","process"] = "thread",
    progress: bool = True,
    baseline_expand: float = 2.0,
    flank_inner: float = 1.5,
    flank_outer: float = 3.0,
    hf_enabled: bool = True,
    hf_cutoff_hz: Optional[float] = None,
    hf_cutoff_frac: float = 0.3,
    hf_resample_dx: Optional[float] = None,
    hf_psd_method: Literal["welch","periodogram"] = "welch",
    hf_welch_nperseg: Optional[int] = None,
    auto_tune: bool = False,
    auto_tune_files: Optional[list[str]] = None,
):
    """Evaluate denoising methods across multiple m/z windows and aggregate results.

    Parameters
    ----------
    x, y : np.ndarray
        m/z axis and intensity values.
    windows : list[tuple[float, float]]
        Each tuple is (min_mz, max_mz) for a window to evaluate.
    per_window_max_peaks : int, default 50
        Max number of strongest peaks (by prominence) to measure within each window.
    min_prominence : float, optional
        Minimum prominence passed to `signal.find_peaks` for reference peak detection.
    rel_height : float, default 0.5
        Relative height used to define FWHM when measuring peaks.
    search_ppm : float, default 20.0
        ±ppm window around each reference m/z used to re-detect peaks after denoising.
    match_min_prominence_ratio, match_min_prominence_abs, match_min_width_pts : floats
        Forwarded to the peak re-matching logic used after denoising.
    resample_to_uniform, target_dx : optional
        Passed through to denoisers that support resampling.
    include_derivatives : bool, default False
        If True, include derivative-style Savitzky-Golay and Gaussian
        candidates inside each window's method grid.
    return_format : {"pandas","polars"}
        Backend for output DataFrames.
    n_jobs : int, default -1
        Workers used within each window’s call to `compare_denoising_methods`.
    parallel_backend : {"thread","process"}, default "thread"
        Parallelism backend.
    progress : bool, default True
        Show progress bars during evaluation.
    baseline_expand, flank_inner, flank_outer : floats
        Baseline mask expansion and flanking-band multipliers forwarded to noise metrics.
    hf_enabled, hf_cutoff_hz, hf_cutoff_frac, hf_resample_dx, hf_psd_method, hf_welch_nperseg : optional
        High-frequency PSD controls forwarded to noise metrics (Welch is lower-variance).

    Returns
    -------
    If return_format == "pandas":
        rollup : pd.DataFrame
            Method-level aggregation across all windows.
        summary_all : pd.DataFrame
            Per-window, per-method summary table (noise and peak metrics).
        detail_all : pd.DataFrame
            Per-peak detail table across all windows.
    If return_format == "polars":
        rollup, summary_all, detail_all : pl.DataFrame
    """
    if auto_tune:
        from ..adaptive import estimate_denoise_params
        _dn_overrides = estimate_denoise_params(auto_tune_files or [])
        if "hf_cutoff_frac" in _dn_overrides:
            hf_cutoff_frac = float(_dn_overrides["hf_cutoff_frac"])
        if "max_peaks" in _dn_overrides:
            per_window_max_peaks = int(_dn_overrides["max_peaks"])

    if return_format == "pandas":
        import pandas as pd
        summaries = []
        details = []
        for (lo, hi) in windows:
            s, d = compare_denoising_methods(
                x, y,
                min_mz=lo, max_mz=hi,
                max_peaks=per_window_max_peaks,
                min_prominence=min_prominence,
                rel_height=rel_height,
                search_ppm=search_ppm,
                match_min_prominence_ratio=match_min_prominence_ratio,
                match_min_prominence_abs=match_min_prominence_abs,
                match_min_width_pts=match_min_width_pts,
                resample_to_uniform=resample_to_uniform,
                include_derivatives=include_derivatives,
                target_dx=target_dx,
                return_format="pandas",
                n_jobs=n_jobs, parallel_backend=parallel_backend, progress=progress,
                baseline_expand=baseline_expand,
                flank_inner=flank_inner,
                flank_outer=flank_outer,
                hf_enabled=hf_enabled,
                hf_cutoff_hz=hf_cutoff_hz,
                hf_cutoff_frac=hf_cutoff_frac,
                hf_resample_dx=hf_resample_dx,
                hf_psd_method=hf_psd_method,
                hf_welch_nperseg=hf_welch_nperseg,
            )
            s = s.copy()
            s["window"] = f"[{lo},{hi}]"
            d = d.copy()
            d["window"] = f"[{lo},{hi}]"
            summaries.append(s)
            details.append(d)

        summary_all = pd.concat(summaries, ignore_index=True)
        detail_all = pd.concat(details,   ignore_index=True)
        rollup = aggregate_method_summaries(
            summary_all,
            unit_label="windows",
            return_format="pandas",
        )
        return rollup, summary_all, detail_all

    elif return_format == "polars":
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("polars is not installed. Install it or use return_format='pandas'.") from e

        summaries: list[pl.DataFrame] = []
        details: list[pl.DataFrame] = []
        for (lo, hi) in windows:
            s, d = compare_denoising_methods(
                x, y,
                min_mz=lo, max_mz=hi,
                max_peaks=per_window_max_peaks,
                min_prominence=min_prominence,
                rel_height=rel_height,
                search_ppm=search_ppm,
                match_min_prominence_ratio=match_min_prominence_ratio,
                match_min_prominence_abs=match_min_prominence_abs,
                match_min_width_pts=match_min_width_pts,
                resample_to_uniform=resample_to_uniform,
                include_derivatives=include_derivatives,
                target_dx=target_dx,
                return_format="polars",
                n_jobs=n_jobs, parallel_backend=parallel_backend, progress=progress,
                baseline_expand=baseline_expand,
                flank_inner=flank_inner,
                flank_outer=flank_outer,
                hf_enabled=hf_enabled,
                hf_cutoff_hz=hf_cutoff_hz,
                hf_cutoff_frac=hf_cutoff_frac,
                hf_resample_dx=hf_resample_dx,
                hf_psd_method=hf_psd_method,
                hf_welch_nperseg=hf_welch_nperseg,
            )
            s = s.with_columns(pl.lit(f"[{lo},{hi}]").alias("window"))
            d = d.with_columns(pl.lit(f"[{lo},{hi}]").alias("window"))
            summaries.append(s)
            details.append(d)

        summary_all = pl.concat(summaries, how="vertical", rechunk=True)
        detail_all = pl.concat(details,   how="vertical", rechunk=True)
        rollup = aggregate_method_summaries(
            summary_all,
            unit_label="windows",
            return_format="polars",
        )
        return rollup, summary_all, detail_all

    else:
        raise ValueError("return_format must be 'pandas' or 'polars'")

# Rank methods

def to_ppm(mz_shift_med: float, mz_ref_median: float) -> float:
    """Convert an absolute m/z shift to parts-per-million (ppm).

    ppm = 1e6 * Δm / m_ref
    """
    return 1e6 * mz_shift_med / mz_ref_median


DEFAULT_SELECTION_CRITERIA: Dict[str, float] = {
    "min_frac_matched": 0.80,
    "max_mz_shift_ppm": 10.0,
    "max_mz_shift_iqr_ppm": 10.0,
    "max_pct_area_med": 25.0,
    "max_pct_height_med": 25.0,
    "max_pct_fwhm_med": 25.0,
    "max_pct_area_iqr": 30.0,
    "max_pct_height_iqr": 30.0,
    "max_pct_fwhm_iqr": 30.0,
    "target_noise_db": 3.0,
    "target_delta_snr_db": 3.0,
    "target_hf_power_reduction_db": 3.0,
    "target_hf_frac_new_global": 0.25,
}


def _resolve_selection_criteria(
    selection_criteria: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Validate and merge dimensionless-selection criteria."""
    criteria = dict(DEFAULT_SELECTION_CRITERIA)
    if selection_criteria is not None:
        unknown = set(selection_criteria) - set(criteria)
        if unknown:
            raise KeyError(f"Unknown selection criteria: {sorted(unknown)}")
        criteria.update(selection_criteria)

    for key, value in criteria.items():
        if not np.isfinite(value):
            raise ValueError(f"selection criterion '{key}' must be finite")

    if not (0.0 < criteria["min_frac_matched"] <= 1.0):
        raise ValueError("selection criterion 'min_frac_matched' must lie in (0, 1]")

    positive_keys = set(criteria) - {"min_frac_matched"}
    for key in positive_keys:
        if criteria[key] <= 0:
            raise ValueError(f"selection criterion '{key}' must be > 0")
    return criteria


def _prepare_ranking_frame_pandas(
    summary_df,
    per_peak_df,
    *,
    min_noise_db: float,
    min_delta_snr_db: float,
    selection_criteria: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Add dimensionless ranking columns and pre-specified pass/fail gates."""
    criteria = _resolve_selection_criteria(selection_criteria)
    s = _to_pandas_df(summary_df)
    peak_df = _to_pandas_df(per_peak_df)

    numeric_cols = (
        "frac_matched",
        "mz_shift_med",
        "mz_shift_iqr",
        "pct_area_med",
        "pct_area_iqr",
        "pct_height_med",
        "pct_height_iqr",
        "pct_fwhm_med",
        "pct_fwhm_iqr",
        "noise_reduction_db",
        "delta_snr_db_med",
        "hf_power_reduction_db",
        "hf_frac_new_global",
    )
    for col in numeric_cols:
        if col not in s.columns:
            s[col] = np.nan
        s[col] = pd.to_numeric(s[col], errors="coerce")

    for col in ("noise_reduction_db", "delta_snr_db_med", "hf_power_reduction_db", "hf_frac_new_global"):
        s[col] = s[col].fillna(0.0)

    mz_ref_median = np.nan
    if "mz_ref" in peak_df.columns:
        mz_ref = pd.to_numeric(peak_df["mz_ref"], errors="coerce").to_numpy(dtype=float)
        if mz_ref.size and np.isfinite(mz_ref).any():
            mz_ref_median = float(np.nanmedian(mz_ref))
    ppm_scale = np.nan
    if np.isfinite(mz_ref_median) and mz_ref_median != 0:
        ppm_scale = 1e6 / mz_ref_median

    s["mz_shift_ppm"] = s["mz_shift_med"].abs() * ppm_scale
    s["mz_shift_iqr_ppm"] = s["mz_shift_iqr"].abs() * ppm_scale
    s["abs_area"] = s["pct_area_med"].abs()
    s["abs_height"] = s["pct_height_med"].abs()
    s["abs_fwhm"] = s["pct_fwhm_med"].abs()
    s["abs_area_iqr"] = s["pct_area_iqr"].abs()
    s["abs_height_iqr"] = s["pct_height_iqr"].abs()
    s["abs_fwhm_iqr"] = s["pct_fwhm_iqr"].abs()

    s["scaled_mz_shift"] = s["mz_shift_ppm"] / criteria["max_mz_shift_ppm"]
    s["scaled_mz_shift_iqr"] = s["mz_shift_iqr_ppm"] / criteria["max_mz_shift_iqr_ppm"]
    s["scaled_abs_area"] = s["abs_area"] / criteria["max_pct_area_med"]
    s["scaled_abs_height"] = s["abs_height"] / criteria["max_pct_height_med"]
    s["scaled_abs_fwhm"] = s["abs_fwhm"] / criteria["max_pct_fwhm_med"]
    s["scaled_abs_area_iqr"] = s["abs_area_iqr"] / criteria["max_pct_area_iqr"]
    s["scaled_abs_height_iqr"] = s["abs_height_iqr"] / criteria["max_pct_height_iqr"]
    s["scaled_abs_fwhm_iqr"] = s["abs_fwhm_iqr"] / criteria["max_pct_fwhm_iqr"]
    s["spread_sum_scaled"] = s[
        [
            "scaled_mz_shift_iqr",
            "scaled_abs_area_iqr",
            "scaled_abs_height_iqr",
            "scaled_abs_fwhm_iqr",
        ]
    ].sum(axis=1, min_count=1)
    s["scaled_noise_reduction"] = s["noise_reduction_db"] / criteria["target_noise_db"]
    s["scaled_delta_snr"] = s["delta_snr_db_med"] / criteria["target_delta_snr_db"]
    s["scaled_hf_power_reduction"] = (
        s["hf_power_reduction_db"] / criteria["target_hf_power_reduction_db"]
    )
    s["scaled_hf_frac"] = s["hf_frac_new_global"] / criteria["target_hf_frac_new_global"]

    pass_checks = {
        "pass_frac_matched": s["frac_matched"] >= criteria["min_frac_matched"],
        "pass_mz_shift_ppm": s["mz_shift_ppm"] <= criteria["max_mz_shift_ppm"],
        "pass_mz_shift_iqr_ppm": s["mz_shift_iqr_ppm"] <= criteria["max_mz_shift_iqr_ppm"],
        "pass_abs_area": s["abs_area"] <= criteria["max_pct_area_med"],
        "pass_abs_height": s["abs_height"] <= criteria["max_pct_height_med"],
        "pass_abs_fwhm": s["abs_fwhm"] <= criteria["max_pct_fwhm_med"],
        "pass_abs_area_iqr": s["abs_area_iqr"] <= criteria["max_pct_area_iqr"],
        "pass_abs_height_iqr": s["abs_height_iqr"] <= criteria["max_pct_height_iqr"],
        "pass_abs_fwhm_iqr": s["abs_fwhm_iqr"] <= criteria["max_pct_fwhm_iqr"],
        "pass_noise_reduction": s["noise_reduction_db"] >= float(min_noise_db),
        "pass_delta_snr": s["delta_snr_db_med"] >= float(min_delta_snr_db),
    }
    for col, mask in pass_checks.items():
        s[col] = mask

    s["passes_peak_preservation"] = (
        s["pass_frac_matched"]
        & s["pass_mz_shift_ppm"]
        & s["pass_mz_shift_iqr_ppm"]
        & s["pass_abs_area"]
        & s["pass_abs_height"]
        & s["pass_abs_fwhm"]
        & s["pass_abs_area_iqr"]
        & s["pass_abs_height_iqr"]
        & s["pass_abs_fwhm_iqr"]
    )
    s["passes_min_denoise"] = s["pass_noise_reduction"] & s["pass_delta_snr"]
    s["passes_selection_criteria"] = s["passes_peak_preservation"] & s["passes_min_denoise"]
    fail_cols = [col for col in pass_checks if col.startswith("pass_")]
    s["failed_criteria_count"] = (~s[fail_cols]).sum(axis=1)
    return s


def rank_methods_pandas(summary_df,
                        per_peak_df,
                        w_match=3.0, w_mz=2.0, w_area=2.0, w_height=1.5, w_fwhm=1.0,
                        w_spread=1.0,
                        w_noise_db=2.0,
                        w_delta_snr_db=1.5,
                        w_hf_db=1.5,
                        w_hf_frac=1.0,
                        min_noise_db=0.5,
                        min_delta_snr_db=1.0,
                        selection_criteria: Optional[Dict[str, float]] = None):
    """Rank methods using explicit gates plus a dimensionless tie-break score.

    The primary scientific selection rule is:
    1. Require peak-preservation and minimum-denoising criteria to pass.
    2. Use a dimensionless score only as a secondary tie-break, not as the
       primary scientific claim.
    """
    s = _prepare_ranking_frame_pandas(
        summary_df,
        per_peak_df,
        min_noise_db=min_noise_db,
        min_delta_snr_db=min_delta_snr_db,
        selection_criteria=selection_criteria,
    )

    penalty_fill = 10.0
    penalty = (
        w_mz * s["scaled_mz_shift"].fillna(penalty_fill)
        + w_area * s["scaled_abs_area"].fillna(penalty_fill)
        + w_height * s["scaled_abs_height"].fillna(penalty_fill)
        + w_fwhm * s["scaled_abs_fwhm"].fillna(penalty_fill)
        + w_spread * s["spread_sum_scaled"].fillna(penalty_fill)
        + w_hf_frac * s["scaled_hf_frac"].fillna(penalty_fill)
    )
    reward = (
        w_match * s["frac_matched"].fillna(0.0)
        + w_noise_db * s["scaled_noise_reduction"].fillna(0.0)
        + w_delta_snr_db * s["scaled_delta_snr"].fillna(0.0)
        + w_hf_db * s["scaled_hf_power_reduction"].fillna(0.0)
    )
    s["selection_penalty"] = penalty
    s["selection_reward"] = reward
    s["selection_score"] = penalty - reward
    s["score"] = s["selection_score"]

    fail_mask = ~s["passes_selection_criteria"]
    s.loc[fail_mask, "score"] = s.loc[fail_mask, "score"] + 1_000_000.0
    s = s.sort_values(
        ["score", "delta_snr_db_med", "frac_matched"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return s


def rank_methods_polars(summary_df: "pl.DataFrame", per_peak_df: "pl.DataFrame",
                        w_match=3.0, w_mz=2.0, w_area=2.0, w_height=1.5, w_fwhm=1.0,
                        w_spread=1.0,
                        w_noise_db=2.0,
                        w_delta_snr_db=1.5,
                        w_hf_db=1.5,
                        w_hf_frac=1.0,
                        min_noise_db: float = 0.5,
                        min_delta_snr_db: float = 1.0,
                        selection_criteria: Optional[Dict[str, float]] = None) -> "pl.DataFrame":
    """Rank methods (polars) using the pandas implementation for identical semantics."""
    if pl is None:
        raise ImportError("polars is not installed. Install it or use rank_methods_pandas instead.")
    ranked = rank_methods_pandas(
        summary_df.to_pandas(),
        per_peak_df.to_pandas(),
        w_match=w_match,
        w_mz=w_mz,
        w_area=w_area,
        w_height=w_height,
        w_fwhm=w_fwhm,
        w_spread=w_spread,
        w_noise_db=w_noise_db,
        w_delta_snr_db=w_delta_snr_db,
        w_hf_db=w_hf_db,
        w_hf_frac=w_hf_frac,
        min_noise_db=min_noise_db,
        min_delta_snr_db=min_delta_snr_db,
        selection_criteria=selection_criteria,
    )
    return pl.DataFrame(ranked)



def rank_method(input_format, summary_df, per_peak_df,
                w_match=3.0, w_mz=2.0, w_area=2.0, w_height=1.5, w_fwhm=1.0,
                w_spread=1.0,
                w_noise_db=2.0,
                w_delta_snr_db=1.5,
                w_hf_db=1.5,
                w_hf_frac=1.0,
                min_noise_db: float = 0.5,
                min_delta_snr_db: float = 1.0,
                selection_criteria: Optional[Dict[str, float]] = None):
    """Dispatch ranking to pandas or polars implementation with identical semantics.

    Returns a DataFrame (pandas or polars) sorted by ascending `score` and includes
    explicit pass/fail flags for denoising and peak-preservation criteria.
    """
    if input_format == 'pandas':
        summary = rank_methods_pandas(summary_df, per_peak_df,
                        w_match=w_match, w_mz=w_mz, w_area=w_area,
                        w_height=w_height, w_fwhm=w_fwhm,
                        w_spread=w_spread,
                        w_noise_db=w_noise_db,
                        w_delta_snr_db=w_delta_snr_db,
                        w_hf_db=w_hf_db,
                        w_hf_frac=w_hf_frac,
                        min_noise_db=min_noise_db,
                        min_delta_snr_db=min_delta_snr_db,
                        selection_criteria=selection_criteria)
    elif input_format == 'polars':
        summary = rank_methods_polars(summary_df, per_peak_df,
                        w_match=w_match, w_mz=w_mz, w_area=w_area,
                        w_height=w_height, w_fwhm=w_fwhm,
                        w_spread=w_spread,
                        w_noise_db=w_noise_db,
                        w_delta_snr_db=w_delta_snr_db,
                        w_hf_db=w_hf_db,
                        w_hf_frac=w_hf_frac,
                        min_noise_db=min_noise_db,
                        min_delta_snr_db=min_delta_snr_db,
                        selection_criteria=selection_criteria)
    else:
        raise ValueError("input_format must be 'pandas' or 'polars'")
    return summary

# ---------------- Pareto visualization -----------------


def decode_method_label(label: str) -> dict:
    """Translate a compact label back into ``noise_filtering`` parameters."""
    family, *parts = label.split(":")
    if family == "wavelet":
        thresh, sigma, wavelet, spins, vst = parts
        return {
            "method": "wavelet",
            "threshold_strategy": thresh,
            "sigma_strategy": sigma,
            "wavelet": wavelet,
            "cycle_spins": int(spins),
            "variance_stabilize": vst,
            # defaults the grid used (see Xpektrion/denoise/denoise_select.py:708-724)
            "threshold_mode": "soft",
            "pywt_mode": "periodization",
            "clip_nonnegative": True,
            "preserve_tic": False,
        }
    if family == "savitzky_golay":
        return {
            "method": "savitzky_golay",
            **{k: int(v) for k, v in (p.split("_", 1) for p in parts)}
        }
    if family == "gaussian":
        return {
            "method": "gaussian",
            "window_length": int(parts[0].split("_", 1)[1]),
            "gaussian_order": int(parts[1].split("_", 1)[1]),
            # sigma is derived as window/6 inside the evaluator
        }
    if family == "median":
        return {"method": "median", "window_length": int(parts[0].split("_", 1)[1])}
    if family == "none":
        return {"method": "none"}
    raise ValueError(f"Unknown label '{label}'")


def _to_pandas_df(summary):
    """Coerce common tabular inputs into a pandas ``DataFrame`` copy."""
    try:
        if isinstance(summary, pl.DataFrame):
            return summary.to_pandas()
    except Exception:
        pass
    return summary.copy() if hasattr(summary, "copy") else pd.DataFrame(summary)
    

def _normalize_and_filter(summary,
                          require_pass=True,
                          require_finite_metrics=True):
    """
    Convert summary to pandas, coerce numerics, and apply the *same* filters.
    """

    df = _to_pandas_df(summary)

    # Required columns for your use cases
    req = {"method"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"summary missing required columns: {sorted(missing)}")

    # Coerce numerics that the plot and ranking use
    if "pct_height_med" in df.columns:
        df["pct_height_med"] = pd.to_numeric(df["pct_height_med"], errors="coerce")
        df["abs_height"] = df["pct_height_med"].abs()
    if "frac_matched" in df.columns:
        df["frac_matched"] = pd.to_numeric(df["frac_matched"], errors="coerce")
    if "delta_snr_db_med" in df.columns:
        df["delta_snr_db_med"] = pd.to_numeric(df["delta_snr_db_med"], errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "selection_score" in df.columns:
        df["selection_score"] = pd.to_numeric(df["selection_score"], errors="coerce")

    # Apply *identical* row filters everywhere
    mask = pd.Series(True, index=df.index)
    if require_pass:
        if "passes_selection_criteria" in df.columns:
            mask &= (df["passes_selection_criteria"] == True)  # noqa: E712
        elif "passes_min_denoise" in df.columns:
            mask &= (df["passes_min_denoise"] == True)  # noqa: E712

    if require_finite_metrics:
        # These are needed for Pareto and usually for quality ranking
        needed = []
        if "abs_height" in df.columns:
            needed.append(np.isfinite(df["abs_height"]))
        if "delta_snr_db_med" in df.columns:
            needed.append(np.isfinite(df["delta_snr_db_med"]))
        if needed:
            m = needed[0]
            for n in needed[1:]:
                m &= n
            mask &= m

    df = df[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after normalization/filters")
    return df


def _compute_pareto_mask(df):
    """
    Non-dominated w.r.t minimize x=abs_height and maximize y=delta_snr_db_med.
    Returns boolean mask of length len(df).
    """
    if not {"abs_height", "delta_snr_db_med"} <= set(df.columns):
        raise ValueError("Pareto needs abs_height and delta_snr_db_med")
    x = df["abs_height"].to_numpy()
    y = df["delta_snr_db_med"].to_numpy()
    n = len(df)
    if n == 0:
        return np.ones(0, dtype=bool)
    # Vectorized: for each point i, check if any other point j dominates it
    # j dominates i iff: y[j] >= y[i] AND x[j] <= x[i] AND (y[j] > y[i] OR x[j] < x[i])
    # Use broadcasting: (n, 1) vs (1, n)
    y_ge = y[:, np.newaxis] >= y[np.newaxis, :]  # (j, i): y[j] >= y[i]
    x_le = x[:, np.newaxis] <= x[np.newaxis, :]  # (j, i): x[j] <= x[i]
    strict = (y[:, np.newaxis] > y[np.newaxis, :]) | (x[:, np.newaxis] < x[np.newaxis, :])
    dominated_by = y_ge & x_le & strict  # (j, i): j dominates i
    # Zero out self-domination (diagonal)
    np.fill_diagonal(dominated_by, False)
    # Point i is non-dominated if no j dominates it
    nd = ~dominated_by.any(axis=0)
    return nd


def select_methods(summary,
                   basis="constrained_pareto_then_snr",
                   top_k=12,
                   require_pass=True,
                   require_finite_metrics=True):
    """
    Returns:
      filtered_df: the post-filter DataFrame (shared!)
      frontier_df: DataFrame of Pareto points (or None if basis='score' and Pareto not computed)
      selected_df: the DataFrame of selected rows to annotate/return (top_k)
    """
    df = _normalize_and_filter(summary, require_pass, require_finite_metrics)

    frontier_df = None
    if basis == "score":
        if "score" not in df.columns:
            raise ValueError("basis='score' requires a 'score' column")
        selected_df = df.sort_values(["score", "delta_snr_db_med"], ascending=[True, False]).head(top_k).copy()

    elif basis in {"constrained_pareto_then_snr", "pareto_then_snr"}:
        nd_mask = _compute_pareto_mask(df)
        frontier_df = df[nd_mask].copy()
        frontier_df = frontier_df.sort_values(["abs_height", "delta_snr_db_med"], ascending=[True, False])
        frontier_df = frontier_df.drop_duplicates(subset=["abs_height"], keep="first")

        order = ["delta_snr_db_med"]
        ascending = [False]
        if "frac_matched" in frontier_df.columns:
            order.append("frac_matched")
            ascending.append(False)
        if "selection_score" in frontier_df.columns:
            order.append("selection_score")
            ascending.append(True)
        elif "score" in frontier_df.columns:
            order.append("score")
            ascending.append(True)
        selected_df = frontier_df.sort_values(order, ascending=ascending).head(top_k).copy()

    elif basis == "pareto_then_score":
        nd_mask = _compute_pareto_mask(df)
        frontier_df = df[nd_mask].copy()
        # Tidy frontier ordering for plotting
        frontier_df = frontier_df.sort_values(["abs_height", "delta_snr_db_med"], ascending=[True, False])
        frontier_df = frontier_df.drop_duplicates(subset=["abs_height"], keep="first")

        # Pick what to annotate/”best” from inside the frontier
        if "score" in df.columns:
            selected_df = frontier_df.sort_values(["score", "delta_snr_db_med"], ascending=[True, False]).head(top_k).copy()
        else:
            selected_df = frontier_df.sort_values("delta_snr_db_med", ascending=False).head(top_k).copy()
    else:
        raise ValueError(
            "basis must be one of "
            "'constrained_pareto_then_snr', 'pareto_then_snr', 'pareto_then_score', or 'score'"
        )

    return df, frontier_df, selected_df


def plot_pareto_delta_snr_vs_height(summary, annotate=True, top_k=12,
                                    out_path=None, ax=None,
                                    basis="constrained_pareto_then_snr",
                                    require_pass=True, require_finite_metrics=True,
                                    save_plot=True, save_pareto=True):
    """Render ΔSNR vs. |%height| with Pareto annotations.

    Parameters mirror :func:`select_methods`; see :class:`DenoisingMethods.plot`
    for additional discussion. The helper creates the Matplotlib figure when
    ``ax`` is omitted and optionally saves both the chart and frontier table.
    """
    import matplotlib.pyplot as plt

    relaxed_for_plot = False
    try:
        df, frontier_df, selected_df = select_methods(
            summary, basis=basis, top_k=top_k,
            require_pass=require_pass, require_finite_metrics=require_finite_metrics
        )
    except ValueError as exc:
        if require_pass and "No rows left after normalization/filters" in str(exc):
            df, frontier_df, selected_df = select_methods(
                summary,
                basis=basis,
                top_k=top_k,
                require_pass=False,
                require_finite_metrics=require_finite_metrics,
            )
            relaxed_for_plot = True
        else:
            raise

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True

    ax.scatter(df["abs_height"], df["delta_snr_db_med"], alpha=0.6, label="All candidates")

    if frontier_df is not None:
        ax.plot(frontier_df["abs_height"], frontier_df["delta_snr_db_med"],
                linewidth=2, marker="o", label="Pareto front")

    ax.set_xlabel("|% height change| (median)")
    ax.set_ylabel("ΔSNR (dB, median)")
    title = "Pareto: ΔSNR vs. |%height| (lower x, higher y is better)"
    if relaxed_for_plot:
        title += "\nNo methods passed the current selection criteria; showing all finite candidates"
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()

    if annotate and not selected_df.empty:
        for _, r in selected_df.iterrows():
            ax.annotate(str(r["method"]),
                        (r["abs_height"], r["delta_snr_db_med"]),
                        xytext=(4, 4), textcoords="offset points")

    if save_plot:
        out_path = OUTPUT_DIR / f"Pareto_plot_top_{top_k}.pdf"
        ax.figure.savefig(out_path, bbox_inches="tight")
    if save_pareto and frontier_df is not None:
        out_path = OUTPUT_DIR / f"Pareto_front_{top_k}.xlsx"
        if _POLARS_AVAILABLE and isinstance(frontier_df, pl.DataFrame):
            frontier_df.write_excel(out_path)
        else:
            frontier_df.to_excel(out_path)
    return ax
