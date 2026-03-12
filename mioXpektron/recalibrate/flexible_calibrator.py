"""
Enhanced Flexible Calibration Module for High-Resolution Mass Spectrometry

Provides manual selection of calibration methods with enhanced features:
- Robust fitting with outlier rejection
- Advanced peak detection methods
- Additional calibration models
- Quality control and validation
- Comprehensive error reporting

Author: Enhanced version for production use
Version: 2.0.0
"""

import os
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Any, Union
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import least_squares, curve_fit, brentq
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.special import voigt_profile
from scipy.stats import median_abs_deviation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings in parallel processing
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Type alias for calibration methods
CalibrationMethod = Literal["quad_sqrt", "linear_sqrt", "poly2", "reflectron", "multisegment", "spline", "physical"]


@dataclass
class FlexibleCalibConfig:
    """Configuration for flexible calibration with method selection.

    Formerly: FlexibleCalibConfig (for backwards compatibility)
    """
    
    reference_masses: List[float]
    calibration_method: CalibrationMethod = "quad_sqrt"
    output_folder: str = "calibrated_spectra"
    max_workers: Optional[int] = None
    
    # Peak detection parameters
    autodetect_tol_da: Optional[float] = 2.0
    autodetect_tol_ppm: Optional[float] = None
    autodetect_method: str = "gaussian"  # Enhanced default
    autodetect_strategy: str = "mz"
    prefer_recompute_from_channel: bool = False
    
    # Robust fitting parameters
    outlier_threshold: float = 3.0
    use_outlier_rejection: bool = True
    max_iterations: int = 3
    
    # Quality control
    min_calibrants: int = 3
    max_ppm_threshold: Optional[float] = 100.0
    fail_on_high_error: bool = False
    
    # Advanced parameters for specific models
    spline_smoothing: Optional[float] = None
    multisegment_breakpoints: List[float] = field(default_factory=lambda: [50, 200, 500])
    instrument_params: Dict[str, float] = field(default_factory=dict)
    
    # Reporting options
    save_diagnostic_plots: bool = False
    verbose: bool = True


# ==================== Shared Helper Functions ====================

def _ppm_error(true_m: npt.NDArray[np.float64], est_m: npt.NDArray[np.float64]) -> float:
    """Calculate median absolute PPM error for robustness."""
    mask = np.isfinite(true_m) & np.isfinite(est_m) & (true_m > 0) & (est_m > 0)
    if not np.any(mask):
        return np.inf
    err_ppm = (est_m[mask] - true_m[mask]) / true_m[mask] * 1e6
    return float(np.median(np.abs(err_ppm)))


def _ppm_to_da(mz: float, ppm: float) -> float:
    """Convert PPM tolerance to Dalton."""
    return mz * (ppm * 1e-6)


def _detect_outliers_huber(residuals: npt.NDArray[np.float64], threshold: float = 3.0) -> npt.NDArray[np.bool_]:
    """Detect outliers using robust MAD-based method."""
    if len(residuals) < 4:
        return np.zeros_like(residuals, dtype=bool)
    
    med = np.median(residuals)
    mad = median_abs_deviation(residuals, scale='normal')
    
    if mad < 1e-10:
        return np.abs(residuals - med) > np.abs(med) * 0.1
    
    modified_z_scores = (residuals - med) / mad
    return np.abs(modified_z_scores) > threshold


def _estimate_noise_level(signal: npt.NDArray[np.float64]) -> float:
    """Estimate noise level using robust statistics."""
    if len(signal) < 10:
        return np.std(signal) * 0.1
    
    diff_signal = np.diff(signal)
    mad = median_abs_deviation(diff_signal, scale='normal')
    median_diff = np.median(diff_signal)
    
    noise_mask = np.abs(diff_signal - median_diff) < 3 * mad
    if noise_mask.sum() > 10:
        return float(median_abs_deviation(diff_signal[noise_mask], scale='normal'))
    else:
        return float(mad)


# ==================== Enhanced Peak Detection ====================

def _fit_gaussian_peak(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                       x0_guess: float) -> Optional[float]:
    """Fit Gaussian peak for accurate center determination."""
    if len(x) < 5:
        return None
    
    def gaussian(x, amp, cen, wid, offset):
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + offset
    
    try:
        p0 = [np.max(y), x0_guess, (np.max(x) - np.min(x)) / 4, np.min(y)]
        bounds = ([0, np.min(x), 0, 0], 
                 [np.inf, np.max(x), np.max(x) - np.min(x), np.max(y)])
        
        popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=1000)
        return float(popt[1])
    except (RuntimeError, ValueError, TypeError):
        return None


def _fit_voigt_peak(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                    x0_guess: float) -> Optional[float]:
    """Fit Voigt profile for asymmetric peaks."""
    if len(x) < 7:
        return None
    
    def voigt(x, amp, cen, sigma, gamma, offset):
        return amp * voigt_profile(x - cen, sigma, gamma) + offset
    
    try:
        p0 = [np.max(y), x0_guess, 1.0, 0.5, np.min(y)]
        bounds = ([0, np.min(x), 0.01, 0.01, 0],
                 [np.inf, np.max(x), 10, 10, np.max(y)])
        
        popt, _ = curve_fit(voigt, x, y, p0=p0, bounds=bounds, maxfev=1000)
        return float(popt[1])
    except (RuntimeError, ValueError, TypeError):
        return None


def _parabolic_peak_center(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                           peak_idx: int) -> Optional[float]:
    """Find peak center using parabolic interpolation."""
    if peak_idx == 0 or peak_idx == len(x) - 1:
        return None
    
    x1, x2, x3 = x[peak_idx - 1], x[peak_idx], x[peak_idx + 1]
    y1, y2, y3 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]
    
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-10:
        return None
    
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    
    if abs(A) < 1e-10:
        return None
    
    xc = -B / (2 * A)
    
    if xc < x1 or xc > x3:
        return None
    
    return float(xc)


def _enhanced_pick_channels(
    df: pd.DataFrame,
    targets: npt.NDArray[np.float64],
    tol_da: Optional[float],
    tol_ppm: Optional[float],
    method: str = "gaussian",
) -> List[int]:
    """Enhanced peak picking with multiple methods."""
    mz = df["m/z"].astype("float64").to_numpy()
    I = df["Intensity"].astype("float64").to_numpy()
    ch = df["Channel"].to_numpy()
    
    out: List[int] = []
    
    for xi in targets:
        tol = _ppm_to_da(xi, tol_ppm) if tol_ppm is not None else (tol_da if tol_da else 2.0)
        left, right = xi - tol, xi + tol
        mask = (mz >= left) & (mz <= right)
        
        if not mask.any():
            out.append(np.nan)
            continue
        
        idxs = np.flatnonzero(mask)
        mzw = mz[idxs]
        Iw = I[idxs]
        chw = ch[idxs]
        
        k_local = int(np.nanargmax(Iw))
        peak_mz = float(mzw[k_local])
        
        if method == "max":
            final_ch = chw[k_local]
            
        elif method == "centroid":
            wsum = Iw.sum()
            if wsum > 0:
                mz_c = float((mzw * Iw).sum() / wsum)
                nearest = int(np.argmin(np.abs(mzw - mz_c)))
                final_ch = chw[nearest]
            else:
                final_ch = chw[k_local]
                
        elif method == "parabolic":
            center = _parabolic_peak_center(mzw, Iw, k_local)
            if center is not None:
                nearest = int(np.argmin(np.abs(mzw - center)))
                final_ch = chw[nearest]
            else:
                final_ch = chw[k_local]
                
        elif method == "gaussian":
            center = _fit_gaussian_peak(mzw, Iw, peak_mz)
            if center is not None:
                nearest = int(np.argmin(np.abs(mzw - center)))
                final_ch = chw[nearest]
            else:
                # Fallback to centroid
                wsum = Iw.sum()
                if wsum > 0:
                    mz_c = float((mzw * Iw).sum() / wsum)
                    nearest = int(np.argmin(np.abs(mzw - mz_c)))
                    final_ch = chw[nearest]
                else:
                    final_ch = chw[k_local]
                    
        elif method == "voigt":
            center = _fit_voigt_peak(mzw, Iw, peak_mz)
            if center is not None:
                nearest = int(np.argmin(np.abs(mzw - center)))
                final_ch = chw[nearest]
            else:
                # Fallback to gaussian
                center = _fit_gaussian_peak(mzw, Iw, peak_mz)
                if center is not None:
                    nearest = int(np.argmin(np.abs(mzw - center)))
                    final_ch = chw[nearest]
                else:
                    final_ch = chw[k_local]
        else:
            final_ch = chw[k_local]
            
        out.append(int(final_ch))
        
    return out


def _enhanced_bootstrap_channels(
    channel: npt.NDArray[np.int_],
    intensity: npt.NDArray[np.float64],
    ref_masses: npt.NDArray[np.float64],
) -> List[int]:
    """Enhanced bootstrap with adaptive thresholds."""
    if len(intensity) < 10 or len(ref_masses) == 0:
        return [np.nan] * len(ref_masses)
    
    # Estimate noise level
    noise_level = _estimate_noise_level(intensity)
    
    # Adaptive peak finding
    min_prominence = max(noise_level * 3, np.nanmax(intensity) * 0.005)
    min_distance = max(10, len(channel) // 5000)
    
    peaks_idx, properties = find_peaks(
        intensity,
        prominence=min_prominence,
        distance=min_distance,
        height=noise_level * 2
    )
    
    if len(peaks_idx) < 3:
        peaks_idx, properties = find_peaks(
            intensity,
            prominence=min_prominence / 2,
            distance=min_distance // 2
        )
        
    if len(peaks_idx) < 2:
        return [np.nan] * len(ref_masses)
    
    # Robust k estimation
    peak_chs = channel[peaks_idx]
    peak_ints = intensity[peaks_idx]
    
    strong_order = np.argsort(peak_ints)[::-1]
    n_strong = min(10, len(peaks_idx))
    strong_peaks_ch = peak_chs[strong_order[:n_strong]]
    
    # Vectorized pairwise k-estimation
    # All upper-triangular peak-channel differences
    pi, pj = np.triu_indices(len(strong_peaks_ch), k=1)
    ch_diffs = strong_peaks_ch[pj] - strong_peaks_ch[pi]
    valid_ch = ch_diffs > 100
    ch_diffs = ch_diffs[valid_ch]

    # All upper-triangular sqrt-mass differences
    sqrt_masses = np.sqrt(ref_masses)
    mi, mj = np.triu_indices(len(ref_masses), k=1)
    sqrt_diffs = sqrt_masses[mj] - sqrt_masses[mi]
    valid_sq = sqrt_diffs > 0.1
    sqrt_diffs = sqrt_diffs[valid_sq]

    if len(ch_diffs) > 0 and len(sqrt_diffs) > 0:
        # Broadcast: all ch_diffs / all sqrt_diffs
        k_all = ch_diffs[:, np.newaxis] / sqrt_diffs[np.newaxis, :]
        k_mask = (k_all > 10) & (k_all < 10000)
        k_estimates = k_all[k_mask].tolist()
    else:
        k_estimates = []
    
    if k_estimates:
        k_rough = np.median(k_estimates)
    else:
        ch_min = np.min(strong_peaks_ch)
        ch_max = np.max(strong_peaks_ch)
        k_rough = (ch_max - ch_min) / (np.sqrt(ref_masses[-1]) - np.sqrt(ref_masses[0]))
    
    # Find peaks near expected channels
    out = []
    for m_ref in ref_masses:
        expected_ch = k_rough * np.sqrt(m_ref)
        tol_ch = max(500, expected_ch * 0.1)
        
        near_peaks = peaks_idx[
            (peak_chs >= expected_ch - tol_ch) & 
            (peak_chs <= expected_ch + tol_ch)
        ]
        
        if len(near_peaks) == 0:
            out.append(np.nan)
        else:
            best_idx = near_peaks[np.argmax(intensity[near_peaks])]
            out.append(int(channel[best_idx]))
            
    return out


# ==================== Calibration Models ====================

def _robust_initial_params_tof(m_ref: npt.NDArray[np.float64], 
                               t_meas: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Estimate initial TOF parameters robustly."""
    sqrt_m = np.sqrt(m_ref)
    
    # Vectorized pairwise slope estimation
    pi, pj = np.triu_indices(len(m_ref), k=1)
    dt = t_meas[pj] - t_meas[pi]
    dsqrt_m = sqrt_m[pj] - sqrt_m[pi]
    valid = (np.abs(dsqrt_m) > 0.1) & (np.abs(dt) > 10)
    k_estimates = (dt[valid] / dsqrt_m[valid]).tolist()
    
    k_init = np.median(k_estimates) if k_estimates else 100.0
    t0_init = np.median(t_meas - k_init * sqrt_m)
    c_init = 0.0
    
    return np.array([k_init, c_init, t0_init])


def _fit_quad_sqrt_robust(m_ref: npt.NDArray[np.float64], 
                   t_meas: npt.NDArray[np.float64],
                   outlier_threshold: float = 3.0,
                   max_iterations: int = 3) -> Optional[Tuple[float, float, float]]:
    """Fit TOF model with iterative outlier rejection."""
    if len(m_ref) < 3:
        return None
    
    x0 = _robust_initial_params_tof(m_ref, t_meas)
    
    mask = np.ones(len(m_ref), dtype=bool)
    best_params = None
    
    for iteration in range(max_iterations):
        m_fit = m_ref[mask]
        t_fit = t_meas[mask]
        
        if len(m_fit) < 3:
            break
            
        bounds = ([0.0, -1.0, -1e4], [1e6, 1.0, 1e4])
        
        def resid(p):
            k, c, t0 = p
            return t_fit - (k * np.sqrt(m_fit) + c * m_fit + t0)
        
        res = least_squares(resid, x0, bounds=bounds, method='trf', loss='huber')
        
        if not res.success:
            break
            
        best_params = res.x
        
        # Check for outliers
        residuals_full = t_meas - (best_params[0] * np.sqrt(m_ref) + 
                                   best_params[1] * m_ref + best_params[2])
        outliers = _detect_outliers_huber(residuals_full[mask], outlier_threshold)
        
        if not np.any(outliers):
            break
            
        temp_mask = mask.copy()
        temp_mask[np.where(mask)[0][outliers]] = False
        
        if temp_mask.sum() < 3:
            break
            
        mask = temp_mask
        x0 = best_params
        
    if best_params is None:
        return None
        
    k, c, t0 = best_params
    return (float(k), float(c), float(t0))


def _fit_reflectron_tof(m_ref: npt.NDArray[np.float64], 
                       t_meas: npt.NDArray[np.float64]) -> Optional[Tuple[float, float, float, float]]:
    """Fit extended TOF model for reflectron geometry."""
    if len(m_ref) < 4:
        return None
    
    sqrt_m = np.sqrt(m_ref)
    quarter_m = np.power(m_ref, 0.25)
    
    bounds = ([0, -1e3, -1, -1e4], [1e6, 1e3, 1, 1e4])
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    eps = 1e-6
    
    # Estimate initial parameters with a least-squares fit and clip into bounds.
    A = np.column_stack([sqrt_m, quarter_m, m_ref, np.ones_like(m_ref)])
    try:
        coeffs, *_ = np.linalg.lstsq(A, t_meas, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    
    if coeffs is None or not np.all(np.isfinite(coeffs)):
        order = np.argsort(sqrt_m)
        dsqrt = sqrt_m[order][-1] - sqrt_m[order][0]
        if abs(dsqrt) < eps:
            return None
        dt = t_meas[order][-1] - t_meas[order][0]
        k1_est = dt / dsqrt if np.isfinite(dt / dsqrt) else 0.0
        base_t0 = np.median(t_meas - k1_est * sqrt_m)
        coeffs = np.array(
            [k1_est, k1_est * 0.05, 0.0, base_t0],
            dtype=float
        )
    
    x0 = np.clip(coeffs, lower + eps, upper - eps)
    
    def resid(p):
        k1, k2, c, t0 = p
        return t_meas - (k1 * sqrt_m + k2 * quarter_m + c * m_ref + t0)
    
    res = least_squares(resid, x0, bounds=bounds, method='trf', loss='huber')
    
    if not res.success:
        return None
        
    k1, k2, c, t0 = res.x
    return (float(k1), float(k2), float(c), float(t0))


def _fit_multisegment_tof(m_ref: npt.NDArray[np.float64], 
                          t_meas: npt.NDArray[np.float64],
                          breakpoints: List[float]) -> Optional[Dict]:
    """Fit piecewise TOF model for different mass ranges."""
    segments = {}
    all_breakpoints = [0] + sorted(breakpoints) + [np.inf]
    
    for i in range(len(all_breakpoints) - 1):
        low, high = all_breakpoints[i], all_breakpoints[i + 1]
        mask = (m_ref >= low) & (m_ref < high)
        
        if mask.sum() >= 3:
            params = _fit_quad_sqrt_robust(m_ref[mask], t_meas[mask])
            if params:
                segments[f"segment_{i}"] = {
                    "range": (low, high),
                    "params": params,
                    "n_points": mask.sum()
                }
    
    if not segments:
        return None
        
    return {"segments": segments, "breakpoints": breakpoints}


def _fit_spline_model(m_ref: npt.NDArray[np.float64], 
                     t_meas: npt.NDArray[np.float64],
                     smoothing: Optional[float] = None) -> Optional[Any]:
    """Fit non-parametric spline model."""
    if len(m_ref) < 4:
        return None
    
    sort_idx = np.argsort(t_meas)
    t_sorted = t_meas[sort_idx]
    m_sorted = m_ref[sort_idx]
    
    sqrt_m = np.sqrt(m_sorted)
    
    if smoothing is None:
        smoothing = len(m_ref) * 0.01
    
    try:
        spline = UnivariateSpline(t_sorted, sqrt_m, s=smoothing, k=3)
        
        m_pred = spline(t_sorted)**2
        ppm = _ppm_error(m_sorted, m_pred)
        
        if ppm > 1000:
            return None
            
        return spline
    except (RuntimeError, ValueError, TypeError):
        return None


def _fit_physical_tof(m_ref: npt.NDArray[np.float64], 
                      t_meas: npt.NDArray[np.float64],
                      instrument_params: Dict[str, float]) -> Optional[Tuple[float, float, float, float]]:
    """Fit physical TOF model based on instrument parameters."""
    if len(m_ref) < 4:
        return None
    
    L = instrument_params.get('flight_length', 1.0)
    V = instrument_params.get('acceleration_voltage', 20000)
    
    def resid(p):
        scale, t_ext, t_det, delta = p
        theoretical_t = scale * L * np.sqrt(m_ref / (2 * V)) + t_ext + t_det
        return t_meas - theoretical_t
    
    bounds = ([0.9, -100, -10, -0.01], [1.1, 100, 10, 0.01])
    x0 = [1.0, 0.0, 0.0, 0.0]
    
    res = least_squares(resid, x0, bounds=bounds, method='trf')
    
    if not res.success:
        return None
    
    return tuple(float(x) for x in res.x)


# Standard models (for compatibility)
def _fit_linear_sqrt(m_ref: npt.NDArray[np.float64], 
                    t_meas: npt.NDArray[np.float64]) -> Optional[Tuple[float, float]]:
    """Fit linear sqrt model."""
    if len(m_ref) < 2:
        return None
    
    valid = np.isfinite(m_ref) & np.isfinite(t_meas) & (m_ref > 0) & (t_meas >= 0)
    if valid.sum() < 2:
        return None
    
    m_ref = m_ref[valid]
    t_meas = t_meas[valid]
    
    y = np.sqrt(m_ref)
    a, b = np.polyfit(t_meas, y, 1)
    return (float(a), float(b))


def _fit_poly2(m_ref: npt.NDArray[np.float64], 
              t_meas: npt.NDArray[np.float64]) -> Optional[Tuple[float, float, float]]:
    """Fit polynomial model."""
    if len(m_ref) < 3:
        return None
    
    valid = np.isfinite(m_ref) & np.isfinite(t_meas) & (m_ref > 0) & (t_meas >= 0)
    if valid.sum() < 3:
        return None
    
    m_ref = m_ref[valid]
    t_meas = t_meas[valid]
    
    p2, p1, p0 = np.polyfit(t_meas, m_ref, 2)
    return (float(p2), float(p1), float(p0))


# Inversion functions
def _invert_quad_sqrt(t: npt.NDArray[np.float64], k: float, c: float, t0: float) -> npt.NDArray[np.float64]:
    """Invert TOF model."""
    dt = np.asarray(t, dtype=np.float64) - float(t0)
    m = np.full_like(dt, np.nan, dtype=np.float64)
    valid = dt >= 0
    
    if not np.any(valid):
        return m
    
    dtv = dt[valid]
    
    if np.isclose(c, 0.0, atol=1e-12):
        x = dtv / float(k)
        x = np.where(x < 0, np.nan, x)
        m[valid] = x**2
        return m
    
    disc = k * k + 4.0 * c * dtv
    disc = np.maximum(disc, 0.0)
    sqrtD = np.sqrt(disc)
    x = (-k + sqrtD) / (2.0 * c)
    x = np.where(x < 0, np.nan, x)
    m[valid] = x**2
    return m


def _invert_reflectron(t: npt.NDArray[np.float64], k1: float, k2: float, 
                       c: float, t0: float) -> npt.NDArray[np.float64]:
    """Invert reflectron TOF model numerically."""
    dt = t - t0
    m_out = np.full_like(dt, np.nan)
    
    for i, dt_i in enumerate(dt):
        if dt_i < 0:
            continue
            
        def equation(m):
            if m <= 0:
                return np.inf
            return k1 * np.sqrt(m) + k2 * np.power(m, 0.25) + c * m - dt_i
        
        try:
            m_solution = brentq(equation, 1e-6, 1e6, xtol=1e-9)
            m_out[i] = m_solution
        except (ValueError, RuntimeError):
            continue
            
    return m_out


def _invert_linear_sqrt(t: npt.NDArray[np.float64], a: float, b: float) -> npt.NDArray[np.float64]:
    """Invert linear sqrt model."""
    x = a * t + b
    x = np.where(x < 0, np.nan, x)
    return x**2


def _invert_poly2(t: npt.NDArray[np.float64], p2: float, p1: float, p0: float) -> npt.NDArray[np.float64]:
    """Invert polynomial model."""
    return (p2 * t + p1) * t + p0


def _invert_spline(t: npt.NDArray[np.float64], spline: Any) -> npt.NDArray[np.float64]:
    """Invert spline model."""
    sqrt_m = spline(t)
    return np.where(sqrt_m > 0, sqrt_m**2, np.nan)


def _invert_physical(t: npt.NDArray[np.float64], scale: float, t_ext: float, 
                     t_det: float, delta: float, L: float, V: float) -> npt.NDArray[np.float64]:
    """Invert physical TOF model."""
    dt = t - t_ext - t_det
    m = 2 * V * (dt / (scale * L))**2
    return np.where(dt > 0, m, np.nan)


# ==================== Main Fitting Function ====================

def _fit_selected_model_enhanced(
    filename: str,
    ref_masses: npt.NDArray[np.float64],
    calib_channels_dict: Dict[str, Sequence[float]],
    config: FlexibleCalibConfig,
) -> Optional[Tuple[str, Dict]]:
    """Fit the user-selected calibration model with enhancements."""
    if filename not in calib_channels_dict:
        logger.warning(f"{filename}: Not found in calibration channels dict")
        return None
    
    t_meas = np.asarray(calib_channels_dict[filename], dtype=float)
    m_ref = np.asarray(ref_masses, dtype=float)
    
    # Filter out NaN channels
    valid = np.isfinite(t_meas)
    n_valid = valid.sum()
    
    min_required = 2 if config.calibration_method == "linear_sqrt" else 3
    if n_valid < min_required:
        logger.warning(f"{filename}: Only {n_valid} valid calibrants (need ≥{min_required})")
        return None
    
    t_meas = t_meas[valid]
    m_ref = m_ref[valid]
    
    logger.debug(f"{filename}: Fitting {config.calibration_method} with {n_valid} calibrants")
    
    # Fit the selected model
    params = None
    method = config.calibration_method
    
    if method == "quad_sqrt":
        if config.use_outlier_rejection:
            params = _fit_quad_sqrt_robust(m_ref, t_meas, 
                                   config.outlier_threshold, 
                                   config.max_iterations)
        else:
            # Simple fit without outlier rejection
            params = _fit_quad_sqrt_robust(m_ref, t_meas, max_iterations=1)
        
        if params is None:
            logger.error(f"{filename}: TOF model fitting failed")
            return None
        m_est = _invert_quad_sqrt(t_meas, *params)
        
    elif method == "reflectron":
        params = _fit_reflectron_tof(m_ref, t_meas)
        if params is None:
            logger.error(f"{filename}: Reflectron model fitting failed")
            return None
        m_est = _invert_reflectron(t_meas, *params)
        
    elif method == "multisegment":
        params = _fit_multisegment_tof(m_ref, t_meas, config.multisegment_breakpoints)
        if params is None:
            logger.error(f"{filename}: Multisegment model fitting failed")
            return None
        # For error calculation, use simple TOF on all points
        simple_params = _fit_quad_sqrt_robust(m_ref, t_meas)
        if simple_params:
            m_est = _invert_quad_sqrt(t_meas, *simple_params)
        else:
            m_est = m_ref  # Fallback
            
    elif method == "spline":
        params = _fit_spline_model(m_ref, t_meas, config.spline_smoothing)
        if params is None:
            logger.error(f"{filename}: Spline model fitting failed")
            return None
        m_est = _invert_spline(t_meas, params)
        
    elif method == "physical":
        if not config.instrument_params:
            logger.error(f"{filename}: Physical model requires instrument parameters")
            return None
        params = _fit_physical_tof(m_ref, t_meas, config.instrument_params)
        if params is None:
            logger.error(f"{filename}: Physical model fitting failed")
            return None
        L = config.instrument_params['flight_length']
        V = config.instrument_params['acceleration_voltage']
        m_est = _invert_physical(t_meas, *params, L, V)
        
    elif method == "linear_sqrt":
        params = _fit_linear_sqrt(m_ref, t_meas)
        if params is None:
            logger.error(f"{filename}: Linear sqrt model fitting failed")
            return None
        m_est = _invert_linear_sqrt(t_meas, *params)
        
    elif method == "poly2":
        params = _fit_poly2(m_ref, t_meas)
        if params is None:
            logger.error(f"{filename}: Poly2 model fitting failed")
            return None
        m_est = _invert_poly2(t_meas, *params)
        
    else:
        logger.error(f"{filename}: Unknown calibration method '{method}'")
        return None
    
    # Calculate calibration error
    ppm = _ppm_error(m_ref, m_est)
    
    # Quality control
    if config.fail_on_high_error and config.max_ppm_threshold:
        if ppm > config.max_ppm_threshold:
            logger.error(f"{filename}: PPM error {ppm:.1f} exceeds threshold {config.max_ppm_threshold}")
            return None
    
    result = {
        "method": method,
        "params": params,
        "ppm": ppm,
        "n_calibrants": n_valid,
        "calibrant_masses": m_ref.tolist(),
        "calibrant_channels": t_meas.tolist(),
        "estimated_masses": m_est.tolist(),
    }
    
    logger.info(f"{filename}: {method.upper()} fitted, ppm={ppm:.1f}, n={n_valid}")
    
    return filename, result


def _apply_model_to_file_enhanced(args: Tuple[str, Dict[str, Dict], str, CalibrationMethod, Dict[str, float]]) -> Optional[str]:
    """Apply the fitted calibration model with enhanced output."""
    file_path, results_map, out_folder, method, instrument_params = args
    fname = os.path.basename(file_path)
    
    if fname not in results_map:
        logger.warning(f"{fname}: No calibration results found")
        return None
    
    rec = results_map[fname]
    params = rec["params"]
    
    try:
        df = pd.read_csv(file_path, sep="\t", header=0, comment="#")
    except Exception as e:
        logger.error(f"{fname}: Failed to read file - {e}")
        return None
    
    if "Channel" not in df.columns:
        logger.error(f"{fname}: Missing 'Channel' column")
        return None
    
    t = df["Channel"].astype(np.float64).to_numpy()
    
    # Apply the calibration model
    if method == "quad_sqrt":
        mz_cal = _invert_quad_sqrt(t, *params)
    elif method == "reflectron":
        mz_cal = _invert_reflectron(t, *params)
    elif method == "linear_sqrt":
        mz_cal = _invert_linear_sqrt(t, *params)
    elif method == "poly2":
        mz_cal = _invert_poly2(t, *params)
    elif method == "spline":
        mz_cal = _invert_spline(t, params)
    elif method == "multisegment":
        mz_cal = np.full_like(t, np.nan)
        segments = params["segments"]
        for seg_info in segments.values():
            low, high = seg_info["range"]
            seg_params = seg_info["params"]
            mz_temp = _invert_quad_sqrt(t, *seg_params)
            mask = (mz_temp >= low) & (mz_temp < high)
            mz_cal[mask] = mz_temp[mask]
    elif method == "physical":
        L = instrument_params.get('flight_length', 1.0)
        V = instrument_params.get('acceleration_voltage', 20000)
        mz_cal = _invert_physical(t, *params, L, V)
    else:
        logger.error(f"{fname}: Unknown method '{method}'")
        return None
    
    # Create output DataFrame
    out_df = pd.DataFrame({
        "Channel": t,      
        "m/z": mz_cal,
        "Intensity": df["Intensity"].to_numpy()
    })
    
    # Save with metadata
    os.makedirs(out_folder, exist_ok=True)
    out_fp = os.path.join(out_folder, fname.replace(".txt", "_calibrated.txt"))
    
    with open(out_fp, 'w') as f:
        f.write(f"# Calibration method: {method}\n")
        f.write(f"# Calibration error: {rec['ppm']:.1f} ppm\n")
        f.write(f"# Number of calibrants: {rec['n_calibrants']}\n")
        f.write(f"# Calibrant masses: {rec['calibrant_masses']}\n")
        out_df.to_csv(f, sep="\t", index=False)
    
    logger.debug(f"{fname}: Saved calibrated spectrum to {out_fp}")
    
    return fname


# ==================== Main Calibrator Class ====================

class FlexibleCalibrator:
    """
    Flexible Calibrator with robust fitting and advanced features.

    Features:
    - User-selected calibration method
    - Robust outlier rejection
    - Advanced peak detection
    - Quality control and validation
    - Comprehensive error reporting
    - Support for advanced models (reflectron, spline, physical)

    Formerly: FlexibleCalibrator (for backwards compatibility)
    """

    def __init__(self, config: Optional[FlexibleCalibConfig] = None):
        """Initialize the flexible calibrator."""
        default_masses = [
            1.0072764666,   # H+ 30 000
            15.0229265168,  # CH3+ 20 000
            22.9892207021,  # Na+ 500 000
            27.0229265168,  # C2H3+ 200 000
            29.0385765812,  # C2H5+ 300 000
            38.9631579065,  # K+    6 000
            41.0385765812,  # C3H5+ 500 000
            43.0542266457,  # C3H7+ 300 000
            57.0698767102,  # C4H9+ 150 000
            58.065674,      # C₃H₈N⁺ (trimethylamine / choline-derived aminium). 300 000
            67.0548,        # C₅H₇⁺ 200 000
            71.0855267746,  # C5H11+ 60 000
            86.096976,      # C5H12N+ (choline fragment; strong in tissue) 250 000
            91.0542266457,  # C7H7+ (tropylium) 250 000
            104.107539,     # C5H14NO+ (choline fragment; strong in tissue) 100 000
            184.073320,     # C5H15NO4P+ (phosphocholine headgroup; hallmark of PC lipids) 70 000
            224.105171,     # C₈H₁₉NO₄P⁺ (phosphorylcholine-related fragment) 8000
            369.351600,     # C27H45+ (cholesterol fragment; very strong if cholesterol abundant) 12 000
        ]
        
        self.config = config or FlexibleCalibConfig(
            reference_masses=default_masses,
            calibration_method="quad_sqrt"
        )
        
        logger.info(f"FlexibleCalibrator initialized with method={self.config.calibration_method}")
    
    def calibrate(
        self,
        files: Sequence[str],
        calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None
    ) -> pd.DataFrame:
        """
        Calibrate all files using the selected calibration method.
        
        Returns:
            DataFrame with calibration summary
        """
        os.makedirs(self.config.output_folder, exist_ok=True)
        ref_masses = np.asarray(self.config.reference_masses, dtype=float)
        method = self.config.calibration_method
        
        logger.info(f"Starting calibration with method={method} for {len(files)} files")
        
        # Autodetect if needed
        if calib_channels_dict is None:
            logger.info("Autodetecting calibrant channels...")
            calib_channels_dict = self._autodetect_channels(files, ref_masses)
        
        # Fit model to all files
        results_map: Dict[str, Dict] = {}
        max_workers = self.config.max_workers or os.cpu_count() or 4

        if max_workers == 1:
            # Single-threaded execution (avoids multiprocessing pickling issues)
            for fp in tqdm(files, desc=f"Fitting {method} model"):
                res = _fit_selected_model_enhanced(
                    os.path.basename(fp), ref_masses,
                    calib_channels_dict, self.config
                )
                if res is not None:
                    fname, rec = res
                    results_map[fname] = rec
        else:
            # Multi-threaded execution
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(_fit_selected_model_enhanced,
                             os.path.basename(fp), ref_masses,
                             calib_channels_dict, self.config): fp
                    for fp in files
                }

                for fut in tqdm(as_completed(futs), total=len(futs),
                              desc=f"Fitting {method} model"):
                    res = fut.result()
                    if res is not None:
                        fname, rec = res
                        results_map[fname] = rec
        
        if not results_map:
            raise RuntimeError(f"No {method} models could be fitted")
        
        logger.info(f"Successfully fitted {len(results_map)}/{len(files)} files")
        
        # Check for high errors
        if self.config.max_ppm_threshold:
            high_error_files = [
                fname for fname, rec in results_map.items()
                if rec["ppm"] > self.config.max_ppm_threshold
            ]
            if high_error_files:
                logger.warning(
                    f"{len(high_error_files)} files exceed PPM threshold "
                    f"({self.config.max_ppm_threshold:.1f})"
                )
        
        # Create summary
        rows = []
        for fname, rec in results_map.items():
            row = {
                "file_name": fname,
                "method": rec["method"],
                "ppm_error": rec["ppm"],
                "n_calibrants": rec["n_calibrants"],
                "calibrant_masses": rec["calibrant_masses"],
                "calibrant_channels": rec["calibrant_channels"],
                "estimated_masses": rec["estimated_masses"],
            }
            rows.append(row)
        
        summary = pd.DataFrame(rows).sort_values("file_name")
        summary_path = os.path.join(self.config.output_folder, 
                                   f"calibration_summary_{method}.tsv")
        summary.to_csv(summary_path, sep="\t", index=False)
        
        # Apply calibration
        success_count = 0
        if max_workers == 1:
            # Single-threaded execution
            for fp in tqdm(files, desc=f"Applying {method} model"):
                result = _apply_model_to_file_enhanced(
                    (fp, results_map, self.config.output_folder, method, self.config.instrument_params)
                )
                if result is not None:
                    success_count += 1
        else:
            # Multi-threaded execution
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                args = ((fp, results_map, self.config.output_folder, method, self.config.instrument_params)
                       for fp in files)
                for result in tqdm(ex.map(_apply_model_to_file_enhanced, args),
                                 total=len(files), desc=f"Applying {method} model"):
                    if result is not None:
                        success_count += 1
        
        logger.info(f"Calibration complete: {success_count}/{len(files)} files processed")
        
        # Print statistics
        mean_ppm = summary["ppm_error"].mean()
        median_ppm = summary["ppm_error"].median()
        logger.info(f"Mean PPM error: {mean_ppm:.1f}, Median PPM error: {median_ppm:.1f}")
        
        return summary
    
    def _autodetect_channels(
        self,
        files: Sequence[str],
        ref_masses: npt.NDArray[np.float64]
    ) -> Dict[str, Sequence[float]]:
        """Autodetect calibrant channels with enhanced methods."""
        autodetected: Dict[str, Sequence[float]] = {}
        
        for fp in tqdm(files, desc="Autodetecting channels"):
            try:
                df = pd.read_csv(fp, sep="\t", header=0, comment="#")
            except Exception as e:
                logger.warning(f"{os.path.basename(fp)}: Failed to read - {e}")
                continue
            
            fname = os.path.basename(fp)
            
            if "Channel" not in df.columns:
                continue
            
            if self.config.prefer_recompute_from_channel or \
               self.config.autodetect_strategy == "bootstrap" or \
               "m/z" not in df.columns:
                # Bootstrap from signal
                ch = df["Channel"].to_numpy()
                y = df["Intensity"].astype("float64").to_numpy()
                autodetected[fname] = _enhanced_bootstrap_channels(ch, y, ref_masses)
            else:
                # Use existing m/z
                autodetected[fname] = _enhanced_pick_channels(
                    df, ref_masses,
                    self.config.autodetect_tol_da,
                    self.config.autodetect_tol_ppm,
                    self.config.autodetect_method
                )
        
        if not autodetected:
            raise RuntimeError("Autodetection failed")
        
        logger.info(f"Autodetected channels for {len(autodetected)}/{len(files)} files")
        
        return autodetected
