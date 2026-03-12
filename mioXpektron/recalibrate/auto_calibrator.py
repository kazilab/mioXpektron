"""
Enhanced Universal Calibration Module for High-Resolution Mass Spectrometry

Improvements:
- Multiple advanced calibration models (TOF, Reflectron, Multi-segment, Spline)
- Robust outlier detection using Huber's method
- Improved peak detection with noise-adaptive thresholds
- Better initial parameter estimation using RANSAC-like approach
- Peak shape fitting (Gaussian, Voigt) for accurate centroid determination
- Comprehensive error handling and validation

Author: Enhanced version for production use
Version: 2.0.0
"""

import os
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Any
from enum import Enum

import numpy as np
import numpy.typing as npt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.optimize import least_squares, curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.special import voigt_profile
from scipy.stats import median_abs_deviation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings in parallel processing
warnings.filterwarnings('ignore', category=RuntimeWarning)


class CalibrationModel(Enum):
    """Enumeration of available calibration models."""
    QUAD_SQRT = "quad_sqrt"  # Also known as TOF model
    LINEAR_SQRT = "linear_sqrt"
    POLY2 = "poly2"
    REFLECTRON = "reflectron"
    MULTISEGMENT = "multisegment"
    SPLINE = "spline"
    PHYSICAL = "physical"


class PeakDetectionMethod(Enum):
    """Enumeration of peak detection methods."""
    MAX = "max"
    CENTROID = "centroid"
    PARABOLIC = "parabolic"
    GAUSSIAN = "gaussian"
    VOIGT = "voigt"


@dataclass
class AutoCalibConfig:
    """Universal calibration configuration with robust options.

    Formerly: AutoCalibConfig (for backwards compatibility)
    """
    
    reference_masses: List[float]
    output_folder: str = "calibrated_spectra"
    max_workers: Optional[int] = None
    
    # Peak detection parameters
    autodetect_tol_da: Optional[float] = 2.0
    autodetect_tol_ppm: Optional[float] = None
    autodetect_method: str = "gaussian"  # Enhanced default
    autodetect_strategy: str = "mz"
    prefer_recompute_from_channel: bool = False
    
    # Robust fitting parameters
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    use_outlier_rejection: bool = True
    max_iterations: int = 3  # Maximum iterations for outlier rejection
    
    # Model selection parameters
    models_to_try: List[str] = field(default_factory=lambda: ["quad_sqrt",
                                                              "reflectron",
                                                              "linear_sqrt",
                                                              "poly2",
                                                              "multisegment",
                                                              "spline",
                                                              "physical"])
    prefer_physical_models: bool = True  # Prefer TOF/Reflectron over polynomial
    
    # Quality control
    min_calibrants: int = 3
    max_ppm_warning: float = 100.0
    max_ppm_error: float = 500.0
    
    # Advanced parameters
    use_bootstrap_init: bool = True  # Use robust initialization
    spline_smoothing: Optional[float] = None  # Auto-determine if None
    multisegment_breakpoints: List[float] = field(default_factory=lambda: [50, 200, 500])
    
    # Instrument parameters (for physical model)
    instrument_params: Dict[str, float] = field(default_factory=dict)


# ==================== Error Calculation and Validation ====================

def _ppm_error(true_m: npt.NDArray[np.float64], est_m: npt.NDArray[np.float64]) -> float:
    """Calculate mean absolute PPM error with robust statistics."""
    mask = np.isfinite(true_m) & np.isfinite(est_m) & (true_m > 0) & (est_m > 0)
    if not np.any(mask):
        return np.inf
    err_ppm = (est_m[mask] - true_m[mask]) / true_m[mask] * 1e6
    # Use median absolute error for robustness
    return float(np.median(np.abs(err_ppm)))


def _ppm_to_da(mz: float, ppm: float) -> float:
    """Convert PPM tolerance to Dalton at given m/z."""
    return mz * (ppm * 1e-6)


def _detect_outliers_huber(residuals: npt.NDArray[np.float64], threshold: float = 3.0) -> npt.NDArray[np.bool_]:
    """
    Detect outliers using Huber's robust method based on MAD.
    
    Args:
        residuals: Array of residuals
        threshold: Number of robust standard deviations for outlier threshold
        
    Returns:
        Boolean mask where True indicates an outlier
    """
    if len(residuals) < 4:
        return np.zeros_like(residuals, dtype=bool)
    
    med = np.median(residuals)
    mad = median_abs_deviation(residuals, scale='normal')
    
    if mad < 1e-10:  # Essentially zero variance
        return np.abs(residuals - med) > np.abs(med) * 0.1  # 10% tolerance
    
    modified_z_scores = (residuals - med) / mad
    return np.abs(modified_z_scores) > threshold


def _estimate_noise_level(signal: npt.NDArray[np.float64]) -> float:
    """
    Estimate noise level using robust MAD estimator on signal derivative.
    
    Args:
        signal: Input signal array
        
    Returns:
        Estimated noise standard deviation
    """
    if len(signal) < 10:
        return np.std(signal) * 0.1
    
    # Use derivative to estimate noise (high-frequency component)
    diff_signal = np.diff(signal)
    # Remove outliers (actual peaks) from noise estimation
    mad = median_abs_deviation(diff_signal, scale='normal')
    median_diff = np.median(diff_signal)
    
    # Only use values within 3 MAD for noise estimation
    noise_mask = np.abs(diff_signal - median_diff) < 3 * mad
    if noise_mask.sum() > 10:
        return float(median_abs_deviation(diff_signal[noise_mask], scale='normal'))
    else:
        return float(mad)


# ==================== Enhanced Peak Detection ====================

def _fit_gaussian_peak(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                       x0_guess: float) -> Optional[float]:
    """
    Fit a Gaussian peak to determine accurate center.
    
    Args:
        x: x-coordinates (m/z or channel)
        y: y-coordinates (intensity)
        x0_guess: Initial guess for peak center
        
    Returns:
        Fitted peak center or None if fitting fails
    """
    if len(x) < 5:
        return None
    
    def gaussian(x, amp, cen, wid, offset):
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + offset
    
    try:
        # Initial parameters
        p0 = [np.max(y), x0_guess, (np.max(x) - np.min(x)) / 4, np.min(y)]
        # Bounds
        bounds = ([0, np.min(x), 0, 0], 
                 [np.inf, np.max(x), np.max(x) - np.min(x), np.max(y)])
        
        popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=1000)
        return float(popt[1])  # Return center
    except:
        return None


def _fit_voigt_peak(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                    x0_guess: float) -> Optional[float]:
    """
    Fit a Voigt profile (convolution of Gaussian and Lorentzian) for asymmetric peaks.
    
    Args:
        x: x-coordinates (m/z or channel)
        y: y-coordinates (intensity)
        x0_guess: Initial guess for peak center
        
    Returns:
        Fitted peak center or None if fitting fails
    """
    if len(x) < 7:
        return None
    
    def voigt(x, amp, cen, sigma, gamma, offset):
        return amp * voigt_profile(x - cen, sigma, gamma) + offset
    
    try:
        # Initial parameters
        p0 = [np.max(y), x0_guess, 1.0, 0.5, np.min(y)]
        # Bounds
        bounds = ([0, np.min(x), 0.01, 0.01, 0],
                 [np.inf, np.max(x), 10, 10, np.max(y)])
        
        popt, _ = curve_fit(voigt, x, y, p0=p0, bounds=bounds, maxfev=1000)
        return float(popt[1])  # Return center
    except:
        return None


def _parabolic_peak_center(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                           peak_idx: int) -> Optional[float]:
    """
    Find peak center using parabolic interpolation around maximum.
    
    Args:
        x: x-coordinates
        y: y-coordinates
        peak_idx: Index of peak maximum
        
    Returns:
        Interpolated peak center or None if not enough points
    """
    if peak_idx == 0 or peak_idx == len(x) - 1:
        return None
    
    # Use three points around maximum
    x1, x2, x3 = x[peak_idx - 1], x[peak_idx], x[peak_idx + 1]
    y1, y2, y3 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]
    
    # Parabolic interpolation formula
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-10:
        return None
    
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    
    if abs(A) < 1e-10:
        return None
    
    xc = -B / (2 * A)
    
    # Sanity check: center should be between x1 and x3
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
    """
    Enhanced peak picking with multiple methods and robust center determination.
    
    Args:
        df: DataFrame with 'm/z', 'Intensity', 'Channel' columns
        targets: Target m/z values
        tol_da: Tolerance in Daltons
        tol_ppm: Tolerance in PPM
        method: Peak detection method
        
    Returns:
        List of channel indices for each target
    """
    mz = df["m/z"].astype("float64").to_numpy()
    I = df["Intensity"].astype("float64").to_numpy()
    ch = df["Channel"].to_numpy()
    
    out: List[int] = []
    
    for xi in targets:
        tol = _ppm_to_da(xi, tol_ppm) if tol_ppm is not None else (tol_da if tol_da else 2.0)
        left, right = xi - tol, xi + tol
        mask = (mz >= left) & (mz <= right)
        
        if not mask.any():
            logger.debug(f"No peak found for target m/z={xi:.4f}")
            out.append(np.nan)
            continue
        
        idxs = np.flatnonzero(mask)
        mzw = mz[idxs]
        Iw = I[idxs]
        chw = ch[idxs]
        
        # Find maximum as initial guess
        k_local = int(np.nanargmax(Iw))
        peak_mz = float(mzw[k_local])
        
        # Apply selected method
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
    """
    Enhanced bootstrap with adaptive thresholds and robust peak detection.
    
    Args:
        channel: Channel array
        intensity: Intensity array
        ref_masses: Reference masses
        
    Returns:
        List of channel indices for each reference mass
    """
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
        logger.warning(f"Bootstrap: Only found {len(peaks_idx)} peaks")
        # Try with relaxed parameters
        peaks_idx, properties = find_peaks(
            intensity,
            prominence=min_prominence / 2,
            distance=min_distance // 2
        )
        
    if len(peaks_idx) < 2:
        return [np.nan] * len(ref_masses)
    
    logger.debug(f"Bootstrap: Found {len(peaks_idx)} peaks with adaptive threshold")
    
    # Robust k estimation using RANSAC-like approach
    peak_chs = channel[peaks_idx]
    peak_ints = intensity[peaks_idx]
    
    # Get strongest peaks for initial estimation
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
    mi_idx, mj_idx = np.triu_indices(len(ref_masses), k=1)
    sqrt_diffs = sqrt_masses[mj_idx] - sqrt_masses[mi_idx]
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
        # Use robust median
        k_rough = np.median(k_estimates)
    else:
        # Fallback to simple estimation
        ch_min = np.min(strong_peaks_ch)
        ch_max = np.max(strong_peaks_ch)
        k_rough = (ch_max - ch_min) / (np.sqrt(ref_masses[-1]) - np.sqrt(ref_masses[0]))
    
    logger.debug(f"Bootstrap: Estimated k={k_rough:.1f} from {len(k_estimates)} estimates")
    
    # Find peaks near expected channels
    out = []
    for m_ref in ref_masses:
        expected_ch = k_rough * np.sqrt(m_ref)
        # Adaptive tolerance based on mass
        tol_ch = max(500, expected_ch * 0.1)
        
        near_peaks = peaks_idx[
            (peak_chs >= expected_ch - tol_ch) & 
            (peak_chs <= expected_ch + tol_ch)
        ]
        
        if len(near_peaks) == 0:
            logger.debug(f"Bootstrap: No peak for m/z={m_ref:.2f} (expected ch={expected_ch:.0f})")
            out.append(np.nan)
        else:
            # Pick strongest peak in window
            best_idx = near_peaks[np.argmax(intensity[near_peaks])]
            out.append(int(channel[best_idx]))
            logger.debug(f"Bootstrap: m/z={m_ref:.2f} -> ch={int(channel[best_idx])}")
            
    return out


# ==================== Calibration Models ====================

def _robust_initial_params_quad_sqrt(m_ref: npt.NDArray[np.float64], 
                               t_meas: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Estimate initial TOF parameters using robust statistics.
    
    Args:
        m_ref: Reference masses
        t_meas: Measured channels
        
    Returns:
        Initial parameters [k, c, t0]
    """
    sqrt_m = np.sqrt(m_ref)
    
    # Vectorized pairwise slope estimation
    pi, pj = np.triu_indices(len(m_ref), k=1)
    dt = t_meas[pj] - t_meas[pi]
    dsqrt_m = sqrt_m[pj] - sqrt_m[pi]
    valid = (np.abs(dsqrt_m) > 0.1) & (np.abs(dt) > 10)
    k_estimates = (dt[valid] / dsqrt_m[valid]).tolist()
    
    k_init = np.median(k_estimates) if k_estimates else 100.0
    
    # Estimate t0 as the intercept
    t0_init = np.median(t_meas - k_init * sqrt_m)
    
    # Start with c = 0 (pure sqrt model)
    c_init = 0.0
    
    return np.array([k_init, c_init, t0_init])


def _fit_quad_sqrt_robust(m_ref: npt.NDArray[np.float64], 
                   t_meas: npt.NDArray[np.float64],
                   outlier_threshold: float = 3.0,
                   max_iterations: int = 3) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    """
    Fit TOF model with iterative outlier rejection.
    
    Args:
        m_ref: Reference masses
        t_meas: Measured channels
        outlier_threshold: Threshold for outlier detection
        max_iterations: Maximum iterations for outlier rejection
        
    Returns:
        Tuple of ("quad_sqrt", (k, c, t0)) if successful
    """
    if len(m_ref) < 3:
        return None
    
    # Initial robust parameter estimation
    x0 = _robust_initial_params_quad_sqrt(m_ref, t_meas)
    
    # Iterative fitting with outlier rejection
    mask = np.ones(len(m_ref), dtype=bool)
    best_params = None
    
    for iteration in range(max_iterations):
        m_fit = m_ref[mask]
        t_fit = t_meas[mask]
        
        if len(m_fit) < 3:
            break
            
        # Bounds for physical parameters
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
            break  # No outliers found
            
        # Update mask
        temp_mask = mask.copy()
        temp_mask[np.where(mask)[0][outliers]] = False
        
        if temp_mask.sum() < 3:
            break  # Too few points remaining
            
        mask = temp_mask
        # Use current best as initial for next iteration
        x0 = best_params
        
    if best_params is None:
        return None
        
    k, c, t0 = best_params
    return ("quad_sqrt", (float(k), float(c), float(t0)))


def _fit_reflectron(m_ref: npt.NDArray[np.float64], 
                       t_meas: npt.NDArray[np.float64]) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
    """
    Fit extended TOF model for reflectron geometry.
    
    Model: t = k1*sqrt(m) + k2*m^(1/4) + c*m + t0
    
    Args:
        m_ref: Reference masses
        t_meas: Measured channels
        
    Returns:
        Tuple of ("reflectron", (k1, k2, c, t0)) if successful
    """
    if len(m_ref) < 4:
        return None
    
    sqrt_m = np.sqrt(m_ref)
    quarter_m = np.power(m_ref, 0.25)
    
    # Initial estimation
    k1_init = (t_meas[-1] - t_meas[0]) / (sqrt_m[-1] - sqrt_m[0])
    x0 = [k1_init, k1_init * 0.1, 0.0, np.min(t_meas)]
    
    def resid(p):
        k1, k2, c, t0 = p
        return t_meas - (k1 * sqrt_m + k2 * quarter_m + c * m_ref + t0)
    
    bounds = ([0, -1e3, -1, -1e4], [1e6, 1e3, 1, 1e4])
    
    res = least_squares(resid, x0, bounds=bounds, method='trf', loss='huber')
    
    if not res.success:
        return None
        
    k1, k2, c, t0 = res.x
    return ("reflectron", (float(k1), float(k2), float(c), float(t0)))


def _invert_reflectron(t: npt.NDArray[np.float64], k1: float, k2: float, 
                       c: float, t0: float) -> npt.NDArray[np.float64]:
    """
    Invert reflectron TOF model numerically.
    
    Args:
        t: Channel array
        k1, k2, c, t0: Model parameters
        
    Returns:
        Calculated m/z array
    """
    from scipy.optimize import brentq
    
    dt = t - t0
    m_out = np.full_like(dt, np.nan)
    
    for i, dt_i in enumerate(dt):
        if dt_i < 0:
            continue
            
        # Define equation to solve
        def equation(m):
            if m <= 0:
                return np.inf
            return k1 * np.sqrt(m) + k2 * np.power(m, 0.25) + c * m - dt_i
        
        try:
            # Find root using Brent's method
            m_solution = brentq(equation, 1e-6, 1e6, xtol=1e-9)
            m_out[i] = m_solution
        except:
            continue
            
    return m_out


def _fit_multisegment(m_ref: npt.NDArray[np.float64], 
                          t_meas: npt.NDArray[np.float64],
                          breakpoints: List[float]) -> Optional[Tuple[str, Dict]]:
    """
    Fit piecewise TOF model for different mass ranges.
    
    Args:
        m_ref: Reference masses
        t_meas: Measured channels
        breakpoints: Mass breakpoints for segments
        
    Returns:
        Tuple of ("multisegment", segment_dict) if successful
    """
    segments = {}
    all_breakpoints = [0] + sorted(breakpoints) + [np.inf]
    
    for i in range(len(all_breakpoints) - 1):
        low, high = all_breakpoints[i], all_breakpoints[i + 1]
        mask = (m_ref >= low) & (m_ref < high)
        
        if mask.sum() >= 3:
            segment_result = _fit_quad_sqrt_robust(m_ref[mask], t_meas[mask])
            if segment_result:
                segments[f"segment_{i}"] = {
                    "range": (low, high),
                    "params": segment_result[1],
                    "n_points": mask.sum()
                }
    
    if not segments:
        return None
        
    return ("multisegment", {"segments": segments, "breakpoints": breakpoints})


def _fit_spline_model(m_ref: npt.NDArray[np.float64], 
                     t_meas: npt.NDArray[np.float64],
                     smoothing: Optional[float] = None) -> Optional[Tuple[str, Any]]:
    """
    Fit non-parametric spline model.
    
    Args:
        m_ref: Reference masses
        t_meas: Measured channels
        smoothing: Smoothing factor (auto if None)
        
    Returns:
        Tuple of ("spline", spline_object) if successful
    """
    if len(m_ref) < 4:
        return None
    
    # Sort by channel for monotonic spline
    sort_idx = np.argsort(t_meas)
    t_sorted = t_meas[sort_idx]
    m_sorted = m_ref[sort_idx]
    
    # Fit through sqrt(m) for better behavior
    sqrt_m = np.sqrt(m_sorted)
    
    if smoothing is None:
        # Automatic smoothing based on data
        smoothing = len(m_ref) * 0.01
    
    try:
        spline = UnivariateSpline(t_sorted, sqrt_m, s=smoothing, k=3)
        
        # Validate spline
        m_pred = spline(t_sorted)**2
        ppm = _ppm_error(m_sorted, m_pred)
        
        if ppm > 1000:  # Spline is too poor
            return None
            
        return ("spline", spline)
    except:
        return None


def _invert_spline(t: npt.NDArray[np.float64], spline: Any) -> npt.NDArray[np.float64]:
    """
    Invert spline model.
    
    Args:
        t: Channel array
        spline: Spline object
        
    Returns:
        Calculated m/z array
    """
    sqrt_m = spline(t)
    return np.where(sqrt_m > 0, sqrt_m**2, np.nan)


# ==================== Standard Models (kept for compatibility) ====================

def _fit_linear_sqrt(m_ref: npt.NDArray[np.float64], 
                    t_meas: npt.NDArray[np.float64]) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Fit linear sqrt model: sqrt(m) = a*t + b."""
    if len(m_ref) < 2:
        return None
    
    valid = np.isfinite(m_ref) & np.isfinite(t_meas) & (m_ref > 0) & (t_meas >= 0)
    if valid.sum() < 2:
        return None
    
    m_ref = m_ref[valid]
    t_meas = t_meas[valid]
    
    y = np.sqrt(m_ref)
    a, b = np.polyfit(t_meas, y, 1)
    return ("linear_sqrt", (float(a), float(b)))


def _invert_linear_sqrt(t: npt.NDArray[np.float64], a: float, b: float) -> npt.NDArray[np.float64]:
    """Invert linear sqrt model."""
    x = a * t + b
    x = np.where(x < 0, np.nan, x)
    return x**2


def _fit_poly2(m_ref: npt.NDArray[np.float64], 
              t_meas: npt.NDArray[np.float64]) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    """Fit polynomial model: m = p2*t^2 + p1*t + p0."""
    if len(m_ref) < 3:
        return None
    
    valid = np.isfinite(m_ref) & np.isfinite(t_meas) & (m_ref > 0) & (t_meas >= 0)
    if valid.sum() < 3:
        return None
    
    m_ref = m_ref[valid]
    t_meas = t_meas[valid]
    
    p2, p1, p0 = np.polyfit(t_meas, m_ref, 2)
    return ("poly2", (float(p2), float(p1), float(p0)))


def _invert_poly2(t: npt.NDArray[np.float64], p2: float, p1: float, p0: float) -> npt.NDArray[np.float64]:
    """Invert polynomial model."""
    return (p2 * t + p1) * t + p0


def _invert_quad_sqrt(t: npt.NDArray[np.float64], k: float, c: float, t0: float) -> npt.NDArray[np.float64]:
    """Invert standard TOF model."""
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


# ==================== Main Fitting Function ====================

def _fit_and_score_models_enhanced(
    filename: str,
    ref_masses: npt.NDArray[np.float64],
    calib_channels_dict: Dict[str, Sequence[float]],
    config: AutoCalibConfig,
) -> Optional[Tuple[str, Dict]]:
    """
    Enhanced model fitting with multiple models and robust statistics.
    
    Args:
        filename: File name
        ref_masses: Reference masses
        calib_channels_dict: Calibration channels
        config: Configuration object
        
    Returns:
        Tuple of (filename, results_dict) if successful
    """
    if filename not in calib_channels_dict:
        logger.warning(f"{filename}: Not found in calibration channels dict")
        return None
    
    t_meas = np.asarray(calib_channels_dict[filename], dtype=float)
    m_ref = np.asarray(ref_masses, dtype=float)
    
    # Filter out NaN channels
    valid = np.isfinite(t_meas)
    n_valid = valid.sum()
    
    if n_valid < config.min_calibrants:
        logger.warning(f"{filename}: Only {n_valid} valid calibrants (need ≥{config.min_calibrants})")
        return None
    
    t_meas = t_meas[valid]
    m_ref = m_ref[valid]
    
    logger.info(f"{filename}: Fitting with {n_valid} calibrants")
    
    # Fit all requested models
    out: Dict[str, Dict] = {}
    
    for model_name in config.models_to_try:
        try:
            if model_name == "quad_sqrt":
                fit_result = _fit_quad_sqrt_robust(m_ref, t_meas, 
                                            config.outlier_threshold,
                                            config.max_iterations)
                if fit_result:
                    _, params = fit_result
                    m_est = _invert_quad_sqrt(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}
                    
            elif model_name == "reflectron":
                fit_result = _fit_reflectron(m_ref, t_meas)
                if fit_result:
                    _, params = fit_result
                    m_est = _invert_reflectron(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}
                    
            elif model_name == "multisegment":
                fit_result = _fit_multisegment(m_ref, t_meas, config.multisegment_breakpoints)
                if fit_result:
                    _, segment_data = fit_result
                    # Complex evaluation for multisegment
                    out[model_name] = {"params": segment_data, "ppm": 0, "n_calibrants": n_valid}
                    
            elif model_name == "spline":
                fit_result = _fit_spline_model(m_ref, t_meas, config.spline_smoothing)
                if fit_result:
                    _, spline = fit_result
                    m_est = _invert_spline(t_meas, spline)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": spline, "ppm": ppm, "n_calibrants": n_valid}
                    
            elif model_name == "linear_sqrt":
                fit_result = _fit_linear_sqrt(m_ref, t_meas)
                if fit_result:
                    _, params = fit_result
                    m_est = _invert_linear_sqrt(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}
                    
            elif model_name == "poly2":
                fit_result = _fit_poly2(m_ref, t_meas)
                if fit_result:
                    _, params = fit_result
                    m_est = _invert_poly2(t_meas, *params)
                    ppm = _ppm_error(m_ref, m_est)
                    out[model_name] = {"params": params, "ppm": ppm, "n_calibrants": n_valid}
                    
        except Exception as e:
            logger.debug(f"{filename}: Failed to fit {model_name}: {e}")
            continue
    
    if not out:
        logger.warning(f"{filename}: All models failed to fit")
        return None
    
    # Select best model
    if config.prefer_physical_models:
        # Prefer physical models if their error is reasonable
        physical_models = ["quad_sqrt", "reflectron"]
        for model in physical_models:
            if model in out and out[model]["ppm"] < config.max_ppm_warning:
                best_name = model
                break
        else:
            # Fall back to lowest error
            best_name = min(out.keys(), key=lambda k: out[k]["ppm"])
    else:
        # Simply choose lowest error
        best_name = min(out.keys(), key=lambda k: out[k]["ppm"])
    
    out["best_model"] = best_name
    
    # Quality warnings
    best_ppm = out[best_name]["ppm"]
    if best_ppm > config.max_ppm_error:
        logger.error(f"{filename}: Best model {best_name} exceeds max PPM ({best_ppm:.1f} > {config.max_ppm_error})")
        return None
    elif best_ppm > config.max_ppm_warning:
        logger.warning(f"{filename}: High PPM error ({best_ppm:.1f})")
    
    logger.info(f"{filename}: Best model={best_name}, ppm={best_ppm:.1f}")
    
    return filename, out


def _apply_model_to_file_enhanced(args: Tuple[str, Dict[str, Dict], str]) -> Optional[str]:
    """
    Apply the best calibration model to a file.
    
    Args:
        args: Tuple of (file_path, best_map, output_folder)
        
    Returns:
        Filename if successful
    """
    file_path, best_map, out_folder = args
    fname = os.path.basename(file_path)
    
    if fname not in best_map:
        return None
    
    rec = best_map[fname]
    model = rec["best_model"]
    params = rec[model]["params"]
    
    try:
        df = pd.read_csv(file_path, sep="\t", header=0, comment="#")
    except Exception as e:
        logger.error(f"Failed to read {fname}: {e}")
        return None
    
    if "Channel" not in df.columns:
        logger.error(f"{fname}: Missing 'Channel' column")
        return None
    
    t = df["Channel"].astype(np.float64).to_numpy()
    
    # Apply the appropriate model
    if model == "quad_sqrt":
        mz_cal = _invert_quad_sqrt(t, *params)
    elif model == "reflectron":
        mz_cal = _invert_reflectron(t, *params)
    elif model == "linear_sqrt":
        mz_cal = _invert_linear_sqrt(t, *params)
    elif model == "poly2":
        mz_cal = _invert_poly2(t, *params)
    elif model == "spline":
        mz_cal = _invert_spline(t, params)
    elif model == "multisegment":
        # Complex application for multisegment
        mz_cal = np.full_like(t, np.nan)
        segments = params["segments"]
        for seg_info in segments.values():
            low, high = seg_info["range"]
            seg_params = seg_info["params"]
            # Apply TOF model for this segment
            # Need to determine which channels belong to this mass range
            # This is approximate without iterative solving
            mz_temp = _invert_quad_sqrt(t, *seg_params)
            mask = (mz_temp >= low) & (mz_temp < high)
            mz_cal[mask] = mz_temp[mask]
    else:
        logger.error(f"Unknown model: {model}")
        return None
    
    # Create output with quality metrics
    out_df = pd.DataFrame({
        "m/z": mz_cal,
        "Intensity": df["Intensity"].to_numpy(),
        "Channel": t
    })
    
    # Add metadata as comments
    os.makedirs(out_folder, exist_ok=True)
    out_fp = os.path.join(out_folder, fname.replace(".txt", "_calibrated.txt"))
    
    with open(out_fp, 'w') as f:
        f.write(f"# Calibration model: {model}\n")
        f.write(f"# Calibration error: {rec[model]['ppm']:.1f} ppm\n")
        f.write(f"# Number of calibrants: {rec[model]['n_calibrants']}\n")
        out_df.to_csv(f, sep="\t", index=False)
    
    return fname


# ==================== Main Calibrator Class ====================

class AutoCalibrator:
    """
    Universal Calibrator with robust fitting and multiple models.

    Features:
    - Multiple calibration models (quad_sqrt, Reflectron, Spline, etc.)
    - Robust outlier detection
    - Advanced peak detection methods
    - Automatic model selection
    - Quality control and validation

    Formerly: AutoCalibrator (for backwards compatibility)
    """

    def __init__(self, config: Optional[AutoCalibConfig] = None):
        """Initialize the universal calibrator."""
        default_masses = [
            1.00782503224,   # H+
            22.9897692820,   # Na+
            38.9637064864,   # K+
            58.065674,       # Organic fragment
            86.096974,       # Organic fragment
            104.107539,      # Organic fragment
            184.073871,      # Organic fragment
            224.105171       # Organic fragment
        ]
        
        self.config = config or AutoCalibConfig(reference_masses=default_masses)
        
        # Validate configuration
        if len(self.config.reference_masses) < 3:
            raise ValueError("At least 3 reference masses required")
        
        logger.info(f"AutoCalibrator initialized with {len(self.config.models_to_try)} models")
    
    def calibrate(self, files: Sequence[str], 
                 calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None) -> pd.DataFrame:
        """
        Calibrate all files with automatic model selection.
        
        Args:
            files: List of file paths
            calib_channels_dict: Optional calibration channels
            
        Returns:
            Summary DataFrame with calibration results
        """
        os.makedirs(self.config.output_folder, exist_ok=True)
        ref_masses = np.asarray(self.config.reference_masses, dtype=float)
        
        # Autodetect if needed
        if calib_channels_dict is None:
            logger.info("Autodetecting calibrant channels...")
            calib_channels_dict = self._autodetect_channels(files, ref_masses)
        
        # Fit models to all files
        best_records: Dict[str, Dict] = {}
        max_workers = self.config.max_workers or os.cpu_count() or 4
        
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(_fit_and_score_models_enhanced, 
                         os.path.basename(fp), ref_masses, 
                         calib_channels_dict, self.config): fp
                for fp in files
            }
            
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Fitting models"):
                res = fut.result()
                if res is not None:
                    fname, rec = res
                    best_records[fname] = rec
        
        if not best_records:
            raise RuntimeError("No models could be fitted for any files")
        
        # Create summary
        rows = []
        for fname, rec in best_records.items():
            row = {
                "file_name": fname,
                "best_model": rec["best_model"],
                "best_ppm": rec[rec["best_model"]]["ppm"],
                "n_calibrants": rec[rec["best_model"]]["n_calibrants"]
            }
            # Add all model errors for comparison
            for model in self.config.models_to_try:
                if model in rec:
                    row[f"{model}_ppm"] = rec[model]["ppm"]
            rows.append(row)
        
        summary = pd.DataFrame(rows).sort_values("file_name")
        summary.to_csv(os.path.join(self.config.output_folder, "calibration_summary.tsv"), 
                      sep="\t", index=False)
        
        # Apply calibration
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            args = ((fp, best_records, self.config.output_folder) for fp in files)
            for _ in tqdm(ex.map(_apply_model_to_file_enhanced, args), 
                         total=len(files), desc="Applying calibration"):
                pass
        
        # Report statistics
        mean_ppm = summary["best_ppm"].mean()
        median_ppm = summary["best_ppm"].median()
        logger.info(f"Calibration complete: Mean PPM = {mean_ppm:.1f}, Median PPM = {median_ppm:.1f}")
        
        return summary
    
    def _autodetect_channels(self, files: Sequence[str], 
                            ref_masses: npt.NDArray[np.float64]) -> Dict[str, Sequence[float]]:
        """Autodetect calibrant channels from files."""
        autodetected: Dict[str, Sequence[float]] = {}
        
        for fp in tqdm(files, desc="Autodetecting channels"):
            try:
                df = pd.read_csv(fp, sep="\t", header=0, comment="#")
            except Exception as e:
                logger.warning(f"Failed to read {os.path.basename(fp)}: {e}")
                continue
            
            fname = os.path.basename(fp)
            
            if "Channel" not in df.columns:
                logger.warning(f"{fname}: No Channel column")
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
            raise RuntimeError("Autodetection failed: no suitable files")
        
        logger.info(f"Autodetected channels for {len(autodetected)}/{len(files)} files")
        
        return autodetected


# ==================== Convenience Functions ====================

def quick_calibrate(
    files: Sequence[str],
    reference_masses: Optional[List[float]] = None,
    output_folder: str = "calibrated_spectra",
    models: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Quick calibration with sensible defaults.
    
    Args:
        files: List of file paths to calibrate
        reference_masses: Optional reference masses (uses defaults if None)
        output_folder: Output directory for calibrated files
        models: List of models to try (uses defaults if None)
        **kwargs: Additional config parameters
        
    Returns:
        DataFrame with calibration summary
        
    Example:
        files = glob.glob("data/*.txt")
        summary = quick_calibrate(files)
    """
    if reference_masses is None:
        reference_masses = [
            1.00782503224,   # H+
            22.9897692820,   # Na+
            38.9637064864,   # K+
        ]
    
    if models is None:
        models = ["quad_sqrt", "reflectron", "linear_sqrt", "poly2"]
    
    config = AutoCalibConfig(
        reference_masses=reference_masses,
        output_folder=output_folder,
        models_to_try=models,
        **kwargs
    )
    
    calibrator = AutoCalibrator(config)
    return calibrator.calibrate(files)


def diagnose_calibration(
    files: Sequence[str],
    reference_masses: List[float],
    output_folder: str = "calibration_diagnostics",
    calib_channels_dict: Optional[Dict[str, Sequence[float]]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive calibration diagnostics.
    
    Tests multiple models and peak detection methods to find optimal settings.
    
    Args:
        files: List of file paths
        reference_masses: Reference masses for calibration
        output_folder: Output directory for diagnostic results
        calib_channels_dict: Optional pre-determined channels
        
    Returns:
        Dictionary with diagnostic DataFrames
        
    Example:
        diagnostics = diagnose_calibration(files, [1.0078, 22.9898, 38.9637])
        print(diagnostics['model_comparison'])
        print(diagnostics['peak_method_comparison'])
    """
    import itertools
    
    results = {}
    os.makedirs(output_folder, exist_ok=True)
    
    # Test different models
    all_models = ["quad_sqrt", "reflectron", "linear_sqrt", "poly2", "spline"]
    model_results = []
    
    logger.info("=" * 60)
    logger.info("CALIBRATION DIAGNOSTICS")
    logger.info("=" * 60)
    
    for models_to_test in [
        ["quad_sqrt"],
        ["reflectron"],
        ["linear_sqrt"],
        ["poly2"],
        ["spline"],
        ["quad_sqrt", "reflectron"],
        all_models
    ]:
        logger.info(f"Testing models: {', '.join(models_to_test)}")
        
        config = AutoCalibConfig(
            reference_masses=reference_masses,
            output_folder=os.path.join(output_folder, "_".join(models_to_test)),
            models_to_try=models_to_test,
            use_outlier_rejection=True
        )
        
        try:
            calibrator = AutoCalibrator(config)
            summary = calibrator.calibrate(files[:min(5, len(files))], calib_channels_dict)
            
            model_results.append({
                "models": ", ".join(models_to_test),
                "mean_ppm": summary["best_ppm"].mean(),
                "median_ppm": summary["best_ppm"].median(),
                "max_ppm": summary["best_ppm"].max(),
                "files_processed": len(summary)
            })
            
            logger.info(f"  Mean PPM: {summary['best_ppm'].mean():.1f}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            model_results.append({
                "models": ", ".join(models_to_test),
                "mean_ppm": np.nan,
                "median_ppm": np.nan,
                "max_ppm": np.nan,
                "files_processed": 0
            })
    
    results['model_comparison'] = pd.DataFrame(model_results)
    
    # Test different peak detection methods
    peak_methods = ["max", "centroid", "parabolic", "gaussian", "voigt"]
    peak_results = []
    
    logger.info("-" * 60)
    logger.info("Testing peak detection methods...")
    logger.info("-" * 60)
    
    for method in peak_methods:
        logger.info(f"Testing {method} peak detection")
        
        config = AutoCalibConfig(
            reference_masses=reference_masses,
            output_folder=os.path.join(output_folder, f"peak_{method}"),
            models_to_try=["quad_sqrt"],
            autodetect_method=method
        )
        
        try:
            calibrator = AutoCalibrator(config)
            summary = calibrator.calibrate(files[:min(5, len(files))])
            
            peak_results.append({
                "method": method,
                "mean_ppm": summary["best_ppm"].mean(),
                "median_ppm": summary["best_ppm"].median(),
                "files_processed": len(summary)
            })
            
            logger.info(f"  Mean PPM: {summary['best_ppm'].mean():.1f}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            peak_results.append({
                "method": method,
                "mean_ppm": np.nan,
                "median_ppm": np.nan,
                "files_processed": 0
            })
    
    results['peak_method_comparison'] = pd.DataFrame(peak_results)
    
    # Save diagnostic results
    for name, df in results.items():
        df.to_csv(os.path.join(output_folder, f"{name}.tsv"), sep="\t", index=False)
    
    # Print recommendations
    logger.info("=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    
    if not results['model_comparison'].empty:
        best_model_idx = results['model_comparison']['mean_ppm'].idxmin()
        if not pd.isna(best_model_idx):
            best_models = results['model_comparison'].loc[best_model_idx, 'models']
            best_ppm = results['model_comparison'].loc[best_model_idx, 'mean_ppm']
            logger.info("Best model configuration: %s", best_models)
            logger.info("  Mean PPM error: %.1f", best_ppm)
    
    if not results['peak_method_comparison'].empty:
        best_peak_idx = results['peak_method_comparison']['mean_ppm'].idxmin()
        if not pd.isna(best_peak_idx):
            best_peak = results['peak_method_comparison'].loc[best_peak_idx, 'method']
            best_peak_ppm = results['peak_method_comparison'].loc[best_peak_idx, 'mean_ppm']
            logger.info("Best peak detection method: %s", best_peak)
            logger.info("  Mean PPM error: %.1f", best_peak_ppm)
    
    logger.info("Diagnostic results saved to: %s", output_folder)
    logger.info("=" * 60)
    
    return results


def validate_calibration(
    original_files: Sequence[str],
    calibrated_files: Sequence[str],
    known_masses: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Validate calibration quality by checking known masses.
    
    Args:
        original_files: Original uncalibrated files
        calibrated_files: Calibrated files
        known_masses: Known masses to check (uses common ions if None)
        
    Returns:
        DataFrame with validation results
    """
    if known_masses is None:
        # Common calibrant and fragment ions
        known_masses = [
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
    
    validation_results = []
    
    for orig_file, cal_file in zip(original_files, calibrated_files):
        try:
            # Read calibrated data
            cal_df = pd.read_csv(cal_file, sep="\t", comment="#")
            mz = cal_df["m/z"].to_numpy()
            intensity = cal_df["Intensity"].to_numpy()
            
            # Check each known mass
            for known_mz in known_masses:
                # Find peak near known mass
                mask = np.abs(mz - known_mz) < 0.5
                if mask.any():
                    idx_max = np.argmax(intensity[mask])
                    found_mz = mz[mask][idx_max]
                    found_int = intensity[mask][idx_max]
                    error_ppm = (found_mz - known_mz) / known_mz * 1e6
                    
                    validation_results.append({
                        "file": os.path.basename(cal_file),
                        "known_mz": known_mz,
                        "found_mz": found_mz,
                        "intensity": found_int,
                        "error_ppm": error_ppm
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to validate {cal_file}: {e}")
    
    if not validation_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(validation_results)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("CALIBRATION VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for known_mz in known_masses:
        mz_data = df[df["known_mz"] == known_mz]["error_ppm"]
        if not mz_data.empty:
            mean_error = mz_data.mean()
            std_error = mz_data.std()
            logger.info("m/z %7.3f: %+6.1f ± %5.1f ppm", known_mz, mean_error, std_error)
    
    overall_mean = df["error_ppm"].abs().mean()
    overall_median = df["error_ppm"].abs().median()
    logger.info("Overall mean absolute error: %.1f ppm", overall_mean)
    logger.info("Overall median absolute error: %.1f ppm", overall_median)
    logger.info("=" * 60)
    
    return df


def batch_process_directory(
    input_dir: str,
    output_dir: str = "calibrated_output",
    pattern: str = "*.txt",
    reference_masses: Optional[List[float]] = None,
    config: Optional[AutoCalibConfig] = None,
    recursive: bool = False
) -> pd.DataFrame:
    """
    Process all matching files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        pattern: File pattern to match
        reference_masses: Reference masses for calibration
        config: Optional configuration object
        recursive: Whether to search subdirectories
        
    Returns:
        Summary DataFrame
    """
    import glob
    
    if recursive:
        files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
    else:
        files = glob.glob(os.path.join(input_dir, pattern))
    
    if not files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")
    
    logger.info("Found %d files to process", len(files))
    
    if config is None:
        if reference_masses is None:
            reference_masses = [1.00782503224, 22.9897692820, 38.9637064864]
        
        config = AutoCalibConfig(
            reference_masses=reference_masses,
            output_folder=output_dir
        )
    
    calibrator = AutoCalibrator(config)
    return calibrator.calibrate(files)


def create_calibration_report(
    summary_df: pd.DataFrame,
    output_file: str = "calibration_report.html"
) -> None:
    """
    Create an HTML report from calibration results.
    
    Args:
        summary_df: Summary DataFrame from calibration
        output_file: Output HTML file path
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calibration Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .good { color: green; font-weight: bold; }
            .warning { color: orange; font-weight: bold; }
            .bad { color: red; font-weight: bold; }
            .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Mass Spectrometry Calibration Report</h1>
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p><strong>Files Processed:</strong> {n_files}</p>
            <p><strong>Mean PPM Error:</strong> <span class="{mean_class}">{mean_ppm:.1f}</span></p>
            <p><strong>Median PPM Error:</strong> <span class="{median_class}">{median_ppm:.1f}</span></p>
            <p><strong>Max PPM Error:</strong> <span class="{max_class}">{max_ppm:.1f}</span></p>
        </div>
        
        <h2>Model Usage</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Files</th>
                <th>Percentage</th>
            </tr>
            {model_rows}
        </table>
        
        <h2>File Details</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Best Model</th>
                <th>PPM Error</th>
                <th>Calibrants</th>
            </tr>
            {file_rows}
        </table>
        
        <p><small>Report generated: {timestamp}</small></p>
    </body>
    </html>
    """
    
    # Calculate statistics
    mean_ppm = summary_df["best_ppm"].mean()
    median_ppm = summary_df["best_ppm"].median()
    max_ppm = summary_df["best_ppm"].max()
    
    # Color coding
    def get_class(value):
        if value < 50:
            return "good"
        elif value < 100:
            return "warning"
        else:
            return "bad"
    
    # Model usage
    model_counts = summary_df["best_model"].value_counts()
    model_rows = ""
    for model, count in model_counts.items():
        pct = count / len(summary_df) * 100
        model_rows += f"<tr><td>{model}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"
    
    # File details
    file_rows = ""
    for _, row in summary_df.iterrows():
        ppm_class = get_class(row["best_ppm"])
        file_rows += f"""<tr>
            <td>{row["file_name"]}</td>
            <td>{row["best_model"]}</td>
            <td class="{ppm_class}">{row["best_ppm"]:.1f}</td>
            <td>{row["n_calibrants"]}</td>
        </tr>\n"""
    
    # Fill template
    from datetime import datetime
    html = html_content.format(
        n_files=len(summary_df),
        mean_ppm=mean_ppm,
        mean_class=get_class(mean_ppm),
        median_ppm=median_ppm,
        median_class=get_class(median_ppm),
        max_ppm=max_ppm,
        max_class=get_class(max_ppm),
        model_rows=model_rows,
        file_rows=file_rows,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    logger.info("Report saved to: %s", output_file)


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Universal Mass Calibrator")
    parser.add_argument("files", nargs="+", help="Input files to calibrate")
    parser.add_argument("-o", "--output", default="calibrated_spectra", 
                       help="Output directory")
    parser.add_argument("-r", "--reference", nargs="+", type=float,
                       help="Reference masses for calibration")
    parser.add_argument("-m", "--models", nargs="+", 
                       choices=["quad_sqrt", "reflectron", "linear_sqrt", "poly2", "spline"],
                       help="Models to try")
    parser.add_argument("--diagnose", action="store_true",
                       help="Run diagnostic mode")
    parser.add_argument("--report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.diagnose:
        # Run diagnostics
        ref_masses = args.reference or [1.00782503224, 22.9897692820, 38.9637064864]
        diagnostics = diagnose_calibration(args.files, ref_masses)
    else:
        # Normal calibration
        config = AutoCalibConfig(
            reference_masses=args.reference or [1.00782503224, 22.9897692820, 38.9637064864],
            output_folder=args.output,
            models_to_try=args.models or ["quad_sqrt", "reflectron", "linear_sqrt", "poly2"]
        )
        
        calibrator = AutoCalibrator(config)
        summary = calibrator.calibrate(args.files)
        
        logger.info("Calibration complete!")
        logger.info("Mean PPM error: %.1f", summary['best_ppm'].mean())
        logger.info("Files processed: %d", len(summary))
        
        if args.report:
            create_calibration_report(summary, 
                                    os.path.join(args.output, "report.html"))
