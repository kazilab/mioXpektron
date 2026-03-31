"""
Shared calibration model backend for the recalibrate subpackage.

All model fitting, inversion, scoring, peak-detection, and helper functions
live here.  Both ``AutoCalibrator`` and ``FlexibleCalibrator`` import from
this module instead of maintaining parallel copies.

Author: Data Analysis Team @KaziLab.se
Version: 0.0.1
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, curve_fit, least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)


# Bootstrap autodetection heuristics. These are pragmatic search defaults for
# channel-only spectra, not instrument physics constants.
_BOOTSTRAP_PROMINENCE_NOISE_MULTIPLIER = 3.0
_BOOTSTRAP_PROMINENCE_SIGNAL_FRACTION = 0.005
_BOOTSTRAP_MIN_PEAK_DISTANCE_POINTS = 10
_BOOTSTRAP_PEAK_DISTANCE_DIVISOR = 5000
_BOOTSTRAP_HEIGHT_NOISE_MULTIPLIER = 2.0
_BOOTSTRAP_RETRY_RELAXATION_FACTOR = 0.5
_BOOTSTRAP_CHANNEL_MARGIN_MIN = 5000.0
_BOOTSTRAP_CHANNEL_MARGIN_FRACTION = 0.01
_BOOTSTRAP_MIN_PEAK_PAIR_SEPARATION = 100.0
_BOOTSTRAP_MIN_SQRT_MASS_PAIR_SEPARATION = 0.1
_BOOTSTRAP_K_GUESS_MIN = 1000.0
_BOOTSTRAP_K_GUESS_MAX = 100000.0
_BOOTSTRAP_K_BIN_WIDTH_MIN = 200.0
_BOOTSTRAP_K_BIN_WIDTH_FRACTION = 0.005
_BOOTSTRAP_T0_BIN_WIDTH_MIN = 1000.0
_BOOTSTRAP_T0_BIN_WIDTH_FRACTION = 0.002
_BOOTSTRAP_EXPECTED_MATCH_TOL_MIN = 600.0
_BOOTSTRAP_EXPECTED_MATCH_TOL_FRACTION = 0.008


# ---------------------------------------------------------------------------
#  Model metadata — flags that govern auto-selection and documentation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelMeta:
    """Metadata for a calibration model family."""
    name: str
    invertible_exactly: bool   # True if channel->m/z can be computed in closed form
    experimental: bool         # True if the model is not yet fully validated
    selection_allowed: bool    # True if the model may be chosen by auto-selection
    description: str           # One-line physical justification


MODEL_REGISTRY: Dict[str, _ModelMeta] = {
    "quad_sqrt": _ModelMeta(
        name="quad_sqrt",
        invertible_exactly=True,
        experimental=False,
        selection_allowed=True,
        description=(
            "Empirical TOF calibration: t = k*sqrt(m) + c*m + t0.  "
            "The sqrt(m) term follows standard TOF scaling; the linear-mass "
            "term absorbs systematic mass-dependent deviations."
        ),
    ),
    "linear_sqrt": _ModelMeta(
        name="linear_sqrt",
        invertible_exactly=True,
        experimental=False,
        selection_allowed=True,
        description=(
            "Simplified TOF: sqrt(m) = a*t + b.  "
            "Two-parameter version of quad_sqrt (no linear mass term)."
        ),
    ),
    "poly2": _ModelMeta(
        name="poly2",
        invertible_exactly=True,
        experimental=False,
        selection_allowed=True,
        description=(
            "Second-order polynomial: m = p2*t^2 + p1*t + p0.  "
            "Purely empirical; useful when the TOF relationship is weakly non-linear."
        ),
    ),
    "reflectron": _ModelMeta(
        name="reflectron",
        invertible_exactly=False,  # inverted numerically via Brent's method
        experimental=False,
        selection_allowed=True,
        description=(
            "Extended TOF for reflectron geometry: "
            "t = k1*sqrt(m) + k2*m^(1/4) + c*m + t0.  "
            "The m^(1/4) term absorbs higher-order reflectron aberrations "
            "(see Cotter, 'Time-of-Flight Mass Spectrometry', ACS 1997, Ch. 4)."
        ),
    ),
    "spline": _ModelMeta(
        name="spline",
        invertible_exactly=True,  # spline evaluation is direct
        experimental=False,
        selection_allowed=True,
        description=(
            "Non-parametric cubic spline through sqrt(m) vs channel.  "
            "Data-driven; requires >= 4 calibrants and risks overfitting."
        ),
    ),
    "multisegment": _ModelMeta(
        name="multisegment",
        invertible_exactly=False,  # channel->m/z requires iterative segment assignment
        experimental=True,
        selection_allowed=False,   # excluded from auto-selection until validated
        description=(
            "Piecewise quad_sqrt across user-defined mass ranges.  "
            "Experimental: segment-boundary assignment is approximate.  "
            "Use only with explicit opt-in via models_to_try."
        ),
    ),
    "physical": _ModelMeta(
        name="physical",
        invertible_exactly=False,
        experimental=True,
        selection_allowed=False,  # not implemented
        description=(
            "Instrument-parameter-driven physical model.  "
            "Not yet implemented; reserved for future use."
        ),
    ),
}

# Compatibility aliases: common names -> canonical model names
_MODEL_ALIASES: Dict[str, str] = {
    "quadratic": "poly2",
    "tof": "quad_sqrt",
    "linear": "linear_sqrt",
}


# ---------------------------------------------------------------------------
#  Error calculation and helper functions
# ---------------------------------------------------------------------------

def _ppm_error(true_m: npt.NDArray[np.float64], est_m: npt.NDArray[np.float64]) -> float:
    """Calculate median absolute PPM error for robustness."""
    mask = np.isfinite(true_m) & np.isfinite(est_m) & (true_m > 0) & (est_m > 0)
    if not np.any(mask):
        return np.inf
    err_ppm = (est_m[mask] - true_m[mask]) / true_m[mask] * 1e6
    return float(np.median(np.abs(err_ppm)))


def _ppm_to_da(mz: float, ppm: float) -> float:
    """Convert PPM tolerance to Dalton at given m/z."""
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
    """Estimate noise level using robust MAD estimator on signal derivative."""
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


# ---------------------------------------------------------------------------
#  Enhanced peak detection
# ---------------------------------------------------------------------------

def _fit_gaussian_peak(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
                       x0_guess: float) -> Optional[float]:
    """Fit Gaussian peak for accurate center determination."""
    if len(x) < 5:
        return None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    unique_x = np.unique(x)
    if len(unique_x) < 2:
        return None
    span = float(unique_x[-1] - unique_x[0])
    if span <= 0.0:
        return None
    step = float(np.median(np.diff(unique_x)))
    min_width = max(step * 0.5, span / 1000.0, np.finfo(np.float64).eps)
    max_width = max(span, min_width * 2.0)

    def gaussian(x, amp, cen, wid, offset):
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + offset

    try:
        peak_to_base = max(float(np.max(y) - np.min(y)), np.finfo(np.float64).eps)
        p0 = [peak_to_base, x0_guess, max(span / 8.0, min_width), float(np.min(y))]
        bounds = (
            [0.0, float(np.min(x)), min_width, float(np.min(y) - peak_to_base)],
            [np.inf, float(np.max(x)), max_width, float(np.max(y))],
        )

        popt, _ = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=1000)
        center = float(popt[1])
        if not np.isfinite(center) or center < np.min(x) or center > np.max(x):
            return None
        return center
    except (RuntimeError, ValueError, TypeError):
        return None


def _fit_voigt_peak(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
                    x0_guess: float) -> Optional[float]:
    """Fit Voigt profile for asymmetric peaks."""
    if len(x) < 7:
        return None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    unique_x = np.unique(x)
    if len(unique_x) < 2:
        return None
    span = float(unique_x[-1] - unique_x[0])
    if span <= 0.0:
        return None
    step = float(np.median(np.diff(unique_x)))
    min_width = max(step * 0.5, span / 1000.0, np.finfo(np.float64).eps)
    max_width = max(span, min_width * 2.0)

    def voigt(x, amp, cen, sigma, gamma, offset):
        return amp * voigt_profile(x - cen, sigma, gamma) + offset

    try:
        peak_to_base = max(float(np.max(y) - np.min(y)), np.finfo(np.float64).eps)
        width_guess = max(span / 8.0, min_width)
        p0 = [peak_to_base, x0_guess, width_guess, width_guess, float(np.min(y))]
        bounds = (
            [0.0, float(np.min(x)), min_width, min_width, float(np.min(y) - peak_to_base)],
            [np.inf, float(np.max(x)), max_width, max_width, float(np.max(y))],
        )

        popt, _ = curve_fit(voigt, x, y, p0=p0, bounds=bounds, maxfev=1000)
        center = float(popt[1])
        if not np.isfinite(center) or center < np.min(x) or center > np.max(x):
            return None
        return center
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


def _local_peak_bounds(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    peak_idx: int,
    *,
    min_fraction: float = 0.25,
    min_points: int = 3,
) -> Tuple[int, int]:
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
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
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

        x_local = x_local[apex_left:apex_right + 1]
        support_profile = support_profile[apex_left:apex_right + 1]

    weights = np.clip(y_local, 0.0, None)
    if apex_fraction is not None and 0.0 < apex_fraction < 1.0:
        weights = weights[apex_left:apex_right + 1]

    if not np.any(weights > 0):
        return None
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None
    return float(np.sum(x_local * weights) / weight_sum)


def _interpolate_channel_for_mz(
    mz_values: npt.NDArray[np.float64],
    channel_values: npt.NDArray[np.float64],
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


def _refined_center_is_plausible(
    center: float,
    peak_mz: float,
    fit_x: npt.NDArray[np.float64],
) -> bool:
    """Reject fitted centers that wander too far from the local apex."""
    if not np.isfinite(center):
        return False

    fit_x = np.asarray(fit_x, dtype=np.float64)
    unique_x = np.unique(fit_x)
    if len(unique_x) < 2:
        return False
    if center < unique_x[0] or center > unique_x[-1]:
        return False

    step = float(np.median(np.diff(unique_x)))
    span = float(unique_x[-1] - unique_x[0])
    max_shift = max(10.0 * step, 0.5 * span)
    return abs(center - peak_mz) <= max_shift


def _resolve_absolute_tolerances(
    target_mz: float,
    tol_da: Optional[float | Sequence[float]],
    tol_ppm: Optional[float],
) -> List[float]:
    """Resolve one or more absolute m/z tolerances for a target mass."""
    if tol_ppm is not None:
        return [_ppm_to_da(target_mz, tol_ppm)]

    if tol_da is None:
        return [_ppm_to_da(target_mz, 200.0)]

    if np.isscalar(tol_da):
        raw_values = [float(tol_da)]
    else:
        raw_values = [float(v) for v in tol_da]

    resolved: List[float] = []
    for raw_tol in raw_values:
        if not np.isfinite(raw_tol) or raw_tol <= 0.0:
            continue
        # Clamp absolute tolerance so it doesn't dominate at low mass:
        # at most +/-500 ppm equivalent, at least 0.05 Da
        max_tol = _ppm_to_da(target_mz, 500.0)
        tol = max(0.05, min(raw_tol, max_tol))
        if not resolved or not np.isclose(tol, resolved[-1], atol=1e-12, rtol=0.0):
            resolved.append(float(tol))

    if not resolved:
        return [_ppm_to_da(target_mz, 200.0)]
    return resolved


def _enhanced_pick_channels(
    df: pd.DataFrame,
    targets: npt.NDArray[np.float64],
    tol_da: Optional[float | Sequence[float]],
    tol_ppm: Optional[float],
    method: str = "gaussian",
    fallback_policy: str = "max",
    return_details: bool = False,
) -> Any:
    """Enhanced peak picking with multiple methods and explicit fallback control.

    Parameters
    ----------
    fallback_policy : {"max", "nan", "raise"}
        Policy applied when the requested non-max method cannot produce a valid
        refined pick. ``"max"`` preserves the current behavior and falls back
        through the existing cascade until the local maximum is used.
    return_details : bool, optional
        If True, return ``(channels, methods_used)`` so callers can inspect
        whether a fallback actually occurred.
    """
    if fallback_policy not in {"max", "nan", "raise"}:
        raise ValueError("fallback_policy must be 'max', 'nan', or 'raise'")

    mz = df["m/z"].astype("float64").to_numpy()
    I = df["Intensity"].astype("float64").to_numpy()
    ch = df["Channel"].to_numpy()

    out: List[float] = []
    methods_used: List[str] = []

    for xi in targets:
        tolerance_candidates = _resolve_absolute_tolerances(xi, tol_da, tol_ppm)
        fallback_result: Optional[Tuple[float, str]] = None
        final_result: Optional[Tuple[float, str]] = None

        for tol in tolerance_candidates:
            left, right = xi - tol, xi + tol
            mask = (mz >= left) & (mz <= right)
            if not mask.any():
                continue

            idxs = np.flatnonzero(mask)
            mzw = mz[idxs]
            Iw = I[idxs]
            chw = ch[idxs]

            k_local = int(np.nanargmax(Iw))
            peak_mz = float(mzw[k_local])

            def _pick_max(label: str = "max") -> Tuple[float, str]:
                return float(chw[k_local]), label

            def _pick_centroid(
                label: str = "centroid",
                *,
                subtract_baseline: bool = True,
                apex_fraction: Optional[float] = 0.75,
            ) -> Optional[Tuple[float, str]]:
                mz_c = _centroid_peak_center(
                    mzw,
                    Iw,
                    k_local,
                    min_points=3,
                    subtract_baseline=subtract_baseline,
                    apex_fraction=apex_fraction,
                )
                if mz_c is None:
                    return None
                channel_c = _interpolate_channel_for_mz(mzw, chw, mz_c)
                if channel_c is None:
                    return None
                return channel_c, label

            def _pick_parabolic(label: str = "parabolic") -> Optional[Tuple[float, str]]:
                center = _parabolic_peak_center(mzw, Iw, k_local)
                if center is None:
                    return None
                channel_c = _interpolate_channel_for_mz(mzw, chw, center)
                if channel_c is None:
                    return None
                return channel_c, label

            def _pick_gaussian(label: str = "gaussian") -> Optional[Tuple[float, str]]:
                local_left, local_right = _local_peak_bounds(mzw, Iw, k_local, min_points=5)
                mz_local = mzw[local_left:local_right + 1]
                I_local = Iw[local_left:local_right + 1]
                center = _fit_gaussian_peak(mz_local, I_local, peak_mz)
                if center is None:
                    return None
                if not _refined_center_is_plausible(center, peak_mz, mz_local):
                    return None
                channel_c = _interpolate_channel_for_mz(mzw, chw, center)
                if channel_c is None:
                    return None
                return channel_c, label

            def _pick_voigt(label: str = "voigt") -> Optional[Tuple[float, str]]:
                local_left, local_right = _local_peak_bounds(mzw, Iw, k_local, min_points=7)
                mz_local = mzw[local_left:local_right + 1]
                I_local = Iw[local_left:local_right + 1]
                center = _fit_voigt_peak(mz_local, I_local, peak_mz)
                if center is None:
                    return None
                if not _refined_center_is_plausible(center, peak_mz, mz_local):
                    return None
                channel_c = _interpolate_channel_for_mz(mzw, chw, center)
                if channel_c is None:
                    return None
                return channel_c, label

            def _handle_failure(failed_method: str) -> Tuple[float, str]:
                if fallback_policy == "max":
                    return _pick_max("max_fallback")
                if fallback_policy == "nan":
                    return np.nan, f"{failed_method}_failed"
                raise RuntimeError(
                    f"Peak-picking method '{method}' failed for target m/z={xi:.4f}"
                )

            if method == "max":
                candidate_result = _pick_max()
            elif method == "centroid":
                result = _pick_centroid()
                candidate_result = _handle_failure("centroid") if result is None else result
            elif method == "centroid_raw":
                result = _pick_centroid(
                    "centroid_raw",
                    subtract_baseline=False,
                    apex_fraction=None,
                )
                candidate_result = _handle_failure("centroid_raw") if result is None else result
            elif method == "parabolic":
                result = _pick_parabolic()
                candidate_result = _handle_failure("parabolic") if result is None else result
            elif method == "gaussian":
                result = _pick_gaussian()
                if result is not None:
                    candidate_result = result
                elif fallback_policy == "max":
                    result = _pick_centroid("centroid_fallback")
                    candidate_result = _handle_failure("gaussian") if result is None else result
                else:
                    candidate_result = _handle_failure("gaussian")
            elif method == "voigt":
                result = _pick_voigt()
                if result is not None:
                    candidate_result = result
                elif fallback_policy == "max":
                    result = _pick_gaussian("gaussian_fallback")
                    candidate_result = _handle_failure("voigt") if result is None else result
                else:
                    candidate_result = _handle_failure("voigt")
            else:
                raise ValueError(f"Unknown autodetect method '{method}'")

            if method == "max":
                final_result = candidate_result
                break

            # When multiple tolerances are provided, keep searching narrower
            # windows until the requested method succeeds. If only fallback
            # picks are available, retain the latest fallback as a last resort.
            if candidate_result[1] != method:
                fallback_result = candidate_result
                continue

            final_result = candidate_result
            break

        if final_result is None:
            if fallback_result is not None:
                final_result = fallback_result
            else:
                logger.debug(f"No peak found for target m/z={xi:.4f}")
                final_result = (np.nan, "none")

        final_ch, method_used = final_result
        out.append(float(final_ch) if np.isfinite(final_ch) else np.nan)
        methods_used.append(method_used)

    if return_details:
        return out, methods_used
    return out


def _enhanced_bootstrap_channels(
    channel: npt.NDArray[np.int_],
    intensity: npt.NDArray[np.float64],
    ref_masses: npt.NDArray[np.float64],
    bootstrap_overrides: Optional[Dict[str, float]] = None,
) -> List[float]:
    """Bootstrap channels from channel-space peaks using a consensus sqrt(m) map.

    The peak-search thresholds in this routine are intentionally heuristic:
    they trade sensitivity against false matches when no calibrated m/z axis is
    available yet. The constants are named explicitly to make those assumptions
    reviewable and easier to tune later.

    When ``bootstrap_overrides`` is provided, its keys (matching the
    ``_BOOTSTRAP_*`` constant names without the leading underscore) take
    precedence over the module-level defaults.
    """
    _ov = bootstrap_overrides or {}

    def _bget(name: str, default: float) -> float:
        return float(_ov.get(name, default))

    if len(intensity) < 10 or len(ref_masses) == 0:
        return [np.nan] * len(ref_masses)

    channel = np.asarray(channel, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    ref_masses = np.asarray(ref_masses, dtype=np.float64)

    order = np.argsort(channel)
    channel = channel[order]
    intensity = intensity[order]

    noise_level = _estimate_noise_level(intensity)

    min_prominence = max(
        noise_level * _bget("BOOTSTRAP_PROMINENCE_NOISE_MULTIPLIER",
                            _BOOTSTRAP_PROMINENCE_NOISE_MULTIPLIER),
        np.nanmax(intensity) * _bget("BOOTSTRAP_PROMINENCE_SIGNAL_FRACTION",
                                     _BOOTSTRAP_PROMINENCE_SIGNAL_FRACTION),
    )
    min_distance = max(
        int(_bget("BOOTSTRAP_MIN_PEAK_DISTANCE_POINTS",
                   _BOOTSTRAP_MIN_PEAK_DISTANCE_POINTS)),
        len(channel) // int(_bget("BOOTSTRAP_PEAK_DISTANCE_DIVISOR",
                                  _BOOTSTRAP_PEAK_DISTANCE_DIVISOR)),
    )

    peaks_idx, properties = find_peaks(
        intensity,
        prominence=min_prominence,
        distance=min_distance,
        height=noise_level * _bget("BOOTSTRAP_HEIGHT_NOISE_MULTIPLIER",
                                     _BOOTSTRAP_HEIGHT_NOISE_MULTIPLIER)
    )

    _retry_factor = _bget("BOOTSTRAP_RETRY_RELAXATION_FACTOR",
                           _BOOTSTRAP_RETRY_RELAXATION_FACTOR)
    if len(peaks_idx) < 3:
        logger.warning(f"Bootstrap: Only found {len(peaks_idx)} peaks")
        peaks_idx, properties = find_peaks(
            intensity,
            prominence=min_prominence * _retry_factor,
            distance=max(
                1,
                int(np.floor(min_distance * _retry_factor)),
            ),
        )

    if len(peaks_idx) < 2:
        return [np.nan] * len(ref_masses)

    logger.debug(f"Bootstrap: Found {len(peaks_idx)} peaks with adaptive threshold")

    peak_chs = channel[peaks_idx]
    peak_ints = intensity[peaks_idx]
    sqrt_masses = np.sqrt(ref_masses)
    strong_order = np.argsort(peak_ints)[::-1]
    n_strong = min(12, len(peak_chs))
    strong_peaks_ch = np.sort(np.asarray(peak_chs[strong_order[:n_strong]], dtype=np.float64))

    def _fallback_params() -> Tuple[float, float]:
        ch_min = float(np.min(strong_peaks_ch))
        ch_max = float(np.max(strong_peaks_ch))
        sqrt_span = max(float(sqrt_masses[-1] - sqrt_masses[0]), np.finfo(np.float64).eps)
        k_guess = max((ch_max - ch_min) / sqrt_span, np.finfo(np.float64).eps)
        t0_guess = ch_min - k_guess * float(sqrt_masses[0])
        return float(k_guess), float(t0_guess)

    candidate_params: List[Tuple[float, float]] = []
    channel_span = max(float(channel[-1] - channel[0]), 1.0)
    channel_margin = max(
        _bget("BOOTSTRAP_CHANNEL_MARGIN_MIN", _BOOTSTRAP_CHANNEL_MARGIN_MIN),
        channel_span * _bget("BOOTSTRAP_CHANNEL_MARGIN_FRACTION",
                             _BOOTSTRAP_CHANNEL_MARGIN_FRACTION),
    )

    for i in range(len(strong_peaks_ch) - 1):
        for j in range(i + 1, len(strong_peaks_ch)):
            ch1 = float(strong_peaks_ch[i])
            ch2 = float(strong_peaks_ch[j])
            ch_diff = ch2 - ch1
            if ch_diff <= _bget("BOOTSTRAP_MIN_PEAK_PAIR_SEPARATION",
                                _BOOTSTRAP_MIN_PEAK_PAIR_SEPARATION):
                continue

            for a in range(len(ref_masses) - 1):
                for b in range(a + 1, len(ref_masses)):
                    sqrt_diff = float(sqrt_masses[b] - sqrt_masses[a])
                    if sqrt_diff <= _bget("BOOTSTRAP_MIN_SQRT_MASS_PAIR_SEPARATION",
                                          _BOOTSTRAP_MIN_SQRT_MASS_PAIR_SEPARATION):
                        continue

                    k_guess = ch_diff / sqrt_diff
                    if (
                        not np.isfinite(k_guess)
                        or k_guess <= 0.0
                        or k_guess < _bget("BOOTSTRAP_K_GUESS_MIN", _BOOTSTRAP_K_GUESS_MIN)
                        or k_guess > _bget("BOOTSTRAP_K_GUESS_MAX", _BOOTSTRAP_K_GUESS_MAX)
                    ):
                        continue

                    t0_guess = ch1 - k_guess * float(sqrt_masses[a])
                    expected = t0_guess + k_guess * sqrt_masses
                    if expected[0] < channel[0] - channel_margin or expected[-1] > channel[-1] + channel_margin:
                        continue

                    candidate_params.append((float(k_guess), float(t0_guess)))

    if candidate_params:
        candidate_arr = np.asarray(candidate_params, dtype=np.float64)
        k_bin_width = max(
            _bget("BOOTSTRAP_K_BIN_WIDTH_MIN", _BOOTSTRAP_K_BIN_WIDTH_MIN),
            float(np.median(candidate_arr[:, 0])) * _bget("BOOTSTRAP_K_BIN_WIDTH_FRACTION",
                                                           _BOOTSTRAP_K_BIN_WIDTH_FRACTION),
        )
        t0_bin_width = max(
            _bget("BOOTSTRAP_T0_BIN_WIDTH_MIN", _BOOTSTRAP_T0_BIN_WIDTH_MIN),
            channel_span * _bget("BOOTSTRAP_T0_BIN_WIDTH_FRACTION",
                                 _BOOTSTRAP_T0_BIN_WIDTH_FRACTION),
        )
        k_bins = np.rint(candidate_arr[:, 0] / k_bin_width).astype(np.int64)
        t0_bins = np.rint(candidate_arr[:, 1] / t0_bin_width).astype(np.int64)
        best_bin, best_count = Counter(zip(k_bins, t0_bins)).most_common(1)[0]
        cluster_mask = (k_bins == best_bin[0]) & (t0_bins == best_bin[1])
        k_fit = float(np.median(candidate_arr[cluster_mask, 0]))
        t0_fit = float(np.median(candidate_arr[cluster_mask, 1]))
        logger.debug(
            "Bootstrap: consensus k=%.1f, t0=%.1f from %d clustered candidates",
            k_fit,
            t0_fit,
            best_count,
        )
    else:
        k_fit, t0_fit = _fallback_params()
        logger.debug("Bootstrap: using edge-based fallback k=%.1f, t0=%.1f", k_fit, t0_fit)

    def _match_prominent_peaks(
        k_model: float,
        t0_model: float,
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
        expected = t0_model + k_model * sqrt_masses
        matched_mass_idx: List[int] = []
        matched_peak_chs: List[float] = []
        used_peak_idx = set()

        for mass_idx, expected_ch in enumerate(expected):
            tol_ch = max(
                _bget("BOOTSTRAP_EXPECTED_MATCH_TOL_MIN",
                       _BOOTSTRAP_EXPECTED_MATCH_TOL_MIN),
                expected_ch * _bget("BOOTSTRAP_EXPECTED_MATCH_TOL_FRACTION",
                                    _BOOTSTRAP_EXPECTED_MATCH_TOL_FRACTION),
            )
            insert_at = int(np.searchsorted(peak_chs, expected_ch))
            lo = max(0, insert_at - 6)
            hi = min(len(peak_chs), insert_at + 7)

            best_peak_idx = None
            best_delta = None
            for peak_idx in range(lo, hi):
                if peak_idx in used_peak_idx:
                    continue
                delta = abs(float(peak_chs[peak_idx]) - float(expected_ch))
                if delta > tol_ch:
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_peak_idx = peak_idx

            if best_peak_idx is None:
                continue

            used_peak_idx.add(best_peak_idx)
            matched_mass_idx.append(mass_idx)
            matched_peak_chs.append(float(peak_chs[best_peak_idx]))

        return (
            np.asarray(matched_mass_idx, dtype=np.int_),
            np.asarray(matched_peak_chs, dtype=np.float64),
        )

    matched_idx = np.asarray([], dtype=np.int_)
    matched_peak_chs = np.asarray([], dtype=np.float64)
    for _ in range(2):
        matched_idx, matched_peak_chs = _match_prominent_peaks(k_fit, t0_fit)
        if len(matched_peak_chs) < 2:
            break

        design = np.column_stack([sqrt_masses[matched_idx], np.ones(len(matched_peak_chs))])
        refined_params, *_ = np.linalg.lstsq(design, matched_peak_chs, rcond=None)
        next_k = float(refined_params[0])
        next_t0 = float(refined_params[1])
        if not np.isfinite(next_k) or next_k <= 0.0 or not np.isfinite(next_t0):
            break
        if np.isclose(next_k, k_fit, rtol=1e-6, atol=1e-3) and np.isclose(next_t0, t0_fit, rtol=1e-6, atol=1e-1):
            k_fit = next_k
            t0_fit = next_t0
            break
        k_fit = next_k
        t0_fit = next_t0

    matched_idx, matched_peak_chs = _match_prominent_peaks(k_fit, t0_fit)
    if len(matched_peak_chs) >= 2:
        matched_expected = t0_fit + k_fit * sqrt_masses[matched_idx]
        residual_scale = float(median_abs_deviation(matched_peak_chs - matched_expected, scale="normal"))
    else:
        residual_scale = np.nan

    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        search_half_window = 300.0
    else:
        search_half_window = float(np.clip(6.0 * residual_scale, 150.0, 800.0))

    logger.debug(
        "Bootstrap: refined k=%.1f, t0=%.1f, matched=%d, local_window=%.1f",
        k_fit,
        t0_fit,
        len(matched_peak_chs),
        search_half_window,
    )

    expected_channels = t0_fit + k_fit * sqrt_masses
    out: List[float] = []
    for mass_idx, expected_ch in enumerate(expected_channels):
        lower = expected_ch - search_half_window
        upper = expected_ch + search_half_window
        if mass_idx > 0:
            lower = max(lower, 0.5 * (expected_channels[mass_idx - 1] + expected_ch))
        if mass_idx < len(expected_channels) - 1:
            upper = min(upper, 0.5 * (expected_ch + expected_channels[mass_idx + 1]))

        left = int(np.searchsorted(channel, lower, side="left"))
        right = int(np.searchsorted(channel, upper, side="right"))
        if right <= left:
            logger.debug("Bootstrap: empty local window for m/z=%.2f", ref_masses[mass_idx])
            out.append(np.nan)
            continue

        local_y = intensity[left:right]
        if len(local_y) == 0 or not np.isfinite(local_y).any():
            out.append(np.nan)
            continue

        best_local_idx = left + int(np.nanargmax(local_y))
        out.append(float(channel[best_local_idx]))
        logger.debug("Bootstrap: m/z=%.2f -> ch=%.0f", ref_masses[mass_idx], channel[best_local_idx])

    return out


# ---------------------------------------------------------------------------
#  Calibration model fitting
# ---------------------------------------------------------------------------

def _robust_initial_params_quad_sqrt(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Estimate initial TOF parameters robustly."""
    sqrt_m = np.sqrt(m_ref)

    pi, pj = np.triu_indices(len(m_ref), k=1)
    dt = t_meas[pj] - t_meas[pi]
    dsqrt_m = sqrt_m[pj] - sqrt_m[pi]
    valid = (np.abs(dsqrt_m) > 0.1) & (np.abs(dt) > 10)
    k_estimates = (dt[valid] / dsqrt_m[valid]).tolist()

    k_init = np.median(k_estimates) if k_estimates else 100.0
    if not np.isfinite(k_init):
        k_init = 100.0
    k_init = max(float(k_init), 1e-6)
    t0_init = np.median(t_meas - k_init * sqrt_m)
    c_init = 0.0

    return np.array([k_init, c_init, t0_init])


def _quad_sqrt_forward(
    m: npt.NDArray[np.float64],
    k: float,
    c: float,
    t0: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the empirical quad_sqrt TOF mapping."""
    m = np.asarray(m, dtype=np.float64)
    return k * np.sqrt(m) + c * m + t0


def _validate_quad_sqrt_params(
    k: float,
    c: float,
    t0: float,
    m_ref: npt.NDArray[np.float64],
) -> Tuple[bool, str]:
    """Check that quad_sqrt parameters are positive and monotone on-range."""
    masses = np.asarray(m_ref, dtype=np.float64)
    valid = np.isfinite(masses) & (masses > 0)
    if not np.any(valid):
        return False, "no positive reference masses available for validation"

    masses = masses[valid]
    m_min = float(np.min(masses))
    m_max = float(np.max(masses))
    span = m_max - m_min

    if span > 0.0:
        # Keep validation local to the calibrated range. Using a span-based
        # lower extrapolation can drive H+-anchored fits toward zero mass and
        # incorrectly reject otherwise valid channel-time models.
        m_eval_min = max(np.finfo(np.float64).tiny, m_min * 0.95)
        m_eval_max = m_max * 1.05
    else:
        m_eval_min = max(np.finfo(np.float64).tiny, m_min * 0.95)
        m_eval_max = m_max * 1.05

    eval_m = np.array([m_eval_min, m_min, m_max, m_eval_max], dtype=np.float64)
    t_eval = _quad_sqrt_forward(eval_m, k, c, t0)

    if not np.all(np.isfinite(t_eval)):
        return False, "model produced non-finite flight times"
    if np.any(t_eval <= 0.0):
        return False, "model produced non-positive flight times"

    # dt/dm = k / (2*sqrt(m)) + c, which is minimized at the largest mass.
    deriv_min = (k / (2.0 * np.sqrt(m_eval_max))) + c
    if not np.isfinite(deriv_min) or deriv_min <= 0.0:
        return False, "model is not monotone increasing over the calibrated mass range"

    return True, ""


def _fit_quad_sqrt_robust(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
    outlier_threshold: float = 3.0,
    max_iterations: int = 3,
) -> Optional[Tuple[float, float, float]]:
    """Fit TOF model with iterative outlier rejection.

    Returns (k, c, t0) or None.
    """
    if len(m_ref) < 3:
        return None

    x0 = _robust_initial_params_quad_sqrt(m_ref, t_meas)
    bounds = ([0.0, -1.0, -1e4], [1e6, 1.0, 1e4])
    lower = np.array(bounds[0], dtype=np.float64)
    upper = np.array(bounds[1], dtype=np.float64)
    eps = 1e-6
    x0 = np.clip(x0, lower + eps, upper - eps)

    mask = np.ones(len(m_ref), dtype=bool)
    best_params = None

    for iteration in range(max_iterations):
        m_fit = m_ref[mask]
        t_fit = t_meas[mask]

        if len(m_fit) < 3:
            break

        def resid(p):
            k, c, t0 = p
            return t_fit - _quad_sqrt_forward(m_fit, k, c, t0)

        res = least_squares(resid, x0, bounds=bounds, method='trf', loss='huber')

        if not res.success:
            break

        best_params = res.x

        residuals_full = t_meas - _quad_sqrt_forward(m_ref, *best_params)
        outliers = _detect_outliers_huber(residuals_full[mask], outlier_threshold)

        if not np.any(outliers):
            break

        temp_mask = mask.copy()
        temp_mask[np.where(mask)[0][outliers]] = False

        if temp_mask.sum() < 3:
            break

        mask = temp_mask
        x0 = np.clip(best_params, lower + eps, upper - eps)

    if best_params is None:
        return None

    k, c, t0 = best_params
    is_valid, reason = _validate_quad_sqrt_params(k, c, t0, m_ref)
    if not is_valid:
        logger.warning("Rejected quad_sqrt fit: %s", reason)
        return None
    return (float(k), float(c), float(t0))


def _fit_reflectron(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
) -> Optional[Tuple[float, float, float, float]]:
    """Fit extended TOF model for reflectron geometry.

    Model: t = k1*sqrt(m) + k2*m^(1/4) + c*m + t0

    The dominant sqrt(m) term follows standard TOF physics (t ~ sqrt(m/z)).
    The m^(1/4) correction is an empirical higher-order term that absorbs
    residual non-linearities from reflectron ion-optics (energy-focusing
    aberrations, fringe-field effects).  See Cotter, "Time-of-Flight Mass
    Spectrometry", ACS, 1997, Ch. 4.

    Returns (k1, k2, c, t0) or None.
    """
    if len(m_ref) < 4:
        return None

    sqrt_m = np.sqrt(m_ref)
    quarter_m = np.power(m_ref, 0.25)

    bounds = ([0, -1e3, -1, -1e4], [1e6, 1e3, 1, 1e4])
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    eps = 1e-6

    # Estimate initial parameters with a least-squares fit and clip into bounds
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


def _fit_multisegment(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
    breakpoints: List[float],
) -> Optional[Dict]:
    """Fit piecewise TOF model for different mass ranges.

    Returns {"segments": ..., "breakpoints": ...} or None.
    """
    segments = {}
    all_breakpoints = [0] + sorted(breakpoints) + [np.inf]

    for i in range(len(all_breakpoints) - 1):
        low, high = all_breakpoints[i], all_breakpoints[i + 1]
        mask = (m_ref >= low) & (m_ref < high)

        if mask.sum() >= 3:
            params = _fit_quad_sqrt_robust(m_ref[mask], t_meas[mask])
            if params is not None:
                segments[f"segment_{i}"] = {
                    "range": (low, high),
                    "params": params,
                    "n_points": int(mask.sum()),
                }

    if not segments:
        return None

    return {"segments": segments, "breakpoints": breakpoints}


def _fit_spline_model(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
    smoothing: Optional[float] = None,
) -> Optional[Any]:
    """Fit non-parametric spline model.  Returns the spline object or None."""
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


def _fit_physical_tof(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
    instrument_params: Dict[str, float],
) -> Optional[Tuple[float, float, float, float]]:
    """Fit physical TOF model based on instrument parameters.

    Returns (scale, t_ext, t_det, delta) or None.
    """
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


def _fit_linear_sqrt(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
) -> Optional[Tuple[float, float]]:
    """Fit linear sqrt model: sqrt(m) = a*t + b.  Returns (a, b) or None."""
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


def _fit_poly2(
    m_ref: npt.NDArray[np.float64],
    t_meas: npt.NDArray[np.float64],
) -> Optional[Tuple[float, float, float]]:
    """Fit polynomial model: m = p2*t^2 + p1*t + p0.  Returns (p2, p1, p0) or None."""
    if len(m_ref) < 3:
        return None

    valid = np.isfinite(m_ref) & np.isfinite(t_meas) & (m_ref > 0) & (t_meas >= 0)
    if valid.sum() < 3:
        return None

    m_ref = m_ref[valid]
    t_meas = t_meas[valid]

    p2, p1, p0 = np.polyfit(t_meas, m_ref, 2)
    return (float(p2), float(p1), float(p0))


# ---------------------------------------------------------------------------
#  Model inversion (channel -> m/z)
# ---------------------------------------------------------------------------

def _invert_quad_sqrt(t: npt.NDArray[np.float64], k: float, c: float, t0: float) -> npt.NDArray[np.float64]:
    """Invert quad_sqrt channel values to m/z.

    Returns NaN where the model has no real positive inverse.
    """
    dt = np.asarray(t, dtype=np.float64) - float(t0)
    m = np.full_like(dt, np.nan, dtype=np.float64)
    valid = np.isfinite(dt) & (dt >= 0.0)

    if not np.any(valid):
        return m

    dtv = dt[valid]
    disc = (k * k) + (4.0 * c * dtv)
    real = disc >= 0.0
    if not np.any(real):
        return m

    x = np.full_like(dtv, np.nan, dtype=np.float64)
    sqrtD = np.sqrt(disc[real])
    denom = k + sqrtD
    good = denom > 0.0
    if np.any(good):
        x_real = np.full_like(sqrtD, np.nan, dtype=np.float64)
        x_real[good] = (2.0 * dtv[real][good]) / denom[good]
        x[real] = x_real

    m[valid] = np.where(x >= 0.0, x * x, np.nan)
    return m


def _invert_reflectron(t: npt.NDArray[np.float64], k1: float, k2: float,
                       c: float, t0: float) -> npt.NDArray[np.float64]:
    """Invert reflectron TOF model numerically via Brent's method."""
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


# ---------------------------------------------------------------------------
#  Apply calibration to a full spectrum
# ---------------------------------------------------------------------------

def apply_model_to_spectrum(
    t: npt.NDArray[np.float64],
    model: str,
    params: Any,
    instrument_params: Optional[Dict[str, float]] = None,
) -> npt.NDArray[np.float64]:
    """Apply a fitted model to a channel array and return calibrated m/z.

    Parameters
    ----------
    t : array
        Channel values.
    model : str
        Model name (e.g. "quad_sqrt", "reflectron", ...).
    params : varies
        Fitted parameters (tuple for most models, dict for multisegment,
        spline object for spline).
    instrument_params : dict, optional
        Required only for the ``physical`` model.

    Returns
    -------
    mz_cal : array
        Calibrated m/z values.
    """
    if model == "quad_sqrt":
        return _invert_quad_sqrt(t, *params)
    elif model == "reflectron":
        return _invert_reflectron(t, *params)
    elif model == "linear_sqrt":
        return _invert_linear_sqrt(t, *params)
    elif model == "poly2":
        return _invert_poly2(t, *params)
    elif model == "spline":
        return _invert_spline(t, params)
    elif model == "multisegment":
        # Two-pass: apply each segment, then fill NaN gaps
        mz_cal = np.full_like(t, np.nan)
        segments = params["segments"]
        seg_list = sorted(segments.values(), key=lambda s: s["range"][0])
        for seg_info in seg_list:
            low, high = seg_info["range"]
            seg_params = seg_info["params"]
            mz_temp = _invert_quad_sqrt(t, *seg_params)
            mask = (mz_temp >= low) & (mz_temp < high)
            mz_cal[mask] = mz_temp[mask]
        # Fill remaining NaN channels with nearest-segment extrapolation
        nan_mask = np.isnan(mz_cal)
        if nan_mask.any() and seg_list:
            global_params = seg_list[len(seg_list) // 2]["params"]
            mz_cal[nan_mask] = _invert_quad_sqrt(t[nan_mask], *global_params)
        return mz_cal
    elif model == "physical":
        if instrument_params is None:
            instrument_params = {}
        L = instrument_params.get('flight_length', 1.0)
        V = instrument_params.get('acceleration_voltage', 20000)
        return _invert_physical(t, *params, L, V)
    else:
        raise ValueError(f"Unknown model '{model}'")
