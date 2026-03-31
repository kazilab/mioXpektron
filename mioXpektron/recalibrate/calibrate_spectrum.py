from pathlib import Path
from typing import Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import least_squares


def local_centroid(mz: np.ndarray,
                   intensity: np.ndarray,
                   target: float,
                   window_ppm: float = 50.0) -> Optional[float]:
    """
    Returns the centroid (intensity-weighted mean) of the peak nearest *target*
    within ±window_ppm.  Returns None if no peak is found.
    """
    delta = target * window_ppm * 1e-6
    mask = (mz >= target - delta) & (mz <= target + delta)
    if not mask.any():
        return None
    idx = mask.nonzero()[0]
    weights = intensity[idx]
    return np.average(mz[idx], weights=weights)


def _tof_model(params: np.ndarray, m_true: np.ndarray) -> np.ndarray:
    """
    Empirical TOF correction model: t = k*sqrt(m) + c*m + t0
    We invert it analytically (solve for m) when calibrating the spectrum.
    """
    k, c, t0 = params
    t = k * np.sqrt(m_true) + c * m_true + t0
    return t


def _tof_residuals(params: np.ndarray,
                   m_meas: np.ndarray,
                   m_true: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
    """
    Residuals for nonlinear least squares in time-of-flight space.
    """
    k, c, t0 = params
    # Convert measured m/z to times assuming t ∝ √m (raw axis ≈ m/z already)
    t_meas = np.sqrt(m_meas)
    t_calc = _tof_model(params, m_true)
    return weights * (t_meas - t_calc)


def _poly_residuals(coeffs: np.ndarray,
                    m_meas: np.ndarray,
                    m_true: np.ndarray,
                    weights: np.ndarray) -> np.ndarray:
    """
    Residuals for direct polynomial fit: m_true = p0 + p1*m_meas + p2*m_meas²
    """
    m_calc = np.polyval(coeffs, m_meas)
    return weights * (m_calc - m_true)


def calibrate_spectrum(mz: np.ndarray,
                       intensity: np.ndarray,
                       standards: Sequence[float],
                       mode: Literal["tof", "poly"] = "tof",
                       window_ppm: float = 50.0):
    """
    Calibrate a ToF-SIMS spectrum.

    Parameters
    ----------
    mz, intensity : 1-D numpy arrays of equal length
    standards     : iterable of reference masses (monoisotopic, in u)
    mode          : "tof" (default) – empirical TOF correction model
                    "poly"          – 2nd-order polynomial
    window_ppm    : search window for peak matching (±ppm)

    Returns
    -------
    mz_cal        : calibrated mass axis
    fit           : scipy OptimizeResult
    report        : pandas DataFrame with residual statistics
    """
    # 1. Match experimental peaks to standards
    exp = []
    refs = []
    wts = []
    for mass in standards:
        centroid = local_centroid(mz, intensity, mass, window_ppm)
        if centroid is not None:
            exp.append(centroid)
            refs.append(mass)
            # weight by peak height to favour well-defined peaks
            wts.append(float(intensity[np.argmin(np.abs(mz - centroid))]))
    if len(exp) < 3:
        raise RuntimeError("Fewer than three standards were found – "
                           "cannot perform reliable calibration.")

    exp = np.array(exp)
    refs = np.array(refs)
    wts = np.sqrt(np.array(wts))       # use sqrt for dynamic-range mitigation

    # 2. Fit chosen calibration model
    if mode == "tof":
        # Initial guess: k ≈ 1, c ≈ 0, t0 ≈ 0
        x0 = np.array([1.0, 0.0, 0.0])
        fit = least_squares(_tof_residuals, x0,
                            args=(exp, refs, wts), method="trf")
        k, c, t0 = fit.x

        # Build calibrated axis by solving t = k√m + cm + t0  for m
        t_meas = np.sqrt(mz)
        # Quadratic: c m + k √m + (t0 - t) = 0
        # Solve for √m using quadratic formula in √m (see derivation)
        a = c
        b = k
        d = t0 - t_meas
        discriminant = b**2 - 4 * a * d
        if np.isclose(a, 0.0):
            # Linear case: k * sqrt(m) = t - t0
            sqrt_m = (t_meas - t0) / k
            sqrt_m = np.where(sqrt_m < 0, np.nan, sqrt_m)
        else:
            # Guard against negative discriminant
            discriminant = np.maximum(discriminant, 0.0)
            sqrt_m = (-b + np.sqrt(discriminant)) / (2 * a)
            sqrt_m = np.where(sqrt_m < 0, np.nan, sqrt_m)
        mz_cal = sqrt_m**2

    elif mode == "poly":
        # 2nd-order polynomial fit (np.polyfit returns highest -> lowest order)
        coeffs = np.polyfit(exp, refs, deg=2, w=wts)
        fit = least_squares(_poly_residuals, coeffs,
                            args=(exp, refs, wts), method="trf")
        coeffs_fit = fit.x
        mz_cal = np.polyval(coeffs_fit, mz)
    else:
        raise ValueError("mode must be 'tof' or 'poly'")

    # 3. Quality metrics
    residuals = mz_cal[np.searchsorted(mz, exp)] - refs
    ppm_err = residuals / refs * 1e6
    report = pd.DataFrame({
        "Reference (u)": refs,
        "Measured (u)": exp,
        "Calibrated (u)": mz_cal[np.searchsorted(mz, exp)],
        "Error (ppm)": ppm_err,
    })

    return mz_cal, fit, report
