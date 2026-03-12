"""
Normalization methods for ToF-SIMS mass spectrometry data.

Provides multiple normalization strategies ranging from simple scaling (TIC, max)
to variance-stabilizing transforms (Poisson, sqrt, VSN) and robust methods
(median, RMS, PQN).  Each function operates on a single 1-D intensity array;
batch helpers live in ``preprocessing.py``.

All public functions share a common contract:
    * Accept a 1-D array-like of intensities.
    * Return a 1-D ``np.ndarray`` of the same length.
    * Replace NaN / negative artefacts with zero by default.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Registry helpers
# ---------------------------------------------------------------------------

def normalization_method_names() -> List[str]:
    """Return a sorted list of available 1-D normalization method names."""
    return sorted(_DISPATCH.keys())


def normalize(
    intensities,
    method: str = "tic",
    **kwargs,
) -> np.ndarray:
    """Apply a named normalization method to a 1-D intensity array.

    Parameters
    ----------
    intensities : array-like
        Raw intensity values (1-D).
    method : str, default ``"tic"``
        Name of the normalization method.  Call
        :func:`normalization_method_names` for the full list.
    **kwargs
        Method-specific keyword arguments forwarded to the underlying
        function (e.g. ``target_tic`` for TIC, ``reference_mz_idx`` for
        selected-ion normalization).

    Returns
    -------
    np.ndarray
        Normalized intensity values.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    func = _DISPATCH.get(method)
    if func is None:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            f"Valid methods: {normalization_method_names()}"
        )
    return func(np.asarray(intensities, dtype=float), **kwargs)


# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------

def _safe_output(arr: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf / negative values with zero."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0
    return arr


# ---------------------------------------------------------------------------
#  Individual methods
# ---------------------------------------------------------------------------

# 1. TIC normalization (existing canonical implementation) -----------------

def tic_normalization(intensities, target_tic=1e6):
    """Scale intensities so the total-ion current equals *target_tic*.

    This is the most common normalisation in ToF-SIMS.  Each spectrum is
    multiplied by ``target_tic / sum(intensities)`` so that all spectra share
    the same TIC.

    Parameters
    ----------
    intensities : array-like
        Raw ion counts or intensities.
    target_tic : float or None
        Desired total-ion current after scaling.  Pass ``None`` to skip.

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)

    if target_tic is None:
        scaled = intensities
    else:
        current_tic = np.nansum(intensities)
        if not np.isfinite(current_tic) or current_tic <= 0:
            warnings.warn(
                f"TIC normalization skipped: current TIC is {current_tic} "
                f"(non-finite or <= 0). Returning zeros.",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros_like(intensities)
        scaled = intensities * (float(target_tic) / current_tic)

    return _safe_output(scaled)


# 2. Median normalization --------------------------------------------------

def median_normalization(intensities, target_median=1.0):
    """Scale intensities so the median equals *target_median*.

    More robust than TIC when a few dominant peaks (e.g. substrate ions)
    inflate the total-ion current.

    Parameters
    ----------
    intensities : array-like
    target_median : float, default 1.0

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    med = np.nanmedian(intensities[intensities > 0]) if np.any(intensities > 0) else 0.0
    if not np.isfinite(med) or med <= 0:
        warnings.warn(
            "Median normalization skipped: median of positive values is "
            f"{med}. Returning zeros.",
            UserWarning, stacklevel=2,
        )
        return np.zeros_like(intensities)
    return _safe_output(intensities * (float(target_median) / med))


# 3. RMS normalization ------------------------------------------------------

def rms_normalization(intensities, target_rms=1.0):
    """Scale intensities so the root-mean-square equals *target_rms*.

    A compromise between TIC (dominated by big peaks) and median
    (ignores peak structure).

    Parameters
    ----------
    intensities : array-like
    target_rms : float, default 1.0

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    rms = np.sqrt(np.nanmean(intensities ** 2))
    if not np.isfinite(rms) or rms <= 0:
        warnings.warn(
            f"RMS normalization skipped: RMS is {rms}. Returning zeros.",
            UserWarning, stacklevel=2,
        )
        return np.zeros_like(intensities)
    return _safe_output(intensities * (float(target_rms) / rms))


# 4. Max normalization ------------------------------------------------------

def max_normalization(intensities):
    """Scale intensities so the maximum value equals 1.

    Parameters
    ----------
    intensities : array-like

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    mx = np.nanmax(np.abs(intensities))
    if not np.isfinite(mx) or mx <= 0:
        return np.zeros_like(intensities)
    return _safe_output(intensities / mx)


# 5. Vector (L2) normalization ---------------------------------------------

def vector_normalization(intensities):
    """Scale intensities to unit L2 norm (vector length = 1).

    Useful for comparing spectral *shape* irrespective of total signal.

    Parameters
    ----------
    intensities : array-like

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    norm = np.linalg.norm(intensities)
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros_like(intensities)
    return _safe_output(intensities / norm)


# 6. SNV (Standard Normal Variate) -----------------------------------------

def snv_normalization(intensities):
    """Standard Normal Variate: centre and scale to unit variance.

    Commonly used before multivariate analysis (PCA, PLS-DA) to remove
    multiplicative scatter effects.

    Parameters
    ----------
    intensities : array-like

    Returns
    -------
    np.ndarray
        Mean-centred, variance-scaled spectrum.  Note: values *can* be
        negative, which is expected for SNV.
    """
    intensities = np.asarray(intensities, dtype=float)
    mean = np.nanmean(intensities)
    std = np.nanstd(intensities)
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(intensities)
    result = (intensities - mean) / std
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# 7. Poisson scaling -------------------------------------------------------

def poisson_scaling(intensities):
    """Poisson (square-root mean) scaling for count data.

    Each channel is divided by ``sqrt(mean_intensity)`` across the
    spectrum.  This equalises the weight of low- and high-count channels
    when ToF-SIMS data follow Poisson statistics.  Widely used before PCA.

    Parameters
    ----------
    intensities : array-like

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    mean_int = np.nanmean(intensities)
    if not np.isfinite(mean_int) or mean_int <= 0:
        return np.zeros_like(intensities)
    return _safe_output(intensities / np.sqrt(mean_int))


# 8. Square-root transform -------------------------------------------------

def sqrt_normalization(intensities):
    """Square-root variance-stabilising transform.

    ``sqrt(intensity)`` stabilises the variance of Poisson-distributed ion
    counts.  Often combined with mean-centering before PCA.

    Parameters
    ----------
    intensities : array-like

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    return _safe_output(np.sqrt(np.clip(intensities, 0, None)))


# 9. Log transform ---------------------------------------------------------

def log_normalization(intensities, pseudo_count=1.0):
    """Log(1 + intensity) transform for high-dynamic-range spectra.

    Parameters
    ----------
    intensities : array-like
    pseudo_count : float, default 1.0
        Added before taking the log to avoid log(0).

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    return _safe_output(np.log(np.clip(intensities, 0, None) + float(pseudo_count)))


# 10. Selected-ion normalization -------------------------------------------

def selected_ion_normalization(intensities, reference_idx=None,
                                reference_intensity=None, target=1.0):
    """Normalise to a single reference peak (e.g. substrate or matrix ion).

    Provide *either* ``reference_idx`` (index into the intensity array) or
    ``reference_intensity`` (the absolute value to divide by).

    Parameters
    ----------
    intensities : array-like
    reference_idx : int, optional
        Index of the reference peak in *intensities*.
    reference_intensity : float, optional
        Absolute intensity value to normalise against.
    target : float, default 1.0
        Target value for the reference peak after normalisation.

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    if reference_intensity is not None:
        ref = float(reference_intensity)
    elif reference_idx is not None:
        if reference_idx < 0 or reference_idx >= len(intensities):
            raise IndexError(
                f"reference_idx={reference_idx} out of bounds for "
                f"spectrum of length {len(intensities)}."
            )
        ref = float(intensities[reference_idx])
    else:
        raise ValueError("Provide either reference_idx or reference_intensity.")

    if not np.isfinite(ref) or ref <= 0:
        warnings.warn(
            f"Selected-ion normalization skipped: reference intensity is "
            f"{ref}. Returning zeros.",
            UserWarning, stacklevel=2,
        )
        return np.zeros_like(intensities)
    return _safe_output(intensities * (float(target) / ref))


# 11. PQN (Probabilistic Quotient Normalization) ---------------------------

def pqn_normalization(intensities, reference=None):
    """Probabilistic Quotient Normalization.

    Designed for compositional data where a few species dominate.  Divides
    each channel by the median quotient relative to a *reference* spectrum.

    Parameters
    ----------
    intensities : array-like
    reference : array-like or None
        Reference spectrum (e.g. median of a dataset).  If ``None``,
        falls back to TIC normalization with a warning.

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    if reference is None:
        warnings.warn(
            "PQN requires a reference spectrum; falling back to TIC "
            "normalization.  For proper PQN, compute the reference from "
            "the full dataset.",
            UserWarning, stacklevel=2,
        )
        return tic_normalization(intensities)

    reference = np.asarray(reference, dtype=float)
    if reference.shape != intensities.shape:
        raise ValueError("reference must have the same shape as intensities.")

    # Step 1: TIC-normalise both
    tic_sample = tic_normalization(intensities)
    tic_ref = tic_normalization(reference)

    # Step 2: compute quotients where reference > 0
    mask = tic_ref > 0
    if not np.any(mask):
        return np.zeros_like(intensities)
    quotients = tic_sample[mask] / tic_ref[mask]
    med_quotient = np.nanmedian(quotients)

    if not np.isfinite(med_quotient) or med_quotient <= 0:
        return np.zeros_like(intensities)
    return _safe_output(tic_sample / med_quotient)


# 12. Median-of-Ratios normalization (DESeq2-style) ------------------------

def median_of_ratios_normalization(intensities, reference=None):
    """DESeq2-style median-of-ratios normalization.

    Computes the geometric mean spectrum as reference, then normalises
    each sample by the median ratio to that reference.  Robust to
    compositional effects.

    Parameters
    ----------
    intensities : array-like
    reference : array-like or None
        Pre-computed geometric-mean reference.  If ``None``, falls back to
        TIC normalization with a warning.

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    if reference is None:
        warnings.warn(
            "Median-of-ratios normalization requires a geometric-mean "
            "reference spectrum computed from the full dataset.  Falling "
            "back to TIC normalization.",
            UserWarning, stacklevel=2,
        )
        return tic_normalization(intensities)

    reference = np.asarray(reference, dtype=float)
    mask = reference > 0
    if not np.any(mask):
        return np.zeros_like(intensities)
    ratios = intensities[mask] / reference[mask]
    size_factor = np.nanmedian(ratios)
    if not np.isfinite(size_factor) or size_factor <= 0:
        return np.zeros_like(intensities)
    return _safe_output(intensities / size_factor)


# 13. VSN (Variance Stabilizing Normalization) -----------------------------

def vsn_normalization(intensities):
    """Variance-stabilising normalization via ``arcsinh`` transform.

    ``arcsinh(x)`` behaves like ``log(2x)`` for large values but handles
    zeros and small values gracefully.  Suitable for high-dynamic-range
    ToF-SIMS spectra.

    Parameters
    ----------
    intensities : array-like

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    return _safe_output(np.arcsinh(np.clip(intensities, 0, None)))


# 14. MinMax normalization --------------------------------------------------

def minmax_normalization(intensities, feature_range=(0.0, 1.0)):
    """Scale intensities to a fixed range (default [0, 1]).

    Parameters
    ----------
    intensities : array-like
    feature_range : tuple of float, default (0.0, 1.0)

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    y_min, y_max = np.nanmin(intensities), np.nanmax(intensities)
    if y_max == y_min:
        return np.full_like(intensities, feature_range[0])
    scaled = (intensities - y_min) / (y_max - y_min)
    new_min, new_max = feature_range
    return _safe_output(scaled * (new_max - new_min) + new_min)


# ---------------------------------------------------------------------------
#  Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "tic": tic_normalization,
    "median": median_normalization,
    "rms": rms_normalization,
    "max": max_normalization,
    "vector": vector_normalization,
    "snv": snv_normalization,
    "poisson": poisson_scaling,
    "sqrt": sqrt_normalization,
    "log": log_normalization,
    "selected_ion": selected_ion_normalization,
    "pqn": pqn_normalization,
    "median_of_ratios": median_of_ratios_normalization,
    "vsn": vsn_normalization,
    "minmax": minmax_normalization,
}
