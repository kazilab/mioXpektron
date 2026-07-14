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


# 7. Robust SNV ------------------------------------------------------------

def robust_snv_normalization(intensities, mad_scale: float = 1.4826):
    """Robust SNV using median and MAD instead of mean and standard deviation.

    This is less sensitive to a few dominant ions than classical SNV and is
    therefore a better fit when substrate/matrix peaks dominate part of the
    spectrum.

    Parameters
    ----------
    intensities : array-like
    mad_scale : float, default 1.4826
        Consistency factor turning MAD into a robust standard deviation
        estimate for approximately Gaussian data.

    Returns
    -------
    np.ndarray
        Median-centred, MAD-scaled spectrum. Negative values are expected.
    """
    intensities = np.asarray(intensities, dtype=float)
    median = np.nanmedian(intensities)
    mad = np.nanmedian(np.abs(intensities - median))
    scale = float(mad_scale) * mad
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(intensities)
    result = (intensities - median) / scale
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# 8. Poisson scaling -------------------------------------------------------

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


# 9. Square-root transform -------------------------------------------------

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


# 10. Log transform ---------------------------------------------------------

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


# 11. Selected-ion normalization -------------------------------------------

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


# 12. Multi-ion robust reference normalization -----------------------------

def multi_ion_reference_normalization(
    intensities,
    reference_indices=None,
    reference_values=None,
    target=1.0,
):
    """Normalize using multiple reference ions and a robust median ratio.

    Parameters
    ----------
    intensities : array-like
    reference_indices : sequence of int
        Indices of stable reference ions in the spectrum.
    reference_values : sequence of float, optional
        Expected intensities for the same reference ions. When provided the
        spectrum is scaled by the median observed/reference ratio. When
        omitted, the median observed intensity is scaled to ``target``.
    target : float, default 1.0
        Target robust centre when ``reference_values`` is omitted.

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    if reference_indices is None:
        raise ValueError("multi_ion_reference requires reference_indices.")

    idx = np.asarray(reference_indices, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("reference_indices must be a non-empty 1-D sequence.")
    if np.any(idx < 0) or np.any(idx >= len(intensities)):
        raise IndexError("reference_indices contain values outside the spectrum bounds.")

    observed = intensities[idx]
    if reference_values is not None:
        ref = np.asarray(reference_values, dtype=float)
        if ref.shape != observed.shape:
            raise ValueError(
                "reference_values must have the same shape as reference_indices."
            )
        mask = np.isfinite(observed) & np.isfinite(ref) & (observed > 0) & (ref > 0)
        if not np.any(mask):
            warnings.warn(
                "Multi-ion reference normalization skipped: no valid positive "
                "observed/reference pairs. Returning zeros.",
                UserWarning, stacklevel=2,
            )
            return np.zeros_like(intensities)
        ratios = observed[mask] / ref[mask]
        size_factor = np.nanmedian(ratios)
        if not np.isfinite(size_factor) or size_factor <= 0:
            return np.zeros_like(intensities)
        scaled = intensities / size_factor
    else:
        mask = np.isfinite(observed) & (observed > 0)
        if not np.any(mask):
            warnings.warn(
                "Multi-ion reference normalization skipped: no valid positive "
                "reference ion intensities. Returning zeros.",
                UserWarning, stacklevel=2,
            )
            return np.zeros_like(intensities)
        robust_ref = np.nanmedian(observed[mask])
        if not np.isfinite(robust_ref) or robust_ref <= 0:
            return np.zeros_like(intensities)
        scaled = intensities * (float(target) / robust_ref)

    return _safe_output(scaled)


# 13. PQN (Probabilistic Quotient Normalization) ---------------------------

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


# 14. Mass-stratified PQN --------------------------------------------------

def mass_stratified_pqn_normalization(
    intensities,
    mz_values=None,
    reference=None,
    strata=None,
):
    """Apply PQN separately across coarse m/z strata.

    This keeps a global TIC-normalised baseline while estimating local PQN
    size factors for different m/z regions.

    Parameters
    ----------
    intensities : array-like
    mz_values : array-like
        m/z axis shared with ``intensities``.
    reference : array-like
        Dataset-level reference spectrum on the same m/z grid.
    strata : sequence of tuple(float, float), optional
        Inclusive/exclusive m/z windows ``[(lo, hi), ...]``. Defaults to
        ``[(0, 100), (100, 400), (400, inf)]``.

    Returns
    -------
    np.ndarray
    """
    intensities = np.asarray(intensities, dtype=float)
    if mz_values is None:
        raise ValueError("mass_stratified_pqn requires mz_values.")
    if reference is None:
        warnings.warn(
            "Mass-stratified PQN requires a dataset-level reference spectrum; "
            "falling back to PQN/TIC behaviour.",
            UserWarning, stacklevel=2,
        )
        return pqn_normalization(intensities)

    mz_values = np.asarray(mz_values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mz_values.shape != intensities.shape or reference.shape != intensities.shape:
        raise ValueError("mz_values and reference must have the same shape as intensities.")

    if strata is None:
        strata = [(0.0, 100.0), (100.0, 400.0), (400.0, np.inf)]

    tic_sample = tic_normalization(intensities)
    tic_ref = tic_normalization(reference)
    global_mask = tic_ref > 0
    if not np.any(global_mask):
        return np.zeros_like(intensities)

    global_factor = np.nanmedian(tic_sample[global_mask] / tic_ref[global_mask])
    if not np.isfinite(global_factor) or global_factor <= 0:
        return np.zeros_like(intensities)

    scaled = np.empty_like(tic_sample)
    covered = np.zeros_like(tic_sample, dtype=bool)

    for lo, hi in strata:
        if np.isinf(hi):
            mask = (mz_values >= float(lo))
        else:
            mask = (mz_values >= float(lo)) & (mz_values < float(hi))
        covered |= mask

        local_mask = mask & (tic_ref > 0)
        if np.any(local_mask):
            local_factor = np.nanmedian(tic_sample[local_mask] / tic_ref[local_mask])
            if not np.isfinite(local_factor) or local_factor <= 0:
                local_factor = global_factor
        else:
            local_factor = global_factor
        scaled[mask] = tic_sample[mask] / local_factor

    scaled[~covered] = tic_sample[~covered] / global_factor
    return _safe_output(scaled)


# 15. Median-of-Ratios normalization (DESeq2-style) ------------------------

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


# 16. VSN (Variance Stabilizing Normalization) -----------------------------

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


# 17. MinMax normalization --------------------------------------------------

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


# 18. Pareto scaling -------------------------------------------------------

def pareto_normalization(intensities, mean=None, std=None, eps: float = 1e-12):
    """Pareto scale a spectrum using dataset-level feature statistics.

    Pareto scaling is a *dataset-level* transform commonly used before PCA:
    each feature is mean-centred and divided by ``sqrt(std_feature)``.  This
    down-weights very intense ions less aggressively than autoscaling while
    still reducing dominance by a few channels.

    Parameters
    ----------
    intensities : array-like
    mean : array-like
        Per-feature dataset mean with the same shape as ``intensities``.
    std : array-like
        Per-feature dataset standard deviation with the same shape as
        ``intensities``.
    eps : float, default 1e-12
        Numerical floor preventing division by zero.

    Returns
    -------
    np.ndarray
        Mean-centred, Pareto-scaled spectrum. Negative values are expected.

    Raises
    ------
    ValueError
        If dataset-level mean/std arrays are not provided.
    """
    intensities = np.asarray(intensities, dtype=float)
    if mean is None or std is None:
        raise ValueError(
            "Pareto normalization requires dataset-level 'mean' and 'std' "
            "arrays matching the spectrum shape."
        )

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.shape != intensities.shape or std.shape != intensities.shape:
        raise ValueError("mean and std must have the same shape as intensities.")

    scale = np.sqrt(np.clip(std, eps, None))
    result = (intensities - mean) / scale
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


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
    "robust_snv": robust_snv_normalization,
    "poisson": poisson_scaling,
    "sqrt": sqrt_normalization,
    "log": log_normalization,
    "selected_ion": selected_ion_normalization,
    "multi_ion_reference": multi_ion_reference_normalization,
    "pqn": pqn_normalization,
    "mass_stratified_pqn": mass_stratified_pqn_normalization,
    "median_of_ratios": median_of_ratios_normalization,
    "vsn": vsn_normalization,
    "minmax": minmax_normalization,
    "pareto": pareto_normalization,
}
