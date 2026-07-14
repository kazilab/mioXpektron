"""
ToF-SIMS 1D denoising & smoothing utilities
------------------------------------------
A focused toolkit for 1D spectra (e.g., ToF‑SIMS / MS) that implements
wavelet‑shrinkage denoising and classical smoothing filters, plus
resampling to a uniform grid when needed. The design emphasizes:

• Peak preservation: track height, FWHM, area, and m/z boundaries.
• Noise reduction: robust σ̂ via MAD (baseline regions) and support for
  variance‑stabilizing transforms (Anscombe) for count‑like noise.
• Practical ergonomics: optional cycle‑spinning for translation
  invariance, TIC preservation, and shape‑preserving interpolation.

This module provides the *primitive operations* used by the higher‑level
selection & evaluation code.
"""

from __future__ import annotations

import warnings
from typing import Optional, Literal, Tuple, Dict, List, Any
import numpy as np
import pywt
from scipy import signal, ndimage
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid


# =============================== helpers ================================== #

def _mad_sigma(x: np.ndarray) -> float:
    """Robust noise estimate via MAD/0.6745 (median absolute deviation)."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.67448975)


def _sure_threshold_fast(
        coefs: np.ndarray,
        sigma: float
        ) -> float:
    """
    Soft-threshold that (approximately) minimizes SURE for Gaussian noise.

    Vectorized O(n log n) evaluation over all unique |w|-knots:
        SURE(t) = n*sigma^2 + Σ min(w_i^2, t^2) - 2*sigma^2 * #{|w_i| <= t}

    Notes
    -----
    - Equivalent to Donoho-Johnstone SureShrink evaluation at the order
      statistics of |w|. We include t=0 implicitly via k=-1 case if desired,
      but practically thresholds >0 dominate.
    """
    y = np.asarray(coefs, dtype=np.float64)
    n = y.size
    if n == 0 or not np.isfinite(sigma) or sigma <= 0:
        return 0.0

    a = np.abs(y)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0

    a.sort()                                 # ascending |w|
    s2 = float(sigma) * float(sigma)
    a2 = a * a

    # 1-based cumulative sum for easy slicing
    csum = np.empty(a2.size + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(a2, out=csum[1:])

    # Evaluate SURE at unique knots t = a[k] using RIGHT boundary (counts <= t)
    uniq = np.unique(a)
    m = np.searchsorted(a, uniq, side="right")      # m_j = #{|w_i| <= uniq[j]}
    t2 = uniq * uniq

    # sum_i min(a_i^2, t^2) = sum_{i<=m} a_i^2 + (n - m) * t^2
    sum_min = csum[m] + (n - m) * t2
    risk = n * s2 + sum_min - 2.0 * s2 * m

    j = int(np.argmin(risk))
    return float(uniq[j])
    

def _sure_threshold_optimized(
        coefs: np.ndarray,
        sigma: float,
        max_candidates: int = 128,
        refine_window: int = 20
        ) -> float:
    """
    Optimized SURE threshold with candidate subsampling, refinement, and inclusion of t=0.
    Addresses approximation risks by refining around the best coarse candidate.
    Biases candidate selection toward larger thresholds.
    """
    y = np.asarray(coefs, dtype=np.float64)
    n = y.size
    if n == 0 or sigma <= 0:
        return 0.0
    
    ay = np.abs(y)
    ay = ay[np.isfinite(ay)]
    if ay.size == 0:
        return 0.0
    
    ay_sorted = np.sort(ay)  # ascending
    s2 = sigma * sigma
    
    # Precompute cumulative sum of ay**2
    ay2_cumsum = np.cumsum(ay_sorted ** 2)
    csum = np.r_[0.0, ay2_cumsum]  # prefixed for csum[m] = sum of first m ay**2
    
    # Adaptive candidate selection: bias toward larger t with power law
    n_candidates = min(max_candidates, max(32, n // 4))
    frac = np.linspace(0, 1, n_candidates) ** 1.5  # bias to higher
    indices = np.unique(np.round(frac * (len(ay_sorted) - 1)).astype(int))
    candidates = np.unique(ay_sorted[indices])  # unique to avoid dups
    
    # Include t=0 explicitly if not already (though often ay_sorted[0] >=0)
    if candidates[0] > 0:
        candidates = np.r_[0.0, candidates]
    
    # Function to compute risk for a given t
    def compute_risk(t: float) -> float:
        """Evaluate the SURE risk at threshold ``t`` for the current spectrum."""
        if t == 0:
            m0 = np.searchsorted(ay_sorted, 0.0, side='right')  # count(|w_i| <= 0)
            # For t=0: sum_min = 0, risk = n*s2 - 2*s2*m0
            return n * s2 - 2.0 * s2 * m0
        m = np.searchsorted(ay_sorted, t, side='right')  # # <= t
        t2 = t * t
        sum_min = csum[m] + (n - m) * t2
        risk = n * s2 + sum_min - 2.0 * s2 * m
        return risk
    
    # Coarse search: find best among candidates
    risks = np.array([compute_risk(t) for t in candidates])
    j = np.argmin(risks)
    best_t_coarse = candidates[j]
    # best_risk_coarse = risks[j]
    
    # Refinement: evaluate dense around the best coarse t
    m_coarse = np.searchsorted(ay_sorted, best_t_coarse, side='right')
    start = max(0, m_coarse - refine_window)
    end = min(n, m_coarse + refine_window + 1)
    fine_candidates = np.unique(ay_sorted[start:end])  # unique in window
    
    # Compute risks for fine candidates
    fine_risks = np.array([compute_risk(t) for t in fine_candidates])
    k = np.argmin(fine_risks)
    best_t = fine_candidates[k]
    
    return float(best_t)


def _bayes_threshold(
        detail: np.ndarray,
        sigma: float
        ) -> float:
    """
    BayesShrink threshold: t = sigma^2 / sqrt(max(var(detail) - sigma^2, 0)).
    Safer fallback: Universal threshold if var_x <= 0.
    """
    d = np.asarray(detail, dtype=np.float64)
    n = d.size
    if n == 0 or sigma <= 0 or not np.isfinite(sigma):
        return 0.0
    var_w = float(np.var(d))
    var_x = max(var_w - sigma * sigma, 0.0)
    if var_x <= 0.0:
        return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))
    return float((sigma * sigma) / np.sqrt(var_x))


def _circular_shift(x: np.ndarray, k: int) -> np.ndarray:
    """Circularly shift a 1D array to the right by k samples (mod N)."""
    return np.roll(x, int(k))


def _inverse_circular_shift(x: np.ndarray, k: int) -> np.ndarray:
    """Inverse circular shift for `_circular_shift` (left by k samples)."""
    return np.roll(x, -int(k))


def _resample_to_uniform(
    x: np.ndarray,
    y: np.ndarray,
    target_dx: Optional[float] = None,
    *,
    forward_kind: Literal["pchip", "linear"] = "pchip",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate (x, y) to a uniform grid (xu, yu) using shape-preserving PCHIP
    (default) or linear interpolation, and return (xu, yu, x_in, idx_in).

    Parameters
    ----------
    x : np.ndarray
        Original, possibly nonuniform x-axis values.
    y : np.ndarray
        Signal values aligned to x.
    target_dx : float, optional
        Desired spacing for the uniform grid. If None, inferred from the median
        positive spacing in x (falls back to mean of diffs if necessary).
    forward_kind : {"pchip", "linear"}
        Interpolant for the forward resampling step; PCHIP preserves peak shapes.

    Returns
    -------
    xu : np.ndarray
        Uniform x-grid spanning [min(x), max(x)].
    yu : np.ndarray
        Resampled y on the uniform grid xu.
    x_in : np.ndarray
        Finite x samples from the caller, in the caller's original order.
    idx_in : np.ndarray
        Indices of finite samples relative to the original arrays.

    Notes
    -----
    Internally, (x, y) are sorted by x to build the forward interpolant. Mapping
    back to the caller's order is done later by evaluating a PCHIP defined on
    (xu, out_u) at x_in; this preserves the caller's order and avoids silent
    reordering.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    idx_in = np.nonzero(mask)[0]
    x_in, y_in = x[mask], y[mask]

    if x_in.size < 2:
        return x_in, y_in, x_in, idx_in

    # sort by x to create the forward interpolant
    order = np.argsort(x_in)
    xs, ys = x_in[order], y_in[order]

    diffs = np.diff(xs)
    pos = diffs[np.isfinite(diffs) & (diffs > 0)]
    if target_dx is None or not np.isfinite(target_dx) or target_dx <= 0:
        target_dx = float(np.median(pos)) if pos.size else float(np.mean(diffs))

    n = max(2, int(np.round((xs[-1] - xs[0]) / target_dx)) + 1)
    xu = np.linspace(xs[0], xs[-1], n)

    if forward_kind == "pchip":
        fwd = PchipInterpolator(xs, ys, extrapolate=True)
        yu = fwd(xu)
    else:
        yu = np.interp(xu, xs, ys, left=ys[0], right=ys[-1])

    return xu, np.asarray(yu, dtype=np.float64), x_in, idx_in


def _anscombe_forward(y: np.ndarray) -> np.ndarray:
    """
    Classical Anscombe VST for Poisson counts:
        z = 2 * sqrt(y + 3/8)
    Requires y >= 0.
    """
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * np.sqrt(y + 3.0 / 8.0)


def _anscombe_inverse_unbiased(z: np.ndarray) -> np.ndarray:
    """
    Unbiased inverse of the classical Anscombe transform (Mäkitalo & Foi).
    Accurate for small counts; asymptotic series with safeguards.
    This is the consistent inverse for the classical Anscombe forward transform
    under a pure Poisson noise model. Reference: Mäkitalo & Foi, IEEE TIP 2013.
    """
    z = np.asarray(z, dtype=np.float64)
    # Prevent division by zero in series
    z2 = np.maximum(z, 1e-12)
    # Asymptotic series (pure Poisson case)
    # I(z) ≈ (z/2)^2 - 1/8 + 1/(8 z^2) - 5/(128 z^4) + 7/(512 z^6) - 21/(1024 z^8)
    inv = (z2 * 0.5) ** 2 - 0.125 \
        + 1.0 / (8.0 * z2 ** 2) \
        - 5.0 / (128.0 * z2 ** 4) \
        + 7.0 / (512.0 * z2 ** 6) \
        - 21.0 / (1024.0 * z2 ** 8)
    # Clip tiny negatives caused by numerical error
    return np.maximum(inv, 0.0)


def _prepare_anscombe_input(
    y: np.ndarray,
    *,
    negative_policy: Literal["warn_clip", "clip", "raise"] = "warn_clip",
) -> np.ndarray:
    """Validate non-negative input for the classical Anscombe transform."""
    y = np.asarray(y, dtype=np.float64)
    neg_mask = np.isfinite(y) & (y < 0.0)
    if not np.any(neg_mask):
        return y

    n_neg = int(np.count_nonzero(neg_mask))
    min_neg = float(np.min(y[neg_mask]))
    message = (
        "Classical Anscombe VST assumes non-negative Poisson-like input, "
        f"but found {n_neg} negative values (min={min_neg:.6g})."
    )

    if negative_policy == "raise":
        raise ValueError(message + " Use variance_stabilize='none' or preprocess the signal before denoising.")
    if negative_policy not in {"warn_clip", "clip"}:
        raise ValueError("anscombe_negative_policy must be 'warn_clip', 'clip', or 'raise'")
    if negative_policy == "warn_clip":
        warnings.warn(
            message + " Clipping negatives to zero before the classical Anscombe transform.",
            RuntimeWarning,
            stacklevel=3,
        )

    y = y.copy()
    y[neg_mask] = 0.0
    return y


# =========================== wavelet denoising ============================ #

def wavelet_denoise(
    intensities: np.ndarray,
    *,
    wavelet: Literal["db4", "db8", "sym5", "sym8", "coif2", "coif3"] = "sym8",
    level: Optional[int] = None,
    threshold_mode: Literal["soft", "hard"] = "soft",
    threshold_strategy: Literal["universal", "bayes", "sure", "sure_opt"] = "universal",
    sigma: Optional[float] = None,
    sigma_strategy: Literal["per_level", "global"] = "per_level",
    variance_stabilize: Literal["none", "anscombe"] = "none",
    anscombe_negative_policy: Literal["warn_clip", "clip", "raise"] = "warn_clip",
    cycle_spins: Literal[0, 4, 8, 16, 32] = 0,
    pywt_mode: str = "periodization",
    clip_nonnegative: bool = True,
    preserve_tic: bool = False,
) -> np.ndarray:
    """
    Wavelet-shrinkage denoising for 1D ToF-SIMS/MS data, with optional
    translation invariance (cycle-spinning) and TIC preservation.

    Parameters
    ----------
    intensities : array-like
        1D intensities (finite values recommended).
    wavelet : {"db4","db8","sym5","sym8","coif2","coif3"}
        Discrete wavelet family used for the DWT.
    level : int or None
        DWT level; if None, chosen conservatively via `pywt.dwt_max_level` (capped at 6).
    threshold_mode : {"soft","hard"}
        Shrinkage mode (soft is default and recommended for denoising).
    threshold_strategy : {"universal","bayes","sure","sure_opt"}
        Threshold selection per detail level:
          • "universal": √(2 log N) scaling (Donoho–Johnstone)
          • "bayes": BayesShrink (subband variance model)
          • "sure": SURE-threshold via full knot evaluation (fast, exact at knots)
          • "sure_opt": SURE with biased candidate subsampling + local refinement (faster)
    sigma : float or None
        If provided, used as a *global* noise std at all levels (overrides sigma_strategy).
    sigma_strategy : {"per_level","global"}
        How to estimate σ when `sigma` is None.
          • "per_level": σ_j via MAD on each detail subband (robust; recommended)
          • "global": a single σ via MAD on the finest detail of the unshifted input
    variance_stabilize : {"none","anscombe"}
        Apply a variance-stabilizing transform prior to denoising. ``"anscombe"``
        uses the classical Anscombe forward transform with the Mäkitalo-Foi
        unbiased inverse, appropriate for non-negative Poisson-like input.
    anscombe_negative_policy : {"warn_clip","clip","raise"}
        How to handle negative values before the classical Anscombe transform.
        ``"warn_clip"`` preserves backward compatibility but emits a warning;
        ``"raise"`` is recommended when negative values are scientifically
        meaningful residuals rather than impossible counts.
    cycle_spins : {0,4,8,16,32}
        Number of circular shifts to average (0 disables; try 8–16 for higher SNR if compute allows).
        If cycle_spins ≥ N on very long vectors, the implementation caps the number of shifts at 2048 for performance.
    pywt_mode : str
        Wavelet signal extension mode. Common choices: 'periodization' (reduces edge artifacts),
        'symmetric', 'reflect'. See PyWavelets docs for full list.
    clip_nonnegative : bool
        Clip negative outputs to zero (recommended for intensities).
    preserve_tic : bool
        If True, rescale output so sum(out) == sum(input) (guarded to avoid division by zero).

    Returns
    -------
    np.ndarray
        Denoised intensities, same length as input.

    Raises
    ------
    ValueError
        If `intensities` is not 1D, or an unknown `threshold_strategy` is provided.
    
    See Also
    --------
    noise_filtering : High-level wrapper supporting classical smoothers and resampling.
    """
    y0 = np.asarray(intensities, dtype=np.float64)
    if y0.ndim != 1:
        raise ValueError("'intensities' must be a 1D array")
    if anscombe_negative_policy not in {"warn_clip", "clip", "raise"}:
        raise ValueError("anscombe_negative_policy must be 'warn_clip', 'clip', or 'raise'")
    
    # Precompute global sigma once (unshifted signal) if requested and not provided.
    # If requested, precompute a single global σ on the *unshifted* input using
    # the finest detail subband (robust to nonstationary low-freq content).
    precomputed_global_sigma: Optional[float] = None
    if sigma is None and sigma_strategy == "global":
        w = pywt.Wavelet(wavelet)
        lvl0 = level if level is not None else max(1, min(6, pywt.dwt_max_level(y0.size, w.dec_len)))
        y0_finite = np.where(np.isfinite(y0), y0, 0.0)
        coeffs0 = pywt.wavedec(y0_finite, wavelet=wavelet, level=lvl0, mode=pywt_mode)
        # cD1 is the finest detail band in PyWavelets: coeffs = [cA_n, cD_n, ..., cD_1]
        finest = coeffs0[-1] if len(coeffs0) >= 2 else coeffs0[0]
        precomputed_global_sigma = _mad_sigma(finest)
    
    def _preprocess(y: np.ndarray) -> np.ndarray:
        """Replace NaNs and optionally apply the forward variance transform."""
        y = np.where(np.isfinite(y), y, 0.0)
        if variance_stabilize == "anscombe":
            return _anscombe_forward(
                _prepare_anscombe_input(y, negative_policy=anscombe_negative_policy)
            )
        return y

    def _postprocess(y: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
        """Undo variance-stabilizing transforms and apply TIC/clip policies."""
        # Inverse VST first (nonlinear), then TIC & clipping
        if variance_stabilize == "anscombe":
            y = _anscombe_inverse_unbiased(y)
        # TIC scaling relative to the original raw input
        if preserve_tic:
            s_in = np.nansum(np.where(np.isfinite(y_raw), y_raw, 0.0))
            s_out = np.nansum(np.where(np.isfinite(y), y, 0.0))
            if np.isfinite(s_in) and np.isfinite(s_out) and s_out > 0:
                y = y * (s_in / s_out)
        if clip_nonnegative:
            y = np.maximum(y, 0.0)
        return y

    def _single_pass(y: np.ndarray) -> np.ndarray:
        """Denoise a single (possibly shifted) realization of the signal."""
        y_v = _preprocess(y)
        w_ = pywt.Wavelet(wavelet)
        # Choose a conservative decomposition level (cap at 6 to avoid over-decomposition on long signals)
        lvl = level if level is not None else max(1, min(6, pywt.dwt_max_level(y_v.size, w_.dec_len)))
        coeffs = pywt.wavedec(y_v, wavelet=wavelet, level=lvl, mode=pywt_mode)

        # decide sigma policy (your existing logic)
        if sigma is not None:
            global_sigma = float(sigma)
        elif sigma_strategy == "global":
            global_sigma = float(precomputed_global_sigma) if precomputed_global_sigma is not None else 0.0
        else:
            global_sigma = None

        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            c = np.where(np.isfinite(c), c, 0.0)
            sigma_j = _mad_sigma(c) if global_sigma is None else global_sigma
            if sigma_j <= 0 or not np.isfinite(sigma_j):
                new_coeffs.append(c)
                continue
            if threshold_strategy == "universal":
                t = sigma_j * np.sqrt(2.0 * np.log(max(y_v.size, 2)))
            elif threshold_strategy == "bayes":
                t = _bayes_threshold(c, sigma_j)
            elif threshold_strategy == "sure":
                t = _sure_threshold_fast(c, sigma_j)
            elif threshold_strategy == "sure_opt":
                t = _sure_threshold_optimized(c, sigma_j)
            else:
                raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_thr = pywt.threshold(c, value=float(t), mode=threshold_mode)
            new_coeffs.append(np.where(np.isfinite(c_thr), c_thr, 0.0))

        out_v = pywt.waverec(new_coeffs, wavelet, mode=pywt_mode)[: y_v.size]
        return out_v

    if cycle_spins and cycle_spins > 0 and y0.size > 8:
        acc = np.zeros_like(y0, dtype=float)
        N = y0.size
        # Use all shifts if spins >= N; otherwise sample evenly-spaced shifts across [0, N)
        # Guard: on very long vectors, using all shifts can be prohibitively expensive.
        # If spins >= N and N is large, cap to a reasonable maximum of evenly spaced shifts.
        max_spins = 2048
        effective_spins = cycle_spins
        if cycle_spins >= N:
            effective_spins = min(N, max_spins)
            if N > max_spins:
                warnings.warn(
                    f"cycle_spins >= N ({cycle_spins} >= {N}); capping to "
                    f"{max_spins} evenly spaced shifts for performance. "
                    f"This may reduce denoising accuracy compared to full "
                    f"cycle-spinning.",
                    UserWarning,
                    stacklevel=2,
                )
        elif cycle_spins > max_spins:
            effective_spins = max_spins
            warnings.warn(
                f"cycle_spins={cycle_spins} exceeds cap of {max_spins}; "
                f"using {max_spins} evenly spaced shifts.",
                UserWarning,
                stacklevel=2,
            )
        if effective_spins >= N:
            steps = np.arange(N, dtype=int)
        else:
            steps = np.unique(np.round(np.linspace(0, N - 1, num=effective_spins)).astype(int))
        for k in steps:
            acc += _inverse_circular_shift(_single_pass(_circular_shift(y0, k)), k)
        denoised_v = acc / float(steps.size)
    else:
        denoised_v = _single_pass(y0)

    # Inverse VST, TIC, clipping
    denoised = _postprocess(denoised_v, y0)
    denoised = np.where(np.isfinite(denoised), denoised, 0.0)
    return denoised


# ==================== traditional smoothing + resampling =================== #

def noise_filtering(
    intensities: np.ndarray,
    *,
    method: Literal["savitzky_golay", "gaussian", "median", "wavelet", "none"] = "wavelet",
    # Savitzky–Golay / median / gaussian parameters
    window_length: int = 15,
    polyorder: int = 3,
    deriv: int = 0,
    gauss_sigma_pts: Optional[float] = None,
    gaussian_order: int = 0,
    # Wavelet parameters
    wavelet: Literal["db4", "db8", "sym5", "sym8", "coif2", "coif3"] = "sym8",
    level: Optional[int] = None,
    threshold_strategy: Literal["universal", "bayes", "sure", "sure_opt"] = "universal",
    threshold_mode: Literal["soft", "hard"] = "soft",
    sigma: Optional[float] = None,
    sigma_strategy: Literal["per_level", "global"] = "per_level",
    variance_stabilize: Literal["none", "anscombe"] = "none",
    anscombe_negative_policy: Literal["warn_clip", "clip", "raise"] = "warn_clip",
    cycle_spins: Literal[0, 4, 8, 16, 32] = 0,
    pywt_mode: str = "periodization",
    # Shared/output behavior
    clip_nonnegative: bool = True,
    preserve_tic: bool = False,
    # Sampling & resampling
    x: Optional[np.ndarray] = None,
    resample_to_uniform: bool = False,
    target_dx: Optional[float] = None,
    forward_interp: Literal["pchip", "linear"] = "pchip",
) -> np.ndarray:
    """
    Apply 1D denoising/smoothing to ToF-SIMS spectra.

    Notes
    -----
    - Savitzky–Golay / Gaussian / Median assume ~uniform sampling. If your m/z
      grid is nonuniform, pass `x` and set `resample_to_uniform=True`.
      The wavelet path can also resample when `resample_to_uniform=True`.
    - Wavelet shrinkage preserves narrow peaks; consider Bayes/SURE and cycle-spins.

    Parameters
    ----------
    intensities : np.ndarray
        1D intensity array.
    method : {'savitzky_golay','gaussian','median','wavelet','none'}
    window_length : int
        Odd window for Savitzky–Golay or median; will be coerced to odd >=3.
    polyorder : int
        For Savitzky–Golay, 0 ≤ polyorder < window_length.
    deriv : int
        For Savitzky–Golay, derivative order (0 = smoothing; 1/2/... compute derivatives).
        Requires `polyorder >= deriv`.
    gauss_sigma_pts : float or None
        If provided, overrides default sigma = window_length/6 for Gaussian filter.
    gaussian_order : int
        For Gaussian filtering, derivative order for `ndimage.gaussian_filter1d`.
        0 = smoothing; >0 computes derivatives.
    wavelet, level, threshold_strategy, threshold_mode, sigma, cycle_spins, pywt_mode
        Passed to wavelet processing (see `wavelet_denoise`).
    sigma_strategy : {"per_level","global"}
        Strategy if `sigma` is None. "per_level" = σ_j via MAD on each detail subband;
        "global" = one σ via MAD on the finest detail of the unshifted input.
    variance_stabilize : {"none","anscombe"}
        Apply variance‑stabilizing transform before denoising. ``"anscombe"``
        uses the classical Anscombe transform for non-negative Poisson-like input.
    anscombe_negative_policy : {"warn_clip","clip","raise"}
        Handling policy for negative values before the classical Anscombe
        transform.
    clip_nonnegative, preserve_tic
        Output behaviors.
    x : np.ndarray or None
        Optional m/z (or channel) axis aligned with intensities.
    resample_to_uniform : bool
        If True and x is provided, internally resample to a uniform grid and back.
    target_dx : float or None
        Target spacing for the uniform grid (if None, inferred).
    forward_interp : {'pchip','linear'}
        Interpolant used when building the uniform-grid signal (PCHIP recommended).

    Returns
    -------
    np.ndarray
        Filtered intensities aligned to the input grid/order.

    Raises
    ------
    ValueError
        If `intensities` or `x` have mismatched shapes or if `intensities` is not 1D.
        If Savitzky–Golay has `polyorder < deriv` after clamping, or if `method` is unknown.
    See Also
    --------
    wavelet_denoise : Core wavelet denoising routine used when `method="wavelet"`.
    """
    y = np.asarray(intensities, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("'intensities' must be a 1D array")

    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and intensities must have the same shape")

    # Work only on finite entries; reconstruct full-length output at the end
    if x is None:
        mask = np.isfinite(y)
    else:
        mask = np.isfinite(y) & np.isfinite(x)
    idx = np.nonzero(mask)[0]

    if idx.size == 0:
        out = np.zeros_like(y, dtype=float)
        return out

    y_work = y[mask]
    x_work = x[mask] if x is not None else None

    def _map_back(out_in_: np.ndarray) -> np.ndarray:
        """Restore filtered samples onto the original indexing with safeguards."""
        # Reconstruct a full-length output aligned to the original indices; apply optional clipping and TIC scaling
        out_full = np.zeros_like(y, dtype=float)
        out_full[mask] = out_in_
        if clip_nonnegative:
            out_full = np.maximum(out_full, 0.0)
        if preserve_tic:
            s_in = y_work.sum()
            s_out = out_in_.sum()
            if np.isfinite(s_in) and np.isfinite(s_out) and s_out > 0:
                scale = s_in / s_out
                out_full[mask] *= scale
        return out_full

    # Optional resampling branch (also available for wavelet)
    use_uniform = (x_work is not None) and resample_to_uniform and method in {
        "savitzky_golay", "gaussian", "median", "wavelet"
    }

    if method == "none":
        return _map_back(y_work.copy())

    if use_uniform:
        xu, yu, x_in, idx_in = _resample_to_uniform(
            x_work, y_work, target_dx=target_dx, forward_kind=forward_interp
        )
        # Apply the chosen filter on the uniform grid
        if method == "savitzky_golay":
            wl = max(3, int(window_length))
            if wl % 2 == 0:
                wl += 1
            poly = int(polyorder)
            poly = min(max(0, poly), wl - 1)
            d = max(0, int(deriv))
            if poly < d:
                raise ValueError("Savitzky–Golay requires polyorder >= deriv")
            out_u = signal.savgol_filter(yu, window_length=wl, polyorder=poly, deriv=d, mode="interp")
            
        elif method == "gaussian":
            sigma_pts = float(gauss_sigma_pts) if gauss_sigma_pts is not None else max(0.5, float(window_length) / 6.0)  # heuristic ≈ window/6
            gorder = max(0, int(gaussian_order))
            out_u = ndimage.gaussian_filter1d(yu, sigma=sigma_pts, order=gorder, mode="nearest")
            
        elif method == "median":
            wl = max(3, int(window_length))
            if wl % 2 == 0:
                wl += 1
            out_u = signal.medfilt(yu, kernel_size=wl)
            
        elif method == "wavelet":
            out_u = wavelet_denoise(
                yu,
                wavelet=wavelet,
                level=level,
                threshold_mode=threshold_mode,
                threshold_strategy=threshold_strategy,
                sigma=sigma,
                sigma_strategy=sigma_strategy,
                variance_stabilize=variance_stabilize,
                anscombe_negative_policy=anscombe_negative_policy,
                cycle_spins=cycle_spins,
                pywt_mode=pywt_mode,
                clip_nonnegative=False,   # clip after mapping back
                preserve_tic=False,       # preserve after mapping back
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Map back to the caller's original x ordering (shape-preserving)
        back_interp = PchipInterpolator(xu, out_u, extrapolate=True)
        out_in = back_interp(x_in)
        return _map_back(np.asarray(out_in, dtype=np.float64))

    # No resampling branch (operate directly on y_work)
    if method == "savitzky_golay":
        wl = max(3, int(window_length))
        if wl % 2 == 0:
            wl += 1
        poly = int(polyorder)
        poly = min(max(0, poly), wl - 1)
        d = max(0, int(deriv))
        if poly < d:
            raise ValueError("Savitzky–Golay requires polyorder >= deriv")
        out_in = signal.savgol_filter(y_work, window_length=wl, polyorder=poly, deriv=d, mode="interp")

    elif method == "gaussian":
        sigma_pts = float(gauss_sigma_pts) if gauss_sigma_pts is not None else max(0.5, float(window_length) / 6.0)  # heuristic ≈ window/6
        gorder = max(0, int(gaussian_order))
        out_in = ndimage.gaussian_filter1d(y_work, sigma=sigma_pts, order=gorder, mode="nearest")

    elif method == "median":
        wl = max(3, int(window_length))
        if wl % 2 == 0:
            wl += 1
        out_in = signal.medfilt(y_work, kernel_size=wl)

    elif method == "wavelet":
        out_in = wavelet_denoise(
            y_work,
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
            threshold_strategy=threshold_strategy,
            sigma=sigma,
            sigma_strategy=sigma_strategy,
            variance_stabilize=variance_stabilize,
            anscombe_negative_policy=anscombe_negative_policy,
            cycle_spins=cycle_spins,
            pywt_mode=pywt_mode,
            clip_nonnegative=True,
            preserve_tic=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return _map_back(np.asarray(out_in, dtype=np.float64))
