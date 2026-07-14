# Data import function
import logging
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
from scipy import signal, optimize, special, integrate
from pybaselines import Baseline

from ..baseline.baseline_base import baseline_correction
from ..denoise.denoise_main import noise_filtering
from ..normalization.normalization import tic_normalization
from ..utils.file_management import _resolve_group
from ..utils.file_management import import_data

_PEAK_RESULT_COLUMNS = [
    "PeakCenter",
    "PeakWidth",
    "Prominences",
    "Amplitude",
    "PeakArea",
    "SampleName",
    "Group",
    "DetectedBy",
    "Deconvoluted",
    "FitModel",
    "AreaDefinition",
    "WidthDefinition",
    "IntegrationMethod",
]


def _empty_peak_properties_df():
    """Return an empty peak table with the canonical result schema."""

    return pd.DataFrame(columns=_PEAK_RESULT_COLUMNS)


def _log_peak_fit_failure_summary(
        sample_name,
        method,
        *,
        single_attempts=0,
        single_failures=0,
        deconv_attempts=0,
        deconv_failures=0,
        ):
    """Emit one warning summarizing analytic fit failures for a spectrum."""
    if single_failures <= 0 and deconv_failures <= 0:
        return

    parts = []
    if single_attempts > 0:
        parts.append(f"{single_failures}/{single_attempts} single-peak fits failed")
    if deconv_attempts > 0:
        parts.append(f"{deconv_failures}/{deconv_attempts} deconvolution fits raised exceptions")

    logger.warning(
        "%s: %s detection skipped some analytic fits (%s)",
        sample_name,
        method,
        "; ".join(parts),
    )


def handle_missing_values(mz_values, intensities, method="interpolation"):
    """Fill missing intensity values using the requested strategy."""

    missing_indices = np.where(np.isnan(intensities))[0]
    if missing_indices.size == 0:
        return mz_values, intensities

    fixed_intensities = intensities.astype(float, copy=True)

    if method == "interpolation":
        valid_indices = np.where(~np.isnan(intensities))[0]
        if valid_indices.size == 0:
            fixed_intensities[missing_indices] = 0.0
        else:
            fixed_intensities[missing_indices] = np.interp(
                mz_values[missing_indices],
                mz_values[valid_indices],
                intensities[valid_indices],
            )
    elif method == "zero":
        fixed_intensities[missing_indices] = 0.0
    elif method == "mean":
        mean_intensity = np.nanmean(intensities)
        fixed_intensities[missing_indices] = 0.0 if np.isnan(mean_intensity) else mean_intensity
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown method for handling missing values: {method}")

    return mz_values, fixed_intensities

#  Robust Noise Level Estimation Function


def _positive_noise_stats(noise_values, fallback_values=None, empty_warning=None):
    """Return median and Gaussian-equivalent MAD for positive finite values."""
    values = np.asarray(noise_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if values.size == 0 and fallback_values is not None:
        if empty_warning:
            logger.warning(empty_warning)
        values = np.asarray(fallback_values, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]

    if values.size == 0:
        return 0.0, 0.0

    median_intensity = np.median(values)
    mad = np.median(np.abs(values - median_intensity))
    # 1.4826 converts MAD to a Gaussian-equivalent sigma. This is a
    # pragmatic thresholding surrogate, not a Poisson-specific noise model.
    robust_std = 1.4826 * mad
    return median_intensity, robust_std


def _resolve_noise_peak_regions(
        intensities,
        peak_indices=None,
        peak_height=None,
        peak_prominence=None,
        min_peak_width=1,
        max_peak_width=75,
        ):
    """Return peak centers and width bounds used to mask likely signal."""
    if peak_indices is not None:
        peaks = np.asarray(peak_indices, dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(intensities))]
        if peaks.size == 0:
            return peaks, np.array([], dtype=float), np.array([], dtype=float)

        try:
            _widths, _height, left_ips, right_ips = signal.peak_widths(
                intensities,
                peaks,
                rel_height=0.5,
            )
        except (ValueError, RuntimeError):
            left_ips = peaks.astype(float)
            right_ips = peaks.astype(float)
        return peaks, np.asarray(left_ips, dtype=float), np.asarray(right_ips, dtype=float)

    if peak_height is None or peak_prominence is None:
        pos = intensities[intensities > 0]
        if pos.size > 0:
            _med, _mad = _positive_noise_stats(pos)
        else:
            _med, _mad = 0.0, 0.0
        if peak_height is None:
            peak_height = _med
        if peak_prominence is None:
            peak_prominence = 3.0 * _mad if _mad > 0 else _med

    peaks, properties = signal.find_peaks(
        intensities,
        height=peak_height,
        prominence=peak_prominence,
        width=(min_peak_width, max_peak_width),
    )
    left_ips = np.asarray(properties.get("left_ips", peaks), dtype=float)
    right_ips = np.asarray(properties.get("right_ips", peaks), dtype=float)
    return np.asarray(peaks, dtype=int), left_ips, right_ips


def _noise_exclusion_mask(
        intensities,
        peak_indices=None,
        window=2,
        peak_height=None,
        peak_prominence=None,
        min_peak_width=1,
        max_peak_width=75,
        ):
    """Build a noise mask using measured peak widths plus an extra point margin."""
    mask = np.ones_like(intensities, dtype=bool)
    peak_indices, left_ips, right_ips = _resolve_noise_peak_regions(
        intensities,
        peak_indices=peak_indices,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
    )

    if peak_indices is None or len(peak_indices) == 0:
        return mask

    n = len(intensities)
    lefts = np.maximum(0, np.floor(left_ips).astype(int) - int(window))
    rights = np.minimum(n, np.ceil(right_ips).astype(int) + int(window) + 1)
    for left, right in zip(lefts, rights):
        if right > left:
            mask[left:right] = False
    return mask


def robust_noise_estimation(
        intensities,
        peak_indices=None,
        window=2,
        peak_height=None,
        peak_prominence=None,
        min_peak_width=1,
        max_peak_width=75
        ):
    """
    Robust noise estimation by excluding regions near detected peaks.

    Parameters
    ----------
    intensities : np.ndarray
        Denoised, baseline-corrected intensities.
    peak_indices : np.ndarray or None
        Indices of detected peaks. If None, function will detect peaks automatically.
    window : int
        Extra number of data points to exclude on each side of the detected
        peak width. The measured peak extent is always masked first.
    peak_height : float or None
        Minimum height for peak detection. If None, defaults to the median
        of positive intensities (data-adaptive).
    peak_prominence : float or None
        Minimum prominence for peak detection. If None, defaults to 3x the
        MAD of positive intensities (data-adaptive).

    Returns
    -------
    median_intensity : float
        Median intensity of noise region.
    robust_std : float
        Robust standard deviation (Gaussian-equivalent MAD) of noise region.
    """
    mask = _noise_exclusion_mask(
        intensities,
        peak_indices=peak_indices,
        window=window,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
    )
    noise_values = intensities[mask]
    return _positive_noise_stats(
        noise_values,
        fallback_values=intensities,
        empty_warning="No positive noise values found; falling back to full intensity array",
    )

# Robust Noise Level Estimation Function


def robust_noise_estimation_mz(
        mz_values,
        intensities,
        min_mz,
        max_mz
        ):
    """
    Estimate noise from a user-specified m/z baseline region.

    Parameters
    ----------
    mz_values : np.ndarray
        m/z axis.
    intensities : np.ndarray
        Corresponding intensity values.
    min_mz, max_mz : float
        m/z window that defines the baseline region.

    Returns
    -------
    median_intensity : float
        Median intensity of the baseline region.
    robust_std : float
        Robust standard deviation (MAD-scaled) of the baseline region.
    """
    baseline_region = intensities[(mz_values >= min_mz) & (mz_values <= max_mz)]
    median_intensity, robust_std = _positive_noise_stats(baseline_region)
    if median_intensity == 0.0 and robust_std == 0.0:
        logger.warning("No positive values in m/z baseline region [%s, %s]", min_mz, max_mz)
        return 0.0, 0.0
    return median_intensity, robust_std


def robust_noise_estimation_mz_dependent(
        mz_values,
        intensities,
        peak_indices=None,
        window=2,
        peak_height=None,
        peak_prominence=None,
        min_peak_width=1,
        max_peak_width=75,
        n_bins=20,
        min_points_per_bin=25,
        ):
    """Estimate local noise as piecewise-constant m/z bins interpolated over the spectrum.

    Returns
    -------
    median_profile : np.ndarray
        Per-point local median noise estimate.
    std_profile : np.ndarray
        Per-point local Gaussian-equivalent robust std estimate.
    """
    mz_values = np.asarray(mz_values, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if mz_values.size != intensities.size:
        raise ValueError("mz_values and intensities must have the same length")
    if mz_values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if min_points_per_bin < 1:
        raise ValueError("min_points_per_bin must be >= 1")

    global_median, global_std = robust_noise_estimation(
        intensities,
        peak_indices=peak_indices,
        window=window,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
    )

    if np.allclose(mz_values.min(), mz_values.max()):
        return (
            np.full_like(intensities, global_median, dtype=float),
            np.full_like(intensities, global_std, dtype=float),
        )

    mask = _noise_exclusion_mask(
        intensities,
        peak_indices=peak_indices,
        window=window,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
    )

    edges = np.linspace(mz_values.min(), mz_values.max(), n_bins + 1)
    centers = []
    medians = []
    stds = []

    for i in range(n_bins):
        if i == n_bins - 1:
            bin_mask = (mz_values >= edges[i]) & (mz_values <= edges[i + 1])
        else:
            bin_mask = (mz_values >= edges[i]) & (mz_values < edges[i + 1])
        noise_values = intensities[bin_mask & mask]
        median_i, std_i = _positive_noise_stats(noise_values)
        if np.count_nonzero(np.isfinite(noise_values) & (noise_values > 0)) < min_points_per_bin:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        medians.append(median_i)
        stds.append(std_i)

    if not centers:
        return (
            np.full_like(intensities, global_median, dtype=float),
            np.full_like(intensities, global_std, dtype=float),
        )

    centers = np.asarray(centers, dtype=float)
    medians = np.asarray(medians, dtype=float)
    stds = np.asarray(stds, dtype=float)

    if centers.size == 1:
        return (
            np.full_like(intensities, medians[0], dtype=float),
            np.full_like(intensities, stds[0], dtype=float),
        )

    median_profile = np.interp(mz_values, centers, medians, left=medians[0], right=medians[-1])
    std_profile = np.interp(mz_values, centers, stds, left=stds[0], right=stds[-1])
    return median_profile, std_profile


def _height_threshold_from_noise_model(
        mz_values,
        intensities,
        min_snr,
        noise_model="global",
        window=2,
        peak_height=None,
        peak_prominence=None,
        min_peak_width=1,
        max_peak_width=75,
        noise_bins=20,
        noise_min_points=25,
        ):
    """Return (median_noise, std_noise, height_threshold) for the requested noise model."""
    if noise_model == "global":
        median_noise, std_noise = robust_noise_estimation(
            intensities,
            peak_indices=None,
            window=window,
            peak_height=peak_height,
            peak_prominence=peak_prominence,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
        )
    elif noise_model == "mz_binned":
        median_noise, std_noise = robust_noise_estimation_mz_dependent(
            mz_values,
            intensities,
            peak_indices=None,
            window=window,
            peak_height=peak_height,
            peak_prominence=peak_prominence,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            n_bins=noise_bins,
            min_points_per_bin=noise_min_points,
        )
    else:
        raise ValueError("noise_model must be 'global' or 'mz_binned'")

    height_thresh = median_noise + min_snr * std_noise
    return median_noise, std_noise, height_thresh


# ---------------------------------------------------------------------------
# Shared width / area computation (trapezoid via cumulative sum)
# ---------------------------------------------------------------------------


def _compute_widths_and_areas(mz_values, intensities, peak_indices, rel_height):
    """Return (peak_widths_mz, areas) for the given peak indices.

    Uses cumulative-trapezoid integration, matching the logic formerly
    duplicated in ``detect_peaks_with_area`` and ``detect_peaks_cwt_with_area``.
    """
    if len(peak_indices) == 0:
        return np.array([]), np.array([])

    results_half = signal.peak_widths(intensities, peak_indices, rel_height=rel_height)
    left_ips = results_half[2]
    right_ips = results_half[3]
    left_bounds = np.clip(np.floor(left_ips).astype(int), 0, len(mz_values) - 1)
    right_bounds = np.clip(np.ceil(right_ips).astype(int), 0, len(mz_values) - 1)
    peak_widths = mz_values[right_bounds] - mz_values[left_bounds]

    dx = np.diff(mz_values)
    avg_y = 0.5 * (intensities[:-1] + intensities[1:])
    cum_area = np.empty(len(mz_values), dtype=float)
    cum_area[0] = 0.0
    np.cumsum(dx * avg_y, out=cum_area[1:])
    areas = cum_area[right_bounds] - cum_area[left_bounds]

    return peak_widths, areas


# Detect peak and measure width and area


def detect_peaks_with_area(
        mz_values,
        intensities,
        sample_name,
        group,
        min_intensity=1,
        min_snr=3,
        min_distance=2,
        window_size=10,
        peak_height=50,
        prominence=10,
        min_peak_width=1,
        max_peak_width=75,
        width_rel_height=0.5,
        noise_model="global",
        noise_bins=20,
        noise_min_points=25,
        verbose=False
        ):
    """
    Fast peak detection in ToF-SIMS or similar spectra, including peak area.
    
    Returns:
    --------
    peak_indices : np.ndarray
        Indices of detected peaks
    peak_properties : dict
        Contains: mz, intensities, widths, prominences, heights, areas
    """
    if not np.any(intensities > min_intensity):
        return _empty_peak_properties_df()

    # Validate m/z is monotonically increasing for correct area integration
    if len(mz_values) > 1 and not np.all(np.diff(mz_values) > 0):
        sort_idx = np.argsort(mz_values)
        mz_values = mz_values[sort_idx]
        intensities = intensities[sort_idx]

    # Clamp sub-threshold intensities to zero but keep the full x-grid so
    # that peak-width and area estimation see the true signal geometry.
    clamped = np.where(intensities > min_intensity, intensities, 0.0)

    if verbose:
        logger.info(f'min intensity: {np.nanmin(intensities)}, max intensity: {np.nanmax(intensities)}')

    median_noise, std_noise, height_thresh = _height_threshold_from_noise_model(
        mz_values,
        clamped,
        min_snr,
        noise_model=noise_model,
        window=window_size,
        peak_height=peak_height,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
        noise_bins=noise_bins,
        noise_min_points=noise_min_points,
    )
    if verbose:
        if np.ndim(median_noise) == 0:
            logger.info(f'median_noise: {median_noise}, std_noise: {std_noise}')
        else:
            logger.info(
                'median_noise range: [%s, %s], std_noise range: [%s, %s]',
                float(np.nanmin(median_noise)),
                float(np.nanmax(median_noise)),
                float(np.nanmin(std_noise)),
                float(np.nanmax(std_noise)),
            )
        if np.ndim(height_thresh) == 0:
            logger.info('height threshold: %s', height_thresh)
        else:
            logger.info(
                'height threshold range: [%s, %s]',
                float(np.nanmin(height_thresh)),
                float(np.nanmax(height_thresh)),
            )

    # Detect peaks on clamped signal
    peaks, props = signal.find_peaks(
        clamped,
        height=height_thresh,
        distance=min_distance,
        prominence=prominence,
        width=(min_peak_width, max_peak_width)
    )

    # Calculate FWHM bounds and areas on the *original* intensities so
    # that tails below min_intensity still contribute to width/area.
    peak_widths, areas = _compute_widths_and_areas(
        mz_values, intensities, peaks, width_rel_height
    )

    peak_properties = {
        'PeakCenter': mz_values[peaks],
        'PeakWidth': peak_widths,
        'Prominences': props.get('prominences', None),
        'Amplitude': props.get('peak_heights', None),
        'PeakArea': areas,
    }
    peak_properties = pd.DataFrame(peak_properties)
    peak_properties['SampleName'] = sample_name
    peak_properties['Group'] = group
    peak_properties['DetectedBy'] = 'local_max'
    peak_properties['Deconvoluted'] = False
    peak_properties['FitModel'] = 'none'
    peak_properties['AreaDefinition'] = 'raw_trapezoid'
    peak_properties['WidthDefinition'] = f'width_at_rel_height={width_rel_height}'
    peak_properties['IntegrationMethod'] = 'trapezoid'
    return peak_properties



def _baseline_simpson(mz, y, left_ip, right_ip):
    """
    Integrate y over [left_ip, right_ip] after subtracting
    a straight-line baseline through the two end-points.

    Returns 0.0 when the interval is degenerate (left_ip ≈ right_ip).
    """
    if right_ip <= left_ip:
        return 0.0

    sample_idx = np.arange(len(mz), dtype=float)
    x_left = np.interp(left_ip, sample_idx, mz)
    x_right = np.interp(right_ip, sample_idx, mz)
    if x_right <= x_left:
        return 0.0

    y_left = np.interp(left_ip, sample_idx, y)
    y_right = np.interp(right_ip, sample_idx, y)

    i0 = int(np.floor(left_ip)) + 1
    i1 = int(np.ceil(right_ip))
    x_seg = np.concatenate(([x_left], mz[i0:i1], [x_right]))
    y_seg = np.concatenate(([y_left], y[i0:i1], [y_right]))

    y_base = np.interp(x_seg, [x_left, x_right], [y_left, y_right])
    y_corr = np.clip(y_seg - y_base, 0, None)
    return integrate.simpson(y=y_corr, x=x_seg)


def detect_peaks_with_area_v2(
        mz, intens, sample_name, group,
        *,
        min_intensity=1, min_snr=3,
        min_distance=2, prominence=10,
        min_peak_width=1, max_peak_width=75,
        rel_height=0.5,
        noise_model="global",
        noise_bins=20,
        noise_min_points=25,
        noise_window=10,
        verbose=False):

    # 0. basic sanity check
    mask = intens > min_intensity
    if not np.any(mask):
        return _empty_peak_properties_df()

    mz, intens = mz[mask], intens[mask]

    # 1. noise estimate
    median_noise, std_noise, hthr = _height_threshold_from_noise_model(
        mz,
        intens,
        min_snr,
        noise_model=noise_model,
        window=noise_window,
        peak_height=None,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
        noise_bins=noise_bins,
        noise_min_points=noise_min_points,
    )
    if verbose:
        if np.ndim(hthr) == 0:
            logger.info(
                "median=%.2f, robust_std=%.2f, height threshold=%.2f",
                median_noise,
                std_noise,
                hthr,
            )
        else:
            logger.info(
                "median range=[%.2f, %.2f], robust_std range=[%.2f, %.2f], threshold range=[%.2f, %.2f]",
                float(np.nanmin(median_noise)),
                float(np.nanmax(median_noise)),
                float(np.nanmin(std_noise)),
                float(np.nanmax(std_noise)),
                float(np.nanmin(hthr)),
                float(np.nanmax(hthr)),
            )

    # 2. peak picking
    peaks, props = signal.find_peaks(intens,
                                     height=hthr,
                                     distance=min_distance,
                                     prominence=prominence,
                                     width=(min_peak_width, max_peak_width))

    if peaks.size == 0:
        return _empty_peak_properties_df()

    # 3. sub-sample widths (FWHM by default)
    widths, h_eval, left_ips, right_ips = signal.peak_widths(
                                            intens, peaks, rel_height=rel_height)

    # 4. area under each peak (baseline-corrected Simpson)
    areas = np.fromiter((_baseline_simpson(mz, intens, l, r)
                         for l, r in zip(left_ips, right_ips)),
                        dtype=float, count=peaks.size)

    # 5. width in m/z units (sub-sample)
    mz_interp = np.interp
    widths_mz = mz_interp(right_ips, np.arange(mz.size), mz) - \
                mz_interp(left_ips,  np.arange(mz.size), mz)

    df = pd.DataFrame({
        'PeakCenter'       : mz[peaks],
        'PeakWidth'        : widths_mz,
        'Prominences'      : props['prominences'],
        'Amplitude'        : props['peak_heights'],
        'PeakArea'         : areas,
        'SampleName'       : sample_name,
        'Group'            : group,
        'DetectedBy'       : 'find_peaks+widths',
        'Deconvoluted'     : False,
        'FitModel'         : 'none',
        'AreaDefinition'   : 'baseline_corrected_simpson',
        'WidthDefinition'  : f'width_at_rel_height={rel_height}',
        'IntegrationMethod': 'simpson',
    })
    return df


def detect_peaks_cwt_with_area(
        mz_values,
        intensities,
        sample_name,
        group,
        min_intensity=1,
        min_snr=3,
        min_distance=2,
        window_size=10,
        peak_height=50,
        prominence=10,
        min_peak_width=1,
        max_peak_width=75,
        width_rel_height=0.5,
        noise_model="global",
        noise_bins=20,
        noise_min_points=25,
        verbose=False
        ):
    """
    Peak detection using Continuous Wavelet Transform (CWT) for ToF-SIMS spectra.

    Returns:
    --------
    peak_properties : pd.DataFrame
        Contains: mz, intensities, widths (approx), amplitudes, areas
    """
    if not np.any(intensities > min_intensity):
        return _empty_peak_properties_df()

    # Clamp sub-threshold intensities but keep the full x-grid
    clamped = np.where(intensities > min_intensity, intensities, 0.0)

    if verbose:
        logger.info(f'min intensity: {np.nanmin(intensities)}, max intensity: {np.nanmax(intensities)}')

    median_noise, std_noise, height_thresh = _height_threshold_from_noise_model(
        mz_values,
        clamped,
        min_snr,
        noise_model=noise_model,
        window=window_size,
        peak_height=peak_height,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
        noise_bins=noise_bins,
        noise_min_points=noise_min_points,
    )

    if verbose:
        if np.ndim(median_noise) == 0:
            logger.info(f'median noise: {median_noise}, std noise: {std_noise}')
        else:
            logger.info(
                'median noise range: [%s, %s], std noise range: [%s, %s]',
                float(np.nanmin(median_noise)),
                float(np.nanmax(median_noise)),
                float(np.nanmin(std_noise)),
                float(np.nanmax(std_noise)),
            )
        if np.ndim(height_thresh) == 0:
            logger.info('height threshold: %s', height_thresh)
        else:
            logger.info(
                'height threshold range: [%s, %s]',
                float(np.nanmin(height_thresh)),
                float(np.nanmax(height_thresh)),
            )

    # Prepare widths array for CWT (must be integers)
    widths = np.arange(min_peak_width, max_peak_width + 1)

    # CWT peak detection on clamped signal
    cwt_peaks = signal.find_peaks_cwt(clamped, widths, min_snr=min_snr)

    # Only retain peaks above the estimated SNR threshold
    cwt_peaks = [
        idx for idx in cwt_peaks
        if clamped[idx] > (height_thresh[idx] if np.ndim(height_thresh) > 0 else height_thresh)
    ]

    # Estimate peak properties using original intensities
    peak_centers = mz_values[cwt_peaks]
    amplitudes = intensities[cwt_peaks]

    # Estimate peak widths and areas on original intensities
    cwt_peaks_arr = np.asarray(cwt_peaks, dtype=int)
    peak_widths, peak_areas = _compute_widths_and_areas(
        mz_values, intensities, cwt_peaks_arr, width_rel_height
    )

    # Build DataFrame
    peak_properties = pd.DataFrame({
        'PeakCenter': peak_centers,
        'PeakWidth': peak_widths,
        'Prominences': [np.nan] * len(cwt_peaks),
        'Amplitude': amplitudes,
        'PeakArea': peak_areas,
    })
    peak_properties['SampleName'] = sample_name
    peak_properties['Group'] = group
    peak_properties['DetectedBy'] = 'cwt'
    peak_properties['Deconvoluted'] = False
    peak_properties['FitModel'] = 'none'
    peak_properties['AreaDefinition'] = 'raw_trapezoid'
    peak_properties['WidthDefinition'] = f'width_at_rel_height={width_rel_height}'
    peak_properties['IntegrationMethod'] = 'trapezoid'

    return peak_properties
    
    
# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

_FWHM_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))   # ≈ 2.35482  (sigma → FWHM)


def _voigt_fwhm(sigma, gamma):
    """
    Standard pseudo-Voigt FWHM approximation (Olivero & Longbothum, 1977).

    Parameters
    ----------
    sigma : float  — Gaussian standard deviation
    gamma : float  — Lorentzian HWHM

    Returns
    -------
    float : Voigt FWHM
    """
    f_G = _FWHM_SIGMA * abs(sigma)   # Gaussian FWHM
    f_L = 2.0 * abs(gamma)           # Lorentzian FWHM
    return 0.5346 * f_L + np.sqrt(0.2166 * f_L**2 + f_G**2)


# ---------------------------------------------------------------------------
# Curve fittings

# Gaussian model for curve fitting


def gaussian(x, amp, cen, sigma):
    """Gaussian lineshape function.
    amp: Peak height at *cen*
    cen: Peak centre (mean)
    sigma: standard deviation sigma of the Gaussian

    returns:
    Gaussian function evaluated at x.
    """
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

# Lorentzian model for curve fitting


def lorentzian(x, A, x0, gamma):
    """Lorentzian lineshape function.
    A: area under the peak (scaling factor)
    x0: center
    gamma: half-width at half-maximum (HWHM)
    """
    return (A / np.pi) * (gamma / ((x - x0)**2 + gamma**2))

# Voigt model for curve fitting

def voigt(x, A, x0, sigma, gamma):
    """
    Voigt profile (convolution of Gaussian and Lorentzian).
    A: area under the peak (scaling factor)
    x0: center
    sigma: standard deviation of Gaussian component
    gamma: HWHM of Lorentzian component
    """
    # z = ((x - x0) + 1j*gamma) / (sigma * np.sqrt(2))
    # v = A * np.real(special.wofz(z)) / (sigma * np.sqrt(2*np.pi))
    return A * special.voigt_profile(x - x0, sigma, gamma)

# Two Gaussian model


def two_gaussians(x, amp1, cen1, wid1, amp2, cen2, wid2):
    return (amp1 * np.exp(-(x - cen1)**2 / (2*wid1**2)) +
            amp2 * np.exp(-(x - cen2)**2 / (2*wid2**2)))


def _median_positive_spacing(x):
    """Return the median positive x spacing or NaN when undefined."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    diffs = np.diff(x)
    pos = diffs[np.isfinite(diffs) & (diffs > 0)]
    if pos.size == 0:
        return np.nan
    return float(np.median(pos))


def _rss(y_true, y_pred):
    """Residual sum of squares with finite-value guarding."""
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return np.inf
    return float(np.sum(resid * resid))


def _bic_score(y_true, y_pred, n_params):
    """Bayesian information criterion for least-squares peak models."""
    n_obs = int(np.count_nonzero(np.isfinite(y_true) & np.isfinite(y_pred)))
    if n_obs <= max(int(n_params), 1):
        return np.inf
    rss = max(_rss(y_true, y_pred), np.finfo(float).tiny)
    return float(n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs))


def _fwhm_points(width_mz, x_fit):
    """Convert an m/z-domain FWHM to approximate sample points."""
    dx = _median_positive_spacing(x_fit)
    if not np.isfinite(dx) or dx <= 0:
        return np.nan
    return float(width_mz / dx)


def _component_widths_are_reasonable(sigmas, x_fit, min_peak_width, max_peak_width):
    """Check fitted Gaussian component widths against detector width limits."""
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.size == 0 or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        return False

    fwhm_pts = np.array(
        [_fwhm_points(abs(sigma) * _FWHM_SIGMA, x_fit) for sigma in sigmas],
        dtype=float,
    )
    if not np.all(np.isfinite(fwhm_pts)):
        return False

    min_width = max(float(min_peak_width), 1.0)
    max_width = max(float(max_peak_width), min_width)
    return bool(np.all((fwhm_pts >= min_width) & (fwhm_pts <= max_width)))


def _fit_single_gaussian_window(x_fit, y_fit, center_hint, min_peak_width, max_peak_width):
    """Fit a single Gaussian to the deconvolution window for BIC comparison."""
    dx = _median_positive_spacing(x_fit)
    sigma_floor = 1e-9
    sigma_guess = max(np.ptp(x_fit) / (6.0 * _FWHM_SIGMA), sigma_floor)
    sigma_lower = sigma_floor
    sigma_upper = max(np.ptp(x_fit), sigma_guess * 10.0, sigma_floor * 10.0)

    if np.isfinite(dx) and dx > 0:
        sigma_lower = max((max(float(min_peak_width), 1.0) * dx) / _FWHM_SIGMA, sigma_floor)
        sigma_upper = max((max(float(max_peak_width), float(min_peak_width)) * dx) / _FWHM_SIGMA, sigma_lower * 1.01)
        sigma_guess = min(max(sigma_guess, sigma_lower), sigma_upper)

    amp_guess = max(float(np.nanmax(y_fit)), 1e-9)
    p0 = [amp_guess, float(center_hint), sigma_guess]
    bounds = (
        [1e-9, float(np.min(x_fit)), sigma_lower],
        [np.inf, float(np.max(x_fit)), sigma_upper],
    )
    return optimize.curve_fit(
        gaussian,
        x_fit,
        y_fit,
        p0=p0,
        maxfev=20000,
        bounds=bounds,
    )[0]


def robust_peak_detection(
        mz_values,
        intensities,
        sample_name,
        group,
        method='Gaussian',
        min_intensity=1,
        min_snr=3,
        min_distance=2,
        window_size=10,
        peak_height=50,
        prominence=10,
        min_peak_width=1,
        max_peak_width=75,
        width_rel_height=0.5,
        distance_threshold=0.1,
        combined=False,
        use_cwt=False,
        noise_model="global",
        noise_bins=20,
        noise_min_points=25,
        deconvolution_min_bic_delta=10.0,
        deconvolution_overlap_factor=0.75,
        deconvolution_replace_singles=True,
        verbose=False
        ):
    """
    Fast peak detection in ToF-SIMS or similar spectra, including peak area.
    
    Returns:
    --------
    peak_indices : np.ndarray
        Indices of detected peaks
    peak_properties : dict
        Contains: mz, intensities, widths, prominences, heights, areas

    Notes
    -----
    Overlapping-peak deconvolution now requires both geometric overlap and a
    BIC improvement over a single-Gaussian window fit. Fitted component widths
    must also remain within the user-specified peak-width bounds.
    """
    if not np.any(intensities > min_intensity):
        return _empty_peak_properties_df()

    # Clamp sub-threshold intensities but keep the full x-grid
    clamped = np.where(intensities > min_intensity, intensities, 0.0)

    if verbose:
        logger.info(f'min intensity: {np.nanmin(intensities)}, max intensity: {np.nanmax(intensities)}')

    median_noise, std_noise, height_thresh = _height_threshold_from_noise_model(
        mz_values,
        clamped,
        min_snr,
        noise_model=noise_model,
        window=window_size,
        peak_height=peak_height,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
        noise_bins=noise_bins,
        noise_min_points=noise_min_points,
    )
    if verbose:
        if np.ndim(median_noise) == 0:
            logger.info(f'median noise: {median_noise}, std noise: {std_noise}')
        else:
            logger.info(
                'median noise range: [%s, %s], std noise range: [%s, %s]',
                float(np.nanmin(median_noise)),
                float(np.nanmax(median_noise)),
                float(np.nanmin(std_noise)),
                float(np.nanmax(std_noise)),
            )

    def _find_local_peaks():
        local_peaks, _props = signal.find_peaks(
            clamped,
            height=height_thresh,
            distance=min_distance,
            prominence=prominence,
            width=(min_peak_width, max_peak_width)
        )
        return np.asarray(local_peaks, dtype=int)

    def _find_cwt_peaks():
        cwt_peaks = signal.find_peaks_cwt(
            clamped,
            widths=np.arange(min_peak_width, max_peak_width+1),
            min_snr=min_snr
        )
        cwt_peaks = np.asarray(cwt_peaks, dtype=int)
        if cwt_peaks.size == 0:
            return cwt_peaks
        cwt_peaks = cwt_peaks[(cwt_peaks >= 0) & (cwt_peaks < len(clamped))]
        if np.ndim(height_thresh) == 0:
            keep = clamped[cwt_peaks] > height_thresh
        else:
            keep = clamped[cwt_peaks] > height_thresh[cwt_peaks]
        return np.unique(cwt_peaks[keep])

    peaks_local = np.array([], dtype=int)
    peaks_cwt = np.array([], dtype=int)

    # find peaks
    if combined:
        peaks_local = _find_local_peaks()
        peaks_cwt = _find_cwt_peaks()
        combined_indices = np.unique(np.concatenate([peaks_local, peaks_cwt]))
    elif use_cwt:
        peaks_cwt = _find_cwt_peaks()
        combined_indices = peaks_cwt.copy()
    else:
        peaks_local = _find_local_peaks()
        combined_indices = peaks_local.copy()

    peaks_local_set = set(peaks_local.tolist())
    peaks_cwt_set = set(peaks_cwt.tolist())
    
    if verbose:
        if np.ndim(height_thresh) == 0:
            logger.info(f"Height threshold: {height_thresh}")
        else:
            logger.info(
                "Height threshold range: [%s, %s]",
                float(np.nanmin(height_thresh)),
                float(np.nanmax(height_thresh)),
            )
        if combined:
            logger.info(f"Local peaks found: {len(peaks_local)} at indices {peaks_local}")
            logger.info(f"CWT peaks found: {len(peaks_cwt)} at indices {peaks_cwt}")
            logger.info(f"Combined peaks used for fitting: {len(combined_indices)} at indices {combined_indices}")
        elif not use_cwt:
            logger.info(f"Local peaks found: {len(peaks_local)} at indices {peaks_local}")
        else:
            logger.info(f"CWT peaks found: {len(peaks_cwt)} at indices {peaks_cwt}")

    peak_width_mz_map = {}
    if combined_indices.size > 0:
        try:
            _, _, left_ips, right_ips = signal.peak_widths(
                clamped,
                combined_indices,
                rel_height=width_rel_height,
            )
            width_mz = (
                np.interp(right_ips, np.arange(len(mz_values)), mz_values)
                - np.interp(left_ips, np.arange(len(mz_values)), mz_values)
            )
            peak_width_mz_map = {
                int(idx): float(width)
                for idx, width in zip(combined_indices, width_mz)
                if np.isfinite(width) and width > 0
            }
        except (ValueError, RuntimeError):
            peak_width_mz_map = {}

    single_fit_attempts = int(len(combined_indices))
    single_fit_failures = 0
    deconv_fit_attempts = 0
    deconv_fit_failures = 0

    all_peak_records = []
    all_deconv_records = []
    deconvolved_source_indices = set()

    def get_two_peak_guess(idx1, idx2, mz_values, intensities):
        return [
            intensities[idx1], mz_values[idx1], 0.2,
            intensities[idx2], mz_values[idx2], 0.2
        ]

    # -- Single Peak Fitting --
    for idx in combined_indices:
        left = max(0, idx - window_size)
        right = min(len(mz_values), idx + window_size + 1)
        x_fit = mz_values[left:right]
        y_fit = intensities[left:right]
        try:
            if method == 'Gaussian':
                popt, _ = optimize.curve_fit(
                    gaussian, 
                    x_fit, 
                    y_fit, 
                    p0=[intensities[idx], mz_values[idx], 0.0002],
                    maxfev=20000,
                    bounds=([1e-9, x_fit.min(), 0], [np.inf, x_fit.max(), np.inf])
                )
                amp = popt[0]
                cen = popt[1]
                fwhm = abs(popt[2]) * 2.35482  # Convert sigma to FWHM, sigma = FWHM / 2 * sqrt(2 * ln2)
                area = popt[0] * abs(popt[2]) * np.sqrt(2 * np.pi)

            elif method =='Lorentzian':
                popt, _ = optimize.curve_fit(
                    lorentzian, 
                    x_fit, 
                    y_fit, 
                    p0=[intensities[idx] * np.pi * 0.0001, mz_values[idx], 0.0001],
                    maxfev=20000,
                    bounds=([1e-9, x_fit.min(), 1e-9], [np.inf, x_fit.max(), np.inf]),
                    method='trf',
                    loss='soft_l1'
                )
                area = popt[0]
                amp = popt[0] / (np.pi * popt[2])  # Convert area to amplitude
                cen = popt[1]
                fwhm = popt[2] * 2  # Convert HWHM to FWHM            
            elif method == 'Voigt':
                popt, _ = optimize.curve_fit(
                    voigt, 
                    x_fit, 
                    y_fit, 
                    p0=[intensities[idx] * np.pi * 0.0001, mz_values[idx], 0.0002, 0.0001],
                    maxfev=20000,
                    bounds=([1e-9, x_fit.min(), 1e-9, 1e-9], [np.inf, x_fit.max(), np.inf, np.inf])
                )
                area = popt[0]  # A is the analytic area of the Voigt profile
                amp = popt[0] / (np.sqrt(2 * np.pi) * popt[2])  # Convert area to amplitude
                cen = popt[1]
                fwhm = _voigt_fwhm(popt[2], popt[3])  # Olivero & Longbothum approximation
            else:
                raise ValueError(f"Unknown fitting method: {method}")
            
            detected_by = []
            if idx in peaks_local_set:
                detected_by.append("local_max")
            if idx in peaks_cwt_set:
                detected_by.append("cwt")
            all_peak_records.append({
                "PeakCenter": cen,
                "PeakArea": area,
                "Prominences": np.nan,
                "Amplitude": amp,
                "PeakWidth": fwhm,
                "DetectedBy": "+".join(detected_by),
                "Deconvoluted": False,
                "FitModel": method,
                "AreaDefinition": "analytic_fit",
                "WidthDefinition": "FWHM",
                "IntegrationMethod": "analytic",
                "_SourceIndex": int(idx),
            })
        except Exception as e:
            single_fit_failures += 1
            if verbose:
                logger.debug(f"Single peak fit failed at idx={idx}: {e}")
            continue

    # -- Multi-Peak Deconvolution --
    for i in range(len(combined_indices) - 1):
        idx1 = combined_indices[i]
        idx2 = combined_indices[i + 1]
        # Only consider if both peaks are above threshold
        if clamped[idx1] < 2*height_thresh or clamped[idx2] < 2*height_thresh:
            continue
        spacing = abs(mz_values[idx2] - mz_values[idx1])
        width_candidates = [peak_width_mz_map.get(int(idx1)), peak_width_mz_map.get(int(idx2))]
        finite_widths = [width for width in width_candidates if width is not None and np.isfinite(width) and width > 0]
        adaptive_threshold = np.nan
        if finite_widths:
            adaptive_threshold = float(deconvolution_overlap_factor * np.mean(finite_widths))
        if distance_threshold is None:
            effective_threshold = adaptive_threshold
        elif np.isfinite(adaptive_threshold):
            effective_threshold = max(float(distance_threshold), adaptive_threshold)
        else:
            effective_threshold = float(distance_threshold)

        if np.isfinite(effective_threshold) and spacing < effective_threshold:
            left = max(0, idx1 - window_size)
            right = min(len(mz_values), idx2 + window_size + 1)
            x_fit = mz_values[left:right]
            y_fit = intensities[left:right]
            guess = get_two_peak_guess(idx1, idx2, mz_values, intensities)
            deconv_fit_attempts += 1
            try:
                popt, _ = optimize.curve_fit(two_gaussians, x_fit, y_fit, p0=guess, maxfev=20000,
                bounds=(
                    [1e-9, x_fit.min(), 0, 1e-9, x_fit.min(), 0],
                    [np.inf, x_fit.max(), np.inf, np.inf, x_fit.max(), np.inf])
                )
                amp1, cen1, wid1, amp2, cen2, wid2 = popt
                components = sorted(
                    [
                        (float(amp1), float(cen1), float(wid1)),
                        (float(amp2), float(cen2), float(wid2)),
                    ],
                    key=lambda item: item[1],
                )
                amp1, cen1, wid1 = components[0]
                amp2, cen2, wid2 = components[1]

                if not _component_widths_are_reasonable(
                    [wid1, wid2],
                    x_fit,
                    min_peak_width,
                    max_peak_width,
                ):
                    if verbose:
                        logger.debug(
                            "Rejected deconvolution at idx1=%s, idx2=%s due to implausible widths",
                            idx1,
                            idx2,
                        )
                    continue

                dominant_idx = idx1 if intensities[idx1] >= intensities[idx2] else idx2
                single_params = _fit_single_gaussian_window(
                    x_fit,
                    y_fit,
                    mz_values[dominant_idx],
                    min_peak_width,
                    max_peak_width,
                )
                bic_single = _bic_score(y_fit, gaussian(x_fit, *single_params), n_params=3)
                bic_double = _bic_score(
                    y_fit,
                    two_gaussians(x_fit, amp1, cen1, wid1, amp2, cen2, wid2),
                    n_params=6,
                )
                if not (bic_double < (bic_single - float(deconvolution_min_bic_delta))):
                    if verbose:
                        logger.debug(
                            "Rejected deconvolution at idx1=%s, idx2=%s because BIC improvement was insufficient "
                            "(single=%.3f, double=%.3f)",
                            idx1,
                            idx2,
                            bic_single,
                            bic_double,
                        )
                    continue

                area1 = amp1 * abs(wid1) * np.sqrt(2 * np.pi)
                area2 = amp2 * abs(wid2) * np.sqrt(2 * np.pi)
                deconvolved_source_indices.update({int(idx1), int(idx2)})
                all_deconv_records.extend([
                    {
                        "PeakCenter": cen1,
                        "PeakArea": area1,
                        "Prominences": np.nan,
                        "Amplitude": amp1,
                        "PeakWidth": abs(wid1) * _FWHM_SIGMA,  # was wid*sqrt(2π) — wrong
                        "DetectedBy": "deconv",
                        "Deconvoluted": True,
                        "FitModel": "two_gaussian",
                        "AreaDefinition": "analytic_fit",
                        "WidthDefinition": "FWHM",
                        "IntegrationMethod": "analytic",
                    },
                    {
                        "PeakCenter": cen2,
                        "PeakArea": area2,
                        "Prominences": np.nan,
                        "Amplitude": amp2,
                        "PeakWidth": abs(wid2) * _FWHM_SIGMA,  # was wid*sqrt(2π) — wrong
                        "DetectedBy": "deconv",
                        "Deconvoluted": True,
                        "FitModel": "two_gaussian",
                        "AreaDefinition": "analytic_fit",
                        "WidthDefinition": "FWHM",
                        "IntegrationMethod": "analytic",
                    }
                ])
            except Exception as e:
                deconv_fit_failures += 1
                if verbose:
                    logger.debug(f"Two-peak fit failed at idx1={idx1}, idx2={idx2}: {e}")
                continue

    if deconvolution_replace_singles and deconvolved_source_indices:
        all_peak_records = [
            record for record in all_peak_records
            if int(record.get("_SourceIndex", -1)) not in deconvolved_source_indices
        ]

    if not all_peak_records and not all_deconv_records:
        peak_properties = _empty_peak_properties_df()
    else:
        peak_properties = pd.DataFrame(all_peak_records + all_deconv_records)
        if "_SourceIndex" in peak_properties.columns:
            peak_properties = peak_properties.drop(columns=["_SourceIndex"])
        peak_properties = peak_properties.drop_duplicates(subset=["PeakCenter", "DetectedBy"], keep='last')

    peak_properties['SampleName']=sample_name
    peak_properties['Group']=group
    _log_peak_fit_failure_summary(
        sample_name,
        method,
        single_attempts=single_fit_attempts,
        single_failures=single_fit_failures,
        deconv_attempts=deconv_fit_attempts,
        deconv_failures=deconv_fit_failures,
    )
    return peak_properties


# Process all peaks to get peak properties


def collect_peak_properties_batch(
        files,
        mz_min=None,
        mz_max=None,
        baseline_method='airpls',
        noise_method='wavelet',
        missing_value_method='interpolation',
        normalization_target=1e8,
        method='Gaussian',
        min_intensity=1,
        min_snr=3,
        min_distance=5,
        window_size=10,
        peak_height=50,
        prominence=50,
        min_peak_width=1,
        max_peak_width=75,
        width_rel_height=0.5,
        distance_threshold=0.01,
        combined=False,
        noise_model="global",
        noise_bins=20,
        noise_min_points=25,
        deconvolution_min_bic_delta=10.0,
        deconvolution_overlap_factor=0.75,
        deconvolution_replace_singles=True,
        ):

    """
    Collect peak properties from a batch of ToF-SIMS files.

    Parameters
    ----------
    files : list of str
        List of file paths to process.
    mz_min, mz_max : float or None
        m/z window for data import (if supported).
    baseline_method : str
        Method for baseline correction.
    noise_method : str
        Noise filtering method.
    missing_value_method : str
        Method for handling missing values.
    normalization_target : float
        Target TIC normalization value.
    min_snr : int or float
        Minimum signal-to-noise ratio for peak detection.
    min_distance : int
        Minimum distance between peaks (in data points).
    prominence : int or float or None
        Minimum peak prominence for detection.
    width_rel_height : float
        Relative height for width calculation (e.g., 0.5 = FWHM).
    noise_model : {"global", "mz_binned"}
        Noise model used to derive peak thresholds.
    noise_bins : int
        Number of m/z bins for ``noise_model="mz_binned"``.
    noise_min_points : int
        Minimum positive noise points per bin before using local estimates.

    Returns
    -------
    peaks_df : pd.DataFrame
        DataFrame with all peak properties for all files.
    """

    all_peak_records = []

    for file_path in files:
        try:
            # Import data
            mz, intensity, sample_name, group = import_data(file_path, mz_min, mz_max)
            logger.info(f"Processing Sample: {sample_name}, Group: {group}")

            # Baseline correction
            intensity_base = baseline_correction(intensity, method=baseline_method)
            logger.debug("Baseline corrected intensities: available")

            # Noise filtering
            intensity_base_noise = noise_filtering(intensity_base, method=noise_method)
            logger.debug("Noise filtered intensities: available")

            # Handle missing values
            _, intensity_base_noise_mis = handle_missing_values(mz, intensity_base_noise, method=missing_value_method)
            logger.debug("Missing values handled: yes")

            # TIC normalization
            intensity_base_noise_mis_norm = tic_normalization(intensity_base_noise_mis, target_tic=normalization_target)
            logger.debug("TIC normalized intensities: available")


            # Peak detection
            if method is None:
                # print("Performing peak detection using local maxima!")
                peak_properties = detect_peaks_with_area(
                    mz_values=mz,
                    intensities=intensity_base_noise_mis_norm,
                    sample_name=sample_name,
                    group=group,
                    min_intensity=min_intensity,
                    min_snr=min_snr,
                    min_distance=min_distance,
                    window_size=window_size,
                    peak_height=peak_height,
                    prominence=prominence,
                    min_peak_width=min_peak_width,
                    max_peak_width=max_peak_width,
                    width_rel_height=width_rel_height,
                    noise_model=noise_model,
                    noise_bins=noise_bins,
                    noise_min_points=noise_min_points,
                )
            elif method == 'cwt':
                # print("Performing peak detection using Continuous Wavelet Transform (CWT)!")
                peak_properties = detect_peaks_cwt_with_area(
                    mz_values=mz,
                    intensities=intensity_base_noise_mis_norm,
                    sample_name=sample_name,
                    group=group,
                    min_intensity=min_intensity,
                    min_snr=min_snr,
                    min_distance=min_distance,
                    window_size=window_size,
                    peak_height=peak_height,
                    prominence=prominence,
                    min_peak_width=min_peak_width,
                    max_peak_width=max_peak_width,
                    width_rel_height=width_rel_height,
                    noise_model=noise_model,
                    noise_bins=noise_bins,
                    noise_min_points=noise_min_points,
                )
            else:
                # print(f"Performing peak detection using {method} method!")
                peak_properties = robust_peak_detection(
                    mz_values=mz,
                    intensities=intensity_base_noise_mis_norm,
                    sample_name=sample_name,
                    group=group,
                    method=method,
                    min_intensity=min_intensity,
                    min_snr=min_snr,
                    min_distance=min_distance,
                    window_size=window_size,
                    peak_height=peak_height,
                    prominence=prominence,
                    min_peak_width=min_peak_width,
                    max_peak_width=max_peak_width,
                    width_rel_height=width_rel_height,
                    distance_threshold=distance_threshold,
                    combined=combined,
                    noise_model=noise_model,
                    noise_bins=noise_bins,
                    noise_min_points=noise_min_points,
                    deconvolution_min_bic_delta=deconvolution_min_bic_delta,
                    deconvolution_overlap_factor=deconvolution_overlap_factor,
                    deconvolution_replace_singles=deconvolution_replace_singles,
                    )
            # pd.DataFrame(peak_properties).to_csv(file_path+'.txt', sep='\t', index=False)
            # Collect properties — vectorized DataFrame append
            if isinstance(peak_properties, pd.DataFrame) and len(peak_properties) > 0:
                cols_to_keep = [
                    'SampleName',
                    'Group',
                    'PeakCenter',
                    'PeakWidth',
                    'Prominences',
                    'PeakArea',
                    'Amplitude',
                    'DetectedBy',
                    'Deconvoluted',
                    'FitModel',
                    'AreaDefinition',
                    'WidthDefinition',
                    'IntegrationMethod',
                ]
                existing_cols = [c for c in cols_to_keep if c in peak_properties.columns]
                batch = peak_properties[existing_cols].copy()
                batch['SampleName'] = batch['SampleName'].astype(str)
                batch['Group'] = batch['Group'].astype(str)
                batch['DetectedBy'] = batch['DetectedBy'].astype(str)
                batch['Deconvoluted'] = batch['Deconvoluted'].astype(str)
                all_peak_records.extend(batch.to_dict('records'))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    peaks_df = pd.DataFrame(all_peak_records)
    
    return peaks_df



def _greedy_mz_bins(mz_array: np.ndarray, tolerance: float) -> np.ndarray:
    """Assign m/z values to bins using a greedy sorted-bin algorithm.

    Guarantees that ``max(mz) - min(mz) <= tolerance`` within every bin,
    unlike DBSCAN which only bounds pairwise neighbour distance and can
    chain peaks into arbitrarily wide clusters.

    Parameters
    ----------
    mz_array : np.ndarray
        1-D array of m/z values (one per peak row).
    tolerance : float
        Hard upper bound on the span of each bin.

    Returns
    -------
    np.ndarray
        Integer bin labels aligned with *mz_array*.
    """
    order = np.argsort(mz_array)
    labels = np.empty(len(mz_array), dtype=int)
    bin_id = 0
    bin_start = mz_array[order[0]]

    for idx in order:
        if mz_array[idx] - bin_start > tolerance:
            bin_id += 1
            bin_start = mz_array[idx]
        labels[idx] = bin_id

    return labels


def align_peaks(
    peaks_df,
    mz_tolerance=0.2,
    mz_rounding_precision=1,
    output="intensity",
):
    """Cluster peaks by m/z and return an aligned feature matrix.

    Uses a greedy sorted-bin algorithm that guarantees every aligned bin
    spans at most *mz_tolerance* in m/z.
    """

    if "PeakCenter" not in peaks_df.columns:
        raise ValueError("peaks_df must contain 'PeakCenter' column")

    if peaks_df.empty:
        return pd.DataFrame()

    working = peaks_df.copy()
    mz_values = working["PeakCenter"].to_numpy()

    # Greedy sorted bins — hard tolerance guarantee
    working["ClusterLabels"] = _greedy_mz_bins(mz_values, mz_tolerance)

    cluster_centers = (
        working.groupby("ClusterLabels", as_index=False)["PeakCenter"].mean()
        .rename(columns={"PeakCenter": "AlignedPeakCenter"})
    )
    working = working.merge(cluster_centers, on="ClusterLabels", how="left")
    working["AlignedPeakCenterRounded"] = (
        working["AlignedPeakCenter"].round(mz_rounding_precision).astype(str)
    )

    output_lower = output.lower()
    if output_lower == "intensity":
        value_col = "Amplitude"
    elif output_lower == "area":
        value_col = "PeakArea"
        if "AreaDefinition" not in working.columns:
            raise ValueError(
                "peaks_df must contain 'AreaDefinition' when output='area'"
            )
        area_definitions = sorted(
            {
                str(value)
                for value in working["AreaDefinition"].dropna().tolist()
                if str(value).strip()
            }
        )
        if not area_definitions:
            raise ValueError(
                "No non-empty AreaDefinition values found for area alignment"
            )
        if len(area_definitions) > 1:
            logger.warning(
                "Aligning PeakArea across mixed AreaDefinition values: %s",
                ", ".join(area_definitions),
            )
    else:
        raise ValueError("Output must be 'intensity' or 'area'")

    if value_col not in working.columns:
        raise ValueError(f"Column '{value_col}' required for output='{output}' is missing")

    # Detect duplicate peaks (same sample, same bin) and warn
    dup_mask = working.duplicated(
        subset=["SampleName", "AlignedPeakCenterRounded"], keep=False
    )
    n_dups = dup_mask.sum()
    if n_dups > 0:
        n_bins_affected = working.loc[
            dup_mask, "AlignedPeakCenterRounded"
        ].nunique()
        logger.warning(
            "%d peaks across %d bins have >1 peak per sample per bin; "
            "keeping the strongest peak in each conflict.",
            n_dups, n_bins_affected,
        )

    index_cols = ["SampleName"]
    if "Group" in working.columns:
        index_cols.append("Group")

    pivot = (
        working.pivot_table(
            index=index_cols,
            columns="AlignedPeakCenterRounded",
            values=value_col,
            aggfunc="max",
        )
        .fillna(0)
        .sort_index()
    )

    pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: float(x)), axis=1)
    return pivot


class PeakAlignIntensityArea:
    """
    Process normalized ToF-SIMS spectra from CSV files, detect peaks, align them across samples,
    and calculate both intensity and area tables for each aligned m/z value.

    Parameters
    ----------
    mz_tolerance : float, optional (default=0.2)
        Maximum distance (in m/z units) for clustering peaks across samples.
    mz_rounding_precision : int, optional (default=1)
        Number of decimal places for rounding aligned m/z values in output tables.
    min_intensity : float, optional (default=1)
        Minimum intensity threshold for considering data points.
    min_snr : float, optional (default=3)
        Minimum signal-to-noise ratio for peak detection.
    min_distance : int, optional (default=2)
        Minimum distance (in data points) between peaks.
    peak_height : float, optional (default=50)
        Minimum peak height for initial peak detection.
    prominence : float, optional (default=10)
        Minimum prominence for peak detection.
    min_peak_width : int, optional (default=1)
        Minimum peak width (in data points).
    max_peak_width : int, optional (default=75)
        Maximum peak width (in data points).
    width_rel_height : float, optional (default=0.5)
        Relative height for peak width calculation (0.5 = FWHM).
    noise_model : {"global", "mz_binned"}, optional (default="global")
        Noise model used for threshold estimation.
    noise_bins : int, optional (default=20)
        Number of m/z bins when using ``noise_model="mz_binned"``.
    noise_min_points : int, optional (default=25)
        Minimum positive noise points per bin for the local model.
    method : str or None, optional (default=None)
        Peak-detection / fitting method. None uses simple local-max detection
        (``detect_peaks_with_area_v2``), ``'cwt'`` uses CWT detection, and
        ``'Gaussian'`` / ``'Lorentzian'`` / ``'Voigt'`` use curve-fit detection
        via ``robust_peak_detection``.
    deconvolution_min_bic_delta : float, optional (default=10.0)
        Minimum BIC improvement required before accepting a two-Gaussian
        deconvolution over a single-peak fit.
    deconvolution_overlap_factor : float, optional (default=0.75)
        Scale factor applied to the mean measured peak width when deriving the
        adaptive deconvolution spacing gate.
    deconvolution_replace_singles : bool, optional (default=True)
        If True, replace overlapping single-peak fits with the accepted
        deconvoluted components in the output table.
    output_dir : str or None, optional
        Directory to save output CSV files. If None, files are not saved.
    verbose : bool, optional (default=False)
        If True, print progress information.

    Examples
    --------
    >>> from mioXpektron.detection import PeakAlignIntensityArea
    >>> import glob
    >>>
    >>> # Get all normalized spectra
    >>> csv_files = glob.glob('output_files/normalized_spectra/*.csv')
    >>>
    >>> # Create analyzer instance
    >>> analyzer = PeakAlignIntensityArea(
    ...     mz_tolerance=0.1,
    ...     min_snr=3,
    ...     output_dir='output_files/peak_analysis'
    ... )
    >>>
    >>> # Process with m/z cutoff
    >>> intensity_table, area_table, peaks_df = analyzer.run(
    ...     csv_files,
    ...     mz_min=50,
    ...     mz_max=500
    ... )
    >>>
    >>> print(f"Detected {len(peaks_df)} peaks across {len(csv_files)} samples")
    >>> print(f"Aligned to {intensity_table.shape[1]} unique m/z values")
    """

    def __init__(
        self,
        mz_tolerance=0.2,
        mz_rounding_precision=1,
        min_intensity=1,
        min_snr=3,
        min_distance=2,
        peak_height=50,
        prominence=10,
        min_peak_width=1,
        max_peak_width=75,
        width_rel_height=0.5,
        noise_model="global",
        noise_bins=20,
        noise_min_points=25,
        method=None,
        deconvolution_min_bic_delta=10.0,
        deconvolution_overlap_factor=0.75,
        deconvolution_replace_singles=True,
        output_dir=None,
        verbose=False,
        group_patterns=None,
        group_fn=None,
    ):
        """Initialize the PeakAlignIntensityArea analyzer with default parameters."""
        self.mz_tolerance = mz_tolerance
        self.mz_rounding_precision = mz_rounding_precision
        self.min_intensity = min_intensity
        self.min_snr = min_snr
        self.min_distance = min_distance
        self.peak_height = peak_height
        self.prominence = prominence
        self.min_peak_width = min_peak_width
        self.max_peak_width = max_peak_width
        self.width_rel_height = width_rel_height
        self.noise_model = noise_model
        self.noise_bins = noise_bins
        self.noise_min_points = noise_min_points
        self.method = method
        self.deconvolution_min_bic_delta = deconvolution_min_bic_delta
        self.deconvolution_overlap_factor = deconvolution_overlap_factor
        self.deconvolution_replace_singles = deconvolution_replace_singles
        self.output_dir = output_dir
        self.verbose = verbose
        self.group_patterns = group_patterns
        self.group_fn = group_fn

    def run(self, csv_files, mz_min=None, mz_max=None):
        """
        Process CSV files and perform peak detection, alignment, and quantification.

        Parameters
        ----------
        csv_files : list of str
            List of paths to normalized spectrum CSV files. Each CSV should have columns:
            'channel', 'mz', 'intensity'
        mz_min : float or None, optional
            Minimum m/z value to consider for peak detection. If None, use full range.
        mz_max : float or None, optional
            Maximum m/z value to consider for peak detection. If None, use full range.

        Returns
        -------
        intensity_table : pd.DataFrame
            DataFrame with samples as rows and aligned m/z values as columns,
            containing peak intensities (amplitudes). Missing peaks are filled with 0.
        area_table : pd.DataFrame
            DataFrame with samples as rows and aligned m/z values as columns,
            containing peak areas. Missing peaks are filled with 0.
        peaks_df : pd.DataFrame
            DataFrame containing all detected peaks with their properties before alignment.
        """
        all_peak_records = []

        for file_path in csv_files:
            try:
                # Extract sample name and group from filename
                sample_name = os.path.basename(file_path).replace('.csv', '')
                group = _resolve_group(
                    sample_name, self.group_patterns, self.group_fn
                )

                if self.verbose:
                    logger.info(f"Processing: {sample_name}")

                # Load CSV file
                df = pd.read_csv(file_path)

                # Check for required columns
                if 'mz' not in df.columns or 'intensity' not in df.columns:
                    logger.warning(f"Skipping {file_path} - missing 'mz' or 'intensity' columns")
                    continue

                mz = df['mz'].values
                intensity = df['intensity'].values

                # Apply m/z cutoff if specified
                if mz_min is not None or mz_max is not None:
                    mask = np.ones(len(mz), dtype=bool)
                    if mz_min is not None:
                        mask &= (mz >= mz_min)
                    if mz_max is not None:
                        mask &= (mz <= mz_max)
                    mz = mz[mask]
                    intensity = intensity[mask]

                    if len(mz) == 0:
                        if self.verbose:
                            logger.warning(f"  No data in m/z range [{mz_min}, {mz_max}]")
                        continue

                if self.verbose:
                    logger.info(f"  m/z range: [{mz.min():.4f}, {mz.max():.4f}]")

                # Detect peaks — dispatch on self.method
                if self.method is None:
                    peak_properties = detect_peaks_with_area_v2(
                        mz=mz,
                        intens=intensity,
                        sample_name=sample_name,
                        group=group,
                        min_intensity=self.min_intensity,
                        min_snr=self.min_snr,
                        min_distance=self.min_distance,
                        prominence=self.prominence,
                        min_peak_width=self.min_peak_width,
                        max_peak_width=self.max_peak_width,
                        rel_height=self.width_rel_height,
                        noise_model=self.noise_model,
                        noise_bins=self.noise_bins,
                        noise_min_points=self.noise_min_points,
                        verbose=self.verbose,
                    )
                elif self.method == 'cwt':
                    peak_properties = detect_peaks_cwt_with_area(
                        mz_values=mz,
                        intensities=intensity,
                        sample_name=sample_name,
                        group=group,
                        min_intensity=self.min_intensity,
                        min_snr=self.min_snr,
                        peak_height=self.peak_height,
                        prominence=self.prominence,
                        min_peak_width=self.min_peak_width,
                        max_peak_width=self.max_peak_width,
                        width_rel_height=self.width_rel_height,
                        noise_model=self.noise_model,
                        noise_bins=self.noise_bins,
                        noise_min_points=self.noise_min_points,
                        deconvolution_min_bic_delta=self.deconvolution_min_bic_delta,
                        deconvolution_overlap_factor=self.deconvolution_overlap_factor,
                        deconvolution_replace_singles=self.deconvolution_replace_singles,
                        verbose=self.verbose,
                    )
                else:
                    peak_properties = robust_peak_detection(
                        mz_values=mz,
                        intensities=intensity,
                        sample_name=sample_name,
                        group=group,
                        method=self.method,
                        min_intensity=self.min_intensity,
                        min_snr=self.min_snr,
                        min_distance=self.min_distance,
                        peak_height=self.peak_height,
                        prominence=self.prominence,
                        min_peak_width=self.min_peak_width,
                        max_peak_width=self.max_peak_width,
                        width_rel_height=self.width_rel_height,
                        noise_model=self.noise_model,
                        noise_bins=self.noise_bins,
                        noise_min_points=self.noise_min_points,
                        deconvolution_min_bic_delta=self.deconvolution_min_bic_delta,
                        deconvolution_overlap_factor=self.deconvolution_overlap_factor,
                        deconvolution_replace_singles=self.deconvolution_replace_singles,
                        verbose=self.verbose,
                    )

                if len(peak_properties) > 0:
                    all_peak_records.append(peak_properties)
                    if self.verbose:
                        logger.info(f"  Detected {len(peak_properties)} peaks")
                else:
                    if self.verbose:
                        logger.info("  No peaks detected")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}", exc_info=self.verbose)
                continue

        # Combine all peak records
        if len(all_peak_records) == 0:
            logger.warning("No peaks detected in any file!")
            empty_table = pd.DataFrame()
            return empty_table, empty_table, _empty_peak_properties_df()

        peaks_df = pd.concat(all_peak_records, ignore_index=True)

        if self.verbose:
            logger.info(f"Total peaks detected: {len(peaks_df)}")
            logger.info(f"Unique samples: {peaks_df['SampleName'].nunique()}")

        # Align peaks and create intensity table
        intensity_table = align_peaks(
            peaks_df,
            mz_tolerance=self.mz_tolerance,
            mz_rounding_precision=self.mz_rounding_precision,
            output="intensity"
        )

        # Align peaks and create area table
        area_table = align_peaks(
            peaks_df,
            mz_tolerance=self.mz_tolerance,
            mz_rounding_precision=self.mz_rounding_precision,
            output="area"
        )

        if self.verbose:
            logger.info(f"Aligned m/z values: {intensity_table.shape[1]}")
            logger.info(f"Intensity table shape: {intensity_table.shape}")
            logger.info(f"Area table shape: {area_table.shape}")

        # Save output files if directory specified
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

            # Save tables
            intensity_path = os.path.join(self.output_dir, 'peak_intensity_table.csv')
            area_path = os.path.join(self.output_dir, 'peak_area_table.csv')
            peaks_path = os.path.join(self.output_dir, 'all_detected_peaks.csv')

            intensity_table.to_csv(intensity_path)
            area_table.to_csv(area_path)
            peaks_df.to_csv(peaks_path, index=False)

            if self.verbose:
                logger.info(f"Saved outputs to {self.output_dir}:")
                logger.info(f"  - {intensity_path}")
                logger.info(f"  - {area_path}")
                logger.info(f"  - {peaks_path}")

        return intensity_table, area_table, peaks_df
