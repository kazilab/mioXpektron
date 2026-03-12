# Data import function
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
from scipy import signal, optimize, special, integrate
from pybaselines import Baseline
from sklearn.cluster import DBSCAN

from ..baseline.baseline_base import baseline_correction
from ..denoise.denoise_main import noise_filtering
from ..normalization.normalization import tic_normalization
from ..utils.file_management import import_data


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

def robust_noise_estimation(
        intensities,
        peak_indices=None,
        window=2,
        peak_height=50,
        peak_prominence=10,
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
        Number of data points to exclude on each side of a peak.
    peak_height : float or None
        Minimum height for peak detection (optional).
    peak_prominence : float or None
        Minimum prominence for peak detection (optional).

    Returns
    -------
    median_intensity : float
        Median intensity of noise region.
    robust_std : float
        Robust standard deviation (MAD scaled) of noise region.
    """
    # If peak_indices not given, detect peaks automatically
    if peak_indices is None:
        peaks, _ = signal.find_peaks(intensities, height=peak_height, prominence=peak_prominence, width=(min_peak_width, max_peak_width))
        peak_indices = peaks

    mask = np.ones_like(intensities, dtype=bool)
    if peak_indices is not None and len(peak_indices) > 0:
        # Vectorized: build all exclusion ranges at once
        peak_indices = np.asarray(peak_indices)
        n = len(intensities)
        lefts = np.maximum(0, peak_indices - window)
        rights = np.minimum(n, peak_indices + window + 1)
        # Create index array of all positions to exclude
        exclude = np.concatenate([np.arange(l, r) for l, r in zip(lefts, rights)])
        mask[exclude] = False

    noise_values = intensities[mask]
    noise_values = noise_values[noise_values>0]
    median_intensity = np.median(noise_values)
    mad = np.median(np.abs(noise_values - median_intensity))
    robust_std = 1.4826 * mad

    return median_intensity, robust_std

# Robust Noise Level Estimation Function


def robust_noise_estimation_mz(
        mz_values,
        intensities,
        min_mz,
        max_mz
        ):
    # Select baseline region based on provided min/max mz
    baseline_region = intensities[(mz_values >= min_mz) & (mz_values <= max_mz)]
    baseline_region = baseline_region[baseline_region>0]
    median_intensity = np.median(baseline_region)
    mad = np.median(np.abs(baseline_region - median_intensity))
    robust_std = mad * 1.4826
    
    return median_intensity, robust_std


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
    nonzero_mask = intensities > min_intensity
    mz_values = mz_values[nonzero_mask]
    intensities = intensities[nonzero_mask]

    # Validate m/z is monotonically increasing for correct area integration
    if len(mz_values) > 1 and not np.all(np.diff(mz_values) > 0):
        sort_idx = np.argsort(mz_values)
        mz_values = mz_values[sort_idx]
        intensities = intensities[sort_idx]

    min_int = np.nanmin(intensities)
    max_int = np.nanmax(intensities)

    if verbose:
        logger.info(f'min intensity: {min_int}, max intensity: {max_int}')

    median_noise, std_noise = robust_noise_estimation(
        intensities,
        peak_indices=None,
        window=window_size,
        peak_height=peak_height,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width)
    if verbose:
        logger.info(f'median_noise: {median_noise}, std_noise: {std_noise}')

    # Estimate set noise height threshold
    height_thresh = median_noise + min_snr * std_noise
    if verbose:
        logger.info('height threshold: %s', height_thresh)

    # Detect peaks
    peaks, props = signal.find_peaks(
        intensities,
        height=height_thresh,
        distance=min_distance,
        prominence=prominence,
        width=(min_peak_width, max_peak_width)
    )

    # Calculate FWHM bounds
    if len(peaks) > 0:
        results_half = signal.peak_widths(intensities, peaks, rel_height=width_rel_height)
        left_ips = results_half[2]
        right_ips = results_half[3]
        # Get integer bounds for integration
        left_bounds = np.floor(left_ips).astype(int)
        right_bounds = np.ceil(right_ips).astype(int)
        # Clip to valid range
        left_bounds = np.clip(left_bounds, 0, len(mz_values) - 1)
        right_bounds = np.clip(right_bounds, 0, len(mz_values) - 1)
        peak_widths = mz_values[right_bounds] - mz_values[left_bounds]
        # Calculate area for each peak using cumulative trapezoid (vectorized)
        dx = np.diff(mz_values)
        avg_y = 0.5 * (intensities[:-1] + intensities[1:])
        cum_area = np.empty(len(mz_values), dtype=float)
        cum_area[0] = 0.0
        np.cumsum(dx * avg_y, out=cum_area[1:])
        areas = cum_area[right_bounds] - cum_area[left_bounds]
    else:
        peak_widths = np.array([])
        areas = np.array([])

    peak_properties = {
        'PeakCenter': mz_values[peaks],
        'PeakWidth': peak_widths,
        'Prominences': props.get('prominences', None),
        'Amplitude': props.get('peak_heights', None),
        'PeakArea': areas,
    }
    peak_properties = pd.DataFrame(peak_properties)
    peak_properties['SampleName']=sample_name
    peak_properties['Group']=group
    peak_properties['DetectedBy']='local_max'
    peak_properties['Deconvoluted']=False
    return peak_properties



def _baseline_simpson(mz, y, left_ip, right_ip):
    """
    Integrate y over [left_ip, right_ip] after subtracting
    a straight-line baseline through the two end-points.
    """
    # slice that fully covers the floating interval
    i0, i1 = int(np.floor(left_ip)), int(np.ceil(right_ip)) + 1
    x_seg, y_seg = mz[i0:i1], y[i0:i1]

    # linear baseline
    y_base = np.interp(x_seg,
                       [mz[int(np.floor(left_ip))], mz[int(np.floor(right_ip))]],
                       [y[int(np.floor(left_ip))],  y[int(np.floor(right_ip))]])
    y_corr = np.clip(y_seg - y_base, 0, None)          # avoid negatives
    return integrate.simpson(y_corr, x_seg)            # higher-order rule


def detect_peaks_with_area_v2(
        mz, intens, sample_name, group,
        *,
        min_intensity=1, min_snr=3,
        min_distance=2, prominence=10,
        min_peak_width=1, max_peak_width=75,
        rel_height=0.5, verbose=False):

    # 0. basic sanity check
    mask = intens > min_intensity
    if not np.any(mask):
        return pd.DataFrame(columns=['PeakCenter','PeakWidth','Prominences',
                                     'Amplitude','PeakArea','SampleName',
                                     'Group','DetectedBy','Deconvoluted'])

    mz, intens = mz[mask], intens[mask]

    # 1. robust noise estimate  (median + k·MAD)
    med  = np.median(intens)
    mad  = 1.4826 * np.median(np.abs(intens - med))   # Gaussian-equivalent σ
    hthr = med + min_snr * mad
    if verbose:
        logger.info(f"median={med:.2f}, MAD={mad:.2f}, height threshold={hthr:.2f}")

    # 2. peak picking
    peaks, props = signal.find_peaks(intens,
                                     height=hthr,
                                     distance=min_distance,
                                     prominence=prominence,
                                     width=(min_peak_width, max_peak_width))

    if peaks.size == 0:
        return pd.DataFrame(columns=['PeakCenter','PeakWidth','Prominences',
                                     'Amplitude','PeakArea','SampleName',
                                     'Group','DetectedBy','Deconvoluted'])

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
        'PeakCenter' : mz[peaks],
        'PeakWidth'  : widths_mz,
        'Prominences': props['prominences'],
        'Amplitude'  : props['peak_heights'],
        'PeakArea'   : areas,
        'SampleName' : sample_name,
        'Group'      : group,
        'DetectedBy' : 'find_peaks+widths',
        'Deconvoluted': False,
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
        verbose=False
        ):
    """
    Peak detection using Continuous Wavelet Transform (CWT) for ToF-SIMS spectra.

    Returns:
    --------
    peak_properties : pd.DataFrame
        Contains: mz, intensities, widths (approx), amplitudes, areas
    """
    nonzero_mask = intensities > min_intensity
    mz_values = mz_values[nonzero_mask]
    intensities = intensities[nonzero_mask]
    min_int = np.nanmin(intensities)
    max_int = np.nanmax(intensities)
    
    if verbose:
        logger.info(f'min intensity: {min_int}, max intensity: {max_int}')

    median_noise, std_noise = robust_noise_estimation(
        intensities,
        peak_indices=None,
        window=window_size,
        peak_height=peak_height,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width)

    if verbose:
        logger.info(f'median noise: {median_noise}, std noise: {std_noise}')

    # Estimate signal threshold
    height_thresh = median_noise + min_snr * std_noise

    if verbose:
        logger.info('height threshold: %s', height_thresh)

    # Prepare widths array for CWT (must be integers)
    widths = np.arange(min_peak_width, max_peak_width + 1)

    # CWT peak detection
    cwt_peaks = signal.find_peaks_cwt(intensities, widths, min_snr=min_snr)

    # Only retain peaks above the estimated SNR threshold
    cwt_peaks = [idx for idx in cwt_peaks if intensities[idx] > height_thresh]

    # Estimate peak properties
    peak_centers = mz_values[cwt_peaks]
    amplitudes = intensities[cwt_peaks]

    # Estimate peak widths using scipy.signal.peak_widths (vectorized)
    cwt_peaks_arr = np.asarray(cwt_peaks, dtype=int)
    if len(cwt_peaks_arr) > 0:
        widths_result = signal.peak_widths(intensities, cwt_peaks_arr, rel_height=width_rel_height)
        left_ips = widths_result[2]
        right_ips = widths_result[3]
        left_bounds = np.clip(np.floor(left_ips).astype(int), 0, len(mz_values) - 1)
        right_bounds = np.clip(np.ceil(right_ips).astype(int), 0, len(mz_values) - 1)
        peak_widths = mz_values[right_bounds] - mz_values[left_bounds]
        # Vectorized area computation via cumulative trapezoid
        dx = np.diff(mz_values)
        avg_y = 0.5 * (intensities[:-1] + intensities[1:])
        cum_area = np.empty(len(mz_values), dtype=float)
        cum_area[0] = 0.0
        np.cumsum(dx * avg_y, out=cum_area[1:])
        peak_areas = cum_area[right_bounds] - cum_area[left_bounds]
    else:
        peak_widths = np.array([])
        peak_areas = np.array([])

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

    return peak_properties
    
    
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
        use_cwt=False,
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
    nonzero_mask = intensities > min_intensity
    mz_values = mz_values[nonzero_mask]
    intensities = intensities[nonzero_mask]
    min_int = np.nanmin(intensities)
    max_int = np.nanmax(intensities)
    
    if verbose:
        logger.info(f'min intensity: {min_int}, max intensity: {max_int}')

    median_noise, std_noise = robust_noise_estimation(
        intensities,
        peak_indices=None,
        window=window_size,
        peak_height=peak_height,
        peak_prominence=prominence,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width)
    if verbose:
        logger.info(f'median noise: {median_noise}, std noise: {std_noise}')
    
    # Estimate set noise height threshold
    height_thresh = median_noise + min_snr * std_noise




    
    # find peaks
    if use_cwt:
        # Detect peaks - CWT-based peak detection
        peaks_cwt = signal.find_peaks_cwt(
            intensities,
            widths=np.arange(min_peak_width, max_peak_width+1),
            min_snr=min_snr
        )

        combined_indices = peaks_cwt.copy()
    else:
        # Detect peaks - Local maxima detection (classic method)
        peaks_local, props = signal.find_peaks(
            intensities,
            height=height_thresh,
            distance=min_distance,
            prominence=prominence,
            width=(min_peak_width, max_peak_width)
        )
        combined_indices = peaks_local.copy() 
    
    if verbose:
        logger.info(f"Height threshold: {height_thresh}")
        if not use_cwt:
            logger.info(f"Local peaks found: {len(peaks_local)} at indices {peaks_local}")
        else:
            logger.info(f"CWT peaks found: {len(peaks_cwt)} at indices {peaks_cwt}")

    all_peak_records = []
    all_deconv_records = []

    def get_two_peak_guess(idx1, idx2, mz_values, intensities):
        return [
            intensities[idx1], mz_values[idx1], 0.2,
            intensities[idx2], mz_values[idx2], 0.2
        ]

    # -- Single Peak Fitting --
    for idx in combined_indices:
        left = max(0, idx - window_size)
        right = min(len(mz_values)+1, idx + window_size + 1)
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
                area = popt[0] # Area = A * sqrt(2*pi) * sigma
                amp = popt[0] / (np.sqrt(2 * np.pi) * popt[2])  # Convert area to amplitude
                cen = popt[1]
                fwhm = popt[2]* np.sqrt(2 * np.pi)  # Convert sigma to FWHM
            else:
                raise ValueError(f"Unknown fitting method: {method}")
            
            detected_by = []
            # Remove undefined 'combined' variable usage
            # if combined:
            #     if idx in peaks_local:
            #         detected_by.append("local_max")
            if 'peaks_local' in locals() and idx in peaks_local:
                detected_by.append("local_max")
            if 'peaks_cwt' in locals() and idx in peaks_cwt:
                detected_by.append("cwt")
            all_peak_records.append({
                "PeakCenter": cen,
                "PeakArea": area,
                "Amplitude": amp,
                "PeakWidth": fwhm,
                "DetectedBy": "+".join(detected_by),
                "Deconvoluted": False
            })
        except Exception as e:
            if verbose:
                logger.debug(f"Single peak fit failed at idx={idx}: {e}")
            continue

    # -- Multi-Peak Deconvolution --
    for i in range(len(combined_indices) - 1):
        idx1 = combined_indices[i]
        idx2 = combined_indices[i + 1]
        # Only consider if both peaks are above threshold
        if intensities[idx1] < 2*height_thresh or intensities[idx2] < 2*height_thresh:
            continue
        if abs(mz_values[idx2] - mz_values[idx1]) < distance_threshold:
            left = max(0, idx1 - window_size)
            right = min(len(mz_values)+1, idx2 + window_size)
            x_fit = mz_values[left:right]
            y_fit = intensities[left:right]
            guess = get_two_peak_guess(idx1, idx2, mz_values, intensities)
            try:
                popt, _ = optimize.curve_fit(two_gaussians, x_fit, y_fit, p0=guess, maxfev=20000,
                bounds=(
                    [1e-9, x_fit.min(), 0, 1e-9, x_fit.min(), 0],
                    [np.inf, x_fit.max(), np.inf, np.inf, x_fit.max(), np.inf])
                )
                amp1, cen1, wid1, amp2, cen2, wid2 = popt
                area1 = amp1 * abs(wid1) * np.sqrt(2 * np.pi)
                area2 = amp2 * abs(wid2) * np.sqrt(2 * np.pi)
                all_deconv_records.extend([
                    {
                        "PeakCenter": cen1,
                        "PeakArea": area1,
                        "Amplitude": amp1,
                        "PeakWidth": abs(wid1) * np.sqrt(2 * np.pi),
                        "DetectedBy": "deconv",
                        "Deconvoluted": True
                    },
                    {
                        "PeakCenter": cen2,
                        "PeakArea": area2,
                        "Amplitude": amp2,
                        "PeakWidth": abs(wid2)  * np.sqrt(2 * np.pi),
                        "DetectedBy": "deconv",
                        "Deconvoluted": True
                    }
                ])
            except Exception as e:
                if verbose:
                    logger.debug(f"Two-peak fit failed at idx1={idx1}, idx2={idx2}: {e}")
                continue

    peak_properties = pd.DataFrame(all_peak_records + all_deconv_records)
    peak_properties = peak_properties.drop_duplicates(subset=["PeakCenter", "DetectedBy"], keep='last')
    peak_properties['SampleName']=sample_name
    peak_properties['Group']=group
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
        combined=False
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
                    min_intensity=1,
                    min_snr=min_snr,
                    min_distance=min_distance,
                    window_size=window_size,
                    peak_height=peak_height,
                    prominence=prominence,
                    min_peak_width=min_peak_width,
                    max_peak_width=max_peak_width,
                    width_rel_height=width_rel_height
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
                    width_rel_height=width_rel_height)
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
                    combined=combined
                    )
            # pd.DataFrame(peak_properties).to_csv(file_path+'.txt', sep='\t', index=False)
            # Collect properties — vectorized DataFrame append
            if isinstance(peak_properties, pd.DataFrame) and len(peak_properties) > 0:
                cols_to_keep = ['SampleName', 'Group', 'PeakCenter', 'PeakWidth',
                                'PeakArea', 'Amplitude', 'DetectedBy', 'Deconvoluted']
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



def align_peaks(
    peaks_df,
    mz_tolerance=0.2,
    mz_rounding_precision=1,
    output="intensity",
):
    """Cluster peaks by m/z and return an aligned feature matrix."""

    if "PeakCenter" not in peaks_df.columns:
        raise ValueError("peaks_df must contain 'PeakCenter' column")

    working = peaks_df.copy()
    mz_values = working[["PeakCenter"]].to_numpy().reshape(-1, 1)
    cluster_labels = DBSCAN(eps=mz_tolerance, min_samples=1).fit_predict(mz_values)
    working["ClusterLabels"] = cluster_labels

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
    else:
        raise ValueError("Output must be 'intensity' or 'area'")

    if value_col not in working.columns:
        raise ValueError(f"Column '{value_col}' required for output='{output}' is missing")

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
        output_dir=None,
        verbose=False
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
        self.output_dir = output_dir
        self.verbose = verbose

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
        import os

        all_peak_records = []

        for file_path in csv_files:
            try:
                # Extract sample name and group from filename
                sample_name = os.path.basename(file_path).replace('.csv', '')
                # Try to extract group from filename pattern (e.g., 'breast_CT-14b_2' -> 'CT')
                parts = sample_name.split('_')
                group = parts[1].split('-')[0] if len(parts) > 1 and '-' in parts[1] else 'Unknown'

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

                # Detect peaks using the v2 function (baseline-corrected Simpson integration)
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
                    verbose=self.verbose
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
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df

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
            import os
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
