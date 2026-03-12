"""High-level utilities for parallel peak extraction and alignment."""

import tqdm
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..detection.detection import (
    detect_peaks_with_area,
    detect_peaks_cwt_with_area,
    robust_peak_detection,
    align_peaks as _align_peaks,
    collect_peak_properties_batch as _collect_peaks_detection,
)
from ..normalization.preprocessing import data_preprocessing


def _process_one_file(
    file_path,
    mz_min,
    mz_max,
    normalization_target,
    method,
    min_intensity,
    min_snr,
    min_distance,
    window_size,
    peak_height,
    prominence,
    min_peak_width,
    max_peak_width,
    width_rel_height,
    distance_threshold,
):
    """Return a list of peak records extracted from ``file_path``."""
    try:
        sample_name, group, mz_vals, norm_int = data_preprocessing(
            file_path=file_path,
            mz_min=mz_min,
            mz_max=mz_max,
            normalization_target=normalization_target,
            verbose=False,
            return_all=False,
        )

        if method is None:
            peak_props = detect_peaks_with_area(
                mz_values=mz_vals,
                intensities=norm_int,
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
            )
        elif method == "cwt":
            peak_props = detect_peaks_cwt_with_area(
                mz_values=mz_vals,
                intensities=norm_int,
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
            )
        else:
            peak_props = robust_peak_detection(
                mz_values=mz_vals,
                intensities=norm_int,
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
            )

        # Vectorized: convert DataFrame to records directly
        if isinstance(peak_props, pd.DataFrame) and len(peak_props) > 0:
            cols_to_keep = ['SampleName', 'Group', 'PeakCenter', 'PeakWidth',
                            'PeakArea', 'Amplitude', 'DetectedBy', 'Deconvoluted']
            existing_cols = [c for c in cols_to_keep if c in peak_props.columns]
            batch = peak_props[existing_cols].copy()
            batch['SampleName'] = batch['SampleName'].astype(str)
            batch['Group'] = batch['Group'].astype(str)
            batch['DetectedBy'] = batch['DetectedBy'].astype(str)
            batch['Deconvoluted'] = batch['Deconvoluted'].astype(str)
            return batch.to_dict('records')
        return []
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"{file_path}: {exc}") from exc


def batch_processing(
    files,
    *,
    max_workers=None,
    mz_min=None,
    mz_max=None,
    normalization_target=1e8,
    method=None,
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
    mz_tolerance=0.2,
    mz_rounding_precision=1,
):
    """Parallel batch peak detection.

    Returns ``(peaks_df, intensity_df, area_df)`` analogous to the legacy
    implementation.
    """
    worker = partial(
        _process_one_file,
        mz_min=mz_min,
        mz_max=mz_max,
        normalization_target=normalization_target,
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
    )

    all_peak_records = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, fp): fp for fp in files}
        for fut in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing ToF-SIMS files",
        ):
            all_peak_records.extend(fut.result())

    if not all_peak_records:
        raise ValueError("No peaks detected in any file. Check parameters or data quality.")

    peaks_df = pd.DataFrame(all_peak_records)
    intensity_df = _align_peaks(
        peaks_df,
        mz_tolerance=mz_tolerance,
        mz_rounding_precision=mz_rounding_precision,
        output="intensity",
    )
    area_df = _align_peaks(
        peaks_df,
        mz_tolerance=mz_tolerance,
        mz_rounding_precision=mz_rounding_precision,
        output="area",
    )
    return peaks_df, intensity_df, area_df


# Backwards-compatible re-exports -----------------------------------------------------
align_peaks = _align_peaks
collect_peak_properties_batch = _collect_peaks_detection
