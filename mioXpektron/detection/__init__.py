"""Peak detection utilities for the Xpektron toolkit."""

from .check_overlapping_peaks import check_overlapping_peaks
from .check_overlapping_peaks2 import check_overlapping_peaks2
from .detection import (
    align_peaks,
    collect_peak_properties_batch,
    detect_peaks_cwt_with_area,
    detect_peaks_with_area,
    detect_peaks_with_area_v2,
    PeakAlignIntensityArea,
    robust_noise_estimation,
    robust_noise_estimation_mz,
    robust_peak_detection,
)

__all__ = [
    "align_peaks",
    "check_overlapping_peaks",
    "check_overlapping_peaks2",
    "collect_peak_properties_batch",
    "detect_peaks_cwt_with_area",
    "detect_peaks_with_area",
    "detect_peaks_with_area_v2",
    "PeakAlignIntensityArea",
    "robust_noise_estimation",
    "robust_noise_estimation_mz",
    "robust_peak_detection",
]
