"""Public package interface for the Xpektron toolkit."""

__version__ = "0.0.2"

# Core imports (always work)
from . import recalibrate, utils

# Optional imports (may fail if dependencies missing)
try:
    from . import baseline
    from .baseline import (
        AggregateParams,
        BaselineBatchCorrector,
        BaselineMethodEvaluator,
        FlatParams,
        ScanForFlatRegion,
        baseline_correction,
        baseline_method_names,
        small_param_grid_preset,
    )
except ImportError:
    baseline = None

try:
    from . import denoise
    from .denoise import (
        BatchDenoising,
        DenoisingMethods,
        batch_denoise,
        compare_denoising_methods,
        compare_methods_in_windows,
        decode_method_label,
        noise_filtering,
        plot_pareto_delta_snr_vs_height,
        rank_method,
        select_methods,
    )
except ImportError:
    denoise = None

try:
    from . import detection
    from .detection import (
        align_peaks,
        check_overlapping_peaks,
        check_overlapping_peaks2,
        collect_peak_properties_batch,
        detect_peaks_cwt_with_area,
        detect_peaks_with_area,
        detect_peaks_with_area_v2,
        robust_noise_estimation,
        robust_noise_estimation_mz,
        robust_peak_detection,
        PeakAlignIntensityArea
    )
except ImportError:
    detection = None

try:
    from . import normalization
    from .normalization import (
        batch_tic_norm,
        data_preprocessing,
        normalize,
        normalization_method_names,
        normalization_target,
        tic_normalization,
        BatchTicNorm,
        NormalizationEvaluator,
        NormalizationMethods,
    )
except ImportError:
    normalization = None

try:
    from . import plotting
    from .plotting import (
        PlotPeak,
        PlotPeaks,
        PlotPeaksConfig,
        plot_overlapping_peaks
    )
except ImportError:
    plotting = None

from .recalibrate import (
    FlexibleCalibrator,
    FlexibleCalibConfig,
    AutoCalibrator,
    AutoCalibConfig,
    FlexibleCalibratorDebug,
    FlexibleCalibConfigDebug,
)

from .utils import import_data

try:
    from .pipeline import run_pipeline, PipelineConfig
except ImportError:
    run_pipeline = None
    PipelineConfig = None

__all__ = [
    "AggregateParams",
    "BaselineBatchCorrector",
    "BaselineMethodEvaluator",
    "BatchDenoising",
    "DenoisingMethods",
    "FlatParams",
    "ScanForFlatRegion",
    "align_peaks",
    "baseline",
    "baseline_correction",
    "baseline_method_names",
    "batch_denoise",
    "FlexibleCalibrator",
    "FlexibleCalibConfig",
    "batch_tic_norm",
    "check_overlapping_peaks",
    "check_overlapping_peaks2",
    "collect_peak_properties_batch",
    "compare_denoising_methods",
    "compare_methods_in_windows",
    "data_preprocessing",
    "decode_method_label",
    "denoise",
    "detection",
    "detect_peaks_cwt_with_area",
    "detect_peaks_with_area",
    "detect_peaks_with_area_v2",
    "import_data",
    "normalization",
    "normalization_target",
    "noise_filtering",
    "PlotPeak",
    "plot_pareto_delta_snr_vs_height",
    "plotting",
    "rank_method",
    "select_methods",
    "recalibrate",
    "robust_noise_estimation",
    "robust_noise_estimation_mz",
    "robust_peak_detection",
    "small_param_grid_preset",
    "tic_normalization",
    "utils",
    "run_pipeline",
    "PipelineConfig",
    "AutoCalibrator",
    "AutoCalibConfig",
    "PlotPeak",
    "PlotPeaks",
    "PlotPeaksConfig",
    "plot_overlapping_peaks",
    "FlexibleCalibratorDebug",
    "FlexibleCalibConfigDebug",
    "BatchTicNorm",
    "normalize",
    "normalization_method_names",
    "NormalizationEvaluator",
    "NormalizationMethods",
    "PeakAlignIntensityArea"
]
