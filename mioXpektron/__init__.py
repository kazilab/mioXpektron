"""Public package interface for the Xpektron toolkit."""

from ._metadata import (
    AUTHOR as __author__,
    COPYRIGHT as __copyright__,
    EMAIL as __email__,
    VERSION as __version__,
)

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
        resample_spectrum,
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
    from . import analysis
    from .analysis import (
        AnalysisConfig,
        AnalysisWorkflow,
        bh_fdr,
        calculate_multiclass_metrics,
        compare_model_results,
        compute_univariate_tests,
        evaluate_all_models,
        get_benchmark_models,
        infer_feature_columns,
        plot_confusion_matrix,
        plot_heatmap_top_features,
        plot_pca,
        plot_tsne,
        plot_umap,
        plot_volcano,
        run_embeddings,
        analysis_capabilities,
        prepare_matrix,
        prepare_ml_data,
        run_analysis,
        run_cnmf,
        run_multi_dataset_comparison,
        tune_top_models,
    )
except ImportError:
    analysis = None
    AnalysisConfig = None
    AnalysisWorkflow = None
    evaluate_all_models = None
    get_benchmark_models = None
    plot_tsne = None
    plot_umap = None
    run_embeddings = None
    analysis_capabilities = None
    prepare_ml_data = None
    run_analysis = None
    run_cnmf = None
    calculate_multiclass_metrics = None
    compare_model_results = None
    plot_confusion_matrix = None
    run_multi_dataset_comparison = None
    tune_top_models = None

try:
    from .pipeline import run_pipeline, PipelineConfig, DEFAULT_REFERENCE_MASSES
except ImportError:
    run_pipeline = None
    PipelineConfig = None
    DEFAULT_REFERENCE_MASSES = None

try:
    from .adaptive import (
        auto_tune_calib_config,
        estimate_autodetect_tolerance,
        estimate_bootstrap_heuristics,
        estimate_denoise_params,
        estimate_flat_params,
        estimate_multisegment_breakpoints,
        estimate_mz_tolerance,
        estimate_normalization_target,
        estimate_outlier_threshold,
        estimate_screening_thresholds,
    )
except ImportError:
    auto_tune_calib_config = None

__all__ = [
    "__author__",
    "__copyright__",
    "__email__",
    "__version__",
    "analysis",
    "AnalysisConfig",
    "AnalysisWorkflow",
    "bh_fdr",
    "compute_univariate_tests",
    "evaluate_all_models",
    "get_benchmark_models",
    "infer_feature_columns",
    "plot_heatmap_top_features",
    "plot_pca",
    "plot_tsne",
    "plot_umap",
    "run_embeddings",
    "analysis_capabilities",
    "plot_volcano",
    "prepare_matrix",
    "prepare_ml_data",
    "run_analysis",
    "run_cnmf",
    "calculate_multiclass_metrics",
    "compare_model_results",
    "plot_confusion_matrix",
    "run_multi_dataset_comparison",
    "tune_top_models",
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
    "resample_spectrum",
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
    "PeakAlignIntensityArea",
    "DEFAULT_REFERENCE_MASSES",
    "auto_tune_calib_config",
    "estimate_autodetect_tolerance",
    "estimate_bootstrap_heuristics",
    "estimate_denoise_params",
    "estimate_flat_params",
    "estimate_multisegment_breakpoints",
    "estimate_mz_tolerance",
    "estimate_normalization_target",
    "estimate_outlier_threshold",
    "estimate_screening_thresholds",
]
