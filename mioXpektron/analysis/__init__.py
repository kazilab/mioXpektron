"""Downstream statistical analysis for aligned peak matrices."""

from .cnmf import (
    choose_k_by_pac,
    plot_pac_vs_k,
    run_cnmf,
    save_consensus_heatmap,
    save_factor_bars,
)
from .compare import (
    categorize_model,
    compare_model_results,
    plot_dataset_model_comparison,
    plot_family_comparison,
    run_multi_dataset_comparison,
    summarize_model_families,
)
from .embeddings import (
    compute_pca,
    compute_tsne,
    compute_umap,
    resolve_embedding_methods,
    run_embeddings,
)
from .metrics import (
    calculate_multiclass_metrics,
    get_class_names,
    plot_confusion_matrix,
)
from .ml import (
    evaluate_all_models,
    evaluate_model,
    explain_with_shap,
    get_benchmark_models,
    model_needs_scaling,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    prepare_ml_data,
    transform_features,
)
from .optional import analysis_capabilities, missing_packages
from .plots import (
    plot_heatmap_top_features,
    plot_pca,
    plot_tsne,
    plot_umap,
    plot_volcano,
)
from .prepare import infer_feature_columns, prepare_matrix
from .stats import bh_fdr, compute_univariate_tests
from .tuning import get_tuning_grid, select_best_tuned_model, tune_top_models
from .workflow import AnalysisConfig, AnalysisWorkflow, run_analysis

__all__ = [
    "AnalysisConfig",
    "AnalysisWorkflow",
    "analysis_capabilities",
    "bh_fdr",
    "calculate_multiclass_metrics",
    "categorize_model",
    "choose_k_by_pac",
    "compare_model_results",
    "compute_pca",
    "compute_tsne",
    "compute_umap",
    "compute_univariate_tests",
    "evaluate_all_models",
    "evaluate_model",
    "explain_with_shap",
    "get_benchmark_models",
    "get_class_names",
    "get_tuning_grid",
    "infer_feature_columns",
    "missing_packages",
    "model_needs_scaling",
    "plot_confusion_matrix",
    "plot_dataset_model_comparison",
    "plot_family_comparison",
    "plot_feature_importance",
    "plot_heatmap_top_features",
    "plot_model_comparison",
    "plot_pac_vs_k",
    "plot_pca",
    "plot_roc_curves",
    "plot_tsne",
    "plot_umap",
    "plot_volcano",
    "prepare_matrix",
    "prepare_ml_data",
    "resolve_embedding_methods",
    "run_analysis",
    "run_cnmf",
    "run_embeddings",
    "run_multi_dataset_comparison",
    "save_consensus_heatmap",
    "save_factor_bars",
    "select_best_tuned_model",
    "summarize_model_families",
    "transform_features",
    "tune_top_models",
]