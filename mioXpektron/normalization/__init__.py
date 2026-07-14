"""Normalization utilities for the Xpektron toolkit."""

from .normalization import (
    normalize,
    normalization_method_names,
    tic_normalization,
    median_normalization,
    rms_normalization,
    max_normalization,
    vector_normalization,
    snv_normalization,
    robust_snv_normalization,
    poisson_scaling,
    sqrt_normalization,
    log_normalization,
    selected_ion_normalization,
    multi_ion_reference_normalization,
    pqn_normalization,
    mass_stratified_pqn_normalization,
    median_of_ratios_normalization,
    vsn_normalization,
    minmax_normalization,
    pareto_normalization,
)
from .normalization_eval import NormalizationEvaluator
from .main import NormalizationMethods
from .tic_count import normalization_target
from .preprocessing import (
    batch_tic_norm,
    data_preprocessing,
    resample_spectrum,
    BatchTicNorm,
)

__all__ = [
    # Dispatch
    "normalize",
    "normalization_method_names",
    # Individual methods
    "tic_normalization",
    "median_normalization",
    "rms_normalization",
    "max_normalization",
    "vector_normalization",
    "snv_normalization",
    "robust_snv_normalization",
    "poisson_scaling",
    "sqrt_normalization",
    "log_normalization",
    "selected_ion_normalization",
    "multi_ion_reference_normalization",
    "pqn_normalization",
    "mass_stratified_pqn_normalization",
    "median_of_ratios_normalization",
    "vsn_normalization",
    "minmax_normalization",
    "pareto_normalization",
    # Evaluation
    "NormalizationEvaluator",
    # Orchestrator
    "NormalizationMethods",
    # Batch / preprocessing
    "batch_tic_norm",
    "data_preprocessing",
    "resample_spectrum",
    "normalization_target",
    "BatchTicNorm",
]
