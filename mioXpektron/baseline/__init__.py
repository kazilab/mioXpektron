"""Baseline correction utilities for the Xpektron toolkit."""

from .baseline_eval import BaselineMethodEvaluator
from .baseline_base import (baseline_correction,
                            baseline_method_names,
                            small_param_grid_preset)
from .baseline_batch import BaselineBatchCorrector
from .flat_window_suggester import (
    AggregateParams,
    FlatParams,
    ScanForFlatRegion,
)

__all__ = [
    "AggregateParams",
    "BaselineBatchCorrector",
    "BaselineMethodEvaluator",
    "FlatParams",
    "ScanForFlatRegion",
    "baseline_correction",
    "baseline_method_names",
    "small_param_grid_preset",
]
