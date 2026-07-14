"""Tests for the high-level pipeline configuration and orchestration."""

import numpy as np
import pandas as pd

from mioXpektron import pipeline
from mioXpektron.pipeline import (
    PipelineConfig,
    DEFAULT_REFERENCE_MASSES,
    _maybe_recalibrate,
    _load_apply_denoise_baseline_normalize,
)


def test_default_reference_masses_sane():
    assert len(DEFAULT_REFERENCE_MASSES) > 10
    assert all(m > 0 for m in DEFAULT_REFERENCE_MASSES)


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.use_recalibration is True
    assert cfg.denoise_method == "wavelet"
    assert cfg.baseline_method == "airpls"
    assert cfg.auto_tune is False


def test_maybe_recalibrate_disabled_returns_original(spectrum_files):
    cfg = PipelineConfig(use_recalibration=False)
    out = _maybe_recalibrate(spectrum_files, {"x": [1, 2, 3]}, cfg)
    assert out == list(spectrum_files)


def test_maybe_recalibrate_no_calib_dict_returns_original(spectrum_files):
    cfg = PipelineConfig(use_recalibration=True)
    out = _maybe_recalibrate(spectrum_files, None, cfg)
    assert out == list(spectrum_files)


def test_load_apply_denoise_baseline_normalize(spectrum_file):
    cfg = PipelineConfig(denoise_method="gaussian", baseline_method="airpls",
                         normalization_target=1e6)
    sample_name, mz, y = _load_apply_denoise_baseline_normalize(spectrum_file, cfg)
    assert sample_name == "sample_CT-1a_1"
    assert mz.shape == y.shape
    assert np.all(np.isfinite(y))


def test_load_apply_no_denoise(spectrum_file):
    cfg = PipelineConfig(denoise_method="none", baseline_method="airpls")
    _, mz, y = _load_apply_denoise_baseline_normalize(spectrum_file, cfg)
    assert y.shape == mz.shape


def test_run_pipeline_end_to_end(spectrum_files):
    cfg = PipelineConfig(
        use_recalibration=False,
        max_workers=1,
        mz_tolerance=0.3,
    )
    intensity_df, area_df = pipeline.run_pipeline(spectrum_files, config=cfg)
    assert isinstance(intensity_df, pd.DataFrame)
    assert isinstance(area_df, pd.DataFrame)
    assert not intensity_df.empty
