"""Unit tests for adaptive parameter estimation helpers."""

import numpy as np
import pandas as pd

from mioXpektron import adaptive


def test_sample_files_returns_all_when_small():
    files = ["a", "b", "c"]
    assert adaptive._sample_files(files, 10) == files


def test_sample_files_subsamples_and_spans_range():
    files = [str(i) for i in range(100)]
    chosen = adaptive._sample_files(files, 5)
    assert len(chosen) == 5
    assert chosen[0] == "0"
    assert chosen[-1] == "99"


def test_estimate_autodetect_tolerance_no_files_fallback():
    tol = adaptive.estimate_autodetect_tolerance([], [23.0, 39.0])
    assert tol == 0.5


def test_estimate_autodetect_tolerance_measures_peak(spectrum_files):
    tol = adaptive.estimate_autodetect_tolerance(spectrum_files, [23.0, 39.0, 55.0])
    assert 0.05 <= tol <= 2.0


def test_estimate_outlier_threshold_too_few_returns_default():
    assert adaptive.estimate_outlier_threshold(np.array([1.0, 2.0])) == 3.0


def test_estimate_outlier_threshold_zero_mad_returns_upper_bound():
    r = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    assert adaptive.estimate_outlier_threshold(r, bounds=(2.0, 5.0)) == 5.0


def test_estimate_outlier_threshold_within_bounds():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 500)
    thr = adaptive.estimate_outlier_threshold(r, bounds=(2.0, 5.0))
    assert 2.0 <= thr <= 5.0


def test_estimate_screening_thresholds_from_table():
    df = pd.DataFrame(
        {
            "mean_abs_ppm": [10.0, 20.0, 30.0, 100.0],
            "valid_fraction": [1.0, 0.9, 0.8, 0.5],
        }
    )
    result = adaptive.estimate_screening_thresholds(df)
    assert 5.0 <= result["screen_max_mean_abs_ppm"] <= 200.0
    assert 0.3 <= result["screen_min_valid_fraction"] <= 1.0


def test_estimate_screening_thresholds_empty_columns():
    assert adaptive.estimate_screening_thresholds(pd.DataFrame()) == {}


def test_estimate_multisegment_breakpoints_fallback_when_too_few():
    assert adaptive.estimate_multisegment_breakpoints([1.0, 2.0]) == [50.0, 200.0, 500.0]


def test_estimate_multisegment_breakpoints_quantiles():
    masses = list(range(1, 101))
    bps = adaptive.estimate_multisegment_breakpoints(masses, n_segments=3)
    assert len(bps) == 2
    assert bps == sorted(bps)


def test_estimate_normalization_target_no_files_fallback():
    assert adaptive.estimate_normalization_target([]) == 1e6


def test_estimate_normalization_target_median_tic(spectrum_files):
    target = adaptive.estimate_normalization_target(spectrum_files)
    assert target > 0


def test_estimate_mz_tolerance_no_files_fallback():
    assert adaptive.estimate_mz_tolerance([]) == 0.2


def test_estimate_mz_tolerance_from_spacing(spectrum_files):
    tol = adaptive.estimate_mz_tolerance(spectrum_files)
    assert 0.01 <= tol <= 1.0


def test_estimate_flat_params_returns_odd_window(spectrum_files):
    result = adaptive.estimate_flat_params(spectrum_files)
    if "savgol_window" in result:
        assert result["savgol_window"] % 2 == 1


def test_estimate_denoise_params_keys(spectrum_files):
    result = adaptive.estimate_denoise_params(spectrum_files)
    assert isinstance(result, dict)
    for key in result:
        assert key in {"hf_cutoff_frac", "max_peaks"}


def test_estimate_bootstrap_heuristics_from_channels(tmp_path):
    # Build a file with a Channel column so bootstrap stats are computed.
    from conftest import make_spectrum, write_spectrum_txt

    mz, y = make_spectrum(n=1000)
    path = write_spectrum_txt(tmp_path / "chan_CT-1a_1.txt", mz, y,
                              channel=np.arange(1000))
    result = adaptive.estimate_bootstrap_heuristics([path])
    assert "BOOTSTRAP_PEAK_DISTANCE_DIVISOR" in result
    assert "BOOTSTRAP_K_GUESS_MAX" in result


def test_auto_tune_calib_config_overrides(spectrum_files):
    masses = [23.0, 39.0, 55.0, 71.0]
    cfg = adaptive.auto_tune_calib_config(spectrum_files, masses)
    assert cfg.auto_screen_reference_masses is True
    assert 0.05 <= cfg.autodetect_tol_da <= 2.0
