"""Tests for the calibration model math and reference-mass selection helpers."""

import numpy as np
import pandas as pd
import pytest

from mioXpektron.recalibrate import _models as m
from mioXpektron.recalibrate.flexible_calibrator import (
    FlexibleCalibConfig,
    summarize_reference_mass_stability,
    select_stable_reference_masses,
    _mask_excluded_reference_masses,
    _filter_reference_masses,
    _filter_calibration_values,
    _absolute_ppm_errors,
    _select_worst_calibrant_index,
)


REFERENCE_MASSES = np.array([23.0, 39.0, 55.0, 91.0, 184.0, 369.0])


# --------------------------- ppm helpers ---------------------------

def test_ppm_error_zero_for_exact():
    m_true = np.array([100.0, 200.0])
    assert m._ppm_error(m_true, m_true) == 0.0


def test_ppm_error_scales():
    m_true = np.array([1_000_000.0])
    m_est = np.array([1_000_001.0])  # +1 ppm
    assert np.isclose(m._ppm_error(m_true, m_est), 1.0, atol=1e-6)


def test_ppm_error_no_valid_returns_inf():
    assert m._ppm_error(np.array([-1.0]), np.array([-1.0])) == np.inf


def test_ppm_to_da():
    assert np.isclose(m._ppm_to_da(1000.0, 10.0), 0.01)


# --------------------------- outlier / noise ---------------------------

def test_detect_outliers_huber_flags_spike():
    residuals = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 50.0])
    flags = m._detect_outliers_huber(residuals, threshold=3.0)
    assert flags[-1]
    assert not flags[:-1].any()


def test_detect_outliers_huber_too_few_all_false():
    flags = m._detect_outliers_huber(np.array([1.0, 2.0, 3.0]))
    assert not flags.any()


def test_estimate_noise_level_positive():
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 10, 200)) + rng.normal(0, 0.1, 200)
    noise = m._estimate_noise_level(signal)
    assert noise > 0


def test_estimate_noise_level_short_signal():
    assert m._estimate_noise_level(np.array([1.0, 2.0, 3.0])) >= 0


# --------------------------- forward / inverse round trips ---------------------------

def test_quad_sqrt_forward_inverse_roundtrip():
    k, c, t0 = 1000.0, 0.01, 50.0
    t = m._quad_sqrt_forward(REFERENCE_MASSES, k, c, t0)
    recovered = m._invert_quad_sqrt(t, k, c, t0)
    np.testing.assert_allclose(recovered, REFERENCE_MASSES, rtol=1e-8, atol=1e-6)


def test_invert_quad_sqrt_nan_below_t0():
    result = m._invert_quad_sqrt(np.array([10.0]), 1000.0, 0.01, 50.0)
    assert np.isnan(result[0])


def test_linear_sqrt_fit_and_invert_roundtrip():
    # Construct data exactly satisfying sqrt(m) = a*t + b.
    a, b = 0.02, 1.0
    t = np.array([10.0, 50.0, 100.0, 200.0])
    masses = (a * t + b) ** 2
    params = m._fit_linear_sqrt(masses, t)
    assert params is not None
    recovered = m._invert_linear_sqrt(t, *params)
    np.testing.assert_allclose(recovered, masses, rtol=1e-6)


def test_fit_linear_sqrt_too_few_points():
    assert m._fit_linear_sqrt(np.array([10.0]), np.array([1.0])) is None


def test_poly2_fit_and_invert_roundtrip():
    p2, p1, p0 = 1e-4, 0.5, 2.0
    t = np.array([10.0, 40.0, 90.0, 160.0])
    masses = (p2 * t + p1) * t + p0
    params = m._fit_poly2(masses, t)
    assert params is not None
    recovered = m._invert_poly2(t, *params)
    np.testing.assert_allclose(recovered, masses, rtol=1e-6)


def test_fit_poly2_too_few_points():
    assert m._fit_poly2(np.array([10.0, 20.0]), np.array([1.0, 2.0])) is None


# --------------------------- quad_sqrt validation & robust fit ---------------------------

def test_validate_quad_sqrt_accepts_monotone():
    ok, reason = m._validate_quad_sqrt_params(1000.0, 0.01, 50.0, REFERENCE_MASSES)
    assert ok is True
    assert reason == ""


def test_validate_quad_sqrt_rejects_non_monotone():
    # Large negative curvature term makes dt/dm negative at the top of the range.
    ok, reason = m._validate_quad_sqrt_params(1000.0, -100.0, 50.0, REFERENCE_MASSES)
    assert ok is False
    assert reason != ""


def test_fit_quad_sqrt_robust_recovers_params():
    k, c, t0 = 1500.0, 0.02, 30.0
    t = m._quad_sqrt_forward(REFERENCE_MASSES, k, c, t0)
    params = m._fit_quad_sqrt_robust(REFERENCE_MASSES, t)
    assert params is not None
    recovered = m._invert_quad_sqrt(t, *params)
    np.testing.assert_allclose(recovered, REFERENCE_MASSES, rtol=1e-4)


def test_fit_quad_sqrt_robust_too_few():
    assert m._fit_quad_sqrt_robust(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


def test_robust_initial_params_shape():
    t = m._quad_sqrt_forward(REFERENCE_MASSES, 1000.0, 0.0, 50.0)
    p = m._robust_initial_params_quad_sqrt(REFERENCE_MASSES, t)
    assert p.shape == (3,)
    assert p[0] > 0  # k positive


# --------------------------- peak fitting ---------------------------

def test_fit_gaussian_peak_center():
    x = np.linspace(90, 110, 41)
    y = 100 * np.exp(-0.5 * ((x - 100.3) / 1.5) ** 2) + 2
    center = m._fit_gaussian_peak(x, y, 100.0)
    assert center is not None
    assert abs(center - 100.3) < 0.2


def test_fit_gaussian_peak_too_few_points():
    assert m._fit_gaussian_peak(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.5) is None


def test_fit_voigt_peak_center():
    x = np.linspace(90, 110, 61)
    from scipy.special import voigt_profile

    y = 100 * voigt_profile(x - 100.2, 1.0, 0.8) + 1
    center = m._fit_voigt_peak(x, y, 100.0)
    assert center is not None
    assert abs(center - 100.2) < 0.3


# --------------------------- apply_model_to_spectrum ---------------------------

def test_apply_model_to_spectrum_quad_sqrt():
    k, c, t0 = 1000.0, 0.01, 50.0
    channels = m._quad_sqrt_forward(REFERENCE_MASSES, k, c, t0)
    mz = m.apply_model_to_spectrum(channels, "quad_sqrt", (k, c, t0))
    np.testing.assert_allclose(mz, REFERENCE_MASSES, rtol=1e-6)


# --------------------------- config ---------------------------

def test_flexible_calib_config_defaults():
    cfg = FlexibleCalibConfig(reference_masses=[23.0, 39.0])
    assert cfg.reference_masses == [23.0, 39.0]
    assert cfg.calibration_method in {"auto", "quad_sqrt"} or isinstance(
        cfg.calibration_method, str
    )


# --------------------------- reference-mass filtering ---------------------------

def test_mask_excluded_reference_masses():
    ref = np.array([23.0, 39.0, 55.0])
    mask = _mask_excluded_reference_masses(ref, [39.0])
    assert mask.tolist() == [True, False, True]


def test_mask_excluded_empty_keeps_all():
    ref = np.array([23.0, 39.0])
    assert _mask_excluded_reference_masses(ref, []).all()


def test_filter_reference_masses_preserves_order():
    out = _filter_reference_masses([23.0, 39.0, 55.0], [39.0])
    np.testing.assert_allclose(out, [23.0, 55.0])


def test_filter_calibration_values_subsets():
    original = [23.0, 39.0, 55.0]
    values = {"file1.txt": [100.0, 200.0, 300.0]}
    out = _filter_calibration_values(values, original, [23.0, 55.0])
    np.testing.assert_allclose(out["file1.txt"], [100.0, 300.0])


def test_filter_calibration_values_length_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        _filter_calibration_values({"f": [1.0]}, [23.0, 39.0], [23.0])


def test_absolute_ppm_errors():
    true = [100.0, 200.0]
    est = [100.0001, 200.0]
    out = _absolute_ppm_errors(true, est)
    assert out[0] > 0
    assert np.isclose(out[1], 0.0)


def test_select_worst_calibrant_index_prefers_above_floor():
    masses = [1.0, 23.0, 55.0]
    abs_ppm = [1000.0, 5.0, 40.0]  # worst overall is m=1 but below floor
    idx = _select_worst_calibrant_index(masses, abs_ppm, exclude_below_mz=1.5)
    assert idx == 2  # 55.0 has worst ppm among masses above floor


# --------------------------- stability & selection ---------------------------

def _make_summary_df():
    return pd.DataFrame(
        {
            "file_name": ["a.txt", "b.txt"],
            "calibrant_masses": [[23.0, 39.0, 55.0], [23.0, 39.0, 55.0]],
            "estimated_masses": [
                [23.0002, 39.02, 55.0001],   # 39 is unstable
                [23.0001, 39.03, 55.0002],
            ],
        }
    )


def test_summarize_reference_mass_stability():
    out = summarize_reference_mass_stability(_make_summary_df())
    assert set(["mass", "mean_abs_ppm", "valid_fraction"]).issubset(out.columns)
    # The 39.0 calibrant has much larger ppm error than 23 or 55.
    ppm_by_mass = dict(zip(out["mass"], out["mean_abs_ppm"]))
    assert ppm_by_mass[39.0] > ppm_by_mass[23.0]


def test_summarize_stability_empty():
    out = summarize_reference_mass_stability(pd.DataFrame())
    assert out.empty
    assert "mean_abs_ppm" in out.columns


def test_select_stable_reference_masses_drops_unstable():
    stability = summarize_reference_mass_stability(_make_summary_df())
    selected, table = select_stable_reference_masses(
        [23.0, 39.0, 55.0],
        stability,
        max_mean_abs_ppm=50.0,
        min_valid_fraction=0.5,
        min_count=1,
    )
    # 39.0 (~500 ppm) should be excluded; 23 and 55 retained.
    assert 39.0 not in set(np.round(selected, 3))
    assert 23.0 in set(np.round(selected, 3))
