"""Unit tests for the 1-D normalization methods in normalization.normalization."""

import numpy as np
import pytest

from mioXpektron.normalization import normalization as norm


@pytest.fixture
def signal():
    return np.array([1.0, 2.0, 3.0, 4.0, 10.0], dtype=float)


def test_method_names_registry_is_nonempty_and_sorted():
    names = norm.normalization_method_names()
    assert "tic" in names
    assert names == sorted(names)
    assert len(names) > 5


def test_normalize_dispatches_to_named_method(signal):
    out = norm.normalize(signal, method="tic", target_tic=100.0)
    assert np.isclose(out.sum(), 100.0)


def test_normalize_unknown_method_raises(signal):
    with pytest.raises(ValueError, match="Unknown normalization method"):
        norm.normalize(signal, method="does-not-exist")


def test_tic_normalization_scales_to_target(signal):
    out = norm.tic_normalization(signal, target_tic=1000.0)
    assert np.isclose(out.sum(), 1000.0)


def test_tic_normalization_none_target_returns_cleaned_input(signal):
    out = norm.tic_normalization(signal, target_tic=None)
    np.testing.assert_allclose(out, signal)


def test_tic_normalization_zero_signal_warns_and_returns_zeros():
    with pytest.warns(UserWarning):
        out = norm.tic_normalization(np.zeros(4), target_tic=100.0)
    assert np.all(out == 0)


def test_median_normalization_sets_median(signal):
    out = norm.median_normalization(signal, target_median=2.0)
    assert np.isclose(np.median(out[out > 0]), 2.0)


def test_median_normalization_all_zero_warns():
    with pytest.warns(UserWarning):
        out = norm.median_normalization(np.zeros(5))
    assert np.all(out == 0)


def test_rms_normalization_sets_rms(signal):
    out = norm.rms_normalization(signal, target_rms=1.0)
    assert np.isclose(np.sqrt(np.mean(out ** 2)), 1.0)


def test_rms_normalization_zero_warns():
    with pytest.warns(UserWarning):
        out = norm.rms_normalization(np.zeros(3))
    assert np.all(out == 0)


def test_max_normalization_peak_is_one(signal):
    out = norm.max_normalization(signal)
    assert np.isclose(out.max(), 1.0)


def test_max_normalization_zero_returns_zeros():
    assert np.all(norm.max_normalization(np.zeros(4)) == 0)


def test_vector_normalization_unit_norm(signal):
    out = norm.vector_normalization(signal)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_vector_normalization_zero_returns_zeros():
    assert np.all(norm.vector_normalization(np.zeros(4)) == 0)


def test_snv_normalization_zero_mean_unit_std(signal):
    out = norm.snv_normalization(signal)
    assert np.isclose(out.mean(), 0.0, atol=1e-9)
    assert np.isclose(out.std(), 1.0, atol=1e-9)


def test_snv_normalization_constant_returns_zeros():
    assert np.all(norm.snv_normalization(np.full(5, 3.0)) == 0)


def test_robust_snv_centered_on_median(signal):
    out = norm.robust_snv_normalization(signal)
    assert np.isclose(np.median(out), 0.0, atol=1e-9)


def test_robust_snv_constant_returns_zeros():
    assert np.all(norm.robust_snv_normalization(np.full(5, 7.0)) == 0)


def test_poisson_scaling_divides_by_sqrt_mean(signal):
    out = norm.poisson_scaling(signal)
    expected = signal / np.sqrt(signal.mean())
    np.testing.assert_allclose(out, expected)


def test_poisson_scaling_zero_returns_zeros():
    assert np.all(norm.poisson_scaling(np.zeros(4)) == 0)


def test_sqrt_normalization_matches_sqrt(signal):
    out = norm.sqrt_normalization(signal)
    np.testing.assert_allclose(out, np.sqrt(signal))


def test_sqrt_normalization_clips_negatives():
    out = norm.sqrt_normalization(np.array([-4.0, 4.0]))
    np.testing.assert_allclose(out, np.array([0.0, 2.0]))


def test_log_normalization_matches_log1p_scaled(signal):
    out = norm.log_normalization(signal, pseudo_count=1.0)
    np.testing.assert_allclose(out, np.log(signal + 1.0))


def test_selected_ion_normalization_by_index(signal):
    out = norm.selected_ion_normalization(signal, reference_idx=4, target=1.0)
    assert np.isclose(out[4], 1.0)


def test_selected_ion_normalization_by_value(signal):
    out = norm.selected_ion_normalization(signal, reference_intensity=10.0, target=2.0)
    np.testing.assert_allclose(out, signal * (2.0 / 10.0))


def test_selected_ion_out_of_bounds_raises(signal):
    with pytest.raises(IndexError):
        norm.selected_ion_normalization(signal, reference_idx=99)


def test_selected_ion_no_reference_raises(signal):
    with pytest.raises(ValueError):
        norm.selected_ion_normalization(signal)


def test_selected_ion_zero_reference_warns(signal):
    with pytest.warns(UserWarning):
        out = norm.selected_ion_normalization(signal, reference_intensity=0.0)
    assert np.all(out == 0)


def test_multi_ion_reference_with_values(signal):
    out = norm.multi_ion_reference_normalization(
        signal, reference_indices=[0, 4], reference_values=[1.0, 5.0]
    )
    # median ratio = median([1/1, 10/5]) = 1.5 → scaled = signal / 1.5
    np.testing.assert_allclose(out, signal / 1.5)


def test_multi_ion_reference_without_values(signal):
    out = norm.multi_ion_reference_normalization(
        signal, reference_indices=[0, 1, 2], target=1.0
    )
    assert np.all(np.isfinite(out))


def test_multi_ion_reference_requires_indices(signal):
    with pytest.raises(ValueError):
        norm.multi_ion_reference_normalization(signal)


def test_multi_ion_reference_out_of_bounds(signal):
    with pytest.raises(IndexError):
        norm.multi_ion_reference_normalization(signal, reference_indices=[0, 99])


def test_multi_ion_reference_shape_mismatch(signal):
    with pytest.raises(ValueError):
        norm.multi_ion_reference_normalization(
            signal, reference_indices=[0, 1], reference_values=[1.0]
        )


@pytest.mark.parametrize(
    "name",
    ["tic", "median", "rms", "max", "vector", "snv", "sqrt", "log", "poisson"],
)
def test_all_simple_methods_preserve_length(signal, name):
    out = norm.normalize(signal, method=name)
    assert out.shape == signal.shape
