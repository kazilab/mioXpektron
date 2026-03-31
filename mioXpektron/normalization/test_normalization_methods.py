import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import Akima1DInterpolator

from mioXpektron import resample_spectrum as public_resample_spectrum
from mioXpektron.normalization import (
    normalization_method_names,
    normalize,
    resample_spectrum,
)
from mioXpektron.normalization.normalization import (
    mass_stratified_pqn_normalization,
    multi_ion_reference_normalization,
    pareto_normalization,
    robust_snv_normalization,
    tic_normalization,
)


def test_normalization_method_registry_includes_new_methods():
    methods = normalization_method_names()
    assert "robust_snv" in methods
    assert "pareto" in methods
    assert "multi_ion_reference" in methods
    assert "mass_stratified_pqn" in methods


def test_resample_spectrum_is_exported_from_public_apis():
    assert public_resample_spectrum is resample_spectrum


def test_resample_spectrum_linear_sorts_and_deduplicates():
    mz_values = np.array([3.0, 1.0, 2.0, 2.0], dtype=float)
    intensities = np.array([9.0, 1.0, 4.0, 5.0], dtype=float)
    target_mz = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.5], dtype=float)

    result = resample_spectrum(
        mz_values,
        intensities,
        target_mz,
        method="linear",
    )

    expected = np.array([0.0, 1.0, 2.5, 4.0, 6.5, 0.0], dtype=float)
    assert_allclose(result, expected)


def test_resample_spectrum_supports_makima_when_available():
    mz_values = np.array([0.0, 1.0, 2.0, 4.0], dtype=float)
    intensities = np.array([0.0, 1.0, 0.5, 2.0], dtype=float)
    target_mz = np.array([-1.0, 0.5, 3.0, 5.0], dtype=float)

    try:
        expected = Akima1DInterpolator(
            mz_values,
            intensities,
            method="makima",
            extrapolate=False,
        )(target_mz)
    except (TypeError, ValueError):
        with pytest.raises(ValueError, match="makima"):
            resample_spectrum(
                mz_values,
                intensities,
                target_mz,
                method="makima",
            )
        return

    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
    outside = (target_mz < mz_values[0]) | (target_mz > mz_values[-1])
    expected = np.where(outside, 0.0, expected)
    expected = np.clip(expected, 0.0, None)

    result = resample_spectrum(
        mz_values,
        intensities,
        target_mz,
        method="makima",
    )
    assert_allclose(result, expected)


def test_robust_snv_uses_median_and_mad():
    intensities = np.array([1.0, 2.0, 100.0], dtype=float)
    result = robust_snv_normalization(intensities)
    expected = np.array(
        [
            (1.0 - 2.0) / 1.4826,
            0.0,
            (100.0 - 2.0) / 1.4826,
        ]
    )
    assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_robust_snv_returns_zeros_when_mad_is_zero():
    intensities = np.array([5.0, 5.0, 5.0], dtype=float)
    result = robust_snv_normalization(intensities)
    assert_allclose(result, np.zeros_like(intensities))


def test_pareto_scaling_requires_dataset_statistics():
    intensities = np.array([2.0, 4.0, 6.0], dtype=float)
    mean = np.array([1.0, 2.0, 3.0], dtype=float)
    std = np.array([1.0, 4.0, 9.0], dtype=float)

    direct = pareto_normalization(intensities, mean=mean, std=std)
    dispatched = normalize(intensities, method="pareto", mean=mean, std=std)

    expected = np.array([1.0, 1.0, 1.0], dtype=float)
    assert_allclose(direct, expected)
    assert_allclose(dispatched, expected)


def test_multi_ion_reference_uses_median_ratio_to_reference_values():
    intensities = np.array([2.0, 10.0, 20.0, 30.0], dtype=float)
    reference_indices = [1, 2]
    reference_values = [5.0, 10.0]

    direct = multi_ion_reference_normalization(
        intensities,
        reference_indices=reference_indices,
        reference_values=reference_values,
    )
    dispatched = normalize(
        intensities,
        method="multi_ion_reference",
        reference_indices=reference_indices,
        reference_values=reference_values,
    )

    expected = np.array([1.0, 5.0, 10.0, 15.0], dtype=float)
    assert_allclose(direct, expected)
    assert_allclose(dispatched, expected)


def test_mass_stratified_pqn_corrects_region_specific_bias():
    intensities = np.array([1.0, 2.0, 40.0, 80.0], dtype=float)
    reference = np.array([1.0, 2.0, 10.0, 20.0], dtype=float)
    mz_values = np.array([50.0, 80.0, 150.0, 180.0], dtype=float)
    strata = [(0.0, 100.0), (100.0, 200.0)]

    direct = mass_stratified_pqn_normalization(
        intensities,
        mz_values=mz_values,
        reference=reference,
        strata=strata,
    )
    dispatched = normalize(
        intensities,
        method="mass_stratified_pqn",
        mz_values=mz_values,
        reference=reference,
        strata=strata,
    )

    expected = tic_normalization(reference)
    assert_allclose(direct, expected)
    assert_allclose(dispatched, expected)
