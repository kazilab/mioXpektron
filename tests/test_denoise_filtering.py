"""Tests for the noise_filtering front-end and wavelet denoising helpers."""

import numpy as np
import pytest

from mioXpektron.denoise.denoise_main import (
    noise_filtering,
    wavelet_denoise,
    _mad_sigma,
    _circular_shift,
    _inverse_circular_shift,
    _anscombe_forward,
    _anscombe_inverse_unbiased,
)


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 50, 1000)
    clean = 100 * np.exp(-0.5 * ((x - 25) / 0.5) ** 2) + 10
    return clean + rng.normal(0, 3, size=x.size)


# --------------------------- helpers ---------------------------

def test_mad_sigma_of_gaussian_close_to_std():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2.0, 5000)
    assert 1.5 < _mad_sigma(x) < 2.5


def test_circular_shift_roundtrip():
    x = np.arange(10.0)
    shifted = _circular_shift(x, 3)
    restored = _inverse_circular_shift(shifted, 3)
    np.testing.assert_allclose(restored, x)


def test_circular_shift_zero_is_identity():
    x = np.arange(5.0)
    np.testing.assert_allclose(_circular_shift(x, 0), x)


def test_anscombe_roundtrip_approx():
    y = np.array([0.0, 1.0, 5.0, 20.0, 100.0])
    z = _anscombe_forward(y)
    back = _anscombe_inverse_unbiased(z)
    # Unbiased inverse should recover the counts to within ~1.
    assert np.all(np.abs(back - y) < 1.5)


# --------------------------- noise_filtering ---------------------------

def test_noise_filtering_requires_1d():
    with pytest.raises(ValueError, match="1D"):
        noise_filtering(np.ones((3, 3)))


def test_noise_filtering_x_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        noise_filtering(np.ones(5), x=np.ones(4))


def test_noise_filtering_all_nan_returns_zeros():
    out = noise_filtering(np.full(10, np.nan), method="gaussian")
    assert out.shape == (10,)
    assert np.all(out == 0)


def test_noise_filtering_none_method_returns_input(noisy_signal):
    out = noise_filtering(noisy_signal, method="none", clip_nonnegative=False)
    np.testing.assert_allclose(out, noisy_signal)


@pytest.mark.parametrize("method", ["savitzky_golay", "gaussian", "median", "wavelet"])
def test_noise_filtering_reduces_noise(noisy_signal, method):
    out = noise_filtering(noisy_signal, method=method)
    assert out.shape == noisy_signal.shape
    # Smoothed signal should have lower point-to-point variation.
    assert np.std(np.diff(out)) < np.std(np.diff(noisy_signal))


def test_noise_filtering_clip_nonnegative(noisy_signal):
    out = noise_filtering(noisy_signal, method="gaussian", clip_nonnegative=True)
    assert np.all(out >= 0)


def test_noise_filtering_preserve_tic(noisy_signal):
    positive = np.clip(noisy_signal, 0, None)
    out = noise_filtering(positive, method="gaussian", clip_nonnegative=True,
                          preserve_tic=True)
    assert np.isclose(out.sum(), positive.sum(), rtol=1e-6)


def test_noise_filtering_unknown_method_raises(noisy_signal):
    with pytest.raises(ValueError):
        noise_filtering(noisy_signal, method="bogus")


def test_noise_filtering_savgol_even_window_coerced(noisy_signal):
    # Even window length should be coerced to odd rather than raising.
    out = noise_filtering(noisy_signal, method="savitzky_golay", window_length=14)
    assert out.shape == noisy_signal.shape


def test_noise_filtering_resample_to_uniform(noisy_signal):
    x = np.sort(np.random.default_rng(2).uniform(0, 50, noisy_signal.size))
    out = noise_filtering(
        noisy_signal, method="gaussian", x=x, resample_to_uniform=True
    )
    assert out.shape == noisy_signal.shape


# --------------------------- wavelet_denoise ---------------------------

@pytest.mark.parametrize("strategy", ["universal", "bayes", "sure"])
def test_wavelet_denoise_strategies(noisy_signal, strategy):
    out = wavelet_denoise(noisy_signal, threshold_strategy=strategy)
    assert out.shape == noisy_signal.shape
    assert np.all(np.isfinite(out))


def test_wavelet_denoise_cycle_spins(noisy_signal):
    out = wavelet_denoise(noisy_signal, cycle_spins=4)
    assert out.shape == noisy_signal.shape


def test_wavelet_denoise_hard_threshold(noisy_signal):
    out = wavelet_denoise(noisy_signal, threshold_mode="hard")
    assert out.shape == noisy_signal.shape
