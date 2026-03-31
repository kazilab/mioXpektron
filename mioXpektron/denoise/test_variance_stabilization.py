import numpy as np
import pytest

from .denoise_main import (
    _anscombe_forward,
    _anscombe_inverse_unbiased,
    noise_filtering,
    wavelet_denoise,
)


def test_classical_anscombe_forward_and_unbiased_inverse_form_consistent_pair():
    counts = np.array([0.0, 1.0, 4.0, 9.0, 25.0, 100.0], dtype=float)

    recovered = _anscombe_inverse_unbiased(_anscombe_forward(counts))

    assert np.all(np.isfinite(recovered))
    np.testing.assert_allclose(recovered, counts, atol=0.5, rtol=0.0)


def test_wavelet_denoise_anscombe_can_raise_on_negative_input():
    signal = np.array([2.0, 1.5, -0.25, 3.0, 2.5], dtype=float)

    with pytest.raises(ValueError, match="Classical Anscombe VST assumes non-negative Poisson-like input"):
        wavelet_denoise(
            signal,
            variance_stabilize="anscombe",
            anscombe_negative_policy="raise",
            cycle_spins=0,
        )


def test_noise_filtering_anscombe_warns_and_clips_negative_input_by_default():
    signal = np.array([2.0, 1.5, -0.25, 3.0, 2.5, 1.0, 0.5, 2.2], dtype=float)

    with pytest.warns(RuntimeWarning, match="Clipping negatives to zero before the classical Anscombe transform"):
        out = noise_filtering(
            signal,
            method="wavelet",
            variance_stabilize="anscombe",
            anscombe_negative_policy="warn_clip",
            cycle_spins=0,
        )

    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
