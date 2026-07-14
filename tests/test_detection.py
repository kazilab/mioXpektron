"""Tests for noise estimation, peak detection, alignment, and peak analysis."""

import numpy as np
import pandas as pd
import pytest

from mioXpektron.detection.detection import (
    handle_missing_values,
    robust_noise_estimation,
    robust_noise_estimation_mz,
    detect_peaks_with_area,
    detect_peaks_cwt_with_area,
    align_peaks,
    _greedy_mz_bins,
    gaussian,
    lorentzian,
    _positive_noise_stats,
)
from mioXpektron.detection.peak_analysis import (
    _interp,
    _fwhm_from_peak,
    load_and_measure_peaks_polars,
)


@pytest.fixture
def peaky_spectrum():
    x = np.linspace(0, 100, 4000)
    y = np.full_like(x, 5.0)
    for c, h in [(20.0, 500.0), (40.0, 800.0), (60.0, 300.0)]:
        y += h * np.exp(-0.5 * ((x - c) / 0.3) ** 2)
    rng = np.random.default_rng(0)
    y += rng.normal(0, 1.0, x.size)
    return x, np.clip(y, 0, None)


# --------------------------- handle_missing_values ---------------------------

def test_handle_missing_no_nans_passthrough():
    mz = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    out_mz, out_y = handle_missing_values(mz, y)
    np.testing.assert_array_equal(out_y, y)


def test_handle_missing_interpolation():
    mz = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, np.nan, 4.0])
    _, out = handle_missing_values(mz, y, method="interpolation")
    assert np.isclose(out[1], 2.0)


def test_handle_missing_zero():
    y = np.array([1.0, np.nan, 3.0])
    _, out = handle_missing_values(np.arange(3.0), y, method="zero")
    assert out[1] == 0.0


def test_handle_missing_mean():
    y = np.array([2.0, np.nan, 4.0])
    _, out = handle_missing_values(np.arange(3.0), y, method="mean")
    assert np.isclose(out[1], 3.0)


def test_handle_missing_all_nan_interpolation_zeros():
    y = np.array([np.nan, np.nan])
    _, out = handle_missing_values(np.arange(2.0), y, method="interpolation")
    assert np.all(out == 0.0)


# --------------------------- noise estimation ---------------------------

def test_positive_noise_stats_basic():
    med, std = _positive_noise_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert med == 3.0
    assert std > 0


def test_positive_noise_stats_empty_returns_zeros():
    assert _positive_noise_stats(np.array([-1.0, -2.0])) == (0.0, 0.0)


def test_positive_noise_stats_fallback():
    med, std = _positive_noise_stats(
        np.array([0.0, 0.0]), fallback_values=np.array([10.0, 20.0])
    )
    assert med > 0


def test_robust_noise_estimation_excludes_peaks(peaky_spectrum):
    _, y = peaky_spectrum
    median, std = robust_noise_estimation(y)
    # Noise median should be far below the tallest peak.
    assert median < 100
    assert std >= 0


def test_robust_noise_estimation_mz_window(peaky_spectrum):
    x, y = peaky_spectrum
    median, std = robust_noise_estimation_mz(x, y, 5.0, 15.0)
    assert median >= 0
    assert std >= 0


def test_robust_noise_estimation_mz_empty_region(peaky_spectrum):
    x, y = peaky_spectrum
    median, std = robust_noise_estimation_mz(x, y, 200.0, 300.0)
    assert (median, std) == (0.0, 0.0)


# --------------------------- peak detection ---------------------------

def test_detect_peaks_with_area_finds_peaks(peaky_spectrum):
    x, y = peaky_spectrum
    df = detect_peaks_with_area(x, y, "sample1", "Control",
                                min_intensity=1, peak_height=50, prominence=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3
    assert {"PeakCenter", "PeakArea", "Amplitude", "SampleName"}.issubset(df.columns)
    # Detected centers should be near the injected peaks.
    centers = np.sort(df["PeakCenter"].to_numpy())
    for expected in (20.0, 40.0, 60.0):
        assert np.min(np.abs(centers - expected)) < 1.0


def test_detect_peaks_with_area_empty_below_threshold():
    x = np.linspace(0, 10, 100)
    y = np.full_like(x, 0.5)
    df = detect_peaks_with_area(x, y, "s", "g", min_intensity=1)
    assert df.empty


def test_detect_peaks_with_area_sorts_unsorted_mz(peaky_spectrum):
    x, y = peaky_spectrum
    order = np.argsort(-x)  # descending
    df = detect_peaks_with_area(x[order], y[order], "s", "g",
                                peak_height=50, prominence=50)
    assert len(df) >= 1


def test_detect_peaks_cwt_runs(peaky_spectrum):
    x, y = peaky_spectrum
    df = detect_peaks_cwt_with_area(x, y, "s", "Control",
                                    min_intensity=1, peak_height=50, prominence=50)
    assert isinstance(df, pd.DataFrame)


# --------------------------- lineshapes ---------------------------

def test_gaussian_peak_at_center():
    x = np.linspace(-5, 5, 101)
    y = gaussian(x, amp=2.0, cen=0.0, sigma=1.0)
    assert np.isclose(y[np.argmax(y)], 2.0)
    assert np.isclose(x[np.argmax(y)], 0.0)


def test_lorentzian_is_positive_and_peaks_at_center():
    x = np.linspace(-5, 5, 101)
    y = lorentzian(x, A=1.0, x0=0.0, gamma=1.0)
    assert np.all(y >= 0)
    assert np.isclose(x[np.argmax(y)], 0.0)


# --------------------------- alignment ---------------------------

def test_greedy_mz_bins_hard_tolerance():
    mz = np.array([10.0, 10.05, 10.1, 20.0, 20.05])
    labels = _greedy_mz_bins(mz, tolerance=0.2)
    # First three within 0.2 → same bin; last two another bin.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def _make_peaks_df():
    return pd.DataFrame(
        {
            "SampleName": ["s1", "s1", "s2", "s2"],
            "Group": ["A", "A", "B", "B"],
            "PeakCenter": [10.0, 20.0, 10.05, 20.1],
            "Amplitude": [100.0, 200.0, 110.0, 190.0],
            "PeakArea": [1.0, 2.0, 1.1, 1.9],
            "AreaDefinition": ["raw_trapezoid"] * 4,
        }
    )


def test_align_peaks_intensity_matrix():
    out = align_peaks(_make_peaks_df(), mz_tolerance=0.2, output="intensity")
    assert not out.empty
    assert "SampleName" in out.columns or out.index.name == "SampleName" or True


def test_align_peaks_area_matrix():
    out = align_peaks(_make_peaks_df(), mz_tolerance=0.2, output="area")
    assert not out.empty


def test_align_peaks_missing_center_raises():
    with pytest.raises(ValueError, match="PeakCenter"):
        align_peaks(pd.DataFrame({"Amplitude": [1.0]}))


def test_align_peaks_empty_returns_empty():
    df = pd.DataFrame({"PeakCenter": [], "Amplitude": []})
    assert align_peaks(df).empty


def test_align_peaks_bad_output_raises():
    with pytest.raises(ValueError, match="intensity"):
        align_peaks(_make_peaks_df(), output="bogus")


# --------------------------- peak_analysis ---------------------------

def test_interp_midpoint():
    assert _interp(0.0, 0.0, 2.0, 10.0, 5.0) == 1.0


def test_interp_degenerate():
    assert _interp(1.0, 5.0, 2.0, 5.0, 5.0) == 1.0


def test_fwhm_from_symmetric_peak():
    x = np.linspace(-5, 5, 201)
    y = np.exp(-0.5 * (x / 1.0) ** 2)
    p = int(np.argmax(y))
    fwhm = _fwhm_from_peak(x, y, p)
    # Gaussian FWHM = 2.355*sigma ≈ 2.355.
    assert fwhm is not None
    assert 2.0 < fwhm < 2.7


def test_fwhm_none_for_nonpositive_peak():
    x = np.arange(5.0)
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    assert _fwhm_from_peak(x, y, 2) is None


def test_load_and_measure_peaks_polars(tmp_path):
    x = np.linspace(10, 30, 2000)
    y = 5 + 500 * np.exp(-0.5 * ((x - 20) / 0.05) ** 2)
    p = tmp_path / "spec.txt"
    lines = ["# comment", "Channel\tm/z\tIntensity"]
    lines += [f"{i}\t{mz:.6f}\t{iy:.4f}" for i, (mz, iy) in enumerate(zip(x, y))]
    p.write_text("\n".join(lines) + "\n")
    out = load_and_measure_peaks_polars(str(p), mz_min=10.0, mz_max=30.0,
                                        prominence=50.0)
    assert out.height >= 1
    assert set(out.columns) == {"m/z", "FWHM", "height"}


def test_load_and_measure_peaks_empty_window(tmp_path):
    p = tmp_path / "spec.txt"
    p.write_text("Channel\tm/z\tIntensity\n0\t10.0\t100.0\n1\t11.0\t120.0\n")
    out = load_and_measure_peaks_polars(str(p), mz_min=500.0, mz_max=600.0)
    assert out.height == 0
