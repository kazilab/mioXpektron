"""Tests for the calibration peak_picker helpers and pick_peaks dispatcher."""

import numpy as np
import pandas as pd
import pytest

from mioXpektron.recalibrate.peak_picker import (
    _ppm_to_da,
    _parabolic_vertex,
    _local_peak_bounds,
    _centroid_peak_center,
    _interpolate_channel_for_mz,
    pick_peaks,
)


def make_peak_df(centers=(23.0, 39.0), width=0.05, n_per=41, span=0.4, seed=0):
    """Build a Channel/m/z/Intensity DataFrame with Gaussian peaks at centers."""
    rng = np.random.default_rng(seed)
    mz_parts, int_parts = [], []
    for c in centers:
        local_mz = np.linspace(c - span, c + span, n_per)
        local_i = 1000 * np.exp(-0.5 * ((local_mz - c) / width) ** 2) + 5
        mz_parts.append(local_mz)
        int_parts.append(local_i)
    mz = np.concatenate(mz_parts)
    intensity = np.concatenate(int_parts) + rng.normal(0, 0.5, size=len(np.concatenate(int_parts)))
    channel = np.arange(len(mz))
    return pd.DataFrame({"Channel": channel, "m/z": mz, "Intensity": intensity})


# --------------------------- helpers ---------------------------

def test_ppm_to_da():
    assert np.isclose(_ppm_to_da(1000.0, 100.0), 0.1)


def test_parabolic_vertex_symmetric():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 0.0])  # apex at x=0
    assert np.isclose(_parabolic_vertex(x, y), 0.0)


def test_parabolic_vertex_offset():
    x = np.array([9.0, 10.0, 11.0])
    y = np.array([1.0, 3.0, 2.0])
    xv = _parabolic_vertex(x, y)
    assert 9.0 <= xv <= 11.0


def test_parabolic_vertex_wrong_length():
    assert _parabolic_vertex(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


def test_parabolic_vertex_degenerate_x():
    assert _parabolic_vertex(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None


def test_local_peak_bounds_covers_apex():
    x = np.linspace(0, 10, 101)
    y = np.exp(-0.5 * ((x - 5) / 0.5) ** 2)
    apex = int(np.argmax(y))
    left, right = _local_peak_bounds(x, y, apex)
    assert left <= apex <= right
    assert right > left


def test_local_peak_bounds_empty():
    assert _local_peak_bounds(np.array([]), np.array([]), 0) == (0, -1)


def test_centroid_peak_center_near_true():
    x = np.linspace(22.5, 23.5, 101)
    y = 1000 * np.exp(-0.5 * ((x - 23.05) / 0.05) ** 2) + 2
    apex = int(np.argmax(y))
    center = _centroid_peak_center(x, y, apex)
    assert center is not None
    assert abs(center - 23.05) < 0.05


def test_centroid_peak_center_empty():
    assert _centroid_peak_center(np.array([]), np.array([]), 0) is None


def test_interpolate_channel_for_mz():
    mz = np.array([10.0, 11.0, 12.0])
    ch = np.array([100.0, 200.0, 300.0])
    assert np.isclose(_interpolate_channel_for_mz(mz, ch, 11.5), 250.0)


def test_interpolate_channel_out_of_range():
    mz = np.array([10.0, 11.0])
    ch = np.array([100.0, 200.0])
    assert _interpolate_channel_for_mz(mz, ch, 99.0) is None


def test_interpolate_channel_non_finite():
    assert _interpolate_channel_for_mz(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.nan) is None


# --------------------------- pick_peaks ---------------------------

@pytest.mark.parametrize("method", ["max", "centroid", "centroid_raw", "parabolic"])
def test_pick_peaks_methods_find_targets(method):
    df = make_peak_df(centers=(23.0, 39.0))
    out = pick_peaks(df, targets=[23.0, 39.0], tol_da=0.3, tol_ppm=None, method=method)
    assert len(out) == 2
    for rec, expected in zip(out, [23.0, 39.0]):
        assert rec["n_points"] > 0
        assert abs(rec["Matched_m/z"] - expected) < 0.1


def test_pick_peaks_no_match_returns_nan():
    df = make_peak_df(centers=(23.0,))
    out = pick_peaks(df, targets=[500.0], tol_da=0.2, tol_ppm=None, method="max")
    assert len(out) == 1
    assert np.isnan(out[0]["Matched_m/z"])
    assert out[0]["method_used"] == "none"


def test_pick_peaks_unknown_method_raises():
    df = make_peak_df(centers=(23.0,))
    with pytest.raises(ValueError, match="Unknown method"):
        pick_peaks(df, targets=[23.0], tol_da=0.3, tol_ppm=None, method="bogus")


def test_pick_peaks_ppm_tolerance():
    df = make_peak_df(centers=(184.0,))
    out = pick_peaks(df, targets=[184.0], tol_da=None, tol_ppm=2000.0, method="max")
    assert out[0]["n_points"] > 0
    assert abs(out[0]["Matched_m/z"] - 184.0) < 0.1
