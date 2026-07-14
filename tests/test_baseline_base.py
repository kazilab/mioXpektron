"""Tests for baseline correction, method registry, parsing, and I/O."""

import numpy as np
import polars as pl
import pytest

from mioXpektron.baseline.baseline_base import (
    baseline_correction,
    baseline_method_names,
    small_param_grid_preset,
    _split_method_and_inline_kwargs,
    _standardize_columns,
    read_spectrum_table,
)


@pytest.fixture
def sloped_spectrum():
    x = np.linspace(0, 100, 1000)
    baseline = 0.5 * x + 20  # linear ramp
    peaks = 200 * np.exp(-0.5 * ((x - 50) / 0.6) ** 2)
    return baseline + peaks


# --------------------------- registry ---------------------------

def test_baseline_method_names_nonempty_sorted():
    names = baseline_method_names()
    assert names == sorted(names)
    assert "airpls" in names
    assert "median_filter" in names
    assert "poly" in names
    # Removed methods should not appear.
    assert "collab_pls" not in names


def test_small_param_grid_defaults():
    grid = small_param_grid_preset()
    assert grid["median_filter"] == [{"window_size": 501}, {"window_size": 1001},
                                     {"window_size": 2001}]


def test_small_param_grid_adaptive_odd_windows():
    grid = small_param_grid_preset(n_points=938000)
    for entry in grid["median_filter"]:
        assert entry["window_size"] % 2 == 1


# --------------------------- inline kwargs parsing ---------------------------

def test_split_method_plain():
    assert _split_method_and_inline_kwargs("airpls") == ("airpls", {})


def test_split_method_with_kwargs():
    name, kw = _split_method_and_inline_kwargs("aspls(lam=1000000.0, p=0.01)")
    assert name == "aspls"
    assert kw == {"lam": 1000000.0, "p": 0.01}


def test_split_method_bool_none_values():
    name, kw = _split_method_and_inline_kwargs("foo(a=true, b=false, c=none)")
    assert kw == {"a": True, "b": False, "c": None}


def test_split_method_empty_args():
    assert _split_method_and_inline_kwargs("foo()") == ("foo", {})


# --------------------------- baseline_correction ---------------------------

@pytest.mark.parametrize("method", ["airpls", "asls", "arpls", "modpoly", "poly"])
def test_baseline_correction_reduces_offset(sloped_spectrum, method):
    corrected = baseline_correction(sloped_spectrum, method=method)
    assert corrected.shape == sloped_spectrum.shape
    # Off-peak regions should be pushed toward zero.
    assert np.median(corrected) < np.median(sloped_spectrum)


def test_baseline_correction_clip_negative(sloped_spectrum):
    corrected = baseline_correction(sloped_spectrum, method="airpls",
                                    clip_negative=True)
    assert np.all(corrected >= 0)


def test_baseline_correction_return_baseline(sloped_spectrum):
    corrected, baseline = baseline_correction(
        sloped_spectrum, method="airpls", return_baseline=True
    )
    assert corrected.shape == baseline.shape
    np.testing.assert_allclose(corrected, np.clip(sloped_spectrum - baseline, 0, None),
                               atol=1e-6)


def test_baseline_correction_median_filter(sloped_spectrum):
    corrected = baseline_correction(sloped_spectrum, method="median_filter",
                                    window_size=51)
    assert corrected.shape == sloped_spectrum.shape


def test_baseline_correction_adaptive_window(sloped_spectrum):
    corrected = baseline_correction(sloped_spectrum, method="adaptive_window",
                                    window_size=51)
    assert np.all(corrected >= 0)


def test_baseline_correction_unknown_method(sloped_spectrum):
    with pytest.raises(ValueError, match="Unknown baseline method"):
        baseline_correction(sloped_spectrum, method="does_not_exist")


def test_baseline_correction_inline_kwargs(sloped_spectrum):
    corrected = baseline_correction(sloped_spectrum, method="airpls(lam=100000.0)")
    assert corrected.shape == sloped_spectrum.shape


# --------------------------- I/O ---------------------------

def test_standardize_columns_aliases():
    df = pl.DataFrame({"Channel": [1, 2], "m/z": [10.0, 20.0], "Counts": [5.0, 6.0]})
    out = _standardize_columns(df)
    assert out.columns == ["channel", "mz", "intensity"]


def test_standardize_columns_fabricates_channel():
    df = pl.DataFrame({"mz": [10.0, 20.0], "intensity": [5.0, 6.0]})
    out = _standardize_columns(df)
    assert "channel" in out.columns
    assert out["channel"].to_list() == [1, 2]


def test_standardize_columns_missing_required():
    df = pl.DataFrame({"foo": [1.0], "bar": [2.0]})
    with pytest.raises(KeyError):
        _standardize_columns(df)


def test_read_spectrum_table_tab(tmp_path):
    p = tmp_path / "spec.txt"
    p.write_text("Channel\tm/z\tIntensity\n1\t10.0\t100.0\n2\t20.0\t200.0\n")
    df = read_spectrum_table(p)
    assert set(df.columns) == {"channel", "mz", "intensity"}
    assert df.height == 2
