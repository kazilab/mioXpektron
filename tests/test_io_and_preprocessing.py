"""Tests for data import, preprocessing, and TIC counting."""

import numpy as np
import pytest

from mioXpektron.utils.file_management import (
    import_data,
    _resolve_group,
    _resolve_column,
)
from mioXpektron.normalization.preprocessing import (
    resample_spectrum,
    data_preprocessing,
    batch_tic_norm,
)
from mioXpektron.normalization.tic_count import normalization_target


# --------------------------- file_management ---------------------------

def test_resolve_group_defaults_cancer():
    assert _resolve_group("breast_CC-1a_1", None, None) == "Cancer"


def test_resolve_group_defaults_control():
    assert _resolve_group("breast_CT-1a_1", None, None) == "Control"


def test_resolve_group_unknown():
    assert _resolve_group("something_else", None, None) == "Unknown"


def test_resolve_group_custom_fn_takes_priority():
    assert _resolve_group("x", {"_CC": "Cancer"}, lambda s: "Override") == "Override"


def test_resolve_group_custom_patterns():
    assert _resolve_group("tumor_1", {"tumor": "Tumor"}, None) == "Tumor"


def test_resolve_column_case_insensitive():
    import polars as pl

    df = pl.DataFrame({"Intensity": [1.0], "m/z": [1.0]})
    assert _resolve_column(df, ("intensity",)) == "Intensity"


def test_resolve_column_missing_raises():
    import polars as pl

    df = pl.DataFrame({"foo": [1.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        _resolve_column(df, ("bar",))


def test_import_data_reads_txt(spectrum_file):
    mz, intensity, sample_name, group = import_data(spectrum_file)
    assert mz.shape == intensity.shape
    assert sample_name == "sample_CT-1a_1"
    assert group == "Control"


def test_import_data_mz_filter(spectrum_file):
    mz, intensity, _, _ = import_data(spectrum_file, mz_min=20.0, mz_max=60.0)
    assert mz.min() >= 20.0
    assert mz.max() <= 60.0


def test_import_data_empty_after_filter_raises(spectrum_file):
    with pytest.raises(ValueError, match="No data"):
        import_data(spectrum_file, mz_min=1e9)


def test_import_data_csv(tmp_path):
    p = tmp_path / "s_CC-1a_1.csv"
    p.write_text("mz,corrected_intensity\n10.0,5.0\n20.0,7.0\n")
    mz, intensity, name, group = import_data(str(p))
    np.testing.assert_allclose(mz, [10.0, 20.0])
    np.testing.assert_allclose(intensity, [5.0, 7.0])
    assert group == "Cancer"


# --------------------------- resample_spectrum ---------------------------

def test_resample_linear_matches_interp():
    mz = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 10.0, 20.0])
    target = np.array([0.5, 1.5])
    out = resample_spectrum(mz, y, target, method="linear")
    np.testing.assert_allclose(out, [5.0, 15.0])


def test_resample_out_of_range_is_zero():
    mz = np.array([1.0, 2.0])
    y = np.array([5.0, 6.0])
    out = resample_spectrum(mz, y, np.array([0.0, 3.0]), method="linear")
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_resample_empty_input_returns_zeros():
    out = resample_spectrum([], [], np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_resample_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        resample_spectrum([1.0, 2.0], [1.0], np.array([1.5]))


def test_resample_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown resample_method"):
        resample_spectrum([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], np.array([1.5]),
                          method="bogus")


@pytest.mark.parametrize("method", ["pchip", "akima", "cubic"])
def test_resample_interpolators_run(method):
    mz = np.linspace(0, 10, 50)
    y = np.sin(mz) + 2
    out = resample_spectrum(mz, y, np.linspace(1, 9, 20), method=method)
    assert out.shape == (20,)
    assert np.all(out >= 0)


def test_resample_collapses_duplicate_mz():
    mz = np.array([1.0, 1.0, 2.0])
    y = np.array([5.0, 99.0, 10.0])
    out = resample_spectrum(mz, y, np.array([1.0]), method="linear")
    assert np.isclose(out[0], 5.0)


# --------------------------- data_preprocessing ---------------------------

def test_data_preprocessing_returns_tuple(spectrum_file):
    sample_name, group, mz, y = data_preprocessing(
        spectrum_file, normalization_target=1e6, verbose=False
    )
    assert group == "Control"
    assert np.isclose(y.sum(), 1e6)


def test_data_preprocessing_no_norm(spectrum_file):
    _, _, _, y = data_preprocessing(
        spectrum_file, normalization_target=None, verbose=False
    )
    assert y.sum() > 0


def test_data_preprocessing_return_all(spectrum_file):
    out = data_preprocessing(
        spectrum_file, normalization_target=1e6, verbose=False, return_all=True
    )
    assert len(out) == 5


def test_data_preprocessing_missing_file():
    with pytest.raises(FileNotFoundError):
        data_preprocessing("/no/such/file.txt", verbose=False)


# --------------------------- batch_tic_norm ---------------------------

def test_batch_tic_norm_writes_files(spectrum_files, tmp_path):
    pattern = str(tmp_path / "*.txt")
    outdir = tmp_path / "normed"
    written = batch_tic_norm(pattern, output_dir=str(outdir), normalization_target=1e6)
    assert len(written) == len(spectrum_files)
    for w in written:
        assert w.endswith("_normalized.txt")


def test_batch_tic_norm_no_match_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        batch_tic_norm(str(tmp_path / "nope_*.txt"))


# --------------------------- tic_count ---------------------------

def test_normalization_target_builds_dataframe(spectrum_files):
    df = normalization_target(spectrum_files)
    assert set(["SampleName", "Group", "TIC-Million"]).issubset(df.columns)
    assert len(df) == len(spectrum_files)
