"""Tests for mioXpektron.utils.file_management."""

import numpy as np
import polars as pl
import pytest

from mioXpektron.utils.file_management import (
    DEFAULT_GROUP_PATTERNS,
    INTENSITY_COLUMN_ALIASES,
    MZ_COLUMN_ALIASES,
    _resolve_column,
    _resolve_group,
    import_data,
)

from conftest import make_spectrum, write_spectrum_txt


class TestResolveGroup:
    def test_default_patterns_detect_control(self):
        assert _resolve_group("breast_CT-1a_1", None, None) == "Control"

    def test_default_patterns_detect_cancer(self):
        assert _resolve_group("breast_CC-2b_3", None, None) == "Cancer"

    def test_unmatched_sample_is_unknown(self):
        assert _resolve_group("random_sample", None, None) == "Unknown"

    def test_case_insensitive_match(self):
        assert _resolve_group("breast_ct-1a_1", None, None) == "Control"

    def test_custom_patterns_take_precedence(self):
        patterns = {"tumor": "Tumor", "normal": "Normal"}
        assert _resolve_group("tumor_5", patterns, None) == "Tumor"
        assert _resolve_group("normal_5", patterns, None) == "Normal"

    def test_group_fn_overrides_patterns(self):
        result = _resolve_group("breast_CT-1a_1", None, lambda s: "Custom")
        assert result == "Custom"


class TestResolveColumn:
    def test_exact_match(self):
        df = pl.DataFrame({"m/z": [1.0], "Intensity": [2.0]})
        assert _resolve_column(df, MZ_COLUMN_ALIASES) == "m/z"

    def test_case_insensitive_alias(self):
        df = pl.DataFrame({"MZ": [1.0]})
        assert _resolve_column(df, MZ_COLUMN_ALIASES) == "MZ"

    def test_intensity_aliases(self):
        df = pl.DataFrame({"corrected_intensity": [1.0]})
        assert _resolve_column(df, INTENSITY_COLUMN_ALIASES) == "corrected_intensity"

    def test_missing_column_raises(self):
        df = pl.DataFrame({"foo": [1.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            _resolve_column(df, MZ_COLUMN_ALIASES)


class TestImportData:
    def test_reads_txt_spectrum(self, tmp_path):
        mz, y = make_spectrum(n=500)
        path = write_spectrum_txt(tmp_path / "breast_CT-1a_1.txt", mz, y)
        out_mz, out_int, name, group = import_data(path)
        assert out_mz.shape == out_int.shape == (500,)
        assert name == "breast_CT-1a_1"
        assert group == "Control"

    def test_mz_filtering_inclusive(self, tmp_path):
        mz, y = make_spectrum(n=500)
        path = write_spectrum_txt(tmp_path / "breast_CC-1a_1.txt", mz, y)
        out_mz, _, _, group = import_data(path, mz_min=20.0, mz_max=50.0)
        assert out_mz.min() >= 20.0
        assert out_mz.max() <= 50.0
        assert group == "Cancer"

    def test_empty_after_filter_raises(self, tmp_path):
        mz, y = make_spectrum(n=200)
        path = write_spectrum_txt(tmp_path / "breast_CT-1a_1.txt", mz, y)
        with pytest.raises(ValueError, match="No data"):
            import_data(path, mz_min=100000.0)

    def test_csv_with_corrected_intensity(self, tmp_path):
        p = tmp_path / "breast_CC-1a_1.csv"
        p.write_text("mz,corrected_intensity\n5.0,1.0\n6.0,2.0\n7.0,3.0\n")
        mz, intensity, _, group = import_data(str(p))
        np.testing.assert_allclose(mz, [5.0, 6.0, 7.0])
        np.testing.assert_allclose(intensity, [1.0, 2.0, 3.0])
        assert group == "Cancer"

    def test_group_fn_applied(self, tmp_path):
        mz, y = make_spectrum(n=100)
        path = write_spectrum_txt(tmp_path / "anything_1.txt", mz, y)
        _, _, _, group = import_data(path, group_fn=lambda s: "MyGroup")
        assert group == "MyGroup"


def test_default_group_patterns_structure():
    assert isinstance(DEFAULT_GROUP_PATTERNS, dict)
    assert set(DEFAULT_GROUP_PATTERNS.values()) == {"Cancer", "Control"}
