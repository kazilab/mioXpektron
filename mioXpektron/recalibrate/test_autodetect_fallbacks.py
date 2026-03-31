import os

import numpy as np
import pandas as pd
import pytest

from . import auto_calibrator, flexible_calibrator
from ._models import _enhanced_pick_channels


def _fallback_spectrum():
    return pd.DataFrame(
        {
            "m/z": [99.9, 100.0, 100.1],
            "Intensity": [1.0, 10.0, 20.0],
            "Channel": [10, 11, 12],
        }
    )


def test_enhanced_pick_channels_defaults_to_max_style_fallback():
    channels, methods_used = _enhanced_pick_channels(
        _fallback_spectrum(),
        np.array([100.0]),
        tol_da=None,
        tol_ppm=2000.0,
        method="gaussian",
        return_details=True,
    )

    assert np.isfinite(channels[0])
    assert 11.0 < channels[0] < 12.0
    assert methods_used == ["centroid_fallback"]


def test_enhanced_pick_channels_can_return_nan_instead_of_fallback():
    channels, methods_used = _enhanced_pick_channels(
        _fallback_spectrum(),
        np.array([100.0]),
        tol_da=None,
        tol_ppm=2000.0,
        method="gaussian",
        fallback_policy="nan",
        return_details=True,
    )

    assert np.isnan(channels[0])
    assert methods_used == ["gaussian_failed"]


def test_enhanced_pick_channels_can_raise_on_failed_refinement():
    with pytest.raises(RuntimeError, match="Peak-picking method 'gaussian' failed"):
        _enhanced_pick_channels(
            _fallback_spectrum(),
            np.array([100.0]),
            tol_da=None,
            tol_ppm=2000.0,
            method="gaussian",
            fallback_policy="raise",
        )


def test_auto_calibrator_forwards_fallback_policy_and_records_methods(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_pick(df, targets, tol_da, tol_ppm, method, fallback_policy="max", return_details=False):
        captured["fallback_policy"] = fallback_policy
        captured["return_details"] = return_details
        return [101] * len(targets), ["gaussian"] * len(targets)

    monkeypatch.setattr(auto_calibrator, "_enhanced_pick_channels", fake_pick)

    fp = tmp_path / "auto.tsv"
    pd.DataFrame(
        {
            "m/z": [10.0, 20.0, 30.0],
            "Intensity": [1.0, 2.0, 3.0],
            "Channel": [100, 101, 102],
        }
    ).to_csv(fp, sep="\t", index=False)

    config = auto_calibrator.AutoCalibConfig(
        reference_masses=[10.0, 20.0, 30.0],
        autodetect_method="gaussian",
        autodetect_fallback_policy="nan",
    )
    calibrator = auto_calibrator.AutoCalibrator(config)

    autodetected = calibrator._autodetect_channels([os.fspath(fp)], np.array(config.reference_masses))

    assert captured == {"fallback_policy": "nan", "return_details": True}
    assert autodetected[fp.name] == [101, 101, 101]
    assert calibrator.last_autodetect_methods[fp.name] == ["gaussian", "gaussian", "gaussian"]


def test_flexible_calibrator_forwards_fallback_policy_and_records_methods(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_pick(df, targets, tol_da, tol_ppm, method, fallback_policy="max", return_details=False):
        captured["fallback_policy"] = fallback_policy
        captured["return_details"] = return_details
        return [201] * len(targets), ["max_fallback"] * len(targets)

    monkeypatch.setattr(flexible_calibrator, "_enhanced_pick_channels", fake_pick)

    fp = tmp_path / "flex.tsv"
    pd.DataFrame(
        {
            "m/z": [10.0, 20.0, 30.0],
            "Intensity": [1.0, 2.0, 3.0],
            "Channel": [200, 201, 202],
        }
    ).to_csv(fp, sep="\t", index=False)

    config = flexible_calibrator.FlexibleCalibConfig(
        reference_masses=[10.0, 20.0, 30.0],
        calibration_method="quad_sqrt",
        autodetect_method="gaussian",
        autodetect_fallback_policy="max",
    )
    calibrator = flexible_calibrator.FlexibleCalibrator(config)

    autodetected = calibrator._autodetect_channels([os.fspath(fp)], np.array(config.reference_masses))

    assert captured == {"fallback_policy": "max", "return_details": True}
    assert autodetected[fp.name] == [201, 201, 201]
    assert calibrator.last_autodetect_methods[fp.name] == [
        "max_fallback",
        "max_fallback",
        "max_fallback",
    ]


def test_invalid_fallback_policy_is_rejected():
    with pytest.raises(ValueError, match="autodetect_fallback_policy"):
        auto_calibrator.AutoCalibConfig(
            reference_masses=[10.0, 20.0, 30.0],
            autodetect_fallback_policy="sometimes",
        )

    with pytest.raises(ValueError, match="autodetect_fallback_policy"):
        flexible_calibrator.FlexibleCalibConfig(
            reference_masses=[10.0, 20.0, 30.0],
            autodetect_fallback_policy="sometimes",
        )
