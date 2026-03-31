import os

import numpy as np
import pandas as pd
import pytest

from . import flexible_calibrator as flexible_calibrator_module
from ._models import (
    _enhanced_bootstrap_channels,
    _enhanced_pick_channels,
    _fit_quad_sqrt_robust,
    _invert_quad_sqrt,
    _quad_sqrt_forward,
    _validate_quad_sqrt_params,
)
from .flexible_calibrator import (
    FlexibleCalibConfig,
    FlexibleCalibrator,
    select_stable_reference_masses,
    summarize_reference_mass_stability,
)


def test_invert_quad_sqrt_returns_nan_when_no_real_root_exists():
    masses = _invert_quad_sqrt(np.array([30.0]), 10.0, -1.0, 0.0)

    assert np.isnan(masses[0])


def test_invert_quad_sqrt_remains_stable_for_tiny_linear_term():
    true_masses = np.array([100.0, 400.0, 900.0])
    k, c, t0 = 1000.0, 1e-13, 25.0
    tof = _quad_sqrt_forward(true_masses, k, c, t0)

    recovered = _invert_quad_sqrt(tof, k, c, t0)

    np.testing.assert_allclose(recovered, true_masses, rtol=0.0, atol=1e-6)


def test_validate_quad_sqrt_rejects_non_monotone_model():
    ok, reason = _validate_quad_sqrt_params(
        10.0,
        -2.0,
        50.0,
        np.array([1.0, 4.0, 9.0, 16.0]),
    )

    assert ok is False
    assert "monotone" in reason


def test_fit_quad_sqrt_robust_rejects_descending_time_axis():
    params = _fit_quad_sqrt_robust(
        np.array([1.0, 4.0, 9.0, 16.0]),
        np.array([100.0, 90.0, 80.0, 70.0]),
        max_iterations=1,
    )

    assert params is None


def test_fit_quad_sqrt_robust_accepts_valid_monotone_data():
    true_masses = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    true_params = (75.0, 0.02, 5.0)
    tof = _quad_sqrt_forward(true_masses, *true_params)

    params = _fit_quad_sqrt_robust(true_masses, tof, max_iterations=1)

    assert params is not None
    recovered = _invert_quad_sqrt(tof, *params)
    np.testing.assert_allclose(recovered, true_masses, rtol=0.0, atol=1e-6)


def test_validate_quad_sqrt_does_not_extrapolate_hydrogen_range_to_zero_mass():
    masses = np.array([
        1.0072764666,
        15.0229265168,
        22.9892207021,
        27.0229265168,
        29.0385765812,
        38.9631579065,
        41.0385765812,
        43.0542266457,
        57.0698767102,
        58.065674,
        67.0548,
        71.0855267746,
        86.096976,
        91.0542266457,
        104.107539,
        184.073320,
        224.105171,
        369.351600,
    ])
    params = (33189.9723, -0.0845105, -651.183862)

    ok, reason = _validate_quad_sqrt_params(*params, masses)

    assert ok is True
    assert reason == ""


def test_fit_quad_sqrt_robust_accepts_real_mz_autodetected_channels():
    masses = np.array([
        1.0072764666,
        15.0229265168,
        22.9892207021,
        27.0229265168,
        29.0385765812,
        38.9631579065,
        41.0385765812,
        43.0542266457,
        57.0698767102,
        58.065674,
        67.0548,
        71.0855267746,
        86.096976,
        91.0542266457,
        104.107539,
        184.073320,
        224.105171,
        369.351600,
    ])
    df = pd.read_csv("data/breast/control/breast_CT-11b_3.txt", sep="\t", comment="#")
    channels, methods = _enhanced_pick_channels(
        df,
        masses,
        tol_da=0.5,
        tol_ppm=None,
        method="max",
        fallback_policy="max",
        return_details=True,
    )

    assert methods == ["max"] * len(masses)

    params = _fit_quad_sqrt_robust(masses, np.asarray(channels, dtype=float))

    assert params is not None


def test_enhanced_bootstrap_channels_recovers_real_file_channels_from_channel_only_data():
    masses = np.array([
        1.0072764666,
        15.0229265168,
        22.9892207021,
        27.0229265168,
        29.0385765812,
        38.9631579065,
        41.0385765812,
        43.0542266457,
        57.0698767102,
        58.065674,
        67.0548,
        71.0855267746,
        86.096976,
        91.0542266457,
        104.107539,
        184.073320,
        224.105171,
        369.351600,
    ])
    df = pd.read_csv("data/breast/control/breast_CT-11b_3.txt", sep="\t", comment="#")

    bootstrap_channels = np.asarray(
        _enhanced_bootstrap_channels(
            df["Channel"].to_numpy(),
            df["Intensity"].astype("float64").to_numpy(),
            masses,
        ),
        dtype=float,
    )
    mz_channels, _ = _enhanced_pick_channels(
        df,
        masses,
        tol_da=0.5,
        tol_ppm=None,
        method="max",
        fallback_policy="max",
        return_details=True,
    )
    mz_channels = np.asarray(mz_channels, dtype=float)

    assert np.isfinite(bootstrap_channels).all()
    np.testing.assert_allclose(bootstrap_channels, mz_channels, rtol=0.0, atol=150.0)


def test_flexible_calibrator_quad_sqrt_succeeds_on_real_file_with_mz_max():
    masses = [
        1.0072764666,
        15.0229265168,
        22.9892207021,
        27.0229265168,
        29.0385765812,
        38.9631579065,
        41.0385765812,
        43.0542266457,
        57.0698767102,
        58.065674,
        67.0548,
        71.0855267746,
        86.096976,
        91.0542266457,
        104.107539,
        184.073320,
        224.105171,
        369.351600,
    ]
    config = FlexibleCalibConfig(
        reference_masses=masses,
        calibration_method="quad_sqrt",
        autodetect_method="max",
        autodetect_fallback_policy="max",
        autodetect_tol_da=0.5,
        autodetect_strategy="mz",
        max_workers=1,
        max_ppm_threshold=None,
        fail_on_high_error=False,
        verbose=False,
    )

    calibrator = FlexibleCalibrator(config)
    summary = calibrator.calibrate(["data/breast/control/breast_CT-11b_3.txt"])

    assert len(summary) == 1
    assert np.isfinite(summary.iloc[0]["ppm_error"])


def test_flexible_calibrator_quad_sqrt_bootstrap_matches_mz_summary_on_real_file():
    masses = [
        1.0072764666,
        15.0229265168,
        22.9892207021,
        27.0229265168,
        29.0385765812,
        38.9631579065,
        41.0385765812,
        43.0542266457,
        57.0698767102,
        58.065674,
        67.0548,
        71.0855267746,
        86.096976,
        91.0542266457,
        104.107539,
        184.073320,
        224.105171,
        369.351600,
    ]
    common_kwargs = dict(
        reference_masses=masses,
        calibration_method="quad_sqrt",
        autodetect_method="max",
        autodetect_fallback_policy="max",
        autodetect_tol_da=0.5,
        max_workers=1,
        max_ppm_threshold=None,
        fail_on_high_error=False,
        verbose=False,
    )

    bootstrap_summary = FlexibleCalibrator(
        FlexibleCalibConfig(
            autodetect_strategy="bootstrap",
            prefer_recompute_from_channel=True,
            **common_kwargs,
        )
    ).calibrate(["data/breast/control/breast_CT-11b_3.txt"])
    mz_summary = FlexibleCalibrator(
        FlexibleCalibConfig(
            autodetect_strategy="mz",
            prefer_recompute_from_channel=False,
            **common_kwargs,
        )
    ).calibrate(["data/breast/control/breast_CT-11b_3.txt"])

    assert np.isfinite(bootstrap_summary.iloc[0]["ppm_error"])
    assert np.isfinite(mz_summary.iloc[0]["ppm_error"])
    assert abs(bootstrap_summary.iloc[0]["ppm_error"] - mz_summary.iloc[0]["ppm_error"]) < 1.0


def test_reference_mass_screening_selects_stable_subset():
    summary = pd.DataFrame(
        [
            {
                "file_name": "a.txt",
                "calibrant_masses": [1.0072764666, 71.0855267746, 86.096976, 104.107539],
                "estimated_masses": [
                    1.0072764666,
                    71.0855267746 * (1.0 - 120e-6),
                    86.096976 * (1.0 + 70e-6),
                    104.107539 * (1.0 + 4e-6),
                ],
            },
            {
                "file_name": "b.txt",
                "calibrant_masses": [1.0072764666, 71.0855267746, 86.096976, 104.107539],
                "estimated_masses": [
                    1.0072764666,
                    71.0855267746 * (1.0 - 110e-6),
                    86.096976 * (1.0 + 60e-6),
                    104.107539 * (1.0 - 5e-6),
                ],
            },
        ]
    )

    stability = summarize_reference_mass_stability(summary, exclude_below_mz=1.5)
    kept, annotated = select_stable_reference_masses(
        [1.0072764666, 71.0855267746, 86.096976, 104.107539],
        stability,
        exclude_below_mz=1.5,
        max_mean_abs_ppm=50.0,
        min_valid_fraction=1.0,
        min_count=2,
    )

    np.testing.assert_allclose(kept, np.array([1.0072764666, 104.107539]))
    selected = dict(zip(annotated["mass"], annotated["selected"]))
    assert selected[71.0855267746] is False
    assert selected[86.096976] is False
    assert selected[104.107539] is True


def test_flexible_calibrator_two_pass_screening_refits_with_reduced_mass_set(
    monkeypatch,
    tmp_path,
):
    masses = [1.0072764666, 41.0385765812, 71.0855267746, 104.107539]
    fp = tmp_path / "screen.tsv"
    pd.DataFrame(
        {
            "Channel": [10.0, 20.0, 30.0],
            "m/z": [1.0, 2.0, 3.0],
            "Intensity": [100.0, 200.0, 150.0],
        }
    ).to_csv(fp, sep="\t", index=False)

    fit_history = []

    def fake_fit(filename, ref_masses, calib_channels_dict, config):
        ref_masses = np.asarray(ref_masses, dtype=float)
        fit_history.append(ref_masses.tolist())
        t_meas = np.asarray(calib_channels_dict[filename], dtype=float)
        estimated = ref_masses.copy()
        for i, mass in enumerate(ref_masses):
            if np.isclose(mass, 71.0855267746, atol=1e-6):
                estimated[i] = mass * (1.0 - 120e-6)
            else:
                estimated[i] = mass * (1.0 + 5e-6)
        ppm = float(np.median(np.abs((estimated - ref_masses) / ref_masses * 1e6)))
        return filename, {
            "method": config.calibration_method,
            "params": {"dummy": True},
            "ppm": ppm,
            "n_calibrants": len(ref_masses),
            "calibrant_masses": ref_masses.tolist(),
            "calibrant_channels": t_meas.tolist(),
            "estimated_masses": estimated.tolist(),
        }

    def fake_apply(args):
        return os.path.basename(args[0])

    monkeypatch.setattr(flexible_calibrator_module, "_fit_selected_model_enhanced", fake_fit)
    monkeypatch.setattr(flexible_calibrator_module, "_apply_model_to_file_enhanced", fake_apply)

    config = FlexibleCalibConfig(
        reference_masses=masses,
        calibration_method="quad_sqrt",
        max_workers=1,
        max_ppm_threshold=None,
        fail_on_high_error=False,
        verbose=False,
        auto_screen_reference_masses=True,
        screen_max_mean_abs_ppm=50.0,
        screen_min_valid_fraction=1.0,
        screen_min_count=1,
    )
    calibrator = FlexibleCalibrator(config)

    summary = calibrator.calibrate(
        [os.fspath(fp)],
        calib_channels_dict={fp.name: [1000.0, 2000.0, 3000.0, 4000.0]},
    )

    assert fit_history == [
        masses,
        [1.0072764666, 41.0385765812, 104.107539],
    ]
    assert calibrator.last_reference_masses_screened_out == [71.0855267746]
    assert calibrator.last_reference_masses_used == [1.0072764666, 41.0385765812, 104.107539]
    assert calibrator.last_reference_mass_screening["selected"].tolist() == [True, False, True]
    assert summary.iloc[0]["calibrant_masses"] == [1.0072764666, 41.0385765812, 104.107539]


def test_refined_methods_return_fractional_channels_on_synthetic_peak():
    channels = np.arange(100, 201, dtype=float)
    mz = 50.0 + 0.002 * (channels - channels[0])
    true_channel = 150.4
    true_mz = 50.0 + 0.002 * (true_channel - channels[0])
    intensity = 1000.0 * np.exp(-0.5 * ((channels - true_channel) / 4.0) ** 2)
    df = pd.DataFrame({"m/z": mz, "Intensity": intensity, "Channel": channels})

    max_channels, _ = _enhanced_pick_channels(
        df,
        np.array([true_mz]),
        tol_da=0.05,
        tol_ppm=None,
        method="max",
        fallback_policy="max",
        return_details=True,
    )
    max_channel = max_channels[0]

    for method in ["centroid", "parabolic", "gaussian", "voigt"]:
        refined_channels, methods_used = _enhanced_pick_channels(
            df,
            np.array([true_mz]),
            tol_da=0.05,
            tol_ppm=None,
            method=method,
            fallback_policy="raise",
            return_details=True,
        )

        refined_channel = refined_channels[0]
        assert methods_used == [method]
        assert np.isfinite(refined_channel)
        assert abs(refined_channel - true_channel) < abs(max_channel - true_channel)
        assert not np.isclose(refined_channel, round(refined_channel))


def test_centroid_uses_local_peak_support_instead_of_whole_window_tail():
    channels = np.arange(0, 241, dtype=float)
    mz = 100.0 + 0.002 * channels
    main_peak = 100.0 * np.exp(-0.5 * ((channels - 80.4) / 3.0) ** 2)
    secondary_peak = 30.0 * np.exp(-0.5 * ((channels - 120.0) / 4.5) ** 2)
    intensity = main_peak + secondary_peak + 2.0
    df = pd.DataFrame({"m/z": mz, "Intensity": intensity, "Channel": channels})

    centroid_channels, methods_used = _enhanced_pick_channels(
        df,
        np.array([mz[80]]),
        tol_da=0.12,
        tol_ppm=None,
        method="centroid",
        fallback_policy="raise",
        return_details=True,
    )

    assert methods_used == ["centroid"]
    assert abs(centroid_channels[0] - 80.4) < 1.0


def test_centroid_beats_centroid_raw_on_asymmetric_tail():
    channels = np.arange(0, 241, dtype=float)
    mz = 100.0 + 0.002 * channels
    true_channel = 80.4
    main_peak = 120.0 * np.exp(-0.5 * ((channels - true_channel) / 3.0) ** 2)
    right_tail = 25.0 * np.exp(-(channels - true_channel) / 10.0)
    right_tail[channels < true_channel] = 0.0
    intensity = main_peak + right_tail + 12.0
    df = pd.DataFrame({"m/z": mz, "Intensity": intensity, "Channel": channels})

    centroid_channels, centroid_methods = _enhanced_pick_channels(
        df,
        np.array([100.0 + 0.002 * true_channel]),
        tol_da=0.12,
        tol_ppm=None,
        method="centroid",
        fallback_policy="raise",
        return_details=True,
    )
    raw_channels, raw_methods = _enhanced_pick_channels(
        df,
        np.array([100.0 + 0.002 * true_channel]),
        tol_da=0.12,
        tol_ppm=None,
        method="centroid_raw",
        fallback_policy="raise",
        return_details=True,
    )

    assert centroid_methods == ["centroid"]
    assert raw_methods == ["centroid_raw"]
    assert abs(centroid_channels[0] - true_channel) < abs(raw_channels[0] - true_channel)
    assert abs(centroid_channels[0] - true_channel) < 1.0


def test_voigt_does_not_accept_large_tailward_jump_on_real_spectrum():
    masses = np.array([15.0229265168])
    df = pd.read_csv("data/breast/control/breast_CT-11b_3.txt", sep="\t", comment="#")

    max_channels, _ = _enhanced_pick_channels(
        df,
        masses,
        tol_da=0.5,
        tol_ppm=None,
        method="max",
        fallback_policy="max",
        return_details=True,
    )
    voigt_channels, methods_used = _enhanced_pick_channels(
        df,
        masses,
        tol_da=0.5,
        tol_ppm=None,
        method="voigt",
        fallback_policy="max",
        return_details=True,
    )

    assert abs(voigt_channels[0] - max_channels[0]) < 20.0
    assert methods_used[0] in {"voigt", "gaussian_fallback", "max_fallback"}
