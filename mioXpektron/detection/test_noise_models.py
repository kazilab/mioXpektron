import numpy as np
import pandas as pd

from . import detection as detection_module
from .detection import (
    PeakAlignIntensityArea,
    detect_peaks_with_area,
    detect_peaks_with_area_v2,
    robust_noise_estimation,
    robust_noise_estimation_mz_dependent,
)


def test_robust_noise_estimation_mz_dependent_tracks_local_noise_levels():
    mz = np.linspace(1.0, 100.0, 400)
    low_noise = np.tile([1.0, 2.0, 1.0, 2.0], 50)
    high_noise = np.tile([1.0, 5.0, 1.0, 5.0], 50)
    intensities = np.concatenate([low_noise, high_noise]).astype(float)

    median_profile, std_profile = robust_noise_estimation_mz_dependent(
        mz,
        intensities,
        peak_indices=np.array([], dtype=int),
        n_bins=4,
        min_points_per_bin=20,
    )

    assert median_profile.shape == mz.shape
    assert std_profile.shape == mz.shape
    assert std_profile[300] > std_profile[50]


def test_detect_peaks_with_area_supports_mz_binned_thresholds(monkeypatch):
    monkeypatch.setattr(
        detection_module,
        "robust_noise_estimation_mz_dependent",
        lambda *args, **kwargs: (
            np.zeros(6, dtype=float),
            np.array([2.0, 2.0, 2.0, 10.0, 10.0, 10.0], dtype=float),
        ),
    )

    df = detect_peaks_with_area(
        mz_values=np.arange(6.0),
        intensities=np.array([0.0, 5.0, 0.0, 0.0, 5.0, 0.0]),
        sample_name="sample",
        group="g",
        min_intensity=0.0,
        min_snr=1.0,
        min_distance=1,
        window_size=1,
        peak_height=None,
        prominence=0.1,
        min_peak_width=1,
        max_peak_width=10,
        noise_model="mz_binned",
    )

    assert df["PeakCenter"].tolist() == [1.0]


def test_detect_peaks_with_area_v2_supports_mz_binned_thresholds(monkeypatch):
    monkeypatch.setattr(
        detection_module,
        "robust_noise_estimation_mz_dependent",
        lambda *args, **kwargs: (
            np.zeros(6, dtype=float),
            np.array([2.0, 2.0, 2.0, 10.0, 10.0, 10.0], dtype=float),
        ),
    )

    df = detect_peaks_with_area_v2(
        mz=np.arange(6.0),
        intens=np.array([1.0, 5.0, 1.0, 1.0, 5.0, 1.0]),
        sample_name="sample",
        group="g",
        min_intensity=0.0,
        min_snr=1.0,
        min_distance=1,
        prominence=0.1,
        min_peak_width=1,
        max_peak_width=10,
        noise_model="mz_binned",
    )

    assert df["PeakCenter"].tolist() == [1.0]


def test_detect_peaks_with_area_v2_uses_shared_height_threshold_helper(monkeypatch):
    calls = {}

    def fake_height_threshold(*args, **kwargs):
        calls["noise_model"] = kwargs["noise_model"]
        calls["window"] = kwargs["window"]
        return 0.0, 1.0, np.array([2.0, 2.0, 2.0, 10.0, 10.0, 10.0], dtype=float)

    monkeypatch.setattr(detection_module, "_height_threshold_from_noise_model", fake_height_threshold)

    df = detect_peaks_with_area_v2(
        mz=np.arange(6.0),
        intens=np.array([1.0, 5.0, 1.0, 1.0, 5.0, 1.0]),
        sample_name="sample",
        group="g",
        min_intensity=0.0,
        min_snr=1.0,
        min_distance=1,
        prominence=0.1,
        min_peak_width=1,
        max_peak_width=10,
        noise_model="mz_binned",
        noise_window=7,
    )

    assert calls == {
        "noise_model": "mz_binned",
        "window": 7,
    }
    assert df["PeakCenter"].tolist() == [1.0]


def test_robust_noise_estimation_excludes_full_peak_width_not_only_center_window():
    x = np.arange(201, dtype=float)
    wide_peak = 200.0 * np.exp(-0.5 * ((x - 100.0) / 12.0) ** 2)
    intensities = 1.0 + wide_peak

    narrow_mask = detection_module._noise_exclusion_mask(
        intensities,
        peak_indices=np.array([100]),
        window=1,
    )
    excluded = np.flatnonzero(~narrow_mask)

    assert excluded[0] < 95
    assert excluded[-1] > 105

    median_noise, std_noise = robust_noise_estimation(
        intensities,
        peak_indices=np.array([100]),
        window=1,
    )

    assert median_noise < 1.5
    assert std_noise < 1.0


def test_peak_align_intensity_area_forwards_noise_model(tmp_path, monkeypatch):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "mz": [1.0, 2.0, 3.0],
            "intensity": [0.0, 5.0, 0.0],
        }
    ).to_csv(csv_path, index=False)

    calls = {}

    def fake_detect_peaks_with_area_v2(**kwargs):
        calls["noise_model"] = kwargs["noise_model"]
        calls["noise_bins"] = kwargs["noise_bins"]
        calls["noise_min_points"] = kwargs["noise_min_points"]
        return detection_module._empty_peak_properties_df()

    monkeypatch.setattr(detection_module, "detect_peaks_with_area_v2", fake_detect_peaks_with_area_v2)

    analyzer = PeakAlignIntensityArea(
        method=None,
        noise_model="mz_binned",
        noise_bins=7,
        noise_min_points=9,
    )
    intensity_table, area_table, peaks_df = analyzer.run([str(csv_path)])

    assert calls == {
        "noise_model": "mz_binned",
        "noise_bins": 7,
        "noise_min_points": 9,
    }
    assert intensity_table.empty
    assert area_table.empty
    assert peaks_df.empty


def test_peak_align_intensity_area_forwards_deconvolution_controls(tmp_path, monkeypatch):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "mz": [1.0, 2.0, 3.0],
            "intensity": [0.0, 5.0, 0.0],
        }
    ).to_csv(csv_path, index=False)

    calls = {}

    def fake_robust_peak_detection(**kwargs):
        calls["deconvolution_min_bic_delta"] = kwargs["deconvolution_min_bic_delta"]
        calls["deconvolution_overlap_factor"] = kwargs["deconvolution_overlap_factor"]
        calls["deconvolution_replace_singles"] = kwargs["deconvolution_replace_singles"]
        return detection_module._empty_peak_properties_df()

    monkeypatch.setattr(detection_module, "robust_peak_detection", fake_robust_peak_detection)

    analyzer = PeakAlignIntensityArea(
        method="Gaussian",
        deconvolution_min_bic_delta=12.0,
        deconvolution_overlap_factor=0.9,
        deconvolution_replace_singles=False,
    )
    intensity_table, area_table, peaks_df = analyzer.run([str(csv_path)])

    assert calls == {
        "deconvolution_min_bic_delta": 12.0,
        "deconvolution_overlap_factor": 0.9,
        "deconvolution_replace_singles": False,
    }
    assert intensity_table.empty
    assert area_table.empty
    assert peaks_df.empty
