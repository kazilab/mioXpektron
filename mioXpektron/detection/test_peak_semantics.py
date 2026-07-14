import numpy as np
import pandas as pd
import pytest

from . import detection as detection_module
from .detection import (
    align_peaks,
    collect_peak_properties_batch,
    detect_peaks_with_area_v2,
    robust_peak_detection,
)


EXPECTED_PEAK_COLUMNS = [
    "PeakCenter",
    "PeakWidth",
    "Prominences",
    "Amplitude",
    "PeakArea",
    "SampleName",
    "Group",
    "DetectedBy",
    "Deconvoluted",
    "FitModel",
    "AreaDefinition",
    "WidthDefinition",
    "IntegrationMethod",
]


def test_align_peaks_logs_mixed_area_definitions_warning(caplog):
    peaks_df = pd.DataFrame(
        [
            {
                "SampleName": "s1",
                "Group": "g1",
                "PeakCenter": 100.0,
                "PeakArea": 10.0,
                "Amplitude": 5.0,
                "AreaDefinition": "raw_trapezoid",
            },
            {
                "SampleName": "s2",
                "Group": "g1",
                "PeakCenter": 100.05,
                "PeakArea": 12.0,
                "Amplitude": 6.0,
                "AreaDefinition": "analytic_fit",
            },
        ]
    )

    aligned = align_peaks(peaks_df, output="area")

    assert not aligned.empty
    assert "mixed AreaDefinition" in caplog.text


def test_detect_peaks_with_area_v2_empty_schema_includes_semantic_columns():
    mz = np.array([1.0, 2.0, 3.0])
    intensities = np.zeros_like(mz)

    df = detect_peaks_with_area_v2(mz, intensities, "sample", "Unknown")

    assert df.empty
    assert list(df.columns) == EXPECTED_PEAK_COLUMNS


def test_robust_peak_detection_combined_mode_returns_semantic_columns():
    mz = np.linspace(0.0, 1.0, 401)
    sigma = 0.02
    intensities = 100.0 * np.exp(-((mz - 0.5) ** 2) / (2.0 * sigma**2))

    df = robust_peak_detection(
        mz_values=mz,
        intensities=intensities,
        sample_name="sample",
        group="Unknown",
        method="Gaussian",
        min_intensity=0,
        min_snr=0,
        min_distance=1,
        window_size=30,
        peak_height=1,
        prominence=1,
        min_peak_width=1,
        max_peak_width=200,
        combined=True,
    )

    assert set(EXPECTED_PEAK_COLUMNS).issubset(df.columns)
    if not df.empty:
        assert set(df["AreaDefinition"].dropna()) == {"analytic_fit"}
        assert set(df["WidthDefinition"].dropna()) == {"FWHM"}


def test_collect_peak_properties_batch_preserves_semantic_metadata(monkeypatch):
    monkeypatch.setattr(
        detection_module,
        "import_data",
        lambda file_path, mz_min, mz_max: (
            np.array([1.0, 2.0, 3.0]),
            np.array([0.0, 5.0, 0.0]),
            "sample",
            "Unknown",
        ),
    )
    monkeypatch.setattr(detection_module, "baseline_correction", lambda intensity, method: intensity)
    monkeypatch.setattr(detection_module, "noise_filtering", lambda intensity, method: intensity)
    monkeypatch.setattr(
        detection_module,
        "handle_missing_values",
        lambda mz_values, intensities, method: (mz_values, intensities),
    )
    monkeypatch.setattr(
        detection_module,
        "tic_normalization",
        lambda intensities, target_tic: intensities,
    )

    calls = {}

    def fake_robust_peak_detection(*, combined=False, **kwargs):
        calls["combined"] = combined
        return pd.DataFrame(
            [
                {
                    "SampleName": "sample",
                    "Group": "Unknown",
                    "PeakCenter": 2.0,
                    "PeakWidth": 0.1,
                    "Prominences": np.nan,
                    "PeakArea": 42.0,
                    "Amplitude": 5.0,
                    "DetectedBy": "local_max",
                    "Deconvoluted": False,
                    "FitModel": "Gaussian",
                    "AreaDefinition": "analytic_fit",
                    "WidthDefinition": "FWHM",
                    "IntegrationMethod": "analytic",
                }
            ]
        )

    monkeypatch.setattr(detection_module, "robust_peak_detection", fake_robust_peak_detection)

    df = collect_peak_properties_batch(["dummy.txt"], method="Gaussian", combined=True)

    assert calls["combined"] is True
    assert list(df.columns) == [
        "SampleName",
        "Group",
        "PeakCenter",
        "PeakWidth",
        "Prominences",
        "PeakArea",
        "Amplitude",
        "DetectedBy",
        "Deconvoluted",
        "FitModel",
        "AreaDefinition",
        "WidthDefinition",
        "IntegrationMethod",
    ]
    assert df.loc[0, "AreaDefinition"] == "analytic_fit"


def test_robust_peak_detection_rejects_deconvolution_without_bic_support(monkeypatch):
    mz = np.linspace(0.0, 1.0, 101)
    intensities = np.zeros_like(mz)
    intensities[40] = 10.0
    intensities[60] = 9.0

    monkeypatch.setattr(
        detection_module,
        "_height_threshold_from_noise_model",
        lambda *args, **kwargs: (0.0, 1.0, 0.0),
    )
    monkeypatch.setattr(
        detection_module.signal,
        "find_peaks",
        lambda *args, **kwargs: (
            np.array([40, 60], dtype=int),
            {
                "prominences": np.array([10.0, 9.0], dtype=float),
                "peak_heights": np.array([10.0, 9.0], dtype=float),
            },
        ),
    )
    monkeypatch.setattr(
        detection_module.signal,
        "peak_widths",
        lambda *args, **kwargs: (
            np.array([8.0, 8.0], dtype=float),
            np.array([5.0, 4.5], dtype=float),
            np.array([36.0, 56.0], dtype=float),
            np.array([44.0, 64.0], dtype=float),
        ),
    )

    def fake_curve_fit(func, x_fit, y_fit, p0=None, **kwargs):
        if func is detection_module.two_gaussians:
            return np.array([9.0, 0.40, 0.01, 8.5, 0.60, 0.01]), np.eye(6)
        return np.array([float(np.nanmax(y_fit)), float(x_fit[np.argmax(y_fit)]), 0.01]), np.eye(3)

    monkeypatch.setattr(detection_module.optimize, "curve_fit", fake_curve_fit)
    monkeypatch.setattr(
        detection_module,
        "_fit_single_gaussian_window",
        lambda *args, **kwargs: np.array([10.0, 0.50, 0.02]),
    )
    monkeypatch.setattr(
        detection_module,
        "_bic_score",
        lambda y_true, y_pred, n_params: 100.0 if n_params == 3 else 95.0,
    )

    df = robust_peak_detection(
        mz_values=mz,
        intensities=intensities,
        sample_name="sample",
        group="Unknown",
        method="Gaussian",
        min_intensity=0.0,
        min_snr=0.0,
        min_distance=1,
        window_size=5,
        peak_height=0.0,
        prominence=0.0,
        min_peak_width=1,
        max_peak_width=20,
        distance_threshold=0.5,
        deconvolution_min_bic_delta=10.0,
    )

    assert not df["Deconvoluted"].any()
    assert set(df["FitModel"]) == {"Gaussian"}


def test_robust_peak_detection_accepts_deconvolution_with_strong_bic_support(monkeypatch):
    mz = np.linspace(0.0, 1.0, 101)
    intensities = np.zeros_like(mz)
    intensities[49] = 12.0
    intensities[51] = 11.0

    monkeypatch.setattr(
        detection_module,
        "_height_threshold_from_noise_model",
        lambda *args, **kwargs: (0.0, 1.0, 0.0),
    )
    monkeypatch.setattr(
        detection_module.signal,
        "find_peaks",
        lambda *args, **kwargs: (
            np.array([49, 51], dtype=int),
            {
                "prominences": np.array([12.0, 11.0], dtype=float),
                "peak_heights": np.array([12.0, 11.0], dtype=float),
            },
        ),
    )
    monkeypatch.setattr(
        detection_module.signal,
        "peak_widths",
        lambda *args, **kwargs: (
            np.array([8.0, 8.0], dtype=float),
            np.array([6.0, 5.5], dtype=float),
            np.array([45.0, 47.0], dtype=float),
            np.array([53.0, 55.0], dtype=float),
        ),
    )

    def fake_curve_fit(func, x_fit, y_fit, p0=None, **kwargs):
        if func is detection_module.two_gaussians:
            return np.array([11.0, 0.49, 0.01, 10.5, 0.51, 0.01]), np.eye(6)
        return np.array([float(np.nanmax(y_fit)), float(x_fit[np.argmax(y_fit)]), 0.01]), np.eye(3)

    monkeypatch.setattr(detection_module.optimize, "curve_fit", fake_curve_fit)
    monkeypatch.setattr(
        detection_module,
        "_fit_single_gaussian_window",
        lambda *args, **kwargs: np.array([12.0, 0.50, 0.02]),
    )
    monkeypatch.setattr(
        detection_module,
        "_bic_score",
        lambda y_true, y_pred, n_params: 100.0 if n_params == 3 else 80.0,
    )

    df = robust_peak_detection(
        mz_values=mz,
        intensities=intensities,
        sample_name="sample",
        group="Unknown",
        method="Gaussian",
        min_intensity=0.0,
        min_snr=0.0,
        min_distance=1,
        window_size=5,
        peak_height=0.0,
        prominence=0.0,
        min_peak_width=1,
        max_peak_width=20,
        distance_threshold=0.5,
        deconvolution_min_bic_delta=10.0,
    )

    assert len(df) == 2
    assert df["Deconvoluted"].all()
    assert set(df["FitModel"]) == {"two_gaussian"}


def test_robust_peak_detection_logs_fit_failure_summary(monkeypatch, caplog):
    mz = np.linspace(0.0, 1.0, 51)
    intensities = np.zeros_like(mz)
    intensities[25] = 10.0

    monkeypatch.setattr(
        detection_module,
        "_height_threshold_from_noise_model",
        lambda *args, **kwargs: (0.0, 1.0, 0.0),
    )
    monkeypatch.setattr(
        detection_module.signal,
        "find_peaks",
        lambda *args, **kwargs: (
            np.array([25], dtype=int),
            {
                "prominences": np.array([10.0], dtype=float),
                "peak_heights": np.array([10.0], dtype=float),
            },
        ),
    )
    monkeypatch.setattr(
        detection_module.signal,
        "peak_widths",
        lambda *args, **kwargs: (
            np.array([6.0], dtype=float),
            np.array([5.0], dtype=float),
            np.array([22.0], dtype=float),
            np.array([28.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        detection_module.optimize,
        "curve_fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )

    df = robust_peak_detection(
        mz_values=mz,
        intensities=intensities,
        sample_name="sample",
        group="Unknown",
        method="Gaussian",
        min_intensity=0.0,
        min_snr=0.0,
        min_distance=1,
        window_size=5,
        peak_height=0.0,
        prominence=0.0,
        min_peak_width=1,
        max_peak_width=20,
    )

    assert df.empty
    assert "single-peak fits failed" in caplog.text
