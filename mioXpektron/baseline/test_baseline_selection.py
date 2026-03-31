from __future__ import annotations

import logging

import matplotlib
import numpy as np

from .baseline_base import baseline_correction
from .baseline_eval import BaselineMethodEvaluator

matplotlib.use("Agg")


def _write_spectrum_csv(tmp_path, name: str = "spec.csv"):
    path = tmp_path / name
    x = np.linspace(10.0, 60.0, 200)
    baseline = 0.02 * x
    peak = 5.0 * np.exp(-((x - 32.0) / 1.4) ** 2)
    y = baseline + peak
    with path.open("w", encoding="utf-8") as handle:
        handle.write("channel,m/z,intensity\n")
        for idx, (mz, intensity) in enumerate(zip(x, y), start=1):
            handle.write(f"{idx},{mz},{intensity}\n")
    return path


def test_evaluator_uses_param_grid_keys_when_methods_are_omitted(tmp_path):
    spectrum_path = _write_spectrum_csv(tmp_path)
    param_grid = {
        "adaptive_window": [{"window_size": 5}],
        "poly": [{"poly_order": 2}],
    }

    evaluator = BaselineMethodEvaluator(files=[spectrum_path], param_grid=param_grid)

    assert evaluator.labels == [
        "adaptive_window(window_size=5)",
        "poly(poly_order=2)",
    ]
    assert evaluator.specs == [
        ("adaptive_window", {"window_size": 5}),
        ("poly", {"poly_order": 2}),
    ]


def test_baseline_correction_accepts_parameterized_method_labels():
    y = np.array([5.0, 4.5, 4.0, 20.0, 4.0, 3.5, 3.0], dtype=float)

    parsed = baseline_correction(y, method="adaptive_window(window_size=3)", clip_negative=False)
    explicit = baseline_correction(y, method="adaptive_window", window_size=3, clip_negative=False)

    assert np.allclose(parsed, explicit)


def test_preview_overlay_uses_ranked_parameterized_methods(tmp_path, caplog):
    spectrum_path = _write_spectrum_csv(tmp_path)
    evaluator = BaselineMethodEvaluator(
        files=[spectrum_path],
        methods=["adaptive_window"],
        param_grid={"adaptive_window": [{"window_size": 5}]},
    )

    summary = evaluator.evaluate(n_jobs=1)
    assert summary["overall_best_spec"] == {
        "label": "adaptive_window(window_size=5)",
        "method": "adaptive_window",
        "kwargs": {"window_size": 5},
    }

    with caplog.at_level(logging.INFO, logger="mioXpektron.baseline.baseline_eval"):
        evaluator.preview_overlay(file=spectrum_path, save_to=None, show_errors=True)

    assert "Unknown baseline method" not in caplog.text
    assert "Successful: 1/1 methods" in caplog.text
