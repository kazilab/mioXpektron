import numpy as np
import pandas as pd

from . import main as denoise_main_api
from .denoise_select import (
    _measure_on_method,
    _measure_one_peak,
    _build_denoising_method_grid,
    aggregate_method_summaries,
    plot_pareto_delta_snr_vs_height,
    rank_methods_pandas,
    select_methods,
)


def _summary_row(method: str, **overrides):
    row = {
        "method": method,
        "peaks_total": 10,
        "peaks_matched": 9,
        "peaks_lost": 1,
        "frac_matched": 0.9,
        "mz_shift_med": 0.0,
        "mz_shift_iqr": 0.1,
        "pct_height_med": -2.0,
        "pct_height_iqr": 1.0,
        "pct_fwhm_med": -3.0,
        "pct_fwhm_iqr": 1.5,
        "pct_area_med": -4.0,
        "pct_area_iqr": 2.0,
        "sigma_raw_global": 2.0,
        "sigma_new_global": 1.0,
        "noise_reduction_db": 6.0,
        "delta_snr_db_med": 4.0,
        "delta_snr_db_iqr": 0.5,
        "hf_power_reduction_db": 3.0,
        "hf_frac_new_global": 0.2,
    }
    row.update(overrides)
    return row


def test_measure_on_method_rejects_monotonic_shoulder():
    x = np.linspace(99.995, 100.005, 101)
    y_raw = 10.0 * np.exp(-((x - 100.0) / 0.0006) ** 2)
    ref_idx = int(np.argmax(y_raw))
    ref = _measure_one_peak(x, y_raw, ref_idx, prominence=10.0)

    # Monotonic data inside the search window has no true local maximum, but
    # the previous implementation still accepted its boundary argmax.
    y_bad = np.linspace(5.0, 1.0, x.size)

    match = _measure_on_method(x, y_bad, ref, search_ppm=1000.0)

    assert match is None


def test_aggregate_method_summaries_preserves_unique_methods():
    summary = pd.DataFrame(
        [
            _summary_row("method_a", peaks_total=10, peaks_matched=8, peaks_lost=2, delta_snr_db_med=3.0),
            _summary_row("method_a", peaks_total=4, peaks_matched=2, peaks_lost=2, delta_snr_db_med=5.0),
            _summary_row("method_b", peaks_total=6, peaks_matched=6, peaks_lost=0, delta_snr_db_med=2.0),
        ]
    )

    rollup = aggregate_method_summaries(summary, unit_label="windows")

    method_a = rollup.loc[rollup["method"] == "method_a"].iloc[0]
    assert len(rollup) == 2
    assert method_a["windows"] == 2
    assert method_a["peaks_total"] == 14
    assert method_a["peaks_matched"] == 10
    assert np.isclose(method_a["frac_matched"], 10 / 14)
    assert np.isclose(method_a["delta_snr_db_med"], 4.0)


def test_build_denoising_method_grid_excludes_derivatives_by_default():
    x = np.array([100.0, 100.1, 100.2], dtype=float)

    default_grid = _build_denoising_method_grid(
        x,
        resample_to_uniform=False,
        target_dx=None,
        include_derivatives=False,
    )
    derivative_grid = _build_denoising_method_grid(
        x,
        resample_to_uniform=False,
        target_dx=None,
        include_derivatives=True,
    )

    assert len(default_grid) == 511
    assert len(derivative_grid) == 561
    assert all(":deriv_0" in name or ":deriv_" not in name for name in default_grid)
    assert all(":order_0" in name or ":order_" not in name for name in default_grid)
    assert any(":deriv_1" in name for name in derivative_grid)
    assert any(":order_1" in name for name in derivative_grid)


def test_rank_methods_pandas_applies_ppm_normalized_selection_criteria():
    summary = pd.DataFrame(
        [
            _summary_row(
                "method_ok",
                frac_matched=0.95,
                mz_shift_med=0.0,
                mz_shift_iqr=0.001,
                pct_height_med=8.0,
                pct_height_iqr=10.0,
                pct_fwhm_med=6.0,
                pct_fwhm_iqr=8.0,
                pct_area_med=7.0,
                pct_area_iqr=9.0,
                noise_reduction_db=4.0,
                delta_snr_db_med=4.0,
            ),
            _summary_row(
                "method_bad_ppm",
                frac_matched=0.95,
                mz_shift_med=0.0,
                mz_shift_iqr=0.020,
                pct_height_med=8.0,
                pct_height_iqr=10.0,
                pct_fwhm_med=6.0,
                pct_fwhm_iqr=8.0,
                pct_area_med=7.0,
                pct_area_iqr=9.0,
                noise_reduction_db=4.0,
                delta_snr_db_med=4.0,
            ),
        ]
    )
    per_peak = pd.DataFrame([{"mz_ref": 1000.0}, {"mz_ref": 1000.0}])

    ranked = rank_methods_pandas(summary, per_peak)

    ok_row = ranked.loc[ranked["method"] == "method_ok"].iloc[0]
    bad_row = ranked.loc[ranked["method"] == "method_bad_ppm"].iloc[0]

    assert np.isclose(ok_row["mz_shift_iqr_ppm"], 1.0)
    assert np.isclose(bad_row["mz_shift_iqr_ppm"], 20.0)
    assert bool(ok_row["pass_mz_shift_iqr_ppm"])
    assert not bool(bad_row["pass_mz_shift_iqr_ppm"])
    assert bool(ok_row["passes_peak_preservation"])
    assert not bool(bad_row["passes_peak_preservation"])
    assert bool(ok_row["passes_selection_criteria"])
    assert not bool(bad_row["passes_selection_criteria"])
    assert int(bad_row["failed_criteria_count"]) >= 1
    assert ranked.iloc[0]["method"] == "method_ok"


def test_select_methods_constrained_pareto_then_snr_prefers_highest_snr():
    summary = pd.DataFrame(
        [
            {
                "method": "low_bias",
                "abs_height": 1.0,
                "delta_snr_db_med": 5.0,
                "frac_matched": 0.97,
                "selection_score": 0.40,
                "passes_selection_criteria": True,
            },
            {
                "method": "high_snr",
                "abs_height": 2.0,
                "delta_snr_db_med": 7.0,
                "frac_matched": 0.93,
                "selection_score": 0.55,
                "passes_selection_criteria": True,
            },
            {
                "method": "dominated",
                "abs_height": 2.5,
                "delta_snr_db_med": 6.0,
                "frac_matched": 0.95,
                "selection_score": 0.35,
                "passes_selection_criteria": True,
            },
            {
                "method": "fails_gate",
                "abs_height": 0.5,
                "delta_snr_db_med": 9.0,
                "frac_matched": 0.99,
                "selection_score": 0.10,
                "passes_selection_criteria": False,
            },
        ]
    )

    _, frontier, selected = select_methods(
        summary,
        basis="constrained_pareto_then_snr",
        top_k=2,
    )

    assert set(frontier["method"]) == {"low_bias", "high_snr"}
    assert list(selected["method"]) == ["high_snr", "low_bias"]


def test_plot_pareto_falls_back_to_all_finite_candidates_when_none_pass():
    import matplotlib.pyplot as plt

    summary = pd.DataFrame(
        [
            {
                "method": "method_a",
                "abs_height": 1.0,
                "delta_snr_db_med": 2.0,
                "passes_selection_criteria": False,
            },
            {
                "method": "method_b",
                "abs_height": 2.0,
                "delta_snr_db_med": 3.0,
                "passes_selection_criteria": False,
            },
        ]
    )

    ax = plot_pareto_delta_snr_vs_height(
        summary,
        save_plot=False,
        save_pareto=False,
    )

    assert "No methods passed the current selection criteria" in ax.get_title()
    plt.close(ax.figure)


def test_compare_in_windows_ranks_aggregated_rollup(monkeypatch):
    rollup = pd.DataFrame([_summary_row("method_a")])
    window_summary = pd.DataFrame(
        [
            {**_summary_row("method_a"), "window": "[400,410]"},
            {**_summary_row("method_a"), "window": "[440,450]"},
        ]
    )
    detail = pd.DataFrame([{"method": "method_a", "mz_ref": 100.0}])
    captured = {}

    def fake_compare_methods_in_windows(*args, **kwargs):
        return rollup, window_summary, detail

    def fake_rank_method(*, summary_df, per_peak_df, input_format, **kwargs):
        captured["summary_df"] = summary_df.copy()
        captured["per_peak_df"] = per_peak_df.copy()
        captured["input_format"] = input_format
        return summary_df

    monkeypatch.setattr(denoise_main_api, "compare_methods_in_windows", fake_compare_methods_in_windows)
    monkeypatch.setattr(denoise_main_api, "rank_method", fake_rank_method)

    dm = denoise_main_api.DenoisingMethods(np.array([100.0, 101.0]), np.array([1.0, 2.0]))
    summary = dm.compare_in_windows(windows=[(400, 410)], save_summary=False)

    assert len(summary) == 1
    assert len(captured["summary_df"]) == 1
    assert "window" not in captured["summary_df"].columns
    assert captured["input_format"] == "pandas"
    assert len(captured["per_peak_df"]) == 1


def test_compare_across_files_aggregates_sample_level_summaries(monkeypatch, tmp_path):
    file_a = tmp_path / "sample_a.txt"
    file_b = tmp_path / "sample_b.txt"
    file_a.write_text("")
    file_b.write_text("")

    def fake_load_txt_spectrum(path):
        amp = 1.0 if path.stem.endswith("a") else 2.0
        return {
            "mz": np.array([99.9, 100.0, 100.1], dtype=float),
            "intensity": np.array([0.0, amp, 0.0], dtype=float),
        }

    def fake_compare_denoising_methods(mz, intensity, **kwargs):
        assert kwargs["n_jobs"] == 3
        assert kwargs["parallel_backend"] == "thread"
        assert kwargs["progress"] is False
        assert kwargs["include_derivatives"] is True
        amp = float(intensity[1])
        summary = pd.DataFrame(
            [
                _summary_row("method_a", delta_snr_db_med=3.0 + amp, noise_reduction_db=5.0 + amp),
                _summary_row("method_b", delta_snr_db_med=2.0 + amp, noise_reduction_db=4.0 + amp),
            ]
        )
        detail = pd.DataFrame(
            [
                {"method": "method_a", "mz_ref": 100.0},
                {"method": "method_b", "mz_ref": 100.0},
            ]
        )
        return summary, detail

    def fake_rank_method(*, summary_df, per_peak_df, input_format, **kwargs):
        assert input_format == "pandas"
        assert "spectra" in summary_df.columns
        return summary_df.sort_values("method").reset_index(drop=True)

    monkeypatch.setattr(denoise_main_api, "load_txt_spectrum", fake_load_txt_spectrum)
    monkeypatch.setattr(denoise_main_api, "compare_denoising_methods", fake_compare_denoising_methods)
    monkeypatch.setattr(denoise_main_api, "rank_method", fake_rank_method)

    ranked, sample_summary_all, detail_all = denoise_main_api.DenoisingMethods.compare_across_files(
        [file_b, file_a],
        min_mz=99.5,
        max_mz=100.5,
        file_n_jobs=1,
        method_n_jobs=3,
        method_parallel_backend="thread",
        include_derivatives=True,
        progress=False,
        save_summary=False,
    )

    assert len(ranked) == 2
    assert all(ranked["spectra"] == 2)
    assert len(sample_summary_all) == 4
    assert set(sample_summary_all["sample"]) == {"sample_a", "sample_b"}
    assert len(detail_all) == 4
