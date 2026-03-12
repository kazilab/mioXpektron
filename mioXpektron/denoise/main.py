"""High-level orchestration helpers for denoising spectra and reviewing results."""

import logging

import numpy as np

logger = logging.getLogger(__name__)
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    pl = None
    _POLARS_AVAILABLE = False

import pandas as pd
from pathlib import Path
from datetime import datetime

from .denoise_main import noise_filtering
from .denoise_batch import batch_denoise
from .denoise_select import (
    compare_denoising_methods,
    compare_methods_in_windows,
    plot_pareto_delta_snr_vs_height,
    rank_method,
    select_methods,
    decode_method_label
)
from ..plotting import PlotPeak


OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DenoisingMethods:
    """Evaluate and visualize denoising strategies for mass spectrometry data.

    Parameters
    ----------
    mz : np.ndarray | pl.Series
        The m/z axis of the spectrum.
    intensity : np.ndarray | pl.Series
        Raw intensity values aligned with ``mz``.
    """

    def __init__(self, mz_values, raw_intensities):
        """Store the raw spectrum that downstream helpers will operate on."""
        self.mz = mz_values
        self.intensity = raw_intensities

    def compare(
        self,
        min_mz,
        max_mz,
        return_format='pandas',
        w_match=3.0,
        w_mz=2.0,
        w_area=2.0,
        w_height=1.5,
        w_fwhm=1.0,
        w_spread=1.0,
        w_noise_db=2.0,
        w_delta_snr_db=1.5,
        save_summary=True
    ):
        """Compare denoising methods across the full spectrum window.

        Parameters
        ----------
        min_mz, max_mz : float
            Bounds for the evaluation window.
        return_format : {"pandas", "polars"}, default "pandas"
            Determines the summary dataframe type returned by the lower-level
            evaluators.
        w_match, w_mz, w_area, w_height, w_fwhm, w_spread, w_noise_db, w_delta_snr_db : float
            Weights applied by :func:`rank_method` when condensing metrics into a
            single score.
        save_summary : bool, default True
            When True and the summary is a pandas object, persist an Excel copy
            in ``OUTPUT_DIR`` for later inspection.

        Returns
        -------
        DataFrame or LazyFrame
            Ranked table whose concrete type depends on ``return_format``.
        """

        summary_df, detail_df = compare_denoising_methods(
            self.mz,
            self.intensity,
            min_mz=min_mz,
            max_mz=max_mz,
            return_format=return_format
        )
        summary = rank_method(
            input_format=return_format,
            summary_df=summary_df,
            per_peak_df=detail_df,
            w_match=w_match,
            w_mz=w_mz,
            w_area=w_area,
            w_height=w_height,
            w_fwhm=w_fwhm,
            w_spread=w_spread,
            w_noise_db=w_noise_db,
            w_delta_snr_db=w_delta_snr_db,
        )
        if save_summary:
            file_name = f"denoise_summary_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            file_path = OUTPUT_DIR / file_name
            if _POLARS_AVAILABLE and isinstance(summary, pl.DataFrame):
                summary.write_excel(file_path)
            elif hasattr(summary, "to_excel"):
                summary.to_excel(file_path)
        return summary

    def compare_in_windows(
        self,
        windows,
        per_window_max_peaks=50,
        min_prominence=None,
        search_ppm=20.0,
        resample_to_uniform=True,
        return_format='pandas',
        w_match=3.0,
        w_mz=2.0,
        w_area=2.0,
        w_height=1.5,
        w_fwhm=1.0,
        w_spread=1.0,
        w_noise_db=2.0,
        w_delta_snr_db=1.5,
        save_summary=True
    ):
        """Compare denoising methods within pre-defined m/z windows.

        Parameters mirror :meth:`compare` with additional controls for window
        segmentation. The return value matches ``return_format`` and includes a
        ranking aggregated across all windows.

        Returns
        -------
        DataFrame or LazyFrame
            Ranked summary consistent with ``return_format``.
        """

        _, window_summary, window_detail = compare_methods_in_windows(
            self.mz,
            self.intensity,
            windows=windows,
            per_window_max_peaks=per_window_max_peaks,
            min_prominence=min_prominence,
            search_ppm=search_ppm,
            resample_to_uniform=resample_to_uniform,
            return_format=return_format,
        )
        summary = rank_method(
            input_format=return_format,
            summary_df=window_summary,
            per_peak_df=window_detail,
            w_match=w_match,
            w_mz=w_mz,
            w_area=w_area,
            w_height=w_height,
            w_fwhm=w_fwhm,
            w_spread=w_spread,
            w_noise_db=w_noise_db,
            w_delta_snr_db=w_delta_snr_db,
        )
        if save_summary:
            file_name = f"denoise_summary_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            file_path = OUTPUT_DIR / file_name
            if _POLARS_AVAILABLE and isinstance(summary, pl.DataFrame):
                summary.write_excel(file_path)
            elif hasattr(summary, "to_excel"):
                summary.to_excel(file_path)
        return summary

    def plot(self, summary, annotate=True, top_k=3, save_plot=True, save_pareto=True):
        """Visualize the Pareto front of SNR gain versus peak-height deviation.

        Parameters
        ----------
        summary : DataFrame or LazyFrame
            Ranking output generated by :meth:`compare` or :meth:`compare_in_windows`.
        annotate : bool, default True
            If True, label the top ``top_k`` points on the Pareto chart.
        top_k : int, default 3
            Number of top-ranked methods to annotate.
        save_plot : bool, default True
            Persist the Matplotlib figure via :func:`plot_pareto_delta_snr_vs_height`.
        save_pareto : bool, default True
            Persist the underlying data used to draw the plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axis used for further customization.
        """

        ax = plot_pareto_delta_snr_vs_height(
            summary,
            annotate=annotate,
            top_k=top_k,
            save_plot=save_plot,
            save_pareto=save_pareto
        )
        
        return ax

    def denoise_check(
        self,
        denoise_params,
        *,
        sample_name='test',
        group=None,
        log_scale_y=False,
        mz_min=0,
        mz_max=500,
        show_peaks=False,
        peak_height=1000,
        peak_prominence=50,
        min_peak_width=1,
        max_peak_width=None,
        figsize=(10,6),
        save_plot=True
    ):
        """Preview a single denoising configuration by plotting selected peaks.

        Parameters
        ----------
        denoise_params : Mapping[str, Any]
            Keyword arguments forwarded directly to :func:`noise_filtering`.
        sample_name : str, default "test"
            Label forwarded to :class:`PlotPeak` for file naming.
        group : str | None, optional
            Group identifier used by :class:`PlotPeak` when saving plots.
        log_scale_y : bool, default False
            Apply ``log1p`` before plotting, useful for high-dynamic-range spectra.
        mz_min, mz_max : float
            m/z bounds for the preview overlay.
        show_peaks : bool, default False
            Highlight top peaks using :class:`PlotPeak` detection settings.
        peak_height, peak_prominence, min_peak_width, max_peak_width : float
            Tuning knobs passed to :class:`PlotPeak` when ``show_peaks`` is True.
        save_plot : bool, default True
            Persist the rendered preview when requested by :class:`PlotPeak`.

        Returns
        -------
        matplotlib.axes.Axes
            Axis returned by :class:`PlotPeak` so callers can layer annotations.
        """

        if not isinstance(denoise_params, dict):
            raise TypeError("denoise_params must be a dict of noise_filtering arguments")

        params = dict(denoise_params)
        params.setdefault('method', 'wavelet')
        denoised_intensity = noise_filtering(
            self.intensity,
            **params,
        )

        if log_scale_y:
            raw_intensity = np.log1p(self.intensity)
            corrected_intensity = np.log1p(denoised_intensity)
        else:
            raw_intensity = self.intensity
            corrected_intensity = denoised_intensity

        plotter = PlotPeak(
            mz_values=self.mz,
            raw_intensities=raw_intensity,
            sample_name=sample_name,
            group=group,
            corrected_intensities=corrected_intensity,
        )
        return plotter.plot(
            mz_min=mz_min,
            mz_max=mz_max,
            show_peaks=show_peaks,
            peak_height=peak_height,
            peak_prominence=peak_prominence,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            figsize=figsize,
            save_plot=save_plot
        )
    
    def method_parameters(self, summary, rank=0, basis="pareto_then_score",
                          require_pass=True, require_finite_metrics=True,
                          save_selected=True):
        """Extract the configuration for a ranked denoising method.

        Parameters
        ----------
        summary : DataFrame | pl.DataFrame
            Ranked output produced by the comparison helpers.
        rank : int, default 0
            Zero-based index of the desired method after Pareto filtering.
        basis : str, default "pareto_then_score"
            Strategy forwarded to :func:`select_methods` when Pareto filtering is
            available.
        require_pass : bool, default True
            If True, discard rows that failed the minimum denoising constraint.
        require_finite_metrics : bool, default True
            Drop methods with NaNs before ranking.
        save_selected : bool, default True
            Persist the filtered table to ``OUTPUT_DIR`` for reproducibility.

        Returns
        -------
        dict
            Parameters suitable for passing into :func:`noise_filtering`.
        """

        try:
            import polars as pl  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            pl = None  # type: ignore

        if pl is not None and isinstance(summary, pl.DataFrame):
            df = summary.to_pandas()
        else:
            if not hasattr(summary, "copy"):
                raise TypeError("summary must be a pandas or polars DataFrame")
            df = summary.copy()

        if "method" not in df.columns:
            raise KeyError("summary must contain a 'method' column")

        columns = set(df.columns)
        can_use_pareto = {"abs_height", "delta_snr_db_med"}.issubset(columns)

        if can_use_pareto:
            _, _, selected_df = select_methods(
                df,
                basis=basis,
                top_k=max(rank + 1, 12),
                require_pass=require_pass,
                require_finite_metrics=require_finite_metrics,
            )
        else:
            selected_df = df.copy()
            if require_pass and "passes_min_denoise" in selected_df.columns:
                passed = selected_df[selected_df["passes_min_denoise"] == True]  # noqa: E712
                if not passed.empty:
                    selected_df = passed
            if "score" in selected_df.columns:
                selected_df = selected_df.sort_values("score", ascending=True)
            selected_df = selected_df.reset_index(drop=True)

        if not (0 <= rank < len(selected_df)):
            if "score" in df.columns:
                df_sorted = df.sort_values("score", ascending=True).reset_index(drop=True)
            else:
                df_sorted = df.reset_index(drop=True)

            if rank >= len(df_sorted):
                raise IndexError("rank out of range for selected methods")

            selected_df = df_sorted

        method_label = selected_df.iloc[rank]["method"]
        if save_selected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"Selected_methods_{timestamp}.xlsx"
            if _POLARS_AVAILABLE and isinstance(selected_df, pl.DataFrame):
                selected_df.write_excel(out_path)
            else:
                selected_df.to_excel(out_path)
        return decode_method_label(method_label)


class BatchDenoising:
    """Run denoising across a batch of spectra with timestamped outputs."""

    _ALLOWED_METHODS = {'wavelet', 'gaussian', 'median', 'savitzky_golay', 'none'}

    def __init__(
        self,
        file_paths,
        *,
        method='wavelet',
        n_workers=None,
        backend='threads',
        progress=True,
        params=None,
    ):
        """Store batch processing parameters for later execution."""
        paths = list(file_paths)
        if not paths:
            raise ValueError("file_paths must contain at least one entry")

        if method not in self._ALLOWED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Valid options: {sorted(self._ALLOWED_METHODS)}"
            )

        self.file_paths = [Path(fp) for fp in paths]
        self.method = method
        self.n_workers = n_workers
        self.backend = backend
        self.progress = progress
        self.params = ({k: v for k, v in dict(params).items() if k != "method"}
               if params is not None else None)

        self.last_output_dir = None
        self.last_results = None

    def _normalized_worker_count(self):
        """Return an executor-friendly worker count (0 => auto)."""
        if self.n_workers is None or self.n_workers <= 0:
            return 0
        return self.n_workers

    def run(self, output_root=None, folder_name='denoised_spectrums', save_result=True):
        """Execute the batch denoising run.

        Parameters
        ----------
        output_root : str | Path | None
            Directory where the timestamped result folder will be created. If
            omitted, defaults to :data:`OUTPUT_DIR`.
        folder_name : str, default "denoised_spectrums"
            Base name for the result folder. A ``YYYYmmdd_HHMM`` timestamp is
            appended automatically.
        save_result : bool, default True
            Persist the executor results dataframe to ``OUTPUT_DIR``.

        Returns
        -------
        list[BatchResult]
            Records describing each processed file.
        """
        if output_root is None:
            output_root = OUTPUT_DIR
        
        output_root = Path(output_root)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = output_root / f"{folder_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = batch_denoise(
            files=[str(path) for path in self.file_paths],
            output_dir=output_dir,
            method=self.method,
            n_workers=self._normalized_worker_count(),
            backend=self.backend,
            progress=self.progress,
            params=self.params,
        )

        ok = [r for r in results if r.status == "ok"]
        bad = [r for r in results if r.status == "error"]
        logger.info(f"OK: {len(ok)} | ERRORS: {len(bad)}")
        logger.info("First 3 outputs: %s", [r.out_file for r in ok[:3]])
        if bad:
            logger.warning("Example error:\n%s", bad[0].message)

        self.last_output_dir = output_dir
        self.last_results = results
        if save_result:
            result_path = output_root / f"denoising_results_{timestamp}.xlsx"
            if _POLARS_AVAILABLE:
                # Convert results to dict list for Polars
                results_dicts = [vars(r) for r in results]
                pl.DataFrame(results_dicts).write_excel(result_path)
            else:
                pd.DataFrame(results).to_excel(result_path)

        return results
