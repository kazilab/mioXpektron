"""High-level orchestration helpers for normalising ToF-SIMS spectra."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    _POLARS_AVAILABLE = False

from .normalization import normalize, normalization_method_names, tic_normalization
from .normalization_eval import NormalizationEvaluator
from ..plotting import PlotPeak

OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NormalizationMethods:
    """Evaluate and apply normalization strategies for ToF-SIMS data.

    Parameters
    ----------
    mz_values : array-like
        The m/z axis shared by all spectra.
    raw_intensities : array-like
        Raw intensity values aligned with ``mz_values``.
    """

    def __init__(self, mz_values, raw_intensities):
        self.mz = np.asarray(mz_values, dtype=float)
        self.intensity = np.asarray(raw_intensities, dtype=float)

    # -- single spectrum helpers -------------------------------------------

    def apply(self, method: str = "tic", **kwargs) -> np.ndarray:
        """Apply a named normalization to the stored spectrum.

        Parameters
        ----------
        method : str
            Normalization method name (see :func:`normalization_method_names`).
        **kwargs
            Method-specific keyword arguments.

        Returns
        -------
        np.ndarray
            Normalized intensity array.
        """
        return normalize(self.intensity, method=method, **kwargs)

    def compare_visual(
        self,
        methods: Optional[List[str]] = None,
        method_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        mz_min: float = 0,
        mz_max: float = 500,
        sample_name: str = "test",
        group: Optional[str] = None,
        figsize: tuple = (12, 8),
        save_plot: bool = True,
    ):
        """Plot the raw spectrum alongside several normalized versions.

        Parameters
        ----------
        methods : list of str, optional
            Normalization methods to overlay.  Defaults to a curated set.
        method_kwargs_map : dict, optional
            ``{method: {kwarg: value}}`` for method-specific parameters.
        mz_min, mz_max : float
            m/z bounds for the preview window.
        sample_name : str
            Label used for file naming.
        group : str or None
            Group identifier.
        figsize : tuple
            Figure size.
        save_plot : bool
            Persist the rendered figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if methods is None:
            methods = ["tic", "median", "rms", "poisson", "sqrt", "vsn"]
        method_kwargs_map = method_kwargs_map or {}

        mask = (self.mz >= mz_min) & (self.mz <= mz_max)
        mz_win = self.mz[mask]

        n = len(methods) + 1
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] / 3 * n),
                                  sharex=True)

        axes[0].plot(mz_win, self.intensity[mask], lw=0.5, color="grey")
        axes[0].set_title("Raw")
        axes[0].set_ylabel("Intensity")

        for i, m in enumerate(methods):
            kwargs = method_kwargs_map.get(m, {})
            try:
                normed = normalize(self.intensity, method=m, **kwargs)
                axes[i + 1].plot(mz_win, normed[mask], lw=0.5)
                axes[i + 1].set_title(m)
                axes[i + 1].set_ylabel("Norm. Int.")
            except Exception as e:
                axes[i + 1].set_title(f"{m} (failed: {e})")

        axes[-1].set_xlabel("m/z")
        fig.suptitle(f"Normalization comparison — {sample_name}", y=1.01)
        fig.tight_layout()

        if save_plot:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            for ext in (".png", ".pdf"):
                fig.savefig(
                    OUTPUT_DIR / f"norm_compare_{sample_name}_{ts}{ext}",
                    bbox_inches="tight", dpi=300,
                )

        return axes

    def normalize_and_check(
        self,
        method: str = "tic",
        method_kwargs: Optional[Dict[str, Any]] = None,
        *,
        sample_name: str = "test",
        group: Optional[str] = None,
        mz_min: float = 0,
        mz_max: float = 500,
        show_peaks: bool = False,
        peak_height: float = 1000,
        peak_prominence: float = 50,
        min_peak_width: int = 1,
        max_peak_width: Optional[int] = None,
        figsize: tuple = (10, 6),
        save_plot: bool = True,
    ):
        """Apply one normalization and visualise the result with peak overlay.

        Parameters
        ----------
        method : str
            Normalization method.
        method_kwargs : dict, optional
            Extra kwargs forwarded to :func:`normalize`.
        sample_name, group : str
            Plot labels.
        mz_min, mz_max : float
            m/z window for the plot.
        show_peaks : bool
            Annotate detected peaks.
        peak_height, peak_prominence, min_peak_width, max_peak_width
            Peak detection tuning passed to :class:`PlotPeak`.
        figsize : tuple
        save_plot : bool

        Returns
        -------
        matplotlib.axes.Axes
        """
        method_kwargs = method_kwargs or {}
        normalized = normalize(self.intensity, method=method, **method_kwargs)

        plotter = PlotPeak(
            mz_values=self.mz,
            raw_intensities=self.intensity,
            sample_name=sample_name,
            group=group,
            corrected_intensities=normalized,
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
            save_plot=save_plot,
        )

    # -- batch evaluation --------------------------------------------------

    @staticmethod
    def evaluate(
        files: List[Union[str, Path]],
        methods: Optional[List[str]] = None,
        method_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        mz_min: Optional[float] = None,
        mz_max: Optional[float] = None,
        n_jobs: int = -1,
        compute_supervised: bool = True,
        save_results: bool = True,
    ):
        """Evaluate normalization methods across multiple spectra files.

        Thin wrapper around :class:`NormalizationEvaluator` that runs
        evaluation, prints a summary, and optionally saves results.

        Parameters
        ----------
        files : list of str or Path
            Spectrum file paths or glob patterns.
        methods : list of str, optional
            Method names to evaluate.
        method_kwargs_map : dict, optional
            Per-method keyword arguments.
        mz_min, mz_max : float, optional
            m/z range for data import.
        n_jobs : int
            Parallel workers (``-1`` = all CPUs).
        compute_supervised : bool
            Run supervised classification (requires scikit-learn).
        save_results : bool
            Save CSV + JSON + plots to ``OUTPUT_DIR``.

        Returns
        -------
        NormalizationEvaluator
            The evaluator instance (call ``.plot()`` for figures).
        """
        evaluator = NormalizationEvaluator(
            files=files,
            methods=methods,
            method_kwargs_map=method_kwargs_map,
            mz_min=mz_min,
            mz_max=mz_max,
            n_jobs=n_jobs,
            compute_supervised=compute_supervised,
        )
        results = evaluator.evaluate()
        evaluator.print_summary()

        if save_results:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"normalization_eval_{ts}.xlsx"
            results.to_excel(out_path, index=False)
            logger.info("Results saved to: %s", out_path)

        return evaluator

    @staticmethod
    def available_methods() -> List[str]:
        """Return sorted list of available normalization method names."""
        return normalization_method_names()
