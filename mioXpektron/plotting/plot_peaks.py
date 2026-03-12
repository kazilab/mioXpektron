import logging
import os
import re
import glob
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ..normalization.normalization import tic_normalization


@dataclass
class PlotPeaksConfig:
    """
    Configuration for PlotPeaks class.

    Parameters
    ----------
    data_dir : str
        Directory containing spectra files.
    file_pattern : str, default "*.txt"
        Glob pattern for matching spectrum files.
    mz_min : float, default 0.0
        Minimum m/z value for the plotting window.
    mz_max : float, default 1000.0
        Maximum m/z value for the plotting window.
    norm_tic : bool, default False
        If True, normalize intensities by total ion count.
    bin_width : float, default 0.001
        Bin width for interpolation grid.
    alpha : float, default 0.18
        Transparency for individual spectra lines.
    show_median : bool, default True
        If True, overlay median curves on the plot.
    show_group_cumulative : bool, default True
        If True, create cumulative intensity plot.
    figsize : tuple, default (10, 6)
        Figure size for overlay plot.
    cumulative_figsize : tuple, default (10, 4)
        Figure size for cumulative plot.
    color_map : dict, optional
        Dictionary mapping group names to colors.
    save_fig : bool, default False
        If True, save figures as PDF files.
    save_path : str, default "../output_files/plots"
        Directory path where PDF files will be saved.
    """

    # Data parameters
    data_dir: str
    file_pattern: str = "*.txt"
    mz_min: float = 0.0
    mz_max: float = 1000.0

    # Processing parameters
    norm_tic: bool = False
    bin_width: float = 0.001

    # Plot parameters
    alpha: float = 0.18
    show_median: bool = True
    show_group_cumulative: bool = True
    figsize: Tuple[int, int] = (10, 6)
    cumulative_figsize: Tuple[int, int] = (10, 4)

    # Color map for groups
    color_map: Optional[Dict[str, str]] = None

    # Save parameters
    save_fig: bool = False
    save_path: str = "output_files/plots"

    def __post_init__(self):
        """Set default color map if not provided."""
        if self.color_map is None:
            self.color_map = {
                "Cancer": "#d62728",    # red
                "Control": "#1f77b4",   # blue
                "Unknown": "#7f7f7f"    # grey
            }


class PlotPeaks:
    """
    Class for plotting overlapping peaks from multiple spectra files.

    Features:
    - Load and group spectra by inferred labels (Cancer/Control/Unknown)
    - Overlay individual spectra with customizable transparency
    - Plot per-group median curves
    - Plot cumulative intensity by group
    - Flexible configuration through PlotPeaksConfig

    Example:
    --------
    >>> config = PlotPeaksConfig(
    ...     data_dir="data/spectra",
    ...     mz_min=100.0,
    ...     mz_max=200.0,
    ...     norm_tic=True,
    ...     save_fig=True,
    ...     save_path="../output_files/plots"
    ... )
    >>> plotter = PlotPeaks(config)
    >>> plotter.load_data()
    >>> plotter.plot_overlay()
    >>> plotter.plot_cumulative()
    """

    def __init__(self, config: Optional[PlotPeaksConfig] = None):
        """
        Initialize PlotPeaks.

        Parameters
        ----------
        config : PlotPeaksConfig, optional
            Configuration object. If None, must set attributes manually.
        """
        self.config = config
        self.files: List[str] = []
        self.grouped_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "Cancer": [],
            "Control": [],
            "Unknown": []
        }
        self._group_inference_func: Callable[[str], str] = self._default_group_inference

    def _default_group_inference(self, path: str) -> str:
        """
        Default group inference from filename.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        str
            Group label: "Cancer", "Control", or "Unknown".
        """
        name = os.path.basename(path).lower()
        if "_cc" in name:
            return "Cancer"
        if "_ct" in name:
            return "Control"
        return "Unknown"

    def set_group_inference(self, func: Callable[[str], str]):
        """
        Set custom group inference function.

        Parameters
        ----------
        func : callable
            Function that takes a file path and returns group label.
        """
        self._group_inference_func = func

    @staticmethod
    def _find_column(df: pl.DataFrame, variations: List[str]) -> str:
        """
        Find column name matching any variation (case-insensitive).

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame to search.
        variations : List[str]
            List of possible column name variations.

        Returns
        -------
        str
            Matching column name.

        Raises
        ------
        ValueError
            If no matching column is found.
        """
        columns_lower = {col.lower(): col for col in df.columns}
        for var in variations:
            if var.lower() in columns_lower:
                return columns_lower[var.lower()]
        raise ValueError(
            f"Could not find column matching any of {variations}. "
            f"Available columns: {df.columns}"
        )

    @staticmethod
    def load_window(
        file_path: str,
        mz_min: float,
        mz_max: float,
        norm_tic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read one spectrum and return (m/z, intensity) in the requested window.

        Parameters
        ----------
        file_path : str
            Path to a tab or comma-separated spectrum with columns for m/z and intensity.
            Column names are case-insensitive and support variations:
            - m/z: "mz", "m/z", "M/Z", "MZ", "Mz"
            - intensity: "intensity", "Intensity", "INTENSITY", "int", "Int"
        mz_min, mz_max : float
            Inclusive m/z window to extract.
        norm_tic : bool, default False
            If True, normalize intensities by total ion count (sum to 1).

        Returns
        -------
        mz : np.ndarray
        inten : np.ndarray
            Intensities scaled by 1e6 (to keep values readable on plots).
        """
        import warnings
        import os

        # Try tab-separated first, then comma-separated
        df = None
        for separator in ["\t", ","]:
            try:
                try:
                    df = pl.read_csv(file_path, separator=separator, comment_prefix="#")
                except TypeError:
                    # For older Polars versions using `comment_char`
                    df = pl.read_csv(file_path, separator=separator, comment_char="#")

                # Check if we got multiple columns (successful parse)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue

        if df is None or len(df.columns) == 1:
            raise ValueError(
                f"Could not parse file {file_path}. "
                f"Tried tab and comma separators. "
                f"Columns found: {df.columns if df is not None else 'None'}"
            )

        # Find m/z column (case-insensitive, multiple variations)
        mz_variations = ["mz", "m/z", "M/Z", "MZ", "Mz"]
        mz_col = PlotPeaks._find_column(df, mz_variations)

        # Find intensity column (case-insensitive, multiple variations)
        intensity_variations = ["intensity", "Intensity", "INTENSITY", "int", "Int"]
        inten_col = PlotPeaks._find_column(df, intensity_variations)

        # Validate m/z column type and content
        if df[mz_col].dtype == pl.String:
            filename = os.path.basename(file_path)
            warnings.warn(
                f"\n⚠️  SKIPPING '{filename}': m/z column has String type (corrupted data). "
                f"The m/z values are missing or invalid.\n",
                UserWarning,
                stacklevel=3
            )
            return np.array([]), np.array([])

        # Check if m/z column is all null
        if df[mz_col].null_count() == len(df):
            filename = os.path.basename(file_path)
            warnings.warn(
                f"\n⚠️  SKIPPING '{filename}': m/z column is entirely empty (all null values).\n",
                UserWarning,
                stacklevel=3
            )
            return np.array([]), np.array([])

        # Try to cast m/z to Float64 if it's not already numeric
        try:
            if df[mz_col].dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                df = df.with_columns(pl.col(mz_col).cast(pl.Float64))
        except Exception as e:
            filename = os.path.basename(file_path)
            warnings.warn(
                f"\n⚠️  SKIPPING '{filename}': Cannot convert m/z column to numeric type. Error: {e}\n",
                UserWarning,
                stacklevel=3
            )
            return np.array([]), np.array([])

        # Filter by m/z range
        try:
            df = df.filter((pl.col(mz_col) >= mz_min) & (pl.col(mz_col) <= mz_max))
        except Exception as e:
            filename = os.path.basename(file_path)
            warnings.warn(
                f"\n⚠️  SKIPPING '{filename}': Error filtering m/z range. Error: {e}\n",
                UserWarning,
                stacklevel=3
            )
            return np.array([]), np.array([])

        mz = df[mz_col].to_numpy()
        inten = df[inten_col].to_numpy()
        if norm_tic:
            inten = tic_normalization(inten, target_tic=1e6)
        return mz, inten

    def load_data(self) -> None:
        """
        Load all files matching the pattern and group them by inferred labels.

        Raises
        ------
        RuntimeError
            If no files match the pattern.
        """
        if self.config is None:
            raise ValueError("Config must be set before loading data")

        # Find files
        pattern = os.path.join(self.config.data_dir, self.config.file_pattern)
        self.files = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise RuntimeError(f"No files matched: {pattern}")

        # Reset grouped curves
        self.grouped_curves = {"Cancer": [], "Control": [], "Unknown": []}

        # Load and group
        for fp in self.files:
            mz, inten = self.load_window(
                fp,
                self.config.mz_min,
                self.config.mz_max,
                norm_tic=self.config.norm_tic
            )
            if mz.size == 0:
                continue
            group = self._group_inference_func(fp)
            self.grouped_curves[group].append((mz, inten))

    def get_group_counts(self) -> Dict[str, int]:
        """
        Get counts of spectra per group.

        Returns
        -------
        dict
            Dictionary with group names as keys and counts as values.
        """
        return {
            group: len(curves)
            for group, curves in self.grouped_curves.items()
        }

    def _save_figure(self, fig: Figure, filename: str) -> None:
        """
        Save figure as PDF to the configured save path.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filename : str
            Filename for the saved PDF (without extension).
        """
        if not self.config.save_fig:
            return

        # Create directory if it doesn't exist
        os.makedirs(self.config.save_path, exist_ok=True)

        # Save as PDF
        filepath = os.path.join(self.config.save_path, f"{filename}.pdf")
        fig.savefig(filepath, format='pdf', bbox_inches='tight')
        logger.info("Figure saved to: %s", filepath)

    def _compute_group_median(self, group: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute median intensity curve for a group.

        Parameters
        ----------
        group : str
            Group name.

        Returns
        -------
        mz_common : np.ndarray
            Common m/z grid.
        median_curve : np.ndarray
            Median intensity values.
        """
        if not self.grouped_curves[group]:
            return np.array([]), np.array([])

        mz_common = np.arange(
            self.config.mz_min,
            self.config.mz_max + self.config.bin_width,
            self.config.bin_width
        )

        interp = []
        for mz, inten in self.grouped_curves[group]:
            interp.append(np.interp(mz_common, mz, inten, left=0.0, right=0.0))
        interp = np.vstack(interp)  # (n, bins)
        median_curve = np.median(interp, axis=0)

        return mz_common, median_curve

    def _compute_group_cumulative(self, group: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulative intensity curve for a group.

        Parameters
        ----------
        group : str
            Group name.

        Returns
        -------
        mz_common : np.ndarray
            Common m/z grid.
        cumulative_curve : np.ndarray
            Cumulative intensity values.
        """
        mz_common = np.arange(
            self.config.mz_min,
            self.config.mz_max + self.config.bin_width,
            self.config.bin_width
        )

        if not self.grouped_curves[group]:
            return mz_common, np.zeros_like(mz_common)

        acc = []
        for mz, inten in self.grouped_curves[group]:
            acc.append(np.interp(mz_common, mz, inten, left=0.0, right=0.0))
        cumulative_curve = np.sum(np.vstack(acc), axis=0)

        return mz_common, cumulative_curve

    def plot_overlay(self, ax: Optional[plt.Axes] = None, show: bool = True) -> Figure:
        """
        Plot overlapping spectra with optional median curves.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool, default True
            If True, call plt.show() at the end.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        if self.config is None:
            raise ValueError("Config must be set before plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize)
        else:
            fig = ax.get_figure()

        counts = self.get_group_counts()

        # Plot individual spectra
        for group, curves in self.grouped_curves.items():
            for mz, inten in curves:
                ax.plot(
                    mz, inten,
                    linewidth=0.7,
                    alpha=self.config.alpha,
                    color=self.config.color_map[group]
                )

        # Plot median curves if requested
        if self.config.show_median:
            for group in ["Cancer", "Control"]:
                if not self.grouped_curves[group]:
                    continue
                mz_common, median_curve = self._compute_group_median(group)
                ax.plot(
                    mz_common, median_curve,
                    color=self.config.color_map[group],
                    lw=2.5,
                    alpha=0.9
                )

        # Configure plot
        ax.set_xlim(self.config.mz_min, self.config.mz_max)
        ax.set_xlabel("m/z")
        ax.set_ylabel(
            "Normalized intensity" if self.config.norm_tic
            else "Intensity (scaled ×1e6)"
        )

        # Title with counts
        title_parts = [f"{group} n={count}" for group, count in counts.items() if count > 0]
        title_counts = ", ".join(title_parts)
        ax.set_title(
            f"Spectra overlay by group ({self.config.mz_min}–{self.config.mz_max} m/z) | {title_counts}"
        )

        # Legend
        handles = []
        labels = []
        for group in ["Cancer", "Control", "Unknown"]:
            if self.grouped_curves[group]:
                h, = ax.plot([], [], color=self.config.color_map[group], lw=3, label=group)
                handles.append(h)
                labels.append(group)

        if handles:
            ax.legend(handles=handles, labels=labels, frameon=False)

        fig.tight_layout()

        # Save figure if requested
        self._save_figure(fig, f"overlay_plot_mz_{self.config.mz_min}_{self.config.mz_max}")

        if show:
            plt.show()

        return fig

    def plot_cumulative(self, ax: Optional[plt.Axes] = None, show: bool = True) -> Figure:
        """
        Plot cumulative intensity curves by group.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        show : bool, default True
            If True, call plt.show() at the end.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        if self.config is None:
            raise ValueError("Config must be set before plotting")

        if not self.config.show_group_cumulative:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.cumulative_figsize)
        else:
            fig = ax.get_figure()

        counts = self.get_group_counts()

        # Plot cumulative curves for each group
        for group in ["Cancer", "Control", "Unknown"]:
            if not self.grouped_curves[group]:
                continue

            mz_common, cum_curve = self._compute_group_cumulative(group)

            if cum_curve.sum() > 0:
                linestyle = "--" if group == "Unknown" else "-"
                linewidth = 1.4 if group == "Unknown" else 1.8
                ax.plot(
                    mz_common, cum_curve,
                    label=f"{group} (n={counts[group]})",
                    color=self.config.color_map[group],
                    lw=linewidth,
                    ls=linestyle
                )

        # Configure plot
        ax.set_xlim(self.config.mz_min, self.config.mz_max)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Cumulative intensity (scaled)")
        ax.set_title(
            f"Cumulative intensity by group ({self.config.mz_min}–{self.config.mz_max} m/z)"
        )
        ax.legend(frameon=False)

        fig.tight_layout()

        # Save figure if requested
        self._save_figure(fig, f"cumulative_plot_mz_{self.config.mz_min}_{self.config.mz_max}")

        if show:
            plt.show()

        return fig

    def plot_all(self) -> Tuple[Figure, Figure]:
        """
        Convenience method to plot both overlay and cumulative plots.

        Returns
        -------
        fig_overlay : matplotlib.figure.Figure
            The overlay plot figure.
        fig_cumulative : matplotlib.figure.Figure
            The cumulative plot figure (or None if disabled).
        """
        fig_overlay = self.plot_overlay(show=False)
        fig_cumulative = self.plot_cumulative(show=False) if self.config.show_group_cumulative else None
        plt.show()
        return fig_overlay, fig_cumulative


# ==================== Backwards Compatibility Function ====================

def plot_overlapping_peaks(
    data_dir: str,
    file_pattern: str,
    mz_min: float,
    mz_max: float,
    norm_tic: bool = False,
    alpha: float = 0.18,
    bin_width: float = 0.001,
    show_median: bool = True,
    show_group_cumulative: bool = True,
):
    """
    Overlay spectra with two colors (Cancer vs Control) inferred from file names.

    DEPRECATED: This function is maintained for backwards compatibility.
    Use PlotPeaks class for new code.

    Parameters
    ----------
    data_dir : str
        Directory containing spectra.
    file_pattern : str
        Glob pattern (e.g., "*.txt").
    mz_min, mz_max : float
        m/z window to visualize.
    norm_tic : bool, default False
        Normalize each spectrum by its TIC prior to plotting.
    alpha : float, default 0.18
        Line transparency for individual spectra.
    bin_width : float, default 0.001
        Common grid step for interpolation (used for medians/cumulative plots).
    show_median : bool, default True
        If True, overlay per-group median curves (thicker lines).
    show_group_cumulative : bool, default True
        If True, plot per-group cumulative intensity curves on a separate figure.

    Notes
    -----
    - Group detection is based on substrings in filenames: "_CC" (Cancer), "_CT" (Control).
    - Files without these markers are labeled "Unknown" and plotted in grey.

    Examples
    --------
    >>> # New recommended approach
    >>> config = PlotPeaksConfig(
    ...     data_dir="data/spectra",
    ...     mz_min=100.0,
    ...     mz_max=200.0
    ... )
    >>> plotter = PlotPeaks(config)
    >>> plotter.load_data()
    >>> plotter.plot_all()
    """
    # Use the new class internally for backwards compatibility
    config = PlotPeaksConfig(
        data_dir=data_dir,
        file_pattern=file_pattern,
        mz_min=mz_min,
        mz_max=mz_max,
        norm_tic=norm_tic,
        alpha=alpha,
        bin_width=bin_width,
        show_median=show_median,
        show_group_cumulative=show_group_cumulative
    )

    plotter = PlotPeaks(config)
    plotter.load_data()
    plotter.plot_all()
