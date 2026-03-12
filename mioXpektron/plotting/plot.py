from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

OUTPUT_DIR = Path("output_files/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class PlotPeak:
    """Helper for plotting raw and processed spectra.

    Parameters
    ----------
    mz_values : array-like of shape (n,)
        m/z axis aligned with the supplied intensities.
    raw_intensities : array-like of shape (n,)
        Primary intensity trace used for plotting.
    sample_name : str
        Label shown on the plot title.
    group : str | None, optional
        Additional grouping label appended to the title.
    corrected_intensities : array-like of shape (n,), optional
        Denoised/baseline-corrected intensities. When provided, displayed as the
        comparison trace and used as the default signal for peak detection.
    """

    mz_values: Iterable[float]
    raw_intensities: Iterable[float]
    sample_name: str
    group: Optional[str] = None
    corrected_intensities: Optional[Iterable[float]] = None

    def __post_init__(self) -> None:
        mz = np.asarray(self.mz_values, dtype=float)
        raw = np.asarray(self.raw_intensities, dtype=float)
        if mz.shape != raw.shape:
            raise ValueError("mz_values and raw_intensities must share the same shape")
        object.__setattr__(self, "_mz", mz)
        object.__setattr__(self, "_raw", raw)

        if self.corrected_intensities is not None:
            corr = np.asarray(self.corrected_intensities, dtype=float)
            if corr.shape != raw.shape:
                raise ValueError("corrected_intensities must match raw_intensities shape")
        else:
            corr = None
        object.__setattr__(self, "_corr", corr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot(
        self,
        *,
        mz_min: Optional[float] = None,
        mz_max: Optional[float] = None,
        show_peaks: bool = True,
        peak_height: Optional[float] = None,
        peak_prominence: Optional[float] = None,
        min_peak_width: float = 1,
        max_peak_width: Optional[float] = None,
        corrected_intensities: Optional[Iterable[float]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_plot: Optional[bool]=True
    ):
        """Plot raw and optional corrected spectra for the configured sample."""

        mz = self._mz
        raw = self._raw
        corr = self._corr if corrected_intensities is None else np.asarray(corrected_intensities, dtype=float)
        if corr is not None and corr.shape != raw.shape:
            raise ValueError("corrected_intensities must match raw data shape")

        mask = np.ones_like(mz, dtype=bool)
        if mz_min is not None:
            mask &= mz >= mz_min
        if mz_max is not None:
            mask &= mz <= mz_max

        mz_region = mz[mask]
        raw_region = raw[mask]
        corr_region = corr[mask] if corr is not None else None

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mz_region, raw_region, label="Raw Spectrum", color="0.5", alpha=0.6)
        if corr_region is not None:
            ax.plot(mz_region, corr_region, label="Corrected Spectrum", color="navy")

        if corr_region is not None and show_peaks:
            peaks, properties = signal.find_peaks(
                corr_region,
                height=peak_height,
                prominence=peak_prominence,
                width=(min_peak_width, max_peak_width),
            )
            if peaks.size:
                ax.plot(mz_region[peaks], corr_region[peaks], "rx", label="Detected Peaks")
                for peak_idx in peaks:
                    ax.text(
                        mz_region[peak_idx],
                        corr_region[peak_idx],
                        f"{mz_region[peak_idx]:.1f}",
                        color="red",
                        fontsize=8,
                        rotation=90,
                        va="bottom",
                    )

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        tag = f" |{self.group}|" if self.group is not None else ""
        ax.set_title(f"Spectrum, {self.sample_name}{tag}")
        ax.legend()
        fig.tight_layout()
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"Intensity_plot_{self.sample_name}_{timestamp}.pdf"
            fig.savefig(out_path, bbox_inches="tight")
        plt.show()
        return ax

    def zoom(
        self,
        report: pd.DataFrame,
        *,
        xlim: Tuple[float, float],
        intensity: Optional[Iterable[float]] = None,
        window_ppm: float = 50.0,
        figsize: Tuple[int, int] = (6, 4),
        save_plot: Optional[bool] = True,
    ):
        """Plot a zoomed segment with calibration-peak diagnostics."""

        left, right = xlim
        mz_raw = self._mz
        inten_source = (
            np.asarray(intensity, dtype=float)
            if intensity is not None
            else (self._corr if self._corr is not None else self._raw)
        )
        if inten_source.shape != mz_raw.shape:
            raise ValueError("intensity must align with mz_values")

        mask = (mz_raw >= left) & (mz_raw <= right)
        if not np.any(mask):
            raise ValueError("No data points fall inside the requested x-range.")
        mz_win = mz_raw[mask]
        inten_win = inten_source[mask]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mz_win, inten_win, lw=0.8, color="0.2")

        for _, row in report.iterrows():
            m_theory = row["Reference (u)"]
            if not (left <= m_theory <= right):
                continue
            m_meas = row["Measured (u)"]
            err_ppm = row["Error (ppm)"]

            d = m_theory * window_ppm * 1e-6
            ax.axvspan(m_theory - d, m_theory + d, alpha=0.10, color="tab:blue")
            ax.axvline(m_theory, ls="--", lw=0.8, color="tab:blue")

            y_meas = np.interp(m_meas, mz_raw, inten_source)
            ax.scatter(m_meas, y_meas, zorder=5, color="tab:red")
            ax.text(
                m_meas,
                y_meas * 1.05,
                f"{err_ppm:+.1f} ppm",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlim(left, right)
        ax.set_xlabel("Uncalibrated m/z (u) or channel index")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Calibration-peak check: {left} – {right} u")
        ax.ticklabel_format(style="plain", axis="x")
        fig.tight_layout()
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"Intensity_zplot_{self.sample_name}_{timestamp}.pdf"
            plt.savefig(out_path, bbox_inches="tight")
        return fig
