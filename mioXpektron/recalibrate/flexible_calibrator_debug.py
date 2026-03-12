"""
Debug version of FlexibleCalibrator with diagnostic logging.

Inherits all functionality from ``flexible_calibrator`` and overrides only the
peak-detection helpers that benefit from verbose diagnostics. This eliminates
code duplication while preserving the diagnostic output needed during
development and validation.

Version: 2.1.0-debug
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

# Re-export everything from the production module so that existing imports
# (``from .flexible_calibrator_debug import FlexibleCalibrator``) keep working.
from .flexible_calibrator import (  # noqa: F401 – re-exports
    FlexibleCalibConfig,
    FlexibleCalibrator,
    CalibrationMethod,
    _ppm_to_da,
    _fit_gaussian_peak,
    _fit_voigt_peak,
    _parabolic_peak_center as _parabolic_peak_center_base,
    _enhanced_pick_channels as _enhanced_pick_channels_base,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic overrides
# ---------------------------------------------------------------------------

def _parabolic_peak_center(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    peak_idx: int,
) -> Optional[float]:
    """Parabolic interpolation with debug logging at each decision point."""
    if peak_idx == 0 or peak_idx == len(x) - 1:
        logger.debug(
            "Parabolic fit failed: peak at edge (peak_idx=%d, len=%d)",
            peak_idx, len(x),
        )
        return None

    x1, x2, x3 = x[peak_idx - 1], x[peak_idx], x[peak_idx + 1]
    y1, y2, y3 = y[peak_idx - 1], y[peak_idx], y[peak_idx + 1]

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-10:
        logger.debug(
            "Parabolic fit failed: denominator too small (%.2e)", abs(denom),
        )
        return None

    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom

    if abs(A) < 1e-10:
        logger.debug(
            "Parabolic fit failed: A coefficient too small (%.2e)", abs(A),
        )
        return None

    xc = -B / (2 * A)

    if xc < x1 or xc > x3:
        logger.debug(
            "Parabolic fit failed: center out of range "
            "(xc=%.4f, range=[%.4f, %.4f])",
            xc, x1, x3,
        )
        return None

    logger.debug(
        "Parabolic fit success: peak_mz=%.4f -> refined_mz=%.4f (shift=%.4f)",
        x2, xc, xc - x2,
    )
    return float(xc)


def _enhanced_pick_channels(
    df: pd.DataFrame,
    targets: npt.NDArray[np.float64],
    tol_da: Optional[float],
    tol_ppm: Optional[float],
    method: str = "gaussian",
) -> List[int]:
    """Enhanced peak picking with diagnostic counters and logging."""
    mz = df["m/z"].astype("float64").to_numpy()
    I = df["Intensity"].astype("float64").to_numpy()
    ch = df["Channel"].to_numpy()

    out: List[int] = []

    # Diagnostic counters
    parabolic_success = 0
    parabolic_fallback = 0
    parabolic_same_as_max = 0
    parabolic_different_from_max = 0

    logger.info("[DIAGNOSTIC] Peak picking method: %s", method)

    for target_idx, xi in enumerate(targets):
        tol = (
            _ppm_to_da(xi, tol_ppm)
            if tol_ppm is not None
            else (tol_da if tol_da else 2.0)
        )
        left, right = xi - tol, xi + tol
        mask = (mz >= left) & (mz <= right)

        if not mask.any():
            logger.debug(
                "[DIAGNOSTIC] Target %.4f: No peaks found in window", xi,
            )
            out.append(np.nan)
            continue

        idxs = np.flatnonzero(mask)
        mzw = mz[idxs]
        Iw = I[idxs]
        chw = ch[idxs]

        k_local = int(np.nanargmax(Iw))
        peak_mz = float(mzw[k_local])
        max_ch = chw[k_local]

        logger.debug(
            "[DIAGNOSTIC] Target %.4f: Found %d points in window, "
            "max at mz=%.4f, ch=%s",
            xi, len(mzw), peak_mz, max_ch,
        )

        if method == "max":
            final_ch = chw[k_local]
            logger.debug(
                "[DIAGNOSTIC] Target %.4f: Using MAX method -> ch=%s",
                xi, final_ch,
            )

        elif method == "centroid":
            wsum = Iw.sum()
            if wsum > 0:
                mz_c = float((mzw * Iw).sum() / wsum)
                nearest = int(np.argmin(np.abs(mzw - mz_c)))
                final_ch = chw[nearest]
                logger.debug(
                    "[DIAGNOSTIC] Target %.4f: CENTROID mz=%.4f -> ch=%s "
                    "(vs max ch=%s)",
                    xi, mz_c, final_ch, max_ch,
                )
            else:
                final_ch = chw[k_local]
                logger.debug(
                    "[DIAGNOSTIC] Target %.4f: CENTROID fallback to MAX "
                    "-> ch=%s",
                    xi, final_ch,
                )

        elif method == "parabolic":
            center = _parabolic_peak_center(mzw, Iw, k_local)
            if center is not None:
                parabolic_success += 1
                nearest = int(np.argmin(np.abs(mzw - center)))
                final_ch = chw[nearest]
                if final_ch == max_ch:
                    parabolic_same_as_max += 1
                else:
                    parabolic_different_from_max += 1
                logger.info(
                    "[DIAGNOSTIC] Target %.4f: PARABOLIC refined_mz=%.4f "
                    "-> ch=%s (max_ch=%s, SAME=%s)",
                    xi, center, final_ch, max_ch, final_ch == max_ch,
                )
            else:
                parabolic_fallback += 1
                final_ch = chw[k_local]
                logger.info(
                    "[DIAGNOSTIC] Target %.4f: PARABOLIC FALLBACK to MAX "
                    "-> ch=%s",
                    xi, final_ch,
                )

        elif method == "gaussian":
            center = _fit_gaussian_peak(mzw, Iw, peak_mz)
            if center is not None:
                nearest = int(np.argmin(np.abs(mzw - center)))
                final_ch = chw[nearest]
            else:
                wsum = Iw.sum()
                if wsum > 0:
                    mz_c = float((mzw * Iw).sum() / wsum)
                    nearest = int(np.argmin(np.abs(mzw - mz_c)))
                    final_ch = chw[nearest]
                else:
                    final_ch = chw[k_local]

        elif method == "voigt":
            center = _fit_voigt_peak(mzw, Iw, peak_mz)
            if center is not None:
                nearest = int(np.argmin(np.abs(mzw - center)))
                final_ch = chw[nearest]
            else:
                center = _fit_gaussian_peak(mzw, Iw, peak_mz)
                if center is not None:
                    nearest = int(np.argmin(np.abs(mzw - center)))
                    final_ch = chw[nearest]
                else:
                    final_ch = chw[k_local]
        else:
            final_ch = chw[k_local]

        out.append(int(final_ch))

    # Print summary statistics for parabolic method
    if method == "parabolic":
        total_targets = len(targets)
        logger.info("[DIAGNOSTIC SUMMARY] " + "=" * 42)
        logger.info(
            "[DIAGNOSTIC SUMMARY] Total targets: %d", total_targets,
        )
        logger.info(
            "[DIAGNOSTIC SUMMARY] Parabolic fit succeeded: %d/%d (%.1f%%)",
            parabolic_success, total_targets,
            100 * parabolic_success / max(1, total_targets),
        )
        logger.info(
            "[DIAGNOSTIC SUMMARY] Parabolic fit failed (fallback to max): "
            "%d/%d (%.1f%%)",
            parabolic_fallback, total_targets,
            100 * parabolic_fallback / max(1, total_targets),
        )
        if parabolic_success > 0:
            logger.info("[DIAGNOSTIC SUMMARY] When parabolic succeeded:")
            logger.info(
                "[DIAGNOSTIC SUMMARY]   - Same channel as max: %d/%d (%.1f%%)",
                parabolic_same_as_max, parabolic_success,
                100 * parabolic_same_as_max / max(1, parabolic_success),
            )
            logger.info(
                "[DIAGNOSTIC SUMMARY]   - Different channel from max: "
                "%d/%d (%.1f%%)",
                parabolic_different_from_max, parabolic_success,
                100 * parabolic_different_from_max / max(1, parabolic_success),
            )
        logger.info("[DIAGNOSTIC SUMMARY] " + "=" * 42)

    return out


# ---------------------------------------------------------------------------
# FlexibleCalibratorDebug – thin subclass that wires in the diagnostic helpers
# ---------------------------------------------------------------------------

class FlexibleCalibratorDebug(FlexibleCalibrator):
    """Debug variant of :class:`FlexibleCalibrator` with verbose peak-picking.

    Inherits all calibration logic and overrides only ``_autodetect_channels``
    to route through the diagnostic versions of ``_enhanced_pick_channels``
    and ``_parabolic_peak_center``.
    """

    def _autodetect_channels(
        self,
        files,
        ref_masses: npt.NDArray[np.float64],
    ) -> Dict[str, list]:
        """Autodetect calibrant channels using diagnostic peak picking."""
        import os
        from tqdm import tqdm
        from .flexible_calibrator import _enhanced_bootstrap_channels

        autodetected: Dict[str, list] = {}

        for fp in tqdm(files, desc="Autodetecting channels (debug)"):
            try:
                df = pd.read_csv(fp, sep="\t", header=0, comment="#")
            except Exception as e:
                logger.warning(
                    "%s: Failed to read - %s", os.path.basename(fp), e,
                )
                continue

            fname = os.path.basename(fp)

            if "Channel" not in df.columns:
                continue

            if (
                self.config.prefer_recompute_from_channel
                or self.config.autodetect_strategy == "bootstrap"
                or "m/z" not in df.columns
            ):
                ch = df["Channel"].to_numpy()
                y = df["Intensity"].astype("float64").to_numpy()
                autodetected[fname] = _enhanced_bootstrap_channels(
                    ch, y, ref_masses,
                )
            else:
                # Use the diagnostic version defined in this module
                autodetected[fname] = _enhanced_pick_channels(
                    df,
                    ref_masses,
                    self.config.autodetect_tol_da,
                    self.config.autodetect_tol_ppm,
                    self.config.autodetect_method,
                )

        if not autodetected:
            raise RuntimeError("Autodetection failed")

        logger.info(
            "Autodetected channels for %d/%d files",
            len(autodetected), len(files),
        )
        return autodetected
