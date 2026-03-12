# Requirements:
#   pip install polars scipy numpy
# File:
#   'blood serum_CT-1a_1.txt' (tab-delimited), with header: Channel  m/z  Intensity
#   comment lines start with '#'

import logging

import polars as pl
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

def _interp(x1, y1, x2, y2, y_target):
    """Linear interpolation to get x at y_target between (x1,y1) and (x2,y2)."""
    if y2 == y1:
        return x1  # degenerate, return left point
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)

def _fwhm_from_peak(mz: np.ndarray, y: np.ndarray, p: int) -> float | None:
    """
    Compute FWHM for peak at index p using linear interpolation of half-maximum crossings.
    Returns None if the half-maximum is not bracketed on either side.
    """
    peak_y = y[p]
    if peak_y <= 0:
        return None
    half = peak_y / 2.0

    # Left crossing
    i = p
    while i > 0 and y[i] >= half:
        i -= 1
    if i == p:  # immediate drop (flat/edge)
        return None
    if i == 0 and y[i] > half:
        return None
    xL = _interp(mz[i], y[i], mz[i+1], y[i+1], half)

    # Right crossing
    j = p
    n = len(y)
    while j < n - 1 and y[j] >= half:
        j += 1
    if j == p:
        return None
    if j == n - 1 and y[j] > half:
        return None
    xR = _interp(mz[j-1], y[j-1], mz[j], y[j], half)

    width = xR - xL
    return width if width > 0 else None

def load_and_measure_peaks_polars(
    path: str,
    mz_min: float,
    mz_max: float,
    min_height: float | None = None,
    prominence: float | None = None,
    distance_pts: int | None = None,
    save_csv: str | None = None,
) -> pl.DataFrame:
    """
    Read ToF-SIMS TXT with columns [Channel, m/z, Intensity], filter m/z range,
    detect peaks, and compute [m/z_center, FWHM, height].

    Parameters
    ----------
    path : str
        Path to TXT (tab-delimited). Lines starting with '#' are ignored.
    mz_min, mz_max : float
        m/z selection window (inclusive).
    min_height : float or None
        Minimum peak height for detection (scipy find_peaks 'height').
    prominence : float or None
        Peak prominence threshold (recommended for robustness).
    distance_pts : int or None
        Minimum horizontal distance (in data points) between neighboring peaks.
    save_csv : str or None
        If provided, save the results to this CSV path.

    Returns
    -------
    Polars DataFrame with columns: 'm/z', 'FWHM', 'height'
    """
    # Read with Polars (handle comments and tab separator)
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        comment_char="#",
        infer_schema_length=10000,
    )
    # normalize column names
    df = df.rename({c: c.strip().lower() for c in df.columns})

    # sanity check expected columns
    expected = {"channel", "m/z", "intensity"}
    missing = expected.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found: {df.columns}")

    # filter range and sort by m/z (important if input is not strictly sorted)
    sub = (
        df.filter((pl.col("m/z") >= mz_min) & (pl.col("m/z") <= mz_max))
          .sort("m/z")
    )

    if sub.height == 0:
        return pl.DataFrame({"m/z": [], "FWHM": [], "height": []})

    mz = sub["m/z"].to_numpy()
    y = sub["intensity"].to_numpy()

    # Peak detection (SciPy)
    kwargs = {}
    if min_height is not None:
        kwargs["height"] = min_height
    if prominence is not None:
        kwargs["prominence"] = prominence
    if distance_pts is not None:
        kwargs["distance"] = distance_pts

    peak_idx, props = find_peaks(y, **kwargs)

    centers = []
    heights = []
    widths = []
    for p in peak_idx:
        centers.append(float(mz[p]))
        # prefer 'peak height' from y-array; if 'peak_heights' supplied, it equals y[p]
        heights.append(float(y[p]))
        w = _fwhm_from_peak(mz, y, p)
        widths.append(float(w) if w is not None else np.nan)

    out = pl.DataFrame({"m/z": centers, "FWHM": widths, "height": heights})

    if save_csv:
        out.write_csv(save_csv)

    return out

# ------------------ Example usage ------------------
if __name__ == "__main__":
    results = load_and_measure_peaks_polars(
        path="blood serum_CT-1a_1.txt",
        mz_min=0.0275,     # example window
        mz_max=0.0286,
        min_height=1,      # set based on your noise level
        prominence=0.5,    # often more robust than height alone
        distance_pts=5,    # avoid double-counting very close peaks
        save_csv=None
    )
    logger.info(results)
