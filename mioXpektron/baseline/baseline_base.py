from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import inspect
import io
import re
import warnings

import numpy as np
import polars as pl
from scipy import signal, ndimage

# pybaselines
try:
    from pybaselines import Baseline
except Exception as e:  # pragma: no cover
    raise ImportError("pybaselines is required. pip install pybaselines") from e

# ---------------------------------------------------------------------
# Universal baseline wrapper (ToF‑SIMS friendly)
# ---------------------------------------------------------------------

def baseline_method_names() -> List[str]:
    """Return a sorted list of available baseline algorithms.

    Based on `pybaselines.Baseline` public callables, plus two custom
    filters ("median_filter", "adaptive_window") and a 'poly' alias.
    A few methods that are not 1‑D safe or impractically slow are removed.
    """
    bl = Baseline()
    skip = {"pentapy_solver", "banded_solver"}
    methods = {
        name
        for name in dir(bl)
        if (not name.startswith("_") and name not in skip and callable(getattr(bl, name)))
    }
    # extras
    methods.update({"median_filter", "adaptive_window", "poly"})
    # remove rarely stable / not applicable methods
    remove = {"collab_pls", "interp_pts", "cwt_br"}
    return sorted([m for m in methods if m not in remove])

def small_param_grid_preset(n_points: Optional[int] = None) -> Dict[str, List[Dict]]:
    """A compact parameter grid for common methods.

    Keys must match `pybaselines.Baseline` method names (plus 'poly' and our two filters).

    Parameters
    ----------
    n_points : int, optional
        Number of data points in spectrum. If provided, window_size will be
        calculated adaptively as a percentage of data size. If None, uses
        moderate defaults suitable for ~100K point spectra.

    Returns
    -------
    dict
        Parameter grid with method names as keys

    Notes
    -----
    Window sizes are calculated as:
    - Small: 0.05% of data (min 51)
    - Medium: 0.10% of data (min 101)
    - Large: 0.20% of data (min 501)

    This adaptive scaling ensures that filter methods perform consistently
    across datasets of different sizes. Fixed window sizes work poorly:
    - For 10K points: window=101 is 1.0% (OK)
    - For 1M points: window=101 is 0.01% (too small, causes jagged baselines)

    Examples
    --------
    >>> # Auto-scale for 938K point spectrum
    >>> grid = small_param_grid_preset(n_points=938000)
    >>> grid['median_filter']
    [{'window_size': 469}, {'window_size': 938}, {'window_size': 1876}]

    >>> # Use defaults for unknown size
    >>> grid = small_param_grid_preset()
    >>> grid['median_filter']
    [{'window_size': 501}, {'window_size': 1001}, {'window_size': 2001}]
    """
    # Calculate adaptive window sizes for filter methods
    if n_points is None:
        # Default: Conservative values for moderate-sized data (better than old 101/301)
        window_sizes = [501, 1001, 2001]
    else:
        # Adaptive: Scale with data size (0.05%, 0.10%, 0.20%)
        ws_small = max(51, int(n_points * 0.0005))   # 0.05% of data
        ws_medium = max(101, int(n_points * 0.001))  # 0.10% of data
        ws_large = max(501, int(n_points * 0.002))   # 0.20% of data

        # Ensure odd numbers for symmetric windows
        ws_small = ws_small if ws_small % 2 == 1 else ws_small + 1
        ws_medium = ws_medium if ws_medium % 2 == 1 else ws_medium + 1
        ws_large = ws_large if ws_large % 2 == 1 else ws_large + 1

        window_sizes = [ws_small, ws_medium, ws_large]

    return {
        # Whittaker/penalized-spline family
        "asls":   [{"lam": 1e5, "p": 0.01}, {"lam": 1e6, "p": 0.001}],
        "iasls":  [{"lam": 1e5, "p": 0.01}, {"lam": 1e6, "p": 0.001}],
        "arpls":  [{"lam": 1e6}, {"lam": 1e7}],
        "drpls":  [{"lam": 1e6}, {"lam": 1e7}],
        "airpls": [{"lam": 1e5}, {"lam": 1e6}],  # adaptive iteratively reweighted
        "aspls":  [{"lam": 1e5, "p": 0.01}, {"lam": 1e6, "p": 0.001}],

        # Polynomial family
        "modpoly":  [{"poly_order": 2}, {"poly_order": 3}, {"poly_order": 4}],
        "imodpoly": [{"poly_order": 2}, {"poly_order": 3}, {"poly_order": 4}],
        "poly":     [{"poly_order": 2}, {"poly_order": 3}, {"poly_order": 4}],

        # Simple filters (ADAPTIVE window sizes based on data size)
        "median_filter":   [{"window_size": ws} for ws in window_sizes],
        "adaptive_window": [{"window_size": ws} for ws in window_sizes],
    }

def baseline_correction(
    intensities: Union[np.ndarray, List[float]],
    method: str = "airpls",
    window_size: int = 101,
    poly_order: int = 4,
    clip_negative: bool = True,
    return_baseline: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Baseline-correct a 1‑D spectrum with `pybaselines` or custom filters.

    Parameters
    ----------
    intensities : array-like
        Raw y values.
    method : str
        Algorithm name; see :func:`baseline_method_names`.
    window_size : int
        Kernel width for the two custom filters.
    poly_order : int
        Polynomial order for the 'poly' alias.
    clip_negative : bool
        If True, negative corrected values are set to 0.
    return_baseline : bool
        If True, also return the estimated baseline.
    **kwargs :
        Forwarded to the chosen algorithm (e.g. lam=1e6, p=0.01).

    Returns
    -------
    corrected or (corrected, baseline)
    """
    y = np.asarray(intensities, dtype=float).ravel()
    if y.ndim != 1:
        raise ValueError("intensities must be 1‑D")  # pragma: no cover

    method_lower = str(method).lower()
    poly_like = {"poly", "modpoly", "imodpoly"}
    needs_rescale = method_lower in poly_like
    scale = 1.0
    y_for_baseline = y
    if needs_rescale:
        finite_mask = np.isfinite(y)
        if finite_mask.any():
            max_mag = float(np.max(np.abs(y[finite_mask])))
            if max_mag > 0:
                scale = max_mag
                y_for_baseline = y / scale
        # If no finite values are present, leave the original array; downstream
        # algorithms will fail fast and the caller already handles exceptions.

    bl = Baseline()
    _skip = {"pentapy_solver", "banded_solver"}
    dispatch = {}
    for name in dir(bl):
        if name.startswith("_") or name in _skip:
            continue
        attr = getattr(bl, name)
        if callable(attr):
            dispatch[name] = attr
    # convenience alias
    dispatch["poly"] = lambda arr, *, poly_order=poly_order, **k: bl.poly(arr, poly_order=poly_order, **k)

    # custom filters
    if method_lower == "median_filter":
        baseline = signal.medfilt(y, kernel_size=window_size)
    elif method_lower == "adaptive_window":
        baseline = ndimage.minimum_filter1d(y, size=window_size)
    else:
        func = dispatch.get(method_lower) or dispatch.get(method)
        if func is None:
            raise ValueError(f"Unknown baseline method: {method}")
        input_y = y_for_baseline if needs_rescale else y
        call_kwargs = kwargs
        if kwargs:
            try:
                sig = inspect.signature(func)
            except (TypeError, ValueError):  # pragma: no cover - builtins without introspection
                pass
            else:
                has_var_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
                )
                if not has_var_kwargs:
                    valid_names = {
                        name
                        for name, param in sig.parameters.items()
                        if param.kind in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        )
                    }
                    valid_names.discard("self")
                    valid_names.discard("y")
                    if valid_names:
                        filtered = {k: v for k, v in kwargs.items() if k in valid_names}
                    else:
                        filtered = {}
                    dropped = set(kwargs) - set(filtered)
                    if dropped:
                        warnings.warn(
                            f"Ignoring unsupported baseline parameters {sorted(dropped)} for method '{method_lower}'",
                            UserWarning,
                        )
                    call_kwargs = filtered
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                result = func(input_y, **call_kwargs)  # pybaselines returns (baseline, params)
        if isinstance(result, tuple):
            baseline, _ = result
        else:  # pragma: no cover - defensive against unexpected return types
            baseline = result
        if needs_rescale:
            baseline = baseline * scale

    if not np.all(np.isfinite(baseline)):
        raise ValueError("Baseline estimation produced non-finite values")

    corrected = y - baseline
    if clip_negative:
        corrected[corrected < 0] = 0.0

    return (corrected, baseline) if return_baseline else corrected

# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------

COL_ALIASES = {
    "channel": {"channel", "chan", "ch", "index", "idx"},
    "mz": {"m/z", "mz", "mass", "moverz", "m_over_z"},
    "intensity": {"intensity", "counts", "signal", "y", "ion_counts"},
}

_WHITESPACE_SPLIT = re.compile(r"\s+")


def _standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename = {}
    lower = {c: str(c).strip().lower() for c in df.columns}
    for std, aliases in COL_ALIASES.items():
        for original, lowered in lower.items():
            if lowered in aliases:
                rename[original] = std
                break
    if rename:
        df = df.rename(rename)

    # Sanity checks & fallbacks. Channel is optional; fabricate if missing to preserve order.
    if "mz" not in df.columns or "intensity" not in df.columns:
        raise KeyError("Input table must include 'm/z' (or alias) and 'intensity' columns.")

    if "channel" not in df.columns:
        df = df.with_row_index(name="channel", offset=1)

    df = df.select(["channel", "mz", "intensity"])
    df = df.with_columns(
        pl.col("channel").cast(pl.Int64),
        pl.col("mz").cast(pl.Float64),
        pl.col("intensity").cast(pl.Float64),
    )
    return df

def _read_with_separator(path: Path, sep: str) -> Optional[pl.DataFrame]:
    try:
        df = pl.read_csv(
            path,
            separator=sep,
            comment_prefix="#",
            infer_schema_length=4096,
            ignore_errors=False,
        )
    except Exception:
        return None
    return df if df.width >= 2 else None


def _read_whitespace_table(path: Path) -> pl.DataFrame:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(_WHITESPACE_SPLIT.sub(",", stripped))
    if not lines:
        raise ValueError(f"No tabular data found in {path}.")
    buffer = io.StringIO("\n".join(lines))
    return pl.read_csv(buffer, separator=",")


def read_spectrum_table(path: Union[str, Path]) -> pl.DataFrame:
    """Read a ToF‑SIMS table from CSV/TSV/TXT with flexible separators."""
    path = Path(path)
    df: Optional[pl.DataFrame] = None
    for sep in (",", "\t", ";"):
        df = _read_with_separator(path, sep)
        if df is not None:
            break
    if df is None:
        df = _read_whitespace_table(path)

    df = _standardize_columns(df)
    df = df.sort("channel")
    return df

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _noise_mask_from_quantile(y_raw: np.ndarray, q: float = 0.2) -> np.ndarray:
    """Boolean mask selecting the lowest-q quantile as baseline-only region."""
    finite = np.isfinite(y_raw)
    if not finite.any():
        raise ValueError("No finite values in spectrum.")
    thresh = np.nanquantile(y_raw[finite], q)
    return (y_raw <= thresh) & finite

@dataclass
class MetricResult:
    rfzn: float  # residual flat-zone noise (RMS)
    nar: float   # negative area ratio
    snr: float   # median SNR of top-K peaks
    bbi: float   # baseline bias index (median y_corr in baseline zones)
    br: float    # baseline roughness (RMS of baseline second derivative in baseline zones)
    nbc: float   # negative bin fraction

def compute_metrics(
    y_corr: np.ndarray,
    y_raw: np.ndarray,
    baseline: Optional[np.ndarray],
    x: Optional[np.ndarray],
    noise_mask: Optional[np.ndarray] = None,
    topk: int = 5,
    raw_noise_quantile: float = 0.2,
) -> MetricResult:
    """Compute RFZN, NAR, SNR, BBI, BR, NBC for a single corrected spectrum.

    Notes
    -----
    * RFZN: RMS of y_corr in baseline-only region (noise_mask). If mask is
      not supplied, it is derived from the **raw** intensities (bottom-q).
    * NAR:  sum(-y_corr[y<0]) / sum(|y_corr|); lower is better.
    * SNR:  median prominence of top-K peaks divided by noise sigma.
    * BBI:  median(y_corr[noise_mask]); lower magnitude is better.
    * BR:   RMS of d²(baseline)/dx² in baseline-only regions; requires `baseline` and `x`.
    * NBC:  fraction of points where y_corr < 0 (before clipping).
    """
    y_corr = np.asarray(y_corr, dtype=float).ravel()
    y_raw  = np.asarray(y_raw, dtype=float).ravel()
    assert y_corr.shape == y_raw.shape

    if noise_mask is None:
        noise_mask = _noise_mask_from_quantile(y_raw, raw_noise_quantile)

    # RFZN
    sigma_noise = float(np.sqrt(np.mean(y_corr[noise_mask] ** 2))) if noise_mask.any() else float("nan")

    # NAR
    denom = float(np.sum(np.abs(y_corr))) or np.nan
    neg_area = float(np.sum(-y_corr[y_corr < 0.0]))
    nar = neg_area / denom if denom and denom > 0 else float("nan")  # lower is better

    # SNR via prominent peaks
    sigma_raw = float(np.sqrt(np.mean((y_raw[noise_mask]) ** 2))) if noise_mask.any() else 0.0
    prom_thr = max(3.0 * sigma_raw, 0.0)
    peaks, props = signal.find_peaks(y_corr, prominence=prom_thr)
    if peaks.size == 0:
        peak_heights = np.array([np.nanmax(y_corr)])
    else:
        peak_heights = np.asarray(props.get("prominences", y_corr[peaks]), dtype=float)
    if peak_heights.size > 1:
        top = np.sort(peak_heights)[-min(topk, peak_heights.size):]
        peak_stat = float(np.median(top))
    else:
        peak_stat = float(peak_heights.item())
    snr = peak_stat / sigma_noise if (sigma_noise and sigma_noise > 0) else float("nan")

    # BBI
    bbi = float(np.nanmedian(y_corr[noise_mask])) if noise_mask.any() else float("nan")

    # BR (needs baseline & x)
    if baseline is None or x is None:
        br = float("nan")
    else:
        x = np.asarray(x, dtype=float).ravel()
        bl = np.asarray(baseline, dtype=float).ravel()
        if x.shape != bl.shape:
            br = float("nan")
        else:
            # second derivative w.r.t. x (handles nonuniform spacing)
            d1 = np.gradient(bl, x, edge_order=2)
            d2 = np.gradient(d1, x, edge_order=2)
            br = float(np.sqrt(np.mean((d2[noise_mask]) ** 2))) if noise_mask.any() else float("nan")

    # NBC
    nbc = float(np.mean(y_corr < 0.0))

    return MetricResult(rfzn=sigma_noise, nar=nar, snr=snr, bbi=bbi, br=br, nbc=nbc)
