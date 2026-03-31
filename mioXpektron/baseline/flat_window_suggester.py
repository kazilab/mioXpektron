
"""
flat_window_suggester_polars.py
--------------------------------
Small application to discover common "flat" m/z windows across a set of
ToF‑SIMS spectra. Inputs are 3‑column tables: Channel, m/z, intensity
(case‑insensitive).

Key changes vs. the original:
- Replaced all pandas operations with Polars (Rust/Arrow backend).
- Added support for providing an explicit *list of file paths or glob patterns*,
  so data can be spread across many folders (no need for a single root dir).
- Kept the numerical core (NumPy/SciPy) for smoothing & derivatives.

What it does (unchanged conceptually)
-------------------------------------
1) Per spectrum:
   - Smooth intensities (Savitzky–Golay) and compute 1st/2nd derivatives.
   - Flag baseline‑candidate points where simultaneously:
       y_raw <= q_y quantile  AND
       |dy/dx| <= q_g quantile  AND
       |d²y/dx²| <= q_c quantile
   - Merge contiguous candidate points into segments; keep segments that
     satisfy minimum width & minimum number of points.

2) Across all spectra:
   - Discretize the global m/z range into bins (width = bin_width).
   - For each file, mark bins covered by any of its segments.
   - Compute the coverage fraction per bin (#files covering bin / #files total).
   - Extract contiguous regions whose coverage ≥ coverage_threshold.
   - Rank regions by mean coverage (then by width) and return top_k windows.

Outputs
-------
- out_dir / per_file_segments.csv             (Polars CSV)
- out_dir / flat_windows_suggestions.csv      (Polars CSV with coverage stats)
- out_dir / flat_windows.json                 (list[[lo, hi], ...])
- out_dir / coverage_curve.(png|pdf)          (plot of coverage vs m/z)
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)
from joblib import Parallel, delayed
from scipy.signal import medfilt, savgol_filter
from tqdm import tqdm

OUTPUT_DIR = Path("output_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Column aliases & helpers
# ---------------------------------------------------------------------

COL_ALIASES = {
    "channel": {"channel", "chan", "index", "idx", "Channel", "CHANNEL", "Index"},
    "mz": {"m/z", "mz", "mass", "moverz", "m_over_z", "Mass", "MZ"},
    "intensity": {"intensity", "counts", "signal", "y", "ion_counts", "Intensity", "COUNTS", "INTENSITY"},
}


def _standardize_columns_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to ['channel','mz','intensity'] if aliases are found.
    If 'channel' missing, insert a 1-based index. Returns only those 3 columns.
    """
    # Map existing cols to lowercase for alias matching
    lower = {c: str(c).strip().lower() for c in df.columns}
    rename: Dict[str, str] = {}

    for std, aliases in COL_ALIASES.items():
        for c, cl in lower.items():
            if cl in aliases:
                rename[c] = std
                break

    if rename:
        df = df.rename(rename)

    cols = set(df.columns)
    if "mz" not in cols or "intensity" not in cols:
        raise KeyError("Input must include 'm/z' (or alias) and 'intensity' columns.")

    if "channel" not in cols:
        df = df.with_columns(pl.arange(1, df.height + 1).alias("channel"))

    # Coerce dtypes and filter invalid rows
    def _safe_cast(col: str, dtype) -> pl.Expr:
        try:
            return pl.col(col).cast(dtype, strict=False)
        except TypeError:
            # Older Polars versions may not accept strict=; fall back to cast
            return pl.col(col).cast(dtype)

    df = df.with_columns(
        _safe_cast("channel", pl.Int64),
        _safe_cast("mz", pl.Float64),
        _safe_cast("intensity", pl.Float64),
    )

    # Keep only required columns; drop NA/inf in mz/intensity
    df = df.select("channel", "mz", "intensity")
    # Filter NaNs/Infs
    df = df.filter(
        pl.col("mz").is_finite() & pl.col("intensity").is_finite()
    )

    return df


def _read_csv_with_fallbacks(path: Path) -> Optional[pl.DataFrame]:
    """Try to read CSV/TSV (and simple whitespace-delimited) with Polars + fallback.
    Returns a Polars DataFrame or None.
    """
    # First, attempt common separators with Polars
    for sep in (",", "\t"):
        try:
            df = pl.read_csv(
                path,
                separator=sep,
                comment_prefix="#",
                infer_schema_length=2048,
                ignore_errors=False,
                try_parse_dates=False,
            )
            if df.width >= 2 and df.height >= 1:
                return df
        except Exception:
            pass

    # Fallback for simple whitespace-delimited tables via NumPy
    # (handles arbitrary runs of whitespace; ignores '#...' comments)
    try:
        # Read header to get names
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            header = None
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                header = re.split(r"\s+", s)
                break

        if header is None:
            return None

        arr = np.genfromtxt(
            path,
            comments="#",
            dtype=None,
            names=True,
            encoding="utf-8",
            autostrip=True,
        )
        if arr.size == 0:
            return None

        # Build DataFrame from structured array
        cols = {}
        names = arr.dtype.names if arr.dtype.names is not None else header
        for name in names:
            try:
                cols[name] = arr[name]
            except Exception:
                pass

        if not cols:
            return None

        return pl.DataFrame(cols)
    except Exception:
        return None


def read_spectrum_table(path: Union[str, Path]) -> pl.DataFrame:
    """Robust reader that returns Polars DataFrame with standardized columns.
    Tries comma, tab, then whitespace-delimited tables (with '#' comments).
    """
    path = Path(path)
    df = _read_csv_with_fallbacks(path)
    if df is None:
        raise ValueError(f"Could not parse table at: {path}")
    # Sort by m/z (derivatives use x order)
    df = _standardize_columns_pl(df).sort("mz")
    return df


# ---------------------------------------------------------------------
# Smoothing / flat-segment detection (NumPy/SciPy)
# ---------------------------------------------------------------------

@dataclass
class FlatParams:
    y_quantile: float = 0.20
    grad_quantile: float = 0.40
    curv_quantile: float = 0.40
    savgol_window: int = 11
    savgol_poly: int = 2
    min_width: float = 0.2
    min_points: int = 20


def _adaptive_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = y.size
    if n < 5:
        return y.copy()
    # window must be odd and <= n
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < poly + 2:
        window = poly + 3  # ensure > poly and odd
    if window % 2 == 0:
        window += 1
    try:
        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    except Exception:
        # fallback to a small median filter
        k = min(max(5, window), n if n % 2 == 1 else n - 1)
        return medfilt(y, kernel_size=k)


def find_flat_segments(
    x: np.ndarray, y: np.ndarray, p: FlatParams
) -> List[Tuple[float, float, int]]:
    """
    Return list of (lo, hi, n_points) flat segments for one spectrum.
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    assert x.shape == y.shape

    # Collapse duplicate m/z values to avoid zero-spacing gradients.
    uniq_x, inverse = np.unique(x, return_inverse=True)
    if uniq_x.size != x.size:
        accum = np.zeros_like(uniq_x, dtype=float)
        counts = np.zeros_like(uniq_x, dtype=int)
        np.add.at(accum, inverse, y)
        np.add.at(counts, inverse, 1)
        y = accum / np.maximum(counts, 1)
        x = uniq_x

    # Smooth intensity
    y_s = _adaptive_savgol(y, p.savgol_window, p.savgol_poly)

    # derivatives w.r.t m/z (handles nonuniform spacing)
    dy = np.gradient(y_s, x, edge_order=2)
    d2 = np.gradient(dy, x, edge_order=2)

    # thresholds (lower quantiles → flat)
    y_thr  = np.nanquantile(y,  p.y_quantile)
    g_thr  = np.nanquantile(np.abs(dy), p.grad_quantile)
    c_thr  = np.nanquantile(np.abs(d2), p.curv_quantile)

    mask = (y <= y_thr) & (np.abs(dy) <= g_thr) & (np.abs(d2) <= c_thr)

    # group contiguous True regions
    segs: List[Tuple[float, float, int]] = []
    if not mask.any():
        return segs
    idx = np.where(mask)[0]
    # break into contiguous runs
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for r in runs:
        if r.size < p.min_points:
            continue
        lo = float(x[r[0]]); hi = float(x[r[-1]])
        if (hi - lo) >= p.min_width:
            segs.append((lo, hi, int(r.size)))
    return segs


# ---------------------------------------------------------------------
# Aggregate across files
# ---------------------------------------------------------------------

@dataclass
class AggregateParams:
    bin_width: float = 0.1
    coverage_threshold: float = 0.5
    top_k: int = 6


def aggregate_common_windows(
    segments_by_file: Dict[str, List[Tuple[float, float, int]]],
    x_minmax: Tuple[float, float],
    agg: AggregateParams,
) -> Tuple[List[Tuple[float, float]], pl.DataFrame]:
    """
    Merge per-file segments into common windows via m/z bin coverage.
    Returns (windows, coverage_table_df[polars]).
    """
    files = list(segments_by_file.keys())
    n_files = len(files)

    x_min, x_max = x_minmax
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return [], pl.DataFrame({"mz_center": [], "coverage": []})

    # Build bin edges and centers
    edges = np.arange(x_min, x_max + agg.bin_width, agg.bin_width, dtype=float)
    if edges.size < 2:
        return [], pl.DataFrame({"mz_center": [], "coverage": []})
    centers = 0.5 * (edges[:-1] + edges[1:])

    cover_counts = np.zeros(centers.size, dtype=int)

    # Vectorized: mark bins covered by segments for each file
    for fname, segs in segments_by_file.items():
        if not segs:
            continue
        # Stack all segment bounds and use broadcasting
        seg_arr = np.array([(lo, hi) for lo, hi, _n in segs])
        # centers (n_bins,) vs seg_arr (n_segs, 2): check if center is in any segment
        in_any = np.any(
            (centers[np.newaxis, :] >= seg_arr[:, 0:1]) &
            (centers[np.newaxis, :] <= seg_arr[:, 1:2]),
            axis=0
        )
        cover_counts += in_any.astype(int)

    coverage = cover_counts / max(n_files, 1)
    # Identify runs with coverage ≥ threshold
    mask = coverage >= float(agg.coverage_threshold)

    windows: List[Tuple[float, float]] = []
    if mask.any():
        idx = np.where(mask)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for r in runs:
            lo = float(edges[r[0]])
            hi = float(edges[r[-1] + 1])
            if (hi - lo) > 0:
                windows.append((lo, hi))

    # Rank windows by mean coverage, pick top_k
    stats: List[Tuple[float, float, float, float, float]] = []
    for (lo, hi) in windows:
        sel = (centers >= lo) & (centers <= hi)
        cov_mean = float(np.mean(coverage[sel])) if np.any(sel) else 0.0
        cov_min  = float(np.min(coverage[sel])) if np.any(sel) else 0.0
        stats.append((lo, hi, hi - lo, cov_mean, cov_min))

    if stats:
        df = pl.DataFrame(
            stats,
            schema=["lo", "hi", "width", "coverage_mean", "coverage_min"],
            orient="row",
        ).sort(by=["coverage_mean", "width"], descending=[True, True])
        if agg.top_k and agg.top_k > 0:
            df = df.head(agg.top_k)
        windows = [(float(r[0]), float(r[1])) for r in df.select(["lo", "hi"]).to_numpy()]
    else:
        df = pl.DataFrame(schema=["lo", "hi", "width", "coverage_mean", "coverage_min"])

    # coverage table (for plotting/export)
    coverage_df = pl.DataFrame({"mz_center": centers, "coverage": coverage})
    return windows, coverage_df


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def _has_glob_chars(s: str) -> bool:
    return any(ch in s for ch in "*?[")


@dataclass
class ScanForFlatRegion:
    files: List[Union[str, Path]] = field(default_factory=list)
    out_dir: Union[str, Path] = "flat_windows_out"
    n_jobs: int = -1
    flat_params: FlatParams = field(default_factory=FlatParams)
    agg_params: AggregateParams = field(default_factory=AggregateParams)
    auto_tune: bool = False

    def _expand_files(self) -> List[Path]:
        paths: List[Path] = []
        for item in self.files:
            s = str(item).strip()
            if not s:
                continue
            if _has_glob_chars(s):
                for hit in Path().glob(s):
                    if hit.is_file():
                        paths.append(hit.resolve())
                continue
            p = Path(s)
            if p.is_file():
                paths.append(p.resolve())
            elif p.is_dir():
                for hit in p.rglob("*"):
                    if hit.is_file():
                        paths.append(hit.resolve())
            else:
                for hit in Path().glob(s):
                    if hit.is_file():
                        paths.append(hit.resolve())

        uniq = sorted(set(paths))
        if not uniq:
            raise FileNotFoundError(
                "No input files found; verify the provided file paths or glob patterns."
            )
        return uniq

    def run(self):
        files = self._expand_files()

        if self.auto_tune:
            from ..adaptive import estimate_flat_params
            from dataclasses import replace as _replace
            overrides = estimate_flat_params([str(f) for f in files])
            if overrides:
                self.flat_params = _replace(self.flat_params, **overrides)

        out_dir = OUTPUT_DIR / Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _worker(path: Path):
            try:
                df = read_spectrum_table(path)
                x = df.get_column("mz").to_numpy()
                y = df.get_column("intensity").to_numpy()
                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)
                segs = find_flat_segments(x, y, self.flat_params)
                return (
                    str(path),
                    segs,
                    (np.min(x) if x.size else np.nan),
                    (np.max(x) if x.size else np.nan),
                )
            except Exception:
                return str(path), [], np.nan, np.nan

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_worker)(p) for p in tqdm(files, desc="Processing spectra")
        )

        segments_by_file: Dict[str, List[Tuple[float, float, int]]] = {}
        xmins: List[float] = []
        xmaxs: List[float] = []
        rows: List[Tuple[str, float, float, float, int]] = []
        for fname, segs, xmin, xmax in results:
            segments_by_file[fname] = segs
            if np.isfinite(xmin):
                xmins.append(float(xmin))
            if np.isfinite(xmax):
                xmaxs.append(float(xmax))
            for lo, hi, n in segs:
                rows.append((fname, lo, hi, hi - lo, n))

        per_file_df = pl.DataFrame(
            rows,
            schema=["file", "lo", "hi", "width", "n_pts"],
            orient="row",
        ).sort(by=["file", "lo"])
        per_file_df.write_csv(out_dir / "per_file_segments.csv")

        x_min = float(np.nanmin(xmins)) if xmins else np.nan
        x_max = float(np.nanmax(xmaxs)) if xmaxs else np.nan
        windows, coverage_df = aggregate_common_windows(
            segments_by_file,
            (x_min, x_max),
            self.agg_params,
        )

        stats: List[dict] = []
        for i, (lo, hi) in enumerate(windows, start=1):
            sub = coverage_df.filter(
                (pl.col("mz_center") >= lo) & (pl.col("mz_center") <= hi)
            )
            if sub.height:
                stats.append(
                    {
                        "window_id": i,
                        "lo": lo,
                        "hi": hi,
                        "width": hi - lo,
                        "coverage_mean": float(sub["coverage"].mean()),
                        "coverage_min": float(sub["coverage"].min()),
                    }
                )
        sug_df = (
            pl.DataFrame(
                stats,
                schema=[
                    "window_id",
                    "lo",
                    "hi",
                    "width",
                    "coverage_mean",
                    "coverage_min",
                ],
            )
            if stats
            else pl.DataFrame(
                schema=[
                    "window_id",
                    "lo",
                    "hi",
                    "width",
                    "coverage_mean",
                    "coverage_min",
                ]
            )
        )
        sug_df.write_csv(out_dir / "flat_windows_suggestions.csv")

        with open(out_dir / "flat_windows.json", "w", encoding="utf-8") as f:
            json.dump([[float(lo), float(hi)] for (lo, hi) in windows], f, indent=2)

        plt.figure(figsize=(10, 4.2))
        x_cov = coverage_df["mz_center"].to_numpy() if coverage_df.height else np.array([])
        y_cov = coverage_df["coverage"].to_numpy() if coverage_df.height else np.array([])
        if x_cov.size:
            plt.plot(x_cov, y_cov, lw=1)
        for lo, hi in windows:
            plt.axvspan(lo, hi, alpha=0.25)
        plt.xlabel("m/z")
        plt.ylabel("Coverage fraction")
        plt.title("Flat-window coverage across spectra (suggested windows shaded)")
        plt.tight_layout()
        for ext in (".pdf", ".png"):
            plt.savefig(out_dir / f"coverage_curve{ext}", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Suggested flat windows (lo, hi):")
        for i, (lo, hi) in enumerate(windows, start=1):
            logger.info(f"  {i:2d}. [{lo:.4f}, {hi:.4f}]  (width={hi - lo:.4f})")

        return windows
