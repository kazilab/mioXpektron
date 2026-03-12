"""Batch-oriented helpers for running denoising over many spectra files."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import traceback, time, os
from typing import Iterable, List, Dict, Any, Callable, Optional, Union
from .denoise_main import noise_filtering

import numpy as np

logger = logging.getLogger(__name__)


def load_txt_spectrum(path: Path) -> Dict[str, np.ndarray]:
    """Parse a plain-text spectrum file into NumPy arrays using Polars.

    Parameters
    ----------
    path : Path
        Location of the ASCII export produced by the instrument or pre-processing
        software. Delimiters can be comma, tab, semicolon, or space.

    Returns
    -------
    dict
        Dictionary containing any detected columns (`channel`, `mz`, and
        `intensity`). Only columns that successfully parse are included; the
        intensity array is always provided.

    Notes
    -----
    The loader uses Polars for fast CSV parsing with automatic delimiter detection,
    tolerates blank lines, and skips rows that fail numeric conversion.
    """
    import polars as pl

    # Try different delimiters
    for sep in ["\t", ",", ";", " "]:
        try:
            df = pl.read_csv(
                path,
                separator=sep,
                has_header=True,
                ignore_errors=True,
                truncate_ragged_lines=True,
                comment_prefix="#",
            )
            if df.shape[0] > 0 and df.shape[1] >= 1:
                break
        except Exception:
            continue
    else:
        # Fallback: try without header
        try:
            df = pl.read_csv(
                path,
                separator="\t",
                has_header=False,
                ignore_errors=True,
                truncate_ragged_lines=True,
                comment_prefix="#",
            )
        except Exception:
            return {"intensity": np.array([], dtype=float)}

    # Normalize column names for matching
    def norm(s: str) -> str:
        """Lowercase tokens and drop spaces for loose matching."""
        return s.strip().lower().replace(" ", "")

    col_rename = {}
    for col in df.columns:
        normed = norm(col)
        if normed in {"m/z", "mz", "mass", "M/Z", "MZ", "M"}:
            col_rename[col] = "mz"
        elif normed in {"intensity", "inten", "i", "Intensity"}:
            col_rename[col] = "intensity"
        elif normed in {"channel", "chan", "ch", "Channel"}:
            col_rename[col] = "channel"

    if col_rename:
        df = df.rename(col_rename)
    else:
        # No header detected, assume column order: Channel, m/z, Intensity
        if df.shape[1] >= 3:
            df = df.rename({df.columns[0]: "channel", df.columns[1]: "mz", df.columns[2]: "intensity"})
        elif df.shape[1] == 2:
            df = df.rename({df.columns[0]: "mz", df.columns[1]: "intensity"})
        elif df.shape[1] == 1:
            df = df.rename({df.columns[0]: "intensity"})

    # Build output dictionary with numeric arrays
    out = {}
    for col_name in ["channel", "mz", "intensity"]:
        if col_name in df.columns:
            try:
                arr = df.select(pl.col(col_name).cast(pl.Float64, strict=False)).to_numpy().ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    out[col_name] = arr
            except Exception:
                pass

    # Ensure intensity is always present
    if "intensity" not in out:
        out["intensity"] = np.array([], dtype=float)

    return out

def save_txt_spectrum(orig: Path, out_path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Persist a spectrum to disk with the same column ordering as the input using Polars.

    Parameters
    ----------
    orig : Path
        Original source file. Currently unused but kept for potential metadata
        handling.
    out_path : Path
        Destination path for the denoised export.
    arrays : dict[str, np.ndarray]
        Columns to write. The function writes whichever of `channel`, `mz`, and
        `intensity` are present, preserving numeric precision to six decimals.
    """
    import polars as pl

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dataframe from arrays
    data_dict = {}
    if "channel" in arrays:
        data_dict["Channel"] = arrays["channel"]
    if "mz" in arrays:
        data_dict["m/z"] = arrays["mz"]
    data_dict["Intensity"] = arrays["intensity"]

    df = pl.DataFrame(data_dict)

    # Write to CSV with tab separator and 6 decimal precision
    df.write_csv(
        out_path,
        separator="\t",
    )

@dataclass
class BatchResult:
    """Outcome metadata for a single denoising file run."""
    file: str
    out_file: Optional[str]
    status: str  # "ok" or "error"
    elapsed_s: float
    n_points: int
    message: str = ""

def _process_one(path: Path, out_dir: Path, method: str, params: Dict[str, Any]) -> BatchResult:
    """Run denoising on a single file and capture timing plus status.

    Parameters
    ----------
    path : Path
        Input spectrum file.
    out_dir : Path
        Directory where the denoised file is written.
    method : str
        Denoising strategy forwarded to :func:`noise_filtering`.
    params : dict[str, Any]
        Additional parameters for :func:`noise_filtering`.

    Returns
    -------
    BatchResult
        Summary of the run, including elapsed time and error context.
    """
    t0 = time.perf_counter()
    try:
        rec = load_txt_spectrum(path)
        y = rec["intensity"]
        if y.size == 0:
            raise ValueError("Empty intensity array")

        y_hat = noise_filtering(y, method=method, **params)
        rec["intensity"] = y_hat

        out_path = out_dir / f"{path.stem}_denoised{path.suffix}"
        save_txt_spectrum(path, out_path, rec)
        return BatchResult(
            file=str(path), out_file=str(out_path), status="ok",
            elapsed_s=time.perf_counter() - t0, n_points=int(y.size),
            message=""
        )
    except Exception as e:
        return BatchResult(
            file=str(path), out_file=None, status="error",
            elapsed_s=time.perf_counter() - t0, n_points=0,
            message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )

def batch_denoise(
    files: Union[Iterable[str], Iterable[Path]],
    output_dir: Union[str, Path],
    method: str = "wavelet",
    n_workers: int = 0,
    backend: str = "threads",
    progress: bool = True,
    params: Optional[Dict[str, Any]] = None,
) -> List[BatchResult]:
    """Apply the configured denoising method to multiple spectrum files.

    Parameters
    ----------
    files : Iterable[str | Path]
        Collection of filesystem paths (glob results, manual list, etc.).
    output_dir : str | Path
        Directory where the denoised outputs will be written.
    method : str, default "wavelet"
        Name of the smoothing routine forwarded to :func:`noise_filtering`.
    n_workers : int, default 0
        Worker count for the executor. ``0`` or ``None`` selects a CPU-aware
        default.
    backend : {"threads", "processes"}, default "threads"
        Execution strategy for the worker pool.
    progress : bool, default True
        If True, wrap the executor iterator in ``tqdm`` when available.
    params : dict | None
        Extra keyword arguments forwarded to :func:`noise_filtering`.

    Returns
    -------
    list[BatchResult]
        Status records describing each attempted file.

    Raises
    ------
    ValueError
        If no input paths exist or an unsupported backend name is provided.
    """
    params = dict(params or {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize file list
    paths = [Path(p) for p in files] if not isinstance(files, (str, Path)) else [Path(files)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise ValueError("No input files exist. Check your 'files' iterable and working directory.")

    # Backend selection
    if backend not in {"threads", "processes"}:
        raise ValueError(f"Invalid backend='{backend}'. Use 'threads' or 'processes'.")

    # Default workers
    if not n_workers or n_workers < 0:
        import os
        n_workers = os.cpu_count() or 4

    worker = lambda p: _process_one(p, out_dir, method, params)

    results: List[BatchResult] = []
    Executor = ThreadPoolExecutor if backend == "threads" else ProcessPoolExecutor

    # IMPORTANT: consume futures via as_completed to ensure work executes
    with Executor(max_workers=n_workers) as ex:
        futs = {ex.submit(worker, p): p for p in paths}
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(as_completed(futs), total=len(futs), desc="Denoising")
            except Exception:
                it = as_completed(futs)
        else:
            it = as_completed(futs)

        for fut in it:
            res = fut.result()
            results.append(res)
            if progress and 'tqdm' in globals():
                pass  # tqdm advances via iteration

    # Surface failures clearly
    errors = [r for r in results if r.status == "error"]
    if errors:
        logger.warning(f"[batch_denoise] {len(errors)} errors encountered. First error:\n{errors[0].message}")

    return results
