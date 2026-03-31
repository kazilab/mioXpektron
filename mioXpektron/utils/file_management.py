# Data import function

import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import polars as pl

# Default group patterns: regex → group label.
# Patterns are tested in insertion order; the first match wins.
DEFAULT_GROUP_PATTERNS: Dict[str, str] = {
    r'_CC(?=_|\b)': 'Cancer',
    r'_CT(?=_|\b)': 'Control',
}

MZ_COLUMN_ALIASES = ("m/z", "mz")
INTENSITY_COLUMN_ALIASES = ("corrected_intensity", "Intensity", "intensity")


def _resolve_group(
    sample_name: str,
    group_patterns: Optional[Dict[str, str]],
    group_fn: Optional[Callable[[str], str]],
) -> str:
    """Return a group label for *sample_name*.

    Priority: *group_fn* (if given) > *group_patterns* (if given) >
    ``DEFAULT_GROUP_PATTERNS``.
    """
    if group_fn is not None:
        return group_fn(sample_name)

    patterns = group_patterns if group_patterns is not None else DEFAULT_GROUP_PATTERNS
    for pattern, label in patterns.items():
        if re.search(pattern, sample_name, flags=re.IGNORECASE):
            return label
    return 'Unknown'


def _resolve_column(df: pl.DataFrame, aliases: Tuple[str, ...]) -> str:
    """Return the first matching column name from *aliases*."""
    by_lower = {col.lower(): col for col in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        match = by_lower.get(alias.lower())
        if match is not None:
            return match
    raise ValueError(f"Missing required columns. Tried aliases: {aliases}")


def import_data(
        file_path: str,
        mz_min: float = None,
        mz_max: float = None,
        group_patterns: Optional[Dict[str, str]] = None,
        group_fn: Optional[Callable[[str], str]] = None,
        ) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Import ToF-SIMS data from a spectrum file.

    Parameters
    ----------
    file_path : str
        Path to the ToF-SIMS data file. Supports tab-delimited ``.txt``
        exports with ``m/z`` + ``Intensity`` columns and CSV exports with
        ``mz`` + ``corrected_intensity`` or ``intensity`` columns.
    mz_min : float, optional
        Minimum m/z value to be imported (inclusive).
    mz_max : float, optional
        Maximum m/z value to be imported (inclusive).
    group_patterns : dict[str, str], optional
        Mapping of ``{regex_pattern: group_label}``.  Patterns are tested
        against the sample name (filename without extension) in order;
        the first match determines the group.  Defaults to
        ``{'_CC...': 'Cancer', '_CT...': 'Control'}``.
    group_fn : callable, optional
        A function ``(sample_name: str) -> str`` that returns the group
        label directly.  When provided this takes priority over
        *group_patterns*.

    Returns
    -------
    mz : np.ndarray
        Mass-to-charge ratio values.
    intensity : np.ndarray
        Intensity values.
    sample_name : str
        Sample name extracted from file name.
    group : str
        Group label derived from the filename.
    """

    path = Path(file_path)
    separator = "," if path.suffix.lower() == ".csv" else "\t"

    # Read file, skip lines starting with '#'
    df = pl.read_csv(file_path, separator=separator, comment_prefix="#")

    mz_col = _resolve_column(df, MZ_COLUMN_ALIASES)
    intensity_col = _resolve_column(df, INTENSITY_COLUMN_ALIASES)

    # Apply m/z filtering (inclusive)
    if mz_min is not None:
        df = df.filter(pl.col(mz_col) >= mz_min)
    if mz_max is not None:
        df = df.filter(pl.col(mz_col) <= mz_max)

    if df.height == 0:
        raise ValueError(f"No data in {file_path} after m/z filtering.")

    mz = df[mz_col].to_numpy()
    intensity = df[intensity_col].to_numpy()
    sample_name = path.stem
    group = _resolve_group(sample_name, group_patterns, group_fn)
    return mz, intensity, sample_name, group
