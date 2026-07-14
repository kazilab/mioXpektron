"""Prepare feature matrices from pipeline or tabular outputs."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

DEFAULT_META_COLS = ("SampleName", "Group")


def _normalize_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _is_meta_column(name: str, meta_cols: Sequence[str]) -> bool:
    return name in meta_cols


def prepare_matrix(
    df: pd.DataFrame,
    *,
    label_col: str = "Group",
    sample_col: str = "SampleName",
    meta_cols: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    coerce_numeric: bool = True,
    fill_na: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build a sample-by-feature matrix and group labels from pipeline output.

    Accepts either:

    * A long table with ``SampleName`` / ``Group`` columns and m/z feature columns
      (typical exported CSV), or
    * An aligned matrix from :func:`~mioXpektron.detection.align_peaks` where
      ``SampleName`` and optionally ``Group`` are index levels.

    Parameters
    ----------
    df
        Input table or aligned feature matrix.
    label_col
        Column or index level containing group labels.
    sample_col
        Column or index level containing sample identifiers.
    meta_cols
        Additional metadata columns to exclude from features. Defaults to
        ``SampleName`` and ``Group`` only.
    feature_cols
        Explicit feature column names. When omitted, all non-metadata columns are used.
    coerce_numeric
        If True, coerce feature columns to numeric (invalid values become NaN).
    fill_na
        Value used to fill missing feature values after coercion.

    Returns
    -------
    X
        Feature matrix (samples x m/z), index aligned with ``meta``.
    y
        Group labels indexed like ``X``.
    meta
        Metadata frame with at least ``sample_col`` and ``label_col``.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    working = df.copy()
    meta_cols = tuple(meta_cols) if meta_cols is not None else DEFAULT_META_COLS
    required_meta = {sample_col, label_col}

    index_names = list(working.index.names or [])
    index_has_meta = any(name in required_meta for name in index_names if name is not None)

    if index_has_meta:
        working = working.reset_index()

    missing = required_meta - set(working.columns)
    if missing:
        raise ValueError(
            f"Input must provide {sorted(required_meta)} as columns or index levels; "
            f"missing: {sorted(missing)}."
        )

    if feature_cols is None:
        feature_cols = [
            col
            for col in working.columns
            if not _is_meta_column(col, meta_cols) and col not in required_meta
        ]
    else:
        feature_cols = list(feature_cols)

    if not feature_cols:
        raise ValueError("No feature columns found in the input DataFrame.")

    meta = working[[sample_col, label_col]].copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta[label_col] = _normalize_label(meta[label_col])

    X = working[feature_cols].copy()
    if coerce_numeric:
        X = X.apply(pd.to_numeric, errors="coerce")
    if fill_na is not None:
        X = X.fillna(fill_na)

    row_index = meta[sample_col].values
    X.index = row_index
    y = pd.Series(meta[label_col].values, index=row_index, name=label_col)

    return X, y, meta


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    meta_cols: Sequence[str] = DEFAULT_META_COLS,
) -> List[str]:
    """Return non-metadata columns, attempting to detect m/z-like headers."""
    cols = [c for c in df.columns if c not in meta_cols]
    mz_like = [c for c in cols if _looks_like_mz(c)]
    return mz_like if mz_like else cols


def _looks_like_mz(name: Union[str, float, int]) -> bool:
    text = str(name).strip()
    if not text:
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return True
    try:
        float(text)
        return True
    except ValueError:
        return False