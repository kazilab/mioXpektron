"""Univariate statistics for aligned feature matrices."""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    adjusted = p[order] * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q = np.empty_like(p, dtype=float)
    q[order] = adjusted
    return np.clip(q, 0, 1)


def _normalize_labels(y: pd.Series) -> pd.Series:
    return y.astype(str).str.strip()


def _sanitize_group_name(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"[^\w]+", "_", text)
    return text.strip("_") or "group"


def _resolve_groups(
    y: pd.Series,
    *,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
) -> Tuple[str, str]:
    labels = _normalize_labels(y)
    counts = labels.value_counts()
    if counts.empty:
        raise ValueError("Group labels are empty.")

    if group_a is not None and group_b is not None:
        if group_a == group_b:
            raise ValueError("group_a and group_b must be different.")
        for name in (group_a, group_b):
            if name not in counts.index:
                raise ValueError(f"Group '{name}' not found in labels: {list(counts.index)}.")
        return group_a, group_b

    if group_a is not None or group_b is not None:
        specified = group_a if group_a is not None else group_b
        if specified not in counts.index:
            raise ValueError(f"Group '{specified}' not found in labels: {list(counts.index)}.")
        others = [g for g in counts.index if g != specified]
        if not others:
            raise ValueError("At least two groups are required for two-group testing.")
        partner = others[0]
        if group_a is not None:
            return group_a, partner
        return partner, group_b  # type: ignore[return-value]

    if len(counts) < 2:
        raise ValueError("At least two groups are required for two-group testing.")
    top_two = counts.index[:2].tolist()
    return top_two[0], top_two[1]


def compute_univariate_tests(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
    reference_group: Optional[str] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Welch t-test per feature with log2 fold-change (group_a / group_b).

    When ``group_a`` and ``group_b`` are omitted, the two largest groups by
    sample count are compared. ``reference_group`` sets the denominator for
    log2 fold-change and defaults to ``group_b``.
    """
    labels = _normalize_labels(y)
    if len(labels) != len(X):
        raise ValueError("X and y must have the same number of samples.")
    if not labels.index.equals(X.index):
        labels = labels.reindex(X.index)

    group_a, group_b = _resolve_groups(labels, group_a=group_a, group_b=group_b)

    if reference_group is None:
        reference_group = group_b
    if reference_group not in (group_a, group_b):
        raise ValueError("reference_group must be group_a or group_b.")

    numerator = group_a if reference_group == group_b else group_b
    denominator = reference_group

    mask_num = labels == numerator
    mask_den = labels == denominator
    if mask_num.sum() == 0 or mask_den.sum() == 0:
        raise ValueError("Each comparison group must contain at least one sample.")

    X_num = X.loc[mask_num]
    X_den = X.loc[mask_den]

    mean_num = X_num.mean(axis=0).values
    mean_den = X_den.mean(axis=0).values
    log2_fc = np.log2((mean_num + eps) / (mean_den + eps))

    Xn = X_num.values
    Xd = X_den.values
    n_num, n_den = Xn.shape[0], Xd.shape[0]
    var_num = np.nanvar(Xn, axis=0, ddof=1)
    var_den = np.nanvar(Xd, axis=0, ddof=1)
    se = np.sqrt(var_num / n_num + var_den / n_den)
    t_stat = np.where(se > 0, (mean_num - mean_den) / se, 0.0)

    num_df = (var_num / n_num + var_den / n_den) ** 2
    den_df = (var_num / n_num) ** 2 / (n_num - 1) + (var_den / n_den) ** 2 / (n_den - 1)
    df_welch = np.where(den_df > 0, num_df / den_df, 1.0)
    pvals = 2.0 * stats.t.sf(np.abs(t_stat), df_welch)
    pvals = np.where(np.isfinite(pvals), pvals, 1.0)
    qvals = bh_fdr(pvals)

    col_num = f"mean_{_sanitize_group_name(numerator)}"
    col_den = f"mean_{_sanitize_group_name(denominator)}"

    res = pd.DataFrame(
        {
            "feature": X.columns,
            col_num: mean_num,
            col_den: mean_den,
            "log2_FC": log2_fc,
            "p_value": pvals,
            "q_value": qvals,
            "group_a": numerator,
            "group_b": denominator,
        }
    ).sort_values("q_value", ascending=True).reset_index(drop=True)
    return res