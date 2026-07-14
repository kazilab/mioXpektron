#!/usr/bin/env python3
"""
Deprecated study-level analysis entry point.

Use :mod:`mioXpektron.analysis` instead. This module re-exports the public API
and keeps a CLI-compatible ``main()`` for legacy scripts.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import pandas as pd

from mioXpektron.analysis import (
    AnalysisConfig,
    AnalysisWorkflow,
    bh_fdr,
    choose_k_by_pac,
    compute_univariate_tests,
    plot_heatmap_top_features,
    plot_pca,
    plot_umap,
    plot_volcano,
    prepare_matrix,
    run_cnmf,
    save_consensus_heatmap,
    save_factor_bars,
)

warnings.warn(
    "mioXpektron.utils.analysis is deprecated; use mioXpektron.analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisConfig",
    "AnalysisWorkflow",
    "bh_fdr",
    "choose_k_by_pac",
    "compute_univariate_tests",
    "main",
    "plot_heatmap_top_features",
    "plot_pca",
    "plot_umap",
    "plot_volcano",
    "prepare_matrix",
    "run_cnmf",
    "save_consensus_heatmap",
    "save_factor_bars",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(
    input_file,
    outdir,
    topn=25,
    umap=False,
    tsne=False,
    shap=False,
    cnmf=False,
    k_list=None,
    cnmf_reps=30,
    cnmf_beta="frobenius",
    ml_benchmark=False,
    ml_tuning=False,
):
    """Legacy CLI wrapper around :class:`~mioXpektron.analysis.AnalysisWorkflow`."""
    ensure_dir(outdir)
    df = pd.read_csv(input_file)
    if "Group" in df.columns:
        df["Group"] = df["Group"].astype(str).str.strip().str.capitalize()

    config = AnalysisConfig(
        outdir=outdir,
        group_a="Cancer",
        group_b="Control",
        reference_group="Control",
        top_n_features=topn,
        run_umap=umap,
        run_tsne=tsne,
        run_ml_benchmark=ml_benchmark,
        run_ml_tuning=ml_tuning,
        run_shap=shap,
        run_cnmf=cnmf,
        cnmf_k_list=k_list,
        cnmf_reps=cnmf_reps,
        cnmf_beta=cnmf_beta,
    )

    results = AnalysisWorkflow(df, config=config).run()
    logger.info("Done. Outputs written to: %s", results["outdir"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mioXpektron analysis workflow")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--outdir", default="analysis_outputs", help="Output directory")
    parser.add_argument("--topn", type=int, default=25, help="Top features for heatmap")
    parser.add_argument("--umap", action="store_true", help="Compute UMAP embedding")
    parser.add_argument("--tsne", action="store_true", help="Compute t-SNE embedding")
    parser.add_argument("--shap", action="store_true", help="SHAP explanation for best model")
    parser.add_argument("--cnmf", action="store_true", help="Run consensus NMF")
    parser.add_argument("--k_list", nargs="+", type=int, default=None, help="cNMF k values")
    parser.add_argument("--cnmf_reps", type=int, default=30, help="cNMF repetitions")
    parser.add_argument("--cnmf_beta", default="frobenius", help="NMF beta loss")
    parser.add_argument("--ml_benchmark", action="store_true", help="Benchmark classifiers")
    parser.add_argument("--ml_tuning", action="store_true", help="Tune top tunable classifiers")
    args = parser.parse_args()
    main(
        args.input,
        args.outdir,
        topn=args.topn,
        umap=args.umap,
        tsne=args.tsne,
        shap=args.shap,
        cnmf=args.cnmf,
        k_list=args.k_list,
        cnmf_reps=args.cnmf_reps,
        cnmf_beta=args.cnmf_beta,
        ml_benchmark=args.ml_benchmark,
        ml_tuning=args.ml_tuning,
    )