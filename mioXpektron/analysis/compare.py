"""Cross-dataset and model-family comparison utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def categorize_model(model_name: str) -> str:
    """Assign a model name to a coarse family label."""
    name = model_name.lower()
    if any(token in name for token in ("logistic", "ridge", "discriminant", "sgd")):
        return "Linear"
    if any(token in name for token in ("tree", "forest", "extra")):
        return "Tree"
    if any(token in name for token in ("boost", "xgb", "lgbm", "lightgbm", "adaboost")):
        return "Boosting"
    if "svm" in name or "svc" in name:
        return "SVM"
    if "knn" in name:
        return "KNN"
    if "naive" in name or "bayes" in name:
        return "Naive Bayes"
    if "mlp" in name:
        return "Neural Network"
    return "Other"


def summarize_model_families(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark metrics by model family."""
    df = results_df[results_df["status"] == "success"].copy()
    if df.empty:
        return pd.DataFrame()
    df["family"] = df["model_name"].map(categorize_model)
    summary = (
        df.groupby("family", as_index=False)
        .agg(
            n_models=("model_name", "count"),
            best_accuracy=("test_accuracy", "max"),
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            mean_train_time=("train_time", "mean"),
        )
        .sort_values("best_accuracy", ascending=False)
    )
    return summary.reset_index(drop=True)


def plot_family_comparison(
    results_df: pd.DataFrame,
    savepath: str,
    *,
    dataset_name: str = "dataset",
) -> None:
    """Plot best and mean accuracy per model family."""
    summary = summarize_model_families(results_df)
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(summary["family"], summary["best_accuracy"], alpha=0.8)
    axes[0].set_title(f"Best accuracy by family — {dataset_name}")
    axes[0].invert_yaxis()

    axes[1].barh(
        summary["family"],
        summary["mean_accuracy"],
        xerr=summary["std_accuracy"].fillna(0),
        alpha=0.8,
    )
    axes[1].set_title(f"Mean accuracy by family — {dataset_name}")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def compare_model_results(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    *,
    dataset_a: str = "dataset_a",
    dataset_b: str = "dataset_b",
) -> pd.DataFrame:
    """Merge two benchmark tables and compute accuracy deltas."""
    left = results_a[["model_name", "test_accuracy", "train_time"]].copy()
    right = results_b[["model_name", "test_accuracy", "train_time"]].copy()
    left.columns = ["model_name", f"{dataset_a}_accuracy", f"{dataset_a}_train_time"]
    right.columns = ["model_name", f"{dataset_b}_accuracy", f"{dataset_b}_train_time"]
    merged = pd.merge(left, right, on="model_name", how="inner")
    merged["accuracy_diff"] = merged[f"{dataset_a}_accuracy"] - merged[f"{dataset_b}_accuracy"]
    return merged.sort_values(f"{dataset_a}_accuracy", ascending=False).reset_index(drop=True)


def plot_dataset_model_comparison(
    comparison_df: pd.DataFrame,
    savepath: str,
    *,
    dataset_a: str = "dataset_a",
    dataset_b: str = "dataset_b",
    top_n: int = 15,
) -> None:
    """Side-by-side accuracy bars for two datasets."""
    df = comparison_df.head(top_n)
    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.35
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.bar(
        x - width / 2,
        df[f"{dataset_a}_accuracy"],
        width,
        label=dataset_a,
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        df[f"{dataset_b}_accuracy"],
        width,
        label=dataset_b,
        alpha=0.8,
    )
    plt.xticks(x, df["model_name"], rotation=90)
    plt.ylabel("Test accuracy")
    plt.title(f"Model comparison: {dataset_a} vs {dataset_b}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def run_multi_dataset_comparison(
    datasets: Mapping[str, Union[pd.DataFrame, str]],
    *,
    outdir: str = "comparison_outputs",
    config: Optional[Any] = None,
    run_ml_benchmark: bool = True,
) -> Dict[str, Any]:
    """Run analysis workflows on multiple datasets and compare ML benchmarks."""
    from .workflow import AnalysisConfig, AnalysisWorkflow

    os.makedirs(outdir, exist_ok=True)
    base_config = config or AnalysisConfig()
    per_dataset: Dict[str, Dict[str, Any]] = {}
    ml_tables: Dict[str, pd.DataFrame] = {}

    for name, data in datasets.items():
        dataset_dir = os.path.join(outdir, name)
        cfg = AnalysisConfig(
            outdir=dataset_dir,
            label_col=base_config.label_col,
            sample_col=base_config.sample_col,
            group_a=base_config.group_a,
            group_b=base_config.group_b,
            reference_group=base_config.reference_group,
            top_n_features=base_config.top_n_features,
            transform=base_config.transform,
            random_state=base_config.random_state,
            embedding_methods=base_config.embedding_methods,
            run_umap=base_config.run_umap,
            run_tsne=base_config.run_tsne,
            umap_n_neighbors=base_config.umap_n_neighbors,
            umap_min_dist=base_config.umap_min_dist,
            tsne_perplexity=base_config.tsne_perplexity,
            run_ml_benchmark=run_ml_benchmark,
            include_xgboost=base_config.include_xgboost,
            run_ml_tuning=base_config.run_ml_tuning,
            ml_tune_top_n=base_config.ml_tune_top_n,
            run_shap=base_config.run_shap,
            run_cnmf=base_config.run_cnmf,
            cnmf_k_list=base_config.cnmf_k_list,
            cnmf_reps=base_config.cnmf_reps,
            cnmf_beta=base_config.cnmf_beta,
        )
        if isinstance(data, str):
            data = pd.read_csv(data)
        results = AnalysisWorkflow(data, config=cfg).run()
        per_dataset[name] = results
        if "ml_results" in results:
            ml_tables[name] = results["ml_results"]

    comparison: Dict[str, Any] = {
        "datasets": per_dataset,
        "ml_tables": ml_tables,
        "outdir": outdir,
    }

    names = list(ml_tables.keys())
    if len(names) >= 2:
        merged = compare_model_results(
            ml_tables[names[0]],
            ml_tables[names[1]],
            dataset_a=names[0],
            dataset_b=names[1],
        )
        comparison["pairwise_comparison"] = merged
        merged.to_csv(os.path.join(outdir, "model_comparison_pairwise.csv"), index=False)
        plot_dataset_model_comparison(
            merged,
            os.path.join(outdir, "model_comparison_pairwise.png"),
            dataset_a=names[0],
            dataset_b=names[1],
        )

    family_rows = []
    for name, table in ml_tables.items():
        summary = summarize_model_families(table)
        if not summary.empty:
            summary.insert(0, "dataset", name)
            family_rows.append(summary)
    if family_rows:
        family_summary = pd.concat(family_rows, ignore_index=True)
        comparison["family_summary"] = family_summary
        family_summary.to_csv(os.path.join(outdir, "model_family_summary.csv"), index=False)

    return comparison