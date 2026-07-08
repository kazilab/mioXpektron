"""End-to-end analysis workflow for aligned peak matrices."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .cnmf import choose_k_by_pac, plot_pac_vs_k, run_cnmf, save_consensus_heatmap, save_factor_bars
from .embeddings import run_embeddings
from .metrics import calculate_multiclass_metrics, plot_confusion_matrix
from .ml import (
    evaluate_model,
    explain_with_shap,
    get_benchmark_models,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    prepare_ml_data,
    transform_features,
)
from .tuning import select_best_tuned_model, tune_top_models
from .plots import plot_heatmap_top_features, plot_volcano
from .prepare import prepare_matrix
from .stats import compute_univariate_tests

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for :class:`AnalysisWorkflow`."""

    outdir: str = "analysis_outputs"
    label_col: str = "Group"
    sample_col: str = "SampleName"
    group_a: Optional[str] = None
    group_b: Optional[str] = None
    reference_group: Optional[str] = None
    top_n_features: int = 25
    transform: str = "log1p"
    random_state: int = 0
    embedding_methods: Optional[List[str]] = None
    run_umap: bool = False
    run_tsne: bool = False
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    tsne_perplexity: float = 30.0
    run_ml_benchmark: bool = False
    include_xgboost: bool = True
    ml_top_n_plot: int = 10
    run_ml_tuning: bool = False
    ml_tune_top_n: int = 3
    run_shap: bool = False
    run_cnmf: bool = False
    cnmf_k_list: Optional[List[int]] = None
    cnmf_reps: int = 30
    cnmf_beta: str = "frobenius"
    cnmf_top_features: int = 15


class AnalysisWorkflow:
    """Orchestrate univariate stats, embeddings, ML, and optional cNMF."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[AnalysisConfig] = None,
        *,
        models: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.data = data
        self.config = config or AnalysisConfig()
        self.models = models
        self.results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute the configured analysis pipeline and write outputs."""
        cfg = self.config
        os.makedirs(cfg.outdir, exist_ok=True)

        X, y, meta = prepare_matrix(
            self.data,
            label_col=cfg.label_col,
            sample_col=cfg.sample_col,
        )
        self.results["X"] = X
        self.results["y"] = y
        self.results["meta"] = meta

        y.value_counts().to_frame("count").to_csv(
            os.path.join(cfg.outdir, "label_counts.csv")
        )

        uni = compute_univariate_tests(
            X,
            y,
            group_a=cfg.group_a,
            group_b=cfg.group_b,
            reference_group=cfg.reference_group,
        )
        self.results["univariate"] = uni
        uni.to_csv(os.path.join(cfg.outdir, "univariate_results.csv"), index=False)

        group_a = str(uni["group_a"].iloc[0]) if not uni.empty else cfg.group_a
        group_b = str(uni["group_b"].iloc[0]) if not uni.empty else cfg.group_b
        plot_volcano(
            uni,
            os.path.join(cfg.outdir, "volcano.png"),
            group_a=group_a,
            group_b=group_b,
        )

        X_embed = transform_features(X.values, method=cfg.transform)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_embed)
        self.results["X_scaled"] = X_scaled

        embedding_coords = run_embeddings(
            X_scaled,
            y,
            cfg.outdir,
            methods=cfg.embedding_methods,
            run_umap=cfg.run_umap,
            run_tsne=cfg.run_tsne,
            random_state=cfg.random_state,
            umap_n_neighbors=cfg.umap_n_neighbors,
            umap_min_dist=cfg.umap_min_dist,
            tsne_perplexity=cfg.tsne_perplexity,
        )
        coords = meta.copy()
        for col, values in embedding_coords.items():
            coords[col] = values
        coords.to_csv(os.path.join(cfg.outdir, "embeddings.csv"), index=False)
        self.results["embeddings"] = coords
        self.results["embedding_coords"] = embedding_coords

        plot_heatmap_top_features(
            X,
            y,
            uni,
            os.path.join(cfg.outdir, f"heatmap_top{cfg.top_n_features}.png"),
            top_n=cfg.top_n_features,
            label_col=cfg.label_col,
        )

        if cfg.run_ml_benchmark:
            self._run_ml_benchmark(X, y)

        if cfg.run_cnmf:
            self._run_cnmf(X, y)

        self.results["outdir"] = cfg.outdir
        logger.info("Analysis complete. Outputs written to %s", cfg.outdir)
        return self.results

    def _run_ml_benchmark(self, X: pd.DataFrame, y: pd.Series) -> None:
        cfg = self.config
        data_dict = prepare_ml_data(
            (X, y),
            label_col=cfg.label_col,
            sample_col=cfg.sample_col,
            transform=cfg.transform,
            random_state=cfg.random_state,
        )
        models = self.models or get_benchmark_models(
            random_state=cfg.random_state,
            include_boosting=cfg.include_xgboost,
        )
        fitted_models: Dict[str, Any] = {}
        rows = []
        for name, model in models.items():
            result, trained = evaluate_model(name, model, data_dict)
            rows.append(result)
            if result.get("status") == "success":
                fitted_models[name] = trained

        ml_results = pd.DataFrame(rows)
        if "test_accuracy" in ml_results.columns:
            ml_results = ml_results.sort_values(
                "test_accuracy", ascending=False, na_position="last"
            )
        ml_results = ml_results.reset_index(drop=True)
        self.results["ml_results"] = ml_results
        ml_results.to_csv(os.path.join(cfg.outdir, "model_performance.csv"), index=False)

        plot_model_comparison(
            ml_results,
            os.path.join(cfg.outdir, "model_comparison.png"),
            top_n=cfg.ml_top_n_plot,
        )
        plot_roc_curves(
            ml_results,
            data_dict,
            fitted_models,
            os.path.join(cfg.outdir, "roc_curves.png"),
        )
        from .compare import plot_family_comparison

        plot_family_comparison(
            ml_results,
            os.path.join(cfg.outdir, "model_family_comparison.png"),
        )

        best_model = None
        best_name = None
        if cfg.run_ml_tuning:
            tuning_df = tune_top_models(
                data_dict,
                ml_results,
                top_n=cfg.ml_tune_top_n,
                random_state=cfg.random_state,
            )
            self.results["ml_tuning"] = tuning_df
            if not tuning_df.empty:
                tuning_export = tuning_df.drop(columns=["best_estimator"], errors="ignore")
                tuning_export.to_csv(
                    os.path.join(cfg.outdir, "model_tuning_results.csv"),
                    index=False,
                )
                best_name, best_model = select_best_tuned_model(tuning_df)

        if best_model is None and not ml_results.empty:
            best_name = str(ml_results.iloc[0]["model_name"])
            best_model = fitted_models.get(best_name)

        if best_model is not None and best_name is not None:
            self._save_best_model_report(best_model, best_name, data_dict, ml_results)

        for name, model in list(fitted_models.items())[:2]:
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plot_feature_importance(
                model,
                data_dict["feature_names"],
                os.path.join(cfg.outdir, f"importance_{safe_name}.png"),
                title=f"Top features — {name}",
            )

        if cfg.run_shap and best_model is not None:
            explain_with_shap(
                best_model,
                data_dict,
                os.path.join(cfg.outdir, "shap_summary.png"),
            )

    def _save_best_model_report(
        self,
        model: Any,
        model_name: str,
        data_dict: Mapping[str, Any],
        ml_results: pd.DataFrame,
    ) -> None:
        cfg = self.config
        y_test = data_dict["y_test"]
        y_pred = model.predict(data_dict["X_test"])
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(data_dict["X_test"])
            except Exception:
                y_proba = None

        metrics = calculate_multiclass_metrics(
            y_test,
            y_pred,
            y_proba=y_proba,
            data_dict=data_dict,
        )
        self.results["best_model_metrics"] = metrics

        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            y_test,
            y_pred,
            os.path.join(cfg.outdir, f"confusion_matrix_{safe_name}.png"),
            data_dict=data_dict,
            title=f"Confusion matrix — {model_name}",
        )
        pd.DataFrame([metrics["overall_metrics"]]).to_csv(
            os.path.join(cfg.outdir, f"best_model_overall_metrics_{safe_name}.csv"),
            index=False,
        )
        metrics["per_class_metrics"].to_csv(
            os.path.join(cfg.outdir, f"best_model_per_class_metrics_{safe_name}.csv"),
            index=False,
        )

    def _run_cnmf(self, X: pd.DataFrame, y: pd.Series) -> None:
        cfg = self.config
        k_list = cfg.cnmf_k_list or [3, 4, 5]
        cnmf_results = run_cnmf(
            X.values.astype(float, copy=False),
            k_list,
            R=cfg.cnmf_reps,
            beta=cfg.cnmf_beta,
            outdir=cfg.outdir,
        )
        self.results["cnmf"] = cnmf_results

        pac_table = pd.DataFrame(
            {
                "k": sorted(cnmf_results.keys()),
                "PAC": [cnmf_results[k]["PAC"] for k in sorted(cnmf_results.keys())],
            }
        )
        pac_table.to_csv(os.path.join(cfg.outdir, "cnmf_PAC_vs_k.csv"), index=False)
        plot_pac_vs_k(cnmf_results, os.path.join(cfg.outdir, "cnmf_PAC_vs_k.png"))

        best_k = choose_k_by_pac(cnmf_results)
        np.save(os.path.join(cfg.outdir, "cnmf_W_best.npy"), cnmf_results[best_k]["W_mean"])
        np.save(os.path.join(cfg.outdir, "cnmf_H_best.npy"), cnmf_results[best_k]["H_mean"])
        save_consensus_heatmap(
            cnmf_results[best_k]["consensus"],
            y,
            os.path.join(cfg.outdir, "cnmf_consensus_best.png"),
            label_col=cfg.label_col,
        )
        save_factor_bars(
            cnmf_results[best_k]["H_mean"],
            list(X.columns),
            cfg.outdir,
            topm=cfg.cnmf_top_features,
        )

        summary_path = os.path.join(cfg.outdir, f"cnmf_summary_k{best_k}.txt")
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write("\nSelected as best k by minimal PAC.\n")


def run_analysis(
    data: Union[pd.DataFrame, str],
    *,
    config: Optional[AnalysisConfig] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper around :class:`AnalysisWorkflow`."""
    if isinstance(data, str):
        data = pd.read_csv(data)
    if config is None:
        config = AnalysisConfig(**kwargs) if kwargs else AnalysisConfig()
    elif kwargs:
        raise ValueError("Pass either config or keyword overrides, not both.")
    return AnalysisWorkflow(data, config=config).run()