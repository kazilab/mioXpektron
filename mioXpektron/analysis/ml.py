"""Machine-learning benchmarking for aligned m/z feature matrices."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .optional import HAVE_SHAP, HAVE_XGBOOST
from .prepare import prepare_matrix

logger = logging.getLogger(__name__)

_HAVE_SHAP = HAVE_SHAP
_HAVE_XGBOOST = HAVE_XGBOOST

if _HAVE_SHAP:
    import shap

if _HAVE_XGBOOST:
    import xgboost as xgb


def model_needs_scaling(model: Any) -> bool:
    """Return True when a classifier benefits from feature scaling."""
    name = model.__class__.__name__.lower()
    tree_based = (
        "decisiontree",
        "randomforest",
        "extratrees",
        "xgb",
        "lgb",
        "gradientboosting",
        "adaboost",
        "bagging",
    )
    return not any(token in name for token in tree_based)


def transform_features(
    X: np.ndarray,
    *,
    method: str = "log1p",
) -> np.ndarray:
    """Apply a variance-stabilising transform before embedding or ML."""
    if method == "none":
        return X.astype(float, copy=False)
    if method == "log1p":
        return np.log1p(np.clip(X, 0.0, None))
    raise ValueError(f"Unknown transform method: {method}")


def _handle_missing(
    X: np.ndarray,
    *,
    strategy: str,
) -> np.ndarray:
    if not np.any(np.isnan(X)):
        return X
    if strategy == "raise":
        raise ValueError("Missing values found in feature matrix.")
    if strategy == "drop":
        raise ValueError("handle_missing='drop' is not supported after matrix preparation.")
    if strategy == "mean":
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_mean, inds[1])
        return X
    if strategy == "zero":
        return np.nan_to_num(X, nan=0.0)
    raise ValueError(f"Invalid handle_missing option: {strategy}")


def prepare_ml_data(
    data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
    *,
    label_col: str = "Group",
    sample_col: str = "SampleName",
    test_size: float = 0.2,
    random_state: int = 42,
    transform: str = "log1p",
    scale_features: bool = True,
    handle_missing: str = "zero",
) -> Dict[str, Any]:
    """Prepare an aligned matrix for supervised classification benchmarks.

    Parameters
    ----------
    data
        Either a table with metadata columns or a ``(X, y)`` tuple from
        :func:`~mioXpektron.analysis.prepare_matrix`.
    """
    if isinstance(data, tuple):
        X_df, y = data[0], data[1]
        meta = pd.DataFrame({sample_col: X_df.index, label_col: y.values})
    else:
        X_df, y, meta = prepare_matrix(
            data,
            label_col=label_col,
            sample_col=sample_col,
        )

    feature_names = list(X_df.columns)
    X_raw = _handle_missing(X_df.values.astype(float), strategy=handle_missing)
    X_raw = transform_features(X_raw, method=transform)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.values)

    if len(np.unique(y_encoded)) < 2:
        raise ValueError("At least two classes are required for classification.")

    stratify = y_encoded if len(np.unique(y_encoded)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": le,
        "class_names": list(le.classes_),
        "feature_names": feature_names,
        "meta": meta,
    }


def get_benchmark_models(
    *,
    random_state: int = 42,
    include_boosting: bool = True,
) -> Dict[str, Any]:
    """Return a compact set of classifiers suitable for m/z matrices."""
    models: Dict[str, Any] = {
        "Logistic Regression (L1)": LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Logistic Regression (L2)": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Random Forest (100)": RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Random Forest (500)": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
    }

    if include_boosting and _HAVE_XGBOOST:
        for n_est, label in ((50, "50"), (100, "100"), (200, "200")):
            models[f"XGBoost ({label})"] = xgb.XGBClassifier(
                n_estimators=n_est,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
            )

    return models


def _roc_auc_score_safe(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
) -> float:
    if y_proba is None:
        return float("nan")
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        return float("nan")


def evaluate_model(
    name: str,
    model: Any,
    data_dict: Mapping[str, Any],
    *,
    cv_folds: int = 5,
    max_train_time_for_cv: float = 30.0,
) -> Tuple[Dict[str, Any], Any]:
    """Fit one classifier and return hold-out and CV metrics."""
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    X_train_raw = data_dict.get("X_train_raw", X_train)

    results: Dict[str, Any] = {"model_name": name}
    try:
        needs_scaling = model_needs_scaling(model)
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None

        n_classes = len(data_dict.get("class_names", [])) or len(np.unique(y_test))
        results.update(
            {
                "status": "success",
                "train_time": train_time,
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "test_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "test_roc_auc": _roc_auc_score_safe(y_test, y_proba, n_classes),
                "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
                "y_pred": y_pred,
                "y_proba": y_proba,
            }
        )
        results["overfit_gap"] = results["train_accuracy"] - results["test_accuracy"]

        if train_time < max_train_time_for_cv:
            params = {k: v for k, v in model.get_params().items() if "__" not in k}
            if needs_scaling:
                cv_estimator: Any = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", model.__class__(**params)),
                    ]
                )
            else:
                cv_estimator = model.__class__(**params)
            cv_scores = cross_val_score(
                cv_estimator,
                X_train_raw,
                y_train,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        else:
            results["cv_mean"] = float("nan")
            results["cv_std"] = float("nan")
    except Exception as exc:
        logger.warning("Model %s failed: %s", name, exc)
        results.update({"status": "failed", "error": str(exc)})
        for key in (
            "train_time",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "train_accuracy",
            "cv_mean",
            "cv_std",
            "overfit_gap",
        ):
            results.setdefault(key, float("nan"))

    return results, model


def evaluate_all_models(
    models: Mapping[str, Any],
    data_dict: Mapping[str, Any],
    *,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """Benchmark a mapping of classifiers and return a sorted results table."""
    rows: List[Dict[str, Any]] = []
    total = len(models)
    for idx, (name, model) in enumerate(models.items(), start=1):
        logger.info("[%d/%d] Evaluating %s", idx, total, name)
        result, _ = evaluate_model(name, model, data_dict)
        rows.append(result)
    df = pd.DataFrame(rows)
    if "test_accuracy" in df.columns:
        df = df.sort_values("test_accuracy", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def plot_model_comparison(
    results_df: pd.DataFrame,
    savepath: str,
    *,
    dataset_name: str = "dataset",
    top_n: int = 10,
) -> None:
    """Bar chart of top model accuracies."""
    df = results_df[results_df["status"] == "success"].head(top_n)
    if df.empty:
        return

    plt.figure(figsize=(8, max(4, 0.4 * len(df))))
    y_pos = np.arange(len(df))
    plt.barh(y_pos, df["test_accuracy"], alpha=0.8)
    plt.yticks(y_pos, df["model_name"], fontsize=9)
    plt.xlabel("Test accuracy")
    plt.title(f"Model comparison — {dataset_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_roc_curves(
    results_df: pd.DataFrame,
    data_dict: Mapping[str, Any],
    models: Mapping[str, Any],
    savepath: str,
    *,
    top_n: int = 2,
) -> None:
    """Plot ROC curves for the top-performing probabilistic models."""
    class_names = data_dict["class_names"]
    if len(class_names) != 2:
        logger.info("ROC curves are only plotted for binary classification.")
        return

    successful = results_df[results_df["status"] == "success"].head(top_n)
    plt.figure(figsize=(6, 5))
    y_test = data_dict["y_test"]

    for _, row in successful.iterrows():
        name = row["model_name"]
        model = models[name]
        if not hasattr(model, "predict_proba"):
            continue
        y_proba = model.predict_proba(data_dict["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    savepath: str,
    *,
    title: str,
    top_n: int = 15,
) -> None:
    """Plot absolute LR coefficients or RF feature importances."""
    values: Optional[np.ndarray] = None
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    if values is None or len(values) != len(feature_names):
        return

    top_idx = np.argsort(values)[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(top_idx)), values[top_idx])
    plt.xticks(range(len(top_idx)), [feature_names[i] for i in top_idx], rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def _resolve_shap_model(model: Any) -> Any:
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def _build_shap_explainer(model: Any, X_background: np.ndarray) -> Any:
    resolved = _resolve_shap_model(model)
    name = resolved.__class__.__name__.lower()
    if "xgb" in name or "randomforest" in name or "gradientboosting" in name:
        return shap.TreeExplainer(resolved)
    if "pipeline" in name:
        return shap.Explainer(model, X_background)
    return shap.Explainer(resolved, X_background)


def explain_with_shap(
    model: Any,
    data_dict: Mapping[str, Any],
    savepath: str,
    *,
    max_samples: int = 100,
    max_background: int = 200,
) -> Optional[np.ndarray]:
    """SHAP beeswarm and bar plots when shap is installed."""
    if not _HAVE_SHAP:
        logger.info(
            "shap is not installed; skipping SHAP explanation. "
            "Install with: pip install mioXpektron[analysis]"
        )
        return None

    X_test = data_dict["X_test"]
    X_train = data_dict["X_train"]
    feature_names = data_dict["feature_names"]

    X_explain = X_test[:max_samples]
    X_background = X_train[:max_background]

    try:
        explainer = _build_shap_explainer(model, X_background)
        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            explanation = explainer(X_explain)
            shap_values = np.asarray(explanation.values)

        base, ext = os.path.splitext(savepath)
        summary_path = savepath if ext else f"{savepath}_summary.png"
        bar_path = f"{base}_bar.png" if ext else f"{savepath}_bar.png"

        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=200, bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.close()

        values_path = f"{base}_values.npy" if ext else f"{savepath}_values.npy"
        np.save(values_path, shap_values)
        return shap_values
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        plt.close()
        return None