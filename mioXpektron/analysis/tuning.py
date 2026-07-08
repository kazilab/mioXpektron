"""Hyperparameter tuning for top benchmark models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from .ml import model_needs_scaling
from .optional import HAVE_XGBOOST as _HAVE_XGBOOST

logger = logging.getLogger(__name__)

if _HAVE_XGBOOST:
    import xgboost as xgb


def get_tuning_grid(model_name: str) -> Optional[Dict[str, list]]:
    """Return a parameter grid for supported model names."""
    name = model_name.lower()
    if "random forest" in name:
        return {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    if "xgboost" in name and _HAVE_XGBOOST:
        return {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }
    if "gradient boosting" in name:
        return {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
    return None


def _build_base_estimator(model_name: str, random_state: int) -> Optional[Any]:
    name = model_name.lower()
    if "random forest" in name:
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    if "xgboost" in name and _HAVE_XGBOOST:
        return xgb.XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    if "gradient boosting" in name:
        return GradientBoostingClassifier(random_state=random_state)
    return None


def tune_top_models(
    data_dict: Mapping[str, Any],
    results_df: pd.DataFrame,
    *,
    top_n: int = 3,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: int = 0,
) -> pd.DataFrame:
    """Grid-search hyperparameters for the top-performing models."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    successful = results_df[results_df["status"] == "success"]
    rows = []
    tuned_count = 0

    for _, row in successful.iterrows():
        if tuned_count >= top_n:
            break
        model_name = str(row["model_name"])
        param_grid = get_tuning_grid(model_name)
        estimator = _build_base_estimator(model_name, random_state)
        if param_grid is None or estimator is None:
            logger.info("No tuning grid for %s; skipping.", model_name)
            continue

        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        if model_needs_scaling(estimator):
            search_estimator: Any = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
            prefix = "model__"
            grid = {f"{prefix}{k}": v for k, v in param_grid.items()}
        else:
            search_estimator = estimator
            grid = param_grid

        search = GridSearchCV(
            search_estimator,
            grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=verbose,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        if hasattr(best, "named_steps"):
            predictor = best.named_steps["model"]
            X_test = best.predict(data_dict["X_test"])
        else:
            predictor = best
            X_test = best.predict(data_dict["X_test"])

        y_test = data_dict["y_test"]
        tuned_accuracy = accuracy_score(y_test, X_test)
        tuned_f1 = f1_score(y_test, X_test, average="weighted", zero_division=0)

        rows.append(
            {
                "model_name": model_name,
                "original_accuracy": float(row["test_accuracy"]),
                "tuned_accuracy": tuned_accuracy,
                "improvement": tuned_accuracy - float(row["test_accuracy"]),
                "tuned_f1": tuned_f1,
                "cv_score": float(search.best_score_),
                "best_params": search.best_params_,
                "best_estimator": best,
            }
        )
        tuned_count += 1

    return pd.DataFrame(rows)


def select_best_tuned_model(
    tuning_df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[Any]]:
    """Return the name and estimator of the best tuned model."""
    if tuning_df.empty:
        return None, None
    best_row = tuning_df.sort_values("tuned_accuracy", ascending=False).iloc[0]
    return str(best_row["model_name"]), best_row.get("best_estimator")