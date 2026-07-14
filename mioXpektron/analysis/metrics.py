"""Classification metrics and diagnostic plots."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_class_names(
    *,
    data_dict: Optional[Mapping[str, Any]] = None,
    label_encoder: Optional[Any] = None,
    class_names: Optional[Sequence[str]] = None,
    n_classes: Optional[int] = None,
) -> List[str]:
    """Resolve human-readable class names for encoded labels."""
    if class_names is not None:
        return list(class_names)

    if data_dict is not None:
        if data_dict.get("class_names") is not None:
            names = data_dict["class_names"]
            return list(names.tolist() if hasattr(names, "tolist") else names)
        if data_dict.get("label_encoder") is not None:
            label_encoder = data_dict["label_encoder"]

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        classes = label_encoder.classes_
        return list(classes.tolist() if hasattr(classes, "tolist") else classes)

    if n_classes is not None:
        return [f"Class_{i}" for i in range(n_classes)]

    raise ValueError("Could not resolve class names.")


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    data_dict: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute overall and per-class classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    names = get_class_names(
        data_dict=data_dict,
        class_names=class_names,
        n_classes=n_classes,
    )

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "jaccard_macro": float(
            jaccard_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "jaccard_weighted": float(
            jaccard_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    if y_proba is not None:
        try:
            if n_classes == 2:
                overall["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                overall["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except ValueError:
            overall["roc_auc"] = float("nan")
    else:
        overall["roc_auc"] = float("nan")

    per_class_rows = []
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class_rows.append(
            {
                "class": name,
                "support": int(cm[i, :].sum()),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
            }
        )

    per_class = pd.DataFrame(per_class_rows).sort_values("f1", ascending=False)
    return {
        "overall_metrics": overall,
        "per_class_metrics": per_class.reset_index(drop=True),
        "confusion_matrix": cm,
        "class_names": names,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    savepath: str,
    *,
    class_names: Optional[Sequence[str]] = None,
    data_dict: Optional[Mapping[str, Any]] = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> np.ndarray:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    names = get_class_names(
        data_dict=data_dict,
        class_names=class_names,
        n_classes=cm.shape[0],
    )
    display = cm.astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, where=row_sums > 0)

    plt.figure(figsize=(max(5, cm.shape[0]), max(4, cm.shape[0] - 1)))
    plt.imshow(display, interpolation="nearest", cmap="Blues")
    plt.colorbar(label="Proportion" if normalize else "Count")
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    fmt = ".2f" if normalize else "d"
    thresh = display.max() / 2 if display.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = display[i, j]
            text = format(value, fmt) if normalize else str(int(cm[i, j]))
            color = "white" if value > thresh else "black"
            plt.text(j, i, text, ha="center", va="center", color=color)

    acc = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f"Accuracy: {acc:.3f}", ha="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close()
    return cm