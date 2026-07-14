"""Tests for analysis stats, matrix preparation, metrics, compare, and embeddings."""

import numpy as np
import pandas as pd
import pytest

from mioXpektron.analysis import stats, prepare, metrics, compare, optional, embeddings


# --------------------------- stats ---------------------------

def test_bh_fdr_monotone_and_bounded():
    p = np.array([0.01, 0.02, 0.03, 0.5, 0.9])
    q = stats.bh_fdr(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    assert np.all(q >= p)


def test_bh_fdr_empty():
    assert stats.bh_fdr(np.array([])).size == 0


def test_sanitize_group_name():
    assert stats._sanitize_group_name("Group A!") == "Group_A"
    assert stats._sanitize_group_name("   ") == "group"


def make_two_group_data(seed=0):
    rng = np.random.default_rng(seed)
    n = 20
    X = pd.DataFrame(
        {
            "10.0": np.concatenate([rng.normal(5, 1, n), rng.normal(10, 1, n)]),
            "20.0": np.concatenate([rng.normal(3, 1, n), rng.normal(3, 1, n)]),
        }
    )
    y = pd.Series(["Control"] * n + ["Cancer"] * n)
    return X, y


def test_compute_univariate_tests_columns():
    X, y = make_two_group_data()
    res = stats.compute_univariate_tests(X, y, group_a="Cancer", group_b="Control")
    for col in ["feature", "log2_FC", "p_value", "q_value", "group_a", "group_b"]:
        assert col in res.columns
    # The separated feature should be significant.
    sig = res[res["feature"] == "10.0"].iloc[0]
    assert sig["p_value"] < 0.05


def test_compute_univariate_tests_auto_groups():
    X, y = make_two_group_data()
    res = stats.compute_univariate_tests(X, y)
    assert len(res) == 2


def test_compute_univariate_tests_length_mismatch():
    X, y = make_two_group_data()
    with pytest.raises(ValueError, match="same number of samples"):
        stats.compute_univariate_tests(X, y.iloc[:5])


def test_resolve_groups_same_group_raises():
    y = pd.Series(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="must be different"):
        stats._resolve_groups(y, group_a="A", group_b="A")


def test_resolve_groups_missing_group_raises():
    y = pd.Series(["A", "B"])
    with pytest.raises(ValueError, match="not found"):
        stats._resolve_groups(y, group_a="Z", group_b="A")


def test_resolve_groups_single_group_raises():
    y = pd.Series(["A", "A"])
    with pytest.raises(ValueError, match="At least two groups"):
        stats._resolve_groups(y)


# --------------------------- prepare ---------------------------

def test_prepare_matrix_from_long_table():
    df = pd.DataFrame(
        {
            "SampleName": ["s1", "s2"],
            "Group": ["Control", "Cancer"],
            "10.0": [1.0, 2.0],
            "20.0": [3.0, 4.0],
        }
    )
    X, y, meta = prepare.prepare_matrix(df)
    assert list(X.columns) == ["10.0", "20.0"]
    assert list(y) == ["Control", "Cancer"]
    assert len(meta) == 2


def test_prepare_matrix_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        prepare.prepare_matrix(pd.DataFrame())


def test_prepare_matrix_missing_meta_raises():
    df = pd.DataFrame({"10.0": [1.0], "20.0": [2.0]})
    with pytest.raises(ValueError, match="missing"):
        prepare.prepare_matrix(df)


def test_prepare_matrix_no_features_raises():
    df = pd.DataFrame({"SampleName": ["s1"], "Group": ["A"]})
    with pytest.raises(ValueError, match="No feature columns"):
        prepare.prepare_matrix(df)


def test_prepare_matrix_coerces_and_fills():
    df = pd.DataFrame(
        {
            "SampleName": ["s1", "s2"],
            "Group": ["A", "B"],
            "10.0": ["1.0", "not-a-number"],
        }
    )
    X, _, _ = prepare.prepare_matrix(df, fill_na=0.0)
    assert X.iloc[1, 0] == 0.0


def test_infer_feature_columns_mz_like():
    df = pd.DataFrame(columns=["SampleName", "Group", "10.5", "20.1"])
    cols = prepare.infer_feature_columns(df)
    assert cols == ["10.5", "20.1"]


def test_looks_like_mz():
    assert prepare._looks_like_mz("12.5")
    assert prepare._looks_like_mz("100")
    assert not prepare._looks_like_mz("")
    assert not prepare._looks_like_mz("abc")


# --------------------------- metrics ---------------------------

def test_get_class_names_from_explicit():
    assert metrics.get_class_names(class_names=["a", "b"]) == ["a", "b"]


def test_get_class_names_from_n_classes():
    assert metrics.get_class_names(n_classes=3) == ["Class_0", "Class_1", "Class_2"]


def test_get_class_names_unresolvable_raises():
    with pytest.raises(ValueError):
        metrics.get_class_names()


def test_calculate_multiclass_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    res = metrics.calculate_multiclass_metrics(y, y, class_names=["A", "B"])
    assert res["overall_metrics"]["accuracy"] == 1.0
    assert res["confusion_matrix"].shape == (2, 2)
    assert "per_class_metrics" in res


def test_calculate_multiclass_metrics_with_proba():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    res = metrics.calculate_multiclass_metrics(
        y_true, y_pred, y_proba=proba, class_names=["A", "B"]
    )
    assert np.isfinite(res["overall_metrics"]["roc_auc"])


def test_plot_confusion_matrix_writes_file(tmp_path):
    y = np.array([0, 0, 1, 1])
    path = tmp_path / "cm.png"
    metrics.plot_confusion_matrix(y, y, str(path), class_names=["A", "B"])
    assert path.exists()


# --------------------------- compare ---------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("LogisticRegression", "Linear"),
        ("RandomForest", "Tree"),
        ("XGBoost", "Boosting"),
        ("SVC", "SVM"),
        ("KNN", "KNN"),
        ("NaiveBayes", "Naive Bayes"),
        ("MLPClassifier", "Neural Network"),
        ("Weird", "Other"),
    ],
)
def test_categorize_model(name, expected):
    assert compare.categorize_model(name) == expected


def test_summarize_model_families():
    df = pd.DataFrame(
        {
            "status": ["success", "success", "failed"],
            "model_name": ["RandomForest", "LogisticRegression", "SVC"],
            "test_accuracy": [0.9, 0.8, 0.0],
            "train_time": [1.0, 0.5, 0.0],
        }
    )
    summary = compare.summarize_model_families(df)
    assert "family" in summary.columns
    assert set(summary["family"]) == {"Tree", "Linear"}


def test_summarize_model_families_empty():
    df = pd.DataFrame({"status": ["failed"], "model_name": ["x"],
                       "test_accuracy": [0.0], "train_time": [0.0]})
    assert compare.summarize_model_families(df).empty


# --------------------------- optional ---------------------------

def test_analysis_capabilities_keys():
    caps = optional.analysis_capabilities()
    assert caps["pca"] is True
    assert set(["umap", "tsne", "xgboost", "shap", "cnmf"]).issubset(caps)


def test_missing_packages_returns_dict():
    assert isinstance(optional.missing_packages(), dict)


# --------------------------- embeddings ---------------------------

def test_resolve_embedding_methods_default_pca():
    assert embeddings.resolve_embedding_methods() == ["pca"]


def test_resolve_embedding_methods_adds_pca():
    methods = embeddings.resolve_embedding_methods(embedding_methods=["tsne"])
    assert methods[0] == "pca"
    assert "tsne" in methods


def test_resolve_embedding_methods_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedding"):
        embeddings.resolve_embedding_methods(embedding_methods=["bogus"])


def test_compute_pca_writes_and_returns(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    path = tmp_path / "pca.png"
    Z, var = embeddings.compute_pca(X, y, str(path))
    assert Z.shape == (20, 2)
    assert var.shape == (2,)
    assert path.exists()
