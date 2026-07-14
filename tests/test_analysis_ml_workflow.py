"""Tests for ML benchmarking, tuning, plots, cNMF, and the analysis workflow."""

import numpy as np
import pandas as pd
import pytest

from mioXpektron.analysis import ml, tuning, plots, cnmf
from mioXpektron.analysis.workflow import AnalysisConfig, AnalysisWorkflow, run_analysis
from mioXpektron.analysis.stats import compute_univariate_tests


def make_classification_table(n_per_class=20, n_features=12, seed=0):
    """Build a long-format table with SampleName / Group / feature columns."""
    rng = np.random.default_rng(seed)
    n = n_per_class * 2
    # First few features are class-discriminative.
    control = rng.normal(5.0, 1.0, size=(n_per_class, n_features))
    cancer = rng.normal(5.0, 1.0, size=(n_per_class, n_features))
    cancer[:, :3] += 4.0  # separable signal
    X = np.vstack([control, cancer])
    cols = {f"{10.0 + i:.1f}": X[:, i] for i in range(n_features)}
    df = pd.DataFrame(cols)
    df.insert(0, "Group", ["Control"] * n_per_class + ["Cancer"] * n_per_class)
    df.insert(0, "SampleName", [f"s{i}" for i in range(n)])
    return df


# --------------------------- ml helpers ---------------------------

def test_model_needs_scaling_tree_false():
    from sklearn.ensemble import RandomForestClassifier

    assert ml.model_needs_scaling(RandomForestClassifier()) is False


def test_model_needs_scaling_linear_true():
    from sklearn.linear_model import LogisticRegression

    assert ml.model_needs_scaling(LogisticRegression()) is True


def test_transform_features_log1p():
    X = np.array([[0.0, 9.0]])
    np.testing.assert_allclose(ml.transform_features(X, method="log1p"),
                               np.log1p(X))


def test_transform_features_none():
    X = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(ml.transform_features(X, method="none"), X)


def test_transform_features_unknown():
    with pytest.raises(ValueError):
        ml.transform_features(np.zeros((1, 1)), method="bogus")


def test_prepare_ml_data_from_tuple():
    df = make_classification_table()
    X = df.drop(columns=["SampleName", "Group"])
    X.index = df["SampleName"].values
    y = pd.Series(df["Group"].values, index=df["SampleName"].values)
    data = ml.prepare_ml_data((X, y))
    assert data["X_train"].shape[1] == X.shape[1]
    assert len(data["class_names"]) == 2
    assert data["X_test"].shape[0] > 0


def test_prepare_ml_data_single_class_raises():
    df = make_classification_table()
    df["Group"] = "Control"
    X = df.drop(columns=["SampleName", "Group"])
    y = pd.Series(df["Group"].values)
    with pytest.raises(ValueError, match="two classes"):
        ml.prepare_ml_data((X, y))


def test_get_benchmark_models_returns_estimators():
    models = ml.get_benchmark_models(include_boosting=False)
    assert "Gaussian Naive Bayes" in models
    assert all(hasattr(m, "fit") for m in models.values())


def test_evaluate_model_success():
    from sklearn.naive_bayes import GaussianNB

    df = make_classification_table()
    X = df.drop(columns=["SampleName", "Group"])
    X.index = df["SampleName"].values
    y = pd.Series(df["Group"].values, index=df["SampleName"].values)
    data = ml.prepare_ml_data((X, y))
    result, model = ml.evaluate_model("GaussianNB", GaussianNB(), data)
    assert result["status"] == "success"
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_evaluate_all_models_table():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    df = make_classification_table()
    X = df.drop(columns=["SampleName", "Group"])
    X.index = df["SampleName"].values
    y = pd.Series(df["Group"].values, index=df["SampleName"].values)
    data = ml.prepare_ml_data((X, y))
    models = {"NB": GaussianNB(), "LR": LogisticRegression(max_iter=500)}
    res = ml.evaluate_all_models(models, data)
    assert len(res) == 2
    assert "test_accuracy" in res.columns


# --------------------------- tuning ---------------------------

def test_get_tuning_grid_random_forest():
    grid = tuning.get_tuning_grid("Random Forest (100)")
    assert "n_estimators" in grid


def test_get_tuning_grid_gradient_boosting():
    grid = tuning.get_tuning_grid("Gradient Boosting")
    assert "learning_rate" in grid


def test_get_tuning_grid_unknown_returns_none():
    assert tuning.get_tuning_grid("Gaussian Naive Bayes") is None


def test_build_base_estimator():
    est = tuning._build_base_estimator("Random Forest (100)", 0)
    assert est is not None
    assert tuning._build_base_estimator("KNN", 0) is None


# --------------------------- plots ---------------------------

def test_plot_volcano_writes_file(tmp_path):
    df = make_classification_table()
    X = df.drop(columns=["SampleName", "Group"])
    y = pd.Series(df["Group"].values)
    uni = compute_univariate_tests(X, y, group_a="Cancer", group_b="Control")
    path = tmp_path / "volcano.png"
    plots.plot_volcano(uni, str(path))
    assert path.exists()


def test_plot_heatmap_top_features_writes_file(tmp_path):
    df = make_classification_table()
    X = df.drop(columns=["SampleName", "Group"])
    y = pd.Series(df["Group"].values)
    uni = compute_univariate_tests(X, y, group_a="Cancer", group_b="Control")
    path = tmp_path / "heatmap.png"
    plots.plot_heatmap_top_features(X, y, uni, str(path), top_n=5)
    assert path.exists()


def test_plot_tsne_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = pd.Series(["A"] * 15 + ["B"] * 15)
    path = tmp_path / "tsne.png"
    plots.plot_tsne(X, y, str(path), perplexity=5.0)
    assert path.exists()


# --------------------------- cnmf ---------------------------

def test_run_cnmf_and_choose_k(tmp_path):
    rng = np.random.default_rng(0)
    # Two clear blocks → rank-2 structure.
    block = rng.random((20, 6))
    X = np.abs(np.vstack([
        np.hstack([block, block * 0.1]),
        np.hstack([block * 0.1, block]),
    ]))
    results = cnmf.run_cnmf(X, [2, 3], R=3, max_iter=100)
    assert set(results.keys()) == {2, 3}
    best_k = cnmf.choose_k_by_pac(results)
    assert best_k in (2, 3)


def test_cnmf_pac_score_bounds():
    consensus = np.eye(5)
    score = cnmf._pac_score(consensus)
    assert 0.0 <= score <= 1.0


# --------------------------- workflow (end to end) ---------------------------

def test_analysis_workflow_minimal(tmp_path):
    df = make_classification_table()
    cfg = AnalysisConfig(outdir=str(tmp_path / "out"), top_n_features=5)
    results = AnalysisWorkflow(df, config=cfg).run()
    assert "univariate" in results
    assert "embeddings" in results
    assert (tmp_path / "out" / "univariate_results.csv").exists()
    assert (tmp_path / "out" / "volcano.png").exists()


def test_workflow_with_ml_benchmark(tmp_path):
    from sklearn.naive_bayes import GaussianNB

    df = make_classification_table()
    cfg = AnalysisConfig(
        outdir=str(tmp_path / "out2"),
        top_n_features=5,
        run_ml_benchmark=True,
        include_xgboost=False,
    )
    results = AnalysisWorkflow(df, config=cfg, models={"NB": GaussianNB()}).run()
    assert "ml_results" in results
    assert (tmp_path / "out2" / "model_performance.csv").exists()


def test_run_analysis_wrapper_kwargs(tmp_path):
    df = make_classification_table()
    results = run_analysis(df, outdir=str(tmp_path / "out3"), top_n_features=5)
    assert "univariate" in results
    assert (tmp_path / "out3" / "embeddings.csv").exists()
