import numpy as np

from digit_robustness.evaluation import classification_metrics, confusion_10x10, top_confusions
from digit_robustness.modeling import MLPConfig, build_logistic_regression, build_mlp, build_random_forest, predict_proba


def test_metrics_and_confusion_matrix_contracts():
    y_true = np.arange(10)
    y_pred = np.arange(10)
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    cm = confusion_10x10(y_true, y_pred)
    assert cm.shape == (10, 10)
    assert cm.trace() == 10
    assert top_confusions(cm).empty


def test_model_builders_expose_expected_hyperparameters():
    lr = build_logistic_regression(C=0.7, solver="lbfgs")
    rf = build_random_forest(n_estimators=20, max_depth=8, n_jobs=1)
    assert lr.C == 0.7 and lr.solver == "lbfgs"
    assert rf.n_estimators == 20 and rf.max_depth == 8


def test_mlp_output_probabilities_are_valid():
    model = build_mlp(output_units=10, config=MLPConfig(16, 8, 1e-3, 8), seed=42)
    x = np.zeros((3, 784), dtype=np.float32)
    p = predict_proba(model, x)
    assert p.shape == (3, 10)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(p).all()
