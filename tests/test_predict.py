import pytest

from fraud_detection.data import split_xy
from fraud_detection.predict import FraudPredictor
from fraud_detection.train import save_artifacts, train_model


@pytest.fixture
def predictor(sample_df, tmp_path):
    X, y = split_xy(sample_df)
    result = train_model(X, y)
    save_artifacts(result, tmp_path)
    return FraudPredictor.from_dir(tmp_path), X


def test_train_model_returns_metrics(sample_df):
    X, y = split_xy(sample_df)
    result = train_model(X, y)
    assert set(result["metrics"]) == {"roc_auc", "recall", "precision", "f1"}
    assert result["metrics"]["roc_auc"] > 0.8


def test_predict_proba_range_and_length(predictor):
    pred, X = predictor
    proba = pred.predict_proba(X)
    assert len(proba) == len(X)
    assert proba.between(0, 1).all()


def test_predict_returns_binary(predictor):
    pred, X = predictor
    labels = pred.predict(X)
    assert set(labels.unique()) <= {0, 1}


def test_predict_invalid_threshold(predictor):
    pred, X = predictor
    with pytest.raises(ValueError):
        pred.predict(X, threshold=1.5)
