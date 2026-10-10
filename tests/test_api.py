import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import app, get_predictor, risk_level
from fraud_detection.data import split_xy
from fraud_detection.predict import FraudPredictor
from fraud_detection.train import save_artifacts, train_model


@pytest.fixture
def client(sample_df, tmp_path):
    X, y = split_xy(sample_df)
    save_artifacts(train_model(X, y), tmp_path)
    predictor = FraudPredictor.from_dir(tmp_path)
    app.dependency_overrides[get_predictor] = lambda: predictor
    yield TestClient(app), X
    app.dependency_overrides.clear()


def rows(X, n=3):
    return X.head(n).to_dict(orient="records")


def test_health(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_one_result_per_row(client):
    c, X = client
    response = c.post("/predict", json={"transactions": rows(X, 3)})
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert 0.0 <= p["fraud_probability"] <= 1.0
        assert p["risk_level"] in {"low", "medium", "high"}


def test_threshold_changes_decision(client):
    c, X = client
    body = {"transactions": rows(X, 5)}
    strict = c.post("/predict?threshold=1.0", json=body).json()["predictions"]
    loose = c.post("/predict?threshold=0.0", json=body).json()["predictions"]
    assert not any(p["is_fraud"] for p in strict)
    assert all(p["is_fraud"] for p in loose)


def test_negative_amount_is_rejected(client):
    c, X = client
    bad = rows(X, 1)
    bad[0]["Amount"] = -5
    assert c.post("/predict", json={"transactions": bad}).status_code == 422


def test_missing_feature_is_rejected(client):
    c, X = client
    bad = rows(X, 1)
    del bad[0]["V14"]
    assert c.post("/predict", json={"transactions": bad}).status_code == 422


def test_empty_batch_is_rejected(client):
    c, _ = client
    assert c.post("/predict", json={"transactions": []}).status_code == 422


def test_invalid_threshold_is_rejected(client):
    c, X = client
    response = c.post("/predict?threshold=2", json={"transactions": rows(X, 1)})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "probability, expected",
    [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.69, "medium"), (0.7, "high")],
)
def test_risk_level_bands(probability, expected):
    assert risk_level(probability) == expected
