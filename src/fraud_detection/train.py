"""Model training and evaluation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_detection.data import load_data, split_xy
from fraud_detection.features import FEATURE_COLUMNS, build_features, fit_scalers


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a class-weighted logistic regression and return artifacts + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler_amount, scaler_time = fit_scalers(X_train)
    train_features = build_features(X_train, scaler_amount, scaler_time)
    test_features = build_features(X_test, scaler_amount, scaler_time)

    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state
    )
    model.fit(train_features, y_train)

    proba = model.predict_proba(test_features)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    return {
        "model": model,
        "scaler_amount": scaler_amount,
        "scaler_time": scaler_time,
        "feature_columns": list(FEATURE_COLUMNS),
        "metrics": metrics,
    }


def save_artifacts(result: dict, out_dir: str | Path) -> Path:
    """Save model, scalers and feature columns using the repo's file names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], out_dir / "fraud_detection_model.pkl")
    joblib.dump(result["scaler_amount"], out_dir / "scaler_amount.pkl")
    joblib.dump(result["scaler_time"], out_dir / "scaler_time.pkl")
    joblib.dump(result["feature_columns"], out_dir / "feature_columns.pkl")
    return out_dir


def main(
    data_path: str = "data/creditcard.csv.gz",
    out_dir: str = "models/retrained",
) -> None:
    """Train on the dataset and save into a separate folder.

    The default output folder is `models/retrained`, so the model used by the
    deployed app is never overwritten by accident.
    """
    X, y = split_xy(load_data(data_path))
    result = train_model(X, y)
    save_artifacts(result, out_dir)
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()