"""Model training and evaluation (random undersampling + logistic regression)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fraud_detection.data import load_data, split_xy
from fraud_detection.features import FEATURE_COLUMNS, build_features, fit_scalers


def undersample(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep every fraud row and an equal number of random legit rows."""
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].sample(n=len(fraud_idx), random_state=random_state).index
    keep = fraud_idx.union(legit_idx)
    return X.loc[keep], y.loc[keep]


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Undersample, train logistic regression and evaluate it two ways.

    `roc_auc`, `recall`, `precision`, `f1` are measured on the balanced test
    split. The `holdout_*` metrics use every row that was not used for training,
    which keeps the real, heavily imbalanced fraud rate.
    """
    X_bal, y_bal = undersample(X, y, random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=test_size, random_state=random_state, stratify=y_bal
    )
    scaler_amount, scaler_time = fit_scalers(X_train)
    train_features = build_features(X_train, scaler_amount, scaler_time)
    test_features = build_features(X_test, scaler_amount, scaler_time)

    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(train_features, y_train)

    proba = model.predict_proba(test_features)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }

    holdout = ~X.index.isin(X_train.index)
    X_hold, y_hold = X[holdout], y[holdout]
    hold_features = build_features(X_hold, scaler_amount, scaler_time)
    hold_proba = model.predict_proba(hold_features)[:, 1]
    hold_pred = (hold_proba >= 0.5).astype(int)
    metrics.update(
        {
            "holdout_roc_auc": float(roc_auc_score(y_hold, hold_proba)),
            "holdout_pr_auc": float(average_precision_score(y_hold, hold_proba)),
            "holdout_recall": float(recall_score(y_hold, hold_pred, zero_division=0)),
            "holdout_precision": float(
                precision_score(y_hold, hold_pred, zero_division=0)
            ),
        }
    )
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
