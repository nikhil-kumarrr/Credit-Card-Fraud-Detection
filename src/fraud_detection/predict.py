"""Inference: load saved artifacts and score transactions."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fraud_detection.features import build_features


class FraudPredictor:
    """Wraps the saved model, scalers and feature columns."""

    def __init__(self, model, scaler_amount, scaler_time, feature_columns):
        self.model = model
        self.scaler_amount = scaler_amount
        self.scaler_time = scaler_time
        self.feature_columns = list(feature_columns)

    @classmethod
    def from_dir(cls, models_dir: str | Path = "models") -> FraudPredictor:
        models_dir = Path(models_dir)
        return cls(
            model=joblib.load(models_dir / "fraud_detection_model.pkl"),
            scaler_amount=joblib.load(models_dir / "scaler_amount.pkl"),
            scaler_time=joblib.load(models_dir / "scaler_time.pkl"),
            feature_columns=joblib.load(models_dir / "feature_columns.pkl"),
        )

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_features(
            X, self.scaler_amount, self.scaler_time, self.feature_columns
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return the fraud probability (0 to 1) for each row."""
        proba = self.model.predict_proba(self._features(X))[:, 1]
        return pd.Series(proba, index=X.index, name="fraud_probability")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Return 1 for fraud and 0 for legit using the given threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(X) >= threshold).astype(int).rename("is_fraud")
