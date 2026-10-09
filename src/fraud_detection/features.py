"""Feature engineering: scaling Amount and Time and aligning model columns."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler

from fraud_detection.data import V_COLUMNS

FEATURE_COLUMNS = [*V_COLUMNS, "scaled_amount", "scaled_time"]


def fit_scalers(X: pd.DataFrame) -> tuple[StandardScaler, StandardScaler]:
    """Fit one StandardScaler for Amount and one for Time."""
    scaler_amount = StandardScaler().fit(X[["Amount"]])
    scaler_time = StandardScaler().fit(X[["Time"]])
    return scaler_amount, scaler_time


def scale_features(
    X: pd.DataFrame, scaler_amount: StandardScaler, scaler_time: StandardScaler
) -> pd.DataFrame:
    """Return a copy of X with scaled Amount and Time.

    The scaled values are stored under `scaled_amount` / `scaled_time` and also
    written back to `Amount` / `Time`, so models saved with either column
    naming convention can be aligned later.
    """
    out = X.copy()
    scaled_amount = scaler_amount.transform(out[["Amount"]]).ravel()
    scaled_time = scaler_time.transform(out[["Time"]]).ravel()
    out["scaled_amount"] = scaled_amount
    out["scaled_time"] = scaled_time
    out["Amount"] = scaled_amount
    out["Time"] = scaled_time
    return out


def align_columns(X: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Select and order columns exactly as the model expects."""
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return X[list(feature_columns)]


def build_features(
    X: pd.DataFrame,
    scaler_amount: StandardScaler,
    scaler_time: StandardScaler,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Scale Amount/Time and return the model-ready feature matrix."""
    scaled = scale_features(X, scaler_amount, scaler_time)
    return align_columns(scaled, feature_columns)