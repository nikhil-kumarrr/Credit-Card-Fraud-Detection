"""Loading and validating the credit card transactions dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
TARGET = "Class"
REQUIRED_COLUMNS = ["Time", "Amount", *V_COLUMNS, TARGET]


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if the dataframe is missing any required column."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the dataset (.csv or .csv.gz) and validate its columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return validate_columns(pd.read_csv(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a validated dataframe into features X and integer labels y."""
    validate_columns(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y
