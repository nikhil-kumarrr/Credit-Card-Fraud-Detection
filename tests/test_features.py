import numpy as np
import pytest

from fraud_detection.data import split_xy
from fraud_detection.features import (
    FEATURE_COLUMNS,
    align_columns,
    build_features,
    fit_scalers,
    scale_features,
)


def test_scale_features_zero_mean_unit_std(sample_df):
    X, _ = split_xy(sample_df)
    scaler_amount, scaler_time = fit_scalers(X)
    scaled = scale_features(X, scaler_amount, scaler_time)
    assert abs(scaled["scaled_amount"].mean()) < 1e-9
    assert scaled["scaled_amount"].std(ddof=0) == pytest.approx(1.0)


def test_scale_features_does_not_modify_input(sample_df):
    X, _ = split_xy(sample_df)
    original = X["Amount"].copy()
    scaler_amount, scaler_time = fit_scalers(X)
    scale_features(X, scaler_amount, scaler_time)
    assert np.allclose(X["Amount"], original)


def test_align_columns_order(sample_df):
    X, _ = split_xy(sample_df)
    aligned = align_columns(X, ["V2", "V1"])
    assert list(aligned.columns) == ["V2", "V1"]


def test_align_columns_missing_raises(sample_df):
    X, _ = split_xy(sample_df)
    with pytest.raises(KeyError):
        align_columns(X, ["V1", "does_not_exist"])


def test_build_features_matches_feature_columns(sample_df):
    X, _ = split_xy(sample_df)
    scaler_amount, scaler_time = fit_scalers(X)
    features = build_features(X, scaler_amount, scaler_time)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.isna().sum().sum() == 0