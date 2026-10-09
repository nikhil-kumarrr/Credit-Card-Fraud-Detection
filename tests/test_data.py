import pandas as pd
import pytest

from fraud_detection.data import load_data, split_xy, validate_columns


def test_validate_columns_ok(sample_df):
    assert validate_columns(sample_df) is sample_df


def test_validate_columns_missing_raises(sample_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(sample_df.drop(columns=["Amount"]))


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_load_data_reads_gzip(sample_df, tmp_path):
    path = tmp_path / "data.csv.gz"
    sample_df.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded.shape == sample_df.shape


def test_split_xy_shapes_and_types(sample_df):
    X, y = split_xy(sample_df)
    assert "Class" not in X.columns
    assert len(X) == len(y) == len(sample_df)
    assert y.dtype == int
    assert isinstance(X, pd.DataFrame)
