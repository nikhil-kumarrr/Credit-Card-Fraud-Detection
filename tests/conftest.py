import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import V_COLUMNS


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic dataset with the same columns as the real one."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df["Time"] = rng.uniform(0, 172_000, size=n)
    df["Amount"] = rng.exponential(80, size=n)
    labels = np.zeros(n, dtype=int)
    labels[:40] = 1
    df["V1"] = df["V1"] + labels * 3.0  # make fraud learnable
    df["Class"] = labels
    return df.sample(frac=1, random_state=1).reset_index(drop=True)
