"""FastAPI service that scores credit card transactions.

Run locally:
    uvicorn fraud_detection.api:app --app-dir src --reload

Set FRAUD_MODELS_DIR to load the model from a different folder (default: models).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel, Field, create_model

from fraud_detection import __version__
from fraud_detection.data import V_COLUMNS
from fraud_detection.predict import FraudPredictor

# One request row: Time, Amount and the 28 anonymised PCA features V1..V28.
Transaction = create_model(
    "Transaction",
    Time=(float, Field(..., description="Seconds since the first transaction")),
    Amount=(float, Field(..., ge=0, description="Transaction amount, never negative")),
    **{col: (float, Field(...)) for col in V_COLUMNS},
)


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class PredictResponse(BaseModel):
    threshold: float
    predictions: list[Prediction]


class Health(BaseModel):
    status: str
    version: str


def risk_level(probability: float) -> str:
    """Same bands as the dashboard: low below 30%, medium below 70%, else high."""
    if probability < 0.3:
        return "low"
    if probability < 0.7:
        return "medium"
    return "high"


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """Load the model once and reuse it for every request."""
    return FraudPredictor.from_dir(os.environ.get("FRAUD_MODELS_DIR", "models"))


app = FastAPI(
    title="Fraud Shield API",
    version=__version__,
    description="Score credit card transactions with the trained fraud model.",
)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", version=__version__)


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    predictor: Annotated[FraudPredictor, Depends(get_predictor)],
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
) -> PredictResponse:
    frame = pd.DataFrame([t.model_dump() for t in body.transactions])
    probabilities = predictor.predict_proba(frame)
    predictions = [
        Prediction(
            fraud_probability=float(p),
            is_fraud=bool(p >= threshold),
            risk_level=risk_level(float(p)),
        )
        for p in probabilities
    ]
    return PredictResponse(threshold=threshold, predictions=predictions)
