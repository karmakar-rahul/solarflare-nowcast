"""
app.py
------
FastAPI inference server for the Solar Flare Early Warning model.

POST /predict  →  accepts 360 minutes of 5-channel GOES data, returns probability.

Usage:
    uvicorn app:app --reload --port 8000

    # Then POST to:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"xrs_short": [...360 values...], "xrs_long": [...360 values...]}'
"""

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List

from src.model import build_model
from src.predictor import FlarePredictor

CHECKPOINT_PATH = "checkpoints/best_model.pt"
SEQ_LEN = 360
LOG_EPS = 1e-9
ROLLING_MAX_W = 30

app = FastAPI(
    title="Solar Flare Early Warning API",
    description=(
        "Binary solar flare forecasting using GOES X-ray time series. "
        "Predicts probability of M/X class flare within the next 60 minutes."
    ),
    version="2.0",
)

# Load model at startup 

predictor: FlarePredictor = None

@app.on_event("startup")
def load_model():
    global predictor
    predictor = FlarePredictor.from_checkpoint(CHECKPOINT_PATH, device="cpu")
    print(f"[app] Model loaded. Threshold: {predictor.threshold:.2f}")


# Request / Response schemas

class PredictRequest(BaseModel):
    """
    Send raw flux values (W/m²) — the API handles log-scaling and feature engineering.
    Either send raw flux OR pre-computed log-scaled values; set `already_log_scaled=true`
    if your values are already in log10 space.
    """
    xrs_short: List[float]          # 0.05–0.4 nm channel, 360 values
    xrs_long:  List[float]          # 0.1–0.8 nm channel, 360 values
    already_log_scaled: bool = False

    @field_validator("xrs_short", "xrs_long")
    @classmethod
    def check_length(cls, v):
        if len(v) != SEQ_LEN:
            raise ValueError(f"Each channel must have exactly {SEQ_LEN} values, got {len(v)}")
        return v


class PredictResponse(BaseModel):
    probability:   float
    flare_warning: bool
    risk_level:    str
    threshold:     float


# Feature engineering (matches training pipeline) 

def _build_features(short: np.ndarray, long_: np.ndarray, already_log: bool) -> np.ndarray:
    if already_log:
        xrs_short = short
        xrs_long  = long_
    else:
        xrs_short = np.log10(np.clip(short, LOG_EPS, None))
        xrs_long  = np.log10(np.clip(long_,  LOG_EPS, None))

    xrs_ratio   = xrs_long - xrs_short
    deriv_short = np.gradient(xrs_short)
    rolling_max = (
        pd.Series(xrs_long)
        .rolling(window=ROLLING_MAX_W, min_periods=1)
        .max()
        .values
    )
    return np.stack([xrs_short, xrs_long, xrs_ratio, deriv_short, rolling_max], axis=1).astype(np.float32)


# Endpoints 

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    short = np.array(req.xrs_short, dtype=np.float32)
    long_ = np.array(req.xrs_long,  dtype=np.float32)

    x = _build_features(short, long_, req.already_log_scaled)  # (360, 5)

    prob, warning    = predictor.predict(x)
    level, _         = predictor.get_risk_level(prob)

    return PredictResponse(
        probability=round(prob, 4),
        flare_warning=warning,
        risk_level=level,
        threshold=predictor.threshold,
    )


@app.get("/live")
def predict_live():
    """Fetch latest GOES data from NOAA SWPC and run prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        from src.goes_fetcher import fetch_latest_goes_array
        x = fetch_latest_goes_array()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch GOES data: {e}")

    prob, warning = predictor.predict(x)
    level, _      = predictor.get_risk_level(prob)

    return PredictResponse(
        probability=round(prob, 4),
        flare_warning=warning,
        risk_level=level,
        threshold=predictor.threshold,
    )