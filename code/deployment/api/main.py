"""
Stage 3: Model API
Serves predictions from the trained model over HTTP.

Input:  models/model.pkl (mounted as a volume)
Output: running prediction API
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# The path is configurable so the same code works locally and in the container
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/model.pkl"))

app = FastAPI(title="ATP Match Prediction API")


class MatchInput(BaseModel):
    """Input schema - mirrors the features used at training time."""

    rank_1: int = Field(..., ge=1, description="ATP rank of player 1")
    rank_2: int = Field(..., ge=1, description="ATP rank of player 2")
    pts_1: int = Field(..., ge=0, description="ATP points of player 1")
    pts_2: int = Field(..., ge=0, description="ATP points of player 2")
    series: str = Field(..., description="Tournament series, e.g. Grand Slam")
    court: str = Field(..., description="Indoor or Outdoor")
    surface: str = Field(..., description="Hard, Clay or Grass")
    round_name: str = Field(..., description="Round, e.g. Quarterfinals")
    best_of: int = Field(..., description="3 or 5 sets")


def load_model():
    """Loads the model on every prediction so the container always serves
    the latest version produced by the pipeline, without a rebuild."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found. Run the pipeline first.")
    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(match: MatchInput):
    """Returns the predicted winner and the model's confidence."""
    model = load_model()

    # Column names must match exactly what the training pipeline expects
    features = pd.DataFrame([{
        "Rank_1": match.rank_1,
        "Rank_2": match.rank_2,
        "Pts_1": match.pts_1,
        "Pts_2": match.pts_2,
        "Best of": match.best_of,
        "rank_diff": match.rank_1 - match.rank_2,
        "pts_diff": match.pts_1 - match.pts_2,
        "Series": match.series,
        "Court": match.court,
        "Surface": match.surface,
        "Round": match.round_name,
    }])

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][prediction])

    return {
        "prediction": prediction,
        "winner": "Player 1" if prediction == 1 else "Player 2",
        "confidence": round(probability, 4),
    }