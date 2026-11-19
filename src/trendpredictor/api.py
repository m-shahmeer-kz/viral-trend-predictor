"""A small FastAPI service.

Run:  uvicorn trendpredictor.api:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import add_video_features
from .model import TrendModel
from .script import make_script

MODEL_PATH = Path(os.getenv("TREND_MODEL", "models/trend_model.joblib"))

app = FastAPI(title="Viral Trend Predictor", version="1.0.0",
              description="Will a trending YouTube video stay on the list longer than its peers?")
_model: TrendModel | None = None


def get_model() -> TrendModel:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"No trained model at {MODEL_PATH}. Run `trendpredictor train` first.")
        _model = TrendModel.load(MODEL_PATH)
    return _model


class Snapshot(BaseModel):
    views: int = Field(ge=0)
    likes: int = Field(ge=0)
    dislikes: int = Field(0, ge=0)
    comment_count: int = Field(0, ge=0)
    views_pct: float = Field(0.5, ge=0, le=1, description="Rank among that day's trending videos (0-1)")


class VideoIn(BaseModel):
    title: str
    tags: list[str] = []
    description: str = ""
    channel_title: str = "unknown"
    category_id: int = 24
    publish_time: datetime
    first_trending_date: datetime
    day1: Snapshot
    day2: Snapshot | None = None
    comments_disabled: bool = False
    ratings_disabled: bool = False
    channel_previous_trending: int = 0
    list_turnover_7d: float = Field(28.0, description="Avg new videos entering the list per day, last week")


class Prediction(BaseModel):
    probability_long_trend: float
    verdict: str


class ScriptIn(BaseModel):
    topic: str
    keywords: list[str] = []
    momentum: float | None = None


def to_frame(v: VideoIn) -> pd.DataFrame:
    tdate = pd.Timestamp(v.first_trending_date).tz_localize(None).normalize()
    row = {
        "title": v.title, "tags": "|".join(v.tags) if v.tags else "[none]", "description": v.description,
        "channel_title": v.channel_title, "category_id": v.category_id,
        "ptime": pd.Timestamp(v.publish_time).tz_localize(None), "tdate": tdate,
        "comments_disabled": v.comments_disabled, "ratings_disabled": v.ratings_disabled,
        **v.day1.model_dump(),
    }
    t = pd.DataFrame([row])
    t["likes_pct"] = t["views_pct"]
    t["comment_count_pct"] = t["views_pct"]
    s2 = None
    if v.day2:
        s2 = pd.DataFrame([{**v.day2.model_dump(), "tdate": tdate + pd.Timedelta(days=1)}])
    t = add_video_features(t, s2)
    t["chan_prev"] = v.channel_previous_trending
    t["turnover7"] = v.list_turnover_7d
    return t


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(video: VideoIn) -> Prediction:
    p = float(get_model().predict_proba(to_frame(video))[0])
    verdict = "likely long trend" if p >= 0.5 else "likely short trend"
    return Prediction(probability_long_trend=round(p, 3), verdict=verdict)


@app.post("/script")
def script(req: ScriptIn) -> dict:
    return make_script(req.topic, req.keywords, req.momentum).to_dict()
