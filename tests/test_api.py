import pytest
from fastapi.testclient import TestClient

import trendpredictor.api as api
from trendpredictor.data import build_dataset, load_trending
from trendpredictor.model import TrendModel


@pytest.fixture
def client(toy_csv, tmp_path, monkeypatch):
    t = build_dataset(load_trending(toy_csv), edge_days=0).table
    path = tmp_path / "m.joblib"
    TrendModel(n_text_components=3).fit(t).save(path)
    monkeypatch.setattr(api, "MODEL_PATH", path)
    monkeypatch.setattr(api, "_model", None)
    return TestClient(api.app)


VIDEO = {
    "title": "NBA Finals Game 4 Highlights", "tags": ["nba", "highlights"],
    "publish_time": "2018-06-08T03:00:00Z", "first_trending_date": "2018-06-09",
    "day1": {"views": 850000, "likes": 9000, "dislikes": 300, "comment_count": 2100, "views_pct": 0.8},
    "day2": {"views": 1900000, "likes": 16000, "dislikes": 500, "comment_count": 3500, "views_pct": 0.9},
}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probability(client):
    r = client.post("/predict", json=VIDEO)
    assert r.status_code == 200
    assert 0 <= r.json()["probability_long_trend"] <= 1


def test_predict_without_day2(client):
    body = {k: v for k, v in VIDEO.items() if k != "day2"}
    assert client.post("/predict", json=body).status_code == 200


def test_script_endpoint(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/script", json={"topic": "nba", "keywords": ["nba", "finals"], "momentum": 1.5})
    assert r.json()["source"] == "template"
