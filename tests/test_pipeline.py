import numpy as np
import pandas as pd

from trendpredictor.data import NUMERIC_FEATURES, build_dataset, caps_ratio, load_trending, text_of
from trendpredictor.model import TrendModel, scores
from trendpredictor.script import make_script, template_script
from trendpredictor.topics import fit_topics, rising_topics
from trendpredictor.video import synth_beat


def test_load_parses_dates_and_dedupes(toy_csv):
    df = load_trending(toy_csv)
    assert df["tdate"].dtype.kind == "M"
    assert not df.duplicated(["video_id", "tdate"]).any()
    assert df["views_pct"].between(0, 1).all()


def test_dataset_has_all_features_and_binary_label(toy_csv):
    t = build_dataset(load_trending(toy_csv), edge_days=2).table
    assert len(t) > 0
    assert set(NUMERIC_FEATURES) <= set(t.columns)
    assert set(t["y"].unique()) <= {0, 1}
    assert not t[NUMERIC_FEATURES].isna().any().any()


def test_no_lookahead_in_channel_history(toy_csv):
    df = load_trending(toy_csv)
    t = build_dataset(df, edge_days=0).table
    first = df.groupby("video_id").head(1)
    for vid, row in t.iterrows():
        same = first[first["channel_title"] == row["channel_title"]]
        # only videos that started trending on an EARLIER day may be counted
        assert row["chan_prev"] == (same["tdate"] < row["tdate"]).sum()


def test_model_trains_and_predicts_probabilities(toy_csv):
    t = build_dataset(load_trending(toy_csv), edge_days=0).table
    m = TrendModel(n_text_components=3).fit(t)
    p = m.predict_proba(t)
    assert p.shape == (len(t),) and ((p >= 0) & (p <= 1)).all()


def test_scores_baseline_and_coverage():
    y = np.array([0, 0, 1, 1])
    r = scores(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert r["accuracy"] == 1.0 and r["majority_baseline"] == 0.5
    assert r["confident_coverage"] == 0.5


def test_text_and_caps_helpers():
    df = pd.DataFrame({"title": ["HELLO world"], "tags": ['"a"|"b"']})
    assert text_of(df).iloc[0] == "hello world  a   b "
    assert abs(caps_ratio("ABcd") - 0.5) < 1e-9


def test_topics_and_momentum(toy_csv):
    first = load_trending(toy_csv).groupby("video_id").head(1)
    words, topic = fit_topics(first, n_topics=3)
    assert len(words) == 3 and len(topic) == len(first)
    top = rising_topics(first, topic, words)
    assert {"keywords", "momentum"} <= set(top.columns)


def test_script_template_works_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = make_script("netflix", ["netflix", "movies"], 2.0)
    assert s.source == "template" and len(s.scenes()) == 5
    assert "2.0x" in template_script("x", ["x"], 2.0).points[0]


def test_beat_is_stereo_and_normalised():
    a = synth_beat(2.0)
    assert a.shape == (88200, 2) and np.abs(a).max() <= 1.0
