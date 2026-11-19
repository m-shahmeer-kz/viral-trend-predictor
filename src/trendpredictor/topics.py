"""Find topics in trending titles/tags and measure which ones are rising."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import text_of

EXTRA_STOP = {"video", "official", "new", "2017", "2018", "ft", "feat", "hd", "vs", "trailer", "episode"}


def fit_topics(videos: pd.DataFrame, n_topics: int = 12, top_words: int = 6, seed: int = 0):
    """Learn ``n_topics`` topics with NMF. Returns (topic_words, topic_of_each_video)."""
    vec = TfidfVectorizer(min_df=5, max_df=0.3, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z]{2,}\b")
    X = vec.fit_transform(text_of(videos))
    vocab = np.array(vec.get_feature_names_out())
    nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    words = []
    for comp in nmf.components_:
        ranked = [w for w in vocab[comp.argsort()[::-1]] if w not in EXTRA_STOP]
        words.append(ranked[:top_words])
    topic = W.argmax(axis=1)
    topic[W.max(axis=1) == 0] = -1
    return words, topic


def rising_topics(videos: pd.DataFrame, topic: np.ndarray, words: list[list[str]],
                  as_of: pd.Timestamp | None = None, window_days: int = 14) -> pd.DataFrame:
    """Compare each topic's share of new trending videos in the last window vs the window before.

    ``momentum`` > 1 means the topic is growing.
    """
    v = videos.assign(topic=topic)
    v = v[v["topic"] >= 0]
    as_of = as_of or v["tdate"].max()
    w = pd.Timedelta(days=window_days)
    recent = v[(v["tdate"] > as_of - w) & (v["tdate"] <= as_of)]
    before = v[(v["tdate"] > as_of - 2 * w) & (v["tdate"] <= as_of - w)]
    share_r = recent["topic"].value_counts(normalize=True)
    share_b = before["topic"].value_counts(normalize=True)
    rows = []
    for k, ws in enumerate(words):
        r, b = share_r.get(k, 0.0), share_b.get(k, 0.0)
        rows.append({"topic": k, "keywords": ", ".join(ws), "share_recent": round(r, 4),
                     "share_before": round(b, 4), "momentum": round((r + 0.005) / (b + 0.005), 2),
                     "videos_recent": int((recent["topic"] == k).sum())})
    return pd.DataFrame(rows).sort_values("momentum", ascending=False).reset_index(drop=True)
