"""Loading the YouTube trending data and turning it into one row per video.

The prediction is made at the end of a video's *second* day on the trending list,
so every feature below only uses information that is available at that moment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Numeric features fed to the model (text features are added separately).
NUMERIC_FEATURES = [
    # size of the video on day 1 (log scale)
    "views_l", "likes_l", "dislikes_l", "comment_count_l",
    # growth from day 1 to day 2 (log ratio, 0 if it dropped off the list)
    "views_g", "likes_g", "dislikes_g", "comment_count_g", "on_day2",
    # rank against the other videos trending that day (0-1)
    "views_pct", "likes_pct", "comment_count_pct", "views_pct2",
    # audience reaction
    "like_rate", "comment_rate", "dislike_ratio", "eng",
    # speed since upload
    "age_h", "vph", "lph", "cph",
    # publishing details
    "pub_hour", "pub_wd", "category_id", "comments_disabled", "ratings_disabled",
    # packaging of the video
    "title_len", "caps", "ntags", "desc_len",
    # context known at prediction time
    "chan_prev", "turnover7",
]

STATS = ["views", "likes", "dislikes", "comment_count"]


def load_trending(path: str | Path) -> pd.DataFrame:
    """Read a Kaggle-style ``USvideos.csv`` and return one clean row per video per day."""
    df = pd.read_csv(path)
    df["tdate"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m")
    df["ptime"] = pd.to_datetime(df["publish_time"], utc=True).dt.tz_localize(None)
    df["tags"] = df["tags"].fillna("[none]")
    df["description"] = df["description"].fillna("")
    for col in ("comments_disabled", "ratings_disabled", "video_error_or_removed"):
        if col in df:
            df[col] = df[col].astype(str).str.lower().eq("true")
    df = df.sort_values(["video_id", "tdate"]).drop_duplicates(["video_id", "tdate"])
    # where each video ranks among that day's trending list
    for col in ("views", "likes", "comment_count"):
        df[col + "_pct"] = df.groupby("tdate")[col].rank(pct=True)
    return df.reset_index(drop=True)


def text_of(frame: pd.DataFrame) -> pd.Series:
    """Title + tags, lower-cased: the text the NLP part of the model reads."""
    tags = frame["tags"].astype(str).str.replace("[none]", "", regex=False).str.replace("|", " ", regex=False)
    return (frame["title"].astype(str) + " " + tags).str.replace('"', " ", regex=False).str.lower()


def caps_ratio(title: str) -> float:
    letters = [ch for ch in str(title) if ch.isalpha()]
    return sum(ch.isupper() for ch in letters) / max(1, len(letters))


def add_video_features(t: pd.DataFrame, s2: pd.DataFrame | None) -> pd.DataFrame:
    """Compute every model feature from day-1 rows ``t`` and matching day-2 rows ``s2``."""
    t = t.copy()
    if s2 is None:
        s2 = pd.DataFrame(index=t.index, columns=["tdate", *STATS, "views_pct"])
    gap = (pd.to_datetime(s2["tdate"]) - t["tdate"]).dt.days
    t["on_day2"] = ((gap >= 1) & (gap <= 2)).astype(int)
    for c in STATS:
        t[c + "_l"] = np.log1p(t[c].astype(float))
        day2 = np.log1p(pd.to_numeric(s2[c], errors="coerce").astype(float))
        t[c + "_g"] = np.where(t["on_day2"] == 1, day2 - t[c + "_l"], 0.0)
    t["views_pct2"] = np.where(t["on_day2"] == 1, pd.to_numeric(s2["views_pct"], errors="coerce"), 0.0)
    views = t["views"].astype(float)
    t["like_rate"] = t["likes"] / (views + 1)
    t["comment_rate"] = t["comment_count"] / (views + 1)
    t["dislike_ratio"] = t["dislikes"] / (t["likes"] + t["dislikes"] + 1)
    t["eng"] = (t["likes"] + t["comment_count"]) / (views + 1)
    age = ((t["tdate"] - t["ptime"]).dt.total_seconds() / 3600).clip(lower=1)
    t["age_h"] = age
    t["vph"] = np.log1p(views / age)
    t["lph"] = np.log1p(t["likes"] / age)
    t["cph"] = np.log1p(t["comment_count"] / age)
    t["pub_hour"] = t["ptime"].dt.hour
    t["pub_wd"] = t["ptime"].dt.weekday
    t["title_len"] = t["title"].astype(str).str.len()
    t["caps"] = t["title"].map(caps_ratio)
    t["ntags"] = t["tags"].map(lambda s: 0 if s == "[none]" else len(str(s).split("|")))
    t["desc_len"] = t["description"].astype(str).str.len()
    t["comments_disabled"] = t["comments_disabled"].astype(int)
    t["ratings_disabled"] = t["ratings_disabled"].astype(int)
    return t


@dataclass
class Dataset:
    table: pd.DataFrame          # one row per video, features + label
    turnover: pd.Series          # new entries per day (7-day average, known the day before)


def build_dataset(df: pd.DataFrame, edge_days: int = 7) -> Dataset:
    """One row per video with features at day 2 and the label.

    Label (``y``): 1 if the video stayed on the trending list longer than the median
    video that *started trending in the same week*. Comparing with same-week peers keeps
    the target fair even though trending behaviour changed during the period.
    Videos at the very start/end of the data are dropped, because their true trending
    span cannot be observed.
    """
    g = df.groupby("video_id")
    first = g.head(1).set_index("video_id")
    second = g.nth(1).set_index("video_id").reindex(first.index)
    t = add_video_features(first, second)

    t["span"] = (g["tdate"].max() - g["tdate"].min()).dt.days + 1
    t["week"] = t["tdate"].dt.to_period("W")
    t["y"] = (t["span"] > t.groupby("week")["span"].transform("median")).astype(int)

    t = t.sort_values("tdate")
    # how many videos from the same channel started trending on EARLIER days (no look-ahead)
    t["chan_prev"] = (t.groupby("channel_title")["tdate"].rank(method="min") - 1).astype(int)
    # how fast the trending list was turning over in the week before
    new_per_day = t.groupby("tdate").size().reindex(
        pd.date_range(t["tdate"].min(), t["tdate"].max()), fill_value=0)
    turnover = new_per_day.rolling(7, min_periods=1).mean().shift(1).bfill()
    t["turnover7"] = t["tdate"].map(turnover)

    lo, hi = df["tdate"].min(), df["tdate"].max()
    keep = (t["tdate"] > lo + pd.Timedelta(days=edge_days)) & (t["tdate"] <= hi - pd.Timedelta(days=edge_days + 1))
    return Dataset(table=t[keep].copy(), turnover=turnover)
