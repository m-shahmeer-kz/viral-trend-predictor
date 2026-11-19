import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_csv(tmp_path):
    """A tiny fake trending file: 60 videos over ~40 days, same columns as the real one."""
    rng = np.random.default_rng(0)
    rows = []
    words = ["music", "nba", "makeup", "netflix", "trailer", "cooking", "news", "game"]
    start = pd.Timestamp("2018-01-01")
    for v in range(60):
        day0 = start + pd.Timedelta(days=int(rng.integers(0, 30)))
        span = int(rng.integers(1, 9))
        views = int(rng.integers(10_000, 2_000_000))
        topic = words[v % len(words)]
        for d in range(span):
            date = day0 + pd.Timedelta(days=d)
            grow = 1 + 0.4 * d
            rows.append({
                "video_id": f"vid{v:03d}", "trending_date": date.strftime("%y.%d.%m"),
                "title": f"{topic.title()} video number {v} {topic}", "channel_title": f"chan{v % 9}",
                "category_id": 10 + v % 5, "publish_time": (day0 - pd.Timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "tags": f'"{topic}"|"{words[(v + 1) % 8]}"|"viral"', "views": int(views * grow),
                "likes": int(views * grow * 0.04), "dislikes": int(views * grow * 0.002),
                "comment_count": int(views * grow * 0.005), "thumbnail_link": "", "comments_disabled": "False",
                "ratings_disabled": "False", "video_error_or_removed": "False", "description": "desc " * (v % 7),
            })
    path = tmp_path / "toy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
