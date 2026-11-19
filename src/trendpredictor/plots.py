"""Charts saved into reports/figures (black / white / grey style)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, GREY, LIGHT = "#111111", "#8a8a8f", "#d9d9dc"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forward_chaining(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, df["accuracy"], 0.38, color=INK, label="Model accuracy")
    ax.bar(x + 0.2, df["majority_baseline"], 0.38, color=LIGHT, label="Always-guess-majority")
    for i, a in enumerate(df["accuracy"]):
        ax.text(i - 0.2, a + 0.01, f"{a:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, df["month"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Each month predicted using only earlier months")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _save(fig, path)


def confidence(out: pd.DataFrame, path: Path) -> None:
    """Accuracy when the model is more and more sure of itself."""
    p, y = out["p"].to_numpy(), out["y"].to_numpy()
    sure = np.abs(p - 0.5)
    th = np.linspace(0, 0.4, 9)
    acc, cov = [], []
    for t in th:
        m = sure >= t
        acc.append(((p[m] >= 0.5) == y[m]).mean())
        cov.append(m.mean())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(cov, acc, "-o", color=INK)
    for c, a, t in zip(cov, acc, th):
        ax.annotate(f"{a:.0%}", (c, a), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of test videos the model makes a call on (most confident first)")
    ax.set_ylabel("Accuracy on those videos")
    ax.set_title("Being pickier makes the model more accurate")
    _save(fig, path)


PRETTY = {
    "chan_prev": "Channel's earlier trending videos", "title + tags text": "Title + tags (NLP)",
    "turnover7": "How fast the list was changing", "pub_wd": "Weekday published", "age_h": "Hours since upload",
    "views_pct2": "Day-2 views rank", "likes_g": "Day-2 like growth", "like_rate": "Likes per view",
    "comment_count_g": "Day-2 comment growth", "desc_len": "Description length", "pub_hour": "Hour published",
    "likes_l": "Likes (day 1)", "ntags": "Number of tags", "views_pct": "Day-1 views rank",
    "likes_pct": "Day-1 likes rank", "views_g": "Day-2 view growth", "vph": "Views per hour",
    "views_l": "Views (day 1)", "category_id": "Category", "eng": "Engagement rate", "caps": "CAPS in title",
}


def importance(names: list[str], values: np.ndarray, path: Path, top: int = 15) -> None:
    order = np.argsort(values)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    labels = [PRETTY.get(n, n) for n in np.array(names)[order]]
    ax.barh(labels, values[order], color=INK)
    ax.set_xlabel("Drop in AUC when the feature is shuffled")
    ax.set_title("What the model relies on most")
    _save(fig, path)


def span_hist(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    spans = table["span"].clip(upper=30)
    ax.hist(spans, bins=np.arange(0.5, 31.5, 1), color=INK)
    ax.set_xlabel("Days on the trending list (30 = 30 or more)")
    ax.set_ylabel("Videos")
    ax.set_title("Most videos trend for under a week")
    _save(fig, path)


def topics_bar(top: pd.DataFrame, path: Path) -> None:
    t = top.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [INK if m > 1 else LIGHT for m in t["momentum"]]
    labels = t["keywords"].str.split(", ").str[:4].str.join(", ")
    ax.barh(labels, t["momentum"], color=colors)
    ax.axvline(1, color=GREY, lw=1, ls="--")
    ax.set_xlabel("Momentum (share last 2 weeks ÷ share 2 weeks before)")
    ax.set_title("Rising topics at the end of the data")
    _save(fig, path)
