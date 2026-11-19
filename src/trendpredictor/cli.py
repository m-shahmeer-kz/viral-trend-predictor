"""Command line entry point.

    trendpredictor train   --data data/USvideos.csv   # evaluate + save model, metrics, charts
    trendpredictor topics  --data data/USvideos.csv   # print rising topics
    trendpredictor video   --data data/USvideos.csv   # script + vertical MP4 for the top rising topic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .data import NUMERIC_FEATURES, build_dataset, load_trending
from .model import TrendModel, evaluate_forward_chaining, evaluate_time_split, save_metrics
from .script import make_script
from .topics import fit_topics, rising_topics

FIG = Path("reports/figures")


def _permutation_auc(model: TrendModel, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """How much AUC drops when one numeric feature (or all text features) is shuffled."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y = test["y"].to_numpy()
    base = roc_auc_score(y, model.predict_proba(test))
    drops = []
    for col in NUMERIC_FEATURES + ["__text__"]:
        shuffled = test.copy()
        cols = ["title", "tags"] if col == "__text__" else [col]
        idx = rng.permutation(len(test))
        for c in cols:
            shuffled[c] = test[c].to_numpy()[idx]
        drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)))
    return np.array(drops)


def cmd_train(args) -> None:
    df = load_trending(args.data)
    ds = build_dataset(df)
    table = ds.table
    print(f"Videos used: {len(table):,}  |  positives: {table['y'].mean():.1%}")

    res, model, out = evaluate_time_split(table, args.test_from)
    print("Time split:", json.dumps(res))
    months = sorted(table["tdate"].dt.strftime("%Y-%m").unique())[2:]
    fc = evaluate_forward_chaining(table, months)
    for r in fc:
        print(f"  {r['month']}: acc {r['accuracy']:.3f}  auc {r['auc']:.3f}  (n={r['n']})")

    test = table[table["tdate"] >= args.test_from]
    drops = _permutation_auc(model, test)
    names = [*NUMERIC_FEATURES, "title + tags text"]

    plots.forward_chaining(fc, FIG / "forward_chaining.png")
    plots.confidence(out, FIG / "confidence.png")
    plots.importance(names, drops, FIG / "importance.png")
    plots.span_hist(table, FIG / "trend_length.png")

    final = TrendModel().fit(table)  # the shipped model uses all labelled data
    final.save(args.model)
    out.sort_values("p", ascending=False).head(15).to_csv("reports/top_predictions.csv", index=False)
    save_metrics("reports/metrics.json", {
        "dataset": {"rows": int(len(df)), "videos_used": int(len(table)),
                    "first_day": str(df["tdate"].min().date()), "last_day": str(df["tdate"].max().date())},
        "time_split": res, "forward_chaining": fc,
        "top_features": sorted(zip(names, map(float, drops)), key=lambda x: -x[1])[:10],
    })
    print(f"Saved model -> {args.model}, metrics -> reports/metrics.json, charts -> {FIG}/")


def _rising(df: pd.DataFrame, n_topics: int):
    first = df.groupby("video_id").head(1)
    words, topic = fit_topics(first, n_topics=n_topics)
    return rising_topics(first, topic, words)


def cmd_topics(args) -> None:
    top = _rising(load_trending(args.data), args.n_topics)
    print(top.to_string(index=False))
    Path("reports").mkdir(exist_ok=True)
    top.to_csv("reports/rising_topics.csv", index=False)
    plots.topics_bar(top, FIG / "rising_topics.png")


def cmd_video(args) -> None:
    from .video import render_video

    top = _rising(load_trending(args.data), args.n_topics)
    solid = top[top["videos_recent"] >= 10]  # ignore topics with only a handful of videos
    best = (solid if len(solid) else top).iloc[0]
    kws = best["keywords"].split(", ")
    script = make_script(kws[0], kws, float(best["momentum"]))
    print("Script:", json.dumps(script.to_dict(), indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(script.to_dict(), indent=2))
    render_video(script.scenes(), args.out, gif_path=args.gif)
    print(f"Video -> {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trendpredictor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "topics", "video"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/USvideos.csv")
        s.add_argument("--n-topics", type=int, default=12)
        if name == "train":
            s.add_argument("--model", default="models/trend_model.joblib")
            s.add_argument("--test-from", default="2018-04-01")
        if name == "video":
            s.add_argument("--out", default="assets/demo.mp4")
            s.add_argument("--gif", default="assets/demo.gif")
    args = p.parse_args(argv)
    {"train": cmd_train, "topics": cmd_topics, "video": cmd_video}[args.cmd](args)


if __name__ == "__main__":
    main()
