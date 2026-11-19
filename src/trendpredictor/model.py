"""The trend model: text (TF-IDF + SVD) + numeric features -> gradient boosting.

Two honest evaluations are provided:
* a single time split (train on the past, test on the most recent months), and
* forward chaining (for each month, train only on what happened before it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .data import NUMERIC_FEATURES, text_of


@dataclass
class TrendModel:
    n_text_components: int = 40
    random_state: int = 0
    tfidf: TfidfVectorizer | None = field(default=None, repr=False)
    svd: TruncatedSVD | None = field(default=None, repr=False)
    clf: HistGradientBoostingClassifier | None = field(default=None, repr=False)

    # ---------- building the feature matrix ----------
    def _matrix(self, table: pd.DataFrame, fit: bool) -> np.ndarray:
        text = text_of(table)
        if fit:
            self.tfidf = TfidfVectorizer(min_df=3, max_features=20000, ngram_range=(1, 2), stop_words="english")
            self.svd = TruncatedSVD(self.n_text_components, random_state=self.random_state)
            txt = self.svd.fit_transform(self.tfidf.fit_transform(text))
        else:
            txt = self.svd.transform(self.tfidf.transform(text))
        num = table[NUMERIC_FEATURES].astype(float).to_numpy()
        return np.hstack([num, txt])

    def feature_names(self) -> list[str]:
        return NUMERIC_FEATURES + [f"text_{i}" for i in range(self.n_text_components)]

    # ---------- train / predict ----------
    def fit(self, table: pd.DataFrame) -> "TrendModel":
        X = self._matrix(table, fit=True)
        self.clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.04, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=self.random_state)
        self.clf.fit(X, table["y"].to_numpy())
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model is not trained yet - call fit() first.")
        return self.clf.predict_proba(self._matrix(table, fit=False))[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TrendModel":
        return joblib.load(path)


# ---------- evaluation ----------
def scores(y: np.ndarray, p: np.ndarray, confident: float = 0.25) -> dict:
    """Accuracy, AUC, F1 and the accuracy on the model's confident calls only."""
    pred = (p >= 0.5).astype(int)
    sure = (p >= 1 - confident) | (p <= confident)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "auc": round(float(roc_auc_score(y, p)), 3),
        "f1": round(float(f1_score(y, pred)), 3),
        "majority_baseline": round(float(max(y.mean(), 1 - y.mean())), 3),
        "confident_accuracy": round(float(accuracy_score(y[sure], pred[sure])), 3) if sure.any() else None,
        "confident_coverage": round(float(sure.mean()), 3),
    }


def time_split(table: pd.DataFrame, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(test_start)
    return table[table["tdate"] < cut], table[table["tdate"] >= cut]


def evaluate_time_split(table: pd.DataFrame, test_start: str = "2018-04-01") -> tuple[dict, TrendModel, pd.DataFrame]:
    train, test = time_split(table, test_start)
    model = TrendModel().fit(train)
    p = model.predict_proba(test)
    out = test[["title", "channel_title", "tdate", "span", "y"]].copy()
    out["p"] = p
    res = scores(test["y"].to_numpy(), p)
    res.update(train_n=int(len(train)), test_from=test_start)
    return res, model, out


def evaluate_forward_chaining(table: pd.DataFrame, months: list[str]) -> list[dict]:
    """For each month: train on everything before it, test on that month."""
    rows = []
    for m in months:
        start = pd.Period(m, "M").start_time
        end = pd.Period(m, "M").end_time
        train = table[table["tdate"] < start]
        test = table[(table["tdate"] >= start) & (table["tdate"] <= end)]
        if len(test) < 100 or len(train) < 300:
            continue
        p = TrendModel().fit(train).predict_proba(test)
        r = scores(test["y"].to_numpy(), p)
        r["month"] = m
        rows.append(r)
    return rows


def save_metrics(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
