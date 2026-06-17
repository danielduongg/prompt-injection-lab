"""
detector.py -- a lightweight, transparent prompt-injection classifier.

Design goal: a cheap, interpretable *input filter* (word + character TF-IDF ->
logistic regression) that can sit in front of an LLM, in the spirit of Anthropic's
Constitutional Classifiers but small enough to train in seconds and reason about.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def build_detector() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def evaluate(model, X, y, threshold: float = 0.5):
    """Score a fitted model on (X, y); return (metrics, roc_curve_dict, proba)."""
    X = np.asarray(X, dtype=object).astype(str)
    y = np.asarray(y).astype(int)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = dict(
        n=int(len(y)), n_pos=int(y.sum()),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        pr_auc=float(average_precision_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        confusion=confusion_matrix(y, pred).tolist(),
    )
    fpr, tpr, _ = roc_curve(y, proba)
    return metrics, {"fpr": fpr.tolist(), "tpr": tpr.tolist()}, proba


def train_and_eval(df, test_size: float = 0.25, seed: int = 42):
    X = df["text"].astype(str).values
    y = df["label"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    model = build_detector().fit(Xtr, ytr)
    metrics, roc, _ = evaluate(model, Xte, yte)
    metrics["n_train"] = int(len(Xtr))
    return model, metrics, roc, (Xte, yte)


@dataclass
class DetectorFilter:
    """Wraps a fitted detector as an input filter for the eval harness."""
    model: object
    threshold: float = 0.5

    def score(self, text: str) -> float:
        return float(self.model.predict_proba([text])[0, 1])

    def flag(self, text: str) -> bool:
        return self.score(text) >= self.threshold
