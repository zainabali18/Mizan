"""Baseline 1: TF-IDF character n-grams + logistic regression.

A classic, CPU-only model that trains in seconds. Character n-grams (2-5
letters, within word boundaries) cope well with Arabic prefixes/suffixes and
dialect spelling, e.g. "بطاقتي" and "البطاقه" share most of their n-grams.
The regularisation strength C is chosen on the validation split only; the
test split is used once, at the end.

Run:  python -m mizan.baseline_tfidf
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .data import PROCESSED, clean_text
from .evaluate import ROOT, evaluate

MODEL_PATH = ROOT / "models" / "tfidf_logreg.joblib"
C_GRID = (1, 3, 10, 30, 100)


def make_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=c, max_iter=3000)),
    ])


def main() -> dict:
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "validation.csv")
    test = pd.read_csv(PROCESSED / "test.csv")

    # 1. Choose C on validation.
    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        model = make_pipeline(c).fit(train.text_clean, train.label)
        f1 = f1_score(val.label, model.predict(val.text_clean), average="macro", zero_division=0)
        print(f"C={c:<4} validation macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    # 2. Retrain on train + validation with the chosen C, then score test once.
    full = pd.concat([train, val])
    model = make_pipeline(best_c).fit(full.text_clean, full.label)

    # Latency is measured end to end on raw text: PII redaction + normalisation + predict.
    start = time.perf_counter()
    preds = [model.predict([clean_text(t)])[0] for t in test.text]
    latency_ms = (time.perf_counter() - start) * 1000 / len(test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    size_mb = Path(MODEL_PATH).stat().st_size / 1e6

    result = evaluate(
        "TF-IDF + logistic regression",
        test.label.tolist(), preds, test.variety.tolist(),
        latency_ms=latency_ms, size_mb=size_mb,
        notes=f"char_wb 2-5 grams, C={best_c} (chosen on validation), CPU",
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
