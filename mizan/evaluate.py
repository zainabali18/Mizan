"""Shared evaluation so every model is scored exactly the same way.

Each experiment writes results/<name>.json with:
  accuracy, macro_f1, per-variety (MSA / PAL) accuracy and macro-F1,
  average latency per query (ms), model size (MB), and notes.
`python -m mizan.report` then turns those files into the README table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def score(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # zero_division=0: an intent the model never predicts counts as F1 0, not an error
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "n": len(y_true),
    }


def evaluate(
    name: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    varieties: Sequence[str],
    latency_ms: float,
    size_mb: float,
    notes: str = "",
    save: bool = True,
) -> dict:
    """Score predictions overall and per Arabic variety, optionally save to results/."""
    if not (len(y_true) == len(y_pred) == len(varieties)):
        raise ValueError("y_true, y_pred and varieties must be the same length")
    result = {"model": name, **score(y_true, y_pred), "by_variety": {}}
    for v in sorted(set(varieties)):
        idx = [i for i, x in enumerate(varieties) if x == v]
        result["by_variety"][v] = score([y_true[i] for i in idx], [y_pred[i] for i in idx])
    result["latency_ms"] = round(float(latency_ms), 2)
    result["size_mb"] = round(float(size_mb), 1)
    result["notes"] = notes
    if save:
        RESULTS.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        (RESULTS / f"{slug}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
