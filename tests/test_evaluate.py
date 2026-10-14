import json

import pytest

from mizan import evaluate as ev
from mizan import report


def test_scores_overall_and_by_variety(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    y_true = ["a", "b", "a", "b"]
    y_pred = ["a", "b", "b", "b"]
    varieties = ["MSA", "MSA", "PAL", "PAL"]
    r = ev.evaluate("Test model", y_true, y_pred, varieties, latency_ms=1.234, size_mb=10)
    assert r["accuracy"] == 0.75
    assert r["by_variety"]["MSA"]["accuracy"] == 1.0
    assert r["by_variety"]["PAL"]["accuracy"] == 0.5
    assert r["latency_ms"] == 1.23
    saved = json.loads((tmp_path / "test_model.json").read_text(encoding="utf-8"))
    assert saved["model"] == "Test model"


def test_unparseable_predictions_count_as_wrong(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    r = ev.evaluate("m", ["a", "b"], ["a", "UNPARSEABLE"], ["MSA", "MSA"], 1, 1, save=False)
    assert r["accuracy"] == 0.5


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ev.evaluate("m", ["a"], ["a", "b"], ["MSA"], 1, 1, save=False)


def test_report_table(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    ev.evaluate("TF-IDF + logistic regression", ["a", "b"], ["a", "a"], ["MSA", "PAL"], 2.2, 10.4)
    md = report.table()
    assert "| TF-IDF + logistic regression | 50.0% |" in md
    assert "2.2 ms" in md and "10.4 MB" in md
