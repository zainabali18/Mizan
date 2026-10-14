"""Tests for the website's content layer: every number shown must match the records."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

from webui import analysis  # noqa: E402
from webui import content as C  # noqa: E402


def test_every_intent_has_a_unique_english_name():
    rows = C.intent_table()
    assert len(rows) == 77 == len(C.BANKING77) == len(set(C.BANKING77))
    assert [r["id"] for r in rows] == list(range(77))
    assert sum(r["train_examples"] for r in rows) == 1078


@pytest.mark.parametrize("arabic,english", [
    ("وصول البطاقة", "Card arrival"), ("حد السن", "Age limit"), ("فيزا أو ماستر كارد", "Visa or Mastercard"),
    ("آبل باي أو جوجل باي", "Apple Pay or Google Pay"), ("البطاقة المفقودة أو المسروقة", "Lost or stolen card"),
    ("تغيير رمز التعريف الشخصي", "Change PIN"), ("دعم أجهزة الصراف الآلي", "ATM support"),
])
def test_spot_check_translations(arabic, english):
    assert C.english_for(arabic) == english


def test_headline_numbers_match_recorded_results():
    h = C.headline()
    rec = {json.loads(p.read_text(encoding="utf-8"))["model"]: json.loads(p.read_text(encoding="utf-8"))
           for p in (ROOT / "results").glob("*.json")}
    cfg = json.loads((ROOT / "mizan_config.json").read_text(encoding="utf-8"))
    assert h["arabert_f1"] == rec["Fine-tuned AraBERT"]["macro_f1"]
    assert h["llm_f1"] == rec["Zero-shot LLM (Qwen2.5-1.5B-Instruct)"]["macro_f1"]
    assert h["tfidf_f1"] == rec["TF-IDF + logistic regression"]["macro_f1"]
    assert h["threshold"] == cfg["confidence_threshold"]
    assert h["auto_share"] + h["review_share"] == pytest.approx(1.0, abs=1e-4)


def test_recorded_llm_errors_exist_in_test_set():
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    for e in C.llm_errors()["examples"]:
        match = test[test.text == e["text"]]
        assert len(match) == 1 and match.iloc[0].label == e["true"]


def test_training_log_best_epoch_is_the_max_validation_f1():
    log = C.training_log()
    best = max(log["epochs"], key=lambda e: e["val_macro_f1"])
    assert best["epoch"] == log["best_epoch"] and len(log["epochs"]) == 20


def test_user_text_is_escaped_and_tags_highlighted():
    out = C.highlight_redactions('<img src=x onerror=alert(1)> [CARD_NUMBER]')
    assert "<img" not in out and "&lt;img" in out
    assert '<mark class="pii"' in out and "[CARD_NUMBER]</mark>" in out


@pytest.mark.parametrize("text,isolated", [
    ("بطاقتي 4111 1111 1111 1111 انسرقت", "4111 1111 1111 1111"),
    ("هويتي 784-1990-1234567-1", "784-1990-1234567-1"),
    ("على AE07 0331 2345 6789 0123 456", "AE07 0331 2345 6789 0123 456"),
])
def test_multi_part_numbers_keep_reading_order(text, isolated):
    assert f'<span class="mz-ltr">{isolated}</span>' in C.display_text(text)


def test_plain_numbers_are_left_alone():
    assert C.display_text("سحبت 500 جنيه") == "سحبت 500 جنيه"


def test_has_arabic():
    assert C.has_arabic("بطاقتي") and not C.has_arabic("my card")


def test_stylesheet_has_no_angle_brackets():
    # Streamlit's HTML sanitiser drops the whole stylesheet if it contains < or >.
    css = (ROOT / "demo" / "webui" / "style.css").read_text(encoding="utf-8")
    assert "<" not in css and ">" not in css


def test_examples_use_fictional_test_values():
    texts = " ".join(e["text"] for e in C.EXAMPLES)
    assert "4111 1111 1111 1111" in texts          # standard Visa test number
    assert all(C.has_arabic(e["text"]) for e in C.EXAMPLES)


def test_prediction_analysis_helpers():
    df = pd.DataFrame({"text": list("abcdef"), "label": ["x", "x", "y", "y", "z", "z"],
                       "pred": ["x", "y", "y", "y", "x", "z"], "score": [.99, .4, .9, .6, .8, .97],
                       "variety": ["MSA"] * 6})
    pi = analysis.per_intent_f1(df)
    assert set(pi.intent) == {"x", "y", "z"} and pi.f1.between(0, 1).all()
    tc = analysis.top_confusions(df)
    assert tc["count"].sum() == 2
    sb = analysis.score_buckets(df)
    assert sb.queries.sum() == 6


def test_spot_checks_match_the_examples_and_labels():
    s = C.spot_checks()
    titles = {e["title"] for e in s["site_examples"]}
    assert len(titles) == len(C.EXAMPLES)
    known = {**C.labels()}
    for row in s["variants"]["rows"] + s["site_examples"]:
        assert row["intent"] in known
        assert (row["route"] == "auto") == (row["score"] >= s["threshold"])


def test_preview_is_not_the_known_failure():
    assert not C.PREVIEW_EXAMPLE.get("note")
    assert sum(bool(e.get("note")) for e in C.EXAMPLES) == 1