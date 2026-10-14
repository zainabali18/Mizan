import json

import pandas as pd
import pytest

from mizan.data import RAW, build, clean_text


def test_clean_text_redacts_then_normalises():
    assert clean_text("بطاقتي 4111 1111 1111 1111 انرفضت!") == "بطاقتي [CARD_NUMBER] انرفضت!"
    assert clean_text("أريد إلغاء التحويل") == "اريد الغاء التحويل"


def test_clean_text_without_normalisation_keeps_letters_but_still_redacts():
    assert clean_text("أريد  إلغاء 0501234567", normalise=False) == "أريد إلغاء [PHONE]"


@pytest.mark.skipif(not (RAW / "arbanking77_train.csv").exists(), reason="raw data not present")
def test_build_on_real_data(tmp_path):
    summary = build(out_dir=tmp_path)
    assert summary["num_labels"] == 77
    assert {s: v["rows"] for s, v in summary["splits"].items()} == {"train": 1078, "validation": 124, "test": 370}
    for split in ("validation", "test"):
        assert summary["splits"][split]["overlap_with_train"] == 0
    assert summary["splits"]["test"]["variety_counts"] == {"PAL": 191, "MSA": 179}

    test = pd.read_csv(tmp_path / "test.csv")
    assert list(test.columns) == ["text", "text_clean", "label", "label_id", "variety"]
    assert test["label_id"].notna().all()

    labels = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert len(labels["label2id"]) == 77
