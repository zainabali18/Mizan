"""Build the processed dataset from the raw ArBanking77 CSVs.

Run from the project root:
    python -m mizan.data

Reads   data/raw/arbanking77_{train,validation,test}.csv   (columns: label, text)
Writes  data/processed/{train,validation,test}.csv          (adds text_clean, label_id, variety)
        data/processed/labels.json                          (label <-> id mapping)
        data/processed/summary.json                         (row counts and checks)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .normalise import normalise_arabic
from .pii import redact

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ("train", "validation", "test")

# The Hugging Face test split is two files joined in order:
# Banking77_Arabized_MSA_test_sample.csv (179 rows) then the PAL file (191 rows).
TEST_MSA_ROWS = 179
TEST_TOTAL_ROWS = 370


def clean_text(text: str) -> str:
    """The exact preprocessing used for training AND at inference time."""
    redacted, _ = redact(text)
    return normalise_arabic(redacted)


def load_raw(split: str, raw_dir: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"arbanking77_{split}.csv")
    missing = {"label", "text"} - set(df.columns)
    if missing:
        raise ValueError(f"{split}: missing columns {missing}")
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip()
    df["text"] = df["text"].str.strip()
    return df.reset_index(drop=True)


def tag_variety(df: pd.DataFrame, split: str) -> pd.Series:
    """MSA vs Palestinian dialect is only known for the test split."""
    if split != "test":
        return pd.Series(["mixed"] * len(df), index=df.index)
    if len(df) != TEST_TOTAL_ROWS:
        raise ValueError(f"expected {TEST_TOTAL_ROWS} test rows, got {len(df)}")
    return pd.Series(["MSA" if i < TEST_MSA_ROWS else "PAL" for i in range(len(df))], index=df.index)


def build(raw_dir: Path = RAW, out_dir: Path = PROCESSED) -> dict:
    frames = {s: load_raw(s, raw_dir) for s in SPLITS}

    labels = sorted(frames["train"]["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    for split in ("validation", "test"):
        unseen = set(frames[split]["label"]) - set(labels)
        if unseen:
            raise ValueError(f"{split} has labels not in train: {unseen}")

    train_texts = set(frames["train"]["text"])
    summary = {"num_labels": len(labels), "splits": {}}
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, df in frames.items():
        df["text_clean"] = df["text"].map(clean_text)
        df["label_id"] = df["label"].map(label2id)
        df["variety"] = tag_variety(df, split)
        overlap = 0 if split == "train" else int(df["text"].isin(train_texts).sum())
        summary["splits"][split] = {
            "rows": len(df),
            "labels_present": int(df["label"].nunique()),
            "duplicates": int(df.duplicated(subset=["text"]).sum()),
            "overlap_with_train": overlap,
            "variety_counts": df["variety"].value_counts().to_dict(),
        }
        df[["text", "text_clean", "label", "label_id", "variety"]].to_csv(
            out_dir / f"{split}.csv", index=False
        )

    (out_dir / "labels.json").write_text(
        json.dumps({"label2id": label2id, "id2label": {i: l for l, i in label2id.items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
