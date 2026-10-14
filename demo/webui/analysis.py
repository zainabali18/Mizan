"""Per-example analysis of the fine-tuned model.

Shown only when results/arabert_test_predictions.csv exists (produced by
notebooks/04_export_predictions.ipynb). Until then the site says plainly that
this analysis has not been run, rather than showing anything invented.
"""
from __future__ import annotations

import pandas as pd

from . import content as C

REQUIRED = {"text", "label", "pred", "score", "variety"}


def load_predictions() -> pd.DataFrame | None:
    if not C.PREDICTIONS.exists():
        return None
    df = pd.read_csv(C.PREDICTIONS)
    return df if REQUIRED <= set(df.columns) else None


def per_intent_f1(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import precision_recall_fscore_support

    labs = sorted(set(df.label))
    p, r, f, n = precision_recall_fscore_support(df.label, df.pred, labels=labs, zero_division=0)
    out = pd.DataFrame({"intent": labs, "precision": p, "recall": r, "f1": f, "support": n})
    return out.sort_values(["f1", "support"], ascending=[True, False]).reset_index(drop=True)


def top_confusions(df: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    wrong = df[df.label != df.pred]
    return (wrong.groupby(["label", "pred"]).size().reset_index(name="count")
            .sort_values("count", ascending=False).head(k).reset_index(drop=True))


def score_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, .5, .7, .85, .95, 1.0001]
    names = ["< 0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95", "≥ 0.95"]
    d = df.assign(bucket=pd.cut(df.score, bins=bins, labels=names, right=False), correct=df.label == df.pred)
    g = d.groupby("bucket", observed=False).agg(queries=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    return g


def render_arabert_section() -> None:
    import streamlit as st

    from . import components as ui

    df = load_predictions()
    st.html('<div class="mz-h3" style="margin-top:1.6rem">Fine-tuned AraBERT: per-intent and per-example analysis</div>')
    if df is None:
        ui.callout(
            "<b>Not yet measured.</b> The fine-tuning run recorded aggregate scores only (overall, MSA and dialect). "
            "Per-intent F1, the most common confusions and example successes and failures need the model's "
            "prediction for each test query. Running <span class='mz-mono'>notebooks/04_export_predictions.ipynb</span> "
            "creates <span class='mz-mono'>results/arabert_test_predictions.csv</span>, and this section then fills in "
            "automatically.", warn=True)
        return

    acc = (df.label == df.pred).mean()
    st.html(f'<p class="mz-p">From {len(df)} exported test predictions (accuracy {C.pct(acc)}).</p>')
    a, b = st.columns(2, gap="large")
    with a:
        pi = per_intent_f1(df).head(8)
        ui.table(["Hardest intents", "F1", "Test n"], [
            [f'<span class="mz-ar" lang="ar">{ui.esc(x.intent)}</span><br><span class="mz-small">{ui.esc(C.english_for(x.intent))}</span>',
             C.pct(x.f1, 0), str(int(x.support))] for x in pi.itertuples()], num_cols=(1, 2))
    with b:
        tc = top_confusions(df)
        ui.table(["Correct intent", "Predicted as", "n"], [
            [ui.esc(C.english_for(x.label)), ui.esc(C.english_for(x.pred)), str(x.count)] for x in tc.itertuples()],
            num_cols=(2,))
    sb = score_buckets(df)
    ui.table(["Model score", "Test queries", "Accuracy"], [
        [ui.esc(str(x.bucket)), str(int(x.queries)), "–" if x.queries == 0 else C.pct(x.accuracy)]
        for x in sb.itertuples()], num_cols=(1, 2),
        caption="If accuracy rises with the score, the score is useful for routing, even though it is not a calibrated probability.")
    ok, bad = df[df.label == df.pred], df[df.label != df.pred]
    rows = []
    for d, tag in ((ok.sort_values("score", ascending=False).head(3), "Correct"), (bad.sort_values("score", ascending=False).head(4), "Wrong")):
        for x in d.itertuples():
            rows.append([tag, f'<div class="mz-ar" lang="ar">{ui.esc(x.text)}</div>',
                         ui.esc(C.english_for(x.label)), ui.esc(C.english_for(x.pred)), f"{x.score:.2f}"])
    ui.table(["", "Customer message", "Correct", "Predicted", "Score"], rows, num_cols=(4,),
             caption="Wrong answers with high scores are the riskiest: they would pass the review threshold.")
