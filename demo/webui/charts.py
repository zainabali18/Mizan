"""Altair charts. Palette validated for colour-blind separation (dataviz validator: all checks pass)."""
from __future__ import annotations

import altair as alt
import pandas as pd

INK, MUTED, GRID = "#14213A", "#545E6E", "#E9E3D7"
SERIES = {"Overall": "#3A5FA0", "MSA": "#12988F", "Dialect (PAL)": "#C2852B"}
FONT = "Plex Arabic, IBM Plex Sans, sans-serif"


def _style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=FONT, background="#FFFFFF")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                            labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, orient="top", title=None,
                              symbolType="square")
            .properties(padding={"left": 8, "right": 16, "top": 8, "bottom": 8}))


def f1_by_model(results: dict, names: dict) -> alt.Chart:
    rows = []
    for key in ("arabert", "tfidf", "llm"):
        r = results[key]
        for subset, val in (("Overall", r["macro_f1"]), ("MSA", r["by_variety"]["MSA"]["macro_f1"]),
                            ("Dialect (PAL)", r["by_variety"]["PAL"]["macro_f1"])):
            rows.append({"model": names[key], "subset": subset, "macro_f1": val, "label": f"{val * 100:.1f}%"})
    df = pd.DataFrame(rows)
    order = [names[k] for k in ("arabert", "tfidf", "llm")]
    sub_order = list(SERIES)
    base = alt.Chart(df).encode(
        y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=13, labelColor=INK)),
        yOffset=alt.YOffset("subset:N", sort=sub_order),
        x=alt.X("macro_f1:Q", title="Macro-F1 on the test set", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=13).encode(
        color=alt.Color("subset:N", scale=alt.Scale(domain=sub_order, range=list(SERIES.values())),
                        sort=sub_order),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("subset:N", title="Test subset"),
                 alt.Tooltip("macro_f1:Q", title="Macro-F1", format=".1%")])
    text = base.mark_text(align="left", dx=5, fontSize=11.5, color=INK).encode(text="label:N")
    return _style((bars + text).properties(height=310))


def training_curve(log: dict) -> alt.Chart:
    df = pd.DataFrame(log["epochs"])
    long = df.melt(id_vars="epoch", value_vars=["val_macro_f1", "val_accuracy"], var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"val_macro_f1": "Validation macro-F1", "val_accuracy": "Validation accuracy"})
    colors = alt.Scale(domain=["Validation macro-F1", "Validation accuracy"], range=["#3A5FA0", "#12988F"])
    lines = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X("epoch:Q", title="Epoch", scale=alt.Scale(domain=[1, len(df)]), axis=alt.Axis(tickCount=10)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("metric:N", scale=colors),
        tooltip=[alt.Tooltip("epoch:Q", title="Epoch"), alt.Tooltip("metric:N", title="Metric"),
                 alt.Tooltip("value:Q", title="Value", format=".1%")])
    best = int(log["best_epoch"])
    rule = alt.Chart(pd.DataFrame({"epoch": [best]})).mark_rule(color=INK, strokeDash=[4, 4], strokeWidth=1.2).encode(x="epoch:Q")
    note = alt.Chart(pd.DataFrame({"epoch": [best], "y": [0.12], "t": [f"Epoch {best}: best validation macro-F1, kept"]})
                     ).mark_text(align="right", dx=-6, fontSize=11.5, color=INK).encode(x="epoch:Q", y="y:Q", text="t:N")
    return _style((lines + rule + note).properties(height=280))
