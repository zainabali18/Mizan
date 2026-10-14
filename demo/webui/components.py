"""Reusable Streamlit components for the Mizan site.

All user-supplied text is HTML-escaped before it is placed in markup.
"""
from __future__ import annotations

import html as _html
from pathlib import Path
from typing import Iterable, Sequence

import streamlit as st

from . import content as C

_CSS_PATH = Path(__file__).parent / "style.css"


def inject_css() -> None:
    # Passing a Path to a .css file is the documented way: Streamlit wraps it in <style>
    # and places it in the event container so it takes no layout space.
    st.html(_CSS_PATH)


def esc(text: str) -> str:
    return _html.escape(str(text))


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def get_classifier():
    """Load the fine-tuned model once per server process (shared by all pages and visitors)."""
    from mizan.service import MizanClassifier

    clf = MizanClassifier()
    clf.warm_up()
    return clf


@st.cache_data(show_spinner=False, max_entries=64)
def classify_example(text: str) -> dict:
    """Cached inference for the fixed examples (e.g. the overview preview)."""
    return get_classifier().classify(text)


# ---------------------------------------------------------------- layout pieces
def page_header(eyebrow: str, title: str, lede: str | None = None) -> None:
    st.html(
        f'<p class="mz-eyebrow">{esc(eyebrow)}</p><h1 class="mz-h1">{title}</h1>'
        + (f'<p class="mz-lede">{lede}</p>' if lede else "")
    )


def section(num: str, title: str, intro: str | None = None) -> None:
    st.html(
        f'<div class="mz-section"><div class="mz-section-head"><span class="mz-section-num">{esc(num)}</span>'
        f'<div><h2 class="mz-h2">{title}</h2>' + (f'<p class="mz-p" style="max-width:46rem">{intro}</p>' if intro else "")
        + "</div></div></div>"
    )


def links(items: Iterable[tuple[str, str]]) -> str:
    return '<div class="mz-links">' + "".join(
        f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(label)}</a>' for label, url in items) + "</div>"


def metric_strip(items: Sequence[tuple[str, str, str]]) -> None:
    """items: (value_html, label, note)."""
    cells = "".join(
        f'<div class="mz-metric"><div class="mz-metric-value">{v}</div>'
        f'<div class="mz-metric-label">{esc(label)}</div><div class="mz-metric-note">{esc(note)}</div></div>'
        for v, label, note in items)
    st.html(f'<div class="mz-metrics" role="list">{cells}</div>')


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], num_cols: Iterable[int] = (),
          highlight: int | None = None, caption: str | None = None) -> None:
    """Rows are trusted HTML strings built by the site (never raw user input)."""
    num_cols = set(num_cols)
    head = "".join(f'<th class="{"num" if i in num_cols else ""}" scope="col">{h}</th>' for i, h in enumerate(headers))
    body = "".join(
        f'<tr class="{"mz-hl" if r == highlight else ""}">'
        + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(row)) + "</tr>"
        for r, row in enumerate(rows))
    cap = f'<p class="mz-fig-cap" style="margin-top:.5rem">{caption}</p>' if caption else ""
    st.html(f'<div class="mz-table-wrap"><table class="mz-table"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>{cap}")


def callout(body_html: str, warn: bool = False) -> None:
    st.html(f'<div class="mz-callout{" mz-callout-warn" if warn else ""}">{body_html}</div>')


def footer() -> None:
    st.html(
        '<footer class="mz-footer"><div><b style="color:var(--ink)">Mizan · ميزان</b>, a portfolio project by '
        f'<a href="{C.LINKEDIN_URL}" target="_blank" rel="noopener">Zainab Ali</a>. '
        "A demonstration: it is not connected to any bank, and no real customer data is used.</div>"
        f'<div><a href="{C.REPO_URL}" target="_blank" rel="noopener">GitHub</a> · '
        f'<a href="{C.MODEL_URL}" target="_blank" rel="noopener">Model</a> · '
        f'Data: <a href="{C.DATASET_URL}" target="_blank" rel="noopener">ArBanking77</a> (SinaLab, Birzeit University)</div>'
        "</footer>"
    )


def empty_state() -> None:
    st.html(
        '<div class="mz-empty"><div class="mz-h3">Your result will appear here</div>'
        '<p class="mz-muted" style="margin:.3rem auto 0;max-width:26rem">Choose an example or write a message in Arabic, '
        "then select <b>Classify message</b>. You'll see each stage of the pipeline:</p>"
        '<div class="mz-empty-steps"><span class="mz-chip">1 · Personal data redacted</span>'
        '<span class="mz-chip">2 · Intent predicted</span><span class="mz-chip">3 · Threshold decision</span></div></div>'
    )


# ---------------------------------------------------------------- result panel
def result_panel(r: dict, original: str, bar_label: str, show_original: bool = True, compact: bool = False) -> str:
    """HTML for one classification result. `r` is the dict returned by MizanClassifier.classify."""
    th = float(r["threshold"])
    score = float(r["confidence"])
    auto = r["route"] == "auto"
    pii = r["pii_detected"]

    steps = []
    if show_original:
        steps.append(("Customer message", f'<div class="mz-msg mz-ar" lang="ar">{C.display_text(original)}</div>'))

    if pii:
        counts = {}
        for k in pii:
            counts[k] = counts.get(k, 0) + 1
        chips = "".join(f'<span class="mz-chip mz-chip-pii">{esc(C.PII_LABELS.get(k, k))}'
                        f'{" ×" + str(n) if n > 1 else ""}</span>' for k, n in counts.items())
        note = f'<div>{chips}</div>'
    else:
        note = ('<div class="mz-small" style="margin-top:.35rem">No supported personal-data pattern found '
                '(Emirates ID, IBAN, card, phone, email).</div>')
    steps.append(("What the model receives (personal data redacted)",
                  f'<div class="mz-msg mz-ar" lang="ar">{C.highlight_redactions(r["redacted_text"])}</div>{note}'))

    intent_en = C.english_for(r["intent"])
    steps.append(("Predicted intent",
                  f'<div class="mz-intent-ar mz-ar" lang="ar">{esc(r["intent"])}</div>'
                  f'<div class="mz-intent-en">{esc(intent_en)}</div>'))

    fill = max(0.0, min(score, 1.0)) * 100
    decision = (
        ('<div class="mz-decision mz-decision-auto" role="status"><span class="mz-decision-icon" aria-hidden="true">✓</span>'
         f'<div><b>Meets the {th:.2f} threshold</b>Eligible for automatic handling.</div></div>')
        if auto else
        ('<div class="mz-decision mz-decision-review" role="status"><span class="mz-decision-icon" aria-hidden="true">!</span>'
         f'<div><b>Below the {th:.2f} threshold: flagged for human review</b>'
         "In this demo the flag is shown only; no review queue is connected.</div></div>")
    )
    steps.append((
        "Model score and decision",
        f'<div class="mz-score"><div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem">'
        f'<span style="font-size:1.5rem;font-weight:600;color:var(--ink)">{score:.2f}</span>'
        f'<span class="mz-small">model score (softmax), not a calibrated probability</span></div>'
        f'<div class="mz-score-track" role="img" aria-label="Score {score:.2f}; threshold {th:.2f}">'
        f'<div class="mz-score-fill" style="width:{fill:.1f}%"></div>'
        f'<div class="mz-score-mark" style="left:calc({th * 100:.1f}% - 1px)"></div></div>'
        f'<div class="mz-score-legend"><span>0</span><span style="color:var(--teal-ink)">threshold {th:.2f}</span><span>1</span></div>'
        f"</div>{decision}"))

    if not compact:
        alts = []
        for j, t in enumerate(r["top_intents"]):
            w = max(0.0, min(float(t["confidence"]), 1.0)) * 100
            alts.append(
                f'<div class="mz-alt{" mz-alt-top" if j == 0 else ""}">'
                f'<div class="mz-alt-name"><span class="mz-ar" lang="ar">{esc(t["intent"])}</span> · '
                f'<span style="color:var(--muted)">{esc(C.english_for(t["intent"]))}</span></div>'
                f'<div class="mz-alt-score">{float(t["confidence"]):.2f}</div>'
                f'<div class="mz-alt-bar"><span style="width:{w:.1f}%"></span></div></div>')
        steps.append(("Top three intents by model score", f'<div class="mz-alts">{"".join(alts)}</div>'))

    body = "".join(
        f'<div class="mz-step"><div class="mz-step-n">{i}</div><div><div class="mz-step-label">{esc(label)}</div>{inner}</div></div>'
        for i, (label, inner) in enumerate(steps, 1))
    meta = (f'<div class="mz-meta"><span>Inference time: <b>{r["latency_ms"]:.0f} ms</b> on this server\'s CPU</span>'
            f'<span>Model: fine-tuned AraBERT (77 intents)</span></div>')
    return (f'<div class="mz-panel"><div class="mz-panel-bar"><span><span class="mz-live-dot"></span>{esc(bar_label)}</span>'
            f'<span>Mizan · ميزان</span></div><div class="mz-flow">{body}</div>{meta}</div>')
