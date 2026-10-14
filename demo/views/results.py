import streamlit as st

from webui import charts
from webui import components as ui
from webui import content as C
from webui import analysis

R = C.results()
H = C.headline()
S = C.data_summary()["splits"]
LOG = C.training_log()

ui.page_header(
    "Results and evaluation",
    "A specialist model, measured against two baselines",
    f"All three approaches were scored by the same code on the same {H['test_n']} held-out Arabic banking queries. "
    "This page shows what was measured, how, and what the numbers do and do not establish.",
)

# ---------------------------------------------------------------- 01 comparison
ui.section("01", "Headline comparison",
           "Macro-F1 is the main metric (defined below). Latency and size come from different hardware, "
           "so the setup column matters when comparing them.")
rows = []
for key in ("arabert", "tfidf", "llm"):
    r = R[key]
    rows.append([
        ui.esc(C.SHORT_NAMES[key]), ui.esc(C.SETUP[key]["training"]),
        C.pct(r["accuracy"]), f"<b>{C.pct(r['macro_f1'])}</b>",
        C.pct(r["by_variety"]["MSA"]["macro_f1"]), C.pct(r["by_variety"]["PAL"]["macro_f1"]),
        f"{r['latency_ms']:.1f} ms<br><span class='mz-small'>{ui.esc(C.SETUP[key]['latency'])}</span>",
        f"{r['size_mb']:,.0f} MB" if r["size_mb"] >= 10 else f"{r['size_mb']:.1f} MB",
    ])
ui.table(["Model", "Training", "Accuracy", "Macro-F1", "MSA F1", "Dialect F1", "Latency / query", "Size"],
         rows, num_cols=(2, 3, 4, 5, 7), highlight=0,
         caption=f"Test set: {S['test']['rows']} queries ({S['test']['variety_counts']['MSA']} MSA, "
                 f"{S['test']['variety_counts']['PAL']} Palestinian dialect). One training run per model; "
                 "no confidence intervals were computed.")

st.altair_chart(charts.f1_by_model(R, C.SHORT_NAMES), width="stretch",
                alt="Bar chart of macro-F1 by model and test subset. Fine-tuned AraBERT: 76.2% overall, 72.6% MSA, "
                    "75.4% dialect. TF-IDF: 61.0%, 52.0%, 64.1%. Zero-shot LLM: 7.4%, 8.6%, 6.4%.")
st.html('<p class="mz-fig-cap">Macro-F1 overall and for each kind of Arabic in the test set. Hover over a bar for its value.</p>')

gap_tfidf = (H["arabert_f1"] - H["tfidf_f1"]) * 100
t1, t2, t3 = st.columns(3, gap="large")
t1.html(f'<div class="mz-h3">+{gap_tfidf:.1f} points over TF-IDF</div><p class="mz-p">Fine-tuning AraBERT on 1,078 '
        "examples beat a tuned classical baseline on every subset. The baseline had slightly <i>more</i> "
        "training data (train + validation).</p>")
t2.html(f'<div class="mz-h3">The zero-shot LLM reached {C.pct(H["llm_f1"])}</div><p class="mz-p">Given only the 77 '
        f"intent names, Qwen2.5-1.5B rarely picked the right one, and in {C.llm_errors()['unparseable_outputs']} of "
        f"{H['test_n']} cases it gave no valid answer. It is also {H['llm_mb'] / H['arabert_mb']:.1f}× larger than the "
        "fine-tuned model.</p>")
t3.html('<div class="mz-h3">Dialect held up</div><p class="mz-p">The fine-tuned model scored '
        f"{C.pct(R['arabert']['by_variety']['PAL']['macro_f1'])} on Palestinian dialect and "
        f"{C.pct(R['arabert']['by_variety']['MSA']['macro_f1'])} on MSA, so it did not depend on formal Arabic. "
        "Only one dialect was tested.</p>")

# ---------------------------------------------------------------- 02 metrics
ui.section("02", "How the metrics are defined")
ui.table(["Metric", "Definition", "Why it is used"], [
    ["<b>Macro-F1</b>", "F1 (the balance of precision and recall) computed for each intent separately, then averaged "
     "with equal weight. Computed with scikit-learn over the intents present in the test labels or predictions.",
     "With 77 intents and few examples each, a model could score well on accuracy by getting common intents right. "
     "Macro-F1 gives rare intents equal weight."],
    ["<b>Accuracy</b>", "Share of test queries whose predicted intent matches the label.", "Easy to read; reported alongside macro-F1."],
    ["<b>MSA / Dialect F1</b>", f"Macro-F1 on the {S['test']['variety_counts']['MSA']} MSA and "
     f"{S['test']['variety_counts']['PAL']} Palestinian-dialect test queries separately.", "Shows whether a model only handles formal Arabic."],
    ["<b>Latency</b>", "Average wall-clock time for one query at a time. TF-IDF and AraBERT: end to end (redaction, "
     "cleaning, prediction). LLM: text generation only; building the prompt was not timed.",
     "A support chatbot answers one customer at a time. Batched throughput would flatter the LLM."],
    ["<b>Size</b>", "Size of the saved model files (TF-IDF, AraBERT) or parameters × 2 bytes in fp16 (LLM).",
     "A proxy for hosting cost and memory."],
])

# ---------------------------------------------------------------- 03 setup
ui.section("03", "Evaluation setup")
c1, c2 = st.columns(2, gap="large")
c1.html(
    '<div class="mz-h3">Data</div><ul class="mz-ticks">'
    f'<li><a class="mz-a" href="{C.DATASET_URL}" target="_blank" rel="noopener">ArBanking77</a> public sample '
    "(SinaLab, Birzeit University): Arabic versions of the 77 Banking77 intents, in MSA and Palestinian dialect.</li>"
    f"<li>Splits used as published: <b>{S['train']['rows']:,}</b> train, <b>{S['validation']['rows']}</b> validation, "
    f"<b>{S['test']['rows']}</b> test. The test set covers {S['test']['labels_present']} of the 77 intents.</li>"
    "<li>Checked: no missing values, no duplicates, and <b>no sentence shared</b> between train and validation or test.</li>"
    "<li>The full corpus has 31,404 queries in five dialects; only the public sample was used.</li></ul>"
)
c2.html(
    '<div class="mz-h3">Protocol</div><ul class="mz-ticks">'
    "<li>Model choices (TF-IDF regularisation, AraBERT's best epoch, the review threshold) were made on "
    "<b>validation data only</b>.</li>"
    "<li>The test set was scored once per model, with the same evaluation function.</li>"
    "<li>Personal-data redaction ran on every input, for every model.</li>"
    "<li><b>Differences to keep in mind:</b> TF-IDF trained on train + validation, AraBERT on train only, and the LLM "
    "not at all. Latencies were measured on different hardware (laptop CPU, Colab CPU, T4 GPU).</li></ul>"
)

# ---------------------------------------------------------------- 04 threshold
ui.section("04", "Human-review threshold",
           "A bank should not answer automatically when the model is unsure. The threshold trades coverage for accuracy.")
auto, review = H["auto_share"] * 100, H["review_share"] * 100
st.html(
    '<div class="mz-panel" style="padding:1.2rem 1.3rem">'
    '<div style="display:flex;height:38px;border-radius:8px;overflow:hidden;gap:2px" role="img" '
    f'aria-label="{auto:.1f}% handled automatically, {review:.1f}% flagged for review">'
    f'<div style="width:{auto:.1f}%;background:#0B6E69;color:#fff;display:flex;align-items:center;padding:0 .8rem;'
    f'font-weight:600;font-size:.9rem">{auto:.1f}% automatic</div>'
    f'<div style="width:{review:.1f}%;background:#E8C78E;color:#4A2E00;display:flex;align-items:center;padding:0 .8rem;'
    f'font-weight:600;font-size:.9rem">{review:.1f}% review</div></div>'
    '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:1rem">'
    f'<div><div class="mz-metric-value">{H["threshold"]:.2f}</div><div class="mz-metric-label">Score threshold</div>'
    '<div class="mz-metric-note">Lowest value at which auto-handled validation answers reached 90% accuracy</div></div>'
    f'<div><div class="mz-metric-value">{C.pct(H["auto_acc"])}</div><div class="mz-metric-label">Accuracy when automatic</div>'
    f'<div class="mz-metric-note">On the test set, against {C.pct(H["arabert_acc"])} for all queries</div></div>'
    f'<div><div class="mz-metric-value">{C.pct(H["review_share"])}</div><div class="mz-metric-label">Flagged for review</div>'
    "<div class=\"mz-metric-note\">Test queries whose score fell below the threshold</div></div></div></div>"
)
st.html('<p class="mz-fig-cap" style="margin-top:.7rem">The validation set is small (124 queries), so the threshold is a '
        "starting point. The test result (89.4%) landed just under the 90% validation target. In production, the "
        "threshold would be tuned on more data and monitored.</p>")

# ---------------------------------------------------------------- 05 training
ui.section("05", "Training run",
           f"AraBERT was fine-tuned for 20 epochs ({LOG['training_minutes']:.1f} minutes on a T4 GPU). "
           "The checkpoint with the best validation macro-F1 was kept.")
st.altair_chart(charts.training_curve(LOG), width="stretch",
                alt="Line chart of validation macro-F1 and accuracy over 20 epochs. Both rise steeply until about "
                    "epoch 7, then level off around 70% macro-F1 and 77% accuracy. Epoch 18 is marked as best.")
st.html('<p class="mz-fig-cap">Validation scores level off after about epoch 11, while training loss keeps falling '
        "(1.08 at epoch 6 to 0.01 at epoch 20), a sign the model is fitting the small training set closely. "
        "Earlier stopping would probably cost little.</p>")

# ---------------------------------------------------------------- 06 errors
ui.section("06", "Where models go wrong")
st.html('<div class="mz-h3">Zero-shot LLM: plausible but wrong neighbours</div>'
        '<p class="mz-p">Recorded mistakes from the test set. The model often chose a related banking topic, '
        "and sometimes missed one whose name nearly matched the question.</p>")
E = C.llm_errors()
ui.table(["Customer message", "Correct intent", "LLM answer"], [
    [f'<div class="mz-ar" lang="ar">{ui.esc(e["text"])}</div><div class="mz-small">{ui.esc(e["gloss"])}</div>',
     f'<span class="mz-ar" lang="ar">{ui.esc(e["true"])}</span><br><span class="mz-small">{ui.esc(C.english_for(e["true"]))}</span>',
     ("<span class='mz-small'>No valid intent number returned</span>" if e["pred"] == "UNPARSEABLE" else
      f'<span class="mz-ar" lang="ar">{ui.esc(e["pred"])}</span><br><span class="mz-small">{ui.esc(C.english_for(e["pred"]))}</span>')]
    for e in E["examples"]])

analysis.render_arabert_section()

# Live spot checks with the real model (results/evidence/live_spot_checks.json)
S = C.spot_checks()
V = S["variants"]
st.html('<div class="mz-h3" style="margin-top:2rem">Fine-tuned AraBERT: what live testing found</div>'
        '<p class="mz-p" style="max-width:46rem">Requests sent by hand to the API running the published model on a '
        "MacBook CPU. These are a handful of messages, so they are observations, not measurements. The same "
        "stolen-card request was sent with one part removed at a time to see what changes the answer.</p>")
ui.table(["Message", "Predicted intent", "Score", "Decision", "Correct"], [
    [f'<div class="mz-ar" lang="ar">{C.display_text(v["text"])}</div><div class="mz-small">{ui.esc(v["name"])}: {ui.esc(v["gloss"])}</div>',
     f'<span class="mz-ar" lang="ar">{ui.esc(v["intent"])}</span><br><span class="mz-small">{ui.esc(C.english_for(v["intent"]))}</span>',
     f'{v["score"]:.3f}', "Automatic" if v["route"] == "auto" else "Review",
     "Yes" if v["intent"] == V["expected"] else "<b>No</b>"]
    for v in V["rows"]], num_cols=[2],
    caption=f"Correct answer for every row: {ui.esc(C.english_for(V['expected']))}. Threshold {S['threshold']:.2f}.")
f1, f2 = st.columns(2, gap="large")
f1.html('<div class="mz-h3">A phone mention overrides the card</div><p class="mz-p">Every version containing '
        "\"رقم جوالي\" (\"my phone number\") was classified as <i>lost or stolen phone</i> with a score near 0.98, "
        "so it was cleared for automatic handling. The threshold cannot catch a mistake the model is confident "
        "about, which is why the score is not treated as a probability of being right.</p>")
f2.html('<div class="mz-h3">Redaction tags lower the score</div><p class="mz-p">Adding a card number, which the model '
        "only sees as <span class='mz-mono'>[CARD_NUMBER]</span>, kept the right intent but cut the score from 0.855 to "
        "0.400. The training data contains no personal data, so the model never saw these tags. A fix is to insert "
        "fictional card numbers, phones and IDs into training examples and redact them in the same way.</p>")
site = S["site_examples"]
auto_right = sum(e["route"] == "auto" and e["verdict"] == "right" for e in site)
flagged = sum(e["route"] == "human" for e in site)
st.html(f'<p class="mz-p" style="max-width:46rem">On the {len(site)} examples in the live demo, {auto_right} got the '
        f"expected intent with scores above 0.95, and {flagged} scored below the threshold and were flagged for review "
        "(a wrong intent, a near miss and a greeting). The stolen-card message was the only confident mistake, and it "
        "is labelled as a known failure in the demo. Right and wrong here are my reading, since these messages "
        f"have no official label. Once loaded, each request took {S['warm_latency_ms']['min']}-"
        f"{S['warm_latency_ms']['max']} ms on the MacBook CPU.</p>")

# ---------------------------------------------------------------- 07 interpretation
ui.section("07", "What the results show, and what they don't")
s1, s2 = st.columns(2, gap="large")
s1.html('<div class="mz-h3">Supported by this evaluation</div><ul class="mz-ticks">'
        "<li>On this dataset, a fine-tuned 543 MB Arabic model clearly beats a zero-shot 1.5B-parameter LLM and a TF-IDF baseline.</li>"
        "<li>About a thousand labelled examples were enough for a usable intent classifier across 77 intents.</li>"
        "<li>The fine-tuned model performed similarly on MSA and Palestinian dialect.</li>"
        "<li>A simple score threshold can raise accuracy on auto-handled queries to about 89%, at the cost of reviewing about a quarter.</li></ul>")
s2.html('<div class="mz-h3 ">Not established</div><ul class="mz-ticks mz-amber">'
        "<li>That fine-tuning beats LLMs in general. Only one small LLM was tested, zero-shot, with one prompt. Larger "
        "models, few-shot prompts or fine-tuned LLMs were not tried.</li>"
        "<li>Performance on Gulf dialects (Emirati, Saudi) or real bank traffic. Neither was in the test data.</li>"
        "<li>Statistical certainty. Each model was trained once on a 370-query test set, with no confidence intervals or "
        "repeated seeds.</li>"
        "<li>That model scores are calibrated probabilities. Calibration was not measured, and live testing "
        "found a wrong answer scored 0.98.</li></ul>")
ui.callout("<b>Published context.</b> SinaLab report an F1 of about 92% (MSA) and 90% (Palestinian) for an AraBERT "
           f"model trained on the <i>full</i> ArBanking77 corpus (<a class='mz-a' href='{C.PAPER_URL}' target='_blank' "
           "rel='noopener'>paper</a>). Mizan used only the public sample, so the numbers are not directly comparable; "
           "the gap suggests what more labelled data could add.")

ui.footer()