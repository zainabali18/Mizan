import streamlit as st

from mizan.pii import redact
from webui import components as ui
from webui import content as C

H = C.headline()
EXAMPLE = C.PREVIEW_EXAMPLE["text"]

hero_text, hero_preview = st.columns([1.15, 1], gap="large")

with hero_text:
    st.html(
        '<p class="mz-eyebrow">Arabic NLP · Banking support · Portfolio project</p>'
        '<h1 class="mz-h1">Understanding Arabic banking requests, and knowing when to ask a human.</h1>'
        '<p class="mz-lede">Mizan reads a customer message written in Modern Standard Arabic or dialect, '
        "removes personal details such as card and Emirates ID numbers, and predicts which of "
        f"<b>77 banking intents</b> it belongs to. When the model's score falls below a set threshold, "
        "the request is flagged for a person instead of being answered automatically.</p>"
    )
    c1, c2 = st.columns(2)
    if c1.button("Try the live demo", type="primary", width="stretch"):
        st.switch_page("views/demo.py")
    if c2.button("Explore the results", width="stretch"):
        st.switch_page("views/results.py")
    st.html(ui.links([("Code on GitHub", C.REPO_URL), ("Model on Hugging Face", C.MODEL_URL),
                      ("Dataset: ArBanking77", C.DATASET_URL)]))

preview_slot = hero_preview.container()   # filled at the end so the page renders before the model loads

st.html('<div style="height:2.2rem"></div>')
ui.metric_strip([
    (C.pct(H["arabert_f1"]), "Macro-F1, fine-tuned AraBERT",
     f"{H['test_n']} held-out test queries across 77 intents"),
    (C.pct(H["llm_f1"]), "Macro-F1, zero-shot open-source LLM",
     "Qwen2.5-1.5B on the same test set, without training"),
    (f'{C.pct(H["auto_share"])}<small> at {C.pct(H["auto_acc"])} acc.</small>', "Handled automatically",
     f"Score ≥ {H['threshold']:.2f}; the other {C.pct(H['review_share'])} are flagged for review"),
    (f'{H["arabert_ms"]:.0f}<small> ms</small>', "Per query on CPU",
     f"One query at a time; {H['arabert_mb']:.0f} MB model, no GPU needed"),
])
st.caption("All figures come from the recorded test-set evaluation. See Results for setup and limitations.")

ui.section("01", "The problem")
p1, p2 = st.columns(2, gap="large")
p1.html(
    '<p class="mz-p">A bank\'s support channel receives the same request phrased many ways. In Arabic, customers '
    "move between formal Modern Standard Arabic and their own dialect, so <i>\"my card hasn't arrived\"</i> "
    "can look quite different from one message to the next. Routing each message to the right answer or team "
    "starts with recognising its intent.</p>"
    '<p class="mz-p">Banking messages also contain personal data. Card numbers, IBANs and ID numbers should not '
    "travel further into a system than they need to.</p>"
)
p2.html(
    '<p class="mz-p">A general-purpose large language model is the obvious first idea. But it is large to host, '
    "and nothing guarantees it can separate 77 closely related banking intents from their names alone.</p>"
    '<div class="mz-callout"><b>The question Mizan tests:</b> can a small open-source Arabic model, fine-tuned on '
    "about a thousand labelled examples and run on CPU, classify these requests better than a general LLM, "
    "while removing personal data first and deferring to a human when unsure?</div>"
)

ui.section("02", "What I built")
st.html(
    '<ol class="mz-list">'
    "<li><b>A data pipeline with personal-data redaction for UAE formats.</b> Emirates ID, IBAN, Luhn-checked card "
    "numbers, phone numbers and emails are replaced with tags before any model sees the text, alongside Arabic "
    "normalisation and leakage checks across the dataset splits.</li>"
    "<li><b>A fair three-way benchmark</b> on the same 370 held-out queries: a TF-IDF classifier, a zero-shot "
    "open-source LLM, and a fine-tuned AraBERT. It reports macro-F1 overall and separately for MSA and dialect.</li>"
    "<li><b>A fine-tuned Arabic model, published openly</b> on Hugging Face, trained in 23 minutes on a free GPU.</li>"
    "<li><b>A human-review threshold chosen on validation data,</b> then checked on the test set: answers that clear "
    f"it were {C.pct(H['auto_acc'])} accurate.</li>"
    "<li><b>A tested service around the model:</b> a Python classifier and FastAPI endpoint, 65 automated tests in "
    "GitHub Actions, and this site.</li></ol>"
)

ui.section("03", "Explore the project")
explore = st.container(key="explore")
e1, e2 = explore.columns(2, gap="large")
with e1:
    st.page_link("views/demo.py", label="Live demo")
    st.caption("Type an Arabic message and watch it pass through redaction, classification and the review threshold.")
    st.page_link("views/results.py", label="Results and evaluation")
    st.caption("The benchmark, metric definitions, training curve, error analysis and what the results do not show.")
with e2:
    st.page_link("views/how_it_works.py", label="How it works")
    st.caption("The pipeline stage by stage, redaction coverage and limits, and where data goes.")
    st.page_link("views/case_study.py", label="Case study")
    st.caption("Goals, decisions and trade-offs, what changed after testing, and what's next.")

ui.footer()

# ---- live preview (runs last so everything above is already visible) ----
with preview_slot:
    try:
        with st.spinner("Loading the Arabic model for a live preview…"):
            r = ui.classify_example(EXAMPLE)
        st.html(ui.result_panel(r, EXAMPLE, "Live output from the deployed model", compact=True))
        st.caption("A sample message. Try your own, including fictional card or ID numbers, in the live demo.")
    except Exception:  # model unavailable: show the real redaction step only
        redacted, _ = redact(EXAMPLE)
        st.html(
            '<div class="mz-panel"><div class="mz-panel-bar"><span>Preview (model not loaded)</span><span>Mizan · ميزان</span></div>'
            f'<div class="mz-flow"><div class="mz-step"><div class="mz-step-n">1</div><div><div class="mz-step-label">Redacted message</div>'
            f'<div class="mz-msg mz-ar" lang="ar">{C.highlight_redactions(redacted)}</div></div></div></div></div>')
        st.caption("The model could not be loaded right now. Please try the live demo again shortly.")