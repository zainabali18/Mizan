import streamlit as st

from webui import components as ui
from webui import content as C

H = C.headline()

ui.page_header(
    "Case study",
    "Building Mizan: decisions, trade-offs and what testing changed",
    "Mizan was designed, built and evaluated by Zainab Ali over one day in October 2026, as a focused demonstration of applied "
    "Arabic NLP: data, evaluation, modelling and a working application.",
)

ui.section("01", "Goal")
g1, g2 = st.columns([1.2, 1], gap="large")
g1.html('<p class="mz-p">Build a credible, honest prototype of an Arabic customer-support component for a regulated '
        "industry: one that classifies requests accurately in formal Arabic and dialect, protects personal data, "
        "and knows when to defer to a person. Then measure whether a small specialist model is the right tool, "
        "rather than assuming a large LLM is.</p>")
g2.html('<div class="mz-h3">What success meant</div><ul class="mz-ticks">'
        "<li>A fair comparison against at least two baselines on held-out data</li>"
        "<li>Results reported separately for MSA and dialect</li>"
        "<li>Personal data removed before any model sees the text</li>"
        "<li>A working, publicly accessible demo backed by tested code</li></ul>")

ui.section("02", "What I did")
ui.table(["Stage", "Work", "Evidence in the repository"], [
    ["<b>Data</b>", "Loaded the ArBanking77 public sample; checked for missing values, duplicates and overlap between "
     "splits; tagged the test set as MSA or dialect; wrote Arabic normalisation and UAE-focused PII redaction.",
     "<span class='mz-mono'>mizan/data.py, pii.py, normalise.py</span>; tests"],
    ["<b>Baselines</b>", "Trained a TF-IDF character n-gram classifier, tuned on validation. Benchmarked a zero-shot "
     "open-source LLM, timing one query at a time.", "<span class='mz-mono'>baseline_tfidf.py</span>, notebook 02, results JSON"],
    ["<b>Model</b>", "Fine-tuned AraBERT on Colab, chose the epoch on validation, published the model to Hugging Face.",
     "Notebook 03, model card, training log"],
    ["<b>Safety</b>", "Chose a human-review threshold on validation data and checked it on the test set.",
     "<span class='mz-mono'>mizan_config.json</span>"],
    ["<b>Engineering</b>", "Built a classifier service and FastAPI endpoint, plus "
     "a test suite that runs on every push (the model is mocked in CI). Built and tested a Docker image, including a run with networking switched off.",
     "<span class='mz-mono'>service.py, api.py, Dockerfile</span>, GitHub Actions"],
    ["<b>Product</b>", "Designed and deployed this site and live demo on Streamlit Community Cloud.",
     "<span class='mz-mono'>demo/</span>"],
])

ui.section("03", "Key decisions and trade-offs")
ui.table(["Decision", "Why", "Trade-off accepted"], [
    ["Macro-F1 as the headline metric", "77 intents with few examples each; accuracy alone would favour common intents.",
     "Less intuitive than accuracy, so both are reported."],
    ["Benchmark a zero-shot LLM before fine-tuning", "Test the common assumption that a general LLM is enough.",
     "Only one small LLM and one prompt were tested."],
    ["Fine-tune a mid-sized Arabic encoder (AraBERT)", "Arabic pretraining, runs on CPU, open weights a bank could self-host.",
     "Cannot generate replies or handle intents it wasn't trained on; 543 MB to host."],
    ["Redact before tokenising, with a Luhn check for cards", "The model never sees identifiers, and amounts aren't mistaken for cards.",
     "Pattern-based: names and addresses aren't covered."],
    ["No letter normalisation for AraBERT (kept for TF-IDF)", "AraBERT was pretrained on natural Arabic spelling.",
     "A design judgement; not compared experimentally."],
    ["Choose the threshold on validation, never on test", "Keeps the test set an honest estimate.",
     "The validation set is small (124 queries), so the threshold is approximate."],
    ["Streamlit Community Cloud for the demo", "Free, deploys from GitHub, runs the tested Python code unchanged.",
     "Third-party hosting, so the demo itself isn't a sovereign deployment (see How it works)."],
])

ui.section("04", "What testing showed, and what it changed")
st.html(
    '<ol class="mz-list">'
    f"<li><b>The LLM baseline confirmed the case for fine-tuning.</b> A zero-shot 1.5B model scored {C.pct(H['llm_f1'])} "
    f"macro-F1, below even TF-IDF ({C.pct(H['tfidf_f1'])}). Its mistakes were plausible but wrong neighbouring intents, "
    "which intent names alone cannot separate. Unparseable answers (40 of 370) were counted as wrong, not dropped.</li>"
    "<li><b>Live testing found a confident mistake.</b> Run through the API with the real model, a stolen-card message "
    "that also gave a phone number was classified as <i>lost or stolen phone</i> with a score of 0.98. Removing parts of "
    "the message one at a time showed two causes: the word \"phone\" overrides \"card\", and redaction tags such as "
    "<span class='mz-mono'>[CARD_NUMBER]</span> lower the score because the model never saw them in training. The example "
    "was taken off the front page, labelled as a known failure in the demo, and written up on the Results page.</li>"
    f"<li><b>The threshold landed just under its target.</b> Chosen for 90% accuracy on validation, it gave "
    f"{C.pct(H['auto_acc'])} on test. With only 124 validation queries, it is presented as approximate, and more "
    "data is listed as a next step.</li>"
    "<li><b>A library upgrade broke training.</b> The newest transformers version removed a training option. Running the "
    "whole notebook first with a tiny stand-in model caught it, and the notebook was rewritten to work across versions "
    "before any GPU time was used.</li>"
    "<li><b>Arabic text displayed numbers in the wrong order.</b> Browser testing of this site showed card and IBAN digit "
    "groups reversed inside right-to-left text. They are now isolated so they read correctly.</li>"
    "<li><b>Hosting changed after a platform restriction.</b> The planned Gradio demo on Hugging Face Spaces became a "
    "paid feature, so the demo moved to Streamlit Community Cloud with no change to the model code.</li></ol>"
)

ui.section("05", "Status: built, measured, proposed")
st.html(
    '<div class="mz-status">'
    '<div><h4>Implemented and tested</h4><ul class="mz-ticks">'
    "<li>Data pipeline, Arabic normalisation, PII redaction (unit tests)</li>"
    "<li>Three-way benchmark on the test set</li>"
    "<li>Fine-tuned AraBERT, published</li><li>Validation-chosen review threshold</li>"
    "<li>Classifier service and FastAPI (unit tests, and checked by hand with the real model)</li><li>Live demo (this site)</li>"
    "<li>Docker image: built on Apple silicon, same results as the local API, non-root, health check passing, and working with networking switched off</li>"
    "<li>Per-intent and per-example analysis of all 370 test predictions</li></ul></div>"
    '<div class="mz-col-grey"><h4>Proposed next steps</h4><ul class="mz-ticks mz-grey">'
    "<li>Evaluate on Gulf dialects (Emirati, Saudi) before any UAE use</li>"
    "<li>Train on the full ArBanking77 corpus</li>"
    "<li>Measure and calibrate the model score</li>"
    "<li>Train with fictional personal data inserted and redacted, so the model learns the tags</li>"
    "<li>Compare a LoRA fine-tuned small LLM</li>"
    "<li>Add a trained PII detector for names and addresses</li>"
    "<li>Connect the review flag to a real queue, with monitoring of review rate and drift</li></ul></div></div>"
)

ui.section("06", "Limitations")
st.html('<ul class="mz-ticks mz-amber" style="max-width:52rem">'
        "<li>Small data: about 14 training examples per intent, and a 370-query test set covering 75 of the 77 intents.</li>"
        "<li>One training run per model and no confidence intervals, so differences of a few points may not be meaningful.</li>"
        "<li>Only MSA and Palestinian dialect were tested; the examples come from a UK-style digital bank translated into Arabic.</li>"
        "<li>The model always picks one of 77 intents; there is no explicit \"out of scope\" class.</li>"
        "<li>It can be confidently wrong: a mention of a phone number pulled a stolen-card message to <i>lost or stolen phone</i> at 0.98.</li>"
        "<li>A demonstration: no customer replies, bank integrations, review queue, access control or audit trail.</li></ul>")

ui.footer()