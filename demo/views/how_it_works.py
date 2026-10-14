import pandas as pd
import streamlit as st

from webui import components as ui
from webui import content as C

H = C.headline()
CFG = C.config()

ui.page_header(
    "How it works",
    "From a customer message to a routing decision",
    "Six stages, all implemented in the open-source <span class='mz-mono'>mizan</span> Python package. The same code "
    "runs in the test suite, the API and this site.",
)

STAGES = [
    ("01", "Message", "Customer text arrives, Arabic or mixed with English. Up to 1,000 characters in the API.", "Browser → app server", False),
    ("02", "Redact", "Emirates ID, IBAN, card, phone and email patterns become tags like [CARD_NUMBER].", "mizan/pii.py", True),
    ("03", "Clean", "Arabic-Indic digits become 0-9 and whitespace is tidied. Letters stay natural for AraBERT.", "mizan/service.py", False),
    ("04", "Tokenise", "The AraBERT tokenizer splits text into sub-words. The first 64 tokens are used.", "Hugging Face transformers", False),
    ("05", "Classify", "Fine-tuned AraBERT scores all 77 intents (softmax). This runs on CPU, in the app's own process.", "mizan/service.py", True),
    ("06", "Decide", f"Top score ≥ {H['threshold']:.2f} means eligible for automatic handling; otherwise it is flagged for human review.", "mizan_config.json", True),
]
st.html('<div class="mz-pipe" role="list" aria-label="Mizan pipeline">' + "".join(
    f'<div class="mz-pipe-stage{" mz-key" if key else ""}" role="listitem"><div class="mz-pipe-n">{n}</div>'
    f'<div class="mz-pipe-title">{t}</div><div class="mz-pipe-desc">{ui.esc(d)}</div>'
    f'<div class="mz-pipe-where">{ui.esc(w)}</div></div>' for n, t, d, w, key in STAGES) + "</div>")
st.html('<p class="mz-fig-cap">Highlighted stages are the project\'s own logic. Output: intent, score, top three '
        "alternatives, redacted text, the personal-data types found, and the routing decision.</p>")

# ---------------------------------------------------------------- 01 model
ui.section("01", "Why fine-tune AraBERT?")
m1, m2 = st.columns(2, gap="large")
m1.html(
    f'<p class="mz-p"><a class="mz-a" href="{C.BASE_MODEL_URL}" target="_blank" rel="noopener">AraBERT v0.2</a> is an '
    "open-source BERT model pretrained on a large Arabic corpus by AUB MIND Lab. It already represents Arabic words "
    "and sentence structure well, so it only needs to learn the mapping from messages to intents.</p>"
    '<p class="mz-p">A small classification layer (77 outputs) was added on top and the whole model was trained for '
    f"20 epochs on 1,078 labelled queries, in {C.training_log()['training_minutes']:.0f} minutes on a free Colab GPU. "
    "The resulting weights are published openly.</p>")
m2.html(
    '<div class="mz-h3">The trade-offs</div><ul class="mz-ticks">'
    f"<li><b>Against a general LLM:</b> {H['llm_mb'] / H['arabert_mb']:.1f}× smaller, runs on CPU, and much more "
    "accurate on this task. But it can only choose among the 77 intents it was trained on and cannot write replies.</li>"
    "<li><b>Against TF-IDF:</b> 15 points more macro-F1, but slower (about 104 ms on CPU, against under 1 ms) and "
    "about 50× larger.</li>"
    "<li><b>Dialect:</b> trained on mixed MSA and Palestinian dialect, and tested on both.</li></ul>")

# ---------------------------------------------------------------- 02 intents
ui.section("02", "How the 77 intents are represented",
           "Each intent is a class with an integer id, an Arabic name from ArBanking77, and the original English "
           "Banking77 name it was translated from. The model outputs one score per id.")
df = pd.DataFrame(C.intent_table())
q = st.text_input("Search intents", placeholder="Search in English or Arabic, e.g. card, transfer, تحويل",
                  label_visibility="collapsed")
if q:
    mask = df.apply(lambda r: q.lower() in r.english.lower() or q in r.arabic or q.lower() in r.banking77, axis=1)
    df = df[mask]
st.dataframe(df.rename(columns={"id": "Id", "arabic": "Arabic intent (as trained)", "english": "English (Banking77)",
                                "banking77": "Banking77 label", "train_examples": "Training examples"}),
             hide_index=True, height=320,
             column_order=["Id", "Arabic intent (as trained)", "English (Banking77)", "Training examples"])
st.html(f'<p class="mz-fig-cap">{len(df)} intents shown. Training examples per intent range from '
        f'{min(r["train_examples"] for r in C.intent_table())} to {max(r["train_examples"] for r in C.intent_table())}.</p>')

# ---------------------------------------------------------------- 03 PII
ui.section("03", "Personal-data redaction: coverage and limits",
           "Pattern-based redaction runs before tokenisation, so the model never receives these values, and the "
           "service returns only the redacted text.")
ui.table(["Type", "What is matched", "Replaced with"],
         [[f"<b>{ui.esc(name)}</b>", ui.esc(desc), f"<span class='mz-mono'>[{k}]</span>"] for k, name, desc in C.PII_TYPES])
r1, r2 = st.columns(2, gap="large")
r1.html('<div class="mz-h3">How it is checked</div><ul class="mz-ticks">'
        "<li>Unit tests cover each type, including Arabic-Indic digits, spaced and dashed formats, and several items in one message.</li>"
        "<li>Ordinary numbers (amounts, counts, years) are tested to make sure they are <i>not</i> redacted. Card numbers must "
        "pass the Luhn checksum.</li>"
        "<li>On all 1,572 dataset queries the redactor changed nothing. That shows no false positives on normal banking text, "
        "but the dataset contains no real personal data, so it does not measure recall.</li></ul>")
r2.html('<div class="mz-h3">What it does not catch</div><ul class="mz-ticks mz-amber">'
        "<li>Names, addresses, dates of birth, passport numbers, or account numbers that are not IBANs.</li>"
        "<li>Formats beyond the tested patterns, for example numbers spelled out in words.</li>"
        "<li>It is a safeguard, not a guarantee. A production system would add a trained PII detector and audit logging.</li></ul>")

# ---------------------------------------------------------------- 04 threshold
ui.section("04", "The review threshold")
st.html(
    '<p class="mz-p" style="max-width:48rem">After training, the model scored every <b>validation</b> query. Thresholds from '
    "0.30 to 0.95 were tried in steps of 0.05, and the lowest one at which the queries above it were at least 90% "
    f"accurate was kept: <b>{H['threshold']:.2f}</b>. It is stored in "
    "<span class='mz-mono'>mizan_config.json</span> and applied to every request. On the test set, "
    f"{C.pct(H['auto_share'])} of queries cleared it, at {C.pct(H['auto_acc'])} accuracy. The score is the model's "
    "top softmax output. It is useful for ranking confidence, but it is <b>not a calibrated probability</b>.</p>")

# ---------------------------------------------------------------- 05 where data goes
ui.section("05", "Where inference runs, and where data goes",
           "The architecture is designed so a bank could host it itself. This public demo is hosted by a third party.")
ui.table(["", "This public demo", "Intended self-hosted deployment"], [
    ["<b>Hosting</b>", "Streamlit Community Cloud (third-party platform)", "A bank's own server, from the project's Dockerfile"],
    ["<b>Model weights</b>", "Downloaded once from Hugging Face when the app starts",
     "Downloaded at image build time; the container then runs with Hugging Face offline mode on"],
    ["<b>Inference</b>", "On CPU inside the app process. No third-party AI API is called.", "Same code, same CPU inference"],
    ["<b>Your message</b>", "Sent from your browser to the app server and processed in memory. The app does not store or log it; "
     "the hosting platform's own policies apply.", "Never leaves the bank's network"],
    ["<b>Status</b>", "Live", "Dockerfile written, <b>not yet built or tested</b>"],
])
cpu1, cpu2 = st.columns(2, gap="large")
cpu1.html('<div class="mz-h3">Running on CPU</div><p class="mz-p">One query took about '
          f"{H['arabert_ms']:.0f} ms on a Colab CPU, comfortably interactive for a chat channel and with no GPU cost. "
          "The demo server's CPU is shared and may be slower; each result shows its own measured time. The "
          f"{H['arabert_mb']:.0f} MB model is loaded once per server process and reused for every request.</p>")
cpu2.html('<div class="mz-h3">Also in the repository</div><ul class="mz-ticks">'
          "<li>A FastAPI service with <span class='mz-mono'>/classify</span>, <span class='mz-mono'>/intents</span> and "
          "<span class='mz-mono'>/health</span>, unit-tested with a stand-in model and checked by hand with the real one.</li>"
          "<li>A TF-IDF baseline that trains in seconds, a possible fallback.</li>"
          "<li>Notebooks for the LLM benchmark and the fine-tuning run.</li></ul>")

ui.footer()