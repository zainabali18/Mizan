"""Mizan live demo (Streamlit).

Local:   streamlit run demo/streamlit_app.py
Cloud:   Streamlit Community Cloud, main file path = demo/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `mizan` importable on Streamlit Cloud
from mizan.service import MizanClassifier  # noqa: E402

st.set_page_config(page_title="Mizan · ميزان", page_icon="⚖️", layout="centered")

EXAMPLES = {
    "Stolen card + personal data (dialect)": "بطاقتي 4111 1111 1111 1111 انسرقت امبارح، رقم جوالي 0501234567",
    "Card hasn't arrived (MSA)": "لقد طلبت بطاقة جديدة منذ أسبوعين ولم تصل بعد",
    "ATM won't give cash (dialect)": "ليش ماكينة الصراف الآلي ما بتعطيني مصاري؟",
    "Top up with Apple Pay (MSA)": "هل يمكنني تعبئة رصيدي باستخدام آبل باي؟",
    "Transfer with an IBAN": "حولت مصاري على IBAN AE07 0331 2345 6789 0123 456 ولسا ما وصلت",
    "Emirates ID verification": "رقم هويتي 784-1990-1234567-1 بس التطبيق ما بقبل يتحقق من هويتي",
    "Vague message (goes to a human)": "مرحبا",
}

st.markdown(
    """
    <style>
      .rtl {direction: rtl; text-align: right; font-size: 1.05rem;}
      textarea {direction: rtl; text-align: right;}
      .pill {display:inline-block; padding:2px 10px; margin:2px; border-radius:999px;
             background:#eef2f7; font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the Arabic model (first run only, about 30 seconds)…")
def load_classifier() -> MizanClassifier:
    clf = MizanClassifier()
    clf.warm_up()
    return clf


st.title("Mizan · ميزان")
st.markdown(
    "**Sovereign Arabic intent classification for banking support.** Type a customer message in "
    "Modern Standard Arabic or dialect. Mizan **removes personal data** (Emirates ID, IBAN, card, phone, email), "
    "classifies it into **one of 77 banking intents** with a fine-tuned open-source AraBERT model running on **CPU**, "
    "and **routes low-confidence requests to a human**."
)
st.caption(
    "[GitHub](https://github.com/zainabali18/Mizan) · "
    "[Model](https://huggingface.co/zainabali18/mizan-arabert-arbanking77) · "
    "76.2% macro-F1 on held-out test data, against 7.4% for a general open-source LLM"
)

clf = load_classifier()

choice = st.selectbox("Try an example", ["(write your own)", *EXAMPLES])
default = EXAMPLES.get(choice, "")
text = st.text_area("Customer message", value=default, height=100, placeholder="اكتب رسالة العميل هنا…")

if st.button("Classify", type="primary", width="stretch") and text.strip():
    r = clf.classify(text)

    st.subheader("Result")
    st.markdown(f"<div class='rtl'><b>{r['intent']}</b></div>", unsafe_allow_html=True)
    if r["route"] == "auto":
        st.success(f"✅ Handled automatically: confidence {r['confidence']:.0%} ≥ threshold {r['threshold']:.0%}")
    else:
        st.warning(f"🧑‍💼 Routed to a human agent: confidence {r['confidence']:.0%} < threshold {r['threshold']:.0%}")

    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{r['confidence']:.0%}")
    c2.metric("Response time (CPU)", f"{r['latency_ms']:.0f} ms")

    st.markdown("**What the model saw** (personal data removed first)")
    st.markdown(f"<div class='rtl'>{r['redacted_text']}</div>", unsafe_allow_html=True)
    if r["pii_detected"]:
        st.markdown("".join(f"<span class='pill'>🔒 {k}</span>" for k in r["pii_detected"]), unsafe_allow_html=True)
    else:
        st.caption("No personal data detected.")

    st.markdown("**Top 3 intents**")
    for t in r["top_intents"]:
        st.markdown(f"<div class='rtl'>{t['intent']}: {t['confidence']:.0%}</div>", unsafe_allow_html=True)
        st.progress(min(max(t["confidence"], 0.0), 1.0))

with st.expander("How it works"):
    st.markdown(
        """
1. **PII redaction.** Emirates IDs, IBANs, card numbers (Luhn-checked), phone numbers and emails are replaced
   with tags *before* the model sees the text. The raw text is never returned or stored.
2. **Classification.** AraBERT (open-source) is fine-tuned on 1,078 Arabic banking queries (ArBanking77) and runs on CPU.
3. **Human handoff.** If confidence is below 85% (chosen on validation data), the request goes to a person.
   On the test set this auto-handles **74%** of requests at **89%** accuracy.

| Model | Macro-F1 | Size | Latency |
|---|---|---|---|
| Zero-shot LLM (Qwen2.5-1.5B) | 7.4% | 3.1 GB | 350 ms (GPU) |
| TF-IDF + logistic regression | 61.0% | 10 MB | 0.4 ms (CPU) |
| **Fine-tuned AraBERT (Mizan)** | **76.2%** | 0.54 GB | 104 ms (CPU) |
"""
    )
st.caption("Demo only. Example card, IBAN and ID numbers are fictional test values.")
