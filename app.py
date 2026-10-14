"""Mizan live demo (Gradio). Runs locally with `python app.py` or on Hugging Face Spaces."""
from __future__ import annotations

import gradio as gr

from mizan.service import MizanClassifier

clf = MizanClassifier()

EXAMPLES = [
    "بطاقتي 4111 1111 1111 1111 انسرقت امبارح، رقم جوالي 0501234567",   # PAL + card + phone
    "لقد طلبت بطاقة جديدة منذ أسبوعين ولم تصل بعد",                      # MSA: card arrival
    "ليش ماكينة الصراف الآلي ما بتعطيني مصاري؟",                          # PAL: ATM declined
    "هل يمكنني تعبئة رصيدي باستخدام آبل باي؟",                             # MSA: Apple Pay top-up
    "حولت مصاري على IBAN AE07 0331 2345 6789 0123 456 ولسا ما وصلت",       # IBAN redaction
    "كم عمر أطفالي لازم يكون عشان يفتحوا حساب؟",                            # age limit
    "مرحبا",                                                                # vague -> human
]


def run(text: str):
    if not text or not text.strip():
        return "Please type a message.", "", {}
    r = clf.classify(text)
    if r["route"] == "auto":
        badge = f"✅ **Handled automatically**: confidence {r['confidence']:.0%} ≥ threshold {r['threshold']:.0%}"
    else:
        badge = f"🧑‍💼 **Routed to a human agent**: confidence {r['confidence']:.0%} < threshold {r['threshold']:.0%}"
    pii = ", ".join(r["pii_detected"]) if r["pii_detected"] else "none"
    summary = (
        f"### Intent: {r['intent']}\n\n{badge}\n\n"
        f"**Personal data removed before the model saw the message:** {pii}  \n"
        f"**Response time:** {r['latency_ms']:.0f} ms on CPU"
    )
    scores = {t["intent"]: t["confidence"] for t in r["top_intents"]}
    return summary, r["redacted_text"], scores


with gr.Blocks(title="Mizan · ميزان") as demo:
    gr.Markdown(
        "# Mizan · ميزان\n"
        "**Sovereign Arabic intent classification for banking support.** Type a customer message in "
        "Modern Standard Arabic or dialect. Mizan removes personal data (Emirates ID, IBAN, card, phone, email), "
        "classifies it into one of 77 banking intents with a fine-tuned open-source AraBERT model running on CPU, "
        "and routes low-confidence requests to a human.  \n"
        "[GitHub](https://github.com/zainabali18/Mizan) · "
        "[Model](https://huggingface.co/zainabali18/mizan-arabert-arbanking77)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Customer message", lines=3, rtl=True, placeholder="اكتب رسالة العميل هنا…")
            btn = gr.Button("Classify", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            out_md = gr.Markdown()
            out_red = gr.Textbox(label="What the model saw (PII redacted)", rtl=True, interactive=False)
            out_lab = gr.Label(label="Top intents", num_top_classes=3)
    btn.click(run, inp, [out_md, out_red, out_lab])
    inp.submit(run, inp, [out_md, out_red, out_lab])

if __name__ == "__main__":
    demo.launch()
