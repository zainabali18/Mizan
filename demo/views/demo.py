import streamlit as st

from webui import components as ui
from webui import content as C

H = C.headline()
MAX_CHARS = 1000

ui.page_header(
    "Live demo",
    "Classify an Arabic banking message",
    "Write a message as a customer would, in Modern Standard Arabic or dialect, or start from an example. "
    "The deployed model processes it in three visible stages.",
)

# ---- state ----
st.session_state.setdefault("message", "")
st.session_state.setdefault("last", None)          # (original_text, result) of the latest successful run
st.session_state.setdefault("runs", 0)


def _use_example() -> None:
    title = st.session_state.get("examples")
    match = next((e for e in C.EXAMPLES if e["title"] == title), None)
    if match:
        st.session_state["message"] = match["text"]


left, right = st.columns([1, 1.12], gap="large")

with left:
    st.html('<div class="mz-h3">1 · Customer message</div>'
            '<p class="mz-muted" style="margin:0 0 .4rem">Pick an example to fill the box, or type your own.</p>')
    st.pills("Examples", [e["title"] for e in C.EXAMPLES], key="examples", label_visibility="collapsed",
             on_change=_use_example, width="stretch", wrap=True)
    with st.form("classify", border=False, enter_to_submit=False):
        st.text_area(
            "Customer message (Arabic)", key="message", height=130, max_chars=MAX_CHARS,
            placeholder="اكتب رسالة العميل هنا… مثال: بطاقتي ما وصلت لحد الآن",
            help="Right-to-left Arabic input. Up to 1,000 characters; the model reads the first 64 tokens.",
        )
        submitted = st.form_submit_button("Classify message", type="primary", width="stretch")

    st.html(
        '<div class="mz-small" style="margin-top:.6rem;line-height:1.6">'
        "<b style='color:var(--ink-2)'>Good inputs:</b> a single banking request about cards, transfers, top-ups, "
        "payments, cash withdrawals, exchange, identity checks or account changes.<br>"
        "<b style='color:var(--ink-2)'>Personal data:</b> use fictional values only. Emirates ID, IBAN, card, phone "
        "and email patterns are redacted before classification; other details (such as names) are <i>not</i>.<br>"
        "<b style='color:var(--ink-2)'>Scope:</b> the model was trained on 77 banking intents only, so any message is "
        "mapped to one of them. Off-topic text still receives a banking label, so check the score and the review flag.</div>"
    )
    with st.expander("What each example tests"):
        for e in C.EXAMPLES:
            st.html(f'<div style="padding:.35rem 0;border-bottom:1px solid var(--line-soft)">'
                    f'<div class="mz-ar" lang="ar" style="font-size:1.02rem">{ui.esc(e["text"])}</div>'
                    f'<div class="mz-small">{ui.esc(e["title"])} · {ui.esc(e["tag"])}</div>'
                    + (f'<div class="mz-small" style="color:var(--amber)">{ui.esc(e["note"])}</div>' if e.get("note") else "")
                    + '</div>')

with right:
    st.html('<div class="mz-h3">2 · Result</div>'
            '<p class="mz-muted" style="margin:0 0 .4rem">Each stage below is real output from the deployed model.</p>')
    result_area = st.container()

with result_area:
    if submitted:
        text = st.session_state["message"]
        if not text.strip():
            st.warning("Please enter a message first: choose an example or type a sentence in Arabic.")
        else:
            if not C.has_arabic(text):
                st.info("This message contains no Arabic. Mizan was trained only on Arabic, so the result is "
                        "unlikely to be meaningful.")
            try:
                with st.spinner("Loading the model (first visit only, up to about 30 seconds)…"
                                if st.session_state["runs"] == 0 else "Classifying…"):
                    r = ui.get_classifier().classify(text)
                st.session_state["last"] = (text, r)
                st.session_state["runs"] += 1
            except Exception as exc:  # noqa: BLE001 - shown to the user, with detail available
                st.session_state["last"] = None
                st.error("Something went wrong while classifying this message. Please try again in a moment.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

    if st.session_state["last"]:
        original, r = st.session_state["last"]
        st.html(ui.result_panel(r, original, "Result from the deployed model"))
        known = next((e for e in C.EXAMPLES if e["text"] == original and e.get("note")), None)
        if known:
            ui.callout(f"<b>Known failure.</b> {ui.esc(known['note'])} A high score is not proof of a right "
                       "answer: see Results, section 06.", warn=True)
        if st.session_state["message"] != original:
            st.caption("The message box has changed since this result. Select Classify message to update it.")
    elif not submitted:
        ui.empty_state()

ui.section("Guide", "Reading the result")
a, b, c = st.columns(3, gap="large")
a.html('<div class="mz-h3">Model score</div><p class="mz-p">The highest softmax output of the classifier, between 0 and 1. '
       "Higher means the model favours one intent more strongly. It has <b>not been calibrated</b>, so 0.90 does "
       "not mean a 90% chance of being correct, and the model can be confidently wrong.</p>")
b.html(f'<div class="mz-h3">Review threshold ({H["threshold"]:.2f})</div><p class="mz-p">Chosen on the validation set as '
       "the lowest score at which auto-handled answers were at least 90% accurate. On the test set, "
       f"{C.pct(H['auto_share'])} of messages cleared it, and those were {C.pct(H['auto_acc'])} accurate.</p>")
c.html('<div class="mz-h3">What the demo does not do</div><p class="mz-p">It does not reply to the customer, connect to '
       "a bank, or send anything to a review queue. The routing decision is shown, not acted on. Messages are "
       "processed in memory and not stored by the app.</p>")

ui.footer()