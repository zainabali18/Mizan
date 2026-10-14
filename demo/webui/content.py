"""Evidence and content for the site.

Every number shown on the site is read from files in the repository
(results/*.json, results/evidence/*.json, mizan_config.json, data/processed/*)
so the site cannot drift from the recorded experiments. Nothing here imports
Streamlit, so it is unit-testable.
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
EVIDENCE = RESULTS / "evidence"
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "mizan_config.json"
PREDICTIONS = RESULTS / "arabert_test_predictions.csv"   # optional, from notebooks/04

REPO_URL = "https://github.com/zainabali18/Mizan"
MODEL_URL = "https://huggingface.co/zainabali18/mizan-arabert-arbanking77"
BASE_MODEL_URL = "https://huggingface.co/aubmindlab/bert-base-arabertv02"
DATASET_URL = "https://huggingface.co/datasets/SinaLab/ArBanking77"
PAPER_URL = "https://aclanthology.org/2023.arabicnlp-1.22"
LINKEDIN_URL = "https://www.linkedin.com/in/zainab-ali-286058260"

MODEL_KEYS = {
    "TF-IDF + logistic regression": "tfidf",
    "Zero-shot LLM (Qwen2.5-1.5B-Instruct)": "llm",
    "Fine-tuned AraBERT": "arabert",
}
SHORT_NAMES = {"tfidf": "TF-IDF + logistic regression", "llm": "Zero-shot LLM (Qwen2.5-1.5B)",
               "arabert": "Fine-tuned AraBERT (Mizan)"}

# How each result was produced. Taken from the code/notebooks that generated it.
SETUP = {
    "tfidf": {
        "training": "Trained on train + validation (1,202 queries); C chosen on validation first",
        "input": "PII-redacted, letter-normalised text; character 2-5-grams",
        "latency": "One query at a time, on a laptop CPU (Apple MacBook Pro)",
    },
    "llm": {
        "training": "None (zero-shot): given the 77 Arabic intent names, numbered, and asked for one number",
        "input": "PII-redacted text inside an instruction prompt",
        "latency": "One query at a time, on an NVIDIA T4 GPU (Colab), fp16",
    },
    "arabert": {
        "training": "Fine-tuned on train (1,078 queries), 20 epochs; best epoch (18) chosen on validation macro-F1",
        "input": "PII-redacted text, letters left natural; max 64 tokens",
        "latency": "One query at a time, on a Colab CPU (13 ms on the T4 GPU)",
    },
}

# The 77 ArBanking77 intents are Arabic versions of the English Banking77 intents.
# Label id -> Banking77 name (same order as data/processed/labels.json).
BANKING77 = [
    "apple_pay_or_google_pay", "reverted_card_payment", "top_up_by_bank_transfer_charge", "cancel_transfer",
    "card_swallowed", "refund_not_showing_up", "supported_cards_and_currencies", "virtual_card_not_working",
    "lost_or_stolen_card", "card_not_working", "verify_source_of_funds", "topping_up_by_card",
    "top_up_by_card_charge", "getting_spare_card", "getting_virtual_card", "get_disposable_virtual_card",
    "get_physical_card", "direct_debit_payment_not_recognised", "pending_card_payment",
    "card_payment_not_recognised", "cash_withdrawal_not_recognised", "exchange_via_app",
    "beneficiary_not_allowed", "lost_or_stolen_phone", "pending_transfer", "terminate_account",
    "card_about_to_expire", "compromised_card", "edit_personal_details", "transfer_fee_charged",
    "verify_top_up", "verify_my_identity", "top_up_by_cash_or_cheque", "automatic_top_up", "change_pin",
    "activate_my_card", "card_delivery_estimate", "receiving_money", "wrong_amount_of_cash_received",
    "transaction_charged_twice", "card_payment_fee_charged", "declined_card_payment", "transfer_timing",
    "age_limit", "disposable_card_limits", "top_up_limits", "atm_support", "country_support",
    "fiat_currency_support", "card_linking", "extra_charge_on_statement", "cash_withdrawal_charge",
    "exchange_charge", "declined_transfer", "declined_cash_withdrawal", "pin_blocked", "exchange_rate",
    "card_payment_wrong_exchange_rate", "wrong_exchange_rate_for_cash_withdrawal", "request_refund",
    "order_physical_card", "top_up_reverted", "contactless_not_working", "unable_to_verify_identity",
    "failed_transfer", "top_up_failed", "pending_top_up", "pending_cash_withdrawal", "visa_or_mastercard",
    "card_acceptance", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "transfer_not_received_by_recipient",
    "why_verify_identity", "passcode_forgotten", "transfer_into_account", "card_arrival",
]

_SPECIAL_CASE = {"pin": "PIN", "atm": "ATM", "apple": "Apple", "google": "Google", "pay": "Pay",
                 "visa": "Visa", "mastercard": "Mastercard"}

# Curated demo inputs. All personal details are fictional test values
# (4111 1111 1111 1111 is the standard Visa test number).
EXAMPLES = [
    {"title": "Card hasn't arrived", "tag": "MSA",
     "text": "لقد طلبت بطاقة جديدة منذ أسبوعين ولم تصل بعد"},
    {"title": "ATM won't give cash", "tag": "Dialect",
     "text": "ليش ماكينة الصراف الآلي ما بتعطيني مصاري؟"},
    {"title": "Top up with Apple Pay", "tag": "MSA",
     "text": "هل يمكنني تعبئة رصيدي باستخدام آبل باي؟"},
    {"title": "Transfer to an IBAN", "tag": "Dialect",
     "text": "حولت مصاري على IBAN AE07 0331 2345 6789 0123 456 ولسا ما وصلت"},
    {"title": "Identity check failing", "tag": "Dialect",
     "text": "رقم هويتي 784-1990-1234567-1 بس التطبيق ما بقبل يتحقق من هويتي"},
    {"title": "Known failure: stolen card + phone", "tag": "Known failure",
     "text": "بطاقتي 4111 1111 1111 1111 انسرقت امبارح، رقم جوالي 0501234567",
     "note": "In testing, mentioning a phone number made the model answer \"lost or stolen phone\" "
             "with a high score, so the threshold does not catch it."},
    {"title": "Vague greeting", "tag": "Edge case",
     "text": "مرحبا"},
]
PREVIEW_EXAMPLE = EXAMPLES[0]   # the overview's live preview: an example the model gets right in testing

PII_TYPES = [
    ("EMIRATES_ID", "Emirates ID", "784-YYYY-NNNNNNN-N, with or without separators; Arabic-Indic digits too"),
    ("IBAN", "IBAN", "Two letters, two check digits, 11-30 characters (e.g. AE…); spaces allowed"),
    ("CARD_NUMBER", "Card number", "13-19 digits that pass the Luhn checksum"),
    ("PHONE", "Phone number", "UAE mobile/landline (+971, 00971, 0…) and other +international numbers"),
    ("EMAIL", "Email address", "name@domain.tld"),
]
PII_LABELS = {k: name for k, name, _ in PII_TYPES}

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_TAG = re.compile(r"\[(EMIRATES_ID|IBAN|CARD_NUMBER|PHONE|EMAIL)\]")


def readable(slug: str) -> str:
    """'card_payment_not_recognised' -> 'Card payment not recognised'."""
    words = [_SPECIAL_CASE.get(w, w) for w in slug.split("_")]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


@lru_cache(maxsize=1)
def labels() -> dict[str, int]:
    return json.loads((PROCESSED / "labels.json").read_text(encoding="utf-8"))["label2id"]


@lru_cache(maxsize=1)
def intent_table() -> list[dict]:
    """One row per intent: id, Arabic label, Banking77 name, readable English, training examples."""
    import pandas as pd

    train_counts = pd.read_csv(PROCESSED / "train.csv").label.value_counts().to_dict()
    rows = []
    for ar, i in sorted(labels().items(), key=lambda kv: kv[1]):
        rows.append({"id": i, "arabic": ar, "banking77": BANKING77[i], "english": readable(BANKING77[i]),
                     "train_examples": int(train_counts.get(ar, 0))})
    return rows


def english_for(arabic_label: str) -> str:
    i = labels().get(arabic_label)
    return readable(BANKING77[i]) if i is not None else "Unknown intent"


@lru_cache(maxsize=1)
def results() -> dict[str, dict]:
    out = {}
    for p in sorted(RESULTS.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        key = MODEL_KEYS.get(r.get("model"))
        if key:
            out[key] = r
    return out


@lru_cache(maxsize=1)
def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def data_summary() -> dict:
    return json.loads((PROCESSED / "summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def training_log() -> dict:
    return json.loads((EVIDENCE / "arabert_training_log.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def spot_checks() -> dict:
    return json.loads((EVIDENCE / "live_spot_checks.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def llm_errors() -> dict:
    return json.loads((EVIDENCE / "llm_error_examples.json").read_text(encoding="utf-8"))


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text or ""))


# Runs of digits (with spaces/dashes), or Latin-led codes such as IBANs and +971 numbers.
_LTR_RUN = re.compile(r"(?:\+|\b[A-Za-z]{2}(?=\d))?\d[\d\s\-A-Z]*\d|\b\d\b")


def bidi_safe(escaped: str) -> str:
    """Isolate number groups so RTL layout doesn't reverse them ('4111 1111' stays in order)."""
    return _LTR_RUN.sub(lambda m: f'<span class="mz-ltr">{m.group(0)}</span>' if " " in m.group(0) or "-" in m.group(0)
                        else m.group(0), escaped)


def display_text(text: str) -> str:
    """Escape user text for HTML and keep multi-part numbers readable in RTL."""
    return bidi_safe(html.escape(text))


def highlight_redactions(redacted: str) -> str:
    """HTML-escape the text, keep numbers in order, and wrap PII placeholders in <mark> elements."""
    safe = display_text(redacted)
    return _TAG.sub(
        lambda m: f'<mark class="pii" title="{PII_LABELS[m.group(1)]} removed">{m.group(0)}</mark>', safe)


def headline() -> dict:
    """The verified numbers used across the site."""
    r, c = results(), config()
    return {
        "arabert_f1": r["arabert"]["macro_f1"], "llm_f1": r["llm"]["macro_f1"], "tfidf_f1": r["tfidf"]["macro_f1"],
        "arabert_acc": r["arabert"]["accuracy"], "arabert_ms": r["arabert"]["latency_ms"],
        "arabert_mb": r["arabert"]["size_mb"], "llm_mb": r["llm"]["size_mb"], "llm_ms": r["llm"]["latency_ms"],
        "test_n": r["arabert"]["n"], "threshold": c["confidence_threshold"],
        "auto_share": c["test_auto_handled_share"], "auto_acc": c["test_auto_handled_accuracy"],
        "review_share": c["test_routed_to_human_share"],
    }