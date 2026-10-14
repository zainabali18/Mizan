"""Arabic text normalisation.

Customers write the same word in many ways: with or without diacritics, with
different alef or yaa forms, with stretched letters (tatweel) or with
Arabic-Indic digits. Normalising these variants means the model sees one
consistent form, which matters most when training data is small.
"""
from __future__ import annotations

import re

# Short vowels and other diacritics (tashkeel), plus the superscript alef.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF_VARIANTS = re.compile("[آأإٱ]")  # آ أ إ ٱ -> ا
_WHITESPACE = re.compile(r"\s+")

# Arabic-Indic (٠-٩) and Eastern Arabic-Indic (۰-۹) digits -> 0-9.
_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩"
    "۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def normalise_digits(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits (so PII patterns match)."""
    return text.translate(_DIGITS)


def normalise_arabic(text: str) -> str:
    """Return a consistent form of Arabic text for modelling.

    Steps: digits -> Western, remove diacritics and tatweel, unify alef forms,
    alef maqsura (ى) -> yaa (ي), taa marbuta (ة) -> haa (ه), collapse whitespace.
    Latin text, numbers and punctuation are kept as they are.
    """
    if not text:
        return ""
    text = normalise_digits(text)
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = _ALEF_VARIANTS.sub("ا", text)
    text = text.replace("ى", "ي")  # ى -> ي
    text = text.replace("ة", "ه")  # ة -> ه
    return _WHITESPACE.sub(" ", text).strip()
