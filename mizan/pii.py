"""Redact personal data (PII) before any text reaches a model or a log.

Banks are regulated: customer identifiers should never be stored in training
data, sent to a model, or written to logs. This module replaces them with
typed placeholders so the request can still be understood ("my card
[CARD_NUMBER] was declined") without exposing the customer.

Detected types (UAE-focused):
  EMIRATES_ID   784-YYYY-NNNNNNN-N (15 digits starting 784, any separators)
  IBAN          e.g. AE07 0331 2345 6789 0123 456 (country code + check digits)
  CARD_NUMBER   13-19 digits that pass the Luhn checksum
  PHONE         UAE mobiles/landlines (+971 / 00971 / 0) and other +international
  EMAIL         name@domain.tld
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalise import normalise_digits

# Separators customers put between digit groups.
_SEP = r"[\s\-.]?"

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    # IBAN: 2 letters, 2 check digits, then 11-30 alphanumerics (spaces allowed).
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b", re.IGNORECASE)),
    # Emirates ID: 784 + year(4) + 7 digits + 1 check digit.
    ("EMIRATES_ID", re.compile(rf"(?<!\d)784{_SEP}\d{{4}}{_SEP}\d{{7}}{_SEP}\d(?!\d)")),
    # Candidate card numbers; confirmed with the Luhn check below.
    ("CARD_NUMBER", re.compile(r"(?<!\d)(?:\d[\s\-]?){12,18}\d(?!\d)")),
    # UAE phone numbers, then generic international numbers.
    ("PHONE", re.compile(rf"(?<![\w+])(?:\+971|00971|0)\s?-?(?:5\d|[234679]){_SEP}\d{{3}}{_SEP}\d{{3,4}}(?!\d)")),
    ("PHONE", re.compile(r"(?<![\w+])\+\d{1,3}[\s\-]?\d(?:[\s\-]?\d){6,13}(?!\d)")),
]


@dataclass(frozen=True)
class Redaction:
    """One piece of personal data that was removed."""

    kind: str
    start: int
    end: int


def luhn_valid(digits: str) -> bool:
    """Return True if the digit string passes the Luhn checksum used by cards."""
    nums = [int(d) for d in digits if d.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def find_pii(text: str) -> list[Redaction]:
    """Find personal data in text. Earlier pattern types win on overlap."""
    found: list[Redaction] = []
    taken = [False] * len(text)
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if any(taken[m.start():m.end()]):
                continue
            if kind == "CARD_NUMBER" and not luhn_valid(m.group()):
                continue
            found.append(Redaction(kind, m.start(), m.end()))
            for i in range(m.start(), m.end()):
                taken[i] = True
    return sorted(found, key=lambda r: r.start)


def redact(text: str) -> tuple[str, list[str]]:
    """Replace personal data with placeholders like [EMIRATES_ID].

    Arabic-Indic digits are converted first so "٧٨٤..." is caught too.
    Returns the redacted text and the list of redacted types, in order.
    """
    text = normalise_digits(text or "")
    redactions = find_pii(text)
    out, last = [], 0
    for r in redactions:
        out.append(text[last:r.start])
        out.append(f"[{r.kind}]")
        last = r.end
    out.append(text[last:])
    return "".join(out), [r.kind for r in redactions]
