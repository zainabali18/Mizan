import pytest

from mizan.pii import luhn_valid, redact


@pytest.mark.parametrize("eid", ["784-1990-1234567-1", "784 1990 1234567 1", "784199012345671"])
def test_emirates_id(eid):
    text, kinds = redact(f"رقم هويتي {eid} لو سمحت")
    assert kinds == ["EMIRATES_ID"]
    assert "[EMIRATES_ID]" in text and "1234567" not in text


def test_emirates_id_in_arabic_indic_digits():
    text, kinds = redact("هويتي ٧٨٤-١٩٩٠-١٢٣٤٥٦٧-١")
    assert kinds == ["EMIRATES_ID"]


@pytest.mark.parametrize("iban", ["AE070331234567890123456", "AE07 0331 2345 6789 0123 456", "ae070331234567890123456"])
def test_iban(iban):
    text, kinds = redact(f"حولوا على {iban} بسرعة")
    assert kinds == ["IBAN"]
    assert "0331" not in text


@pytest.mark.parametrize("card", ["4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111"])
def test_card_number_with_valid_luhn(card):
    text, kinds = redact(f"بطاقتي {card} انرفضت")
    assert kinds == ["CARD_NUMBER"]


def test_long_number_failing_luhn_is_not_a_card():
    _, kinds = redact("رقم الطلب 4111111111111112")
    assert "CARD_NUMBER" not in kinds


@pytest.mark.parametrize("phone", ["+971 50 123 4567", "00971501234567", "0501234567", "04 123 4567", "+44 7700 900123"])
def test_phone_numbers(phone):
    text, kinds = redact(f"اتصلوا على {phone}")
    assert kinds == ["PHONE"], (phone, text)


def test_email():
    text, kinds = redact("ايميلي zainab.ali@example.com")
    assert kinds == ["EMAIL"] and "example.com" not in text


def test_multiple_items_keep_order():
    text, kinds = redact("هويتي 784-1990-1234567-1 وجوالي 0501234567")
    assert kinds == ["EMIRATES_ID", "PHONE"]
    assert text == "هويتي [EMIRATES_ID] وجوالي [PHONE]"


@pytest.mark.parametrize("safe", ["سحبت 500 جنيه", "اشتريت 10 الاف درهم", "حاولت 5 مرات", "مر 3 أيام", "في 2024"])
def test_ordinary_numbers_are_not_redacted(safe):
    text, kinds = redact(safe)
    assert kinds == [] and text == safe


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("123")
