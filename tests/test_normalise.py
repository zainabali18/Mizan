from mizan.normalise import normalise_arabic, normalise_digits


def test_removes_diacritics_and_tatweel():
    assert normalise_arabic("بِطَاقَتِي") == "بطاقتي"
    assert normalise_arabic("بطـــاقة") == "بطاقه"


def test_unifies_alef_yaa_and_taa_marbuta():
    assert normalise_arabic("أإآ") == "ااا"
    assert normalise_arabic("على") == "علي"
    assert normalise_arabic("البطاقة") == "البطاقه"


def test_converts_arabic_indic_digits():
    assert normalise_digits("٥٠٠ درهم") == "500 درهم"
    assert normalise_digits("۱۲۳") == "123"


def test_collapses_whitespace_and_keeps_latin_text():
    assert normalise_arabic("  Apple   Pay  ") == "Apple Pay"


def test_empty_input():
    assert normalise_arabic("") == ""
