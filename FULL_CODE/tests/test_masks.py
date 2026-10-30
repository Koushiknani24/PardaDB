import datetime as dt
import os
import stat

import pytest

from pardadb import masks


@pytest.mark.parametrize(
    "entity, value, expected",
    [
        ("card", "4111 1111 1111 1111", "4111-XXXX-XXXX-1111"),
        ("card", "4111-1111-1111-1111", "4111-XXXX-XXXX-1111"),
        ("aadhaar", "2345 6789 0123", "XXXX XXXX 0123"),
        ("pan", "ABCPK1234F", "XXXXXX234F"),
        ("ssn", "900-12-3456", "XXX-XX-3456"),
        ("account", "16139099060456", "XXXXXXXXXX0456"),
        ("phone", "+91 98765 43210", "+91 XXXXXX3210"),
        ("phone", "09876543210", "+91 XXXXXX3210"),
        ("phone", "+1 (415) 555-0134", "+1 (XXX) XXX-0134"),
        ("email", "priya.nair@example.com", "p***@example.com"),
        ("person", "Komal Sharma", "Komal S****"),
        ("person", "", ""),
        ("address", "12 MG Road", "XX MG Road"),
        ("dob", "1990-05-12", "1990"),
        ("postal", "560010", "560XXX"),
    ],
)
def test_partial_value(entity, value, expected):
    assert masks.partial_value(entity, value) == expected


def test_partial_card_never_leaks_middle_digits():
    out = masks.apply("PARTIAL", "5555555555554444", {"entity": "card"})
    assert out == "5555-XXXX-XXXX-4444"


def test_partial_premasked_aadhaar_kept_as_is():
    assert masks.apply("PARTIAL", "XXXX XXXX 4321", {"entity": "aadhaar"}) == "XXXX XXXX 4321"


def test_token_is_deterministic_and_format_independent():
    t = masks.token("phone", "+91 98765 43210")
    assert t.startswith("tok_") and len(t) == 16
    assert masks.token("phone", "09876543210") == t  # same person, other format
    assert masks.token("phone", "9876543211") != t
    assert masks.token("person", "+91 98765 43210") != t  # entity is part of the key material


def test_token_depends_on_key(monkeypatch):
    before = masks.token("person", "Komal Sharma")
    monkeypatch.setattr(masks, "_KEY", b"a-different-key")
    assert masks.token("person", "Komal Sharma") != before


def test_hmac_key_generated_once_with_private_permissions(monkeypatch, isolated_state):
    monkeypatch.delenv("PARDADB_HMAC_KEY")
    key = masks.hmac_key()
    path = isolated_state / ".hmac_key"
    assert len(key) == 64 and path.read_text().strip().encode() == key
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    monkeypatch.setattr(masks, "_KEY", None)
    assert masks.hmac_key() == key  # re-read, not regenerated


@pytest.mark.parametrize(
    "value, label",
    [
        (0, "< 10K"),
        (-500, "< 10K"),
        (9_999.99, "< 10K"),
        (10_000, "10K-1L"),
        ("99999", "10K-1L"),
        (1_00_000, "1L-10L"),
        (10_00_000, "10L-1Cr"),
        (1_00_00_000, "> 1Cr"),
        ("abc", "[REDACTED]"),
        (None, "[REDACTED]"),
    ],
)
def test_bucket_boundaries(value, label):
    assert masks.bucket(value) == label


def test_generalize():
    assert masks.generalize("dob", "1990-05-12") == "30-39"
    assert masks.generalize("dob", "not a date") == "[REDACTED]"
    assert masks.generalize("postal", "560010") == "560XXX"
    assert masks.generalize("gender", "female") == "F"
    assert masks.generalize("pan", "ABCPK1234F") == "[REDACTED]"


@pytest.mark.parametrize(
    ("born", "today", "band"),
    [
        ("1996-11-06", dt.date(2026, 10, 30), "20-29"),  # a week short of 30: still 29
        ("1996-10-30", dt.date(2026, 10, 30), "30-39"),  # 30th birthday today
        ("2000-02-29", dt.date(2030, 2, 28), "20-29"),  # leap-day birthday, day before it counts
        ("2000-02-29", dt.date(2030, 3, 1), "30-39"),
    ],
)
def test_generalize_dob_age_boundaries(born, today, band):
    assert masks.generalize("dob", born, today=today) == band


@pytest.mark.parametrize("strategy", ["FULL", "REDACT", "HASH", "PARTIAL", "BUCKET", "GENERALIZE", "SPANS_TYPED"])
def test_apply_keeps_null(strategy):
    assert masks.apply(strategy, None, {"entity": "card"}) is None


def test_apply_unknown_strategy_fails_closed():
    assert masks.apply("SOMETHING_NEW", "4111111111111111", {"entity": "card"}) == "[REDACTED]"
    assert masks.apply("REDACT", "x", None) == "[REDACTED]"


TICKET = "My card 4111 1111 1111 1111 was charged, call me on 9876543210.\nRegards,\nKomal Sharma"


def test_mask_text_per_role():
    assert masks.mask_text(TICKET, typed=True) == ("My card [CARD] was charged, call me on [PHONE].\nRegards,\n[NAME]")
    assert masks.mask_text(TICKET, typed=False) == (
        "My card 4111-XXXX-XXXX-1111 was charged, call me on +91 XXXXXX3210.\nRegards,\nKomal S****"
    )
    assert masks.mask_text("no pii here", typed=True) == "no pii here"


def test_mask_text_marks_point_at_replacements():
    out, marks = masks.mask_text(TICKET, typed=True, with_marks=True)
    assert [(out[a:b], ent) for a, b, ent in marks] == [("[CARD]", "card"), ("[PHONE]", "phone"), ("[NAME]", "person")]


def test_mask_text_ai_spans_are_masked():
    text = "Spoke with Ravi at the counter"
    assert masks.mask_text(text, typed=True, extra=[(11, 15, "person")]) == "Spoke with [NAME] at the counter"


def test_mask_json():
    doc = '{"pan": "ABCPK1234F", "note": "call 9876543210", "x": 1}'
    info = {"json_keys": ["pan"]}
    assert masks.mask_json(doc, info, redact=False) == '{"note": "call +91 XXXXXX3210", "pan": "XXXXXX234F", "x": 1}'
    assert masks.mask_json(doc, info, redact=True) == '{"note": "call [PHONE]", "pan": "[REDACTED]", "x": 1}'
    assert masks.mask_json("{broken", info, redact=False) == "[REDACTED]"
