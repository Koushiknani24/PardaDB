import pytest

from pardadb import detectors as d


@pytest.mark.parametrize(
    "num, ok",
    [
        ("4111111111111111", True),
        ("5555555555554444", True),
        ("4012888888881881", True),
        ("4111111111111112", False),  # one digit off
        ("7306089151955451", False),  # 16-digit txn id shape
        ("424242424242", False),  # 12 digits: below the 13-digit minimum
        ("4" + "0" * 18 + "4", False),  # 20 digits: above the 19-digit maximum
        ("", False),
        ("4111-1111-1111-1111", False),  # luhn() expects bare digits
    ],
)
def test_luhn(num, ok):
    assert d.luhn(num) is ok


@pytest.mark.parametrize(
    "value, kind",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (42, "number"),
        (3.5, "number"),
        ("4111 1111 1111 1111", "card"),
        ("4111-1111-1111-1111", "card"),
        ("4111111111111112", "digits"),  # fails Luhn -> just a long id
        ("2345 6789 0123", "aadhaar"),
        ("234567890123", "aadhaar"),
        ("XXXX XXXX 1234", "aadhaar_masked"),
        ("xxxx-xxxx-1234", "aadhaar_masked"),
        ("****  ****1234", "word"),  # double separator is not the masked Aadhaar shape
        ("ABCPK1234F", "pan"),
        ("900-12-3456", "ssn"),
        ("123 45 6789", "ssn"),
        ("+91 98765 43210", "phone"),
        ("+91-9876543210", "phone"),
        ("09876543210", "phone"),
        ("9876543210", "phone"),
        ("+1 (415) 555-0134", "phone"),
        ("priya.nair@example.com", "email"),
        ("komal@okaxis", "upi"),
        ("SBIN0001234", "ifsc"),
        ("1990-05-12", "date"),
        ("2024-01-01 10:20:00", "date"),
        ('{"pan": "ABCPK1234F"}', "json"),
        ("Komal Sharma", "name"),
        ("Saffron Textiles Ltd", "org"),
        ("12 MG Road, Bangalore", "address"),
    ],
)
def test_classify_value(value, kind):
    assert d.classify_value(value) == kind


@pytest.mark.parametrize(
    "value",
    [
        "1234 5678 9012",  # Aadhaar never starts with 0 or 1
        "2345 6789-0123",  # mixed separators
        "ABCXK1234F",  # 4th PAN letter must be an entity-type code
        "abcpk1234f",  # PAN is upper case
        "5876543210",  # Indian mobiles start with 6-9
        "{not json",
    ],
)
def test_classify_value_rejects_lookalikes(value):
    assert d.classify_value(value) not in {"aadhaar", "pan", "phone", "json", "card"}


def test_canonical_forms_unify_formats():
    phones = ["+91 98765 43210", "09876543210", "9876543210", "+91-98765-43210"]
    assert {d.canonical("phone", p) for p in phones} == {"+919876543210"}
    assert d.canonical("phone", "+1 (415) 555-0134") == "+14155550134"
    assert d.canonical("card", "4111 1111-1111 1111") == "4111111111111111"
    assert d.canonical("pan", " ABCPK1234F ") == "abcpk1234f"
    assert d.canonical("person", "Komal   SHARMA") == "komal sharma"


def test_find_spans_empty():
    assert d.find_spans("") == []
    assert d.find_spans(None) == []
    assert d.find_spans("nothing sensitive here at all") == []


def test_find_spans_ticket_text():
    text = "My card 4111 1111 1111 1111 was charged, call me on 9876543210.\nRegards,\nKomal Sharma"
    spans = d.find_spans(text)
    assert [(text[s:e], ent) for s, e, ent in spans] == [
        ("4111 1111 1111 1111", "card"),
        ("9876543210", "phone"),
        ("Komal Sharma", "person"),
    ]


def test_find_spans_identifiers():
    text = (
        "PAN ABCPK1234F, upi komal@okaxis, aadhaar 2345 6789 0123, ssn 900-12-3456, "
        "us +1 (415) 555-0134, mail priya.nair@example.com"
    )
    found = {text[s:e]: ent for s, e, ent in d.find_spans(text)}
    assert found == {
        "ABCPK1234F": "pan",
        "komal@okaxis": "upi",
        "2345 6789 0123": "aadhaar",
        "900-12-3456": "ssn",
        "+1 (415) 555-0134": "phone",
        "priya.nair@example.com": "email",
    }


def test_find_spans_non_luhn_16_digits_is_account_not_card():
    text = "txn 7306089151955451 failed"
    assert [(text[s:e], ent) for s, e, ent in d.find_spans(text)] == [("7306089151955451", "account")]


def test_find_spans_spans_never_overlap():
    text = "Email 9876543210@ybl or a.b@example.com; card 4111111111111111; acct 16139099060456"
    spans = d.find_spans(text)
    assert spans == sorted(spans)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_name_index_matches_known_names_case_insensitively():
    idx = d.NameIndex(["Komal Sharma", "Single", None, ""])
    assert idx.names == {"komal sharma"}  # one-word and empty names are ignored
    text = "called by komal sharma yesterday"
    assert [(text[s:e], ent) for s, e, ent in d.find_spans(text, idx)] == [("komal sharma", "person")]
    assert d.NameIndex().find("Komal Sharma") == []


def test_find_spans_extra_ai_spans_lose_to_rules():
    text = "card 4111111111111111 for Ravi near Gandhi Park"
    extra = [(5, 21, "person"), (26, 30, "person"), (31, 47, "address")]
    found = [(text[s:e], ent) for s, e, ent in d.find_spans(text, extra=extra)]
    assert found == [("4111111111111111", "card"), ("Ravi", "person (AI)"), ("near Gandhi Park", "address (AI)")]


@pytest.mark.parametrize(
    ("text", "address"),
    [
        ("pls update my address to 21, 15th Cross, Anand Path, Shanti Puram Surat 395008. thx",
         "21, 15th Cross, Anand Path, Shanti Puram Surat 395008"),
        ("H.No 29-70/6, Main Road 6, Thiruvananthapuram 695037 is my new address",
         "H.No 29-70/6, Main Road 6, Thiruvananthapuram 695037"),
        ("Address: Flat 302, Krishna Towers, Mumbai 400020.\nThanks",
         "Flat 302, Krishna Towers, Mumbai 400020"),
        ("Sent Rs 331 to my brother, he lives at 12 Gandhi Road Pune 411001",
         "12 Gandhi Road Pune 411001"),
    ],
)  # fmt: skip
def test_address_spans_find_written_addresses(text, address):
    spans = [text[a:b] for a, b, ent in d.find_spans(text) if ent == "address"]
    assert spans == [address]


@pytest.mark.parametrize(
    "text",
    [
        "Sent Rs 125000 to my brother at Gandhi Road branch",  # an amount is not a pincode
        "card 4111 1111 1111 1111 kho gaya, call 9000011111",  # no address at all
        "Pincode 400020 only, no street",  # pincode without a house number and street
        "",
    ],
)
def test_address_spans_ignore_non_addresses(text):
    assert d.address_spans(text) == []
