import json

import pytest

from pardadb import detectors, scanner


@pytest.mark.parametrize(
    "table, column, tag, entity",
    [
        ("cards", "card_no", "PII_HIGH", "card"),
        ("cards", "col_7", "PII_HIGH", "pan"),  # meaningless header: values decide
        ("customers", "full_name", "PII_MEDIUM", "person"),
        ("customers", "mobile", "PII_MEDIUM", "phone"),
        ("customers", "dob", "PII_QUASI", "dob"),
        ("customers", "pincode", "PII_QUASI", "postal"),
        ("customers", "balance", "FINANCIAL_CONFIDENTIAL", "money"),
        ("customers", "customer_id", "INTERNAL_ID", "customer_id"),
        ("support_tickets", "ticket_text", "PII_FREE_TEXT", "text"),
    ],
)
def test_synthetic_bank_tags(cols, table, column, tag, entity):
    c = cols[(table, column)]
    assert (c["tag"], c["entity"]) == (tag, entity)
    assert c["status"] == "auto" and c["confidence"] >= 0.8


@pytest.mark.parametrize("column", ["rrn", "txn_id", "card_ref"])
def test_lookalike_identifiers_stay_clean(cols, column):
    c = cols[("cards", column)]
    assert c["tag"] is None and c["status"] == "clean"


def test_card_evidence_mentions_luhn(cols):
    assert any("all pass Luhn" in e for e in cols[("cards", "card_no")]["evidence"])
    assert any("reference number" in e for e in cols[("cards", "rrn")]["evidence"])


def test_scan_summary_and_index(catalog):
    s = catalog["summary"]
    assert s == {"tagged": 11, "review": 0, "clean": 4}
    idx = scanner.index(catalog)
    assert ("cards", "col_7") in idx
    assert ("cards", "rrn") not in idx  # only tagged columns are indexed


def test_scan_reports_progress(backend):
    events = []
    scanner.scan(backend, progress=lambda ev, **kw: events.append(ev), sample_size=5)
    assert events.count("table") == 3
    assert events.count("column") == 15


def test_empty_column_is_clean():
    c = scanner.classify_column("t", "card_no", [], 1.0)
    assert c["tag"] is None and c["evidence"] == ["no non-null values to sample"]


def test_aadhaar_with_premasked_values():
    vals = ["2345 6789 0123", "XXXX XXXX 1234", "3456 7890 1234", "XXXX XXXX 9876", "4567 8901 2345"]
    c = scanner.classify_column("kyc", "col_3", vals, 0.0)
    assert (c["tag"], c["entity"]) == ("PII_HIGH", "aadhaar")
    assert any("already masked at source" in e for e in c["evidence"])


def test_twelve_digit_reference_numbers_with_one_leading_digit_are_clean():
    vals = [f"5{n:011d}" for n in range(100, 120)]
    c = scanner.classify_column("t", "col_9", vals, 0.0)
    assert c["tag"] is None
    assert "reference numbers, not Aadhaar" in c["evidence"][-1]


def test_mostly_non_card_digits_are_not_cards():
    ids = [f"7{n:015d}" for n in range(1, 30) if not detectors.luhn(f"7{n:015d}")][:5]
    vals = ["4111111111111111"] * 5 + ids  # 50% cards, 50% non-Luhn ids
    assert scanner.classify_column("t", "col_x", vals, 0.0)["entity"] != "card"


def test_header_only_hint_goes_to_review():
    c = scanner.classify_column("t", "email", ["n/a", "unknown", "none", "pending"], 0.0)
    assert (c["entity"], c["confidence"], c["status"]) == ("email", 0.6, "review")


def test_branch_and_bank_domain_contacts_are_business_data():
    assert scanner.classify_column("branches", "branch_phone", ["9876543210"] * 5, 0.0)["tag"] is None
    own = ["ops@saffronbank.example", "help@saffronbank.example"]
    assert scanner.classify_column("t", "emailid", own, 0.0)["tag"] is None


def test_business_records_clears_contacts_in_tables_without_people():
    cat = {
        "columns": [
            {"table": "vendors", "column": "phone", "tag": "PII_MEDIUM", "entity": "phone", "evidence": []},
            {"table": "people", "column": "nm", "tag": "PII_MEDIUM", "entity": "person", "evidence": []},
            {"table": "people", "column": "phone", "tag": "PII_MEDIUM", "entity": "phone", "evidence": []},
        ]
    }
    scanner.business_records(cat)
    assert [c["tag"] for c in cat["columns"]] == [None, "PII_MEDIUM", "PII_MEDIUM"]


def test_header_tokens_split_camel_and_snake_case():
    assert scanner.header_tokens("CUSTOMER_ID") == ["customer", "id"]
    assert scanner.header_tokens("custNm") == ["cust", "nm"]
    assert scanner.header_hint("rrn")[2] is True  # negative header


def _kyc(doc: dict[str, object]) -> list[str]:
    return [json.dumps(doc)] * 5


@pytest.mark.parametrize(
    ("doc", "keys"),
    [
        pytest.param({"kyc": {"pan": "ABCPK1234F", "ok": True}}, ["pan"], id="nested-object"),
        pytest.param({"contacts": [{"Mobile": "9876543210"}, {"mobile": "9123456780"}]}, ["mobile"], id="list"),
        pytest.param([{" PAN ": "ABCPK1234F"}], ["pan"], id="top-level-list-normalised"),
        pytest.param({"pan": {"ABCPK1234F": True}}, ["pan"], id="key-is-pii-flags-parent"),
        pytest.param({"Father_Name": "Rahul Verma", "dms_ref": "651238604723"}, ["father_name"], id="top-level"),
    ],
)
def test_json_keys_found_at_any_depth_and_normalised(doc: object, keys: list[str]) -> None:
    c = scanner.classify_column("kyc_records", "kyc_meta", _kyc(doc), 0.0)
    assert (c["tag"], c["entity"]) == ("SENSITIVE_JSON", "json")
    assert c["json_keys"] == keys


def test_json_walk_stops_at_depth_cap() -> None:
    doc: object = {"pan": "ABCPK1234F"}
    for _ in range(scanner.JSON_MAX_DEPTH + 1):
        doc = {"wrap": doc}
    c = scanner.classify_column("t", "meta", _kyc(doc), 0.0)
    assert c["tag"] is None and "json_keys" not in c
    shallow = {"wrap": {"wrap": {"pan": "ABCPK1234F"}}}
    assert scanner.classify_column("t", "meta", _kyc(shallow), 0.0)["json_keys"] == ["pan"]


def test_json_walk_caps_pairs_per_document() -> None:
    doc = {f"k{i}": i for i in range(scanner.JSON_MAX_PAIRS * 3)}
    assert len(scanner.json_pairs({"root": doc})) == scanner.JSON_MAX_PAIRS
