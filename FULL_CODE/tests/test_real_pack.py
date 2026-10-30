"""Optional checks against the organisers' problem pack; skipped when it is not present."""

from pathlib import Path

import pytest

from pardadb import backends, config, scanner

REAL_PACK_DB = Path(__file__).resolve().parent.parent / "data" / "sql" / "sqlite" / "saffron_bank.db"

pytestmark = pytest.mark.skipif(not REAL_PACK_DB.is_file(), reason="problem pack database not present")


@pytest.fixture(scope="module")
def pack():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "SQLITE_PATH", REAL_PACK_DB)
        b = backends.SQLiteBackend().connect()  # mode=ro: cannot modify the pack
    yield b
    b.conn.close()


def classify(b, table, column):
    return scanner.classify_column(table, column, b.sample(table, column, 200), 0.0)


@pytest.mark.parametrize(
    "table, column, entity",
    [
        ("loan_applications", "col_7", "pan"),
        ("cards", "CARD_NO", "card"),
        ("customers", "cust_nm", "person"),
        ("kyc_records", "aadhaar_no", "aadhaar"),
        ("support_tickets", "ticket_text", "text"),
        ("transactions", "ref_no", "text"),  # some references are real card numbers: masked value by value
    ],
)
def test_pack_pii_columns_tagged(pack, table, column, entity):
    assert classify(pack, table, column)["entity"] == entity


@pytest.mark.parametrize("table, column", [("transactions", "rrn"), ("transactions", "txn_id")])
def test_pack_lookalikes_clean(pack, table, column):
    assert classify(pack, table, column)["tag"] is None
