"""Shared fixtures: a tiny synthetic bank in a temp dir, so the suite never needs the problem pack.

Environment is pointed at throwaway directories *before* pardadb is imported, because
config.py reads its paths at import time. Nothing here touches FULL_CODE/.state or the
real databases, and the ML sidecar URL points at a closed port so no test hits the network.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="pardadb-tests-"))
atexit.register(shutil.rmtree, _TMP, True)
os.environ["PARDADB_STATE"] = str(_TMP / "state")
os.environ["PARDADB_DATASET"] = str(_TMP / "dataset")
os.environ["PARDADB_SQLITE"] = str(_TMP / "bank.db")
os.environ["PARDADB_HMAC_KEY"] = "test-hmac-key-not-a-secret"
os.environ["PARDADB_ML_URL"] = "http://127.0.0.1:9"  # discard port: always refused
os.environ["PARDADB_ML_TIMEOUT"] = "1"

import pytest  # noqa: E402

from pardadb import audit, backends, config, detectors, engine, masks, ml, scanner  # noqa: E402


def luhn_complete(prefix15: str) -> str:
    """Append the Luhn check digit to a 15-digit prefix."""
    return next(prefix15 + d for d in "0123456789" if detectors.luhn(prefix15 + d))


def luhn_fail(prefix15: str) -> str:
    good = luhn_complete(prefix15)
    return good[:-1] + str((int(good[-1]) + 1) % 10)


NAMES = [
    "Komal Sharma",
    "Anthony Fernandes",
    "Rahul Verma",
    "Priya Nair",
    "Arjun Mehta",
    "Sneha Iyer",
    "Vikram Singh",
    "Neha Gupta",
    "Rohan Das",
    "Anita Desai",
    "Suresh Kumar",
    "Meera Joshi",
]
PHONES = ["+91 98765 43210", "9876543211", "+91-9123456780", "09812345678"] * 3
PANS = ["ABCPK1234F", "BCDPF2345G", "CDEPJ3456H", "DEFPL4567J", "EFGPM5678K", "FGHPN6789L"] * 2
TICKETS = [
    "My card 4111 1111 1111 1111 was charged twice, please call me on 9876543210 today.\nRegards,\nKomal Sharma",
    "Please update my nominee details, my name is Anthony Fernandes and the branch is closed.",
    "Card chori ho gaya train me. Account 16139099060456. Jaldi block karo pls, very urgent issue.",
    "The staff at the Mumbai branch were very helpful in closing my FD. Thank you for the support.",
    "Received an OTP I never asked for, mail me at priya.nair@example.com with the details please.",
    "Spoke to Arjun Mehta about the loan EMI, he said it would be sorted out within a week.",
]
CARDS = ["4111111111111111"] + [luhn_complete(f"4{i:02d}{12345678900 + i:012d}") for i in range(1, 12)]


def _rows():
    customers, cards = [], []
    for i, name in enumerate(NAMES):
        cid = 1001 + i
        dob = f"{1960 + i * 3}-0{1 + i % 9}-1{i % 9}"
        customers.append((cid, name, PHONES[i], dob, f"5600{10 + i}", 1500.5 + i * 250_000))
        card = CARDS[i] if i % 3 else " ".join(CARDS[i][k : k + 4] for k in range(0, 16, 4))
        rrn = f"{3 + i % 6}{i:03d}{4567890 + i:07d}{i % 10}"  # 12 digits, Aadhaar-shaped reference number
        txn = luhn_fail(f"7{i:02d}6089151955{i:02d}")  # 16 digits that fail Luhn: not a card
        cards.append((9000 + i, cid, card, PANS[i], rrn, txn))
    tickets = [(i + 1, 1001 + i, t) for i, t in enumerate(TICKETS)]
    return customers, cards, tickets


def build_bank(path: Path) -> Path:
    customers, cards, tickets = _rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, full_name TEXT, mobile TEXT,
                                dob TEXT, pincode TEXT, balance REAL);
        CREATE TABLE cards (card_ref INTEGER PRIMARY KEY, customer_id INTEGER, card_no TEXT,
                            col_7 TEXT, rrn TEXT, txn_id TEXT);
        CREATE TABLE support_tickets (ticket_id INTEGER PRIMARY KEY, customer_id INTEGER, ticket_text TEXT);
        """
    )
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)
    con.executemany("INSERT INTO cards VALUES (?,?,?,?,?,?)", cards)
    con.executemany("INSERT INTO support_tickets VALUES (?,?,?)", tickets)
    con.commit()
    con.close()
    return path


@pytest.fixture(scope="session")
def bank_db() -> Path:
    return build_bank(config.SQLITE_PATH)


@pytest.fixture(scope="session")
def backend(bank_db):
    b = backends.SQLiteBackend().connect()
    assert Path(config.SQLITE_PATH) == bank_db
    yield b
    b.conn.close()


@pytest.fixture(scope="session")
def catalog(backend):
    return scanner.scan(backend)


@pytest.fixture(scope="session")
def cols(catalog):
    """{(table, column): scanner result} for the synthetic bank."""
    return {(c["table"], c["column"]): c for c in catalog["columns"]}


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Fresh state dir per test (audit log, playground), ML sidecar switched off."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(audit, "_last", None)
    monkeypatch.setattr(ml.client, "enabled", False)
    monkeypatch.setattr(masks, "_KEY", None)
    return tmp_path / "state"


@pytest.fixture
def eng(backend, catalog):
    return engine.Engine(backend, catalog)
