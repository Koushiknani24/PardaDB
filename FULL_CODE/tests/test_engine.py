from collections.abc import Iterator
from pathlib import Path

import pytest

from pardadb import audit, backends, config, engine, ml, scanner

ROW_SQL = "SELECT full_name, card_no, mobile, dob, pincode, balance FROM customers JOIN cards USING (customer_id)"


def first_row(eng, sql, role):
    res = eng.run(sql, role, record=False)
    assert res["ok"], res
    return res["rows"][0], {c["name"]: c["strategy"] for c in res["columns"]}


def test_same_query_three_roles(eng):
    admin, _ = first_row(eng, ROW_SQL, "admin")
    support, s_strat = first_row(eng, ROW_SQL, "support")
    analyst, a_strat = first_row(eng, ROW_SQL, "analyst")
    assert admin == ["Komal Sharma", "4111 1111 1111 1111", "+91 98765 43210", "1960-01-10", "560010", 1500.5]
    assert support == ["Komal S****", "4111-XXXX-XXXX-1111", "+91 XXXXXX3210", "1960", "560XXX", "[REDACTED]"]
    assert analyst[0].startswith("tok_") and analyst[2].startswith("tok_")
    assert analyst[1] == "[REDACTED]"
    assert analyst[3:] == ["60-69", "560XXX", "< 10K"]
    assert s_strat["card_no"] == "PARTIAL" and a_strat["card_no"] == "REDACT"


def test_select_star_masks_every_tagged_column(eng):
    res = eng.run("SELECT * FROM cards", "analyst", record=False)
    by_name = {c["name"]: c["strategy"] for c in res["columns"]}
    assert by_name == {
        "card_ref": "FULL",
        "customer_id": "HASH",
        "card_no": "REDACT",
        "col_7": "REDACT",
        "rrn": "FULL",
        "txn_id": "FULL",
    }
    col7 = [c["name"] for c in res["columns"]].index("col_7")
    assert {r[col7] for r in res["rows"]} == {"[REDACTED]"}


def test_alias_and_case_do_not_bypass_masking(eng):
    row, strat = first_row(eng, "SELECT CARD_NO AS x FROM CARDS", "support")
    assert row == ["4111-XXXX-XXXX-1111"] and strat == {"x": "PARTIAL"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT substr(card_no, 1, 16) AS p FROM cards",
        "SELECT card_no || '' AS p FROM cards",
        "SELECT (SELECT card_no FROM cards LIMIT 1) AS p",
        "SELECT CASE WHEN full_name = 'Komal Sharma' THEN 1 ELSE 0 END AS p FROM customers",
        "SELECT max(balance) AS p FROM customers",
    ],
)
def test_derived_expressions_are_redacted(eng, sql):
    for role in ("support", "analyst"):
        row, strat = first_row(eng, sql, role)
        assert row == ["[REDACTED]"] and strat == {"p": "REDACT"}, role
    assert first_row(eng, sql, "admin")[1] == {"p": "FULL"}


def test_tokens_consistent_across_queries(eng):
    a, _ = first_row(eng, "SELECT full_name FROM customers", "analyst")
    b, _ = first_row(eng, "SELECT full_name AS n FROM customers JOIN cards USING (customer_id)", "analyst")
    assert a == b


def test_free_text_spans_per_role(eng):
    sql = "SELECT ticket_text FROM support_tickets WHERE ticket_id = 1"
    admin = eng.run(sql, "admin", record=False)
    assert "4111 1111 1111 1111" in admin["rows"][0][0]
    assert {m[2] for m in admin["marks"]["0:0"]} == {"card", "phone", "person"}  # highlighted, not masked
    assert first_row(eng, sql, "analyst")[0] == [
        "My card [CARD] was charged twice, please call me on [PHONE] today.\nRegards,\n[NAME]"
    ]
    support = first_row(eng, sql, "support")[0][0]
    assert "4111-XXXX-XXXX-1111" in support and "Komal S****" in support


@pytest.mark.parametrize(
    "sql, verb",
    [
        ("SELECT customer_id FROM customers WHERE full_name = 'Komal Sharma'", "filter"),
        ("SELECT customer_id FROM customers WHERE full_name LIKE 'K%'", "filter"),
        ("SELECT card_ref FROM cards WHERE substr(card_no, 1, 1) = '4' AND 1 = 1", "filter"),
        ("SELECT full_name FROM customers ORDER BY balance", "sort"),
        ("SELECT customer_id, balance FROM customers ORDER BY 2 DESC", "sort"),
        ("SELECT substr(full_name, 1, 1) AS f, count(*) AS n FROM customers GROUP BY 1", "group"),
        ("SELECT count(*) AS n FROM customers GROUP BY balance > 100000", "group"),
        ("SELECT c.customer_id FROM customers c JOIN cards k ON c.full_name = k.col_7", "join"),
        ("SELECT full_name, row_number() OVER (ORDER BY balance) AS r FROM customers", "sort"),
        (
            "SELECT 1 AS n FROM customers WHERE customer_id IN (SELECT customer_id FROM cards WHERE col_7 = 'X')",
            "filter",
        ),
        ("WITH t AS (SELECT full_name AS x FROM customers) SELECT x FROM t WHERE x LIKE 'K%'", "filter"),
        ("SELECT customer_id FROM customers LIMIT (SELECT count(*) FROM customers WHERE balance > 100000)", "filter"),
        ("SELECT ticket_id FROM support_tickets WHERE ticket_text LIKE '%4111%'", "filter"),
        ("SELECT full_name FROM customers UNION SELECT col_7 FROM cards ORDER BY 1", "sort"),
    ],
)
def test_inference_guard_blocks_non_admin(eng, sql, verb):
    for role in ("support", "analyst"):
        res = eng.run(sql, role, record=False)
        assert res["blocked"] and not res["ok"], (role, res)
        if verb:
            assert f"cannot {verb}" in res["reason"]
    assert eng.run(sql, "admin", record=False)["ok"]


def test_inference_guard_allows_is_null(eng):
    for role in ("support", "analyst"):
        for sql in (
            "SELECT count(*) AS n FROM customers WHERE full_name IS NOT NULL",
            "SELECT count(*) AS n FROM cards WHERE card_no IS NULL",
        ):
            assert eng.run(sql, role, record=False)["ok"], (role, sql)


def test_sqlite_is_operator_is_not_treated_as_null_check(eng):
    res = eng.run("SELECT customer_id FROM customers WHERE full_name IS 'Komal Sharma'", "support", record=False)
    assert res["blocked"]


def test_analyst_can_group_and_join_on_tokens(eng):
    res = eng.run("SELECT mobile, count(*) AS n FROM customers GROUP BY mobile", "analyst", record=False)
    assert res["ok"] and all(r[0].startswith("tok_") for r in res["rows"])
    sql = "SELECT k.card_ref FROM customers c JOIN cards k ON c.customer_id = k.customer_id"
    assert eng.run(sql, "analyst", record=False)["ok"]
    # ...but filtering on a tokenised column would reveal it
    assert eng.run("SELECT 1 AS one FROM customers WHERE customer_id = 1001", "analyst", record=False)["blocked"]
    assert eng.run("SELECT 1 AS one FROM customers WHERE customer_id = 1001", "support", record=False)["ok"]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers (customer_id) VALUES (1)",
        "UPDATE customers SET full_name = 'x'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "CREATE TABLE x AS SELECT * FROM customers",
        "ALTER TABLE customers ADD COLUMN y TEXT",
        "WITH t AS (SELECT 1) INSERT INTO customers (customer_id) SELECT * FROM t",
        "PRAGMA table_info(customers)",
        "SELECT * INTO copy FROM customers",
    ],
)
def test_writes_blocked_for_every_role(eng, sql):
    for role in engine.ROLES:
        res = eng.run(sql, role, record=False)
        assert res["blocked"] and not res["ok"], (role, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 AS one; DROP TABLE customers",
        "SELECT * FROM customers; DROP TABLE cards",
        "SELECT 1 AS one; SELECT 2 AS two",
        "SELECT 1 AS one;; DELETE FROM customers;",
        "SELECT 1 AS one; /* c */ DROP TABLE customers",
    ],
)
def test_stacked_statements_are_refused_whole(eng: engine.Engine, backend: backends.Backend, sql: str) -> None:
    for role in engine.ROLES:
        res = eng.run(sql, role, record=False)
        assert res["ok"] is False and res["blocked"] is True, (role, res)
        assert res["reason"].startswith("only one statement per query")
        assert "rows" not in res
    assert backend.execute("SELECT count(*) FROM customers")[1] == [[12]]
    assert backend.execute("SELECT count(*) FROM cards")[1] == [[12]]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 AS one;",
        "SELECT 1 AS one;;",
        "SELECT 1 AS one; -- DROP TABLE customers",
        "SELECT 1 AS one; /* trailing note */",
        "-- leading comment\nSELECT 1 AS one;",
    ],
)
def test_trailing_semicolons_and_comments_still_run(eng: engine.Engine, sql: str) -> None:
    res = eng.run(sql, "support", record=False)
    assert res["ok"] is True, res
    assert res["rows"] == [[1]]


@pytest.mark.parametrize("sql", ["", ";", "-- only a comment", "/* nothing */ ;"])
def test_empty_input_is_an_error(eng: engine.Engine, sql: str) -> None:
    res = eng.run(sql, "support", record=False)
    assert res["ok"] is False and res["blocked"] is False
    assert res["error"] == "SQL syntax error: no statement to run"


def test_stacked_statements_refused_on_writable_playground(isolated_state: Path) -> None:
    pb = backends.PlaygroundBackend().connect()
    try:
        eng = engine.Engine(pb, scanner.scan(pb))
        before = pb.execute("SELECT count(*) FROM test_customers")[1]
        for sql in (
            "SELECT full_name FROM test_customers; DROP TABLE test_customers",
            "INSERT INTO test_customers (full_name) VALUES ('x'); DROP TABLE test_customers",
        ):
            res = eng.run(sql, "admin", record=False)
            assert res["blocked"] is True and "only one statement" in res["reason"], res
        assert "test_customers" in pb.tables()
        assert pb.execute("SELECT count(*) FROM test_customers")[1] == before
    finally:
        pb.conn.close()


@pytest.mark.parametrize("fn", ["load_extension('x')", "sleep(5)", "readfile('/etc/passwd')", "pg_sleep(1)"])
def test_banned_functions(eng, fn):
    res = eng.run(f"SELECT {fn} AS v", "admin", record=False)
    assert res["blocked"] and "is blocked" in res["reason"]


@pytest.mark.parametrize("sql", ["SELECT FROM WHERE (", "SELECT 1 FROM customers WHERE"])
def test_syntax_error_is_reported_not_raised(eng, sql):
    res = eng.run(sql, "support", record=False)
    assert res["ok"] is False and res["blocked"] is False
    assert res["error"].startswith("SQL syntax error")


def test_unknown_table_is_an_error(eng):
    res = eng.run("SELECT * FROM no_such_table", "admin", record=False)
    assert res["ok"] is False and not res["blocked"]


def test_auto_limit(eng, monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 3)
    tree, _ = eng.plan("SELECT customer_id FROM customers", "admin")
    assert "LIMIT 3" in tree.sql()
    res = eng.run("SELECT customer_id FROM customers", "admin", record=False)
    assert res["row_count"] == 3 and res["truncated"] is True
    res = eng.run("SELECT customer_id FROM customers LIMIT 2", "admin", record=False)
    assert res["row_count"] == 2 and res["truncated"] is False
    assert eng.run("SELECT customer_id FROM customers LIMIT 50", "admin", record=False)["row_count"] == 3


def test_runs_and_blocks_are_audited(eng):
    eng.run("SELECT full_name FROM customers", "support", user="u1")
    eng.run("SELECT full_name FROM customers ORDER BY balance", "analyst", user="u2")
    log = audit.read()
    assert log["chain_ok"] and log["total"] == 2
    blocked, ok = log["records"]  # newest first
    assert ok["status"] == "OK" and ok["masked"] == {"full_name": "PARTIAL"} and ok["rows"] == 12
    assert blocked["status"] == "BLOCKED" and blocked["user"] == "u2"


def test_browse_filter_goes_through_guard(eng):
    res = eng.browse("customers", "analyst", filter_col="full_name", filter_val="Komal Sharma")
    assert res["blocked"]
    res = eng.browse("customers", "support", limit=5)
    assert res["ok"] and res["total"] == 12 and len(res["rows"]) == 5
    assert eng.browse("nope", "admin")["error"] == "unknown table nope"


def test_ai_spans_used_for_free_text(eng, monkeypatch):
    text = "Spoke to Arjun Mehta about the loan EMI, he said it would be sorted out within a week."
    monkeypatch.setattr(ml.client, "enabled", True)
    monkeypatch.setitem(ml.client.info, "status", "ready")
    monkeypatch.setattr(ml.client, "spans_for", lambda texts: {text: [(54, 60, "person")]} if text in texts else {})
    res = eng.run("SELECT ticket_text FROM support_tickets WHERE ticket_id = 6", "admin", record=False)
    assert res["ai"] is True
    assert [m[2] for m in res["marks"]["0:0"]] == ["person", "person (AI)"]


def test_playground_accepts_writes_and_masks_reads(isolated_state):
    pb = backends.PlaygroundBackend().connect()
    try:
        assert (isolated_state / "playground.db").exists()
        eng = engine.Engine(pb, scanner.scan(pb))
        res = eng.run(
            "INSERT INTO test_customers (full_name, card_no) VALUES ('Komal Sharma', '4111111111111111')",
            "support",
            record=False,
        )
        assert res["ok"] and res["write"] and res["affected"] == 1
        rows = eng.run("SELECT card_no FROM test_customers", "support", record=False)["rows"]
        assert ["4111-XXXX-XXXX-1111"] in rows
    finally:
        pb.conn.close()


def test_bank_backend_is_physically_read_only(backend):
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        backend.execute("DELETE FROM customers")


KYC_DOCS = [
    '{"pan": "ABCPK1234F", "father_name": "Rahul Verma", "src": "BRANCH"}',
    '{"pan": "BCDPF2345G", "father_name": "Arjun Mehta", "src": "VIDEO"}',
    '{"pan": "CDEPJ3456H", "father_name": "Rohan Das", "src": "BRANCH"}',
    '{"pan": "DEFPL4567J", "father_name": "Vikram Singh", "src": "VIDEO"}',
    '{"pan": "EFGPM5678K", "father_name": "Suresh Kumar", "src": "BRANCH"}',
]


@pytest.fixture
def kyc_eng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[engine.Engine]:
    """Engine over a throwaway read-only SQLite file holding a JSON column (kyc_records.kyc_meta)."""
    import sqlite3

    path = tmp_path / "kyc.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE kyc_records (kyc_id INTEGER PRIMARY KEY, kyc_meta TEXT)")
    con.executemany("INSERT INTO kyc_records VALUES (?, ?)", list(enumerate(KYC_DOCS, 1)))
    con.commit()
    con.close()
    monkeypatch.setattr(config, "SQLITE_PATH", path)
    b = backends.SQLiteBackend().connect()
    yield engine.Engine(b, scanner.scan(b))
    b.conn.close()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT json_extract(kyc_meta, '$.pan') AS p FROM kyc_records ORDER BY kyc_id",
        "SELECT kyc_meta -> '$.pan' AS p FROM kyc_records ORDER BY kyc_id",
        "SELECT kyc_meta ->> '$.pan' AS p FROM kyc_records ORDER BY kyc_id",
    ],
)
def test_json_extraction_runs_on_sqlite_and_is_redacted(kyc_eng: engine.Engine, sql: str) -> None:
    admin = kyc_eng.run(sql, "admin", record=False)
    assert admin["ok"] is True, admin
    assert [r[0] for r in admin["rows"]] == ["ABCPK1234F", "BCDPF2345G", "CDEPJ3456H", "DEFPL4567J", "EFGPM5678K"]
    for role in ("support", "analyst"):
        res = kyc_eng.run(sql.replace(" ORDER BY kyc_id", ""), role, record=False)
        assert res["ok"] is True, (role, res)
        assert res["columns"][0]["strategy"] == "REDACT"
        assert {r[0] for r in res["rows"]} == {"[REDACTED]"}


def test_render_spells_json_extract_for_sqlite_only() -> None:
    tree = engine.parse_single("SELECT json_extract(m, '$.a') AS a, m ->> '$.b' AS b FROM t", "sqlite")
    sql = engine.render(tree, "sqlite")
    assert "->" not in sql
    assert sql == "SELECT JSON_EXTRACT(m, '$.a') AS a, JSON_EXTRACT(m, '$.b') AS b FROM t"
    assert engine.render(tree, "mysql") == tree.sql(dialect="mysql")


@pytest.mark.parametrize(
    ("sql", "write"),
    [
        ("SELECT 1", False),
        ("INSERT INTO t VALUES (1)", True),
        ("UPDATE t SET a = 1", True),
        ("CREATE TABLE t (a INT)", True),
        ("DROP TABLE t", True),
        ("SELECT 1; DROP TABLE t", False),  # stacked: refused by the engine, not treated as one write
        ("selec * frm", False),
    ],
)
def test_is_write_classifies_single_statements(sql: str, write: bool) -> None:
    assert engine.is_write(sql, "sqlite") is write
