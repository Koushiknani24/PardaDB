"""Speed caches (plan LRU, one-pass lineage, cached spans) must never change a result."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import sqlglot

from pardadb import backends, detectors, engine, server

QUERIES = [
    "SELECT COUNT(*) FROM customers",
    "SELECT * FROM customers",
    "SELECT full_name AS n, upper(mobile) AS m FROM customers",
    "SELECT c.full_name, k.card_no FROM customers c JOIN cards k ON c.customer_id = k.customer_id",
    "SELECT x.full_name FROM (SELECT full_name FROM customers) x",
    "SELECT full_name FROM customers UNION SELECT ticket_text FROM support_tickets",
    "WITH t AS (SELECT mobile FROM customers) SELECT mobile FROM t",
    "SELECT ticket_text FROM support_tickets",
]


def _stable(res: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in res.items() if k != "elapsed_ms"}


def _per_column(eng: engine.Engine, tree: sqlglot.exp.Expression) -> list[tuple[str, set[tuple[str, str]], bool]]:
    """The old way: one full lineage() call per output column."""
    out = []
    for name in tree.named_selects:
        node = eng._lineage(name, tree.copy())
        out.append((name, engine._source_columns(node), engine._is_direct(node)))
    return out


@pytest.mark.parametrize("sql", QUERIES)
def test_one_pass_lineage_matches_per_column_lineage(eng: engine.Engine, sql: str) -> None:
    tree, _ = eng.plan(sql, "admin")
    expected = _per_column(eng, tree)
    got = [(n, engine._source_columns(node), engine._is_direct(node)) for n, node in eng._output_lineage(tree.copy())]
    assert got == expected
    assert [n for n, _, _ in got] == tree.named_selects


@pytest.mark.parametrize("role", engine.ROLES)
@pytest.mark.parametrize("sql", QUERIES)
def test_cached_plan_gives_the_same_result(
    backend: backends.Backend, catalog: dict[str, Any], sql: str, role: str
) -> None:
    eng = engine.Engine(backend, catalog)
    first = eng.run(sql, role, record=False)
    assert (sql, role) in eng._plans
    again = eng.run(sql, role, record=False)
    fresh = engine.Engine(backend, catalog).run(sql, role, record=False)
    assert _stable(first) == _stable(again) == _stable(fresh)


def test_plan_cache_is_bounded_and_skips_refusals(eng: engine.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "PLAN_CACHE_SIZE", 3)
    for i in range(5):
        assert eng.run(f"SELECT {i} AS v", "admin", record=False)["ok"]
    assert list(eng._plans) == [(f"SELECT {i} AS v", "admin") for i in (2, 3, 4)]
    res = eng.run("SELECT full_name FROM customers WHERE mobile = '1'", "support", record=False)
    assert res["blocked"] is True
    assert eng.run("SELECT nope FROM missing", "admin", record=False)["ok"] is False
    assert len(eng._plans) == 3 and all(role == "admin" for _, role in eng._plans)


def test_result_does_not_share_lists_with_the_cached_plan(eng: engine.Engine) -> None:
    res = eng.run("SELECT full_name FROM customers", "support", record=False)
    res["columns"][0]["source"].append("tampered")
    again = eng.run("SELECT full_name FROM customers", "support", record=False)
    assert again["columns"][0]["source"] == ["customers.full_name"]


def test_parallel_queries_match_serial_results(backend: backends.Backend, catalog: dict[str, Any]) -> None:
    eng = engine.Engine(backend, catalog)
    jobs = [(sql, role) for sql in QUERIES for role in engine.ROLES]
    serial = {job: _stable(engine.Engine(backend, catalog).run(*job, record=False)) for job in jobs}
    got: dict[tuple[str, str], dict[str, Any]] = {}
    lock = threading.Lock()

    def worker(chunk: list[tuple[str, str]]) -> None:
        for job in chunk * 3:
            res = _stable(eng.run(*job, record=False))
            with lock:
                got[job] = res

    threads = [threading.Thread(target=worker, args=(jobs[i::4],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == serial


def test_schema_change_installs_an_engine_with_an_empty_plan_cache(isolated_state: Path) -> None:
    rt = server.Runtime(["playground"])
    rt.connect("playground")
    rt.run_scan("playground")
    try:
        old = rt.engines["playground"]
        before = old.run("SELECT * FROM test_customers", "support", record=False)
        assert old._plans and "speed_col" not in [c["name"] for c in before["columns"]]
        write = old.run("ALTER TABLE test_customers ADD COLUMN speed_col TEXT", "admin", record=False)
        rt.after_write(write)
        new = rt.engines["playground"]
        assert new is not old and not new._plans
        after = new.run("SELECT * FROM test_customers", "support", record=False)
        assert [c["name"] for c in after["columns"]][-1] == "speed_col"
    finally:
        rt.conns["playground"].conn.close()


def test_cached_spans_match_a_fresh_scan() -> None:
    idx = detectors.NameIndex(["Komal Sharma", "Arjun Mehta"])
    text = "Spoke to Arjun Mehta, card 4111 1111 1111 1111, call 9876543210.\nRegards,\nKomal Sharma"
    first = detectors.find_spans(text, idx)
    detectors._rule_spans.cache_clear()
    fresh = detectors.find_spans(text, detectors.NameIndex(["Komal Sharma", "Arjun Mehta"]))
    assert first == fresh == detectors.find_spans(text, idx)
    assert {text[a:b] for a, b, ent in first if ent == "person"} == {"Arjun Mehta", "Komal Sharma"}


def test_name_index_hands_out_fresh_lists() -> None:
    idx = detectors.NameIndex(["Komal Sharma"])
    idx.find("hi Komal Sharma").append((0, 1, "junk"))
    assert idx.find("hi Komal Sharma") == [(3, 15, "person")]


def test_rule_span_cache_is_bounded() -> None:
    detectors._rule_spans.cache_clear()
    for i in range(detectors.SPAN_CACHE_SIZE + 10):
        detectors.find_spans(f"ticket {i}")
    assert detectors._rule_spans.cache_info().currsize == detectors.SPAN_CACHE_SIZE
