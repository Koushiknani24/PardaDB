"""Reliability: shared connections under threads, reconnect after a drop, watcher recovery,
atomic event numbers and the request-body size limit."""

from __future__ import annotations

import http.client
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pg8000.dbapi
import pymysql
import pytest

from pardadb import backends, config, http_api, server

THREADS = 8


def run_threads(target: Callable[[int], None], n: int = THREADS) -> None:
    """Start `n` threads on target(i), wait for all, and fail if any is still running."""
    threads = [threading.Thread(target=target, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)


# ------------------------------------------------------------------ a scripted fake driver
class FakeCursor:
    """Raises the connection's queued errors in order, otherwise returns one row: [[1]]."""

    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn
        self.description = [("x",)]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.conn.statements.append(sql)
        if self.conn.backend.failures:
            raise self.conn.backend.failures.pop(0)
        self.conn.backend.enter()

    def fetchall(self) -> list[tuple[int]]:
        return [(1,)]

    fetchmany = fetchall


class FakeConn:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.statements: list[str] = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self) -> None:
        pass


class FakeBackend(backends.Backend):
    """A backend whose 'driver' records opens and detects two threads inside it at once."""

    name = "fake"

    def __init__(self, failures: list[BaseException] | None = None) -> None:
        super().__init__()
        self.failures = list(failures or [])
        self.opened = 0
        self.inside = 0
        self.overlaps = 0

    def _open(self) -> None:
        self.opened += 1
        self.conn = FakeConn(self)

    def enter(self) -> None:
        """Simulate driver work; count any moment where a second thread is using the connection."""
        self.inside += 1
        self.overlaps += self.inside > 1
        time.sleep(0.0005)
        self.inside -= 1


LOST_MYSQL = pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")


# ------------------------------------------------------------------ thread safety
def test_driver_calls_never_overlap_across_threads() -> None:
    b = FakeBackend().connect()
    run_threads(lambda i: [b.execute("SELECT 1") for _ in range(30)])
    assert b.overlaps == 0
    assert len(b.conn.statements) == THREADS * 30


def test_concurrent_queries_on_one_sqlite_backend_are_correct(backend: backends.Backend, eng: Any) -> None:
    expected_ids = [[1001 + i] for i in range(12)]
    expected_fp = backend.fingerprint("customers")
    failures: list[str] = []

    def worker(i: int) -> None:
        for k in range(30):
            ids = backend.execute("SELECT customer_id FROM customers ORDER BY customer_id")[1]
            res = eng.run("SELECT customer_id, pincode FROM customers ORDER BY customer_id", "support", record=False)
            fp = backend.fingerprint("customers") if k % 10 == 0 else expected_fp
            if ids != expected_ids or not res["ok"] or res["row_count"] != 12 or fp != expected_fp:
                failures.append(f"thread {i} query {k}: {res.get('error')}")

    run_threads(worker)
    assert failures == []


# ------------------------------------------------------------------ reconnect
@pytest.fixture
def own_sqlite(bank_db: Path) -> Iterator[backends.SQLiteBackend]:
    b = backends.SQLiteBackend().connect()
    yield b
    b.conn.close()


def test_closed_connection_is_reopened_and_query_retried_once(
    own_sqlite: backends.SQLiteBackend, caplog: pytest.LogCaptureFixture
) -> None:
    old = own_sqlite.conn
    old.close()
    assert own_sqlite.execute("SELECT COUNT(*) FROM customers") == (["COUNT(*)"], [[12]])
    assert own_sqlite.conn is not old
    assert "retrying the statement once" in caplog.text


def test_reopened_sqlite_connection_is_still_read_only(own_sqlite: backends.SQLiteBackend) -> None:
    own_sqlite.conn.close()
    own_sqlite.ping()
    with pytest.raises(sqlite3.OperationalError):
        own_sqlite.execute("DELETE FROM customers")


def test_second_failure_after_reconnect_propagates() -> None:
    b = FakeBackend([LOST_MYSQL, pymysql.err.OperationalError(2006, "MySQL server has gone away")]).connect()
    with pytest.raises(pymysql.err.OperationalError, match="gone away"):
        b.execute("SELECT 1")
    assert b.opened == 2  # initial connect + exactly one reconnect


def test_statement_error_does_not_reconnect(own_sqlite: backends.SQLiteBackend) -> None:
    conn = own_sqlite.conn
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        own_sqlite.execute("SELECT nope FROM customers")
    assert own_sqlite.conn is conn


def test_timed_out_statement_reconnects_but_is_not_rerun() -> None:
    timeout = pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query (timed out)")
    b = FakeBackend([timeout]).connect()
    with pytest.raises(pymysql.err.OperationalError, match="timed out"):
        b.execute("SELECT SLEEP(60)")
    assert b.opened == 2
    assert b.execute("SELECT 1") == (["x"], [[1]])  # the fresh connection works


def test_playground_write_is_not_retried() -> None:
    pb = backends.PlaygroundBackend().connect()
    try:
        before = pb.row_count("test_customers")
        pb.conn.close()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            pb.write("INSERT INTO test_customers (full_name) VALUES ('Rahul Verma')")
        assert pb.row_count("test_customers") == before  # reopened, but the insert never ran
    finally:
        pb.conn.close()


@pytest.mark.parametrize(
    ("error", "lost"),
    [
        (sqlite3.ProgrammingError("Cannot operate on a closed database."), True),
        (sqlite3.ProgrammingError("Incorrect number of bindings supplied."), False),
        (sqlite3.OperationalError("no such table: x"), False),
        (pymysql.err.InterfaceError(0, ""), True),
        (LOST_MYSQL, True),
        (pymysql.err.OperationalError(2006, "MySQL server has gone away"), True),
        (pymysql.err.OperationalError(1045, "Access denied"), False),
        (pymysql.err.ProgrammingError(1064, "You have an error in your SQL syntax"), False),
        (pg8000.dbapi.InterfaceError("network error"), True),
        (pg8000.dbapi.InterfaceError("connection is closed"), True),
        (pg8000.dbapi.InterfaceError("The parameter x can't be of type list."), False),
        (pg8000.dbapi.DatabaseError({"C": "42P01", "M": "relation does not exist"}), False),
    ],
)
def test_connection_lost_recognises_each_driver(error: BaseException, lost: bool) -> None:
    assert backends.connection_lost(error) is lost


# ------------------------------------------------------------------ watcher
@pytest.fixture
def rt(bank_db: Path) -> Iterator[server.Runtime]:
    runtime = server.Runtime(names=("sqlite",))
    yield runtime
    for b in runtime.conns.values():
        b.conn.close()


def test_to_reconnect_includes_never_connected_and_lost_backends() -> None:
    rt = server.Runtime(names=("sqlite", "mysql", "postgres", "playground"))
    rt.conns = {"sqlite": FakeBackend(), "mysql": FakeBackend(), "postgres": FakeBackend()}
    rt.dbs["sqlite"]["status"] = "connected"
    rt.dbs["mysql"]["status"] = "unreachable"
    rt.dbs["postgres"]["status"] = "connected"
    assert rt.to_reconnect() == ["mysql", "playground"]


def test_to_reconnect_is_empty_when_everything_is_up() -> None:
    rt = server.Runtime(names=("sqlite",))
    rt.conns = {"sqlite": FakeBackend()}
    rt.dbs["sqlite"]["status"] = "connected"
    assert rt.to_reconnect() == []


def test_lost_backend_is_reconnected_by_the_watcher(
    rt: server.Runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rt.connect("sqlite")
    rt.fingerprint("sqlite")
    rt.run_scan("sqlite")
    b, eng = rt.conns["sqlite"], rt.engines["sqlite"]

    good_path = config.SQLITE_PATH
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "missing" / "bank.db")  # the database goes away
    b.conn.close()
    rt.check_schemas()
    assert rt.dbs["sqlite"]["status"] == "unreachable"
    assert rt.to_reconnect() == ["sqlite"]

    rt.retry_unreachable()  # still down: stays unreachable, nothing crashes
    assert rt.dbs["sqlite"]["status"] == "unreachable"

    monkeypatch.setattr(config, "SQLITE_PATH", good_path)  # the database is back
    rt.retry_unreachable()
    assert rt.dbs["sqlite"]["status"] == "connected"
    assert rt.conns["sqlite"] is b and rt.engines["sqlite"] is eng  # engine and catalog stay valid
    res = eng.run("SELECT customer_id FROM customers ORDER BY customer_id LIMIT 2", "support", record=False)
    assert res["ok"] and res["rows"] == [[1001], [1002]]
    assert rt.to_reconnect() == []


def test_lost_is_logged_once_not_every_watch_cycle(rt: server.Runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    rt.connect("sqlite")
    rt.run_scan("sqlite")
    monkeypatch.setattr(config, "SQLITE_PATH", Path("/nonexistent/pardadb/bank.db"))
    rt.conns["sqlite"].conn.close()
    for _ in range(3):
        rt.check_schemas()
        rt.retry_unreachable()
    assert sum("lost" in e["msg"] for e in rt.events) == 1


# ------------------------------------------------------------------ events
def test_event_numbers_are_unique_under_threads(rt: server.Runtime) -> None:
    run_threads(lambda i: [rt.event("info", f"thread {i} event {k}") for k in range(100)])
    seqs = [e["seq"] for e in rt.events]
    assert rt.seq == THREADS * 100
    assert seqs == list(range(rt.seq - len(seqs) + 1, rt.seq + 1))  # unique, gap-free, in order


# ------------------------------------------------------------------ request body limit
def test_body_length_accepts_up_to_the_limit() -> None:
    assert http_api.body_length(None) == 0
    assert http_api.body_length(str(http_api.MAX_BODY_BYTES)) == 64 * 1024


def test_oversized_body_is_rejected_with_413_and_a_clear_message() -> None:
    with pytest.raises(http_api.BodyTooLarge, match="^request body is larger than 64 KB$") as err:
        http_api.body_length(str(http_api.MAX_BODY_BYTES + 1))
    assert err.value.status == 413


@pytest.mark.parametrize("header", ["abc", "-1", "1.5"])
def test_bad_content_length_is_a_400(header: str) -> None:
    with pytest.raises(http_api.BadRequest) as err:
        http_api.body_length(header)
    assert err.value.status == 400


def test_http_server_answers_oversized_post_with_413(rt: server.Runtime) -> None:
    httpd = http_api.serve(rt, port=0)  # any free port
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        conn.putrequest("POST", "/api/query")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(http_api.MAX_BODY_BYTES + 1))
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 413
        assert resp.read() == b'{"ok": false, "error": "request body is larger than 64 KB"}'
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
