"""Read-only database adapters. Each exposes the same small interface, so the
engine never cares which database it talks to. Values are normalised (money,
dates, JSON) so every backend returns identical masked output."""

from __future__ import annotations

import contextlib
import datetime as dt
import decimal
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from . import config

try:
    import pymysql
except ImportError:  # optional driver: MySQL is then simply unavailable
    pymysql = None
try:
    import pg8000.dbapi
except ImportError:  # optional driver: PostgreSQL is then simply unavailable
    pg8000 = None

log = logging.getLogger(__name__)

Row = list[Any]
Schema = dict[str, dict[str, str]]  # {table: {column: declared type}}
T = TypeVar("T")
B = TypeVar("B", bound="Backend")

# The database playground lives in FULL_CODE/db/ as plain SQL so the schema is reviewable.
PLAYGROUND_SQL = (config.REPO / "db" / "playground_schema.sql", config.REPO / "db" / "playground_seed.sql")


# MySQL client codes meaning the socket is gone: server has gone away, lost during query / at handshake.
MYSQL_LOST_CODES = frozenset({2006, 2013, 2055})
# pg8000 raises InterfaceError with exactly these messages once its socket is unusable.
PG_LOST_MESSAGES = frozenset({"network error", "connection is closed"})


def driver_errors() -> tuple[type[BaseException], ...]:
    """Exception types any backend can raise for connection or query problems."""
    errors: list[type[BaseException]] = [sqlite3.Error, OSError]
    if pymysql is not None:
        errors.append(pymysql.err.MySQLError)
    if pg8000 is not None:
        errors.append(pg8000.dbapi.Error)
    return tuple(errors)


DRIVER_ERRORS = driver_errors()


def connection_lost(e: BaseException) -> bool:
    """True when `e` means the connection itself is unusable (not a bad statement)."""
    if isinstance(e, sqlite3.ProgrammingError):
        return "closed" in str(e)  # "Cannot operate on a closed database."
    if pymysql is not None and isinstance(e, pymysql.err.InterfaceError):
        return True
    if pymysql is not None and isinstance(e, pymysql.err.OperationalError):
        return bool(e.args) and e.args[0] in MYSQL_LOST_CODES
    if pg8000 is not None and isinstance(e, pg8000.dbapi.InterfaceError):
        return str(e) in PG_LOST_MESSAGES
    return False


def timed_out(e: BaseException) -> bool:
    """True when the statement hit the driver's read timeout (re-running it would double the wait)."""
    return isinstance(e.__cause__, TimeoutError) or "timed out" in str(e)


def normalise(v: Any) -> Any:
    """Driver value -> plain JSON-friendly value, identical across SQLite, MySQL and PostgreSQL."""
    if isinstance(v, decimal.Decimal):
        v = float(v)
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 2)
    if isinstance(v, dt.datetime):
        return v.isoformat(timespec="seconds")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return v


class Backend:
    """Base adapter: subclasses supply _open(), tables(), columns() and dialect quirks.

    One connection is shared by HTTP threads, the watcher and TUI workers, and no driver allows
    concurrent use of a connection, so every use goes through run(), which holds `self.lock`.
    """

    name = label = dialect = ""
    detail = ""
    writable = False

    def __init__(self) -> None:
        self.conn: Any = None  # DB-API connection of whichever driver the subclass uses
        self._schema: Schema | None = None
        self.lock = threading.RLock()

    # -- interface -------------------------------------------------------
    def _open(self) -> None:
        """Open `self.conn` and apply the session settings (read-only, timeouts)."""
        raise NotImplementedError

    def quote(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    def sql(self, template: str, limit: int | None = None, **identifiers: str) -> str:
        """Fill a constant SQL template with table/column names, always quoted for this dialect.

        Only identifiers go in here (never values: those use driver parameters), so a name read
        from the database catalog cannot change the meaning of the statement. `{random}` becomes
        this dialect's random function and `{limit}` a whole number.
        """
        fill = {key: self.quote(name) for key, name in identifiers.items()}
        fill["random"] = self.random_sql()
        if limit is not None:
            fill["limit"] = int(limit)
        return template.format(**fill)

    def tables(self) -> list[str]:
        raise NotImplementedError

    def columns(self, table: str) -> list[tuple[str, str]]:
        """[(column, declared type)] in table order."""
        raise NotImplementedError

    def row_count(self, table: str) -> int:
        return self.execute(self.sql("SELECT COUNT(*) FROM {t}", t=table))[1][0][0]

    def random_sql(self) -> str:
        return "RANDOM()"

    # -- connection ------------------------------------------------------
    def connect(self: B) -> B:
        """Open the connection; returns self so `SQLiteBackend().connect()` chains."""
        with self.lock:
            self._open()
        return self

    def reconnect(self) -> None:
        """Drop the current connection (it may already be dead) and open a new one."""
        with self.lock:
            with contextlib.suppress(*DRIVER_ERRORS):
                self.conn.close()
            self._open()

    def run(self, work: Callable[[Any], T], retry: bool = True) -> T:
        """Call `work(cursor)` holding the connection lock.

        If the connection turns out to be gone, it is reopened, and `work` is run once more when
        `retry` is set (reads only: writes and timed-out statements are never re-run). A second
        failure propagates to the caller.
        """
        with self.lock:
            try:
                return work(self.conn.cursor())
            except DRIVER_ERRORS as e:
                if not connection_lost(e):
                    raise
                self.reconnect()
                if not retry or timed_out(e):
                    raise
                log.warning("%s: connection lost (%s); reconnected, retrying the statement once", self.name, e)
                return work(self.conn.cursor())

    # -- shared ----------------------------------------------------------
    def execute(self, sql: str, params: tuple[Any, ...] = (), limit: int | None = None) -> tuple[list[str], list[Row]]:
        """Run one read statement; return (column names, normalised rows), at most `limit` rows when given."""

        def work(cur: Any) -> tuple[list[str], list[Row]]:
            cur.execute(sql, params) if params else cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(limit) if limit else cur.fetchall()
            return cols, [[normalise(v) for v in r] for r in rows]

        return self.run(work)

    def schema(self) -> Schema:
        with self.lock:
            if self._schema is None:
                self._schema = {t: {c: (ty or "TEXT") for c, ty in self.columns(t)} for t in self.tables()}
            return self._schema

    def refresh_schema(self) -> Schema:
        """Forget the cached schema and read it again, picking up new tables and columns."""
        with self.lock:
            self._schema = None
            return self.schema()

    def sample(self, table: str, column: str, n: int = 200) -> list[Any]:
        """Random non-null sample: scan cost depends on n, not on table size."""
        sql = self.sql(
            "SELECT {c} FROM {t} WHERE {c} IS NOT NULL ORDER BY {random} LIMIT {limit}", limit=n, c=column, t=table
        )
        return [r[0] for r in self.execute(sql)[1]]

    def null_rate(self, table: str, column: str, total: int) -> float:
        if not total:
            return 1.0
        n = self.execute(self.sql("SELECT COUNT({c}) FROM {t}", c=column, t=table))[1][0][0]
        return round(1 - n / total, 3)

    def fingerprint(self, table: str) -> tuple[str, int]:
        """SHA-256 over every raw row in primary-key order: proves the data never changed."""
        cols = [c for c, _ in self.columns(table)]
        sql = self.sql("SELECT * FROM {t} ORDER BY {c}", t=table, c=cols[0])

        def work(cur: Any) -> tuple[str, int]:
            cur.execute(sql)
            h, n = hashlib.sha256(), 0
            while rows := cur.fetchmany(2000):
                for r in rows:
                    h.update(repr([str(normalise(v)) for v in r]).encode())
                n += len(rows)
            return h.hexdigest(), n

        return self.run(work)

    def ping(self) -> float:
        t0 = time.perf_counter()
        self.execute("SELECT 1")
        return round((time.perf_counter() - t0) * 1000, 2)


class SQLiteBackend(Backend):
    name, label, dialect = "sqlite", "SQLite", "sqlite"

    def _open(self) -> None:
        path = config.SQLITE_PATH
        self.detail = f"{path.name} (read-only)"
        # mode=ro: the engine physically cannot modify the file
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only = ON")

    def tables(self) -> list[str]:
        return [
            r[0]
            for r in self.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )[1]
        ]

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[1], r[2]) for r in self.execute(self.sql("PRAGMA table_info({t})", t=table))[1]]


class MySQLBackend(Backend):
    name, label, dialect = "mysql", "MySQL 8", "mysql"

    def _open(self) -> None:
        c = config.MYSQL
        self.detail = f"{c['host']}:{c['port']}/{c['database']} (read-only session)"
        self.db = c["database"]
        self.conn = pymysql.connect(
            **c,
            autocommit=True,
            charset="utf8mb4",
            connect_timeout=3,
            read_timeout=config.DB_TIMEOUT_S,
            write_timeout=config.DB_TIMEOUT_S,
        )
        self.conn.cursor().execute("SET SESSION TRANSACTION READ ONLY")

    def quote(self, ident: str) -> str:
        return "`" + ident.replace("`", "``") + "`"

    def random_sql(self) -> str:
        return "RAND()"

    def tables(self) -> list[str]:
        return [
            r[0]
            for r in self.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name",
                (self.db,),
            )[1]
        ]

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [
            (r[0], r[1])
            for r in self.execute(
                "SELECT column_name, column_type FROM information_schema.columns "
                "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
                (self.db, table),
            )[1]
        ]


class PostgresBackend(Backend):
    name, label, dialect = "postgres", "PostgreSQL 16", "postgres"

    def _open(self) -> None:
        c = config.POSTGRES
        self.detail = f"{c['host']}:{c['port']}/{c['database']} (read-only session)"
        self.conn = pg8000.dbapi.connect(**c, timeout=config.DB_TIMEOUT_S)
        self.conn.autocommit = True
        self.conn.cursor().execute("SET default_transaction_read_only = on")

    def tables(self) -> list[str]:
        return [
            r[0]
            for r in self.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
            )[1]
        ]

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [
            (r[0], r[1])
            for r in self.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' "
                "AND table_name=%s ORDER BY ordinal_position",
                (table,),
            )[1]
        ]


class PlaygroundBackend(SQLiteBackend):
    """A small writable SQLite file for your own test tables (CREATE / INSERT / UPDATE / ALTER).
    The bank databases stay read-only; only this one accepts writes."""

    name, label, dialect = "playground", "Playground (writable)", "sqlite"
    writable = True

    def _open(self) -> None:
        path = config.state_path("playground.db")
        fresh = not path.exists()
        self.detail = f"{path.name} (your test tables)"
        self.conn = sqlite3.connect(path, check_same_thread=False)
        if fresh:
            for script in PLAYGROUND_SQL:
                self.conn.executescript(script.read_text())
            self.conn.commit()

    def write(self, sql: str) -> int:
        """Run one write statement and commit; returns the affected row count (-1 for DDL).

        Never retried: if the connection drops mid-write it is reopened and the error propagates.
        """

        def work(cur: Any) -> int:
            cur.execute(sql)
            self.conn.commit()
            self._schema = None
            return cur.rowcount

        return self.run(work, retry=False)


BACKENDS: dict[str, type[Backend]] = {
    "sqlite": SQLiteBackend,
    "mysql": MySQLBackend,
    "postgres": PostgresBackend,
    "playground": PlaygroundBackend,
}


def connect(name: str) -> Backend:
    """Instantiate and connect the backend registered under `name`."""
    return BACKENDS[name]().connect()
