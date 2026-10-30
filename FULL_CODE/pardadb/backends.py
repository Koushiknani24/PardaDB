"""Read-only database adapters. Each exposes the same small interface, so the
engine never cares which database it talks to. Values are normalised (money,
dates, JSON) so every backend returns identical masked output."""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
import sqlite3
import time

from . import config

# The database playground lives in FULL_CODE/db/ as plain SQL so the schema is reviewable.
PLAYGROUND_SQL = (config.REPO / "db" / "playground_schema.sql", config.REPO / "db" / "playground_seed.sql")


def driver_errors() -> tuple[type[BaseException], ...]:
    """Exception types any backend can raise for connection or query problems."""
    errors: list[type[BaseException]] = [sqlite3.Error, OSError]
    try:
        import pymysql

        errors.append(pymysql.err.MySQLError)
    except ImportError:
        pass
    try:
        import pg8000.dbapi

        errors.append(pg8000.dbapi.Error)
    except ImportError:
        pass
    return tuple(errors)


def normalise(v):
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
    name = label = dialect = ""
    detail = ""

    def __init__(self):
        self.conn = None
        self._schema = None

    # -- interface -------------------------------------------------------
    def connect(self):
        raise NotImplementedError

    def quote(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    def tables(self) -> list[str]:
        raise NotImplementedError

    def columns(self, table: str) -> list[tuple[str, str]]:
        """[(column, declared type)] in table order."""
        raise NotImplementedError

    def row_count(self, table: str) -> int:
        return self.execute(f"SELECT COUNT(*) FROM {self.quote(table)}")[1][0][0]

    def random_sql(self) -> str:
        return "RANDOM()"

    # -- shared ----------------------------------------------------------
    def execute(self, sql: str, params=(), limit: int | None = None):
        cur = self.conn.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit) if limit else cur.fetchall()
        return cols, [[normalise(v) for v in r] for r in rows]

    def schema(self) -> dict:
        if self._schema is None:
            self._schema = {t: {c: (ty or "TEXT") for c, ty in self.columns(t)} for t in self.tables()}
        return self._schema

    def sample(self, table: str, column: str, n: int = 200) -> list:
        """Random non-null sample: scan cost depends on n, not on table size."""
        q = self.quote
        sql = (
            f"SELECT {q(column)} FROM {q(table)} WHERE {q(column)} IS NOT NULL "
            f"ORDER BY {self.random_sql()} LIMIT {int(n)}"
        )
        return [r[0] for r in self.execute(sql)[1]]

    def null_rate(self, table: str, column: str, total: int) -> float:
        if not total:
            return 1.0
        q = self.quote
        n = self.execute(f"SELECT COUNT({q(column)}) FROM {q(table)}")[1][0][0]
        return round(1 - n / total, 3)

    def fingerprint(self, table: str) -> tuple[str, int]:
        """SHA-256 over every raw row in primary-key order: proves the data never changed."""
        cols = [c for c, _ in self.columns(table)]
        cur = self.conn.cursor()
        cur.execute(f"SELECT * FROM {self.quote(table)} ORDER BY {self.quote(cols[0])}")
        h, n = hashlib.sha256(), 0
        while True:
            rows = cur.fetchmany(2000)
            if not rows:
                break
            for r in rows:
                h.update(repr([str(normalise(v)) for v in r]).encode())
                n += 1
        return h.hexdigest(), n

    def ping(self) -> float:
        t0 = time.perf_counter()
        self.execute("SELECT 1")
        return round((time.perf_counter() - t0) * 1000, 2)


class SQLiteBackend(Backend):
    name, label, dialect = "sqlite", "SQLite", "sqlite"

    def connect(self):
        path = config.SQLITE_PATH
        self.detail = f"{path.name} (read-only)"
        # mode=ro: the engine physically cannot modify the file
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.execute("PRAGMA query_only = ON")
        return self

    def tables(self):
        return [
            r[0]
            for r in self.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )[1]
        ]

    def columns(self, table):
        return [(r[1], r[2]) for r in self.execute(f"PRAGMA table_info({self.quote(table)})")[1]]


class MySQLBackend(Backend):
    name, label, dialect = "mysql", "MySQL 8", "mysql"

    def connect(self):
        import pymysql

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
        return self

    def quote(self, ident):
        return "`" + ident.replace("`", "``") + "`"

    def random_sql(self):
        return "RAND()"

    def tables(self):
        return [
            r[0]
            for r in self.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name",
                (self.db,),
            )[1]
        ]

    def columns(self, table):
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

    def connect(self):
        import pg8000.dbapi

        c = config.POSTGRES
        self.detail = f"{c['host']}:{c['port']}/{c['database']} (read-only session)"
        self.conn = pg8000.dbapi.connect(**c, timeout=config.DB_TIMEOUT_S)
        self.conn.autocommit = True
        self.conn.cursor().execute("SET default_transaction_read_only = on")
        return self

    def tables(self):
        return [
            r[0]
            for r in self.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
            )[1]
        ]

    def columns(self, table):
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

    def connect(self):
        path = config.state_path("playground.db")
        fresh = not path.exists()
        self.detail = f"{path.name} (your test tables)"
        self.conn = sqlite3.connect(path, check_same_thread=False)
        if fresh:
            for script in PLAYGROUND_SQL:
                self.conn.executescript(script.read_text())
            self.conn.commit()
        return self

    def write(self, sql: str) -> int:
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()
        self._schema = None
        return cur.rowcount


BACKENDS = {
    "sqlite": SQLiteBackend,
    "mysql": MySQLBackend,
    "postgres": PostgresBackend,
    "playground": PlaygroundBackend,
}


def connect(name: str) -> Backend:
    return BACKENDS[name]().connect()
