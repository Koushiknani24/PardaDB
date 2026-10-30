"""PardaDB runtime: connects databases, scans, keeps live monitoring state, and serves
the dashboard + JSON API with the standard-library HTTP server (no framework)."""

from __future__ import annotations

import collections
import csv
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sqlglot
from sqlglot import exp

from . import __version__, audit, backends, config, detectors, engine, ml, scanner

log = logging.getLogger(__name__)
DB_ERRORS = backends.driver_errors()

STATIC = Path(__file__).resolve().parent / "dashboard"
PAGES = {"/": "landing.html", "/dashboard": "index.html", "/index.html": "index.html"}
ASSET_TYPES = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".css": "text/css",
    ".ico": "image/x-icon",
}
PRESETS = [
    # --- SELECT: read bank data, masked per role
    {
        "category": "SELECT (read and mask)",
        "title": "Customer look-up",
        "expect": "Names, phones and balances masked differently per role",
        "sql": "SELECT c.cust_nm, c.mob_no, c.dob_dt, c.pincode, a.account_no, a.curr_bal FROM customers c "
        "JOIN accounts a ON a.customer_id = c.customer_id LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "Cards",
        "expect": "Support sees 4111-XXXX-XXXX-1111, Analyst sees nothing",
        "sql": "SELECT CARD_NO, NAME_ON_CARD, CARD_TYPE, CREDIT_LIMIT FROM cards LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "KYC (Aadhaar, PAN, JSON)",
        "expect": "Pre-masked Aadhaar handled; JSON keys masked",
        "sql": "SELECT customer_id, aadhaar_no, pan_no, kyc_meta FROM kyc_records LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "Hidden PAN column (col_7)",
        "expect": "A column with a meaningless name is still protected",
        "sql": "SELECT app_id, loan_type, declared_income_inr, col_7, remarks FROM loan_applications "
        "WHERE col_7 IS NOT NULL LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "Support tickets (free text)",
        "expect": "PII inside Hinglish text masked span by span",
        "sql": "SELECT ticket_id, category, ticket_text, notes FROM support_tickets LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "Transactions",
        "expect": "Narrations and counterparties masked, amounts bucketed",
        "sql": "SELECT txn_ts, amount_inr, channel, counterparty, narration FROM transactions LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "NRI customers (SSN)",
        "expect": "SSN in three formats, US phones",
        "sql": "SELECT customer_id, ssn, us_phone, us_zip, residency_status FROM international_customers LIMIT 10",
    },
    {
        "category": "SELECT (read and mask)",
        "title": "Analytics on tokens",
        "expect": "Analyst can still count per customer via tokens",
        "sql": "SELECT customer_id, COUNT(*) AS accounts FROM accounts "
        "GROUP BY customer_id ORDER BY accounts DESC LIMIT 10",
    },
    # --- bypass tricks and probes: masked or blocked
    {
        "category": "Bypass and probe attempts",
        "title": "Rename with an alias",
        "expect": "Masked: lineage follows the alias",
        "sql": "SELECT x FROM (SELECT CARD_NO AS x FROM cards) t LIMIT 5",
    },
    {
        "category": "Bypass and probe attempts",
        "title": "Cut with SUBSTR",
        "expect": "Redacted: derived from a masked column",
        "sql": "SELECT SUBSTR(CARD_NO, 1, 12) AS first_digits FROM cards LIMIT 5",
    },
    {
        "category": "Bypass and probe attempts",
        "title": "UNION two PAN columns",
        "expect": "Masked: both branches are tagged",
        "sql": "SELECT col_7 AS pan FROM loan_applications WHERE col_7 IS NOT NULL "
        "UNION SELECT pan_no FROM kyc_records LIMIT 5",
    },
    {
        "category": "Bypass and probe attempts",
        "title": "WHERE on a card number",
        "expect": "Blocked for Support and Analyst",
        "sql": "SELECT customer_id FROM cards WHERE CARD_NO LIKE '4111%'",
    },
    {
        "category": "Bypass and probe attempts",
        "title": "ORDER BY balance",
        "expect": "Blocked: sorting reveals rank",
        "sql": "SELECT customer_id, acct_type FROM accounts ORDER BY curr_bal DESC LIMIT 5",
    },
    # --- writes on bank data: always refused
    {
        "category": "Writes on bank data (blocked)",
        "title": "DELETE bank rows",
        "expect": "Blocked: read-only by design",
        "sql": "DELETE FROM accounts",
    },
    {
        "category": "Writes on bank data (blocked)",
        "title": "UPDATE a balance",
        "expect": "Blocked: read-only by design",
        "sql": "UPDATE accounts SET curr_bal = 0 WHERE account_no = '53545092010380'",
    },
    {
        "category": "Writes on bank data (blocked)",
        "title": "INSERT a fake customer",
        "expect": "Blocked: read-only by design",
        "sql": "INSERT INTO customers (customer_id, cust_nm) VALUES ('CIF-HACKER01', 'Mallory')",
    },
    {
        "category": "Writes on bank data (blocked)",
        "title": "DROP a table",
        "expect": "Blocked: read-only by design",
        "sql": "DROP TABLE transactions",
    },
    # --- playground: your own test tables
    {
        "category": "Playground: CREATE",
        "title": "Create your own table",
        "expect": "A new test table; PardaDB scans it automatically",
        "sql": "CREATE TABLE my_complaints "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, cust_nm TEXT, mob_no TEXT, complaint TEXT)",
    },
    {
        "category": "Playground: INSERT",
        "title": "Insert into your table",
        "expect": "Your own values and free text",
        "sql": "INSERT INTO my_complaints (cust_nm, mob_no, complaint) VALUES ('Ritika Sharma', '+91-90000 12345', "
        "'Bhai mera card 5555 5555 5555 4444 block karo, Aadhaar 2345 6789 0123 hai. Regards, Ritika Sharma')",
    },
    {
        "category": "Playground: INSERT",
        "title": "Insert a ticket",
        "expect": "Free text into the sample tickets table",
        "sql": "INSERT INTO test_tickets (customer_name, channel, ticket_text) VALUES ('Arjun Mehta', 'App Chat', "
        "'PAN DEFPA4567J update karna hai, account 98765432109876. Call me on 9000054321')",
    },
    {
        "category": "Playground: INSERT",
        "title": "Insert a customer",
        "expect": "A new row with every PII type",
        "sql": "INSERT INTO test_customers (full_name, mobile, email, card_no, aadhaar, pan, balance, notes) VALUES "
        "('Neha Kulkarni', '+91 99999 55555', 'neha.k@mail.example', '4111 1111 1111 1111', '4567 8901 2345', "
        "'EFGPN5678K', 87000.25, 'Asked to call on 9999955555 after 5 pm')",
    },
    {
        "category": "Playground: SELECT",
        "title": "Read your table",
        "expect": "Masked per role once the re-scan finishes",
        "sql": "SELECT * FROM my_complaints ORDER BY id DESC",
    },
    {
        "category": "Playground: SELECT",
        "title": "Read tickets",
        "expect": "Inserted free text masked span by span",
        "sql": "SELECT * FROM test_tickets ORDER BY id DESC",
    },
    {
        "category": "Playground: SELECT",
        "title": "Read customers",
        "expect": "Sample customers masked per role",
        "sql": "SELECT * FROM test_customers ORDER BY id DESC",
    },
    {
        "category": "Playground: UPDATE / DELETE",
        "title": "Update a ticket",
        "expect": "Change free text; read it again masked",
        "sql": "UPDATE test_tickets "
        "SET ticket_text = 'New number is 9000077777, old card 4012 8888 8888 1881 band karo' "
        "WHERE id = 1",
    },
    {
        "category": "Playground: UPDATE / DELETE",
        "title": "Delete a ticket",
        "expect": "Removes one test row",
        "sql": "DELETE FROM test_tickets WHERE id = 3",
    },
    {
        "category": "Playground: ALTER / DROP",
        "title": "Add a column",
        "expect": "Schema change: the new column is scanned automatically",
        "sql": "ALTER TABLE test_customers ADD COLUMN alt_mobile TEXT",
    },
    {
        "category": "Playground: ALTER / DROP",
        "title": "Drop your table",
        "expect": "Removes my_complaints from the playground",
        "sql": "DROP TABLE my_complaints",
    },
]


def load_users():
    try:
        with open(config.USERS_CSV, newline="", encoding="utf-8") as f:
            return [
                {"username": r["username"], "display_name": r["display_name"], "role": r["role"].lower()}
                for r in csv.DictReader(f)
            ]
    except OSError:
        return [{"username": r, "display_name": r.title(), "role": r} for r in engine.ROLES]


class ProcStats:
    """CPU %, RSS, threads and state-dir size of this process, from /proc (no psutil)."""

    def __init__(self):
        self.tick = os.sysconf("SC_CLK_TCK")
        self.last = (time.monotonic(), self._cpu())

    def _cpu(self):
        try:
            with open("/proc/self/stat") as fh:
                f = fh.read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / self.tick
        except OSError:
            return time.process_time()

    def sample(self):
        now, cpu = time.monotonic(), self._cpu()
        t0, c0 = self.last
        self.last = (now, cpu)
        pct = 100 * (cpu - c0) / (now - t0) if now > t0 else 0.0
        rss = threads = 0
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) / 1024
                    elif line.startswith("Threads:"):
                        threads = int(line.split()[1])
        except OSError:
            pass
        disk = (
            sum(p.stat().st_size for p in config.STATE_DIR.glob("*") if p.is_file()) / 1e6
            if config.STATE_DIR.exists()
            else 0
        )
        return {
            "pid": os.getpid(),
            "cpu_percent": round(pct, 1),
            "rss_mb": round(rss, 1),
            "threads": threads,
            "disk_mb": round(disk, 2),
        }


class Runtime:
    def __init__(self, names=("sqlite", "mysql", "postgres", "playground"), log=None):
        self.started = time.time()
        self.lock = threading.RLock()
        self.log_cb = log
        self.events = collections.deque(maxlen=200)
        self.stats = {"queries": 0, "blocked": 0, "masked_cells": 0}
        self.proc = ProcStats()
        self.proc_cache = (0.0, self.proc.sample())
        self.users = load_users()
        self.policy = engine.load_policy()
        self.order = list(names)
        self.dbs = {
            n: {
                "name": n,
                "label": backends.BACKENDS[n].label,
                "status": "connecting",
                "detail": "",
                "tables": 0,
                "rows": 0,
                "latency_ms": None,
            }
            for n in names
        }
        self.conns, self.catalogs, self.engines, self.name_index, self.before = {}, {}, {}, {}, {}
        self.scan = {
            "state": "idle",
            "backend": None,
            "tables_total": 0,
            "tables_done": 0,
            "columns_total": 0,
            "columns_done": 0,
            "current": None,
            "tagged": 0,
            "elapsed_s": 0,
            "started_at": None,
            "finished_at": None,
        }
        self.active = None

    # ------------------------------------------------------------ events
    def event(self, level, msg):
        self.seq = getattr(self, "seq", 0) + 1
        ev = {"seq": self.seq, "ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg}
        self.events.append(ev)
        log.log({"warn": logging.WARNING, "err": logging.ERROR}.get(level, logging.INFO), msg)
        if self.log_cb:
            self.log_cb(level, msg)

    # ------------------------------------------------------------ lifecycle
    def boot(self):
        for n in self.order:
            self.connect(n)
        if not self.active:
            self.event("warn", "no database reachable yet")
        for n in self.order:
            if n in self.conns:
                self.fingerprint(n)
                self.run_scan(n)
        threading.Thread(target=self.watch, daemon=True, name="schema-watch").start()

    def connect(self, n, quiet=False):
        info = self.dbs[n]
        try:
            b = backends.connect(n)
            tables = b.tables()
            rows = sum(b.row_count(t) for t in tables)
            info.update(status="connected", detail=b.detail, tables=len(tables), rows=rows, latency_ms=b.ping())
            self.conns[n] = b
            self.active = self.active or n
            self.event("ok", f"{info['label']} connected: {len(tables)} tables, {rows:,} rows ({b.detail})")
        except DB_ERRORS as e:
            info.update(status="unreachable", detail=f"{type(e).__name__}: {str(e).splitlines()[0][:80]}")
            if not quiet:
                self.event("warn", f"{info['label']} unreachable (retrying every 10 s): {info['detail']}")

    def fingerprint(self, n):
        b = self.conns[n]
        t0 = time.perf_counter()
        self.before[n] = {t: b.fingerprint(t) for t in b.tables()}
        self.event(
            "info",
            f"{self.dbs[n]['label']}: SHA-256 fingerprints of {len(self.before[n])} tables recorded "
            f"({time.perf_counter() - t0:.1f}s) as proof the data never changes",
        )

    def run_scan(self, n):
        b = self.conns[n]
        schema = b.schema()
        with self.lock:
            self.scan.update(
                state="scanning",
                backend=n,
                tables_total=len(schema),
                tables_done=0,
                columns_total=sum(len(c) for c in schema.values()),
                columns_done=0,
                current=None,
                tagged=0,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                finished_at=None,
            )
        self.event(
            "info",
            f"Scanning {self.dbs[n]['label']}: {self.scan['columns_total']} columns in "
            f"{len(schema)} tables (sample of {config.SAMPLE_SIZE} values each)",
        )
        t0 = time.perf_counter()

        def progress(kind, table, done, total, tables_done, **kw):
            with self.lock:
                self.scan.update(
                    columns_done=done,
                    tables_done=tables_done,
                    current=f"{table}.{kw.get('column')}" if kind == "column" else table,
                    elapsed_s=round(time.perf_counter() - t0, 2),
                )
            if kind == "table":
                self.event("info", f"  scanned {table} ({kw.get('rows', 0):,} rows)")

        cat = scanner.scan(b, progress)
        scanner.save(cat)
        self.install(n, cat)
        s = cat["summary"]
        with self.lock:
            self.scan.update(
                state="done",
                columns_done=self.scan["columns_total"],
                tables_done=len(schema),
                current=None,
                tagged=s["tagged"] + s["review"],
                elapsed_s=cat["elapsed_s"],
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
        self.event(
            "ok",
            f"Scan complete on {self.dbs[n]['label']}: {s['tagged'] + s['review']} sensitive columns "
            f"({s['review']} for review), {s['clean']} clean, in {cat['elapsed_s']}s",
        )

    def install(self, n, cat):
        b = self.conns[n]
        names = []
        for c in cat["columns"]:
            if c["entity"] == "person":
                q = f"SELECT DISTINCT {b.quote(c['column'])} FROM {b.quote(c['table'])}"
                names += [r[0] for r in b.execute(q)[1] if r[0]]
        self.name_index[n] = detectors.NameIndex(names)
        self.catalogs[n] = cat
        self.engines[n] = engine.Engine(b, cat, self.policy, self.name_index[n])

    def watch(self):
        """Every 10 s retry unreachable databases; every 30 s check live ones for new columns."""
        tick = 0
        while True:
            time.sleep(10)
            tick += 1
            for n in self.order:
                if n not in self.conns:
                    self.connect(n, quiet=True)
                    if n in self.conns:
                        self.fingerprint(n)
                        self.run_scan(n)
            if tick % 3:
                continue
            for n, b in list(self.conns.items()):
                try:
                    self.dbs[n]["latency_ms"] = b.ping()
                    b._schema = None
                    live = {(t.lower(), c.lower()) for t, cols in b.schema().items() for c in cols}
                    known = {(c["table"].lower(), c["column"].lower()) for c in self.catalogs[n]["columns"]}
                    new = live - known
                    if new:
                        self.event(
                            "warn", f"{self.dbs[n]['label']}: {len(new)} new column(s) {sorted(new)[:3]}: re-scanning"
                        )
                        self.run_scan(n)
                except DB_ERRORS as e:
                    self.dbs[n]["status"] = "unreachable"
                    self.event("warn", f"{self.dbs[n]['label']} lost: {e}")

    # ------------------------------------------------------------ AI model add-on
    def ml_start(self):
        """Start (or attach to) the AI model sidecar; safe to call from any thread."""
        if ml.client.info["status"] in ("starting", "ready") and ml.client.enabled:
            return
        ml.client.info["status"] = "starting"
        self.event("info", "Activating the AI model (fine-tuned GLiNER) for names and addresses in free text...")
        ok = ml.client.start_sidecar()
        i = ml.client.info
        self.event(
            "ok" if ok else "warn",
            f"AI model {'on' if ok else i['status']}"
            + (f" ({i['device']})" if ok else " - free text masked with rules only"),
        )

    def ml_stop(self):
        ml.client.stop(shutdown_external=True)
        self.event("info", "AI model turned off: free text is masked with rules only")

    # ------------------------------------------------------------ API helpers
    def backend_name(self, q):
        n = (q or {}).get("backend") or self.active
        if n not in self.engines:
            raise KeyError(f"backend '{n}' is not connected or not scanned yet")
        return n

    def route(self, n, sql):
        """Send statements about playground tables (or CREATE TABLE) to the playground."""
        pg = self.conns.get("playground")
        if not pg or n == "playground":
            return n
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            tables = {t.name.lower() for t in tree.find_all(exp.Table)}
        except sqlglot.errors.SqlglotError:
            return n  # the engine reports the syntax error to the user
        mine = {t.lower() for t in self.conns[n].tables()} if n in self.conns else set()
        if isinstance(tree, exp.Create) and not tables & mine:
            return "playground"
        play = {t.lower() for t in pg.tables()}
        return "playground" if tables and tables <= play and not tables & mine else n

    def after_write(self, res):
        if res.get("write") and "playground" in self.conns:
            self.event(
                "ok",
                f"Playground write by {res.get('user') or res['role']}: {max(res['affected'], 0)} row(s); re-scanning",
            )
            self.run_scan("playground")
        return res

    def role_of(self, user, role):
        for u in self.users:
            if user and u["username"] == user:
                return u["role"]
        return role if role in engine.ROLES else "analyst"

    def count(self, res):
        with self.lock:
            self.stats["queries"] += 1
            self.stats["blocked"] += 1 if res.get("blocked") else 0
            self.stats["masked_cells"] += res.get("masked_cells", 0)
            self.stats["last_query"] = {
                "ms": res.get("elapsed_ms"),
                "role": res.get("role"),
                "user": res.get("user"),
                "backend": res.get("backend"),
                "rows": res.get("row_count", 0),
                "at": time.strftime("%H:%M:%S"),
                "status": "OK" if res.get("ok") else ("BLOCKED" if res.get("blocked") else "ERROR"),
            }
        if res.get("blocked"):
            self.event("block", f"BLOCKED {res.get('user') or res['role']}: {res['reason'][:110]}")
        elif res.get("ok"):
            self.event(
                "info",
                f"{res.get('user') or res['role']} ran a query on {res['backend']}: "
                f"{res['row_count']} rows, {res.get('masked_cells', 0)} cells masked",
            )
        return res

    def status(self):
        now = time.monotonic()
        if now - self.proc_cache[0] >= 1.5:
            self.proc_cache = (now, self.proc.sample())
        with self.lock:
            return {
                "version": __version__,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                "uptime_s": int(time.time() - self.started),
                "process": self.proc_cache[1],
                "active_backend": self.active,
                "backends": [self.dbs[n] for n in self.order],
                "scan": dict(self.scan),
                "stats": dict(self.stats),
                "events": list(self.events)[-50:],
                "tags": self.tag_counts(),
                "ml": dict(ml.client.info, url=ml.URL),
            }

    def tag_counts(self):
        cat = self.catalogs.get(self.active) or {}
        out = {}
        for c in cat.get("columns", []):
            if c["tag"]:
                out[c["tag"]] = out.get(c["tag"], 0) + 1
        return out

    def tables(self, n):
        cat = self.catalogs[n]
        by = collections.defaultdict(list)
        for c in cat["columns"]:
            by[c["table"]].append(c)
        out = []
        b = self.conns[n]
        for t, cols in by.items():
            strat = engine.Engine.strategy
            e = self.engines[n]
            out.append(
                {
                    "table": t,
                    "rows": b.row_count(t),
                    "sensitive": sum(1 for c in cols if c["tag"]),
                    "tags": sorted({c["tag"] for c in cols if c["tag"]}),
                    "columns": [
                        {
                            "name": c["column"],
                            "type": c.get("type"),
                            "tag": c["tag"],
                            "entity": c["entity"],
                            "confidence": c["confidence"],
                            "evidence": c["evidence"],
                            "strategy": {r: strat(e, r, c if c["tag"] else None) for r in engine.ROLES},
                        }
                        for c in cols
                    ],
                }
            )
        return {"backend": n, "tables": out}

    def integrity(self, n):
        b = self.conns[n]
        rows, ok = [], True
        for t, (h0, c0) in self.before[n].items():
            h1, c1 = b.fingerprint(t)
            same = h0 == h1 and c0 == c1
            ok &= same
            rows.append({"table": t, "rows": c1, "before": h0, "after": h1, "unchanged": same})
        self.event(
            "ok" if ok else "warn",
            f"Integrity check on {self.dbs[n]['label']}: "
            f"{'all ' + str(len(rows)) + ' tables unchanged' if ok else 'CHANGES DETECTED'}",
        )
        return {"backend": n, "all_unchanged": ok, "tables": rows}


# ------------------------------------------------------------------ request validation
MAX_BODY_BYTES = 64 * 1024
MAX_SQL_CHARS = 20_000
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", *config.EXTRA_HOSTS)


class BadRequest(ValueError):
    """The client sent invalid input; answered with HTTP 400."""


def int_param(q: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(q.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a whole number") from None
    if not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return value


def str_param(q: dict, key: str, max_len: int = 200, required: bool = False) -> str | None:
    value = q.get(key)
    if value is None or value == "":
        if required:
            raise BadRequest(f"'{key}' is required")
        return None
    if not isinstance(value, str) or len(value) > max_len:
        raise BadRequest(f"'{key}' must be text of at most {max_len} characters")
    return value


def sql_param(body: dict) -> str:
    sql = str_param(body, "sql", MAX_SQL_CHARS, required=True)
    return sql.strip()


def make_handler(rt: Runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PardaDB"

        def log_message(self, *a):  # keep the terminal quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "charset" not in ctype else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def _asset(self, path: str):
            """Serve a file from dashboard/assets/ (fixed folder, known types, no path traversal)."""
            root = (STATIC / "assets").resolve()
            target = (STATIC / path.lstrip("/")).resolve()
            ctype = ASSET_TYPES.get(target.suffix.lower())
            if ctype is None or root not in target.parents or not target.is_file():
                return self._send(404, {"error": "not found"})
            return self._send(200, target.read_bytes(), ctype)

        def _check_origin(self):
            """Only answer requests addressed to this machine (blocks DNS-rebinding pages)."""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in ALLOWED_HOSTS:
                raise BadRequest(f"requests must be addressed to 127.0.0.1, not {host!r}")

        def _body(self) -> dict:
            # JSON only: a cross-site form post cannot set this type without a CORS preflight we never allow.
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                raise BadRequest("POST bodies must be sent as application/json")
            n = int_param({"n": self.headers.get("Content-Length") or 0}, "n", 0, 0, MAX_BODY_BYTES)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BadRequest("request body is not valid JSON") from None
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            return body

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                self._check_origin()
                if u.path in PAGES:
                    page = STATIC / PAGES[u.path]
                    if not page.exists():  # no landing page built: go straight to the dashboard
                        page = STATIC / "index.html"
                    return self._send(200, page.read_bytes(), "text/html")
                if u.path.startswith("/assets/"):
                    return self._asset(u.path)
                if u.path == "/api/status":
                    return self._send(200, rt.status())
                if u.path == "/api/users":
                    return self._send(200, rt.users)
                if u.path == "/api/presets":
                    return self._send(200, PRESETS)
                if u.path == "/api/policy":
                    return self._send(200, rt.policy)
                if u.path == "/api/catalog":
                    n = rt.backend_name(q)
                    cat = rt.catalogs[n]
                    return self._send(
                        200,
                        {
                            "backend": n,
                            "summary": cat["summary"],
                            "elapsed_s": cat["elapsed_s"],
                            "columns": cat["columns"],
                        },
                    )
                if u.path == "/api/tables":
                    return self._send(200, rt.tables(rt.backend_name(q)))
                if u.path == "/api/browse":
                    n = rt.backend_name(q)
                    role = rt.role_of(q.get("user"), q.get("role"))
                    res = rt.engines[n].browse(
                        str_param(q, "table", required=True),
                        role,
                        str_param(q, "user"),
                        int_param(q, "offset", 0, 0, 10_000_000),
                        int_param(q, "limit", 25, 1, 100),
                        str_param(q, "filter_col"),
                        str_param(q, "filter_val", 500),
                    )
                    return self._send(200, rt.count(res))
                if u.path == "/api/audit":
                    return self._send(200, audit.read(int_param(q, "limit", 100, 1, 1000)))
                return self._send(404, {"error": "not found"})
            except BadRequest as e:
                return self._send(400, {"ok": False, "error": str(e)})
            except KeyError as e:
                return self._send(409, {"ok": False, "error": str(e).strip("'\"")})
            except Exception:  # last-resort handler: log details, return a safe message
                log.exception("%s %s failed", self.command, self.path)
                return self._send(500, {"ok": False, "error": "internal error; details are in .state/pardadb.log"})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                self._check_origin()
                body = self._body()
                if u.path == "/api/query":
                    sql = sql_param(body)
                    n = rt.route(rt.backend_name(body), sql)
                    user = str_param(body, "user")
                    res = rt.engines[n].run(sql, rt.role_of(user, str_param(body, "role")), user)
                    return self._send(200, rt.after_write(rt.count(res)))
                if u.path == "/api/compare":
                    sql = sql_param(body)
                    n = rt.route(rt.backend_name(body), sql)
                    return self._send(
                        200, {r: rt.count(rt.engines[n].run(sql, r, f"compare:{r}")) for r in engine.ROLES}
                    )
                if u.path == "/api/ml":
                    action = str_param(body, "action", required=True)
                    if action not in ("start", "stop"):
                        raise BadRequest("'action' must be 'start' or 'stop'")
                    if action == "stop":
                        rt.ml_stop()
                    else:
                        threading.Thread(target=rt.ml_start, daemon=True, name="ml").start()
                    return self._send(200, {"ok": True, "ml": dict(ml.client.info)})
                if u.path == "/api/integrity":
                    return self._send(200, rt.integrity(rt.backend_name(body)))
                if u.path == "/api/scan":
                    n = rt.backend_name(body) if body.get("backend") in rt.conns or not body.get("backend") else None
                    if n is None:
                        return self._send(409, {"ok": False, "error": "backend not connected"})
                    threading.Thread(target=rt.run_scan, args=(n,), daemon=True).start()
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "not found"})
            except BadRequest as e:
                return self._send(400, {"ok": False, "error": str(e)})
            except KeyError as e:
                return self._send(409, {"ok": False, "error": str(e).strip("'\"")})
            except Exception:  # last-resort handler: log details, return a safe message
                log.exception("%s %s failed", self.command, self.path)
                return self._send(500, {"ok": False, "error": "internal error; details are in .state/pardadb.log"})

    return Handler


def serve(rt: Runtime, host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), make_handler(rt))
    httpd.daemon_threads = True
    return httpd
