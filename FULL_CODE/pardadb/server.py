"""PardaDB runtime: connects databases, scans, and keeps live monitoring state. The dashboard
and JSON API that expose it live in http_api.py (re-exported here for existing callers)."""

from __future__ import annotations

import collections
import csv
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import sqlglot
from sqlglot import exp

from . import __version__, backends, config, detectors, engine, ml, scanner
from .http_api import (  # noqa: F401 - re-exported: cli.py, the TUIs and tests use server.<name>
    ALLOWED_HOSTS,
    ASSET_TYPES,
    MAX_BODY_BYTES,
    MAX_SQL_CHARS,
    PAGES,
    PRESETS,
    STATIC,
    BadRequest,
    int_param,
    make_handler,
    serve,
    sql_param,
    str_param,
)

log = logging.getLogger(__name__)
DB_ERRORS = backends.driver_errors()


def load_users() -> list[dict[str, str]]:
    """Dashboard users from users.csv, or one user per role when the file is missing."""
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

    def __init__(self) -> None:
        self.tick = os.sysconf("SC_CLK_TCK")
        self.last = (time.monotonic(), self._cpu())

    def _cpu(self) -> float:
        try:
            with open("/proc/self/stat") as fh:
                f = fh.read().rsplit(")", 1)[1].split()
            return (int(f[11]) + int(f[12])) / self.tick
        except OSError:
            return time.process_time()

    def sample(self) -> dict[str, float | int]:
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
    """Shared state behind the API and TUIs: connections, catalogs, engines, stats and events."""

    def __init__(
        self,
        names: Sequence[str] = ("sqlite", "mysql", "postgres", "playground"),
        log: Callable[[str, str], object] | None = None,
    ) -> None:
        self.started = time.time()
        self.lock = threading.RLock()
        self.log_cb = log
        self.events: collections.deque[dict[str, Any]] = collections.deque(maxlen=200)
        self.stats: dict[str, Any] = {"queries": 0, "blocked": 0, "masked_cells": 0}
        self.proc = ProcStats()
        self.proc_cache = (0.0, self.proc.sample())
        self.users = load_users()
        self.policy = engine.load_policy()
        self.order = list(names)
        self.dbs: dict[str, dict[str, Any]] = {
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
        self.conns: dict[str, backends.Backend] = {}
        self.catalogs: dict[str, dict[str, Any]] = {}
        self.engines: dict[str, engine.Engine] = {}
        self.name_index: dict[str, detectors.NameIndex] = {}
        # per backend: table -> (SHA-256, row count) taken at connect time, re-checked by integrity()
        self.before: dict[str, dict[str, tuple[str, int]]] = {}
        self.scan: dict[str, Any] = {
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
        self.active: str | None = None

    # ------------------------------------------------------------ events
    def event(self, level: str, msg: str) -> None:
        self.seq = getattr(self, "seq", 0) + 1
        ev = {"seq": self.seq, "ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg}
        self.events.append(ev)
        log.log({"warn": logging.WARNING, "err": logging.ERROR}.get(level, logging.INFO), msg)
        if self.log_cb:
            self.log_cb(level, msg)

    # ------------------------------------------------------------ lifecycle
    def boot(self) -> None:
        """Connect, fingerprint and scan every database, then start the background watcher."""
        for n in self.order:
            self.connect(n)
        if not self.active:
            self.event("warn", "no database reachable yet")
        for n in self.order:
            if n in self.conns:
                self.fingerprint(n)
                self.run_scan(n)
        threading.Thread(target=self.watch, daemon=True, name="schema-watch").start()

    def connect(self, n: str, quiet: bool = False) -> None:
        """Try to connect backend `n`; on failure mark it unreachable (warn unless `quiet`)."""
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

    def fingerprint(self, n: str) -> None:
        b = self.conns[n]
        t0 = time.perf_counter()
        self.before[n] = {t: b.fingerprint(t) for t in b.tables()}
        self.event(
            "info",
            f"{self.dbs[n]['label']}: SHA-256 fingerprints of {len(self.before[n])} tables recorded "
            f"({time.perf_counter() - t0:.1f}s) as proof the data never changes",
        )

    def run_scan(self, n: str) -> None:
        """Scan backend `n` for PII with live progress, then install the new catalog and engine."""
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

        def progress(kind: str, table: str, done: int, total: int, tables_done: int, **kw: Any) -> None:
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

    def install(self, n: str, cat: dict[str, Any]) -> None:
        """Make catalog `cat` live for backend `n`, rebuilding its name index and masking engine."""
        b = self.conns[n]
        names = []
        for c in cat["columns"]:
            if c["entity"] == "person":
                q = b.sql("SELECT DISTINCT {c} FROM {t}", c=c["column"], t=c["table"])
                names += [r[0] for r in b.execute(q)[1] if r[0]]
        self.name_index[n] = detectors.NameIndex(names)
        self.catalogs[n] = cat
        self.engines[n] = engine.Engine(b, cat, self.policy, self.name_index[n])

    def watch(self) -> None:
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
    def ml_start(self) -> None:
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

    def ml_stop(self) -> None:
        ml.client.stop(shutdown_external=True)
        self.event("info", "AI model turned off: free text is masked with rules only")

    # ------------------------------------------------------------ API helpers
    def backend_name(self, q: dict[str, Any] | None) -> str:
        """The requested backend (default: the active one); KeyError if it is not ready."""
        n = (q or {}).get("backend") or self.active
        if n not in self.engines:
            raise KeyError(f"backend '{n}' is not connected or not scanned yet")
        return n

    def route(self, n: str, sql: str) -> str:
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

    def after_write(self, res: dict[str, Any]) -> dict[str, Any]:
        """Re-scan the playground after a write so new columns and values get masked."""
        if res.get("write") and "playground" in self.conns:
            self.event(
                "ok",
                f"Playground write by {res.get('user') or res['role']}: {max(res['affected'], 0)} row(s); re-scanning",
            )
            self.run_scan("playground")
        return res

    def role_of(self, user: str | None, role: str | None) -> str:
        """A known user's role wins over the requested one; unknown roles fall back to analyst."""
        for u in self.users:
            if user and u["username"] == user:
                return u["role"]
        return role if role in engine.ROLES else "analyst"

    def count(self, res: dict[str, Any]) -> dict[str, Any]:
        """Record a query result in the live stats and event log, and pass it through."""
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

    def status(self) -> dict[str, Any]:
        """Snapshot for /api/status and the TUIs (process stats re-sampled at most every 1.5 s)."""
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

    def tag_counts(self) -> dict[str, int]:
        cat = self.catalogs.get(self.active) or {}
        out = {}
        for c in cat.get("columns", []):
            if c["tag"]:
                out[c["tag"]] = out.get(c["tag"], 0) + 1
        return out

    def tables(self, n: str) -> dict[str, Any]:
        """Per-table view of backend `n`'s catalog with each column's masking strategy per role."""
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

    def integrity(self, n: str) -> dict[str, Any]:
        """Re-fingerprint every table of `n` and compare with the hashes taken at connect time."""
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
