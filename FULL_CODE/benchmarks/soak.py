"""Long-run soak test: drive a running PardaDB over HTTP with parallel clients and record what happens.

    python benchmarks/soak.py --url http://127.0.0.1:8765 --minutes 10 --clients 8

Each client loops over the read and probe presets (SELECTs, bypass attempts, blocked writes) on every
connected bank database and all three roles, the same requests the dashboard sends. A sampler records the
server's RAM and CPU every few seconds. The result is written to benchmarks/soak.json: queries run,
errors, latency percentiles, throughput, memory over time and whether the audit hash chain still verifies.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("soak")

ROLES = ("admin", "support", "analyst")
BANK_BACKENDS = ("sqlite", "mysql", "postgres")
# Presets that only read: SELECTs and probes (the probes are expected to come back blocked, not to fail).
READ_CATEGORIES = ("SELECT (read and mask)", "Bypass and probe attempts", "Writes on bank data (blocked)")
OUT = Path(__file__).resolve().parent / "soak.json"


@dataclass
class Tally:
    """Counters shared by the client threads (guarded by `lock`)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    latencies_ms: list[float] = field(default_factory=list)
    ok: int = 0
    blocked: int = 0
    errors: int = 0
    error_samples: list[str] = field(default_factory=list)
    per_backend: dict[str, int] = field(default_factory=dict)

    def record(self, backend: str, ms: float, result: dict[str, Any] | None, error: str | None) -> None:
        with self.lock:
            self.latencies_ms.append(ms)
            self.per_backend[backend] = self.per_backend.get(backend, 0) + 1
            if error is not None:
                self.errors += 1
                if len(self.error_samples) < 5:
                    self.error_samples.append(error)
            elif result and result.get("blocked"):
                self.blocked += 1
            else:
                self.ok += 1


def call(url: str, path: str, body: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
    """GET (no body) or POST JSON to the PardaDB API and return the decoded reply."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - URL is the operator's own --url
        return json.load(r)


def client(url: str, jobs: list[tuple[str, str, str]], stop: threading.Event, tally: Tally, offset: int) -> None:
    """Run (backend, role, sql) jobs round-robin until `stop` is set."""
    ring = itertools.islice(itertools.cycle(jobs), offset, None)
    for backend, role, sql in ring:
        if stop.is_set():
            return
        started = time.perf_counter()
        result, error = None, None
        try:
            result = call(url, "/api/query", {"sql": sql, "role": role, "backend": backend})
            if not result.get("ok") and not result.get("blocked"):
                error = f"{backend}/{role}: {result.get('error')}"
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            error = f"{backend}/{role}: {type(e).__name__}: {e}"
        tally.record(backend, (time.perf_counter() - started) * 1000, result, error)


def sampler(url: str, stop: threading.Event, every_s: float, samples: list[dict[str, float]]) -> None:
    """Record the server's RSS and CPU every `every_s` seconds."""
    t0 = time.monotonic()
    while not stop.wait(every_s):
        try:
            proc = call(url, "/api/status", timeout=10)["process"]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
            log.warning("status sample failed: %s", e)
            continue
        samples.append({"t_s": round(time.monotonic() - t0, 1), "rss_mb": proc["rss_mb"], "cpu": proc["cpu_percent"]})


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(pct / 100 * len(ordered)))], 2)


def build_jobs(url: str, backends: list[str]) -> list[tuple[str, str, str]]:
    """Every read/probe preset on every backend for every role."""
    presets = [p["sql"] for p in call(url, "/api/presets") if p.get("category") in READ_CATEGORIES]
    return [(b, r, sql) for sql in presets for b in backends for r in ROLES]


def connected_backends(url: str) -> list[str]:
    status = call(url, "/api/status")
    names = [b.get("name", k) if isinstance(b, dict) else k for k, b in _backends(status).items()]
    live = [n for n in names if n in BANK_BACKENDS and _backends(status)[n].get("status") == "connected"]
    return live or ["sqlite"]


def _backends(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = status.get("backends", {})
    return {b["name"]: b for b in raw} if isinstance(raw, list) else raw


def run(url: str, minutes: float, clients: int, sample_every_s: float) -> dict[str, Any]:
    """Run the soak test and return the report."""
    backends = connected_backends(url)
    jobs = build_jobs(url, backends)
    audit_before = call(url, "/api/audit?limit=1")
    log.info("soak: %d clients, %.1f min, backends %s, %d distinct jobs", clients, minutes, backends, len(jobs))
    stop, tally, samples = threading.Event(), Tally(), []
    threads = [threading.Thread(target=sampler, args=(url, stop, sample_every_s, samples), daemon=True)]
    threads += [
        threading.Thread(target=client, args=(url, jobs, stop, tally, i * 7), daemon=True) for i in range(clients)
    ]
    started = time.monotonic()
    for t in threads:
        t.start()
    stop.wait(minutes * 60)
    stop.set()
    for t in threads:
        t.join(timeout=60)
    elapsed = time.monotonic() - started
    audit_after = call(url, "/api/audit?limit=1")
    rss = [s["rss_mb"] for s in samples]
    total = tally.ok + tally.blocked + tally.errors
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": url,
        "clients": clients,
        "minutes": round(elapsed / 60, 2),
        "backends": backends,
        "roles": list(ROLES),
        "distinct_queries": len(jobs),
        "queries": total,
        "ok": tally.ok,
        "blocked_as_expected": tally.blocked,
        "errors": tally.errors,
        "error_samples": tally.error_samples,
        "per_backend": tally.per_backend,
        "throughput_qps": round(total / elapsed, 1) if elapsed else 0.0,
        "latency_ms": {
            "p50": percentile(tally.latencies_ms, 50),
            "p95": percentile(tally.latencies_ms, 95),
            "p99": percentile(tally.latencies_ms, 99),
            "max": round(max(tally.latencies_ms, default=0.0), 2),
            "mean": round(statistics.fmean(tally.latencies_ms), 2) if tally.latencies_ms else 0.0,
        },
        "rss_mb": {"start": rss[0] if rss else None, "end": rss[-1] if rss else None, "max": max(rss, default=None)},
        "samples": samples,
        "audit": {
            "records_added": audit_after["total"] - audit_before["total"],
            "chain_ok_after": audit_after["chain_ok"],
            "segments_after": len(audit_after.get("segments", [])),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8765", help="a running PardaDB")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--sample-every", type=float, default=5.0, help="seconds between RAM/CPU samples")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = run(a.url.rstrip("/"), a.minutes, a.clients, a.sample_every)
    a.out.write_text(json.dumps(report, indent=2))
    log.info(
        "%d queries in %.1f min (%d errors), p95 %.1f ms, RSS %s -> %s MB, audit chain ok: %s; wrote %s",
        report["queries"],
        report["minutes"],
        report["errors"],
        report["latency_ms"]["p95"],
        report["rss_mb"]["start"],
        report["rss_mb"]["end"],
        report["audit"]["chain_ok_after"],
        a.out,
    )


if __name__ == "__main__":
    main()
