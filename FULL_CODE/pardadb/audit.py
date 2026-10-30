"""Hash-chained audit log: each record stores the SHA-256 of the previous one,
so editing or deleting any past record breaks the chain."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from . import config

_lock = threading.Lock()
_last: str | None = None  # hash of the newest record, loaded lazily from disk


def _path() -> Path:
    return config.state_path("audit.jsonl")


def _digest(rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _tail_hash() -> str:
    global _last
    if _last is None:
        _last = "0" * 64
        p = _path()
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            if lines:
                _last = json.loads(lines[-1])["hash"]
    return _last


def append(rec: dict[str, Any]) -> dict[str, Any]:
    """Timestamp `rec`, link it to the previous record's hash and append it to the log."""
    global _last
    with _lock:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec, "prev": _tail_hash()}
        rec["hash"] = _digest(rec)
        with open(_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _last = rec["hash"]
        return rec


def read(limit: int = 100) -> dict[str, Any]:
    """Newest `limit` records plus whether the whole hash chain still verifies."""
    p = _path()
    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    prev, ok = "0" * 64, True
    for r in recs:
        if r.get("prev") != prev or _digest(r) != r.get("hash"):
            ok = False
            break
        prev = r["hash"]
    return {"chain_ok": ok, "total": len(recs), "records": list(reversed(recs[-limit:]))}
