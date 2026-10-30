"""Hash-chained audit log: each record stores the SHA-256 of the previous one,
so editing or deleting any past record breaks the chain.

Several processes (the web server, the CLI) may append at once, so every append holds an
exclusive ``flock`` on a sibling ``audit.lock`` file and links to the tail hash it reads
from disk under that lock. When the active ``audit.jsonl`` grows past
``config.AUDIT_MAX_BYTES`` it is renamed to ``audit-<UTC time>-<n>.jsonl`` and the chain
continues in a fresh file: its first record's ``prev`` is the rotated file's last hash.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from . import config

log = logging.getLogger(__name__)

GENESIS = "0" * 64  # `prev` of the very first record
_SEGMENT = re.compile(r"audit-\d{8}T\d{6}Z-(\d+)\.jsonl")
_TAIL_CHUNK = 4096
_LOCK_POLL_S = 0.005

# Fast path: (path, inode, size, hash) of the active file right after this process's last
# append. It is only trusted while the active file still has that inode and size, which
# every append by any process (size grows) and every rotation (new inode) changes.
_last: tuple[str, int, int, str] | None = None


def _path() -> Path:
    return config.state_path("audit.jsonl")


def _digest(rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _segment_index(path: Path) -> int:
    m = _SEGMENT.fullmatch(path.name)
    return int(m.group(1)) if m else -1


def _segments() -> list[Path]:
    """Rotated segments, oldest first."""
    rotated = [p for p in _path().parent.glob("audit-*.jsonl") if _segment_index(p) >= 0]
    return sorted(rotated, key=_segment_index)


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    """Hold an exclusive flock on audit.lock, giving up after config.AUDIT_LOCK_TIMEOUT_S."""
    with open(config.state_path("audit.lock"), "a") as fd:  # closing the file releases the lock
        deadline = time.monotonic() + config.AUDIT_LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"audit log locked for over {config.AUDIT_LOCK_TIMEOUT_S}s") from None
                time.sleep(_LOCK_POLL_S)
        yield


def _parse(line: bytes) -> dict[str, Any] | None:
    """The record on one log line, or None for a torn or corrupt line."""
    try:
        rec = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return rec if isinstance(rec, dict) else None


def _tail_hash(f: IO[bytes]) -> str | None:
    """Hash of the last intact record, reading the file backwards a chunk at a time."""
    pos, buf = f.seek(0, os.SEEK_END), b""
    while pos > 0 and buf.count(b"\n") < 3:  # two complete lines even if the last one is torn
        step = min(_TAIL_CHUNK, pos)
        pos -= step
        f.seek(pos)
        buf = f.read(step) + buf
    lines = buf.split(b"\n")[1 if pos > 0 else 0 :]  # the first piece may be a partial line
    for line in reversed(lines):
        rec = _parse(line) if line.strip() else None
        if rec and isinstance(rec.get("hash"), str):
            return rec["hash"]
    return None


def _chain_tail(active: IO[bytes], st: os.stat_result) -> str:
    """Hash the next record must link to: the cached one, the active file's tail, or the newest segment's."""
    if _last is not None and _last[:3] == (str(_path()), st.st_ino, st.st_size):
        return _last[3]
    if st.st_size and (h := _tail_hash(active)):
        return h
    for seg in reversed(_segments()):
        with open(seg, "rb") as f:
            if h := _tail_hash(f):
                return h
    return GENESIS


def _ends_with_newline(f: IO[bytes], size: int) -> bool:
    f.seek(size - 1)
    return f.read(1) == b"\n"


def _rotate(path: Path) -> None:
    n = max((_segment_index(p) for p in _segments()), default=0) + 1
    target = path.with_name(f"audit-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{n:06d}.jsonl")
    os.replace(path, target)
    log.info("audit log rotated to %s", target.name)


def append(rec: dict[str, Any]) -> dict[str, Any]:
    """Timestamp `rec`, link it to the previous record's hash and append it to the log."""
    global _last
    path = _path()
    with _exclusive_lock(), open(path, "a+b") as f:
        st = os.fstat(f.fileno())
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec, "prev": _chain_tail(f, st)}
        rec["hash"] = _digest(rec)
        line = json.dumps(rec, ensure_ascii=False).encode() + b"\n"
        if st.st_size and not _ends_with_newline(f, st.st_size):
            line = b"\n" + line  # keep a torn line (crash mid-write) on its own line, where verify flags it
        f.write(line)
        f.flush()
        size = os.fstat(f.fileno()).st_size
        _last = (str(path), st.st_ino, size, rec["hash"])
        if size > config.AUDIT_MAX_BYTES:
            _rotate(path)
    return rec


def _verify_segment(f: IO[bytes], size: int, prev: str, newest: deque[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Check one segment's first `size` bytes against the chain, collecting records into `newest`."""
    count, first_break, consumed = 0, None, 0
    for lineno, line in enumerate(f, 1):
        if consumed >= size:
            break
        consumed += len(line)
        rec = _parse(line)
        if rec is not None:
            count += 1
            newest.append(rec)
        if rec is None or rec.get("prev") != prev or _digest(rec) != rec.get("hash"):
            first_break = first_break or lineno
        if rec is not None:
            prev = str(rec.get("hash"))
    report = {"count": count, "chain_ok": first_break is None, "broken_line": first_break}
    return report, prev


@contextmanager
def _snapshot() -> Iterator[tuple[list[Path], IO[bytes] | None, int]]:
    """Rotated segments (oldest first) plus the open active file and its size, taken under the lock.

    Rotated segments never change again; the active file is read only up to its size at this
    instant, so concurrent appends and rotations never show a half-written view.
    """
    with _exclusive_lock():
        rotated, path = _segments(), _path()
        active = open(path, "rb") if path.exists() else None  # noqa: SIM115 - closed below
        size = os.fstat(active.fileno()).st_size if active else 0
    try:
        yield rotated, active, size
    finally:
        if active:
            active.close()


def _segment_files(rotated: list[Path], active: IO[bytes] | None, size: int) -> Iterator[tuple[str, IO[bytes], int]]:
    """(name, open file, bytes to read) for every segment, oldest first."""
    for seg in rotated:
        with open(seg, "rb") as f:
            yield seg.name, f, os.fstat(f.fileno()).st_size
    if active:
        yield _path().name, active, size


def read(limit: int = 100) -> dict[str, Any]:
    """Newest `limit` records plus whether the hash chain verifies across every segment."""
    newest: deque[dict[str, Any]] = deque(maxlen=limit)
    prev, segments = GENESIS, []
    with _snapshot() as (rotated, active, active_size):
        for name, f, size in _segment_files(rotated, active, active_size):
            report, prev = _verify_segment(f, size, prev, newest)
            segments.append({"file": name, **report})
    chain_ok = all(s["chain_ok"] for s in segments)
    if not chain_ok:
        log.warning("audit hash chain broken: %s", [s for s in segments if not s["chain_ok"]])
    return {
        "chain_ok": chain_ok,
        "total": sum(s["count"] for s in segments),
        "records": list(reversed(newest)),
        "segments": segments,
    }
