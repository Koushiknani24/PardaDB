import fcntl
import json
import multiprocessing
from pathlib import Path

import pytest

from pardadb import audit, config


def _write(n: int = 3, tag: str = "u") -> Path:
    for i in range(n):
        audit.append({"user": f"{tag}{i}", "role": "support", "status": "OK", "sql": f"SELECT {i}"})
    return audit._path()


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _edit(path: Path, index: int, **changes: str) -> None:
    lines = _lines(path)
    lines[index] = json.dumps({**json.loads(lines[index]), **changes})
    path.write_text("\n".join(lines) + "\n")


def _all_files() -> list[Path]:
    active = audit._path()
    return [*audit._segments(), *([active] if active.exists() else [])]


def _append_worker(state_dir: str, max_bytes: int, tag: str, n: int) -> None:
    """Runs in a separate process: point at the test's state dir and append n records."""
    config.STATE_DIR = Path(state_dir)
    config.AUDIT_MAX_BYTES = max_bytes
    _write(n, tag)


@pytest.fixture
def small_segments(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(config, "AUDIT_MAX_BYTES", 1000)
    return 1000


def test_empty_log_is_valid() -> None:
    assert audit.read() == {"chain_ok": True, "total": 0, "records": [], "segments": []}


def test_chain_links_and_verifies() -> None:
    path = _write(3)
    recs = [json.loads(line) for line in _lines(path)]
    assert recs[0]["prev"] == audit.GENESIS
    assert [r["prev"] for r in recs[1:]] == [r["hash"] for r in recs[:-1]]
    log = audit.read(limit=2)
    assert log["chain_ok"] and log["total"] == 3
    assert [r["user"] for r in log["records"]] == ["u2", "u1"]  # newest first, limited
    assert log["segments"] == [{"file": "audit.jsonl", "count": 3, "chain_ok": True, "broken_line": None}]


def test_chain_verifies_after_many_appends() -> None:
    _write(500)
    log = audit.read(limit=1)
    assert log["chain_ok"] and log["total"] == 500
    assert log["records"][0]["user"] == "u499"


def test_tampering_with_a_record_breaks_the_chain() -> None:
    path = _write(3)
    _edit(path, 1, role="admin")
    log = audit.read()
    assert log["chain_ok"] is False
    assert log["segments"][0]["broken_line"] == 2


def test_tampering_with_prev_breaks_the_chain() -> None:
    path = _write(3)
    _edit(path, 2, prev="f" * 64)
    assert audit.read()["chain_ok"] is False


def test_tampering_and_rehashing_one_record_still_breaks_the_chain() -> None:
    path = _write(3)
    lines = _lines(path)
    rec = json.loads(lines[0])
    rec["sql"] = "SELECT harmless"
    rec["hash"] = audit._digest(rec)  # attacker recomputes this record's hash...
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    log = audit.read()
    assert log["chain_ok"] is False  # ...but the next record's prev no longer matches
    assert log["segments"][0]["broken_line"] == 2


def test_deleting_a_record_breaks_the_chain() -> None:
    path = _write(3)
    lines = _lines(path)
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert audit.read()["chain_ok"] is False


def test_chain_continues_after_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    _write(2)
    monkeypatch.setattr(audit, "_last", None)  # a new process has no cached tail
    _write(1)
    log = audit.read()
    assert log["chain_ok"] and log["total"] == 3


def test_stale_cached_tail_is_not_trusted() -> None:
    _write(1)
    stale = audit._last
    _write(1, tag="other")  # stands in for another process appending after our last write
    audit._last = stale
    _write(1, tag="late")
    log = audit.read()
    assert log["chain_ok"] and log["total"] == 3
    assert log["records"][0]["prev"] == log["records"][1]["hash"]


def test_rotation_happens_at_the_threshold(small_segments: int) -> None:
    _write(30)
    rotated = audit._segments()
    assert len(rotated) >= 2
    for seg in rotated:
        assert audit._SEGMENT.fullmatch(seg.name)
        size, last_line = seg.stat().st_size, _lines(seg)[-1]
        assert size > small_segments >= size - len(last_line.encode()) - 1  # rotated by the append that crossed it
    assert all(f.stat().st_size <= small_segments for f in _all_files()[len(rotated) :])  # active file, if any


def test_chain_verifies_across_segments(small_segments: int) -> None:
    _write(30)
    log = audit.read(limit=5)
    assert log["chain_ok"] and log["total"] == 30
    assert sum(s["count"] for s in log["segments"]) == 30
    files = _all_files()
    assert [s["file"] for s in log["segments"]] == [f.name for f in files]
    assert [r["user"] for r in log["records"]] == ["u29", "u28", "u27", "u26", "u25"]
    for older, newer in zip(files, files[1:], strict=False):
        assert json.loads(_lines(newer)[0])["prev"] == json.loads(_lines(older)[-1])["hash"]


def test_chain_continues_into_a_new_file_after_rotation(small_segments: int, monkeypatch: pytest.MonkeyPatch) -> None:
    while not audit._segments():
        _write(1)
    assert not audit._path().exists()
    monkeypatch.setattr(audit, "_last", None)
    _write(1, tag="fresh")
    first = json.loads(_lines(audit._path())[0])
    assert first["prev"] == json.loads(_lines(audit._segments()[-1])[-1])["hash"]
    assert audit.read()["chain_ok"]


def test_tampering_in_an_older_segment_is_detected(small_segments: int) -> None:
    _write(30)
    _edit(audit._segments()[0], 1, role="admin")
    log = audit.read()
    assert log["chain_ok"] is False
    assert [s["chain_ok"] for s in log["segments"]] == [False] + [True] * (len(log["segments"]) - 1)
    assert log["total"] == 30


def test_deleting_a_rotated_segment_breaks_the_chain(small_segments: int) -> None:
    _write(30)
    audit._segments()[1].unlink()
    log = audit.read()
    assert log["chain_ok"] is False
    assert log["segments"][1]["broken_line"] == 1


def test_torn_last_line_is_reported_and_kept_visible() -> None:
    path = _write(3)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])  # crash mid-write of the third record
    log = audit.read()
    assert log["chain_ok"] is False and log["total"] == 2
    assert log["segments"][0]["broken_line"] == 3
    audit._last = None
    rec = audit.append({"user": "after", "role": "support", "status": "OK", "sql": "SELECT 1"})
    assert rec["prev"] == json.loads(_lines(path)[1])["hash"]  # links to the last intact record
    assert len(_lines(path)) == 4  # the torn line stays on its own line
    log = audit.read()
    assert log["chain_ok"] is False and log["total"] == 3
    assert log["records"][0]["user"] == "after"


def test_append_times_out_when_the_lock_is_held(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "AUDIT_LOCK_TIMEOUT_S", 0.05)
    with open(config.state_path("audit.lock"), "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError):
            _write(1)
    assert audit.read()["total"] == 0


@pytest.mark.parametrize("max_bytes", [50_000_000, 3000], ids=["one-file", "rotating"])
def test_concurrent_processes_keep_one_valid_chain(
    isolated_state: Path, monkeypatch: pytest.MonkeyPatch, max_bytes: int
) -> None:
    monkeypatch.setattr(config, "AUDIT_MAX_BYTES", max_bytes)
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_worker, args=(str(isolated_state), max_bytes, f"w{i}-", 50)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert [p.exitcode for p in procs] == [0, 0, 0, 0]
    log = audit.read(limit=1000)
    assert log["chain_ok"] and log["total"] == 200
    assert len({r["hash"] for r in log["records"]}) == 200
    assert (len(log["segments"]) > 1) == (max_bytes == 3000)
