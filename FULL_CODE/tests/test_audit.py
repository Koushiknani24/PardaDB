import json

from pardadb import audit


def _write(n=3):
    for i in range(n):
        audit.append({"user": f"u{i}", "role": "support", "status": "OK", "sql": f"SELECT {i}"})
    return audit._path()


def test_empty_log_is_valid():
    assert audit.read() == {"chain_ok": True, "total": 0, "records": []}


def test_chain_links_and_verifies():
    path = _write(3)
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert recs[0]["prev"] == "0" * 64
    assert [r["prev"] for r in recs[1:]] == [r["hash"] for r in recs[:-1]]
    log = audit.read(limit=2)
    assert log["chain_ok"] and log["total"] == 3
    assert [r["user"] for r in log["records"]] == ["u2", "u1"]  # newest first, limited


def test_tampering_with_a_record_breaks_the_chain():
    path = _write(3)
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["role"] = "admin"
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    assert audit.read()["chain_ok"] is False


def test_tampering_and_rehashing_one_record_still_breaks_the_chain():
    path = _write(3)
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["sql"] = "SELECT harmless"
    rec["hash"] = audit._digest(rec)  # attacker recomputes this record's hash...
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    assert audit.read()["chain_ok"] is False  # ...but the next record's prev no longer matches


def test_deleting_a_record_breaks_the_chain():
    path = _write(3)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert audit.read()["chain_ok"] is False


def test_chain_continues_after_restart(monkeypatch):
    _write(2)
    monkeypatch.setattr(audit, "_last", None)  # simulate a new process reading the tail from disk
    _write(1)
    log = audit.read()
    assert log["chain_ok"] and log["total"] == 3
