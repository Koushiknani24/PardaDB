import io
import json
import urllib.error

import pytest

from pardadb import ml

TEXT = "Please suhmit Form 15G for Chargeback MISMATCH. Anthony D'Costa called. Bolo karo\nKomal ji. Flat 12, MG Road"


def span(sub, text=TEXT):
    s = text.index(sub)
    return s, s + len(sub)


@pytest.mark.parametrize("sub", ["suhmit Form 15G", "Chargeback", "MISMATCH", "cusssst"])
def test_refine_drops_non_names(sub):
    text = TEXT + " cusssst"
    assert ml.refine(text, *span(sub, text), "person", 0.99) is None


def test_refine_keeps_apostrophe_name():
    assert ml.refine(TEXT, *span("Anthony D'Costa"), "person", 0.9) == span("Anthony D'Costa")


def test_refine_trims_hinglish_word_and_newline():
    assert ml.refine(TEXT, *span("karo\nKomal"), "person", 0.9) == span("Komal")


def test_refine_single_unknown_word_needs_high_score(monkeypatch):
    monkeypatch.setattr(ml, "NAME_WORDS", frozenset({"komal"}))
    text = "met Zorawar today"
    assert ml.refine(text, *span("Zorawar", text), "person", 0.9) == span("Zorawar", text)
    assert ml.refine(text, *span("Zorawar", text), "person", 0.5) is None
    assert ml.refine("PAISA", 0, 5, "person", 0.99) is None  # shouted word, not a known name
    assert ml.refine("S. Rao", 0, 6, "person", 0.5) is None  # unknown two-word name, low score


def test_refine_lowercase_full_name_from_dictionary(monkeypatch):
    monkeypatch.setattr(ml, "NAME_WORDS", frozenset({"rekha", "mittal"}))
    text = "paid by rekha mittal"
    assert ml.refine(text, *span("rekha mittal", text), "person", 0.75) == span("rekha mittal", text)
    assert ml.refine(text, *span("rekha mittal", text), "person", 0.6) is None


def test_refine_address():
    assert ml.refine(TEXT, *span("Flat 12, MG Road"), "address", 0.9) == span("Flat 12, MG Road")
    assert ml.refine("flat 12 near temple", 0, 19, "address", 0.5) is None  # low score
    assert ml.refine("somewhere vague", 0, 15, "address", 0.9) is None  # no digit, no address word


class FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


def ready_client():
    c = ml.MLClient()
    c.info["status"] = "ready"
    return c


def test_spans_for_batches_refines_and_caches(monkeypatch):
    texts = ["Spoke with Ravi Kumar", "Chargeback raised"]
    calls = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        calls.append(body)
        return FakeResponse(
            {
                "ms": 7,
                "spans": [
                    [[11, 21, "person", 0.95], [0, 5, "organisation", 0.9]],
                    [[0, 10, "person", 0.95]],  # "Chargeback": dropped by refine
                ],
            }
        )

    monkeypatch.setattr(ml.urllib.request, "urlopen", fake_urlopen)
    c = ready_client()
    out = c.spans_for(texts)
    assert out == {"Spoke with Ravi Kumar": [(11, 21, "person")], "Chargeback raised": []}
    assert len(calls) == 1 and calls[0]["texts"] == texts
    assert c.info["last_ms"] == 7
    assert c.spans_for(texts) == out  # served from cache
    assert len(calls) == 1 and c.info["cache_hits"] == 2


def test_spans_for_sidecar_unreachable_falls_back_quietly(monkeypatch):
    def refuse(*a, **k):
        raise urllib.error.URLError(ConnectionRefusedError(111, "refused"))

    monkeypatch.setattr(ml.urllib.request, "urlopen", refuse)
    c = ml.MLClient()  # never seen the sidecar: health check fails
    assert c.spans_for(["Spoke with Ravi Kumar"]) == {}
    assert c.ready is False

    c = ready_client()  # was up, then died mid-query
    assert c.spans_for(["Spoke with Ravi Kumar"]) == {}
    assert c.info["status"] == "unreachable"


def test_spans_for_real_closed_port_returns_empty():
    # PARDADB_ML_URL points at port 9 (see conftest), so this exercises the real socket path
    c = ml.MLClient()
    assert c.spans_for(["Komal Sharma called"]) == {}


def test_disabled_client_never_calls_out(monkeypatch):
    monkeypatch.setattr(ml.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    c = ready_client()
    c.enabled = False
    assert c.ready is False and c.health() is False
    assert c.spans_for(["x"]) == {}
