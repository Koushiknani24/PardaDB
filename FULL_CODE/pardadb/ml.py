"""Client for the optional ML sidecar (fine-tuned GLiNER, see ../ml/serve.py).

The model finds what rules cannot (person names and addresses inside free text).
It is called once per query with every free-text cell batched, results are cached,
and if the sidecar is not running PardaDB silently falls back to rules only.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import threading
import time
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import config

URL = config.ML_URL
ML_DIR = config.ML_DIR
THRESHOLD = config.ML_THRESHOLD
LABELS = {"person": "person", "address": "address"}  # organisations (merchants, employers) are not personal data

# ------------------------------------------------------------------ precision filter
# The model is good at names and addresses but sometimes tags typos ("cusst", "earlierrr"),
# banking words ("Chargeback", "courier") or Hinglish words ("nahi aaya"). These checks keep
# only spans that look like a real name or address.
_DATA = Path(__file__).resolve().parent / "resources"
_FILTER = json.loads((_DATA / "span_filter.json").read_text())
NAME_WORDS = frozenset((_DATA / "name_words.txt").read_text().split())
STOP = frozenset(_FILTER["stop_words"])
ADDRESS_WORDS = frozenset(_FILTER["address_words"])
_TOKEN = re.compile(r"[A-Za-z][A-Za-z.'-]*")


def _name_token(tok: str) -> bool:
    core = tok.strip(".'-")
    if not core or re.search(r"(.)\1\1", core.lower()) or core.lower() in STOP:
        return False
    if len(core) == 1:
        return tok.endswith(".") and core.isupper()  # an initial like "S."
    return core[0].isupper() and (core[1:].islower() or core.isupper() or "'" in core or "-" in core)


def refine(text: str, s: int, e: int, label: str, score: float) -> tuple[int, int] | None:
    """Return (start, end) of a cleaned span, or None to drop it."""
    seg = text[s:e]
    if label == "address":
        low = seg.lower()
        ok = score >= 0.6 and (any(ch.isdigit() for ch in seg) or any(w in low.split() for w in ADDRESS_WORDS))
        return (s, e) if ok else None
    toks = [(m.start() + s, m.end() + s, m.group()) for m in _TOKEN.finditer(seg)]
    lowered = [t for t in toks if t[2].lower() not in STOP]
    if len(lowered) >= 2 and score >= 0.7 and all(t[2].lower() in NAME_WORDS for t in lowered):
        return lowered[0][0], lowered[-1][1]  # a full name typed in lower case, e.g. "rekha mittal"
    while toks and not _name_token(toks[0][2]):
        toks.pop(0)
    while toks and not _name_token(toks[-1][2]):
        toks.pop()
    if (
        not toks
        or any(ch.isdigit() for ch in text[toks[0][0] : toks[-1][1]])
        or not all(_name_token(t[2]) for t in toks)
    ):
        return None
    known = sum(t[2].strip(".'-").lower() in NAME_WORDS for t in toks)
    if len(toks) == 1:
        word = toks[0][2].strip(".'-")
        if word.isupper() and not known:
            return None  # a shouted word ("MISMATCH", "PAISA") is a name only if it is a known name
        if not (known or (score >= 0.75 and len(word) >= 4)):
            return None
    elif not (known or score >= 0.8):
        return None
    return toks[0][0], toks[-1][1]


def _call(path: str, payload: dict[str, Any] | None = None, timeout: float = config.ML_TIMEOUT_S) -> dict[str, Any]:
    """GET (no payload) or POST JSON to the sidecar and return its JSON reply."""
    data = None if payload is None else json.dumps(payload).encode()
    # URL comes from config.ML_URL, which is checked to be http(s) at start-up.
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.load(r)


class MLClient:
    """Starts, health-checks and queries the sidecar, with an LRU cache of span results."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cache: OrderedDict[str, list[tuple[int, int, str]]] = OrderedDict()
        self.info: dict[str, Any] = {
            "status": "off",
            "device": None,
            "params_m": None,
            "rss_mb": None,
            "texts": 0,
            "cache_hits": 0,
            "last_ms": None,
        }
        self.next_check = 0.0
        self.proc: subprocess.Popen[bytes] | None = None
        self.enabled = True  # the user can switch the model off from the TUI or dashboard

    # ------------------------------------------------------------ lifecycle
    def start_sidecar(self) -> bool:
        """Start ../ml/serve.py with the ML virtualenv, if it exists."""
        self.enabled = True
        py = ML_DIR / ".venv" / "bin" / "python"
        if not (py.exists() and (ML_DIR / "serve.py").exists()):
            self.info["status"] = "not installed"
            return False
        if self.health():
            return True
        # The command is fixed (our own interpreter and script); no user input reaches it.
        with open(config.state_path("ml_sidecar.log"), "a") as log:
            self.proc = subprocess.Popen(  # noqa: S603
                [str(py), "serve.py", "--port", URL.rsplit(":", 1)[-1]],
                cwd=ML_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.info["status"] = "starting"
        for _ in range(90):
            time.sleep(1)
            if self.health():
                return True
            if self.proc.poll() is not None:
                break
        self.info["status"] = "failed to start"
        return False

    def stop(self, shutdown_external: bool = False) -> None:
        """Stop the sidecar we started; with `shutdown_external`, also ask an external one to exit."""
        self.enabled = False
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc = None
        elif shutdown_external:
            with contextlib.suppress(OSError):
                _call("/shutdown", {}, timeout=2)
        self.info.update(status="off", device=None, rss_mb=None)

    def health(self) -> bool:
        """Ping the sidecar and refresh `info`; False when disabled or unreachable."""
        if not self.enabled:
            return False
        try:
            h = _call("/health", timeout=1.5)
            self.info.update(status="ready", device=h.get("device"), params_m=h.get("params_m"), rss_mb=h.get("rss_mb"))
            return True
        except OSError:
            if self.info["status"] == "ready":
                self.info["status"] = "unreachable"
            return False

    @property
    def ready(self) -> bool:
        """True when the sidecar is usable; re-checks a down sidecar at most every 30 s."""
        if not self.enabled:
            return False
        if self.info["status"] != "ready" and time.monotonic() >= self.next_check:
            self.next_check = time.monotonic() + 30
            self.health()
        return self.info["status"] == "ready"

    # ------------------------------------------------------------ inference
    def spans_for(self, texts: Iterable[str]) -> dict[str, list[tuple[int, int, str]]]:
        """{text: [(start, end, entity), ...]} for every text, using the cache and one batched call."""
        out: dict[str, list[tuple[int, int, str]]] = {}
        todo: list[str] = []
        with self.lock:
            for t in texts:
                if t in self.cache:
                    self.cache.move_to_end(t)
                    out[t] = self.cache[t]
                    self.info["cache_hits"] += 1
                elif t not in todo:
                    todo.append(t)
        if not todo or not self.ready:
            return out
        try:
            for i in range(0, len(todo), 128):
                chunk = todo[i : i + 128]
                res = _call("/spans", {"texts": chunk, "threshold": THRESHOLD})
                self.info["last_ms"] = res.get("ms")
                with self.lock:
                    for t, sp in zip(chunk, res["spans"], strict=False):
                        found = []
                        for s_, e_, lab, score in sp:
                            if lab in LABELS:
                                r = refine(t, s_, e_, lab, score)
                                if r:
                                    found.append((r[0], r[1], LABELS[lab]))
                        out[t] = self.cache[t] = found
                        self.info["texts"] += 1
                    while len(self.cache) > 20000:
                        self.cache.popitem(last=False)
        except OSError:
            self.info["status"] = "unreachable"
        return out


client = MLClient()
