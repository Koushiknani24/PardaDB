"""PardaDB ML sidecar: serves the fine-tuned GLiNER model over local HTTP.

    .venv/bin/python serve.py [--port 8766] [--model models/pardadb-pii] [--cpu]

POST /spans  {"texts": [...], "threshold": 0.5}  ->  {"spans": [[[start, end, label, score], ...], ...], "ms": 3.1}
GET  /health ->  {"ok": true, "model": ..., "device": ..., "labels": [...], "params_m": 45.2, "rss_mb": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from gliner import GLiNER  # noqa: E402

LABELS = ["person", "address", "organization"]


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    return 0.0


class Model:
    def __init__(self, path: str, device: str):
        self.path, self.device = path, device
        self.model = GLiNER.from_pretrained(path).to(device).eval()
        self.params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)
        self.lock = threading.Lock()
        self.calls = 0

    def spans(self, texts: list[str], threshold: float) -> list:
        with self.lock, torch.no_grad():
            res = self.model.inference(texts, LABELS, threshold=threshold, batch_size=16)
        self.calls += len(texts)
        return [[[e["start"], e["end"], e["label"], round(float(e["score"]), 3)] for e in r] for r in res]


def handler(m: Model):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(
                    200,
                    {
                        "ok": True,
                        "model": m.path,
                        "device": m.device,
                        "labels": LABELS,
                        "params_m": m.params_m,
                        "rss_mb": rss_mb(),
                        "texts_served": m.calls,
                    },
                )
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/spans":
                return self._send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            texts = [str(t) for t in body.get("texts", [])][:256]
            t0 = time.perf_counter()
            out = m.spans(texts, float(body.get("threshold", 0.5))) if texts else []
            self._send(200, {"spans": out, "ms": round((time.perf_counter() - t0) * 1000, 2)})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--model", default=str(ROOT / "models" / "pardadb-pii"))
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    m = Model(a.model, device)
    m.spans(["warm up the model, Ritika Sharma called from Pune"], 0.5)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(m))
    print(
        f"PardaDB ML sidecar ready on 127.0.0.1:{a.port} ({device}, {m.params_m} M params, {rss_mb()} MB RAM)",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
