"""PardaDB ML sidecar: serves the fine-tuned GLiNER model (any compressed variant) over local HTTP.

    .venv/bin/python serve.py [--port 8766] [--model models/pardadb-pii-int8w] [--cpu] [--threads 4]

``--model`` takes the float32 checkpoint or any directory built by ``compress.py``; the default is the
chosen int8 variant (``models/pardadb-pii-int8w``) when it has been built, else ``models/pardadb-pii``.

POST /spans    {"texts": [...], "threshold": 0.5}  ->  {"spans": [[[start, end, label, score], ...], ...], "ms": 3.1}
GET  /health   ->  {"ok": true, "model": ..., "format": ..., "weights_mb": ..., "device": ..., "rss_mb": ..., ...}
POST /shutdown ->  {"ok": true}, then the server exits
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import compress
import torch

log = logging.getLogger("serve")

LABELS = ["person", "address", "organization"]
MAX_TEXTS = 256  # per request
BATCH = 16


def _params_m(model: Any) -> float | None:
    """Parameter count in millions, or None for a runtime that exposes no torch parameters (ONNX)."""
    params = getattr(model, "parameters", None)
    count = sum(p.numel() for p in params()) if callable(params) else 0
    return round(count / 1e6, 1) if count else None


class Model:
    """A loaded model variant plus the lock and counters the HTTP handlers share."""

    def __init__(self, path: Path, device: str, threads: int) -> None:
        self.path, self.device = path, device
        manifest = compress.read_manifest(path)
        self.format = manifest["format"]
        self.weights_mb = round(sum((path / w).stat().st_size for w in manifest["weights"]) / 1e6, 1)
        self.model = compress.load_model(path, device=device, threads=threads)
        self.params_m = _params_m(self.model)
        self.lock = threading.Lock()
        self.calls = 0

    def spans(self, texts: list[str], threshold: float) -> list[list[list[Any]]]:
        """Entity spans per text as ``[start, end, label, score]`` lists."""
        with self.lock, torch.inference_mode():
            res = self.model.inference(texts, LABELS, threshold=threshold, batch_size=BATCH)
            self.calls += len(texts)
        return [[[e["start"], e["end"], e["label"], round(float(e["score"]), 3)] for e in r] for r in res]

    def health(self) -> dict[str, Any]:
        """Status payload for ``GET /health``."""
        return {
            "ok": True,
            "model": str(self.path),
            "format": self.format,
            "weights_mb": self.weights_mb,
            "device": self.device,
            "labels": LABELS,
            "params_m": self.params_m,
            "rss_mb": compress.current_rss_mb(),
            "texts_served": self.calls,
        }


def handler(m: Model) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class bound to model ``m``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            """Silence the default per-request stderr log."""

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            """Serve ``/health``."""
            if self.path == "/health":
                self._send(200, m.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            """Serve ``/spans`` and ``/shutdown``."""
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/spans":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                texts = [str(t) for t in body.get("texts", [])][:MAX_TEXTS]
                threshold = float(body.get("threshold", 0.5))
            except (ValueError, TypeError, AttributeError) as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            t0 = time.perf_counter()
            out = m.spans(texts, threshold) if texts else []
            self._send(200, {"spans": out, "ms": round((time.perf_counter() - t0) * 1000, 2)})

    return Handler


def main() -> None:
    """CLI entry point: load the model, warm it up and serve until ``POST /shutdown``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--model", type=Path, default=compress.default_model_dir(), help="default: %(default)s")
    ap.add_argument("--cpu", action="store_true", help="run on CPU even when CUDA is available")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch / ONNX Runtime")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    m = Model(args.model, device, args.threads)
    m.spans(["warm up the model, Ritika Sharma called from Pune"], 0.5)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler(m))
    log.info(
        "PardaDB ML sidecar ready on 127.0.0.1:%d (%s, %s, %.1f MB weights, %s MB RAM)",
        args.port,
        m.format,
        device,
        m.weights_mb,
        compress.current_rss_mb(),
    )
    srv.serve_forever()
    srv.server_close()
    log.info("sidecar stopped")


if __name__ == "__main__":
    main()
