"""HTTP layer of the PardaDB runtime: request validation, the dashboard/JSON API handler and
the standard-library server that hosts it (no web framework)."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import ParseResult, parse_qs, urlparse

from . import audit, config, engine, ml

if TYPE_CHECKING:
    from .server import Runtime

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "dashboard"
PAGES = {"/": "landing.html", "/dashboard": "index.html", "/index.html": "index.html"}
ASSET_TYPES = {
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".css": "text/css",
    ".ico": "image/x-icon",
}
# Demo queries shown in the dashboard and the TUI; kept as data so they can change without code edits.
PRESETS: list[dict[str, str]] = json.loads(
    (Path(__file__).resolve().parent / "resources" / "presets.json").read_text(encoding="utf-8")
)

# ------------------------------------------------------------------ request validation
MAX_BODY_BYTES = 64 * 1024
MAX_SQL_CHARS = 20_000
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]", *config.EXTRA_HOSTS)


class BadRequest(ValueError):
    """The client sent invalid input; answered with HTTP `status` (400)."""

    status = 400


class BodyTooLarge(BadRequest):
    """The request body exceeds MAX_BODY_BYTES; answered with HTTP 413."""

    status = 413


def int_param(q: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    """Read `key` from `q` as a whole number within [lo, hi], or raise BadRequest."""
    try:
        value = int(q.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a whole number") from None
    if not lo <= value <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return value


def str_param(q: dict[str, Any], key: str, max_len: int = 200, required: bool = False) -> str | None:
    """Read `key` from `q` as bounded text; empty means None unless `required`."""
    value = q.get(key)
    if value is None or value == "":
        if required:
            raise BadRequest(f"'{key}' is required")
        return None
    if not isinstance(value, str) or len(value) > max_len:
        raise BadRequest(f"'{key}' must be text of at most {max_len} characters")
    return value


def body_length(header: str | None) -> int:
    """The Content-Length of a POST body: a whole number of at most MAX_BODY_BYTES."""
    try:
        n = int(header or 0)
    except ValueError:
        raise BadRequest("Content-Length must be a whole number") from None
    if n < 0:
        raise BadRequest("Content-Length must not be negative")
    if n > MAX_BODY_BYTES:
        raise BodyTooLarge(f"request body is larger than {MAX_BODY_BYTES // 1024} KB")
    return n


def sql_param(body: dict[str, Any]) -> str:
    """The required, length-limited `sql` field of a POST body, trimmed."""
    sql = str_param(body, "sql", MAX_SQL_CHARS, required=True)
    return sql.strip()


def make_handler(rt: Runtime) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to one Runtime (the stdlib server takes a class)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "PardaDB"

        def log_message(self, *a: Any) -> None:  # keep the terminal quiet
            pass

        def _send(self, code: int, body: Any, ctype: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "charset" not in ctype else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def _asset(self, path: str) -> None:
            """Serve a file from dashboard/assets/ (fixed folder, known types, no path traversal)."""
            root = (STATIC / "assets").resolve()
            target = (STATIC / path.lstrip("/")).resolve()
            ctype = ASSET_TYPES.get(target.suffix.lower())
            if ctype is None or root not in target.parents or not target.is_file():
                return self._send(404, {"error": "not found"})
            return self._send(200, target.read_bytes(), ctype)

        def _check_origin(self) -> None:
            """Only answer requests addressed to this machine (blocks DNS-rebinding pages)."""
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in ALLOWED_HOSTS:
                raise BadRequest(f"requests must be addressed to 127.0.0.1, not {host!r}")

        def _body(self) -> dict[str, Any]:
            # JSON only: a cross-site form post cannot set this type without a CORS preflight we never allow.
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                raise BadRequest("POST bodies must be sent as application/json")
            n = body_length(self.headers.get("Content-Length"))
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BadRequest("request body is not valid JSON") from None
            if not isinstance(body, dict):
                raise BadRequest("request body must be a JSON object")
            return body

        def _guarded(self, route: Callable[[], None]) -> None:
            """Run a route after the origin check, mapping errors to 400 / 413 / 409 / 500 responses."""
            try:
                self._check_origin()
                return route()
            except BadRequest as e:
                return self._send(e.status, {"ok": False, "error": str(e)})
            except KeyError as e:
                return self._send(409, {"ok": False, "error": str(e).strip("'\"")})
            except Exception:  # last-resort handler: log details, return a safe message
                log.exception("%s %s failed", self.command, self.path)
                return self._send(500, {"ok": False, "error": "internal error; details are in .state/pardadb.log"})

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            self._guarded(lambda: self._route_get(u, q))

        def do_POST(self) -> None:
            u = urlparse(self.path)
            self._guarded(lambda: self._route_post(u, self._body()))

        def _route_get(self, u: ParseResult, q: dict[str, str]) -> None:
            if u.path in PAGES:
                page = STATIC / PAGES[u.path]
                if not page.exists():  # no landing page built: go straight to the dashboard
                    page = STATIC / "index.html"
                return self._send(200, page.read_bytes(), "text/html")
            if u.path.startswith("/assets/"):
                return self._asset(u.path)
            if u.path == "/api/status":
                return self._send(200, rt.status())
            if u.path == "/api/users":
                return self._send(200, rt.users)
            if u.path == "/api/presets":
                return self._send(200, PRESETS)
            if u.path == "/api/policy":
                return self._send(200, rt.policy)
            if u.path == "/api/catalog":
                return self._send(200, self._catalog(q))
            if u.path == "/api/tables":
                return self._send(200, rt.tables(rt.backend_name(q)))
            if u.path == "/api/browse":
                return self._send(200, rt.count(self._browse(q)))
            if u.path == "/api/audit":
                return self._send(200, audit.read(int_param(q, "limit", 100, 1, 1000)))
            return self._send(404, {"error": "not found"})

        def _catalog(self, q: dict[str, str]) -> dict[str, Any]:
            n = rt.backend_name(q)
            cat = rt.catalogs[n]
            return {"backend": n, "summary": cat["summary"], "elapsed_s": cat["elapsed_s"], "columns": cat["columns"]}

        def _browse(self, q: dict[str, str]) -> dict[str, Any]:
            n = rt.backend_name(q)
            role = rt.role_of(q.get("user"), q.get("role"))
            return rt.engines[n].browse(
                str_param(q, "table", required=True),
                role,
                str_param(q, "user"),
                int_param(q, "offset", 0, 0, 10_000_000),
                int_param(q, "limit", 25, 1, 100),
                str_param(q, "filter_col"),
                str_param(q, "filter_val", 500),
            )

        def _route_post(self, u: ParseResult, body: dict[str, Any]) -> None:
            if u.path == "/api/query":
                return self._send(200, rt.after_write(rt.count(self._query(body))))
            if u.path == "/api/compare":
                sql = sql_param(body)
                n = rt.route(rt.backend_name(body), sql)
                return self._send(200, {r: rt.count(rt.engines[n].run(sql, r, f"compare:{r}")) for r in engine.ROLES})
            if u.path == "/api/ml":
                return self._send(200, self._ml(body))
            if u.path == "/api/integrity":
                return self._send(200, rt.integrity(rt.backend_name(body)))
            if u.path == "/api/scan":
                return self._scan(body)
            return self._send(404, {"error": "not found"})

        def _query(self, body: dict[str, Any]) -> dict[str, Any]:
            sql = sql_param(body)
            n = rt.route(rt.backend_name(body), sql)
            user = str_param(body, "user")
            return rt.engines[n].run(sql, rt.role_of(user, str_param(body, "role")), user)

        def _ml(self, body: dict[str, Any]) -> dict[str, Any]:
            action = str_param(body, "action", required=True)
            if action not in ("start", "stop"):
                raise BadRequest("'action' must be 'start' or 'stop'")
            if action == "stop":
                rt.ml_stop()
            else:  # loading the model takes seconds: answer now, the dashboard polls /api/status
                threading.Thread(target=rt.ml_start, daemon=True, name="ml").start()
            return {"ok": True, "ml": dict(ml.client.info)}

        def _scan(self, body: dict[str, Any]) -> None:
            n = rt.backend_name(body) if body.get("backend") in rt.conns or not body.get("backend") else None
            if n is None:
                return self._send(409, {"ok": False, "error": "backend not connected"})
            threading.Thread(target=rt.run_scan, args=(n,), daemon=True).start()
            return self._send(200, {"ok": True})

    return Handler


def serve(rt: Runtime, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create (but do not start) the threaded HTTP server for the dashboard and API."""
    httpd = ThreadingHTTPServer((host, port), make_handler(rt))
    httpd.daemon_threads = True
    return httpd
