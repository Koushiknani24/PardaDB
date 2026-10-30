"""PardaDB command line.

pardadb                      start everything: connect, scan, dashboard at http://127.0.0.1:8765,
                             then an interactive terminal UI (type `help`)
pardadb start [--db sqlite,mysql,postgres] [--port 8765] [--no-browser]
pardadb scan [--db sqlite]                    scan once and print the PII catalog
pardadb query --user support_ravi --sql "..."  run one query as a user (or --role)
pardadb compare --sql "..."                   same query as Admin, Support and Analyst
pardadb verify [--db sqlite]                  print SHA-256 fingerprints of every table
pardadb status [--port 8765]                  status of a running PardaDB
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.request
import webbrowser

from rich import box
from rich.console import Console
from rich.table import Table

from . import __version__, backends, config, engine, scanner, server

con = Console(highlight=False)
LEVEL = {"ok": "green", "info": "cyan", "warn": "yellow", "block": "bold red"}
TAG = {
    "PII_HIGH": "bold red",
    "PII_MEDIUM": "yellow",
    "PII_QUASI": "green",
    "FINANCIAL_CONFIDENTIAL": "magenta",
    "PII_FREE_TEXT": "blue",
    "SENSITIVE_JSON": "bright_blue",
    "INTERNAL_ID": "white",
}


def banner():
    con.print(
        f"[bold white on #1f4687] PardaDB [/] [dim]v{__version__} · automatic PII discovery and "
        f"role-based masking · raw data is never changed[/]"
    )


def log(level, msg):
    con.print(f"[dim]{time.strftime('%H:%M:%S')}[/] [{LEVEL.get(level, 'white')}]{msg}[/]")


def cmd_start(a):
    banner()
    use_tui = sys.stdin.isatty() and not a.no_tui
    rt = server.Runtime([d.strip() for d in a.db.split(",") if d.strip()], log=None if use_tui else log)
    try:
        httpd = server.serve(rt, a.host, a.port)
    except OSError as e:
        con.print(
            f"[red]Cannot listen on {a.host}:{a.port}: {e}. Is PardaDB already running? "
            f"Try `pardadb status` or --port.[/]"
        )
        return 1
    url = f"http://{a.host}:{a.port}"
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    con.print(f"[bold green]Dashboard ready:[/] [link={url}]{url}[/link]  [dim](Ctrl+C to stop)[/]")
    if not a.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    threading.Thread(target=rt.boot, daemon=True, name="boot").start()
    from . import ml

    if a.ml:
        threading.Thread(target=rt.ml_start, daemon=True, name="ml").start()
    elif ml.client.health():
        rt.event("ok", f"AI model sidecar found on {ml.client.info['device']}: free text uses rules + model")
    if use_tui:
        app = None
        if not a.classic:
            try:
                from .app import PardaApp

                app = PardaApp(rt, url)
            except ImportError:  # textual not installed: fall back to the line-based UI
                con.print(
                    "[yellow]textual is not installed; using the classic terminal UI "
                    "(pip install textual for the full-screen one)[/]"
                )
        if app is not None:
            time.sleep(0.5)  # let the first connections report before the screen takes over
            app.run()
        else:
            from .tui import TUI

            time.sleep(1.5)  # let the first connections report before the prompt appears
            TUI(rt, url).loop()
        con.print("[dim]stopping PardaDB[/]")
        httpd.shutdown()
        ml.client.stop()
        return 0
    try:
        while True:
            time.sleep(a.heartbeat)
            p = rt.status()["process"]
            s = rt.stats
            log(
                "info",
                f"monitoring · CPU {p['cpu_percent']}% · RAM {p['rss_mb']} MB · threads {p['threads']} · "
                f"queries {s['queries']} · blocked {s['blocked']}",
            )
    except KeyboardInterrupt:
        con.print("\n[dim]stopping PardaDB[/]")
        httpd.shutdown()
        ml.client.stop()
    return 0


def _engine(dbname):
    b = backends.connect(dbname)
    cat = scanner.load(dbname)
    if cat is None:
        cat = scanner.scan(b)
        scanner.save(cat)
    return engine.Engine(b, cat)


def cmd_scan(a):
    banner()
    b = backends.connect(a.db)
    cat = scanner.scan(b, lambda kind, table, done, total, **kw: kind == "table" and log("info", f"scanned {table}"))
    scanner.save(cat)
    tb = Table(title=f"PII catalog: {b.label} ({cat['elapsed_s']}s)", box=box.SIMPLE_HEAVY)
    for c in ("table", "column", "tag", "entity", "conf", "evidence"):
        tb.add_column(c, overflow="fold")
    for c in cat["columns"]:
        if c["tag"] or a.all:
            tb.add_row(
                c["table"],
                c["column"],
                f"[{TAG.get(c['tag'], 'dim')}]{c['tag'] or 'clean'}[/]",
                c["entity"] or "",
                f"{c['confidence']:.2f}",
                "; ".join(c["evidence"])[:90],
            )
    con.print(tb)
    s = cat["summary"]
    con.print(f"[green]{s['tagged']} tagged[/], [yellow]{s['review']} for review[/], [dim]{s['clean']} clean[/]")
    return 0


def _print_result(res, title):
    if not res.get("ok"):
        con.print(
            f"[bold red]{title}: {'BLOCKED' if res.get('blocked') else 'ERROR'}[/] "
            f"{res.get('reason') or res.get('error')}"
        )
        return
    tb = Table(title=title, box=box.SIMPLE_HEAVY)
    for c in res["columns"]:
        tb.add_column(f"{c['name']}\n[dim]{c['strategy']}[/]", overflow="fold", max_width=36)
    for r in res["rows"][:20]:
        tb.add_row(*["NULL" if v is None else str(v) for v in r])
    con.print(tb)


def cmd_query(a):
    users = {u["username"]: u["role"] for u in server.load_users()}
    role = users.get(a.user, a.role or "analyst")
    _print_result(_engine(a.db).run(a.sql, role, a.user), f"{a.user or role} on {a.db}")
    return 0


def cmd_compare(a):
    e = _engine(a.db)
    for role in engine.ROLES:
        _print_result(e.run(a.sql, role, f"cli:{role}"), role.upper())
    return 0


def cmd_verify(a):
    b = backends.connect(a.db)
    tb = Table(title=f"SHA-256 fingerprints: {b.label}", box=box.SIMPLE_HEAVY)
    tb.add_column("table")
    tb.add_column("rows", justify="right")
    tb.add_column("sha256")
    for t in b.tables():
        h, n = b.fingerprint(t)
        tb.add_row(t, f"{n:,}", h[:32] + "…")
    con.print(tb)
    return 0


def cmd_status(a):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/status", timeout=3) as r:
            s = json.load(r)
    except OSError:
        con.print("[yellow]PardaDB is not running. Start it with `pardadb`.[/]")
        return 1
    p = s["process"]
    con.print(f"PardaDB v{s['version']} · up {s['uptime_s']}s · CPU {p['cpu_percent']}% · RAM {p['rss_mb']} MB")
    for b in s["backends"]:
        con.print(f"  {b['label']:<14} {b['status']:<12} {b['tables']} tables {b['rows']:,} rows")
    sc = s["scan"]
    con.print(f"  scan: {sc['state']} {sc['columns_done']}/{sc['columns_total']} columns, {sc['tagged']} sensitive")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pardadb", description="Automatic PII discovery and role-based dynamic masking.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("start", help="connect, scan, serve the dashboard and keep monitoring")
    p.add_argument("--db", default="sqlite,mysql,postgres,playground")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--heartbeat", type=int, default=30, help="seconds between terminal status lines")
    p.add_argument("--no-tui", action="store_true", help="plain monitoring log instead of the interactive terminal UI")
    p.add_argument("--ml", action="store_true", help="also start the AI model sidecar (needs ml/.venv; ~1.4 GB RAM)")
    p.add_argument("--classic", action="store_true", help="line-based terminal UI instead of the full-screen one")
    p = sub.add_parser("scan", help="scan once and print the catalog")
    p.add_argument("--db", default="sqlite")
    p.add_argument("--all", action="store_true", help="also list clean columns")
    for name in ("query", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--sql", required=True)
        p.add_argument("--db", default="sqlite")
        if name == "query":
            p.add_argument("--user")
            p.add_argument("--role", choices=engine.ROLES)
    p = sub.add_parser("verify")
    p.add_argument("--db", default="sqlite")
    p = sub.add_parser("status")
    p.add_argument("--port", type=int, default=8765)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["start", *argv]  # plain `pardadb` starts everything
    a = ap.parse_args(argv)
    # Details go to a log file: the terminal belongs to the TUI and the console output.
    logging.basicConfig(
        filename=config.state_path("pardadb.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return {
        "start": cmd_start,
        "scan": cmd_scan,
        "query": cmd_query,
        "compare": cmd_compare,
        "verify": cmd_verify,
        "status": cmd_status,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
