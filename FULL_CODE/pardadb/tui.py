"""Interactive terminal UI: full control of a running PardaDB from the terminal.

Type `help` for commands. Plain SQL is run as the current user/role.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import webbrowser
from typing import Any

with contextlib.suppress(ImportError):  # arrow keys and history for input(); absent on Windows
    import readline  # noqa: F401


from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import audit, engine, server

TAG = {
    "PII_HIGH": "bold red",
    "PII_MEDIUM": "yellow",
    "PII_QUASI": "green",
    "FINANCIAL_CONFIDENTIAL": "magenta",
    "PII_FREE_TEXT": "blue",
    "SENSITIVE_JSON": "bright_blue",
    "INTERNAL_ID": "white",
}
STRAT = {
    "FULL": "green",
    "PARTIAL": "yellow",
    "HASH": "magenta",
    "REDACT": "red",
    "BUCKET": "magenta",
    "GENERALIZE": "green",
    "SPANS": "blue",
    "JSON": "bright_blue",
}
LEVEL = {"ok": "green", "info": "cyan", "warn": "yellow", "block": "bold red"}
HELP = [
    ("status | s", "databases, scan progress, CPU / RAM, counters"),
    ("events [n]", "recent monitoring events"),
    ("use <sqlite|mysql|postgres>", "switch database"),
    ("role <admin|support|analyst>", "act as a role"),
    ("user <username> | users", "act as a user from users.csv / list users"),
    ("tables | t", "tables with row counts and sensitive columns"),
    ("browse <table> [col=value]", "page through a table as the current role (filter goes through the guard)"),
    ("next | prev", "page the last browse"),
    ("<SQL> | q <SQL>", "run a query as the current role"),
    ("compare <SQL> | c <SQL>", "same query as Admin, Support and Analyst"),
    ("presets | p <n>", "list / run a demo query"),
    ("open <row> <column>", "show one cell of the last result in full"),
    ("cols [fit|all|sensitive|a,b,c]", "which columns to show (default: as many as fit)"),
    ("wide", "toggle full-width cells"),
    ("catalog [table] [all]", "PII catalog with tags, confidence and evidence"),
    ("scan", "re-scan the current database"),
    ("policy", "tag x role policy"),
    ("audit [n]", "hash-chained audit log"),
    ("verify", "prove raw tables are unchanged (SHA-256)"),
    ("dash", "open the web dashboard"),
    ("clear | help | quit", ""),
]


class TUI:
    """Line-based REPL over a running Runtime: each `c_<word>` method is the command `<word>`."""

    def __init__(self, rt: server.Runtime, url: str) -> None:
        self.rt, self.url = rt, url
        self.con = Console(highlight=False)
        self.role: str = "admin"
        self.user: str | None = None
        self.wide = False
        self.colmode: str | list[str] = "fit"  # fit | all | sensitive | [names]
        self.last: dict[str, Any] | None = None  # last query/browse result
        self.browse_args: tuple[str, int, str | None, str | None] | None = None  # (table, offset, filter col, value)
        self.seen = 0

    # ------------------------------------------------------------ helpers
    @property
    def db(self) -> str | None:
        return self.rt.active

    def eng(self) -> engine.Engine | None:
        return self.rt.engines.get(self.db)

    def flush_events(self) -> None:
        """Print monitoring events that arrived since the last prompt."""
        evs = [e for e in self.rt.events if e["seq"] > self.seen]
        for e in evs:
            self.print_event(e)
        if evs:
            self.seen = evs[-1]["seq"]

    def prompt(self) -> str:
        """Readline prompt; \\x01/\\x02 wrap the colour codes so readline measures the width correctly."""
        who = f"{self.user} ({self.role})" if self.user else self.role
        st = self.rt.scan["state"]
        busy = " · scanning" if st == "scanning" else ""
        return f"\x01\x1b[1;34m\x02pardadb\x01\x1b[0m\x02 [{self.db or 'no db'}{busy}] {who} › "

    def cell(self, v: Any) -> str:
        """Escape a value for rich markup and truncate it unless wide mode is on."""
        if v is None:
            return "[dim]NULL[/]"
        s = str(v).replace("[", "\\[")
        return s if self.wide or len(s) <= 30 else s[:29] + "…"

    def show_result(self, res: dict[str, Any], title: str) -> None:
        """Render a query/browse result (or its block/error) and remember it for `open` and `cols`."""
        self.last = res
        if not res.get("ok"):
            if res.get("blocked"):
                self.con.print(
                    Panel(
                        f"[bold red]Blocked by inference guard[/]\n{res['reason']}",
                        title=title,
                        border_style="red",
                        expand=False,
                    )
                )
            else:
                self.con.print(f"[red]{title}: {res.get('error')}[/]")
            return
        cols = res["columns"]
        idx = self.pick_columns(cols, res["rows"])
        tb = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False, header_style="bold")
        tb.add_column("#", style="dim", justify="right", no_wrap=True)
        for i in idx:
            c = cols[i]
            short = {"FINANCIAL_CONFIDENTIAL": "FINANCIAL", "SENSITIVE_JSON": "JSON", "PII_FREE_TEXT": "FREE_TEXT"}
            tag = f"[{TAG.get(c['tag'], 'dim')}]{short.get(c['tag'], c['tag']) or 'clean'}[/]"
            tb.add_column(
                f"{c['name']}\n{tag} [{STRAT.get(c['strategy'], 'white')}]{c['strategy']}[/]",
                no_wrap=not self.wide,
                overflow="fold" if self.wide else "ellipsis",
                max_width=None if self.wide else 28,
            )
        marks = res.get("marks") or {}
        for i, r in enumerate(res["rows"][:50]):
            cells = []
            for j in idx:
                m = marks.get(f"{i}:{j}")
                c = self.cell(r[j])
                cells.append(f"[{'red' if m[0][3] else 'cyan'}]●{len(m)}[/] {c}" if m else c)
            tb.add_row(str(i), *cells)
        self.con.print(tb)
        if len(idx) < len(cols):
            self.con.print(
                f"[dim]showing {len(idx)} of {len(cols)} columns · `cols all` · `cols sensitive` · "
                f"`cols name1,name2` · `open <row> <column>`[/]"
            )
        extra = f" of {res['total']:,}" if res.get("total") is not None else ""
        self.con.print(
            f"[dim]{res['row_count']} rows{extra} · {res.get('masked_cells', 0)} cells masked · "
            f"{res.get('elapsed_ms')} ms · `open <row> <column>` shows a cell in full[/]"
        )

    def pick_columns(self, cols: list[dict[str, Any]], rows: list[list[Any]]) -> list[int]:
        """Columns to show: chosen list, sensitive only, all, or as many as fit the terminal."""
        n = len(cols)
        if isinstance(self.colmode, list):
            idx = [i for i, c in enumerate(cols) if c["name"].lower() in self.colmode]
            return idx or list(range(n))
        if self.colmode == "all":
            return list(range(n))
        if self.colmode == "sensitive":
            idx = [i for i, c in enumerate(cols) if c["tag"]]
            return idx or list(range(n))
        width, used, idx = self.con.width, 6, []
        for i, c in enumerate(cols):
            w = min(28, max([len(c["name"]), 14] + [len(str(r[i])) for r in rows[:15] if r[i] is not None])) + 3
            if idx and used + w > width:
                break
            idx.append(i)
            used += w
        return idx

    def c_cols(self, *args: str) -> None:
        arg = " ".join(args).strip().lower()
        self.colmode = (
            "fit"
            if arg in ("", "fit")
            else arg
            if arg in ("all", "sensitive")
            else [a.strip() for a in arg.split(",") if a.strip()]
        )
        self.con.print(f"columns: {self.colmode}")
        if self.last and self.last.get("ok"):
            self.show_result(self.last, "last result")

    def run_sql(self, sql: str) -> None:
        """Run SQL as the current role on the backend the runtime routes it to."""
        e = self.eng()
        if not e:
            return self.con.print("[yellow]No scanned database yet; wait for the scan to finish.[/]")
        n = self.rt.route(self.db, sql)
        res = self.rt.after_write(self.rt.count(self.rt.engines[n].run(sql, self.role, self.user)))
        self.show_result(res, f"{self.user or self.role} on {n}")

    # ------------------------------------------------------------ commands
    def c_status(self, *_: str) -> None:
        s = self.rt.status()
        p, sc, st = s["process"], s["scan"], s["stats"]
        tb = Table(box=box.SIMPLE, title="Databases")
        for c in ("database", "status", "tables", "rows", "latency", "detail"):
            tb.add_column(c)
        for b in s["backends"]:
            col = {"connected": "green", "unreachable": "red"}.get(b["status"], "yellow")
            mark = " ◀" if b["name"] == self.db else ""
            tb.add_row(
                b["label"] + mark,
                f"[{col}]{b['status']}[/]",
                str(b["tables"]),
                f"{b['rows']:,}",
                f"{b['latency_ms']} ms" if b["latency_ms"] is not None else "–",
                b["detail"][:60],
            )
        self.con.print(tb)
        lq = st.get("last_query") or {}
        self.con.print(
            f"scan [bold]{sc['state']}[/] {sc['columns_done']}/{sc['columns_total']} columns · "
            f"{sc['tagged']} sensitive · {sc['elapsed_s']}s   |   CPU {p['cpu_percent']}% · RAM {p['rss_mb']} MB · "
            f"threads {p['threads']} · disk {p['disk_mb']} MB   |   queries {st['queries']} · blocked "
            f"{st['blocked']} · masked cells {st['masked_cells']}"
            + (f" · last query {lq.get('ms')} ms ({lq.get('status')})" if lq else "")
        )

    def print_event(self, e: dict[str, Any]) -> None:
        """One event line; the message is escaped so brackets in it (SQL, lists) are printed, not parsed as markup."""
        self.con.print(f"[dim]{e['ts']}[/] [{LEVEL.get(e['level'], 'white')}]{escape(str(e['msg']))}[/]")

    def count_arg(self, n: str, usage: str) -> int | None:
        """`n` as a positive int, or None after printing the command's usage (like `p` does for bad input)."""
        try:
            v = int(n)
        except ValueError:
            v = 0
        if v > 0:
            return v
        self.con.print(f"[yellow]{escape(f'usage: {usage}  (n must be a positive whole number, got {n!r})')}[/]")
        return None

    def c_events(self, n: str = "20", *_: str) -> None:
        k = self.count_arg(n, "events [n]")
        if k is None:
            return
        for e in list(self.rt.events)[-k:]:
            self.print_event(e)

    def c_use(self, name: str | None = None, *_: str) -> None:
        if name not in self.rt.engines:
            ok = [n for n in self.rt.engines]
            return self.con.print(
                f"[yellow]'{name}' is not connected and scanned. Available: {', '.join(ok) or 'none'}[/]"
            )
        self.rt.active = name
        self.con.print(f"[green]using {name}[/]")

    def c_role(self, role: str | None = None, *_: str) -> None:
        if role not in engine.ROLES:
            return self.con.print(f"[yellow]roles: {', '.join(engine.ROLES)}[/]")
        self.role, self.user = role, None

    def c_user(self, name: str | None = None, *_: str) -> None:
        u = next((u for u in self.rt.users if u["username"] == name), None)
        if not u:
            return self.c_users()
        self.user, self.role = u["username"], u["role"]

    def c_users(self, *_: str) -> None:
        tb = Table(box=box.SIMPLE)
        for c in ("username", "name", "role"):
            tb.add_column(c)
        for u in self.rt.users:
            tb.add_row(u["username"], u["display_name"], u["role"])
        self.con.print(tb)

    def c_tables(self, *_: str) -> None:
        if not self.eng():
            return self.con.print("[yellow]not scanned yet[/]")
        tb = Table(box=box.SIMPLE, title=f"Tables on {self.db}")
        for c in ("table", "rows", "sensitive", "tags"):
            tb.add_column(c)
        for t in self.rt.tables(self.db)["tables"]:
            tb.add_row(
                t["table"],
                f"{t['rows']:,}",
                str(t["sensitive"]),
                " ".join(f"[{TAG.get(g, 'white')}]{g}[/]" for g in t["tags"]),
            )
        self.con.print(tb)

    def c_browse(self, table: str | None = None, *rest: str, offset: int = 0) -> None:
        """Show one 15-row page of a table; a `col=value` argument filters through the guard."""
        if not table:
            return self.con.print("[yellow]browse <table> [column=value][/]")
        fcol = fval = None
        for r in rest:
            if "=" in r:
                fcol, fval = r.split("=", 1)
        eng = self.eng()
        if not eng:
            return self.con.print("[yellow]not scanned yet[/]")
        self.browse_args = (table, offset, fcol, fval)
        res = eng.browse(table, self.role, self.user, offset, 15, fcol, fval)
        self.show_result(self.rt.count(res), f"{table} as {self.user or self.role} · rows {offset + 1}-{offset + 15}")

    def c_next(self, *_: str, step: int = 15) -> None:
        if not self.browse_args:
            return self.con.print("[yellow]browse a table first[/]")
        t, off, fc, fv = self.browse_args
        off = max(0, off + step)
        self.c_browse(t, *([f"{fc}={fv}"] if fc else []), offset=off)

    def c_prev(self, *_: str) -> None:
        self.c_next(step=-15)

    def c_compare(self, *sql: str) -> None:
        q = " ".join(sql)
        if not q or not self.eng():
            return self.con.print("[yellow]compare <SQL>[/]")
        for r in engine.ROLES:
            self.show_result(self.rt.count(self.eng().run(q, r, f"tui:{r}")), r.upper())

    def c_presets(self, *_: str) -> None:
        for i, p in enumerate(server.PRESETS):
            self.con.print(f"[bold]{i:>2}[/] {p['title']} [dim]— {p['expect']}[/]")

    def c_p(self, n: str | None = None, *_: str) -> None:
        """Run demo preset number n, or list the presets when n is missing or invalid."""
        try:
            p = server.PRESETS[int(n)]
        except (TypeError, ValueError, IndexError):
            return self.c_presets()
        self.con.print(f"[dim]{p['sql']}[/]")
        self.run_sql(p["sql"])

    def c_open(self, row: str | None = None, col: str | None = None, *_: str) -> None:
        """Show one cell of the last result in full, highlighting detected PII spans."""
        res = self.last
        if not res or not res.get("ok"):
            return self.con.print("[yellow]run a query or browse first[/]")
        names = [c["name"].lower() for c in res["columns"]]
        try:
            ci = int(col) if str(col).isdigit() else names.index(str(col).lower())
            v, meta = res["rows"][int(row)][ci], res["columns"][ci]
        except (TypeError, ValueError, IndexError):
            return self.con.print(f"[yellow]open <row> <column>; columns: {', '.join(names)}[/]")
        marks = (res.get("marks") or {}).get(f"{int(row)}:{ci}", [])
        title = f"row {row} · {meta['name']} · {meta['tag'] or 'clean'} · {meta['strategy']}"
        if marks:
            body = Text(str(v))
            for a, b, _ent, masked in marks:
                body.stylize("bold white on red" if masked else "bold black on cyan", a, b)
            kinds: dict[str, int] = {}
            for m in marks:
                kinds[m[2]] = kinds.get(m[2], 0) + 1
            how = "masked" if marks[0][3] else "detected (Admin sees them in full)"
            self.con.print(
                Panel(
                    body,
                    title=title,
                    border_style="cyan",
                    subtitle=f"{len(marks)} PII span(s) {how}: "
                    + ", ".join(f"{k}{' x' + str(n) if n > 1 else ''}" for k, n in kinds.items()),
                )
            )
            return
        with contextlib.suppress(TypeError, ValueError):
            v = json.dumps(json.loads(v), indent=2, ensure_ascii=False)
        self.con.print(Panel(str(v), title=title, border_style="cyan"))

    def c_wide(self, *_: str) -> None:
        self.wide = not self.wide
        self.con.print(f"wide cells {'on' if self.wide else 'off'}")

    def c_catalog(self, *args: str) -> None:
        cat = self.rt.catalogs.get(self.db)
        if not cat:
            return self.con.print("[yellow]not scanned yet[/]")
        show_all = "all" in args
        table = next((a for a in args if a != "all"), None)
        tb = Table(box=box.SIMPLE_HEAVY, title=f"PII catalog · {self.db} · {cat['summary']}")
        for c in ("table", "column", "tag", "entity", "conf", "status", "evidence"):
            tb.add_column(c, overflow="fold")
        for c in cat["columns"]:
            if (c["tag"] or show_all) and (not table or c["table"] == table):
                tb.add_row(
                    c["table"],
                    c["column"],
                    f"[{TAG.get(c['tag'], 'dim')}]{c['tag'] or 'clean'}[/]",
                    c["entity"] or "",
                    f"{c['confidence']:.2f}",
                    c["status"],
                    "; ".join(c["evidence"])[:80],
                )
        self.con.print(tb)

    def c_scan(self, *_: str) -> None:
        if self.db not in self.rt.conns:
            return self.con.print("[yellow]no database[/]")
        self.rt.run_scan(self.db)

    def c_policy(self, *_: str) -> None:
        tb = Table(box=box.SIMPLE, title="Policy: tag x role")
        tb.add_column("tag")
        for r in engine.ROLES:
            tb.add_column(r)
        for tag, roles in self.rt.policy["tags"].items():
            tb.add_row(f"[{TAG.get(tag, 'white')}]{tag}[/]", *[roles[r] for r in engine.ROLES])
        self.con.print(tb)

    def c_audit(self, n: str = "15", *_: str) -> None:
        k = self.count_arg(n, "audit [n]")
        if k is None:
            return
        a = audit.read(k)
        self.con.print(
            "[green]hash chain intact[/]" if a["chain_ok"] else "[bold red]hash chain BROKEN[/]",
            f"[dim]({a['total']} records)[/]",
        )
        tb = Table(box=box.SIMPLE)
        for c in ("time", "who", "db", "status", "sql", "hash"):
            tb.add_column(c, overflow="fold")
        for r in a["records"]:
            col = "green" if r["status"] == "OK" else "red"
            tb.add_row(
                r["ts"][11:],
                str(r.get("user") or r["role"]),
                r["backend"],
                f"[{col}]{r['status']}[/]",
                r["sql"][:70],
                r["hash"][:10],
            )
        self.con.print(tb)

    def c_verify(self, *_: str) -> None:
        res = self.rt.integrity(self.db)
        tb = Table(box=box.SIMPLE, title=f"Integrity · {self.db}")
        for c in ("table", "rows", "before", "after", "unchanged"):
            tb.add_column(c)
        for t in res["tables"]:
            tb.add_row(
                t["table"],
                f"{t['rows']:,}",
                t["before"][:16],
                t["after"][:16],
                "[green]yes[/]" if t["unchanged"] else "[red]NO[/]",
            )
        self.con.print(tb)

    def c_dash(self, *_: str) -> None:
        webbrowser.open(self.url)
        self.con.print(f"opened {self.url}")

    def c_help(self, *_: str) -> None:
        tb = Table(box=box.SIMPLE, title="PardaDB commands")
        tb.add_column("command", style="bold")
        tb.add_column("what it does")
        for c, d in HELP:
            tb.add_row(c, d)
        self.con.print(tb)

    # ------------------------------------------------------------ loop
    ALIASES = {"s": "status", "t": "tables", "q": "sql", "c": "compare", "?": "help", "h": "help"}

    def loop(self) -> None:
        """Read-eval loop until quit/EOF; plain SQL runs directly, other words dispatch to `c_<word>`."""
        self.con.print(
            f"[bold]Terminal control is ready.[/] Dashboard: {self.url} · type [bold]help[/] for commands, "
            f"or just type SQL."
        )
        while True:
            self.flush_events()
            try:
                line = input(self.prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                self.con.print()
                return
            if not line:
                continue
            word = line.split()[0].lower()
            word = self.ALIASES.get(word, word)
            if word in ("quit", "exit"):
                return
            if word == "clear":
                self.con.clear()
                continue
            if word == "sql":
                self.run_sql(line.split(None, 1)[1] if " " in line else "")
                continue
            if word in ("select", "with", "insert", "update", "delete", "drop", "create", "alter", "("):
                self.run_sql(line)
                continue
            fn = getattr(self, f"c_{word}", None)
            if not fn:
                self.con.print(f"[yellow]unknown command '{word}'. Type help.[/]")
                continue
            try:
                args = shlex.split(line)[1:] if word not in ("compare",) else line.split()[1:]
                fn(*args)
            except Exception as e:  # keep the TUI alive whatever happens
                self.con.print(f"[red]{type(e).__name__}: {e}[/]")
