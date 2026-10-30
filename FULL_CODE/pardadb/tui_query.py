"""Query tab (presets, run, compare roles), Audit tab (hash chain) and Integrity tab (SHA-256 check) of the TUI."""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual import on, work
from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, OptionList, Select, Static, Switch, TabbedContent

from . import audit, engine, server
from .tui_base import PardaBase
from .tui_widgets import AUDIT_STYLE, QUERY_ROWS, Result, SqlArea, fill_grid, one_line, outcome_text


def result_summary(res: Result) -> Text:
    """Body of a successful result's heading: WRITE badge with affected rows, or row and mask counts."""
    if res.get("write"):
        return Text.assemble(
            (" WRITE ", "bold black on yellow"),
            (f" {max(res.get('affected', 0), 0)} row(s) affected", "yellow"),
        )
    more = f" (showing {QUERY_ROWS})" if res["row_count"] > QUERY_ROWS else ""
    return Text.assemble(
        (f"{res['row_count']} rows{more}", "bold green"),
        (f" · {res.get('masked_cells', 0)} cells masked", "magenta"),
    )


def audit_row(r: dict[str, Any]) -> list[Any]:
    """Cells of one audit-log grid row: time, user, role, backend, status, SQL, detail and hash prefix."""
    st = r.get("status", "")
    sql = " ".join(str(r.get("sql", "")).split())
    det = r.get("reason") or (
        ", ".join(f"{k}:{v}" for k, v in (r.get("masked") or {}).items())
        if r.get("masked")
        else f"{r.get('rows', '')} rows"
        if "rows" in r
        else ""
    )
    return [
        str(r.get("ts", ""))[11:] or r.get("ts", ""),
        r.get("user") or "",
        r.get("role") or "",
        r.get("backend") or "",
        Text(st, AUDIT_STYLE.get(st, "white")),
        Text(sql[:70] + ("…" if len(sql) > 70 else "")),
        Text(str(det)[:60], "red" if st == "BLOCKED" else "dim"),
        Text(str(r.get("hash", ""))[:12], "dim"),
    ]


class QueryTab(PardaBase):
    """Query, Audit and Integrity tabs."""

    # ------------------------------------------------------------------ query
    @on(OptionList.OptionSelected, "#q-presets")
    def _preset_clicked(self, ev: OptionList.OptionSelected) -> None:
        p = server.PRESETS[int(ev.option.id)]
        now = time.monotonic()
        # OptionList has no double-click event, so two selections of one preset within 0.6 s count as one.
        double = self._last_preset == (ev.option.id,) and now - self._last_preset_t < 0.6
        self._last_preset, self._last_preset_t = (ev.option.id,), now
        self.query_one("#q-sql", SqlArea).text = p["sql"]
        self.query_one("#q-msg", Static).update(
            Text.assemble(
                (f"{p.get('category', '')}: ", "dim"),
                (p["title"], "bold"),
                ("   expect: ", "dim"),
                (p["expect"], "italic"),
            )
        )
        if double:
            self.action_run_query()

    def _query_args(self) -> tuple[str, str, str, str] | None:
        """(backend, sql, role, user) for the editor's query, or None after warning what is missing.

        A query that will run is added to the editor's history (up/down recall it).
        """
        editor = self.query_one("#q-sql", SqlArea)
        sql = editor.text.strip()
        n = self.rt.active
        if not sql:
            self.notify("Type a SQL query first", severity="warning")
            return None
        if n not in self.rt.engines:
            self.notify("No database scanned yet", severity="warning")
            return None
        user = self.query_one("#q-user", Select).value
        role = self.rt.role_of(user, None)
        editor.remember(sql)
        return n, sql, role, user

    @on(Button.Pressed, "#q-run")
    def action_run_query(self) -> None:
        """Run the editor's query as the selected user; from another tab, only switch to Query."""
        tabs = self.query_one("#tabs", TabbedContent)
        if tabs.active != "query":
            self.action_tab("query")
            return
        a = self._query_args()
        if a:
            self.query_one("#q-msg", Static).update(Text(f"Running as {a[3]} ({a[2]})…", "dim"))
            self.do_query(*a)

    @work(thread=True, exclusive=True, group="query")
    def do_query(self, n: str, sql: str, role: str, user: str) -> None:
        try:
            n2 = self.rt.route(n, sql)
            res = self.rt.count(self.rt.engines[n2].run(sql, role, user))
        except Exception as e:
            res = {"ok": False, "error": one_line(f"{type(e).__name__}: {e}"), "role": role, "user": user}
        self.call_from_thread(self.show_query, [res])
        if res.get("write"):
            try:
                self.rt.after_write(res)
            except Exception as e:
                self.call_from_thread(self.notify, one_line(f"Re-scan failed: {e}"), severity="error")

    @on(Button.Pressed, "#q-compare")
    def _compare(self) -> None:
        a = self._query_args()
        if a:
            self.query_one("#q-msg", Static).update(Text("Comparing Admin, Support and Analyst…", "dim"))
            self.do_compare(a[0], a[1])

    @work(thread=True, exclusive=True, group="query")
    def do_compare(self, n: str, sql: str) -> None:
        out = []
        try:
            n2 = self.rt.route(n, sql)
            for r in engine.ROLES:
                res = self.rt.count(self.rt.engines[n2].run(sql, r, f"compare:{r}"))
                out.append(res)
        except Exception as e:
            out.append({"ok": False, "error": one_line(f"{type(e).__name__}: {e}"), "role": "?"})
        self.call_from_thread(self.show_query, out)

    def show_query(self, results: list[Result]) -> None:
        """Show up to three results (one per role when comparing), each under its own heading."""
        self.q_res = (results + [None, None, None])[:3]
        summary = [self._show_result(i) for i in range(3)]
        self.query_one("#q-msg", Static).update(
            Text(" · ".join(s for s in summary if s) + "   click any cell for its full value", "dim")
        )
        self.query_one("#q-results", VerticalScroll).scroll_home(animate=False)

    def _show_result(self, i: int) -> str | None:
        """Render result slot i; returns its outcome (OK, BLOCKED or ERROR), or None for an empty slot."""
        title, grid = self.query_one(f"#q-title-{i}", Static), self.query_one(f"#q-grid-{i}", DataTable)
        res = self.q_res[i]
        title.display = res is not None
        grid.display = bool(res and res.get("ok"))
        if res is None:
            return None
        who = f"{res.get('user') or ''}".replace("compare:", "")
        head = Text.assemble(
            (f" {str(res.get('role') or '?').upper()} ", "bold white on #1f4687"),
            " ",
            (who, "bold"),
            (f" on {res.get('backend', '')}", "dim"),
            "  ",
        )
        if res.get("ok"):
            self.colmap[f"q-grid-{i}"] = fill_grid(
                grid,
                res,
                QUERY_ROWS,
                self.query_one("#q-sens", Switch).value,
                self.query_one("#q-hl", Switch).value,
            )
            head.append_text(result_summary(res))
            outcome = "OK"
        else:
            head.append_text(outcome_text(res))
            outcome = "BLOCKED" if res.get("blocked") else "ERROR"
        head.append(f"  {res.get('elapsed_ms', '-')} ms", "dim")
        title.update(head)
        return outcome

    # ------------------------------------------------------------------ audit
    @work(thread=True, exclusive=True, group="audit")
    def load_audit(self) -> None:
        try:
            data = audit.read(50)
        except Exception as e:
            data = {"chain_ok": False, "total": 0, "records": [], "error": str(e)}
        self.call_from_thread(self.show_audit, data)

    def show_audit(self, data: dict[str, Any]) -> None:
        """Show whether the audit hash chain is intact and the newest records."""
        ok = data.get("chain_ok")
        self.query_one("#au-chain", Static).update(
            Text.assemble(
                (" CHAIN INTACT " if ok else " CHAIN BROKEN ", "bold white on green" if ok else "bold white on red"),
                (f"  {data.get('total', 0):,} records, each holding the SHA-256 of the previous one", "dim"),
                (f"   {one_line(data['error'])}" if data.get("error") else "", "red"),
                ("   (showing newest 50)" if data.get("records") else "   no queries yet: run one on tab 4", "dim"),
            )
        )
        g = self.query_one("#au-grid", DataTable)
        g.clear()
        for r in data.get("records", []):
            g.add_row(*audit_row(r))

    @on(Button.Pressed, "#au-refresh")
    def _audit_refresh(self) -> None:
        self.load_audit()

    # ------------------------------------------------------------------ integrity
    @on(Button.Pressed, "#in-verify")
    def _verify(self) -> None:
        n = self.rt.active
        if n not in self.rt.before:
            self.notify("No connected backend to verify", severity="warning")
            return
        self.query_one("#in-note", Static).update(Text(f"Hashing every table of {self.rt.dbs[n]['label']}…", "yellow"))
        self.do_verify(n)

    @work(thread=True, exclusive=True, group="integrity")
    def do_verify(self, n: str) -> None:
        try:
            res = self.rt.integrity(n)
        except Exception as e:
            res = {"all_unchanged": False, "tables": [], "error": one_line(f"{type(e).__name__}: {e}")}
        self.call_from_thread(self.show_verify, n, res)

    def show_verify(self, n: str, res: dict[str, Any]) -> None:
        """Green or red banner plus one row per table comparing start-up and current fingerprints."""
        b = self.query_one("#in-banner", Static)
        ok = res.get("all_unchanged") and not res.get("error")
        b.set_classes("ok" if ok else "bad")
        label = self.rt.dbs[n]["label"]
        b.update(
            Text(
                f"✔  {label}: ALL {len(res['tables'])} TABLES UNCHANGED  (SHA-256 identical to start-up)"
                if ok
                else f"✘  {label}: {res.get('error') or 'CHANGES DETECTED'}",
                "bold white",
            )
        )
        g = self.query_one("#in-grid", DataTable)
        g.clear()
        for t in res["tables"]:
            g.add_row(
                Text(t["table"], "bold"),
                Text(f"{t['rows']:,}", justify="right"),
                Text(t["before"][:24] + "…", "dim"),
                Text(t["after"][:24] + "…", "dim"),
                Text("✔ unchanged", "bold green") if t["unchanged"] else Text("✘ CHANGED", "bold red"),
            )
        self.query_one("#in-note", Static).update(
            Text("Verified just now. Masking never writes to the source data.", "dim")
        )
