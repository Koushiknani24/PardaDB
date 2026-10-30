"""Explorer tab (paged, masked table browsing) and Catalog tab (PII scan results) of the TUI."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on, work
from textual.widgets import Button, DataTable, Input, OptionList, Select, Static, Switch
from textual.widgets.option_list import Option

from . import engine
from .tui_base import PardaBase
from .tui_widgets import (
    PAGE,
    STRAT_STYLE,
    TAG_SHORT,
    TAG_STYLE,
    InfoModal,
    Result,
    cell_modal,
    fill_grid,
    one_line,
    outcome_text,
    strat,
)

CAT_DEBOUNCE_S = 0.25  # wait for a pause in typing before re-filtering the catalog


def catalog_detail(c: dict[str, Any], pol: dict[str, str]) -> InfoModal:
    """Modal with a catalog column's classification, its evidence and its masking per role."""
    head = Text.assemble(
        (f"{c['table']}.{c['column']}", "bold"),
        ("   tag ", "dim"),
        (c["tag"] or "clean", TAG_STYLE.get(c["tag"], "dim")),
        ("   entity ", "dim"),
        (c["entity"] or "-", "bold"),
        ("   confidence ", "dim"),
        (f"{c['confidence']:.2f}", "bold"),
        ("   type ", "dim"),
        (str(c.get("type") or "-"), "bold"),
    )
    body = Text()
    for e in c.get("evidence") or ["no evidence: column looks clean"]:
        body.append("• " + str(e) + "\n")
    legend = Text("Masking: ", "dim")
    for r in engine.ROLES:
        s = strat(pol.get(r, "FULL"))
        legend.append(f"{r} ", "bold")
        legend.append(f"{s}   ", STRAT_STYLE.get(s, "white"))
    return InfoModal(head, body, legend)


class ExplorerTab(PardaBase):
    """Explorer and Catalog tabs, plus the cell-detail modal shared with the Query tab."""

    # ------------------------------------------------------------------ explorer
    @work(thread=True, exclusive=True, group="tables")
    def load_tables(self, n: str) -> None:
        try:
            data = self.rt.tables(n)
        except Exception as e:
            data = {"error": one_line(f"{type(e).__name__}: {e}")}
        self.call_from_thread(self.show_tables, n, data)

    def show_tables(self, n: str, data: dict[str, Any]) -> None:
        """Fill the table list; keep the current table if it still exists, else browse the first one."""
        if n != self.rt.active:
            return
        ol = self.query_one("#ex-tables", OptionList)
        if "error" in data:
            self.query_one("#ex-msg", Static).update(Text(data["error"], "bold red"))
            return
        self.ex_meta = {t["table"]: t for t in data["tables"]}
        opts = []
        for t in data["tables"]:
            p = Text(t["table"], "bold")
            p.append(f"\n  {t['rows']:,} rows · ", "dim")
            p.append(f"{t['sensitive']} sensitive", "bold red" if t["sensitive"] else "green")
            opts.append(Option(p, id=t["table"]))
        ol.set_options(opts)
        if self.ex_table not in self.ex_meta:
            self.ex_table = next(iter(self.ex_meta), None)
            self.ex_offset, self.ex_filter = 0, (None, None)
        if self.ex_table:
            ol.highlighted = list(self.ex_meta).index(self.ex_table)
            self.set_filter_columns()
            self.browse()

    def set_filter_columns(self) -> None:
        cols = self.ex_meta.get(self.ex_table, {}).get("columns", [])
        sel = self.query_one("#ex-fcol", Select)
        sel.set_options(
            [(c["name"] + (f"  [{TAG_SHORT.get(c['tag'], c['tag'])}]" if c["tag"] else ""), c["name"]) for c in cols]
        )

    @on(OptionList.OptionSelected, "#ex-tables")
    def _table_clicked(self, ev: OptionList.OptionSelected) -> None:
        self.ex_table, self.ex_offset, self.ex_filter = ev.option.id, 0, (None, None)
        self.query_one("#ex-fval", Input).value = ""
        self.set_filter_columns()
        self.browse()

    def browse(self) -> None:
        """Load the current page of the current table, as the selected role, in a worker."""
        n = self.rt.active
        if not self.ex_table or n not in self.rt.engines:
            return
        self.query_one("#ex-msg", Static).update(Text(f"Loading {self.ex_table} as {self.role}…", "dim"))
        self.do_browse(n, self.ex_table, self.role, self.ex_offset, *self.ex_filter)

    @work(thread=True, exclusive=True, group="browse")
    def do_browse(self, n: str, table: str, role: str, offset: int, fcol: str | None, fval: str | None) -> None:
        try:
            res = self.rt.count(self.rt.engines[n].browse(table, role, self.user_for(role), offset, PAGE, fcol, fval))
        except Exception as e:
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        res.setdefault("role", role)
        self.call_from_thread(self.show_browse, res)

    def show_browse(self, res: Result) -> None:
        """Render a browse result (or its error) with the page counter and masking summary."""
        grid, msg = self.query_one("#ex-grid", DataTable), self.query_one("#ex-msg", Static)
        page, title = self.query_one("#ex-page", Static), self.query_one("#ex-title", Static)
        self.ex_res = res if res.get("ok") else None
        meta = self.ex_meta.get(self.ex_table, {})
        title.update(
            Text.assemble(
                (str(self.ex_table), "bold"),
                (" as ", "dim"),
                (self.role.upper(), "bold cyan"),
                (f"   {meta.get('sensitive', 0)} sensitive columns", "red"),
            )
        )
        if not res.get("ok"):
            grid.clear(columns=True)
            page.update("")
            msg.update(outcome_text(res))
            return
        self.colmap["ex-grid"] = fill_grid(
            grid, res, PAGE, self.query_one("#ex-sens", Switch).value, self.query_one("#ex-hl", Switch).value
        )
        total = res.get("total")
        lo = res["offset"] + 1 if res["row_count"] else 0
        hi = res["offset"] + res["row_count"]
        page.update(
            Text.assemble(
                ("rows ", "dim"),
                (f"{lo:,}-{hi:,}", "bold"),
                (" of ", "dim"),
                (f"{total:,}" if total is not None else "?", "bold"),
            )
        )
        fcol, fval = self.ex_filter
        msg.update(
            Text.assemble(
                (f"{res.get('masked_cells', 0)} cells masked", "magenta"),
                (f" · {res.get('elapsed_ms')} ms", "dim"),
                (f" · filter {fcol} = {fval}" if fcol else "", "cyan"),
                ("   click any cell to see its full value and PII spans", "dim"),
            )
        )

    @on(Button.Pressed, ".role")
    def _role_clicked(self, ev: Button.Pressed) -> None:
        self.action_role(ev.button.id.split("-", 1)[1])

    def action_role(self, role: str) -> None:
        """View as `role`: Explorer role buttons and re-browse, plus the Query tab's user picker."""
        self.role = role
        for b in self.query(".role").results(Button):
            b.variant = "primary" if b.id == f"role-{role}" else "default"
        user = next((u for u, r in self.q_user_role.items() if r == role), None)
        if user:
            self.query_one("#q-user", Select).value = user
        if self.active_tab() == "explorer":
            self.browse()
        self.notify(f"Viewing as {role.upper()}", timeout=2)

    def action_page(self, step: int) -> None:
        """Previous (-1) or next (+1) Explorer page."""
        if step < 0:
            self._prev()
        else:
            self._next()

    @on(Button.Pressed, "#ex-apply")
    def _apply_filter(self) -> None:
        col = self.query_one("#ex-fcol", Select).value
        val = self.query_one("#ex-fval", Input).value
        if col == Select.NULL:
            self.notify("Pick a column to filter on", severity="warning")
            return
        self.ex_filter, self.ex_offset = (col, val), 0
        self.browse()

    @on(Input.Submitted, "#ex-fval")
    def _filter_submitted(self) -> None:
        self._apply_filter()

    @on(Button.Pressed, "#ex-clear")
    def _clear_filter(self) -> None:
        self.query_one("#ex-fval", Input).value = ""
        self.query_one("#ex-fcol", Select).clear()
        self.ex_filter, self.ex_offset = (None, None), 0
        self.browse()

    @on(Button.Pressed, "#ex-prev")
    def _prev(self) -> None:
        if self.ex_offset > 0:
            self.ex_offset = max(0, self.ex_offset - PAGE)
            self.browse()

    @on(Button.Pressed, "#ex-next")
    def _next(self) -> None:
        total = (self.ex_res or {}).get("total")
        if self.ex_res and (total is None or self.ex_offset + PAGE < total):
            self.ex_offset += PAGE
            self.browse()

    @on(Switch.Changed, ".view")
    def _view_toggled(self, ev: Switch.Changed) -> None:
        if ev.switch.id.startswith("ex-"):
            if self.ex_res:
                self.show_browse(self.ex_res)
        elif any(self.q_res):
            self.show_query([r for r in self.q_res if r is not None])

    # ------------------------------------------------------------------ result cells -> modal
    @on(DataTable.CellSelected)
    def _cell_clicked(self, ev: DataTable.CellSelected) -> None:
        tid = ev.data_table.id or ""
        res = self.ex_res if tid == "ex-grid" else self.q_res[int(tid[-1])] if tid.startswith("q-grid-") else None
        if not res or not res.get("ok"):
            return
        r, c = ev.coordinate.row, ev.coordinate.column
        cm = self.colmap.get(tid) or []
        c = cm[c] if c < len(cm) else c
        if r < len(res["rows"]) and c < len(res["columns"]):
            self.push_screen(cell_modal(res, r, c))

    # ------------------------------------------------------------------ catalog
    def load_catalog(self) -> None:
        """Fill the Catalog grid for the active backend, honouring the text filter and 'show clean'."""
        n = self.rt.active
        cat = self.rt.catalogs.get(n)
        if not cat:
            return
        self.cat_backend = n
        q = self.query_one("#cat-filter", Input).value.strip().lower()
        show_clean = self.query_one("#cat-clean", Switch).value
        sig = (n, id(cat), q, show_clean)
        if sig == self._cat_sig:  # same catalog and filter: the grid is already right
            return
        self._cat_sig = sig
        grid = self.query_one("#cat-grid", DataTable)
        grid.clear()
        shown = 0
        for c in cat["columns"]:
            if not show_clean and c["status"] == "clean":
                continue
            ev = "; ".join(map(str, c.get("evidence") or []))
            if q and q not in " ".join(map(str, (c["table"], c["column"], c["tag"], c["entity"], ev))).lower():
                continue
            st = c["status"]
            grid.add_row(
                c["table"],
                Text(c["column"], "bold"),
                Text(c["tag"] or "clean", TAG_STYLE.get(c["tag"], "dim")),
                c["entity"] or "",
                Text(f"{c['confidence']:.2f}", justify="right"),
                Text(st, "green" if st == "auto" else "yellow" if st == "review" else "dim"),
                Text(ev[:90] + ("…" if len(ev) > 90 else ""), "dim"),
            )
            shown += 1
            if shown >= 400:
                break
        s = cat["summary"]
        self.query_one("#cat-summary", Static).update(
            Text.assemble(
                (f" {self.rt.dbs[n]['label']} ", "bold white on #1f4687"),
                "  ",
                (f"{s['tagged']} tagged", "bold green"),
                ("  ·  ", "dim"),
                (f"{s['review']} for review", "bold yellow"),
                ("  ·  ", "dim"),
                (f"{s['clean']} clean", "dim"),
                (f"   scanned in {cat.get('elapsed_s')} s   showing {shown}", "dim"),
                ("   click a row for full evidence", "dim"),
            )
        )

    @on(Input.Changed, "#cat-filter")
    def _cat_typed(self) -> None:
        if self._cat_timer:
            self._cat_timer.stop()
        self._cat_timer = self.set_timer(CAT_DEBOUNCE_S, self.load_catalog)

    @on(Switch.Changed, "#cat-clean")
    def _cat_filter(self) -> None:
        self.load_catalog()

    @on(DataTable.RowSelected, "#cat-grid")
    def _cat_row(self, ev: DataTable.RowSelected) -> None:
        row = ev.data_table.get_row(ev.row_key)
        cat = self.rt.catalogs.get(self.cat_backend) or {}
        c = next((x for x in cat.get("columns", []) if x["table"] == str(row[0]) and x["column"] == str(row[1])), None)
        if not c:
            return
        pol = self.rt.policy.get("tags", {}).get(c["tag"] or "", {})
        self.push_screen(catalog_detail(c, pol))

    @on(Button.Pressed, "#cat-rescan")
    def _rescan(self) -> None:
        n = self.rt.active
        if n not in self.rt.conns:
            self.notify("No connected backend to scan", severity="warning")
            return
        if self.rt.scan.get("state") == "scanning":
            self.notify("A scan is already running", severity="warning")
            return
        self.notify(f"Re-scanning {self.rt.dbs[n]['label']}… (progress on the Monitor tab)")
        self.do_scan(n)

    @work(thread=True, group="scan")
    def do_scan(self, n: str) -> None:
        try:
            self.rt.run_scan(n)
        except Exception as e:
            self.call_from_thread(self.notify, one_line(f"Scan failed: {e}"), severity="error")
