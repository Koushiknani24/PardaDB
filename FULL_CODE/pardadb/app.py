"""Full-screen, mouse-clickable terminal UI for PardaDB (Textual).

Runs in the same process as the dashboard and talks to the Runtime directly (no HTTP).
Everything slow (browse, query, integrity, scan) runs in worker threads so the UI never freezes.
"""

from __future__ import annotations

import time
import webbrowser

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    OptionList,
    ProgressBar,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    Tabs,
    TextArea,
)
from textual.widgets.option_list import Option

from . import audit, engine, masks, server

TAG_STYLE = {
    "PII_HIGH": "bold red",
    "PII_MEDIUM": "yellow",
    "PII_QUASI": "green",
    "FINANCIAL_CONFIDENTIAL": "magenta",
    "PII_FREE_TEXT": "blue",
    "SENSITIVE_JSON": "bright_blue",
    "INTERNAL_ID": "white",
}
TAG_SHORT = {
    "FINANCIAL_CONFIDENTIAL": "FINANCIAL",
    "PII_FREE_TEXT": "FREE_TEXT",
    "SENSITIVE_JSON": "JSON",
    "INTERNAL_ID": "ID",
    "PII_MEDIUM": "PII_MED",
}
STRAT_STYLE = {
    "FULL": "green",
    "PARTIAL": "yellow",
    "HASH": "magenta",
    "BUCKET": "magenta",
    "REDACT": "red",
    "GENERALIZE": "green",
    "SPANS": "blue",
    "JSON": "bright_blue",
}
LEVEL_STYLE = {"ok": "green", "info": "cyan", "warn": "yellow", "block": "bold red"}
STATUS_STYLE = {"connected": "bold green", "unreachable": "bold red", "connecting": "yellow"}
AUDIT_STYLE = {"OK": "green", "BLOCKED": "bold red", "WRITE": "yellow"}
MARK_MASKED, MARK_SEEN = "bold white on red", "bold black on cyan"
MARK_AI = "bold white on #7c3aed"  # spans found by the AI model (SLM), not by rules


def mark_style(ent, flag):
    return MARK_AI if "(AI)" in str(ent) else MARK_MASKED if flag else MARK_SEEN


PAGE, QUERY_ROWS, CELL_MAX = 25, 50, 40
_FLAT = {ord("\n"): " ", ord("\r"): " ", ord("\t"): " "}


def strat(s):
    return masks.DISPLAY.get(s, s) if s else "FULL"


def dur(s):
    s = int(s or 0)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def col_label(c):
    t = Text(str(c["name"]), style="bold")
    t.append("\n")
    if c.get("tag"):
        t.append(TAG_SHORT.get(c["tag"], c["tag"]), TAG_STYLE.get(c["tag"], "white"))
        t.append(" ")
    s = strat(c.get("strategy"))
    t.append(s, STRAT_STYLE.get(s, "white"))
    return t


def cell_text(v, strategy, marks=None, limit=CELL_MAX, highlight=True):
    if v is None:
        return Text("NULL", style="dim italic")
    s = str(v)
    plain = not highlight or strategy in (None, "FULL")
    t = Text(s.translate(_FLAT), style="" if plain else STRAT_STYLE.get(strategy, ""))
    for a, b, ent, flag in (marks or []) if highlight else []:
        t.stylize(mark_style(ent, flag), a, b)
    if limit and len(s) > limit:
        t.truncate(limit - 1)
        t.append("…", "dim")
    return t


def fill_grid(table: DataTable, res, max_rows, sens_only=False, highlight=True):
    """Render a result; returns the result-column index of each displayed column."""
    table.clear(columns=True)
    cols = res.get("columns") or []
    show = (
        [i for i, c in enumerate(cols) if c.get("tag") or c.get("strategy") not in (None, "FULL")]
        if sens_only and not res.get("write")
        else list(range(len(cols)))
    )
    for i in show:
        table.add_column(col_label(cols[i]))
    marks = res.get("marks") or {}
    table.add_rows(
        [
            [cell_text(row[ci], cols[ci]["strategy"], marks.get(f"{ri}:{ci}"), highlight=highlight) for ci in show]
            for ri, row in enumerate((res.get("rows") or [])[:max_rows])
        ]
    )
    return show


class ClickTable(DataTable):
    """DataTable where a single mouse click on a cell/row selects it (stock DataTable needs two)."""

    def _on_click(self, event):
        # Textual also runs DataTable._on_click (it dispatches along the MRO); that one only posts a
        # selection when the clicked cell already had the cursor, so post it ourselves otherwise.
        meta = event.style.meta
        if (
            meta.get("row", -1) >= 0
            and meta.get("column", -1) >= 0
            and not meta.get("out_of_bounds")
            and self.show_cursor
            and self.cursor_type != "none"
            and Coordinate(meta["row"], meta["column"]) != self.cursor_coordinate
        ):
            self.call_later(self._post_selected_message)


def preset_options():
    """Presets grouped by operation type: a disabled heading row per category."""
    opts, last = [], object()
    for i, p in enumerate(server.PRESETS):
        cat = p.get("category") or "Presets"
        if cat != last:
            if opts:
                opts.append(None)
            style = (
                "bold yellow"
                if cat.startswith("Playground")
                else "bold red"
                if "blocked" in cat or "Bypass" in cat
                else "bold cyan"
            )
            opts.append(Option(Text(cat.upper(), style), disabled=True))
            last = cat
        opts.append(Option(Text("  " + p["title"]), id=str(i)))
    return opts


# ---------------------------------------------------------------------------------------------- modals
class InfoModal(ModalScreen):
    """Shows a full value (or any detail) with a legend; Esc or Close dismisses it."""

    BINDINGS = [Binding("escape", "close", "Close", priority=True), Binding("q", "close", "Close", show=False)]
    DEFAULT_CSS = """
    InfoModal { align: center middle; background: $background 60%; }
    #m-box { width: 100; max-width: 95%; height: auto; max-height: 90%; border: thick $accent;
             background: $panel; padding: 1 2; }
    #m-scroll { height: auto; max-height: 24; margin: 1 0; border: round $primary-darken-1; padding: 0 1; }
    #m-legend { margin-bottom: 1; }
    #m-close { width: 100%; }
    """

    def __init__(self, head: Text, body: Text, legend: Text | None = None):
        super().__init__()
        self.head, self.body, self.legend = head, body, legend

    def compose(self) -> ComposeResult:
        with Vertical(id="m-box"):
            yield Static(self.head, id="m-head")
            with VerticalScroll(id="m-scroll"):
                yield Static(self.body, id="m-body")
            yield Static(self.legend or Text(""), id="m-legend")
            yield Button("Close  (Esc)", id="m-close", variant="primary")

    @on(Button.Pressed, "#m-close")
    def action_close(self):
        self.dismiss()


def cell_modal(res, row, col):
    c = res["columns"][col]
    v = res["rows"][row][col]
    marks = (res.get("marks") or {}).get(f"{row}:{col}") or []
    s = strat(c.get("strategy"))
    head = Text.assemble(
        ("Column ", "dim"),
        (str(c["name"]), "bold"),
        "   ",
        ("tag ", "dim"),
        (c.get("tag") or "none", TAG_STYLE.get(c.get("tag"), "dim")),
        ("  entity ", "dim"),
        (c.get("entity") or "-", "bold"),
        "\n",
        ("Role ", "dim"),
        ((res.get("role") or "?").upper(), "bold cyan"),
        ("  strategy ", "dim"),
        (s, STRAT_STYLE.get(s, "white")),
        ("  row ", "dim"),
        (str(row + 1), "bold"),
    )
    if c.get("source"):
        src = ", ".join(".".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x) for x in c["source"])
        head.append(f"\nsource: {src}", "dim")
    body = Text("NULL", style="dim italic")
    if v is not None:
        body = Text(str(v))
        for a, b, e, flag in marks:
            body.stylize(mark_style(e, flag), a, b)
    legend = None
    if marks:
        rule = [m for m in marks if "(AI)" not in str(m[2])]
        ai = [m for m in marks if "(AI)" in str(m[2])]
        how = "masked" if marks[0][3] else "detected, shown in full to Admin"
        legend = Text()
        if rule:
            legend.append_text(
                Text.assemble(
                    ("   ", MARK_MASKED if marks[0][3] else MARK_SEEN),
                    " ",
                    (f"{len(rule)} found by rules ({how}): ", "bold"),
                    ", ".join(dict.fromkeys(str(m[2]) for m in rule)),
                )
            )
        if ai:
            if rule:
                legend.append("\n")
            legend.append_text(
                Text.assemble(
                    ("   ", MARK_AI),
                    " ",
                    (f"{len(ai)} found by the AI model ({how}): ", "bold #a78bfa"),
                    ", ".join(dict.fromkeys(str(m[2]).replace(" (AI)", "") for m in ai)),
                )
            )
    return InfoModal(head, body, legend)


# ---------------------------------------------------------------------------------------------- app
class PardaApp(App):
    TITLE = "PardaDB"
    CSS = """
    Screen { background: $surface; }
    #topbar { dock: top; height: 1; background: #12233f; color: white; }
    .title { margin: 1 0 0 0; color: $accent; text-style: bold; }
    .side { width: 38; padding: 0 1; border-right: solid $primary-darken-2; }
    .side Select { width: 100%; }
    .bar { height: 3; }
    .lbl { height: 3; width: auto; content-align: left middle; padding: 0 1; }
    .grow { width: 1fr; }
    .line { height: auto; margin-top: 1; }
    Button { min-width: 9; margin: 0 1 0 0; }
    Button.role { min-width: 11; }
    #ex-main, #q-main { padding: 0 1; }
    #ex-fcol { width: 32; }
    #ex-fval { width: 1fr; }
    #ex-msg { height: auto; }
    #ex-grid, #cat-grid, #au-grid, #in-grid { height: 1fr; }
    #ex-tables, #q-presets { height: 1fr; }
    #q-sql { height: 8; }
    #q-results { height: 1fr; }
    .qtitle { height: auto; margin-top: 1; }
    .qgrid { height: auto; max-height: 30; }
    #mon-left { width: 1fr; padding: 0 1; }
    #mon-ml-row { height: auto; }
    #mon-ml { width: 1fr; padding-top: 1; }
    #mon-right { width: 1fr; padding: 0 1; border-left: solid $primary-darken-2; }
    #mon-dbs { height: auto; }
    #mon-bar { width: 100%; }
    #mon-bar Bar { width: 1fr; }
    #mon-events-box { height: 1fr; }
    .backend { width: 32; }
    #cat-filter { width: 1fr; }
    #cat-summary, #au-chain, #in-note { height: auto; }
    #in-banner { height: 5; content-align: center middle; text-style: bold; margin: 1 0; display: none; color: white; }
    #in-banner.ok { background: $success-darken-1; display: block; }
    #in-banner.bad { background: $error-darken-1; display: block; }
    """
    BINDINGS = [
        Binding("1", "tab('monitor')", "Monitor"),
        Binding("2", "tab('explorer')", "Explorer"),
        Binding("3", "tab('catalog')", "Catalog"),
        Binding("4", "tab('query')", "Query"),
        Binding("5", "tab('audit')", "Audit"),
        Binding("6", "tab('integrity')", "Integrity"),
        Binding("ctrl+r", "run_query", "Run query", priority=True),
        Binding("d", "dashboard", "Web dashboard"),
        Binding("m", "toggle_ml", "AI model on/off"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, rt: server.Runtime, url: str = "http://127.0.0.1:8765"):
        super().__init__()
        self.rt, self.url = rt, url
        self.role = "support"
        self.ex_backend = self.ex_table = None
        self.ex_offset, self.ex_filter, self.ex_res, self.ex_meta = 0, (None, None), None, {}
        self.cat_backend = None
        self.q_res = [None, None, None]
        self._seq, self._events, self._db_sig, self._eng_sig, self._scan_sig = 0, [], None, None, None
        self._mon_sig, self._opt_names = None, None
        self.colmap = {}
        self._last_preset, self._last_preset_t = None, 0.0

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        users = self.rt.users or [{"username": r, "display_name": r.title(), "role": r} for r in engine.ROLES]
        default_user = next((u["username"] for u in users if u["role"] == "support"), users[0]["username"])
        yield Static(id="topbar")
        with TabbedContent(initial="monitor", id="tabs"):
            with TabPane("1 Monitor", id="monitor"), Horizontal():
                with Vertical(id="mon-left"):
                    yield Static("Databases  [dim](click a scanned one to make it active)[/]", classes="title")
                    yield ClickTable(id="mon-dbs", cursor_type="row", zebra_stripes=True)
                    yield Static("PII scan", classes="title")
                    yield Static(id="mon-scan")
                    yield ProgressBar(id="mon-bar", show_eta=False)
                    yield Static("Resources and counters", classes="title")
                    yield Static(id="mon-res")
                    yield Static("AI model  [dim](names and addresses in free text)[/]", classes="title")
                    with Horizontal(id="mon-ml-row"):
                        yield Static(id="mon-ml", classes="grow")
                        yield Button("Activate AI model  (m)", id="ml-toggle", variant="success")
                with Vertical(id="mon-right"):
                    yield Static("Live events  [dim](newest first)[/]", classes="title")
                    with VerticalScroll(id="mon-events-box"):
                        yield Static(id="mon-events")
            with TabPane("2 Explorer", id="explorer"), Horizontal():
                with Vertical(classes="side"):
                    yield Select([], prompt="backend", id="ex-backend", classes="backend")
                    yield Static("Tables  [dim](click to browse)[/]", classes="title")
                    yield OptionList(id="ex-tables")
                with Vertical(id="ex-main"):
                    yield Static(id="ex-title", classes="line")
                    with Horizontal(classes="bar"):
                        yield Static("View as", classes="lbl")
                        yield Button("Admin", id="role-admin", classes="role")
                        yield Button("Support", id="role-support", classes="role", variant="primary")
                        yield Button("Analyst", id="role-analyst", classes="role")
                        yield Static(classes="grow")
                        yield Static("sensitive only", classes="lbl")
                        yield Switch(value=False, id="ex-sens", classes="view")
                        yield Static("highlight masked", classes="lbl")
                        yield Switch(value=True, id="ex-hl", classes="view")
                    with Horizontal(classes="bar"):
                        yield Static("Filter", classes="lbl")
                        yield Select([], prompt="column", id="ex-fcol")
                        yield Input(placeholder="value (use % for LIKE)", id="ex-fval")
                        yield Button("Apply", id="ex-apply", variant="success")
                        yield Button("Clear", id="ex-clear")
                    yield Static(id="ex-msg")
                    yield ClickTable(id="ex-grid", header_height=2, zebra_stripes=True, cursor_type="cell")
                    with Horizontal(classes="bar"):
                        yield Button("< Prev", id="ex-prev")
                        yield Button("Next >", id="ex-next")
                        yield Static(id="ex-page", classes="lbl grow")
            with TabPane("3 Catalog", id="catalog"):
                with Horizontal(classes="bar"):
                    yield Select([], prompt="backend", id="cat-backend", classes="backend")
                    yield Input(placeholder="filter by table, column, tag, entity or evidence", id="cat-filter")
                    yield Static("show clean", classes="lbl")
                    yield Switch(value=False, id="cat-clean")
                    yield Button("Re-scan", id="cat-rescan", variant="warning")
                yield Static(id="cat-summary")
                yield ClickTable(id="cat-grid", zebra_stripes=True, cursor_type="row")
            with TabPane("4 Query", id="query"), Horizontal():
                with Vertical(classes="side"):
                    yield Select(
                        [
                            (
                                u["display_name"]
                                if u["role"].lower() in u["display_name"].lower()
                                else f"{u['display_name']} ({u['role']})",
                                u["username"],
                            )
                            for u in users
                        ],
                        value=default_user,
                        allow_blank=False,
                        id="q-user",
                    )
                    yield Select([], prompt="backend", id="q-backend", classes="backend")
                    yield Static("Presets  [dim](click to load, double-click to run)[/]", classes="title")
                    yield OptionList(*preset_options(), id="q-presets")
                with Vertical(id="q-main"):
                    yield TextArea(server.PRESETS[0]["sql"], id="q-sql", soft_wrap=True)
                    with Horizontal(classes="bar"):
                        yield Button("Run  (ctrl+r)", id="q-run", variant="success")
                        yield Button("Compare 3 roles", id="q-compare", variant="primary")
                        yield Static(classes="grow")
                        yield Static("sensitive only", classes="lbl")
                        yield Switch(value=False, id="q-sens", classes="view")
                        yield Static("highlight masked", classes="lbl")
                        yield Switch(value=True, id="q-hl", classes="view")
                    yield Static(id="q-msg", classes="line")
                    with VerticalScroll(id="q-results"):
                        for i in range(3):
                            yield Static(id=f"q-title-{i}", classes="qtitle")
                            yield ClickTable(
                                id=f"q-grid-{i}",
                                header_height=2,
                                zebra_stripes=True,
                                cursor_type="cell",
                                classes="qgrid",
                            )
            with TabPane("5 Audit", id="audit"):
                with Horizontal(classes="bar"):
                    yield Button("Refresh", id="au-refresh", variant="primary")
                    yield Static(id="au-chain", classes="lbl grow")
                yield ClickTable(id="au-grid", zebra_stripes=True, cursor_type="row")
            with TabPane("6 Integrity", id="integrity"):
                with Horizontal(classes="bar"):
                    yield Select([], prompt="backend", id="in-backend", classes="backend")
                    yield Button("Verify SHA-256 fingerprints", id="in-verify", variant="success")
                    yield Static(
                        "Re-hashes every table and compares with the fingerprints taken at start-up.",
                        id="in-note",
                        classes="lbl grow",
                    )
                yield Static(id="in-banner")
                yield ClickTable(id="in-grid", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self):
        t = self.query_one("#mon-dbs", DataTable)
        for c in ("", "Database", "Status", "Tables", "Rows", "Latency", "Detail"):
            t.add_column(c)
        t = self.query_one("#cat-grid", DataTable)
        for c in ("table", "column", "tag", "entity", "conf", "status", "evidence"):
            t.add_column(c)
        t = self.query_one("#au-grid", DataTable)
        for c in ("time", "user", "role", "backend", "status", "sql", "detail", "hash"):
            t.add_column(c)
        t = self.query_one("#in-grid", DataTable)
        for c in ("table", "rows", "fingerprint at start", "fingerprint now", "result"):
            t.add_column(c)
        for i in (1, 2):
            self.query_one(f"#q-title-{i}").display = False
            self.query_one(f"#q-grid-{i}").display = False
        self.query_one("#q-grid-0").display = False
        self.query_one("#q-msg", Static).update(Text(server.PRESETS[0]["expect"], style="dim"))
        self.query_one("#ex-msg", Static).update(Text("Waiting for the first scan to finish…", style="yellow"))
        self.tick()
        self.set_interval(1.5, self.tick)

    # ------------------------------------------------------------------ periodic status
    def tick(self):
        try:
            st = self.rt.status()
        except Exception:
            return
        self.update_topbar(st)
        self.sync_backends()
        self.update_monitor(st)
        sc = st["scan"]
        sig = (sc["state"], sc["backend"], sc.get("finished_at"))
        if sig != self._scan_sig:
            self._scan_sig = sig
            if sc["state"] == "done":  # catalogs changed: refresh what shows that backend
                if sc["backend"] == self.cat_backend:
                    self.cat_backend = None
                if sc["backend"] == self.ex_backend:
                    self.ex_backend = None
                self.refresh_views()

    def update_topbar(self, st):
        p, bs, s = st["process"], st["backends"], st["stats"]
        conn = sum(b["status"] == "connected" for b in bs)
        act = self.rt.dbs.get(st["active_backend"] or "", {}).get("label") or "connecting…"
        t = Text()
        t.append(" PardaDB ", "bold white on #d9730d")
        t.append(f" v{st['version']} ", "dim")
        sep = (" │ ", "#5b6b8a")
        t.append_text(
            Text.assemble(
                sep,
                ("backend ", "dim"),
                (act, "bold cyan"),
                sep,
                ("DBs ", "dim"),
                (f"{conn}/{len(bs)}", "bold green" if conn == len(bs) else "bold yellow"),
                sep,
                ("CPU ", "dim"),
                (f"{p['cpu_percent']}%", "bold"),
                sep,
                ("RAM ", "dim"),
                (f"{p['rss_mb']} MB", "bold"),
                sep,
                ("last query ", "dim"),
            )
        )
        lq = s.get("last_query")
        if lq:
            t.append(f"{lq['ms']} ms" if lq.get("ms") is not None else "-", "bold")
            t.append(
                f" {lq['status']}",
                "bold white on red" if lq["status"] == "BLOCKED" else "green" if lq["status"] == "OK" else "yellow",
            )
        else:
            t.append("-", "dim")
        t.append_text(
            Text.assemble(
                sep,
                ("queries ", "dim"),
                (str(s["queries"]), "bold"),
                ("  blocked ", "dim"),
                (str(s["blocked"]), "bold red" if s["blocked"] else "bold"),
                sep,
                ("up ", "dim"),
                (dur(st["uptime_s"]), "bold"),
            )
        )
        mlst = (st.get("ml") or {}).get("status", "off")
        t.append_text(
            Text.assemble(
                sep,
                ("AI ", "dim"),
                (
                    "on" if mlst == "ready" else mlst,
                    "bold green" if mlst == "ready" else "bold yellow" if mlst == "starting" else "dim",
                ),
            )
        )
        sc = st["scan"]
        if sc["state"] == "scanning":
            pct = 100 * sc["columns_done"] // max(sc["columns_total"], 1)
            t.append_text(Text.assemble(sep, (f"scanning {pct}%", "bold yellow")))
        self.query_one("#topbar", Static).update(t)

    def update_ml(self, st):
        i = st.get("ml") or {}
        stt = i.get("status", "off")
        if stt == "ready":
            msg = Text.assemble(
                ("ON", "bold green"),
                f"  fine-tuned GLiNER, {i.get('params_m')} M params, {i.get('device')}  ·  "
                f"{i.get('texts', 0)} texts analysed, {i.get('cache_hits', 0)} cache hits",
            )
        elif stt == "starting":
            msg = Text.assemble(("STARTING", "bold yellow"), "  loading the model (about 10-20 s)…")
        else:
            msg = Text.assemble((stt.upper(), "bold"), "  free text is masked with rules only")
        self.query_one("#mon-ml", Static).update(msg)
        b = self.query_one("#ml-toggle", Button)
        b.label = (
            "Turn off AI model  (m)"
            if stt == "ready"
            else "Starting…"
            if stt == "starting"
            else "Activate AI model  (m)"
        )
        b.variant = "error" if stt == "ready" else "success"
        b.disabled = stt == "starting"

    @on(Button.Pressed, "#ml-toggle")
    def on_ml_toggle(self):
        self.action_toggle_ml()

    def action_toggle_ml(self):
        if (self.rt.status().get("ml") or {}).get("status") == "starting":
            return
        self._ml_worker()

    @work(thread=True, exclusive=True, group="ml")
    def _ml_worker(self):
        from . import ml

        if ml.client.info.get("status") == "ready" and ml.client.enabled:
            self.rt.ml_stop()
        else:
            self.rt.ml_start()
        self.call_from_thread(self.tick)

    def update_monitor(self, st):
        self.update_ml(st)
        bs = st["backends"]
        sig = tuple((b["name"], b["status"], b["tables"], b["rows"], b["latency_ms"], b["detail"]) for b in bs) + (
            st["active_backend"],
            tuple(self.rt.engines),
        )
        if sig != self._db_sig:
            self._db_sig = sig
            t = self.query_one("#mon-dbs", DataTable)
            t.clear()
            for b in bs:
                act = b["name"] == st["active_backend"]
                ready = b["name"] in self.rt.engines
                t.add_row(
                    Text("▶" if act else "", "bold cyan"),
                    Text(b["label"], "bold" if act else ""),
                    Text(
                        ("● " + b["status"]) + ("" if ready or b["status"] != "connected" else " (scanning)"),
                        STATUS_STYLE.get(b["status"], "white"),
                    ),
                    Text(str(b["tables"]), justify="right"),
                    Text(f"{b['rows']:,}", justify="right"),
                    Text(f"{b['latency_ms']} ms" if b["latency_ms"] is not None else "-", justify="right"),
                    Text(str(b["detail"] or "")[:60], "dim"),
                    key=b["name"],
                )
        sc, p, s = st["scan"], st["process"], st["stats"]
        msig = (
            sc["state"],
            sc["columns_done"],
            sc["current"],
            p["cpu_percent"],
            p["rss_mb"],
            s["queries"],
            s["masked_cells"],
            st["uptime_s"] // 5,
        )
        if msig != self._mon_sig:
            self._mon_sig = msig
            label = self.rt.dbs.get(sc["backend"] or "", {}).get("label", "")
            bar = self.query_one("#mon-bar", ProgressBar)
            if sc["state"] == "scanning":
                txt = Text.assemble(
                    ("Scanning ", "bold yellow"),
                    (f"{label}: ", "bold"),
                    (str(sc["current"] or "…"), "cyan"),
                    (
                        f"   {sc['columns_done']}/{sc['columns_total']} columns · "
                        f"{sc['tables_done']}/{sc['tables_total']} tables · "
                        f"{sc['elapsed_s']} s",
                        "dim",
                    ),
                )
                bar.update(total=max(sc["columns_total"], 1), progress=sc["columns_done"])
            elif sc["state"] == "done":
                txt = Text.assemble(
                    ("Scan complete ", "bold green"),
                    (f"on {label}: ", "bold"),
                    (f"{sc['tagged']} sensitive columns", "bold red"),
                    f" of {sc['columns_total']} in {sc['elapsed_s']} s",
                )
                bar.update(total=max(sc["columns_total"], 1), progress=max(sc["columns_total"], 1))
            else:
                txt = Text("Waiting for a database to connect…", "yellow")
                bar.update(total=1, progress=0)
            self.query_one("#mon-scan", Static).update(txt)
            lq = s.get("last_query") or {}
            res = Text.assemble(
                ("CPU ", "dim"),
                (f"{p['cpu_percent']}%", "bold"),
                ("   RAM ", "dim"),
                (f"{p['rss_mb']} MB", "bold"),
                ("   threads ", "dim"),
                (str(p["threads"]), "bold"),
                ("   state on disk ", "dim"),
                (f"{p['disk_mb']} MB", "bold"),
                ("   pid ", "dim"),
                (str(p["pid"]), "bold"),
                "\n",
                ("Queries ", "dim"),
                (f"{s['queries']:,}", "bold cyan"),
                ("   Blocked ", "dim"),
                (f"{s['blocked']:,}", "bold red" if s["blocked"] else "bold"),
                ("   Masked cells ", "dim"),
                (f"{s['masked_cells']:,}", "bold magenta"),
                ("   Uptime ", "dim"),
                (dur(st["uptime_s"]), "bold"),
                "\n",
                ("Last query ", "dim"),
                (f"{lq.get('status', '-')} ", "bold red" if lq.get("status") == "BLOCKED" else "bold green"),
                (
                    f"{lq.get('ms', '-')} ms · {lq.get('user') or lq.get('role') or ''} · {lq.get('backend') or ''} · "
                    f"{lq.get('rows', 0)} rows · {lq.get('at', '')}"
                    if lq
                    else "",
                    "dim",
                ),
            )
            self.query_one("#mon-res", Static).update(res)
        new = [e for e in st["events"] if e["seq"] > self._seq]
        if new:
            self._seq = new[-1]["seq"]
            self._events = (self._events + new)[-60:]
            out = Text()
            for e in reversed(self._events):
                out.append(f"{e['ts']} ", "dim")
                out.append(str(e["msg"]) + "\n", LEVEL_STYLE.get(e["level"], "white"))
            self.query_one("#mon-events", Static).update(out)

    # ------------------------------------------------------------------ backend selection
    def ready(self):
        return [n for n in self.rt.order if n in self.rt.engines]

    def sync_backends(self):
        names = self.ready()
        sig = (tuple(names), self.rt.active)
        if sig == self._eng_sig:
            return
        first = self._eng_sig is None or not self._eng_sig[0]
        self._eng_sig = sig
        opts = [(self.rt.dbs[n]["label"], n) for n in names]
        reset = names != self._opt_names
        self._opt_names = names
        for sel in self.query(".backend").results(Select):
            if reset:
                sel.set_options(opts)
            if self.rt.active in names and sel.value != self.rt.active:
                sel.value = self.rt.active
        if names and (first or self.rt.active not in names):
            if self.rt.active not in names:
                self.rt.active = names[0]
            self.refresh_views()

    @on(Select.Changed, ".backend")
    def _backend_changed(self, ev: Select.Changed):
        v = ev.value
        if v == Select.NULL or v not in self.rt.engines:
            return
        if v != self.rt.active:
            self.rt.active = v
            self._eng_sig = None
            self.sync_backends()
        self.refresh_views()

    def refresh_views(self):
        n = self.rt.active
        if n not in self.rt.engines:
            return
        if self.ex_backend != n:
            self.ex_backend = n
            self.load_tables(n)
        if self.cat_backend != n:
            self.load_catalog()

    @on(DataTable.RowSelected, "#mon-dbs")
    def _db_clicked(self, ev: DataTable.RowSelected):
        n = ev.row_key.value
        if n in self.rt.engines and n != self.rt.active:
            self.rt.active = n
            self._eng_sig = None
            self.sync_backends()
            self.refresh_views()
            self.notify(f"Active backend: {self.rt.dbs[n]['label']}")
        elif n not in self.rt.engines:
            self.notify(f"{self.rt.dbs[n]['label']} is {self.rt.dbs[n]['status']}; not scanned yet", severity="warning")

    # ------------------------------------------------------------------ explorer
    @work(thread=True, exclusive=True, group="tables")
    def load_tables(self, n):
        try:
            data = self.rt.tables(n)
        except Exception as e:
            data = {"error": f"{type(e).__name__}: {e}"}
        self.call_from_thread(self.show_tables, n, data)

    def show_tables(self, n, data):
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

    def set_filter_columns(self):
        cols = self.ex_meta.get(self.ex_table, {}).get("columns", [])
        sel = self.query_one("#ex-fcol", Select)
        sel.set_options(
            [(c["name"] + (f"  [{TAG_SHORT.get(c['tag'], c['tag'])}]" if c["tag"] else ""), c["name"]) for c in cols]
        )

    @on(OptionList.OptionSelected, "#ex-tables")
    def _table_clicked(self, ev: OptionList.OptionSelected):
        self.ex_table, self.ex_offset, self.ex_filter = ev.option.id, 0, (None, None)
        self.query_one("#ex-fval", Input).value = ""
        self.set_filter_columns()
        self.browse()

    def user_for(self, role):
        return next((u["username"] for u in self.rt.users if u["role"] == role), f"tui:{role}")

    def browse(self):
        n = self.rt.active
        if not self.ex_table or n not in self.rt.engines:
            return
        self.query_one("#ex-msg", Static).update(Text(f"Loading {self.ex_table} as {self.role}…", "dim"))
        self.do_browse(n, self.ex_table, self.role, self.ex_offset, *self.ex_filter)

    @work(thread=True, exclusive=True, group="browse")
    def do_browse(self, n, table, role, offset, fcol, fval):
        try:
            res = self.rt.count(self.rt.engines[n].browse(table, role, self.user_for(role), offset, PAGE, fcol, fval))
        except Exception as e:
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        res.setdefault("role", role)
        self.call_from_thread(self.show_browse, res)

    def show_browse(self, res):
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
            if res.get("blocked"):
                msg.update(Text.assemble((" BLOCKED ", "bold white on red"), " ", (str(res.get("reason")), "bold red")))
            else:
                msg.update(Text.assemble((" ERROR ", "bold white on red"), " ", (str(res.get("error")), "red")))
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
    def _role_clicked(self, ev: Button.Pressed):
        self.role = ev.button.id.split("-", 1)[1]
        for b in self.query(".role").results(Button):
            b.variant = "primary" if b.id == ev.button.id else "default"
        self.browse()

    @on(Button.Pressed, "#ex-apply")
    def _apply_filter(self):
        col = self.query_one("#ex-fcol", Select).value
        val = self.query_one("#ex-fval", Input).value
        if col == Select.NULL:
            self.notify("Pick a column to filter on", severity="warning")
            return
        self.ex_filter, self.ex_offset = (col, val), 0
        self.browse()

    @on(Input.Submitted, "#ex-fval")
    def _filter_submitted(self):
        self._apply_filter()

    @on(Button.Pressed, "#ex-clear")
    def _clear_filter(self):
        self.query_one("#ex-fval", Input).value = ""
        self.query_one("#ex-fcol", Select).clear()
        self.ex_filter, self.ex_offset = (None, None), 0
        self.browse()

    @on(Button.Pressed, "#ex-prev")
    def _prev(self):
        if self.ex_offset > 0:
            self.ex_offset = max(0, self.ex_offset - PAGE)
            self.browse()

    @on(Button.Pressed, "#ex-next")
    def _next(self):
        total = (self.ex_res or {}).get("total")
        if self.ex_res and (total is None or self.ex_offset + PAGE < total):
            self.ex_offset += PAGE
            self.browse()

    @on(Switch.Changed, ".view")
    def _view_toggled(self, ev: Switch.Changed):
        if ev.switch.id.startswith("ex-"):
            if self.ex_res:
                self.show_browse(self.ex_res)
        elif any(self.q_res):
            self.show_query([r for r in self.q_res if r is not None])

    # ------------------------------------------------------------------ result cells -> modal
    @on(DataTable.CellSelected)
    def _cell_clicked(self, ev: DataTable.CellSelected):
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
    def load_catalog(self):
        n = self.rt.active
        cat = self.rt.catalogs.get(n)
        if not cat:
            return
        self.cat_backend = n
        q = self.query_one("#cat-filter", Input).value.strip().lower()
        show_clean = self.query_one("#cat-clean", Switch).value
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
    @on(Switch.Changed, "#cat-clean")
    def _cat_filter(self):
        self.load_catalog()

    @on(DataTable.RowSelected, "#cat-grid")
    def _cat_row(self, ev: DataTable.RowSelected):
        row = ev.data_table.get_row(ev.row_key)
        cat = self.rt.catalogs.get(self.cat_backend) or {}
        c = next((x for x in cat.get("columns", []) if x["table"] == str(row[0]) and x["column"] == str(row[1])), None)
        if not c:
            return
        pol = self.rt.policy.get("tags", {}).get(c["tag"] or "", {})
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
        self.push_screen(InfoModal(head, body, legend))

    @on(Button.Pressed, "#cat-rescan")
    def _rescan(self):
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
    def do_scan(self, n):
        try:
            self.rt.run_scan(n)
        except Exception as e:
            self.call_from_thread(self.notify, f"Scan failed: {e}", severity="error")

    # ------------------------------------------------------------------ query
    @on(OptionList.OptionSelected, "#q-presets")
    def _preset_clicked(self, ev: OptionList.OptionSelected):
        p = server.PRESETS[int(ev.option.id)]
        now = time.monotonic()
        double = self._last_preset == (ev.option.id,) and now - self._last_preset_t < 0.6
        self._last_preset, self._last_preset_t = (ev.option.id,), now
        self.query_one("#q-sql", TextArea).text = p["sql"]
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

    def _query_args(self):
        sql = self.query_one("#q-sql", TextArea).text.strip()
        n = self.rt.active
        if not sql:
            self.notify("Type a SQL query first", severity="warning")
            return None
        if n not in self.rt.engines:
            self.notify("No database scanned yet", severity="warning")
            return None
        user = self.query_one("#q-user", Select).value
        role = self.rt.role_of(user, None)
        return n, sql, role, user

    @on(Button.Pressed, "#q-run")
    def action_run_query(self):
        tabs = self.query_one("#tabs", TabbedContent)
        if tabs.active != "query":
            self.action_tab("query")
            return
        a = self._query_args()
        if a:
            self.query_one("#q-msg", Static).update(Text(f"Running as {a[3]} ({a[2]})…", "dim"))
            self.do_query(*a)

    @work(thread=True, exclusive=True, group="query")
    def do_query(self, n, sql, role, user):
        try:
            n2 = self.rt.route(n, sql)
            res = self.rt.count(self.rt.engines[n2].run(sql, role, user))
        except Exception as e:
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "role": role, "user": user}
        self.call_from_thread(self.show_query, [res])
        if res.get("write"):
            try:
                self.rt.after_write(res)
            except Exception as e:
                self.call_from_thread(self.notify, f"Re-scan failed: {e}", severity="error")

    @on(Button.Pressed, "#q-compare")
    def _compare(self):
        a = self._query_args()
        if a:
            self.query_one("#q-msg", Static).update(Text("Comparing Admin, Support and Analyst…", "dim"))
            self.do_compare(a[0], a[1])

    @work(thread=True, exclusive=True, group="query")
    def do_compare(self, n, sql):
        out = []
        try:
            n2 = self.rt.route(n, sql)
            for r in engine.ROLES:
                res = self.rt.count(self.rt.engines[n2].run(sql, r, f"compare:{r}"))
                out.append(res)
        except Exception as e:
            out.append({"ok": False, "error": f"{type(e).__name__}: {e}", "role": "?"})
        self.call_from_thread(self.show_query, out)

    def show_query(self, results):
        self.q_res = (results + [None, None, None])[:3]
        summary = []
        for i in range(3):
            title, grid = self.query_one(f"#q-title-{i}", Static), self.query_one(f"#q-grid-{i}", DataTable)
            res = self.q_res[i]
            title.display = res is not None
            grid.display = bool(res and res.get("ok"))
            if res is None:
                continue
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
                if res.get("write"):
                    head.append_text(
                        Text.assemble(
                            (" WRITE ", "bold black on yellow"),
                            (f" {max(res.get('affected', 0), 0)} row(s) affected", "yellow"),
                        )
                    )
                else:
                    more = f" (showing {QUERY_ROWS})" if res["row_count"] > QUERY_ROWS else ""
                    head.append_text(
                        Text.assemble(
                            (f"{res['row_count']} rows{more}", "bold green"),
                            (f" · {res.get('masked_cells', 0)} cells masked", "magenta"),
                        )
                    )
                summary.append("OK")
            elif res.get("blocked"):
                head.append_text(
                    Text.assemble((" BLOCKED ", "bold white on red"), " ", (str(res.get("reason")), "bold red"))
                )
                summary.append("BLOCKED")
            else:
                head.append_text(Text.assemble((" ERROR ", "bold white on red"), " ", (str(res.get("error")), "red")))
                summary.append("ERROR")
            head.append(f"  {res.get('elapsed_ms', '-')} ms", "dim")
            title.update(head)
        self.query_one("#q-msg", Static).update(
            Text(" · ".join(summary) + "   click any cell for its full value", "dim")
        )
        self.query_one("#q-results", VerticalScroll).scroll_home(animate=False)

    # ------------------------------------------------------------------ audit
    @work(thread=True, exclusive=True, group="audit")
    def load_audit(self):
        try:
            data = audit.read(50)
        except Exception as e:
            data = {"chain_ok": False, "total": 0, "records": [], "error": str(e)}
        self.call_from_thread(self.show_audit, data)

    def show_audit(self, data):
        ok = data.get("chain_ok")
        self.query_one("#au-chain", Static).update(
            Text.assemble(
                (" CHAIN INTACT " if ok else " CHAIN BROKEN ", "bold white on green" if ok else "bold white on red"),
                (f"  {data.get('total', 0):,} records, each holding the SHA-256 of the previous one", "dim"),
                (f"   {data['error']}" if data.get("error") else "", "red"),
                ("   (showing newest 50)", "dim"),
            )
        )
        g = self.query_one("#au-grid", DataTable)
        g.clear()
        for r in data.get("records", []):
            st = r.get("status", "")
            sql = " ".join(str(r.get("sql", "")).split())
            det = r.get("reason") or (
                ", ".join(f"{k}:{v}" for k, v in (r.get("masked") or {}).items())
                if r.get("masked")
                else f"{r.get('rows', '')} rows"
                if "rows" in r
                else ""
            )
            g.add_row(
                str(r.get("ts", ""))[11:] or r.get("ts", ""),
                r.get("user") or "",
                r.get("role") or "",
                r.get("backend") or "",
                Text(st, AUDIT_STYLE.get(st, "white")),
                Text(sql[:70] + ("…" if len(sql) > 70 else "")),
                Text(str(det)[:60], "red" if st == "BLOCKED" else "dim"),
                Text(str(r.get("hash", ""))[:12], "dim"),
            )

    @on(Button.Pressed, "#au-refresh")
    def _audit_refresh(self):
        self.load_audit()

    # ------------------------------------------------------------------ integrity
    @on(Button.Pressed, "#in-verify")
    def _verify(self):
        n = self.rt.active
        if n not in self.rt.before:
            self.notify("No connected backend to verify", severity="warning")
            return
        self.query_one("#in-note", Static).update(Text(f"Hashing every table of {self.rt.dbs[n]['label']}…", "yellow"))
        self.do_verify(n)

    @work(thread=True, exclusive=True, group="integrity")
    def do_verify(self, n):
        try:
            res = self.rt.integrity(n)
        except Exception as e:
            res = {"all_unchanged": False, "tables": [], "error": f"{type(e).__name__}: {e}"}
        self.call_from_thread(self.show_verify, n, res)

    def show_verify(self, n, res):
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

    # ------------------------------------------------------------------ tabs, keys
    @on(TabbedContent.TabActivated)
    def _tab_activated(self, ev: TabbedContent.TabActivated):
        pid = ev.pane.id if ev.pane else ""
        if pid == "audit":
            self.load_audit()
        elif pid in ("explorer", "catalog"):
            self.refresh_views()

    def action_tab(self, tab: str):
        tc = self.query_one("#tabs", TabbedContent)
        # Move focus to the tab strip first: if a widget in the old pane kept focus, Textual would
        # hand it to a sibling in that pane while hiding it, and TabbedContent would switch back.
        tc.query_one(Tabs).focus()
        tc.active = tab

    def action_dashboard(self):
        webbrowser.open(self.url)
        self.notify(f"Opening {self.url}")
