"""Full-screen, mouse-clickable terminal UI for PardaDB (Textual).

Runs in the same process as the dashboard and talks to the Runtime directly (no HTTP).
Everything slow (browse, query, integrity, scan) runs in worker threads so the UI never freezes.
This module holds the layout, key bindings and tab switching; each tab's behaviour lives in
tui_monitor, tui_explorer and tui_query, and shared widgets in tui_widgets.
"""

from __future__ import annotations

import webbrowser

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
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
)

from . import engine, server
from .tui_explorer import ExplorerTab
from .tui_monitor import TICK_S, MonitorTab
from .tui_query import QueryTab
from .tui_widgets import ClickTable, SqlArea, help_modal, preset_options

# Column headers of the grids whose columns never change (result grids get theirs per query).
FIXED_COLUMNS = {
    "#mon-dbs": ("", "Database", "Status", "Tables", "Rows", "Latency", "Detail"),
    "#cat-grid": ("table", "column", "tag", "entity", "conf", "status", "evidence"),
    "#au-grid": ("time", "user", "role", "backend", "status", "sql", "detail", "hash"),
    "#in-grid": ("table", "rows", "fingerprint at start", "fingerprint now", "result"),
}
# Bindings shown in the footer (and active) only on these tabs; every other binding applies everywhere.
TAB_ACTIONS = {"role": ("explorer", "query"), "page": ("explorer",)}


class PardaApp(MonitorTab, ExplorerTab, QueryTab):
    """The PardaDB TUI: six tabs (Monitor, Explorer, Catalog, Query, Audit, Integrity) over one Runtime."""

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
        Binding("a", "role('admin')", "Admin"),
        Binding("s", "role('support')", "Support"),
        Binding("n", "role('analyst')", "aNalyst"),
        Binding("left_square_bracket", "page(-1)", "Prev page", key_display="["),
        Binding("right_square_bracket", "page(1)", "Next page", key_display="]"),
        Binding("ctrl+r", "run_query", "Run query", priority=True),
        Binding("d", "dashboard", "Web dashboard"),
        Binding("m", "toggle_ml", "AI model on/off"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("f1", "help", "Help", show=False),
        Binding("q", "quit", "Quit"),
    ]

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        with TabbedContent(initial="monitor", id="tabs"):
            yield from self._compose_monitor()
            yield from self._compose_explorer()
            yield from self._compose_catalog()
            yield from self._compose_query()
            yield from self._compose_audit()
            yield from self._compose_integrity()
        yield Footer()

    def _compose_monitor(self) -> ComposeResult:
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

    def _compose_explorer(self) -> ComposeResult:
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

    def _compose_catalog(self) -> ComposeResult:
        with TabPane("3 Catalog", id="catalog"):
            with Horizontal(classes="bar"):
                yield Select([], prompt="backend", id="cat-backend", classes="backend")
                yield Input(placeholder="filter by table, column, tag, entity or evidence", id="cat-filter")
                yield Static("show clean", classes="lbl")
                yield Switch(value=False, id="cat-clean")
                yield Button("Re-scan", id="cat-rescan", variant="warning")
            yield Static(id="cat-summary")
            yield ClickTable(id="cat-grid", zebra_stripes=True, cursor_type="row")

    def _user_select(self) -> Select:
        """Picker of the dashboard users to run queries as, defaulting to the first Support user."""
        users = self.rt.users or [{"username": r, "display_name": r.title(), "role": r} for r in engine.ROLES]
        self.q_user_role = {u["username"]: u["role"] for u in users}
        default_user = next((u["username"] for u in users if u["role"] == "support"), users[0]["username"])
        return Select(
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

    def _compose_query(self) -> ComposeResult:
        with TabPane("4 Query", id="query"), Horizontal():
            with Vertical(classes="side"):
                yield self._user_select()
                yield Select([], prompt="backend", id="q-backend", classes="backend")
                yield Static("Presets  [dim](click to load, double-click to run)[/]", classes="title")
                yield OptionList(*preset_options(), id="q-presets")
            with Vertical(id="q-main"):
                yield SqlArea(server.PRESETS[0]["sql"], id="q-sql", soft_wrap=True)
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

    def _compose_audit(self) -> ComposeResult:
        with TabPane("5 Audit", id="audit"):
            with Horizontal(classes="bar"):
                yield Button("Refresh", id="au-refresh", variant="primary")
                yield Static(id="au-chain", classes="lbl grow")
            yield ClickTable(id="au-grid", zebra_stripes=True, cursor_type="row")

    def _compose_integrity(self) -> ComposeResult:
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

    def on_mount(self) -> None:
        for sel, cols in FIXED_COLUMNS.items():
            t = self.query_one(sel, DataTable)
            for i, c in enumerate(cols):
                t.add_column(c, key=f"c{i}")
        for i in (1, 2):
            self.query_one(f"#q-title-{i}").display = False
            self.query_one(f"#q-grid-{i}").display = False
        self.query_one("#q-grid-0").display = False
        self.query_one("#q-msg", Static).update(Text(server.PRESETS[0]["expect"], style="dim"))
        wait = Text("Scanning… tables appear when the first scan finishes.", "yellow")
        self.query_one("#ex-msg", Static).update(wait)
        self.poll_status()
        self.set_interval(TICK_S, self.tick)

    # ------------------------------------------------------------------ tabs, keys
    @on(TabbedContent.TabActivated)
    def _tab_activated(self, ev: TabbedContent.TabActivated) -> None:
        pid = ev.pane.id if ev.pane else ""
        self.refresh_bindings()  # footer hints follow the tab
        if pid == "audit":
            self.load_audit()
        elif pid in ("explorer", "catalog"):
            self.refresh_views()
        elif pid == "monitor" and self.st:
            self.update_monitor(self.st)  # it was not redrawn while hidden

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide tab-specific bindings (roles, paging) from the footer and keys on tabs they do not apply to."""
        tabs = TAB_ACTIONS.get(action)
        if tabs is None:
            return True
        try:
            return self.active_tab() in tabs
        except NoMatches:  # still composing
            return False

    def action_help(self) -> None:
        self.push_screen(help_modal())

    def action_tab(self, tab: str) -> None:
        tc = self.query_one("#tabs", TabbedContent)
        # Move focus to the tab strip first: if a widget in the old pane kept focus, Textual would
        # hand it to a sibling in that pane while hiding it, and TabbedContent would switch back.
        tc.query_one(Tabs).focus()
        tc.active = tab

    def action_dashboard(self) -> None:
        webbrowser.open(self.url)
        self.notify(f"Opening {self.url}")
