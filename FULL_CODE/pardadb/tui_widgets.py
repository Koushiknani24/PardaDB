"""Reusable pieces of the Textual TUI: colour styles, cell formatting, ClickTable and the detail modal."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static
from textual.widgets.option_list import Option

from . import masks, server

# A query/browse result as returned by the engine: JSON-like, keys vary with the outcome.
Result = dict[str, Any]
# One highlighted span inside a cell: [start, end, entity, masked flag].
Mark = list[Any]

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

PAGE, QUERY_ROWS, CELL_MAX = 25, 50, 40
_FLAT = {ord("\n"): " ", ord("\r"): " ", ord("\t"): " "}


def mark_style(ent: object, flag: object) -> str:
    """Highlight style for one span: AI-found, masked by rules, or detected but shown in full."""
    return MARK_AI if "(AI)" in str(ent) else MARK_MASKED if flag else MARK_SEEN


def strat(s: str | None) -> str:
    """Short display name of a masking strategy (no strategy means the value is shown in full)."""
    return masks.DISPLAY.get(s, s) if s else "FULL"


def dur(s: float | None) -> str:
    """Compact duration such as '1h 05m', '3m 07s' or '42s'."""
    s = int(s or 0)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def col_label(c: dict[str, Any]) -> Text:
    """Two-line column header: name, then sensitivity tag and masking strategy."""
    t = Text(str(c["name"]), style="bold")
    t.append("\n")
    if c.get("tag"):
        t.append(TAG_SHORT.get(c["tag"], c["tag"]), TAG_STYLE.get(c["tag"], "white"))
        t.append(" ")
    s = strat(c.get("strategy"))
    t.append(s, STRAT_STYLE.get(s, "white"))
    return t


def cell_text(
    v: Any,
    strategy: str | None,
    marks: list[Mark] | None = None,
    limit: int = CELL_MAX,
    highlight: bool = True,
) -> Text:
    """One grid cell, flattened to a single line, coloured by strategy and truncated to `limit`."""
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


def fill_grid(
    table: DataTable, res: Result, max_rows: int, sens_only: bool = False, highlight: bool = True
) -> list[int]:
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


def outcome_text(res: Result) -> Text:
    """Red BLOCKED/ERROR badge with the reason, for a result that did not run."""
    if res.get("blocked"):
        return Text.assemble((" BLOCKED ", "bold white on red"), " ", (str(res.get("reason")), "bold red"))
    return Text.assemble((" ERROR ", "bold white on red"), " ", (str(res.get("error")), "red"))


class ClickTable(DataTable):
    """DataTable where a single mouse click on a cell/row selects it (stock DataTable needs two)."""

    def _on_click(self, event: events.Click) -> None:
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


def preset_options() -> list[Option | None]:
    """Presets grouped by operation type: a disabled heading row per category."""
    opts: list[Option | None] = []
    last: object = object()
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

    def __init__(self, head: Text, body: Text, legend: Text | None = None) -> None:
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
    def action_close(self) -> None:
        self.dismiss()


def _cell_head(res: Result, row: int, c: dict[str, Any]) -> Text:
    """Modal heading for one cell: column, tag, entity, viewing role, strategy and source columns."""
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
    return head


def _marks_legend(marks: list[Mark]) -> Text | None:
    """Legend that counts rule-found and AI-found spans separately, or None when nothing was found."""
    if not marks:
        return None
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
    return legend


def cell_modal(res: Result, row: int, col: int) -> InfoModal:
    """Modal with the full value of one result cell, its PII spans highlighted."""
    c = res["columns"][col]
    v = res["rows"][row][col]
    marks = (res.get("marks") or {}).get(f"{row}:{col}") or []
    body = Text("NULL", style="dim italic")
    if v is not None:
        body = Text(str(v))
        for a, b, e, flag in marks:
            body.stylize(mark_style(e, flag), a, b)
    return InfoModal(_cell_head(res, row, c), body, _marks_legend(marks))
