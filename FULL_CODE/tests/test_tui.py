"""Textual TUI (pilot tests on the synthetic bank) and the classic line TUI's argument handling."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from rich.console import Console
from textual.pilot import Pilot
from textual.widgets import Button, DataTable, Select, Static, TabbedContent

from pardadb import server
from pardadb.app import PardaApp
from pardadb.tui import TUI
from pardadb.tui_monitor import IDLE_AFTER, SLOW_EVERY, activity_sig, waiting_text
from pardadb.tui_widgets import InfoModal, SqlArea, one_line, outcome_text

TABS = ("monitor", "explorer", "catalog", "query", "audit", "integrity")


@pytest.fixture(scope="module")
def rt(bank_db: Any) -> server.Runtime:
    """A runtime over the synthetic bank, connected and scanned the way `pardadb` boots it."""
    r = server.Runtime(["sqlite"], log=None)
    r.connect("sqlite")
    r.fingerprint("sqlite")
    r.run_scan("sqlite")
    return r


def run_app(rt: server.Runtime, body: Callable[[PardaApp, Pilot], Awaitable[None]]) -> None:
    """Run `body` against a headless PardaApp once its Monitor tab shows the database."""

    async def main() -> None:
        app = PardaApp(rt, "http://127.0.0.1:1")
        async with app.run_test(size=(160, 50)) as pilot:
            for _ in range(100):
                if app.query_one("#mon-dbs", DataTable).row_count:
                    break
                await pilot.pause(0.05)
            await body(app, pilot)

    asyncio.run(main())


async def settle(app: PardaApp, pilot: Pilot) -> None:
    """Let workers finish and their results reach the screen."""
    await pilot.pause(0.05)
    await app.workers.wait_for_complete()
    await pilot.pause(0.05)


def text_of(app: PardaApp, sel: str) -> str:
    return str(app.query_one(sel, Static).render())


# ---------------------------------------------------------------- Textual app
def test_app_starts_and_every_tab_opens(rt: server.Runtime) -> None:
    async def body(app: PardaApp, pilot: Pilot) -> None:
        assert "PardaDB" in text_of(app, "#topbar")
        assert app.query_one("#mon-dbs", DataTable).get_row("sqlite")[1].plain == "SQLite"
        tabs = app.query_one("#tabs", TabbedContent)
        for key, tab in zip("123456", TABS, strict=True):
            await pilot.press(key)
            await settle(app, pilot)
            assert tabs.active == tab
        assert app.query_one("#ex-grid", DataTable).row_count > 0  # loaded when the tab was opened
        assert app.query_one("#cat-grid", DataTable).row_count > 0

    run_app(rt, body)


def test_role_hotkeys_change_the_role_only_where_roles_apply(rt: server.Runtime) -> None:
    async def body(app: PardaApp, pilot: Pilot) -> None:
        await pilot.press("a")  # Monitor tab: no roles here, so the key is ignored
        assert app.role == "support"
        await pilot.press("2")
        await settle(app, pilot)
        await pilot.press("a")
        await settle(app, pilot)
        assert app.role == "admin"
        assert app.query_one("#role-admin", Button).variant == "primary"
        assert app.query_one("#role-support", Button).variant == "default"
        assert "ADMIN" in text_of(app, "#ex-title")
        await pilot.press("n")
        await settle(app, pilot)
        assert app.role == "analyst"
        assert app.query_one("#q-user", Select).value == "analyst"  # the Query tab follows
        assert "ANALYST" in text_of(app, "#ex-title")

    run_app(rt, body)


def test_help_screen_opens_and_closes(rt: server.Runtime) -> None:
    async def body(app: PardaApp, pilot: Pilot) -> None:
        await pilot.press("question_mark")
        assert isinstance(app.screen, InfoModal)
        help_text = str(app.screen.query_one("#m-body", Static).render())
        assert "a / s / n" in help_text and "Integrity" in help_text
        await pilot.press("escape")
        assert not isinstance(app.screen, InfoModal)
        assert app.query_one("#tabs", TabbedContent).active == "monitor"

    run_app(rt, body)


def test_query_history_recalls_previous_queries(rt: server.Runtime) -> None:
    async def body(app: PardaApp, pilot: Pilot) -> None:
        await pilot.press("4")
        editor = app.query_one("#q-sql", SqlArea)
        for sql in ("SELECT full_name FROM customers", "SELECT mobile FROM customers"):
            editor.text = sql
            await pilot.press("ctrl+r")
            await settle(app, pilot)
        assert "SUPPORT" in text_of(app, "#q-title-0")
        editor.focus()
        editor.text = "SELECT 1"  # unsent draft
        await pilot.press("up")
        assert editor.text == "SELECT mobile FROM customers"
        await pilot.press("up")
        assert editor.text == "SELECT full_name FROM customers"
        await pilot.press("up")  # oldest entry: stays put
        assert editor.text == "SELECT full_name FROM customers"
        await pilot.press("down", "down")
        assert editor.text == "SELECT 1"  # back to the draft

    run_app(rt, body)


def test_blocked_query_shows_one_readable_blocked_line(rt: server.Runtime) -> None:
    async def body(app: PardaApp, pilot: Pilot) -> None:
        await pilot.press("4")
        app.query_one("#q-sql", SqlArea).text = "DELETE FROM customers"
        await pilot.press("ctrl+r")
        await settle(app, pilot)
        title = text_of(app, "#q-title-0")
        assert "BLOCKED" in title and "read-only" in title
        assert "\n" not in title
        assert not app.query_one("#q-grid-0", DataTable).display
        assert "BLOCKED" in text_of(app, "#q-msg")

    run_app(rt, body)


def test_idle_app_polls_less_often(rt: server.Runtime) -> None:
    app = PardaApp(rt)
    polls: list[int] = []
    app.poll_status = lambda: polls.append(1)  # type: ignore[method-assign]
    for _ in range(SLOW_EVERY * 2):
        app.tick()
    assert len(polls) == SLOW_EVERY * 2  # busy: every tick polls
    polls.clear()
    app._quiet_ticks = IDLE_AFTER
    for _ in range(SLOW_EVERY * 2):
        app.tick()
    assert len(polls) == 2  # idle: one poll per SLOW_EVERY ticks


# ---------------------------------------------------------------- helpers
def _status(state: str = "idle", db_status: str = "unreachable", cpu: float = 1.0) -> dict[str, Any]:
    return {
        "scan": {"state": state, "columns_done": 3, "columns_total": 12},
        "stats": {"queries": 0},
        "events": [],
        "active_backend": None,
        "ml": {"status": "off"},
        "backends": [{"status": db_status, "detail": "connection refused\n  (errno 111)"}],
        "process": {"cpu_percent": cpu},
    }


def test_waiting_text_explains_what_the_app_is_waiting_for() -> None:
    assert waiting_text(_status()).plain == (
        "Database not connected - retrying every 10 s.  connection refused (errno 111)"
    )
    assert waiting_text(_status("scanning", "connected")).plain.startswith("Scanning… 25%")
    assert "about to start" in waiting_text(_status("idle", "connected")).plain


def test_activity_signature_ignores_cpu_but_not_scans() -> None:
    assert activity_sig(_status(cpu=1.0)) == activity_sig(_status(cpu=50.0))
    assert activity_sig(_status()) != activity_sig(_status("scanning"))


def test_errors_render_on_one_line() -> None:
    assert one_line("near 'FROM':\n  syntax   error") == "near 'FROM': syntax error"
    assert one_line("x" * 500, 20) == "x" * 19 + "…"
    t = outcome_text({"ok": False, "error": "Traceback\nline 2\nline 3"})
    assert t.plain == " ERROR  Traceback line 2 line 3"


# ---------------------------------------------------------------- classic TUI
@pytest.fixture
def classic(rt: server.Runtime) -> tuple[TUI, io.StringIO]:
    t = TUI(rt, "http://127.0.0.1:1")
    out = io.StringIO()
    t.con = Console(file=out, width=1000, color_system=None)
    return t, out


def test_classic_browse_before_any_scan_says_not_scanned(classic: tuple[TUI, io.StringIO]) -> None:
    t, out = classic
    t.rt = server.Runtime(["sqlite"], log=None)  # nothing connected or scanned yet
    t.c_browse("customers")
    assert "not scanned yet" in out.getvalue()


@pytest.mark.parametrize("cmd", ["c_events", "c_audit"])
@pytest.mark.parametrize("arg", ["abc", "0", "-3", "2.5"])
def test_classic_count_arguments_are_validated(classic: tuple[TUI, io.StringIO], cmd: str, arg: str) -> None:
    t, out = classic
    getattr(t, cmd)(arg)
    assert "positive whole number" in out.getvalue()
    assert f"{cmd[2:]} [n]" in out.getvalue()  # the usage's brackets are printed, not parsed as markup
    assert f"'{arg}'" in out.getvalue()


def test_classic_events_with_a_valid_count_prints_that_many(classic: tuple[TUI, io.StringIO]) -> None:
    t, out = classic
    t.c_events("2")
    lines = out.getvalue().strip().splitlines()
    last_two = list(t.rt.events)[-2:]
    assert [ln.split(" ", 1)[0] for ln in lines] == [e["ts"] for e in last_two]
    assert lines[-1].endswith(last_two[-1]["msg"].strip())
