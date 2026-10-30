"""Shared state of the Textual TUI, inherited by each tab's controller class.

Each tab's handlers live in their own App subclass (tui_monitor, tui_explorer, tui_query) and
PardaApp combines them. They must be App subclasses rather than plain mixins: Textual registers
@on handlers per class through App's metaclass, so a plain mixin's handlers would never fire.
"""

from __future__ import annotations

from typing import Any

from textual.app import App

from . import server
from .tui_widgets import Result


class PardaBase(App):
    """Runtime handle and per-tab state shared by every tab of the TUI."""

    def __init__(self, rt: server.Runtime, url: str = "http://127.0.0.1:8765") -> None:
        super().__init__()
        self.rt, self.url = rt, url
        self.role = "support"
        self.ex_backend: str | None = None
        self.ex_table: str | None = None
        self.ex_offset = 0
        self.ex_filter: tuple[str | None, str | None] = (None, None)
        self.ex_res: Result | None = None
        self.ex_meta: dict[str, dict[str, Any]] = {}
        self.cat_backend: str | None = None
        self.q_res: list[Result | None] = [None, None, None]
        # Signatures of what is on screen, so the 1.5 s tick only redraws what changed.
        self._seq = 0
        self._events: list[dict[str, Any]] = []
        self._db_sig: tuple[Any, ...] | None = None
        self._eng_sig: tuple[tuple[str, ...], str | None] | None = None
        self._scan_sig: tuple[Any, ...] | None = None
        self._mon_sig: tuple[Any, ...] | None = None
        self._opt_names: list[str] | None = None
        # Result-column index of each displayed grid column, per grid id (columns can be hidden).
        self.colmap: dict[str, list[int]] = {}
        self._last_preset: tuple[str | None] | None = None
        self._last_preset_t = 0.0

    def ready(self) -> list[str]:
        """Backends that finished their first scan, in configured order."""
        return [n for n in self.rt.order if n in self.rt.engines]

    def user_for(self, role: str) -> str:
        """First dashboard user with this role, or a synthetic TUI user name."""
        return next((u["username"] for u in self.rt.users if u["role"] == role), f"tui:{role}")
