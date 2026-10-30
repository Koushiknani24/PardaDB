"""Monitor tab and the top status bar of the TUI, plus active-backend selection and the AI model toggle."""

from __future__ import annotations

import logging
from typing import Any

from rich.text import Text
from textual import on, work
from textual.widgets import Button, DataTable, ProgressBar, Select, Static

from .tui_base import PardaBase, Status
from .tui_widgets import LEVEL_STYLE, STATUS_STYLE, dur, one_line, same_cell

log = logging.getLogger(__name__)

TICK_S = 1.5  # status poll interval while something is happening
IDLE_AFTER = 20  # polls (30 s) with no activity before slowing down
SLOW_EVERY = 4  # when idle, poll on every 4th tick only (6 s)


def topbar_text(st: Status, active_label: str) -> Text:
    """One-line summary: active backend, connected DBs, CPU/RAM, last query, counters, AI and scan."""
    p, bs, s = st["process"], st["backends"], st["stats"]
    conn = sum(b["status"] == "connected" for b in bs)
    t = Text()
    t.append(" PardaDB ", "bold white on #d9730d")
    t.append(f" v{st['version']} ", "dim")
    sep = (" │ ", "#5b6b8a")
    t.append_text(
        Text.assemble(
            sep,
            ("backend ", "dim"),
            (active_label, "bold cyan"),
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
    return t


def waiting_text(st: Status) -> Text:
    """Placeholder for views that need a scanned database: scan progress, or the reconnect notice."""
    sc, bs = st["scan"], st["backends"]
    if sc["state"] == "scanning":
        pct = 100 * sc["columns_done"] // max(sc["columns_total"], 1)
        return Text(f"Scanning… {pct}% of the columns classified; tables appear when the PII scan finishes.", "yellow")
    if any(b["status"] == "connected" for b in bs):
        return Text("Connected; the PII scan is about to start…", "yellow")
    why = next((one_line(b["detail"], 90) for b in bs if b["detail"]), "")
    return Text(f"Database not connected - retrying every 10 s.  {why}".rstrip(), "yellow")


def activity_sig(st: Status) -> tuple[Any, ...]:
    """What counts as activity for the adaptive poll rate (CPU and uptime always move, so they are left out)."""
    sc, s = st["scan"], st["stats"]
    ev = st["events"][-1]["seq"] if st["events"] else 0
    return (
        sc["state"],
        sc["columns_done"],
        s["queries"],
        ev,
        st["active_backend"],
        (st.get("ml") or {}).get("status"),
        tuple(b["status"] for b in st["backends"]),
    )


def db_row(b: dict[str, Any], active: bool, ready: bool) -> list[Text]:
    """Cells of one Monitor database row: active marker, label, status, tables, rows, latency, detail."""
    return [
        Text("▶" if active else "", "bold cyan"),
        Text(b["label"], "bold" if active else ""),
        Text(
            ("● " + b["status"]) + ("" if ready or b["status"] != "connected" else " (scanning)"),
            STATUS_STYLE.get(b["status"], "white"),
        ),
        Text(str(b["tables"]), justify="right"),
        Text(f"{b['rows']:,}", justify="right"),
        Text(f"{b['latency_ms']} ms" if b["latency_ms"] is not None else "-", justify="right"),
        Text(one_line(b["detail"] or "", 60), "dim"),
    ]


def resources_text(st: Status) -> Text:
    """Process resources, query counters and the last query, for the Monitor tab."""
    p, s = st["process"], st["stats"]
    lq = s.get("last_query") or {}
    return Text.assemble(
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


class MonitorTab(PardaBase):
    """Periodic status refresh, the Monitor tab, backend selection and the AI model switch."""

    # ------------------------------------------------------------------ periodic status
    def tick(self) -> None:
        """Timer callback: poll the runtime every tick, or every SLOW_EVERY-th tick once idle."""
        self._tick_n += 1
        if self._quiet_ticks >= IDLE_AFTER and self._tick_n % SLOW_EVERY:
            return
        self.poll_status()

    def wake(self) -> None:
        """Back to the fast poll rate and poll now (after a user action that changes the runtime)."""
        self._quiet_ticks = 0
        self.poll_status()

    def poll_status(self) -> None:
        """Start a status poll in a worker unless one is still running (a slow poll is never queued twice)."""
        if not self._polling:
            self._polling = True
            self._poll_worker()

    @work(thread=True, group="status")
    def _poll_worker(self) -> None:
        """Take the status snapshot off the UI thread (it waits on the runtime lock during scans)."""
        try:
            st = self.rt.status()
        except (OSError, RuntimeError, KeyError, ValueError) as e:
            log.warning("TUI status poll failed: %s", e)
            return
        finally:
            self._polling = False
        self.call_from_thread(self.apply_status, st)

    def apply_status(self, st: Status) -> None:
        """Redraw from a snapshot: top bar always, the Monitor tab only while visible; reload views after a scan."""
        self.st = st
        sig = activity_sig(st)
        self._quiet_ticks = self._quiet_ticks + 1 if sig == self._act_sig else 0
        self._act_sig = sig
        self.update_topbar(st)
        self.sync_backends()
        self._collect_events(st)
        if self.active_tab() == "monitor":
            self.update_monitor(st)
        if not self.ready():
            self.show_waiting(st)
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

    def update_topbar(self, st: Status) -> None:
        act = self.rt.dbs.get(st["active_backend"] or "", {}).get("label") or "connecting…"
        t = topbar_text(st, act)
        if t.plain != self._top_plain:
            self._top_plain = t.plain
            self.query_one("#topbar", Static).update(t)

    def show_waiting(self, st: Status) -> None:
        """Friendly placeholder in the Explorer and Catalog until the first scan finishes."""
        t = waiting_text(st)
        if t.plain != self._wait_plain:
            self._wait_plain = t.plain
            self.query_one("#ex-msg", Static).update(t)
            self.query_one("#cat-summary", Static).update(t)

    def update_ml(self, st: Status) -> None:
        """Show the AI model state and set the toggle button's label to match."""
        i = st.get("ml") or {}
        stt = i.get("status", "off")
        sig = (stt, i.get("texts"), i.get("cache_hits"), i.get("device"))
        if sig == self._ml_sig:
            return
        self._ml_sig = sig
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
    def on_ml_toggle(self) -> None:
        self.action_toggle_ml()

    def action_toggle_ml(self) -> None:
        """Start or stop the AI model in a worker; ignored while it is still starting."""
        if ((self.st or {}).get("ml") or {}).get("status") == "starting":
            return
        self._ml_worker()

    @work(thread=True, exclusive=True, group="ml")
    def _ml_worker(self) -> None:
        from . import ml

        if ml.client.info.get("status") == "ready" and ml.client.enabled:
            self.rt.ml_stop()
        else:
            self.rt.ml_start()
        self.call_from_thread(self.wake)

    def update_monitor(self, st: Status) -> None:
        """Refresh the Monitor tab: database table, scan progress, resources and live events."""
        self.update_ml(st)
        self._update_db_table(st)
        self._update_scan(st)
        self._render_events()

    def _update_db_table(self, st: Status) -> None:
        bs = st["backends"]
        sig = tuple((b["name"], b["status"], b["tables"], b["rows"], b["latency_ms"], b["detail"]) for b in bs) + (
            st["active_backend"],
            tuple(self.rt.engines),
        )
        if sig == self._db_sig:
            return
        self._db_sig = sig
        t = self.query_one("#mon-dbs", DataTable)
        rows = {b["name"]: db_row(b, b["name"] == st["active_backend"], b["name"] in self.rt.engines) for b in bs}
        if [k.value for k in t.rows] != list(rows):
            t.clear()
            for name, cells in rows.items():
                t.add_row(*cells, key=name)
            return
        for name, cells in rows.items():  # same databases: patch only the cells that changed
            for i, (old, new) in enumerate(zip(t.get_row(name), cells, strict=True)):
                if not same_cell(old, new):
                    t.update_cell(name, f"c{i}", new)

    def _update_scan(self, st: Status) -> None:
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
        if msig == self._mon_sig:
            return
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
        self.query_one("#mon-res", Static).update(resources_text(st))

    def _collect_events(self, st: Status) -> None:
        """Keep the newest 60 events (cheap, every poll); they are drawn only while Monitor is visible."""
        new = [e for e in st["events"] if e["seq"] > self._seq]
        if not new:
            return
        self._seq = new[-1]["seq"]
        self._events = (self._events + new)[-60:]
        self._events_dirty = True

    def _render_events(self) -> None:
        if not self._events_dirty:
            return
        self._events_dirty = False
        out = Text()
        for e in reversed(self._events):
            out.append(f"{e['ts']} ", "dim")
            out.append(str(e["msg"]) + "\n", LEVEL_STYLE.get(e["level"], "white"))
        self.query_one("#mon-events", Static).update(out)

    # ------------------------------------------------------------------ backend selection
    def sync_backends(self) -> None:
        """Keep every backend picker in step with the scanned backends and the active one."""
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
    def _backend_changed(self, ev: Select.Changed) -> None:
        v = ev.value
        if v == Select.NULL or v not in self.rt.engines:
            return
        if v != self.rt.active:
            self.rt.active = v
            self._eng_sig = None
            self.sync_backends()
        self.refresh_views()

    def refresh_views(self) -> None:
        """Reload the Explorer and Catalog when they show a backend other than the active one.

        This runs once per backend change (the periodic poll never touches them), so both are
        preloaded in the background and open instantly.
        """
        n = self.rt.active
        if n not in self.rt.engines:
            return
        if self.ex_backend != n:
            self.ex_backend = n
            self.load_tables(n)
        if self.cat_backend != n:
            self.load_catalog()

    @on(DataTable.RowSelected, "#mon-dbs")
    def _db_clicked(self, ev: DataTable.RowSelected) -> None:
        n = ev.row_key.value
        if n in self.rt.engines and n != self.rt.active:
            self.rt.active = n
            self._eng_sig = None
            self.sync_backends()
            self.refresh_views()
            self.notify(f"Active backend: {self.rt.dbs[n]['label']}")
        elif n not in self.rt.engines:
            self.notify(f"{self.rt.dbs[n]['label']} is {self.rt.dbs[n]['status']}; not scanned yet", severity="warning")
