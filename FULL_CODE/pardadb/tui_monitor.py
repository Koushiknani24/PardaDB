"""Monitor tab and the top status bar of the TUI, plus active-backend selection and the AI model toggle."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on, work
from textual.widgets import Button, DataTable, ProgressBar, Select, Static

from .tui_base import PardaBase
from .tui_widgets import LEVEL_STYLE, STATUS_STYLE, dur

Status = dict[str, Any]  # Runtime.status(): JSON-like snapshot of backends, scan, process and stats


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
        """Poll the runtime and redraw whatever changed; reload views once a scan finishes."""
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

    def update_topbar(self, st: Status) -> None:
        act = self.rt.dbs.get(st["active_backend"] or "", {}).get("label") or "connecting…"
        self.query_one("#topbar", Static).update(topbar_text(st, act))

    def update_ml(self, st: Status) -> None:
        """Show the AI model state and set the toggle button's label to match."""
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
    def on_ml_toggle(self) -> None:
        self.action_toggle_ml()

    def action_toggle_ml(self) -> None:
        if (self.rt.status().get("ml") or {}).get("status") == "starting":
            return
        self._ml_worker()

    @work(thread=True, exclusive=True, group="ml")
    def _ml_worker(self) -> None:
        from . import ml

        if ml.client.info.get("status") == "ready" and ml.client.enabled:
            self.rt.ml_stop()
        else:
            self.rt.ml_start()
        self.call_from_thread(self.tick)

    def update_monitor(self, st: Status) -> None:
        """Refresh the Monitor tab: database table, scan progress, resources and live events."""
        self.update_ml(st)
        self._update_db_table(st)
        self._update_scan(st)
        self._update_events(st)

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

    def _update_events(self, st: Status) -> None:
        new = [e for e in st["events"] if e["seq"] > self._seq]
        if not new:
            return
        self._seq = new[-1]["seq"]
        self._events = (self._events + new)[-60:]
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
        """Reload the Explorer and Catalog when they show a backend other than the active one."""
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
