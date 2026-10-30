"""Measured benchmark numbers for the dashboard's Benchmarks tab (GET /api/benchmarks).

Nothing here is computed from hand-typed constants: every value is read from a report file written
by a measurement script (ml/benchmark.py, ml/evaluate.py, ml/eval_saffron.py, ml/eval_pipeline.py,
the soak test). Each section names the file it came from; a missing or unreadable file gives a
section with status "missing" or "invalid" and no data, so the page can say "not measured yet".
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

from . import config

log = logging.getLogger(__name__)

REPORTS = config.ML_DIR / "reports"
SOAK = config.REPO / "benchmarks" / "soak.json"
LABELS = ("person", "address", "organization")

Status = Literal["ok", "missing", "invalid"]


class Section(TypedDict):
    """One block of the page: where its numbers came from and the numbers themselves."""

    source: str
    status: Status
    detail: str
    data: dict[str, Any] | None


class Benchmarks(TypedDict):
    """Everything GET /api/benchmarks returns."""

    generated_at: str
    compression: Section
    base_model: Section
    fine_tuned: Section
    saffron: Section
    pipeline: Section
    soak: Section


def display_path(path: Path) -> str:
    """`path` relative to the repository root when it lives inside it, else as given."""
    try:
        return str(path.resolve().relative_to(config.REPO))
    except ValueError:
        return str(path)


def load_json(path: Path) -> tuple[Status, str, Any]:
    """(status, detail, parsed JSON) for a report file; never raises for a bad file."""
    if not path.is_file():
        return "missing", "not measured yet", None
    try:
        return "ok", "", json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        log.warning("benchmark report %s unreadable: %s", path, e)
        return "invalid", f"unreadable: {e}", None


def section(path: Path, extract: Callable[[Any], dict[str, Any]]) -> Section:
    """Load `path` and pass it through `extract`; a file of the wrong shape becomes "invalid"."""
    status, detail, raw = load_json(path)
    data = None
    if status == "ok":
        try:
            data = extract(raw)
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration) as e:
            log.warning("benchmark report %s has an unexpected shape: %r", path, e)
            status, detail = "invalid", f"unexpected format: {e!r}"
    return {"source": display_path(path), "status": status, "detail": detail, "data": data}


# ------------------------------------------------------------------ extractors (pure)
def variant_row(v: dict[str, Any], chosen: str) -> dict[str, Any]:
    """One compressed model variant, flagged as chosen and/or failed."""
    keys = (
        "variant",
        "format",
        "weights_mb",
        "dir_mb",
        "bank_f1",
        "golden_f1",
        "saffron_name_recall",
        "narration_recall",
        "cpu_ms_per_text",
        "peak_rss_mb",
        "notes",
    )
    row = {k: v.get(k) for k in keys}
    row.update(
        verified=bool(v.get("verified")),
        chosen=v["variant"] == chosen,
        failed=str(v.get("notes") or "").upper().startswith("FAILED"),
    )
    return row


def headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Size and bank F1 of the float32 reference against the chosen variant."""
    ref = next(r for r in rows if r["format"] == "fp32")
    best = next(r for r in rows if r["chosen"])
    return {
        "from_variant": ref["variant"],
        "to_variant": best["variant"],
        "from_mb": ref["weights_mb"],
        "to_mb": best["weights_mb"],
        "times_smaller": round(ref["weights_mb"] / best["weights_mb"], 1),
        "from_f1": ref["bank_f1"],
        "to_f1": best["bank_f1"],
    }


def compression(raw: dict[str, Any]) -> dict[str, Any]:
    """Variants table, headline and measurement conditions from ml/reports/compression.json."""
    rows = [variant_row(v, raw["chosen"]) for v in raw["variants"]]
    return {
        "chosen": raw["chosen"],
        "conditions": raw.get("description", ""),
        "device": raw["env"]["device"],
        "threads": raw["env"]["threads"],
        "threshold": raw["variants"][0].get("threshold"),
        "rationale": raw.get("rationale", ""),
        "headline": headline(rows),
        "variants": rows,
    }


def per_label(block: dict[str, Any]) -> dict[str, float]:
    """F1 per entity label that has test cases (support > 0)."""
    return {lab: block[lab]["f1"] for lab in LABELS if block.get(lab, {}).get("support", 0) > 0}


def support(block: dict[str, Any]) -> int:
    """Number of labelled entities in a test set."""
    return sum(block[lab]["support"] for lab in LABELS if lab in block)


def model_eval(raw: dict[str, Any]) -> dict[str, Any]:
    """Bank and hand-written (golden) F1, overall and per label, from evaluate.py's report."""
    bank, golden = raw["bank_test"], raw["golden"]
    return {
        "model": raw["model"],
        "device": raw["device"],
        "threshold": raw["threshold"],
        "bank_f1": bank["micro"]["f1"],
        "golden_f1": golden["micro"]["f1"],
        "bank_labels": per_label(bank),
        "golden_labels": per_label(golden),
        "bank_n": support(bank),
        "golden_n": support(golden),
    }


def saffron(raw: dict[str, Any]) -> dict[str, Any]:
    """Name and narration recall per detector on the organisers' data (eval_saffron.py)."""
    rows = {k: v for k, v in raw.items() if isinstance(v, dict)}
    return {
        "name_cases": raw["customer_name_cases"],
        "narration_cases": raw["narration_cases"],
        "rules_only": rows["rules_only"],
        "base": rows["base"],
        "fine_tuned": rows["fine_tuned"],
    }


def pipeline(raw: dict[str, Any]) -> dict[str, Any]:
    """Rules vs model vs full product recall, and model additions (eval_pipeline.py)."""
    names = raw["customer_names"]
    return {
        "date": raw["date"],
        "model_dir": raw["model_dir"],
        "device": raw["device"],
        "threads": raw["threads"],
        "threshold": raw["threshold"],
        "n": names["n"],
        "model_only": names["model_only"],
        "with_bank_name_list": names["with_bank_name_list"],
        "without_name_list": names["without_name_list"],
        "free_text": raw["free_text"],
    }


def soak(raw: dict[str, Any]) -> dict[str, Any]:
    """The soak test report as written by the soak run (shape owned by that script)."""
    if not isinstance(raw, dict):
        raise TypeError("soak report must be a JSON object")
    return raw


def collect(reports: Path = REPORTS, soak_path: Path = SOAK) -> Benchmarks:
    """Assemble every benchmark section from its report file."""
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compression": section(reports / "compression.json", compression),
        "base_model": section(reports / "base.json", model_eval),
        "fine_tuned": section(reports / "ft.json", model_eval),
        "saffron": section(reports / "saffron_full.json", saffron),
        "pipeline": section(reports / "pipeline.json", pipeline),
        "soak": section(soak_path, soak),
    }
