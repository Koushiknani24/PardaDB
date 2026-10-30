"""Evaluate the full PardaDB masking pipeline on the organisers' Saffron Bank data (CPU, in-process).

Three detectors are compared on the same 918 customer-name cases that ``eval_saffron.py`` uses
(every ticket or loan remark that contains the customer's own name):

* rules only:  ``detectors.find_spans`` with the bank's NameIndex, built like ``server.Runtime.install``
* model only:  raw model spans labelled "person" at the runtime threshold ``config.ML_THRESHOLD``
* product:     rules + model spans passed through ``ml.refine`` and merged by ``find_spans(extra=...)``,
               exactly as ``ml.MLClient.spans_for`` and ``engine.Engine`` combine them at query time

It also counts, over every unique ticket / notes / remarks text, how many spans the product adds
on top of rules and how many raw model spans the precision filter drops. Run once::

    .venv/bin/python eval_pipeline.py        # writes reports/pipeline.json
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import compress  # noqa: E402
import evaluate as ev  # noqa: E402

from pardadb import backends, config, detectors, ml, scanner  # noqa: E402

log = logging.getLogger("eval_pipeline")

THREADS = 4
BATCH = 16
OUT = ROOT / "reports" / "pipeline.json"
FREE_TEXT_SQL = (
    "SELECT ticket_text FROM support_tickets UNION SELECT notes FROM support_tickets "
    "UNION SELECT remarks FROM loan_applications"
)
Span = tuple[int, int, str]
RawSpan = tuple[int, int, str, float]


def bank_name_index() -> detectors.NameIndex:
    """Scan the read-only SQLite bank and index every value of a person column, like the server does."""
    backend = backends.connect("sqlite")
    catalog = scanner.scan(backend)
    names: list[Any] = []
    for col in catalog["columns"]:
        if col["entity"] == "person":
            q = backend.sql("SELECT DISTINCT {c} FROM {t}", c=col["column"], t=col["table"])
            names += [r[0] for r in backend.execute(q)[1] if r[0]]
    return detectors.NameIndex(names)


def free_texts() -> list[str]:
    """Every unique non-empty ticket text, ticket note and loan remark."""
    con = sqlite3.connect(f"file:{config.SQLITE_PATH}?mode=ro", uri=True)
    try:
        return sorted({r[0] for r in con.execute(FREE_TEXT_SQL) if r[0]})
    finally:
        con.close()


def model_spans(model: Any, texts: list[str]) -> dict[str, list[RawSpan]]:
    """Raw ``(start, end, label, score)`` model spans per text at the runtime threshold."""
    out: dict[str, list[RawSpan]] = {}
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        res = model.inference(chunk, ev.LABELS, threshold=config.ML_THRESHOLD, batch_size=BATCH)
        for text, ents in zip(chunk, res, strict=True):
            out[text] = [(e["start"], e["end"], e["label"], float(e["score"])) for e in ents]
    return out


def refined(text: str, raw: Iterable[RawSpan]) -> list[Span]:
    """Model spans that survive the precision filter, labelled as ``MLClient.spans_for`` does."""
    out = []
    for s, e, label, score in raw:
        if label in ml.LABELS and (r := ml.refine(text, s, e, label, score)):
            out.append((r[0], r[1], ml.LABELS[label]))
    return out


def is_person(entity: str) -> bool:
    """True for rule ("person") and model ("person (AI)") name spans."""
    return entity.split(" ")[0] == "person"


def name_recall(golds: list[list[Span]], preds: list[list[Span]]) -> dict[str, float]:
    """Share of known names covered by a person span, and by any masking span at all."""
    as_person = [[(s, e, "person") for s, e, ent in p if is_person(ent)] for p in preds]
    as_any = [[(s, e, "person") for s, e, _ in p] for p in preds]
    return {"name_recall": ev.recall(golds, as_person), "masked_any_label": ev.recall(golds, as_any)}


def recall_section(model: Any, names: detectors.NameIndex) -> dict[str, Any]:
    """Rules vs model vs product name recall on the 918 customer-name cases.

    Measured twice: with the bank's own name list (what the running product does) and without it. The
    list contains the very names being searched for, so the second run is the fair test of names the bank
    does not already hold (third parties, nominees, typos).
    """
    (texts, golds), _ = ev.saffron_set(limit=10**9)
    raw = model_spans(model, texts)
    model_only = [[(s, e, lab) for s, e, lab, _ in raw[t] if lab == "person"] for t in texts]

    def rules_and_product(index: detectors.NameIndex | None) -> dict[str, dict[str, float]]:
        rules = [detectors.find_spans(t, index) for t in texts]
        product = [detectors.find_spans(t, index, refined(t, raw[t])) for t in texts]
        return {"rules_only": name_recall(golds, rules), "product": name_recall(golds, product)}

    return {
        "n": len(texts),
        "model_only": name_recall(golds, model_only),
        "with_bank_name_list": rules_and_product(names),
        "without_name_list": rules_and_product(None),
    }


def additions_section(model: Any, names: detectors.NameIndex) -> dict[str, Any]:
    """What the model adds over rules, and what the precision filter drops, on every free text."""
    texts = free_texts()
    t0 = time.perf_counter()
    raw = model_spans(model, texts)
    ms_per_text = (time.perf_counter() - t0) * 1000 / len(texts)
    kept = {t: refined(t, raw[t]) for t in texts}
    candidates = sum(1 for t in texts for s in raw[t] if s[2] in ml.LABELS)
    added = sum(1 for t in texts for s in detectors.find_spans(t, names, kept[t]) if s[2].endswith("(AI)"))
    return {
        "n_texts": len(texts),
        "rule_spans": sum(len(detectors.find_spans(t, names)) for t in texts),
        "model_candidate_spans": candidates,
        "kept_by_precision_filter": sum(len(v) for v in kept.values()),
        "dropped_by_precision_filter": candidates - sum(len(v) for v in kept.values()),
        "extra_spans_over_rules": added,
        "texts_with_extra_spans": sum(
            any(s[2].endswith("(AI)") for s in detectors.find_spans(t, names, kept[t])) for t in texts
        ),
        "model_cpu_ms_per_text": round(ms_per_text, 2),
    }


def main() -> None:
    """Run both measurements and write ``reports/pipeline.json``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    names = bank_name_index()
    log.info("name index: %d full names", len(names.names))
    model = compress.load_model(compress.CHOSEN_DIR, device="cpu", threads=THREADS)
    report = {
        "description": "Full PardaDB pipeline on the organisers' Saffron Bank SQLite data (ml/eval_pipeline.py).",
        "date": time.strftime("%Y-%m-%d"),
        "model_dir": str(compress.CHOSEN_DIR.relative_to(ROOT)),
        "device": "cpu",
        "threads": THREADS,
        "threshold": config.ML_THRESHOLD,
        "name_index_size": len(names.names),
        "customer_names": recall_section(model, names),
        "free_text": additions_section(model, names),
    }
    OUT.write_text(json.dumps(report, indent=2))
    log.info("wrote %s\n%s", OUT, json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
