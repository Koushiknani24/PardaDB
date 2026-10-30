"""CPU benchmark of one model variant: accuracy, speed, memory and size, as a JSON report.

Uses the test sets, threshold and scoring of ``evaluate.py`` so numbers compare with ``reports/ft.json``
(which was measured on GPU). Run one benchmark at a time: the RSS figure is for the whole process.

    .venv/bin/python benchmark.py --variant fp32 --model-dir models/pardadb-pii --out reports/compression/fp32.json
    .venv/bin/python benchmark.py --variant fp16 --model-dir models/pardadb-pii-fp16 \\
        --out reports/compression/fp16.json --threads 4

Sizes use 1 MB = 1e6 bytes. Peak RSS is ``ru_maxrss`` of this process (Python, torch, model, inference).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import compress
import evaluate as ev
import torch

log = logging.getLogger("benchmark")

MB = 1e6
BATCH = 16  # evaluate.predict default batch size
WARMUP_BATCHES = 2


def default_threshold() -> float:
    """Threshold of the fp32 reference report (``reports/ft.json``), else evaluate.py's default 0.4."""
    ref = ev.ROOT / "reports" / "ft.json"
    if ref.is_file():
        return float(json.loads(ref.read_text(encoding="utf-8"))["threshold"])
    return 0.4


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far (Linux reports ru_maxrss in KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / MB, 1)


def sizes(model_dir: Path) -> dict[str, Any]:
    """Weights-file size (files listed in compression.json) and whole-directory size, in MB."""
    weights = compress.read_manifest(model_dir).get("weights", [compress.WEIGHTS])
    files = [p for p in model_dir.rglob("*") if p.is_file()]
    return {
        "weights_files": weights,
        "weights_mb": round(sum((model_dir / w).stat().st_size for w in weights) / MB, 2),
        "dir_mb": round(compress.dir_size_mb(model_dir), 2),
        "dir_files": sorted(str(p.relative_to(model_dir)) for p in files),
        "unit": "1 MB = 1e6 bytes",
    }


def cpu_latency(model: Any, texts: list[str], threshold: float) -> dict[str, float]:
    """Per-text latency over ``texts`` in batches of ``BATCH`` (after warm-up); median and mean ms/text."""
    batches = [texts[i : i + BATCH] for i in range(0, len(texts), BATCH)]
    for chunk in batches[:WARMUP_BATCHES]:
        ev.predict(model, chunk, ev.LABELS, threshold, batch=BATCH)
    per_text = []
    t_all = time.perf_counter()
    for chunk in batches:
        t0 = time.perf_counter()
        ev.predict(model, chunk, ev.LABELS, threshold, batch=BATCH)
        per_text.append((time.perf_counter() - t0) * 1000 / len(chunk))
    total = time.perf_counter() - t_all
    return {
        "median_ms_per_text": round(statistics.median(per_text), 2),
        "mean_ms_per_text": round(total * 1000 / len(texts), 2),
        "p90_ms_per_text": round(statistics.quantiles(per_text, n=10)[-1], 2),
        "texts": len(texts),
        "batch_size": BATCH,
    }


def run(variant: str, model_dir: Path, threads: int, threshold: float) -> dict[str, Any]:
    """Load ``model_dir`` through compress.load_model on CPU and measure everything for the report."""
    torch.set_num_threads(threads)
    rss_before = compress.current_rss_mb()
    t0 = time.perf_counter()
    model = compress.load_model(model_dir, device="cpu", threads=threads)
    load_s = time.perf_counter() - t0
    report: dict[str, Any] = {
        "variant": variant,
        "model_dir": str(model_dir),
        "manifest": compress.read_manifest(model_dir),
        "device": "cpu",
        "threads": threads,
        "threshold": threshold,
        "labels": ev.LABELS,
        "size": sizes(model_dir),
        "load_seconds": round(load_s, 2),
        "rss_mb_before_load": rss_before,
        "rss_mb_after_load": compress.current_rss_mb(),
    }
    with torch.inference_mode():
        texts, golds = ev.bank_set()
        report["bank_test"] = ev.prf(golds, ev.predict(model, texts, ev.LABELS, threshold))
        log.info("bank micro F1 %.3f", report["bank_test"]["micro"]["f1"])
        gt, gg = ev.golden_set()
        report["golden"] = ev.prf(gg, ev.predict(model, gt, ev.LABELS, threshold))
        log.info("golden micro F1 %.3f", report["golden"]["micro"]["f1"])
        (nt, ng), (rt, rg) = ev.saffron_set()
        report["saffron_customer_name_recall"] = {
            "recall": ev.recall(ng, ev.predict(model, nt, ev.LABELS, threshold)),
            "n": len(nt),
        }
        report["saffron_narration_counterparty_recall"] = {
            "recall": ev.recall(rg, ev.predict(model, rt, ev.LABELS, threshold), any_label=True),
            "n": len(rt),
        }
        report["cpu_latency_bank_test"] = cpu_latency(model, texts, threshold)
    report["peak_rss_mb"] = peak_rss_mb()
    report["env"] = {
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    return report


def summary(r: dict[str, Any]) -> str:
    """One-line human summary of a report."""
    return (
        f"{r['variant']}: weights {r['size']['weights_mb']} MB, dir {r['size']['dir_mb']} MB, "
        f"bank F1 {r['bank_test']['micro']['f1']}, golden F1 {r['golden']['micro']['f1']}, "
        f"names {r['saffron_customer_name_recall']['recall']}, "
        f"narration {r['saffron_narration_counterparty_recall']['recall']}, "
        f"{r['cpu_latency_bank_test']['median_ms_per_text']} ms/text, peak RSS {r['peak_rss_mb']} MB"
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--variant", required=True, help="name recorded in the report, e.g. fp32, fp16, int8")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=None, help="default: threshold in reports/ft.json")
    args = ap.parse_args()

    threshold = default_threshold() if args.threshold is None else args.threshold
    report = run(args.variant, args.model_dir, args.threads, threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("%s", summary(report))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
