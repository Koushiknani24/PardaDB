"""Evaluate a GLiNER model for PardaDB: accuracy and speed.

Test sets (none used in training):
  golden   our hand-written test set (ml/golden): names, addresses, negatives
  bank     held-out split of our generator (unseen names and merchant words)
  saffron  real Saffron Bank text whose correct answer is known from the database:
           a customer's own name inside their ticket or loan remark, and the
           counterparty name inside a transaction narration (recall only)

    .venv/bin/python evaluate.py --model knowledgator/gliner-pii-edge-v1.0 --tag base
    .venv/bin/python evaluate.py --model models/pardadb-pii --tag ft
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from gliner import GLiNER  # noqa: E402

LABELS = ["person", "address", "organization"]
PARDA_LABELS = [
    "person",
    "email",
    "phone number",
    "address",
    "organization",
    "date of birth",
    "aadhaar number",
    "pan number",
    "upi id",
    "bank account number",
    "ifsc code",
    "passport number",
    "credit card number",
    "ip address",
    "username",
    "password",
]
KIND = {"Person": "person", "Address": "address", "Organization": "organization"}
GOLDEN = ROOT / "golden"
DB = ROOT.parent / "data" / "sql" / "sqlite" / "saffron_bank.db"


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def overlaps(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def predict(
    model: GLiNER, texts: list[str], labels: list[str], threshold: float, batch: int = 16
) -> list[list[tuple[int, int, str]]]:
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        if hasattr(model, "inference"):
            res = model.inference(chunk, labels, threshold=threshold, batch_size=batch)
        else:
            res = [model.predict_entities(t, labels, threshold=threshold) for t in chunk]
        out += [[(e["start"], e["end"], e["label"]) for e in r if e["label"] in LABELS] for r in res]
    return out


def prf(golds: list[list[tuple[int, int, str]]], preds: list[list[tuple[int, int, str]]]) -> dict:
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    leak_free = 0
    for gold, pred in zip(golds, preds, strict=False):
        missed = 0
        for g in gold:
            if any(overlaps(g, p) and p[2] == g[2] for p in pred):
                tp[g[2]] += 1
            else:
                fn[g[2]] += 1
                missed += 1
        for p in pred:
            if not any(overlaps(g, p) and p[2] == g[2] for g in gold):
                fp[p[2]] += 1
        leak_free += missed == 0
    res = {}
    for lab in LABELS:
        t, f_p, f_n = tp[lab], fp[lab], fn[lab]
        if t + f_p + f_n == 0:
            continue
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        res[lab] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "support": t + f_n,
        }
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P = T / (T + FP) if T + FP else 0.0
    R = T / (T + FN) if T + FN else 0.0
    res["micro"] = {
        "precision": round(P, 3),
        "recall": round(R, 3),
        "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0,
    }
    res["zero_leak_cases"] = round(leak_free / len(golds), 3)
    return res


def golden_set() -> tuple[list[str], list[list[tuple[int, int, str]]]]:
    texts, golds = [], []
    for name in ("names.jsonl", "negatives.jsonl"):
        for r in read_jsonl(GOLDEN / name):
            texts.append(r["text"])
            golds.append([(e["start"], e["end"], KIND[e["kind"]]) for e in r.get("entities", []) if e["kind"] in KIND])
    return texts, golds


def bank_set() -> tuple[list[str], list[list[tuple[int, int, str]]]]:
    rows = read_jsonl(ROOT / "dataset" / "test.chars.jsonl")
    return [r["text"] for r in rows], [[(e["start"], e["end"], e["label"]) for e in r["entities"]] for r in rows]


def saffron_set(
    limit: int = 400,
) -> tuple[
    tuple[list[str], list[list[tuple[int, int, str]]]],
    tuple[list[str], list[list[tuple[int, int, str]]]],
]:
    """Spans whose identity is known from the database. Recall only: other names
    in the same text (nominees, co-applicants) are not labelled here."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    texts, golds = [], []
    q = (
        "SELECT t.ticket_text, c.cust_nm FROM support_tickets t JOIN customers c USING(customer_id) "
        "UNION ALL SELECT l.remarks, c.cust_nm FROM loan_applications l JOIN customers c USING(customer_id) "
        "WHERE l.remarks IS NOT NULL"
    )
    for text, name in con.execute(q):
        i = text.lower().find(name.lower())
        if i >= 0:
            texts.append(text)
            golds.append([(i, i + len(name), "person")])
    names = texts[:limit], golds[:limit]
    ntexts, ngolds = [], []
    for narr, cp in con.execute(
        "SELECT narration, counterparty FROM transactions WHERE counterparty IS NOT NULL "
        "AND counterparty NOT LIKE '%@%' ORDER BY txn_id LIMIT 3000"
    ):
        i = narr.find(cp)
        if i >= 0:
            ntexts.append(narr)
            ngolds.append([(i, i + len(cp), "any")])
    return names, (ntexts[:limit], ngolds[:limit])


def recall(
    golds: list[list[tuple[int, int, str]]], preds: list[list[tuple[int, int, str]]], any_label: bool = False
) -> float:
    hit = tot = 0
    for gold, pred in zip(golds, preds, strict=False):
        for g in gold:
            tot += 1
            hit += any(overlaps(g, p) and (any_label or p[2] == g[2]) for p in pred)
    return round(hit / tot, 3) if tot else 0.0


def latency(model: GLiNER, texts: list[str], labels: list[str], threshold: float, device: str, n: int = 200) -> float:
    texts = texts[:n]
    predict(model, texts[:16], labels, threshold)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    predict(model, texts, labels, threshold, batch=16)
    if device == "cuda":
        torch.cuda.synchronize()
    return round((time.perf_counter() - t0) * 1000 / len(texts), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = GLiNER.from_pretrained(args.model).to(args.device).eval()
    report = {"model": args.model, "threshold": args.threshold, "device": args.device}
    with torch.no_grad():
        t, g = golden_set()
        report["golden"] = prf(g, predict(model, t, LABELS, args.threshold))
        t, g = bank_set()
        report["bank_test"] = prf(g, predict(model, t, LABELS, args.threshold))
        (nt, ng), (rt, rg) = saffron_set()
        report["saffron_customer_name_recall"] = {
            "recall": recall(ng, predict(model, nt, LABELS, args.threshold)),
            "n": len(nt),
        }
        report["saffron_narration_counterparty_recall"] = {
            "recall": recall(rg, predict(model, rt, LABELS, args.threshold), True),
            "n": len(rt),
        }
        report["ms_per_text_3_labels"] = latency(model, nt, LABELS, args.threshold, args.device)
        report["ms_per_text_16_labels"] = latency(model, nt, PARDA_LABELS, args.threshold, args.device)
    out = ROOT / "reports" / f"{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
