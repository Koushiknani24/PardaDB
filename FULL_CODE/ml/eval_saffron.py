"""Full evaluation on the organisers' Saffron Bank data: every ticket/remark that contains the
customer's own name, and every narration that contains its non-UPI counterparty (3,000 sampled).
Compares PardaDB rules alone (context patterns, no name dictionary) with the base and fine-tuned models."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # FULL_CODE/, so `import pardadb` works
import evaluate as ev
import torch
from gliner import GLiNER

from pardadb import detectors

(nt, ng), (rt, rg) = ev.saffron_set(limit=10**9)
out = {"customer_name_cases": len(nt), "narration_cases": len(rt)}


def rules(texts):
    return [[(s, e, "person") for s, e, k in detectors.find_spans(t) if k == "person"] for t in texts]


out["rules_only"] = {"name_recall": ev.recall(ng, rules(nt)), "narration_recall": ev.recall(rg, rules(rt), True)}
for tag, path in [("base", "knowledgator/gliner-pii-edge-v1.0"), ("fine_tuned", "models/pardadb-pii")]:
    m = GLiNER.from_pretrained(path).to("cuda").eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        pn = ev.predict(m, nt, ev.LABELS, 0.4)
        pr = ev.predict(m, rt, ev.LABELS, 0.4)
        el = time.perf_counter() - t0
    out[tag] = {
        "name_recall": ev.recall(ng, pn),
        "narration_recall": ev.recall(rg, pr, True),
        "ms_per_text": round(el * 1000 / (len(nt) + len(rt)), 2),
    }
    both = [a + b for a, b in zip(pn, rules(nt), strict=False)]
    out[tag + "_plus_rules"] = {"name_recall": ev.recall(ng, both)}
Path("reports/saffron_full.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
