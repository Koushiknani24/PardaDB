"""PII scanner: schema metadata + random value samples -> tag, confidence and evidence per column.

Values decide; the header only adjusts confidence. Scan cost is
columns x sample size, independent of how many rows a table holds.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import config, detectors
from .masks import json_key

if TYPE_CHECKING:
    from .backends import Backend

ENTITY_TAG = {
    "card": "PII_HIGH",
    "aadhaar": "PII_HIGH",
    "pan": "PII_HIGH",
    "ssn": "PII_HIGH",
    "account": "PII_HIGH",
    "person": "PII_MEDIUM",
    "phone": "PII_MEDIUM",
    "email": "PII_MEDIUM",
    "upi": "PII_MEDIUM",
    "address": "PII_MEDIUM",
    "dob": "PII_QUASI",
    "postal": "PII_QUASI",
    "gender": "PII_QUASI",
    "residency": "PII_QUASI",
    "money": "FINANCIAL_CONFIDENTIAL",
    "credit_score": "FINANCIAL_CONFIDENTIAL",
    "text": "PII_FREE_TEXT",
    "json": "SENSITIVE_JSON",
    "customer_id": "INTERNAL_ID",
}
# header token -> (entity, strength)
LEXICON = {
    "nm": ("person", 1),
    "name": ("person", 1),
    "naam": ("person", 1),
    "pita": ("person", 1),
    "father": ("person", 1),
    "nominee": ("person", 0.5),
    "holder": ("person", 0.5),
    "mob": ("phone", 1),
    "mobile": ("phone", 1),
    "phone": ("phone", 1),
    "ph": ("phone", 0.5),
    "tel": ("phone", 1),
    "contact": ("phone", 0.5),
    "email": ("email", 1),
    "emailid": ("email", 1),
    "mail": ("email", 1),
    "addr": ("address", 1),
    "address": ("address", 1),
    "pata": ("address", 1),
    "dob": ("dob", 1),
    "birth": ("dob", 1),
    "janm": ("dob", 1),
    "pincode": ("postal", 1),
    "pin": ("postal", 0.5),
    "zip": ("postal", 1),
    "gender": ("gender", 1),
    "sex": ("gender", 1),
    "residency": ("residency", 1),
    "visa": ("residency", 1),
    "pan": ("pan", 1),
    "aadhaar": ("aadhaar", 1),
    "aadhar": ("aadhaar", 1),
    "uid": ("aadhaar", 1),
    "ssn": ("ssn", 1),
    "card": ("card", 0.5),
    "bal": ("money", 1),
    "balance": ("money", 1),
    "amt": ("money", 1),
    "amount": ("money", 1),
    "inr": ("money", 0.5),
    "limit": ("money", 1),
    "income": ("money", 1),
    "outstanding": ("money", 1),
    "sanctioned": ("money", 1),
    "salary": ("money", 1),
    "score": ("credit_score", 0.5),
    "acct": ("account", 1),
    "account": ("account", 1),
}
NEGATIVE = {
    "rrn",
    "ref",
    "txn",
    "code",
    "ifsc",
    "branch",
    "status",
    "type",
    "ts",
    "kyc",
    "ticket",
    "app",
    "rate",
    "pct",
    "variant",
    "network",
    "mode",
    "channel",
    "dms",
    "maker",
    "checker",
    "verified",
    "by",
}
STRONG_ENTITIES = ("card", "aadhaar", "pan", "ssn", "email", "upi", "phone")


def header_tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def header_hint(name: str) -> tuple[str | None, float, bool, list[str]]:
    """(entity, strength, negative, tokens): what a column header alone suggests."""
    toks = header_tokens(name)
    best, strength = None, 0.0
    for t in toks:
        if t in LEXICON and LEXICON[t][1] > strength:
            best, strength = LEXICON[t]
    negative = any(t in NEGATIVE for t in toks)
    return best, strength, negative, toks


def _rate(counts: Counter[str | None], n: int, *ents: str) -> float:
    return sum(counts[e] for e in ents) / n if n else 0.0


JSON_MAX_DEPTH = 8  # nesting levels walked per sampled document
JSON_MAX_PAIRS = 500  # (key, value) pairs inspected per sampled document


def json_pairs(doc: Any) -> list[tuple[str, Any]]:
    """(key, scalar) pairs of a parsed JSON document at any depth, capped at JSON_MAX_DEPTH levels and
    JSON_MAX_PAIRS pairs. List items inherit their list's key, and an object key is also reported as a
    value of its parent key, so {"pan": {"ABCPK1234F": true}} marks "pan" as holding a PAN."""
    pairs: list[tuple[str, Any]] = []
    stack: list[tuple[Any, str | None, int]] = [(doc, None, 0)]
    while stack and len(pairs) < JSON_MAX_PAIRS:
        node, key, depth = stack.pop()
        if depth > JSON_MAX_DEPTH:
            continue
        if isinstance(node, dict):
            for k, v in node.items():
                if key is not None:
                    pairs.append((key, k))
                stack.append((v, str(k), depth + 1))
        elif isinstance(node, list):
            stack.extend((v, key, depth + 1) for v in node)
        elif key is not None:
            pairs.append((key, node))
    return pairs[:JSON_MAX_PAIRS]


def _is_pii_pair(key: str, value: Any) -> bool:
    """Does `value`, found under JSON key `key`, look like personal data?"""
    hint, _, negative, _ = header_hint(key)
    if negative and hint is None:
        return False  # e.g. dms_ref: a document reference number, not an identity
    kind = detectors.classify_value(value)
    return (
        kind in STRONG_ENTITIES
        or kind == "aadhaar_masked"
        or (hint is not None and kind in ("name", "digits", "aadhaar"))
    )


def json_pii_keys(values: list[Any]) -> Counter[str]:
    """{normalised key: documents where it holds PII} over sampled JSON values, nested objects and lists included."""
    keys: Counter[str] = Counter()
    for v in values:
        try:
            doc = json.loads(v)
        except (TypeError, ValueError, RecursionError):
            continue
        keys.update({json_key(k) for k, val in json_pairs(doc) if _is_pii_pair(k, val)})
    return keys


def classify_column(table: str, column: str, values: list[Any], null_rate: float) -> dict[str, Any]:
    """Decide one column's tag from its sampled values and header."""
    n = len(values)
    hint, strength, negative, toks = header_hint(column)
    res: dict[str, Any] = {
        "table": table,
        "column": column,
        "tag": None,
        "entity": None,
        "confidence": 0.0,
        "evidence": [],
        "status": "clean",
        "null_rate": null_rate,
        "sampled": n,
    }
    if n == 0:
        res["evidence"].append("no non-null values to sample")
        return res
    kinds = [detectors.classify_value(v) for v in values]
    counts = Counter(kinds)
    distinct = len({str(v) for v in values}) / n
    ev = res["evidence"]

    def tag(entity: str, conf: float, *why: str, status: str | None = None) -> dict[str, Any]:
        res.update(entity=entity, tag=ENTITY_TAG[entity], confidence=round(min(conf, 1.0), 2))
        res["status"] = status or ("auto" if conf >= 0.8 else "review")
        ev.extend(why)
        return res

    hint_note = f"header '{column}' suggests {hint}" if hint else f"header '{column}' gives no signal"

    # JSON documents: scan key paths
    if _rate(counts, n, "json") >= 0.8:
        keys = json_pii_keys(values)
        if keys:
            res["json_keys"] = sorted(keys)
            return tag(
                "json",
                0.95,
                f"JSON documents in {counts['json']}/{n} samples",
                "PII keys: " + ", ".join(f"{k} ({c})" for k, c in keys.most_common()),
            )
        ev.append("JSON documents without PII keys")
        return res

    # Aadhaar (including pre-masked) vs look-alike reference numbers
    aad = counts["aadhaar"] + counts["aadhaar_masked"]
    if aad / n >= 0.6:
        firsts = {str(v).strip()[0] for v, k in zip(values, kinds, strict=False) if k == "aadhaar"}
        if negative and hint != "aadhaar":
            ev.append(f"12-digit values but header '{column}' marks a reference number")
            return res
        if counts["aadhaar_masked"] or len(firsts) >= 5 or hint == "aadhaar":
            notes = [f"Aadhaar format in {aad}/{n} samples"]
            if counts["aadhaar_masked"]:
                notes.append(f"{counts['aadhaar_masked']} already masked at source")
            notes.append(hint_note)
            return tag("aadhaar", aad / n + (0.1 if hint == "aadhaar" else 0), *notes)
        ev.append(f"12-digit values but first digit only {sorted(firsts)}: reference numbers, not Aadhaar")
        return res

    # Strong formatted identifiers
    for ent in ("card", "pan", "email", "upi", "phone"):
        r = _rate(counts, n, ent)
        if r >= 0.6:
            if ent in ("email", "phone") and ("branch" in toks or "branch" in table.lower()):
                ev.append(f"{ent} values belong to a branch (business contact, not personal data)")
                return res
            if ent == "email" and all(str(v).lower().endswith("@" + detectors.ORG_DOMAIN) for v in values):
                ev.append("emails on the bank's own domain (business contact)")
                return res
            if ent == "upi" and r < 0.9 and _rate(counts, n, "name", "org") >= 0.1:
                break  # names and UPI handles mixed (e.g. counterparty): handled as free text below
            extra = " (all pass Luhn)" if ent == "card" else ""
            return tag(
                ent, r + (0.1 if hint == ent else 0), f"{ent} format in {counts[ent]}/{n} samples{extra}", hint_note
            )

    # SSN: formatted 3-2-4 or bare 9 digits with an ssn header
    ssn_like = sum(1 for v in values if detectors.RE_SSN.match(str(v).strip()))
    if ssn_like / n >= 0.6 and (hint == "ssn" or counts["ssn"] / n >= 0.3):
        return tag(
            "ssn",
            ssn_like / n + (0.1 if hint == "ssn" else 0),
            f"SSN format (dashed, spaced or bare) in {ssn_like}/{n} samples",
            hint_note,
        )

    # Long digit strings: account numbers vs ids
    if counts["digits"] / n >= 0.8 and not negative:
        luhn_ok = sum(detectors.luhn(detectors.digits(str(v))) for v in values)
        if hint == "account" or "acct" in toks or "linked" in toks:
            return tag(
                "account",
                0.9 + 0.1 * (hint == "account"),
                f"{counts['digits']}/{n} long digit strings",
                f"Luhn passes {luhn_ok}/{n}: not cards",
                hint_note,
            )
        ev.append(f"long digit strings, Luhn passes {luhn_ok}/{n}, no account header: identifiers")
        return res

    if "customer" in toks and "id" in toks:
        return tag("customer_id", 0.9, "customer identifier: links records to a person", "tokenised for analysts")

    # Dates: only birth dates are personal
    if counts["date"] / n >= 0.8:
        if hint == "dob":
            return tag("dob", 0.95, f"dates in {counts['date']}/{n} samples", hint_note)
        ev.append("dates without a birth-date header")
        return res

    # Numbers: money, credit score, postal codes
    if _rate(counts, n, "number", "digits") >= 0.8:
        if hint in ("money", "credit_score") and not negative:
            return tag(hint, 0.9, f"numeric in {n}/{n} samples", hint_note)
        if hint == "postal":
            return tag("postal", 0.9, "postal codes", hint_note)
        ev.append("numeric values without a money or postal header")
        return res

    if hint == "gender" and len({str(v) for v in values}) <= 6:
        return tag("gender", 0.85, "low-cardinality gender values", hint_note)
    if hint == "residency":
        return tag("residency", 0.75, "immigration status values", hint_note, status="review")

    # Names (after excluding orgs, cities and statuses)
    names = counts["name"]
    if names / n >= 0.6 and (distinct >= 0.4 or hint == "person"):
        return tag(
            "person",
            names / n + (0.1 if hint == "person" else 0),
            f"person-name shape in {names}/{n} samples",
            f"distinct ratio {distinct:.2f}",
            hint_note,
        )

    if counts["address"] / n >= 0.5 or (hint == "address" and distinct >= 0.3):
        conf = max(counts["address"] / n, 0.7 if hint == "address" else 0)
        return tag(
            "address",
            conf + (0.1 if hint == "address" else 0),
            f"address shape in {counts['address']}/{n} samples",
            hint_note,
        )

    # Free text (and mixed columns such as counterparty): count PII spans inside values
    texty = _rate(counts, n, "text", "name", "upi", "org", "word")
    if texty >= 0.5 and (counts["text"] or counts["upi"]) and n:
        with_spans, ents = 0, Counter()
        for v in values:
            spans = detectors.find_spans(str(v))
            if spans:
                with_spans += 1
                ents.update(e for _, _, e in spans)
        if with_spans / n >= 0.1 or (counts["upi"] and counts["name"]):
            mix = ", ".join(f"{e} {c}" for e, c in ents.most_common(5))
            return tag(
                "text",
                0.8 + 0.2 * min(1, with_spans / n),
                f"PII inside text in {with_spans}/{n} samples",
                mix or "names and UPI handles mixed",
            )
    if counts["org"] / n >= 0.5:
        ev.append("organisation names (employer, merchant): not personal data")
        return res
    if hint and strength >= 1 and not negative:
        return tag(hint, 0.6, hint_note, "values inconclusive", status="review")
    ev.append(f"no PII pattern ({', '.join(f'{k} {c}' for k, c in counts.most_common(3))})")
    return res


def scan(
    backend: Backend, progress: Callable[..., object] | None = None, sample_size: int | None = None
) -> dict[str, Any]:
    """Scan every column of every table. `progress(event, **info)` reports live status."""
    sample_size = sample_size or config.SAMPLE_SIZE
    schema = backend.schema()
    total_cols = sum(len(c) for c in schema.values())
    catalog = {"backend": backend.name, "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "columns": []}
    started = time.perf_counter()
    done = 0
    for ti, (table, cols) in enumerate(schema.items()):
        rows = backend.row_count(table)
        for column in cols:
            if progress:
                progress("column", table=table, column=column, done=done, total=total_cols, tables_done=ti)
            values = backend.sample(table, column, sample_size)
            res = classify_column(table, column, values, backend.null_rate(table, column, rows))
            res["type"] = cols[column]
            catalog["columns"].append(res)
            done += 1
        if progress:
            progress("table", table=table, rows=rows, done=done, total=total_cols, tables_done=ti + 1)
    business_records(catalog)
    catalog["elapsed_s"] = round(time.perf_counter() - started, 2)
    catalog["summary"] = summary(catalog)
    return catalog


def business_records(catalog: dict[str, Any]) -> None:
    """Contacts and quasi-identifiers only matter in tables that describe people.
    A table with no person, identity or customer link (e.g. branches) holds business data."""
    by_table = {}
    for c in catalog["columns"]:
        by_table.setdefault(c["table"], []).append(c)
    people = {"person", "aadhaar", "pan", "ssn", "card", "account", "customer_id"}
    for table, cols in by_table.items():
        if any(c["entity"] in people for c in cols):
            continue
        for c in cols:
            if c["tag"] in ("PII_MEDIUM", "PII_QUASI") and c["entity"] in ("address", "postal", "phone", "email"):
                c["evidence"].append(f"table '{table}' has no person or customer link: business record, not personal")
                c.update(tag=None, entity=None, confidence=0.0, status="clean")


def summary(catalog: dict[str, Any]) -> dict[str, int]:
    cols = catalog["columns"]
    return {
        "tagged": sum(1 for c in cols if c["tag"] and c["status"] == "auto"),
        "review": sum(1 for c in cols if c["tag"] and c["status"] == "review"),
        "clean": sum(1 for c in cols if not c["tag"]),
    }


def index(catalog: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """{(table_lower, column_lower): column info} for fast lookup."""
    return {(c["table"].lower(), c["column"].lower()): c for c in catalog.get("columns", []) if c["tag"]}


def save(catalog: dict[str, Any]) -> None:
    config.state_path(f"catalog_{catalog['backend']}.json").write_text(json.dumps(catalog, indent=1))


def load(backend_name: str) -> dict[str, Any] | None:
    p = config.state_path(f"catalog_{backend_name}.json")
    return json.loads(p.read_text()) if p.exists() else None
