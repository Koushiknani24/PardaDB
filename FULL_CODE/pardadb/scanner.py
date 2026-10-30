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
from dataclasses import dataclass
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


@dataclass
class _Column:
    """One column under classification: its sample, what the header says, and the result being built."""

    table: str
    column: str
    values: list[Any]
    hint: str | None
    strength: float
    negative: bool
    toks: list[str]
    kinds: list[str | None]
    counts: Counter[str | None]
    res: dict[str, Any]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def distinct(self) -> float:
        return len({str(v) for v in self.values}) / self.n

    @property
    def hint_note(self) -> str:
        if self.hint:
            return f"header '{self.column}' suggests {self.hint}"
        return f"header '{self.column}' gives no signal"

    def rate(self, *ents: str) -> float:
        return _rate(self.counts, self.n, *ents)

    def tag(self, entity: str, conf: float, *why: str, status: str | None = None) -> dict[str, Any]:
        """Tag the column as `entity`; confidence >= 0.8 is applied automatically, lower goes to review."""
        self.res.update(entity=entity, tag=ENTITY_TAG[entity], confidence=round(min(conf, 1.0), 2))
        self.res["status"] = status or ("auto" if conf >= 0.8 else "review")
        self.res["evidence"].extend(why)
        return self.res

    def clean(self, why: str) -> dict[str, Any]:
        """Leave the column untagged, recording why."""
        self.res["evidence"].append(why)
        return self.res


# Each rule returns the finished result when it decides the column, or None to defer to the next rule.


def _json_rule(c: _Column) -> dict[str, Any] | None:
    """JSON documents: tagged when any key path holds PII."""
    if c.rate("json") < 0.8:
        return None
    keys = json_pii_keys(c.values)
    if not keys:
        return c.clean("JSON documents without PII keys")
    c.res["json_keys"] = sorted(keys)
    return c.tag(
        "json",
        0.95,
        f"JSON documents in {c.counts['json']}/{c.n} samples",
        "PII keys: " + ", ".join(f"{k} ({cnt})" for k, cnt in keys.most_common()),
    )


def _aadhaar_rule(c: _Column) -> dict[str, Any] | None:
    """Aadhaar (including pre-masked) vs look-alike 12-digit reference numbers."""
    n, counts = c.n, c.counts
    aad = counts["aadhaar"] + counts["aadhaar_masked"]
    if aad / n < 0.6:
        return None
    firsts = {str(v).strip()[0] for v, k in zip(c.values, c.kinds, strict=False) if k == "aadhaar"}
    if c.negative and c.hint != "aadhaar":
        return c.clean(f"12-digit values but header '{c.column}' marks a reference number")
    if counts["aadhaar_masked"] or len(firsts) >= 5 or c.hint == "aadhaar":
        notes = [f"Aadhaar format in {aad}/{n} samples"]
        if counts["aadhaar_masked"]:
            notes.append(f"{counts['aadhaar_masked']} already masked at source")
        notes.append(c.hint_note)
        return c.tag("aadhaar", aad / n + (0.1 if c.hint == "aadhaar" else 0), *notes)
    return c.clean(f"12-digit values but first digit only {sorted(firsts)}: reference numbers, not Aadhaar")


def _formatted_id_rule(c: _Column) -> dict[str, Any] | None:
    """Strong formatted identifiers: card, PAN, email, UPI, phone (minus business contacts)."""
    for ent in ("card", "pan", "email", "upi", "phone"):
        r = c.rate(ent)
        if r < 0.6:
            continue
        if ent in ("email", "phone") and ("branch" in c.toks or "branch" in c.table.lower()):
            return c.clean(f"{ent} values belong to a branch (business contact, not personal data)")
        if ent == "email" and all(str(v).lower().endswith("@" + detectors.ORG_DOMAIN) for v in c.values):
            return c.clean("emails on the bank's own domain (business contact)")
        if ent == "upi" and r < 0.9 and c.rate("name", "org") >= 0.1:
            return None  # names and UPI handles mixed (e.g. counterparty): handled as free text
        extra = " (all pass Luhn)" if ent == "card" else ""
        return c.tag(
            ent, r + (0.1 if c.hint == ent else 0), f"{ent} format in {c.counts[ent]}/{c.n} samples{extra}", c.hint_note
        )
    return None


def _card_reference_rule(c: _Column) -> dict[str, Any] | None:
    """Reference columns that carry some real card numbers (e.g. transactions.ref_no): mask value by value."""
    cards = c.counts["card"]
    if not cards or cards / c.n < 0.05:
        return None
    return c.tag("text", 0.85, f"Luhn-valid card numbers in {cards}/{c.n} samples", "masked value by value")


def _ssn_rule(c: _Column) -> dict[str, Any] | None:
    """SSN: formatted 3-2-4 or bare 9 digits with an ssn header."""
    n = c.n
    ssn_like = sum(1 for v in c.values if detectors.RE_SSN.match(str(v).strip()))
    if ssn_like / n < 0.6 or not (c.hint == "ssn" or c.counts["ssn"] / n >= 0.3):
        return None
    return c.tag(
        "ssn",
        ssn_like / n + (0.1 if c.hint == "ssn" else 0),
        f"SSN format (dashed, spaced or bare) in {ssn_like}/{n} samples",
        c.hint_note,
    )


def _long_digits_rule(c: _Column) -> dict[str, Any] | None:
    """Long digit strings: account numbers (by header) vs plain identifiers."""
    n = c.n
    if c.counts["digits"] / n < 0.8 or c.negative:
        return None
    luhn_ok = sum(detectors.luhn(detectors.digits(str(v))) for v in c.values)
    if c.hint == "account" or "acct" in c.toks or "linked" in c.toks:
        return c.tag(
            "account",
            0.9 + 0.1 * (c.hint == "account"),
            f"{c.counts['digits']}/{n} long digit strings",
            f"Luhn passes {luhn_ok}/{n}: not cards",
            c.hint_note,
        )
    return c.clean(f"long digit strings, Luhn passes {luhn_ok}/{n}, no account header: identifiers")


def _customer_id_rule(c: _Column) -> dict[str, Any] | None:
    """A customer_id header links records to a person whatever its values look like."""
    if "customer" in c.toks and "id" in c.toks:
        return c.tag("customer_id", 0.9, "customer identifier: links records to a person", "tokenised for analysts")
    return None


def _date_rule(c: _Column) -> dict[str, Any] | None:
    """Dates: only birth dates are personal."""
    if c.counts["date"] / c.n < 0.8:
        return None
    if c.hint == "dob":
        return c.tag("dob", 0.95, f"dates in {c.counts['date']}/{c.n} samples", c.hint_note)
    return c.clean("dates without a birth-date header")


def _numeric_rule(c: _Column) -> dict[str, Any] | None:
    """Numbers: money, credit score or postal code, decided by the header."""
    if c.rate("number", "digits") < 0.8:
        return None
    if c.hint in ("money", "credit_score") and not c.negative:
        return c.tag(c.hint, 0.9, f"numeric in {c.n}/{c.n} samples", c.hint_note)
    if c.hint == "postal":
        return c.tag("postal", 0.9, "postal codes", c.hint_note)
    return c.clean("numeric values without a money or postal header")


def _categorical_rule(c: _Column) -> dict[str, Any] | None:
    """Gender and residency columns, recognised by header (gender also needs few distinct values)."""
    if c.hint == "gender" and len({str(v) for v in c.values}) <= 6:
        return c.tag("gender", 0.85, "low-cardinality gender values", c.hint_note)
    if c.hint == "residency":
        return c.tag("residency", 0.75, "immigration status values", c.hint_note, status="review")
    return None


def _person_name_rule(c: _Column) -> dict[str, Any] | None:
    """Person names (detectors already exclude orgs, cities and statuses)."""
    names, distinct = c.counts["name"], c.distinct
    if names / c.n < 0.6 or not (distinct >= 0.4 or c.hint == "person"):
        return None
    return c.tag(
        "person",
        names / c.n + (0.1 if c.hint == "person" else 0),
        f"person-name shape in {names}/{c.n} samples",
        f"distinct ratio {distinct:.2f}",
        c.hint_note,
    )


def _address_rule(c: _Column) -> dict[str, Any] | None:
    """Addresses by value shape, or by header when values are varied enough."""
    addr = c.counts["address"] / c.n
    if not (addr >= 0.5 or (c.hint == "address" and c.distinct >= 0.3)):
        return None
    conf = max(addr, 0.7 if c.hint == "address" else 0)
    return c.tag(
        "address",
        conf + (0.1 if c.hint == "address" else 0),
        f"address shape in {c.counts['address']}/{c.n} samples",
        c.hint_note,
    )


def _free_text_rule(c: _Column) -> dict[str, Any] | None:
    """Free text (and mixed columns such as counterparty): count PII spans inside values."""
    counts, n = c.counts, c.n
    if c.rate("text", "name", "upi", "org", "word") < 0.5 or not (counts["text"] or counts["upi"]):
        return None
    with_spans, ents = 0, Counter()
    for v in c.values:
        spans = detectors.find_spans(str(v))
        if spans:
            with_spans += 1
            ents.update(e for _, _, e in spans)
    if not (with_spans / n >= 0.1 or (counts["upi"] and counts["name"])):
        return None
    mix = ", ".join(f"{e} {cnt}" for e, cnt in ents.most_common(5))
    return c.tag(
        "text",
        0.8 + 0.2 * min(1, with_spans / n),
        f"PII inside text in {with_spans}/{n} samples",
        mix or "names and UPI handles mixed",
    )


def _fallback_rule(c: _Column) -> dict[str, Any]:
    """No value pattern decided: organisation names stay clean, a strong header alone goes to review."""
    if c.counts["org"] / c.n >= 0.5:
        return c.clean("organisation names (employer, merchant): not personal data")
    if c.hint and c.strength >= 1 and not c.negative:
        return c.tag(c.hint, 0.6, c.hint_note, "values inconclusive", status="review")
    return c.clean(f"no PII pattern ({', '.join(f'{k} {cnt}' for k, cnt in c.counts.most_common(3))})")


# Order matters: the first rule that returns a result decides the column.
_RULES: tuple[Callable[[_Column], dict[str, Any] | None], ...] = (
    _json_rule,
    _aadhaar_rule,
    _formatted_id_rule,
    _card_reference_rule,
    _ssn_rule,
    _long_digits_rule,
    _customer_id_rule,
    _date_rule,
    _numeric_rule,
    _categorical_rule,
    _person_name_rule,
    _address_rule,
    _free_text_rule,
)


def classify_column(table: str, column: str, values: list[Any], null_rate: float) -> dict[str, Any]:
    """Decide one column's tag from its sampled values and header."""
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
        "sampled": len(values),
    }
    if not values:
        res["evidence"].append("no non-null values to sample")
        return res
    kinds = [detectors.classify_value(v) for v in values]
    col = _Column(table, column, values, hint, strength, negative, toks, kinds, Counter(kinds), res)
    for rule in _RULES:
        decided = rule(col)
        if decided is not None:
            return decided
    return _fallback_rule(col)


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
