"""Masking strategies. Each takes (value, column info, context) and returns what the role may see.
Values are canonicalised first, so every stored format masks and hashes the same way."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from typing import Any

from . import config, detectors

log = logging.getLogger(__name__)

Span = tuple[int, int, str]  # (start, end, entity) inside a text value

_KEY: bytes | None = None


def hmac_key() -> bytes:
    """Token key: PARDADB_HMAC_KEY if set, else a random key created once in the state dir."""
    global _KEY
    if _KEY is None:
        env = os.environ.get("PARDADB_HMAC_KEY", "").strip()
        if env.startswith("change-me"):
            log.warning("PARDADB_HMAC_KEY is the .env.example placeholder; using a generated key instead")
            env = ""
        if env:
            _KEY = env.encode()
        else:
            p = config.state_path(".hmac_key")
            if not p.exists():
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(secrets.token_hex(32))
            _KEY = p.read_text().strip().encode()
    return _KEY


def token(entity: str, v: Any) -> str:
    """Keyed HMAC of the canonical value: same person -> same token, joins and GROUP BY still work."""
    raw = f"{entity}:{detectors.canonical(entity, v)}".encode()
    return "tok_" + hmac.new(hmac_key(), raw, hashlib.sha256).hexdigest()[:12]


def _last4_mask(d: str, fill: str = "X") -> str:
    return fill * max(len(d) - 4, 0) + d[-4:]


def partial_value(entity: str, v: Any) -> str:
    """Entity-aware partial reveal, e.g. a card keeps its first and last four digits."""
    s = str(v).strip()
    if entity == "card":
        d = detectors.digits(s)
        return f"{d[:4]}-XXXX-XXXX-{d[-4:]}"
    if entity == "aadhaar":
        return "XXXX XXXX " + detectors.digits(s)[-4:]
    if entity == "pan":
        return "XXXXXX" + s[-4:]
    if entity == "ssn":
        return "XXX-XX-" + detectors.digits(s)[-4:]
    if entity == "account":
        return _last4_mask(detectors.digits(s))
    if entity == "phone":
        c = detectors.canonical("phone", s)
        return ("+1 (XXX) XXX-" if c.startswith("+1") else "+91 XXXXXX") + c[-4:]
    if entity in ("email", "upi"):
        user, _, dom = s.partition("@")
        return user[:1] + "***@" + dom
    if entity == "person":
        parts = s.split()
        return " ".join([parts[0]] + [p[0] + "****" for p in parts[1:]]) if parts else s
    if entity == "address":
        return re.sub(r"\d", "X", s)
    if entity == "dob":
        return s[:4]
    if entity == "postal":
        return s[:3] + "X" * (len(s) - 3)
    if entity in ("gender", "residency", "customer_id"):
        return s
    return _last4_mask(s)


def age_on(born: dt.date, today: dt.date) -> int:
    """Completed years of age on `today` (exact, leap-year safe)."""
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def generalize(entity: str, v: Any, today: dt.date | None = None) -> str:
    """Coarsen a value: date of birth to a 10-year age band, postal code to its prefix, gender to M/F."""
    s = str(v).strip()
    if entity == "dob":
        try:
            born = dt.date.fromisoformat(s[:10])
            age = age_on(born, today or dt.date.today())
            lo = age // 10 * 10
            return f"{lo}-{lo + 9}"
        except ValueError:
            return "[REDACTED]"
    if entity == "postal":
        return s[:3] + "X" * (len(s) - 3)
    if entity == "gender":
        return {"m": "M", "male": "M", "f": "F", "female": "F"}.get(s.lower(), s)
    return "[REDACTED]"


def bucket(v: Any) -> str:
    """Amount range label in Indian units (K, lakh, crore); non-numbers are redacted."""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "[REDACTED]"
    for hi, label in ((10_000, "< 10K"), (1_00_000, "10K-1L"), (10_00_000, "1L-10L"), (1_00_00_000, "10L-1Cr")):
        if n < hi:
            return label
    return "> 1Cr"


SPAN_LABEL = {
    "card": "[CARD]",
    "aadhaar": "[AADHAAR]",
    "pan": "[PAN]",
    "ssn": "[SSN]",
    "account": "[ACCOUNT]",
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "upi": "[UPI]",
    "person": "[NAME]",
    "address": "[ADDRESS]",
}


def mask_text(
    text: str,
    typed: bool,
    names: detectors.NameIndex | None = None,
    with_marks: bool = False,
    extra: list[Span] | None = None,
) -> str | tuple[str, list[list[int | str]]]:
    """Mask each PII span inside free text. With `with_marks`, also return
    [start, end, entity] of every masked span in the *output* text."""
    spans = detectors.find_spans(text, names, extra)
    out: list[str] = []
    marks: list[list[int | str]] = []
    pos = length = 0
    for s, e, ent in spans:
        out.append(text[pos:s])
        length += s - pos
        base = ent.split(" (")[0]
        rep = SPAN_LABEL.get(base, "[PII]") if typed else partial_value(base, text[s:e])
        marks.append([length, length + len(rep), ent])
        out.append(rep)
        length += len(rep)
        pos = e
    out.append(text[pos:])
    masked = "".join(out)
    return (masked, marks) if with_marks else masked


_NOTE_KEYS = frozenset({"remarks", "note", "notes"})
_JSON_ENTITY = {"name": "person", "aadhaar_masked": "aadhaar", "digits": "account"}
# value kinds that identify a person on their own: such text is hidden even when it is an object key
_KEY_PII_KINDS = frozenset({"card", "aadhaar", "aadhaar_masked", "pan", "ssn", "phone", "email", "upi"})
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


def json_key(key: str) -> str:
    """Form a JSON key is matched in: case-insensitive, surrounding whitespace ignored ("PAN " -> "pan")."""
    return key.strip().lower()


def _json_scope(parent: str | None, key: str, pii_keys: frozenset[str]) -> str:
    """Key a nested value is masked under: a PII parent wins, then a note parent, then the value's own key."""
    if parent is not None and (parent in pii_keys or (parent in _NOTE_KEYS and key not in pii_keys)):
        return parent
    return key


def _key_leaks(key: str, parent: str | None, pii_keys: frozenset[str]) -> bool:
    """True when an object key may itself be personal data, e.g. the PAN in {"pan": {"ABCPK1234F": true}}.

    Keys shaped like an identifier (PAN, card, phone, ...) are hidden anywhere. Under a PII or note key
    only plain field names ("value", "verified") survive: names, numbers and free text are hidden.
    """
    kind = detectors.classify_value(key)
    if kind in _KEY_PII_KINDS:
        return True
    sensitive = parent is not None and (parent in pii_keys or parent in _NOTE_KEYS)
    return sensitive and (kind != "word" or not _FIELD_NAME.fullmatch(key.strip()))


def _mask_json_leaf(v: Any, key: str | None, pii_keys: frozenset[str], redact: bool) -> Any:
    """Mask one scalar: values under a PII key are redacted or partially revealed, note strings get span masking."""
    if v is None:
        return None
    if key in pii_keys:
        if redact:
            return "[REDACTED]"
        kind = detectors.classify_value(v)
        return partial_value(_JSON_ENTITY.get(kind, kind), v)
    if key in _NOTE_KEYS and isinstance(v, str):
        return mask_text(v, typed=redact)
    return v


def _mask_json_object(
    node: dict[str, Any], scope: str | None, pii_keys: frozenset[str], redact: bool
) -> dict[str, Any]:
    """Mask every member of an object. Keys that may be PII become "[REDACTED]_n", unique within the object."""
    out: dict[str, Any] = {}
    hidden = 0
    for k, v in node.items():
        nk = json_key(k)
        value = _mask_json_node(v, _json_scope(scope, nk, pii_keys), pii_keys, redact)
        if _key_leaks(k, scope, pii_keys):
            hidden += 1
            while f"[REDACTED]_{hidden}" in node:
                hidden += 1
            k = f"[REDACTED]_{hidden}"
        out[k] = value
    return out


def _mask_json_node(node: Any, key: str | None, pii_keys: frozenset[str], redact: bool) -> Any:
    """Walk a parsed JSON value: objects at any depth, arrays of anything (items inherit the array's key)."""
    if isinstance(node, dict):
        return _mask_json_object(node, key, pii_keys, redact)
    if isinstance(node, list):
        return [_mask_json_node(v, key, pii_keys, redact) for v in node]
    return _mask_json_leaf(node, key, pii_keys, redact)


def mask_json(v: Any, info: dict[str, Any], redact: bool) -> str:
    """Mask the PII keys of a JSON document at any depth (redact or partial) and the PII spans in its note fields.
    Keys match case-insensitively and whitespace-trimmed; keys that are themselves PII are replaced.
    Anything that cannot be parsed or walked (invalid or absurdly deep JSON) is redacted whole: fail closed."""
    pii_keys = frozenset(json_key(k) for k in info.get("json_keys", []))
    try:
        doc = _mask_json_node(json.loads(v), None, pii_keys, redact)
    except (TypeError, ValueError, RecursionError) as exc:
        log.debug("JSON value redacted whole: %s", type(exc).__name__)
        return "[REDACTED]"
    return json.dumps(doc, ensure_ascii=False, sort_keys=True)


def apply(strategy: str, v: Any, info: dict[str, Any] | None, ctx: dict[str, Any] | None = None) -> Any:
    """Apply one strategy to one value."""
    if v is None or strategy == "FULL":
        return v
    entity = (info or {}).get("entity") or ""
    ctx = ctx or {}
    if strategy == "REDACT":
        return "[REDACTED]"
    if strategy == "HASH":
        return token(entity, v)
    if strategy == "PARTIAL":
        if entity == "aadhaar" and detectors.RE_PREMASKED_AADHAAR.match(str(v).strip()):
            return "XXXX XXXX " + str(v)[-4:]
        return partial_value(entity, v)
    if strategy == "BUCKET":
        return bucket(v)
    if strategy == "GENERALIZE":
        return generalize(entity, v)
    if strategy in ("SPANS_PARTIAL", "SPANS_TYPED"):
        return mask_text(str(v), strategy == "SPANS_TYPED", ctx.get("names"))
    if strategy in ("JSON_PARTIAL", "JSON_REDACT"):
        return mask_json(v, info or {}, strategy == "JSON_REDACT")
    return "[REDACTED]"


DISPLAY = {"SPANS_PARTIAL": "SPANS", "SPANS_TYPED": "SPANS", "JSON_PARTIAL": "JSON", "JSON_REDACT": "JSON"}
