"""Value-level PII detectors, canonicalisers and the free-text span finder.

Rules are tuned for messy legacy bank data:
- cards must pass Luhn (16-digit transaction ids do not),
- Aadhaar is recognised by format and context, never gated on Verhoeff
  (synthetic and legacy data often fail it), and pre-masked values count,
- SSNs in the 900-999 test range are accepted,
- every phone/card/Aadhaar format is canonicalised before masking or hashing.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

Span = tuple[int, int, str]  # (start, end, entity) inside a text

ORG_DOMAIN = "saffronbank.example"

RE_PREMASKED_AADHAAR = re.compile(r"^[Xx*]{4}[ -]?[Xx*]{4}[ -]?\d{4}$")
RE_CARD = re.compile(r"^\d{4}(?:[ -]?\d{4}){2}[ -]?\d{1,7}$")
RE_AADHAAR = re.compile(r"^[2-9]\d{3}([ -]?)\d{4}\1\d{4}$")
RE_SSN = re.compile(r"^\d{3}([- ]?)\d{2}\1\d{4}$")
RE_PAN = re.compile(r"^[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]$")
RE_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")
RE_UPI = re.compile(r"^[\w.-]{2,}@[a-z]{2,}$", re.I)
RE_PHONE_IN = re.compile(r"^(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}$")
RE_PHONE_US = re.compile(r"^\+1 ?\(\d{3}\) ?\d{3}-\d{4}$")
RE_IFSC = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RE_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
RE_DIGITS = re.compile(r"^\d{9,18}$")
RE_POSTAL = re.compile(r"^\d{5,6}(?:-\d{4})?$")
RE_NAME = re.compile(r"^(?:[A-Z][a-z]+|[A-Z]{2,})(?:[ .]+(?:[A-Z][a-z]+|[A-Z]{2,}|[A-Z]\.?)){1,3}$")
ORG_WORDS = {
    "co",
    "ltd",
    "llp",
    "pvt",
    "inc",
    "works",
    "engineering",
    "textiles",
    "cements",
    "analytics",
    "logistics",
    "bank",
    "store",
    "stores",
    "jewellers",
    "restaurant",
    "pharmacy",
    "bakery",
    "travels",
    "hospital",
    "salon",
    "fashion",
    "market",
    "electronics",
    "motors",
    "fibernet",
    "apartments",
    "heights",
    "nagar",
    "township",
    "society",
    "enclave",
    "colony",
    "park",
    "gardens",
    "infotech",
    "solutions",
    "services",
    "industries",
    "foods",
    "traders",
    "enterprises",
    "steel",
    "corporation",
}
ADDRESS_WORDS = {
    "cross",
    "marg",
    "road",
    "lane",
    "street",
    "nagar",
    "colony",
    "apt",
    "plot",
    "floor",
    "dr",
    "ave",
    "avenue",
    "rd",
    "st",
    "blvd",
    "sector",
    "block",
    "flat",
    "house",
    "main",
    "layout",
    "society",
    "apartments",
    "township",
    "enclave",
    "way",
    "ct",
    "court",
    "drive",
}


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn(d: str) -> bool:
    if not d.isdigit() or not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2:
            n *= 2
            n -= 9 if n > 9 else 0
        total += n
    return total % 10 == 0


def classify_value(v: Any) -> str | None:
    """Best entity guess for a single cell value (no column context)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return "number"
    s = str(v).strip()
    if not s:
        return None
    if RE_PREMASKED_AADHAAR.match(s):
        return "aadhaar_masked"
    if s[:1] in "{[":
        try:
            if isinstance(json.loads(s), (dict, list)):
                return "json"
        except ValueError:
            pass
    if RE_CARD.match(s) and luhn(digits(s)):
        return "card"
    if RE_AADHAAR.match(s):
        return "aadhaar"
    if RE_SSN.match(s):
        return "ssn"
    if RE_PAN.match(s):
        return "pan"
    if RE_PHONE_US.match(s):
        return "phone"
    if RE_EMAIL.match(s):
        return "email"
    if RE_UPI.match(s):
        return "upi"
    if RE_PHONE_IN.match(s):
        return "phone"
    if RE_IFSC.match(s):
        return "ifsc"
    if RE_DATE.match(s) or re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", s):
        return "date"
    if RE_DIGITS.match(s):
        return "digits"
    if RE_NUMBER.match(s):
        return "number"
    words = s.split()
    low = {w.strip(",.").lower() for w in words}
    if len(words) >= 10 and len(s) > 50:
        return "text"  # sentences (complaints, notes) are free text even if they mention a 'block' or 'road'
    if len(words) >= 2 and any(ch.isdigit() for ch in s) and low & ADDRESS_WORDS and len(s) < 160:
        return "address"
    if RE_NAME.match(s) and not low & ORG_WORDS:
        return "name"
    if len(words) >= 2 and low & ORG_WORDS and len(s) < 80:
        return "org"
    if len(s) > 25 and len(words) >= 4:
        return "text"
    return "word"


# ---------------------------------------------------------------- canonical forms


def canonical(entity: str, v: Any) -> str:
    """One canonical string per real-world value, so formats never split a token."""
    s = str(v).strip()
    if entity in ("card", "aadhaar", "ssn", "account"):
        return digits(s)
    if entity == "phone":
        d = digits(s)
        if s.startswith("+1"):
            return "+1" + d[-10:]
        return "+91" + d[-10:]
    if entity in ("email", "upi", "pan"):
        return s.lower()
    if entity == "person":
        return " ".join(s.lower().split())
    return s


# ---------------------------------------------------------------- free-text spans

NAME = r"(?:[A-Z][a-z]+|[A-Z]{2,})(?:[ ](?:[A-Z][a-z]+|[A-Z]{2,})){1,3}"
SPAN_RULES = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*[a-z]{2,}", re.I)),
    ("upi", re.compile(r"[\w.-]{2,}@(?:[a-z]*upi|ok[a-z]+|ybl|paytm|neelpay|apl|ibl|axl)\b", re.I)),
    ("card", re.compile(r"(?<![\d-])\d{4}(?:[ -]?\d{4}){2}[ -]?\d{3,7}(?![\d-])")),
    ("aadhaar", re.compile(r"(?<![\d-])[2-9]\d{3}[ -]\d{4}[ -]\d{4}(?![\d-])")),
    ("pan", re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b")),
    ("ssn", re.compile(r"\b9\d{2}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"\+1 ?\(\d{3}\) ?\d{3}-\d{4}")),
    ("phone", re.compile(r"(?<![\w-])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?![\d-])")),
    ("account", re.compile(r"(?<![\w])\d{11,18}(?![\d])")),
]
# Indian postal addresses inside free text: a house number, an address word, then a 6-digit pincode.
PINCODE = re.compile(r"(?<![\d₹])(?<!Rs )(?<!Rs\. )(?<!INR )\b[1-9]\d{5}\b(?![\d,.]\d)")
HOUSE_NO = re.compile(r"(?<![\w/.-])\d{1,4}(?:[/-]\d{1,4}){0,2}[A-Z]?(?=[,\s])")
HOUSE_PREFIX = re.compile(r"(?:Flat|Plot|House|H\.\s?No\.?|Door No\.?|#)\s*$", re.I)
# the column-level address words plus words that mostly appear inside written addresses
ADDRESS_TEXT_WORDS = {
    "path", "residency", "heights", "towers", "gardens", "park", "puram", "extension", "vihar",
    "near", "temple", "chowk", "bazaar", "gali", "estate", "complex", "phase",
}  # fmt: skip
ADDRESS_HINT = re.compile(r"\b(?:" + "|".join(sorted(ADDRESS_WORDS | ADDRESS_TEXT_WORDS)) + r")\b", re.I)
ADDRESS_WINDOW = 110


def address_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of addresses: the shortest 'house number … pincode' stretch that has an address word."""
    spans = []
    for pin in PINCODE.finditer(text):
        lo = max(0, pin.start() - ADDRESS_WINDOW)
        window = text[lo : pin.start()]
        if "\n" in window:
            lo += window.rindex("\n") + 1
        starts = [m.start() for m in HOUSE_NO.finditer(text, lo, pin.start())]
        for start in reversed(starts):
            prefix = HOUSE_PREFIX.search(text, max(0, start - 12), start)
            begin = prefix.start() if prefix else start
            if ADDRESS_HINT.search(text, begin, pin.start()):
                spans.append((begin, pin.end()))
                break
    return spans


NAME_CONTEXT = [
    re.compile(r"(?:Regards|regards|Thanks|Thank you|thanks & regards|Thanks & regards),?\s*\n\s*(" + NAME + ")"),
    re.compile(r"(?i:my name is)\s+(" + NAME + ")"),
    re.compile(r"(?i:cust)\s+(" + NAME + r")\s+(?i:called)"),
    re.compile(r"(?i:co-applicant:?)\s+(" + NAME + ")"),
    re.compile(r"(?i:joint holder:?)\s+(" + NAME + ")"),
    re.compile(r"(?i:poa holder)\s+(" + NAME + ")"),
    re.compile(r"(?i:nominee)\s+(" + NAME + ")"),
    re.compile(r"(?i:add)\s+(" + NAME + r")\s+(?i:as nominee)"),
    re.compile(r"(?i:spoke to)\s+(" + NAME + ")"),
    re.compile(r"^(?:UPI/CR|UPI/DR)/\d+/(" + NAME + ")/"),
    re.compile(r"^(?:NEFT|RTGS)/[A-Z0-9]+/(" + NAME + ")/"),
    re.compile(r"^IMPS/P2A/\d+/(" + NAME + ")/"),
]


class NameIndex:
    """Full names known to the bank (from person-tagged columns), matched in free text."""

    def __init__(self, names: Iterable[Any] = ()) -> None:
        self.names = {" ".join(n.lower().split()) for n in names if n and " " in str(n).strip()}
        self.max_words = max((n.count(" ") + 1 for n in self.names), default=0)

    def find(self, text: str) -> list[Span]:
        """Spans of known full names in `text`, preferring the longest match at each word."""
        if not self.names:
            return []
        out = []
        toks = [(m.start(), m.end(), m.group().lower()) for m in re.finditer(r"[A-Za-z]+", text)]
        i = 0
        while i < len(toks):
            hit = None
            for k in range(min(self.max_words, len(toks) - i), 1, -1):
                cand = " ".join(t[2] for t in toks[i : i + k])
                if cand in self.names:
                    hit = (toks[i][0], toks[i + k - 1][1], "person")
                    i += k
                    break
            if hit:
                out.append(hit)
            else:
                i += 1
        return out


def find_spans(text: str, names: NameIndex | None = None, extra: Iterable[Span] | None = None) -> list[Span]:
    """Non-overlapping (start, end, entity) spans of PII inside free text.
    `extra` adds spans from the ML model, labelled e.g. "person (AI)"."""
    if not text:
        return []
    found = []
    for entity, rx in SPAN_RULES:
        for m in rx.finditer(text):
            s = m.group()
            if entity == "card" and not luhn(digits(s)):
                entity_here = "account" if len(digits(s)) >= 11 else None
                if not entity_here:
                    continue
                found.append((m.start(), m.end(), entity_here))
                continue
            found.append((m.start(), m.end(), entity))
    for rx in NAME_CONTEXT:
        for m in rx.finditer(text):
            found.append((m.start(1), m.end(1), "person"))
    found += [(s_, e_, "address") for s_, e_ in address_spans(text)]
    if names:
        found += names.find(text)
    if extra:
        rule_spans = list(found)
        for s_, e_, ent in extra:
            # identifiers found by rules win; the model only adds names and addresses rules missed
            if not any(a < e_ and s_ < b for a, b, _ in rule_spans):
                found.append((s_, e_, f"{ent} (AI)"))
    # keep the earliest, longest span when two overlap
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, last_end = [], -1
    for s, e, ent in found:
        if s >= last_end:
            out.append((s, e, ent))
            last_end = e
    return out
