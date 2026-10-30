"""Build PardaDB's NER training data: the India-first corpus from ml/datagen plus Saffron-style
banking text (support tickets, agent notes, loan remarks, account notes and
transaction narrations).

The model only learns the entities rules cannot catch: person, address and
organization. Identifiers (cards, Aadhaar, PAN, phones, UPI, accounts, RRNs) stay
in the text unlabelled, so the model learns they are *not* names.

    python gen_data.py --out data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

DATAGEN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATAGEN_DIR))

from datagen import entities, names  # noqa: E402
from datagen.generate import aligned, generate_examples, to_gliner  # noqa: E402

LABELS = ["person", "address", "organization"]

# Merchant words are split so the test split uses words never seen in training.
MERCHANT_WORDS = {
    "train": [
        "Kaveri",
        "Ganga",
        "Tulsi",
        "Pragati",
        "Banyan",
        "Neelgiri",
        "Sundar",
        "Indra",
        "Mogra",
        "Parijat",
        "Kesar",
        "Jasmine",
        "Lotus",
        "Vasant",
        "Shanti",
        "Sagar",
        "Ashoka",
        "Anand",
        "Krishna",
        "Saraswati",
        "Godavari",
        "Narmada",
        "Himalaya",
        "Vindhya",
        "Aravali",
    ],
    "dev": ["Malabar", "Konark", "Hampi", "Chandan", "Gulmohar"],
    "test": ["Kaziranga", "Sahyadri", "Nilgiri", "Brahmaputra", "Chilika", "Ajanta"],
}
MERCHANT_TYPES = [
    "JEWELLERS",
    "RESTAURANT",
    "PHARMACY",
    "MOBILE STORE",
    "BAKERY",
    "TRAVELS",
    "STEEL WORKS",
    "TEXTILES LTD",
    "DISTRICT HOSPITAL",
    "SALON",
    "FASHION",
    "MARKET",
    "ELECTRONICS",
    "CO-OP BANK",
    "MOTORS",
    "SUPERMART",
    "CEMENTS LTD",
    "LOGISTICS LLP",
    "FIBERNET",
    "DIAGNOSTICS",
    "SWEETS",
    "OPTICALS",
]
CITIES = [
    "MUMBAI",
    "PUNE",
    "CHENNAI",
    "PATNA",
    "HYDERABAD",
    "GUWAHATI",
    "VIJAYAWADA",
    "WARANGAL",
    "KOCHI",
    "INDORE",
    "NAGPUR",
    "SURAT",
    "JAIPUR",
    "LUCKNOW",
    "BHOPAL",
    "MYSURU",
    "TIRUPATI",
    "KAKINADA",
]
UPI_PSP = ["saffronupi", "pragatiupi", "tulsiupi", "neelpay", "okaxis", "ybl", "paytm"]
EMOJI = ["", "", "", " 🙏", " 😡", " 😢", " 🙏🙏", " pls"]

# {person} {PERSON} {organization} {ORGANIZATION} {address} are entities; [x] are plain fillers.
TEMPLATES = [
    # support tickets: English, Hinglish, typos, emoji
    (
        "There is a charge of Rs [amount] from {organization} on my credit card [card] which I did not "
        "authorise. Please reverse it.\nThank you,\n{person}"
    ),
    (
        "Dear Sir/Madam,\nMy debit card ending [last4] was lost while travelling. Please block it "
        "immediately and issue a replacement card to my registered address.\nRegards,\n{person}"
    ),
    (
        "MERA CARD KHO GAYA H, CARD NO [card]. PLEASE TURANT BLOCK KAR DO, KOI MISUSE NA HO JAYE MERA "
        "NUMBER [phone] HAI, CALL KARO"
    ),
    "purana number band ho gaya hai ([phone]), naya number [phone] update kar do. OTP nahi aa raha[emoji]",
    "Please add {person} as nominee for my account [acct].\nRegards,\n{person}",
    (
        "Minimum balance charges of Rs [amount] have been deducted from my salary account [acct]. A "
        "salary account should be zero balance. Please reverse.\nThanks & regards\n{person}\n[phone]"
    ),
    "uploaded pan card on app but it says name mismatch. my name is {person} same as on pan [pan]. pls fix",
    "KYC upate karna hai Aadhaar no [aadhaar] hai aur PAN [pan] Branch jaana zaroori h kya ya online ho jayega?",
    "Paisa galat UPI ID pe chala gaya [upi] Rs [amount] Wapas mangwa sakte ho kya? Mera UPI [upi] hai[emoji]",
    (
        "Bhai UPI se paymnt kiya {organization} pe, psa cut gaya, transaction failed dikha raha hai. "
        "ref no [rrn], mera number [phone]"
    ),
    "video kyc beech me disconnect ho gaya 2 baar. customer id [cif]. dobara link bhejo",
    "pls update my address to {address}. shifted last month. a/c [acct] thx",
    "Sent Rs.[amount] to my brother {person} on [upi], money deducted but not received[emoji]",
    "statement email pe nahi aa raha. mera email [email] hai, check karo",
    (
        "Hi team I lost my wallet last night, my credit card [card] was inside. pls block it asap. "
        "call me on [phone] A/c no: [acct] thx"
    ),
    (
        "There is a charge of Rs [amount] from {organization} on my credit card [card]. You can also "
        "reach me at [email]. A/c no: [acct]\nRegards,\n{person}"
    ),
    "I am {person}, my DOB is [date]. Please update my Aadhaar [aadhaar] in the account [acct].",
    "{person} ji ka cheque bounce ho gaya, unka account [acct] hai. please check",
    (
        "Please update my PAN in the account; TDS is being deducted at a higher rate. PAN: [pan], "
        "account no [acct].\nThank you,\n{person}"
    ),
    # agent notes
    "Cust shared full card no [card] in chat - advised not to share card details on chat. Card hotlisted.",
    "Mobile update request [phone] -> [phone]. OTP verified on old number.",
    "Spoke to {person}, explained charges as per schedule. Callback on [phone].",
    "Statement re-sent; password hint shared on reg mob [phone]",
    "Auto-reversal done, Rs.[amount] credited back to a/c [acct].",
    "Visited {organization} branch with {person} for re-KYC.",
    # loan remarks
    "Cust {person} called twice for status.",
    "CUST {PERSON} CALLED TWICE FOR STATUS.",
    "Co-applicant: {person}, mob [phone]. ITR verified.",
    "co-applicant: {person}, mob [phone]. salary slips verified.",
    "Co-applicant PAN [pan] - name match OK.",
    "Income docs verified. Employer {organization} confirmed employment over mail.",
    # account notes
    "Joint holder: {person} (mob [phone])",
    "POA holder {person}, contact [phone]",
    "Nominee {person} added; address {address}",
    # narrations and counterparties
    "UPI/CR/[rrn]/{PERSON}/[upi]/bday gift",
    "UPI/CR/[rrn]/{PERSON}/[upi]/loan return",
    "NEFT/SAFFN[digits11]/{PERSON}/{ORGANIZATION}",
    "NEFT/SAFFN[digits11]/{ORGANIZATION}/SALARY [month] 26",
    "IMPS/P2A/[rrn]/{PERSON}/XXXXXXXXXX[last4]/transfer",
    "POS [card6]XXXXXX[last4] {ORGANIZATION} [CITY]",
    "POS/[card]/{ORGANIZATION}/[CITY]",
    "BIL/BBPS/{ORGANIZATION}/[digits10]",
    "{PERSON}",
    "{ORGANIZATION}",
    "{person}",
]
NEGATIVES = [
    "Duplicate of earlier complaint. Closing.",
    "Cust irate. Explained charges as per schedule.",
    "Known issue on app version, fix in next release.",
    "Asked cust to visit branch with originals for reactivation.",
    "Feedback shared with branch ([phone]).",
    "User ID unlocked after verification on [phone].",
    "PAN [pan] updated in CBS. TDS refund to be processed in next cycle.",
    "NRI cust - US contact +1 ([area]) 555-01[nn]. Callback scheduled.",
    "My account [acct] has become dormant. I want to reactivate it.",
    "Form 15G submitted but still tds cut on my FD ending [last4].",
    "atm didnttt give cash but sms came for debit Rs.[amount]. ref [rrn]",
    "Legal and technical valuation done.",
    "Old a/c no [acct12] migrated from legacy CBS",
    "Cust alt email [email]",
    "Cheque book [number] lvs issued",
    "Sweep-in linked to FD",
    "Locker attached",
    "UPI/[rrn]/[upi]/dinner",
    "UPI/[rrn]/[upi]/Payment from Ph",
    "ECOM PUR/[DOMAIN].EXAMPLE/************[last4]",
    "ATW/[card6]XXXXXX[last4]/SAFF[digits4]/[CITY]",
    "TAX PMT/ADV TAX/PAN [pan]/CHALLAN 280",
    "INT.PD:01-07-2026 TO 30-09-2026",
    "RD INSTALLMENT [month] 26",
    "MIN BAL CHGS INCL GST",
    "SMS ALERT CHGS INCL GST",
    "video kyc link expired, customer id [cif]",
    "Branch Service",
    "[CITY]",
]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _phone(rng: random.Random) -> str:
    d = rng.choice("9876") + _digits(rng, 9)
    return rng.choice(
        [
            d,
            "+91-" + d,
            "+91 " + d[:5] + " " + d[5:],
            "0" + d,
            d[:5] + " " + d[5:],
            "+91-" + d[:5] + "-" + d[5:],
            "+91" + d,
        ]
    )


def _card(rng: random.Random) -> str:
    raw = entities._card(rng).replace(" ", "").replace("-", "")
    return rng.choice(
        [raw, " ".join(raw[i : i + 4] for i in range(0, 16, 4)), "-".join(raw[i : i + 4] for i in range(0, 16, 4))]
    )


def _aadhaar(rng: random.Random) -> str:
    d = rng.choice("23456789") + _digits(rng, 11)
    return rng.choice([d, f"{d[:4]} {d[4:8]} {d[8:]}", f"{d[:4]}-{d[4:8]}-{d[8:]}"])


def merchant(rng: random.Random, split: str) -> str:
    return f"{rng.choice(MERCHANT_WORDS[split])} {rng.choice(MERCHANT_TYPES)}"


def filler(key: str, rng: random.Random, split: str) -> str:
    if key == "amount":
        return rng.choice([str(rng.randint(100, 99999)), f"{rng.randint(1, 99)},{_digits(rng, 3)}"])
    if key == "card":
        return _card(rng)
    if key == "card6":
        return _card(rng).replace(" ", "").replace("-", "")[:6]
    if key == "last4":
        return _digits(rng, 4)
    if key == "phone":
        return _phone(rng)
    if key == "acct":
        return rng.choice("123456789") + _digits(rng, 13)
    if key == "acct12":
        return "0" + _digits(rng, 11)
    if key == "pan":
        return entities.pan(rng)
    if key == "aadhaar":
        return _aadhaar(rng)
    if key == "upi":
        local = rng.choice(
            [
                rng.choice("9") + _digits(rng, 9),
                names.sample_name(rng, "india", split).lower().replace(" ", rng.choice([".", "_", ""])),
            ]
        )
        return f"{local}@{rng.choice(UPI_PSP)}"
    if key == "email":
        return entities._email(rng, "india", split)
    if key == "rrn":
        return rng.choice("56") + _digits(rng, 11)
    if key == "cif":
        return "CIF-" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(8))
    if key == "date":
        return f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-{rng.randint(1945, 2005)}"
    if key.startswith("digits"):
        return _digits(rng, int(key[6:]))
    if key == "number":
        return str(rng.choice([10, 25, 50, 100]))
    if key == "month":
        return rng.choice(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])
    if key == "CITY":
        return rng.choice(CITIES)
    if key == "DOMAIN":
        return rng.choice(["MEDIKART", "TRAVELDESK", "SAFFRONRETAIL", "SHOPKART", "BOOKMYTRIP"])
    if key == "area":
        return str(rng.choice([212, 408, 512, 650, 713, 732]))
    if key == "nn":
        return _digits(rng, 2)
    if key == "emoji":
        return rng.choice(EMOJI)
    raise KeyError(key)


def entity(slot: str, rng: random.Random, split: str) -> tuple[str, str]:
    label = slot.lower()
    if label == "person":
        value = names.sample_name(rng, "india", split)
        value = value.lower() if rng.random() < 0.08 and slot == "person" else value
    elif label == "organization":
        value = merchant(rng, split) if rng.random() < 0.8 else rng.choice(entities.ORG_REAL)
        value = value.title() if slot == "organization" and rng.random() < 0.6 else value
    else:
        value = entities._india_address(rng)
    return label, value.upper() if slot.isupper() else value


def render(template: str, rng: random.Random, split: str) -> dict:
    """Fill `{entity}` and `[filler]` slots, returning a char-offset record."""
    text, ents, i = "", [], 0
    while i < len(template):
        ch = template[i]
        if ch in "{[":
            close = "}" if ch == "{" else "]"
            j = template.index(close, i)
            key = template[i + 1 : j]
            if ch == "{":
                label, value = entity(key, rng, split)
                ents.append({"start": len(text), "end": len(text) + len(value), "label": label})
                text += value
            else:
                text += filler(key, rng, split)
            i = j + 1
        else:
            text += ch
            i += 1
    return {"text": text, "entities": ents}


def bank_examples(n: int, seed: int, split: str) -> list[tuple[dict, dict]]:
    out = []
    for k in range(n):
        rng = random.Random(f"pardadb-bank/{seed}/{split}/{k}")
        pool = NEGATIVES if k % 5 == 0 else TEMPLATES
        for _ in range(20):
            chars = render(rng.choice(pool), rng, split)
            if aligned(chars):
                out.append((to_gliner(chars), chars))
                break
    return out


def parda_examples(n: int, seed: int, split: str) -> list[tuple[dict, dict]]:
    """The general India-first corpus from ml/datagen, keeping only our three labels."""
    out = []
    for _record, chars in generate_examples(n, seed, split):
        keep = [e for e in chars["entities"] if e["label"] in LABELS]
        chars = {"text": chars["text"], "entities": keep}
        out.append((to_gliner(chars), chars))
    return out


def write(path: Path, rows: list[tuple[dict, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fg, open(path.with_suffix(".chars.jsonl"), "w", encoding="utf-8") as fc:
        for g, c in rows:
            fg.write(json.dumps(g, ensure_ascii=False) + "\n")
            fc.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--bank", type=int, default=9000)
    ap.add_argument("--general", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(args.seed)
    sizes = {"train": (args.bank, args.general), "dev": (600, 300), "test": (800, 0)}
    for split, (nb, ng) in sizes.items():
        rows = bank_examples(nb, args.seed, split) + parda_examples(ng, args.seed, split)
        rng.shuffle(rows)
        write(out / f"{split}.jsonl", rows)
        n_ent = sum(len(g["ner"]) for g, _ in rows)
        print(f"{split}: {len(rows)} examples, {n_ent} entities")


if __name__ == "__main__":
    main()
