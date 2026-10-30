# 1448 · B1: PardaDB, Smart Data Masking for PII Protection

**PardaDB finds personal data in any SQL database by itself and shows each role only what it may see, at the moment a query runs, without ever changing the data.**

Team AVNI · TalentAP Mega Hackathon 2026 · Problem statement B1 (BFSI): *Smart Data Masking System for PII Protection*.

## What it does

Inside a bank, support agents, analysts and administrators all query the same customer tables. Without column-level controls, every one of them can read every card number, Aadhaar number and balance. PardaDB sits between those people and the databases:

1. **Scans** every table by itself: it samples values, checks formats and checksums, reads column names (even abbreviated or Hindi ones like `nm`, `mob`, `pita`), and tags each column `PII_HIGH`, `PII_MEDIUM`, `PII_QUASI`, `FINANCIAL_CONFIDENTIAL`, `PII_FREE_TEXT` or `SENSITIVE_JSON`, with a confidence and the evidence.
2. **Masks at query time** by role: Admin sees everything, Support sees partial values (`4111-XXXX-XXXX-1111`), Analyst sees redactions, irreversible tokens (`tok_65d85106690a`) or ranges (`30-39`, `1L-10L`). PII inside free-text tickets, notes and JSON is masked piece by piece.
3. **Blocks probing**: the inference guard refuses queries that would reveal hidden values indirectly (`WHERE CARD_NO LIKE '4111%'`, `ORDER BY curr_bal`), and every write to bank data.
4. **Proves it**: SHA-256 fingerprints show every table is unchanged; every query, allowed or blocked, goes into a hash-chained (tamper-evident) audit log.

It works on **SQLite, MySQL 8 and PostgreSQL 16** with identical output, and runs in one small process (about 45 MB of RAM).

**Who uses it and what they see:** bank staff query through the web dashboard, the full-screen terminal UI, the `pardadb` CLI or the REST API. Each sees the same rows, masked for their role. Evaluators can switch Admin / Support / Analyst on the same query and watch the results change in real time.

Measured on the organisers' Saffron Bank data (9 tables, 120 columns, 129,260 rows):

| Result | Value |
|---|---|
| Columns scanned | 120 in about 0.5 s: 46 tagged, 2 for review, 72 clean |
| Hard cases found | `col_7` (no useful name) found to hold PAN; look-alike `rrn` and `txn_id` left clean |
| Addresses inside free text found by rules | 164 of 164 |
| Customer names inside tickets and remarks | rules alone 0.911 recall; rules + optional AI model 0.993 |
| Same masked output on all three databases | yes, including identical tokens |

## How to run

**Requirements:** Python 3.10+, and [uv](https://docs.astral.sh/uv/) (recommended) or pip. Docker is only needed for the MySQL and PostgreSQL demo.

**1. Put the problem pack's files in `FULL_CODE/data/`** (this folder is git-ignored and never committed). PardaDB expects the pack layout:

```
FULL_CODE/data/sql/sqlite/saffron_bank.db
FULL_CODE/data/data/users.csv
FULL_CODE/data/docker-compose.yml        (for MySQL / PostgreSQL)
```

Copy the whole `B1_Smart_Data_Masking_PII` folder's contents there, or point `PARDADB_DATASET` at it.

**2. Install and configure:**

```bash
cd FULL_CODE
uv sync --extra dev               # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env              # then fill in the database passwords from the pack README
```

**3. (Optional) start MySQL and PostgreSQL** with the pack's Docker setup:

```bash
cd data && docker compose up -d && cd ..
```

**4. Run:**

```bash
uv run pardadb                    # or: python main.py
```

This connects to every database it can reach, fingerprints every table, scans every column, opens the terminal UI, and serves the web app at <http://127.0.0.1:8765> (landing page; **DEMO** opens the dashboard). Databases that are not running yet are retried every 10 seconds.

| Command | What it does |
|---|---|
| `pardadb` | Start everything and keep monitoring (`--db sqlite,mysql,postgres`, `--port`, `--no-browser`, `--no-tui`, `--ml`) |
| `pardadb scan [--db sqlite] [--all]` | Scan once and print the PII catalog with evidence |
| `pardadb query --user support_ravi --sql "SELECT ..."` | Run one query as a user (or `--role analyst`) |
| `pardadb compare --sql "SELECT ..."` | The same query as Admin, Support and Analyst |
| `pardadb verify [--db sqlite]` | SHA-256 fingerprints of every table |
| `pardadb status` | Health of a running PardaDB |

**API / Postman:** import `FULL_CODE/api/pardadb.postman_collection.json` and run it against a running PardaDB. Its 15 requests assert the masking per role, the blocked bypass attempts, input validation and the integrity proof.

**Environment variables** (all optional, see `FULL_CODE/.env.example`): `PARDADB_DATASET`, `PARDADB_MYSQL_*`, `PARDADB_PG_*`, `PARDADB_HMAC_KEY` (token key; a random one is created in `.state/` if unset), `PARDADB_ML_URL`, `PARDADB_ML_THRESHOLD`, `PARDADB_ALLOWED_HOSTS`. Logs go to `FULL_CODE/.state/pardadb.log`.

### Optional AI add-on

Rules handle everything with a format. A small open model (GLiNER, 45 M parameters), fine-tuned by us, adds person names and addresses inside free text that have no format. It runs as a local sidecar; no data leaves the machine. If it is off or down, PardaDB masks with rules only.

```bash
cd FULL_CODE/ml
uv venv && uv pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python gen_data.py                # rebuilds dataset/ exactly (fixed seeds)
.venv/bin/python train.py                   # about 3 minutes on a small GPU -> models/pardadb-pii (safetensors)
.venv/bin/python evaluate.py --model models/pardadb-pii --tag ft
```

Then start PardaDB with `pardadb --ml`, or switch the model on from the dashboard header or the TUI (`m`). Model weights are not committed (git-ignored); the command above recreates them.

## How to run the tests

```bash
cd FULL_CODE
uv run pytest                     # 217 tests, about 1 second, no network and no problem pack needed
uv run ruff check . && uv run ruff format --check .
```

The tests build a small synthetic bank in a temporary folder and mock the AI sidecar. `tests/test_real_pack.py` additionally checks the real Saffron data when `FULL_CODE/data/` is present.

## Architecture

```
  Web dashboard + landing page   Terminal UI   CLI `pardadb`   REST API / Postman
                 \                     |              |              /
                  +--------------- server.py (one process) ----------+
                  |  control plane                 data plane        |
                  |  scanner.py  -> PII catalog    engine.py         |
                  |  detectors.py   (tag, conf.,   sqlglot parse ->  |
                  |                  evidence)     lineage ->        |
                  |  policy.json (tag x role)      inference guard ->|
                  |                                masks.py ->       |
                  |                                audit.py (chain)  |
                  +------------------------+-------------------------+
                                           | read-only connections (backends.py)
            SQLite (mode=ro)     MySQL 8 (read-only session)     PostgreSQL 16 (read-only session)
            Playground SQLite (writable test tables; schema in db/)
                                           |
                  optional: ml/serve.py, fine-tuned GLiNER on localhost (ml.py client + precision filter)
```

| Path in `FULL_CODE/` | Responsibility |
|---|---|
| `main.py` | Entry point (same as the `pardadb` command) |
| `pardadb/config.py` | All settings: paths, connections, limits, AI flags; reads `.env` |
| `pardadb/backends.py` | Read-only SQLite / MySQL / PostgreSQL adapters with one interface; writable Playground |
| `pardadb/detectors.py` | Value detectors (Luhn, Aadhaar, PAN, SSN, phones, email, UPI, addresses), free-text spans |
| `pardadb/scanner.py` | Column classification: tag, confidence, evidence; the PII catalog |
| `pardadb/policy.json` | Tag × role → masking strategy |
| `pardadb/masks.py` | Masking strategies: partial, keyed HMAC tokens, redaction, ranges, age bands |
| `pardadb/engine.py` | Query parsing, lineage, inference guard, masking of results |
| `pardadb/audit.py` | Hash-chained audit log |
| `pardadb/ml.py` | Client for the optional AI sidecar, with a precision filter |
| `pardadb/server.py` | Runtime, background monitoring, JSON API with input validation |
| `pardadb/cli.py`, `tui.py`, `app.py` | CLI, classic and full-screen terminal UIs |
| `pardadb/dashboard/` | Offline web dashboard and landing page (no CDN) |
| `db/` | Playground schema and seed SQL (bank schemas come from the problem pack) |
| `api/` | Postman collection |
| `ml/` | AI add-on: data generator, dataset, training, evaluation, serving |
| `tests/` | pytest suite |

Bank databases are never written: SQLite opens with `mode=ro` and `query_only`, MySQL and PostgreSQL sessions are set read-only, and the engine refuses write statements before they reach a database.

## AI usage

**In the product (optional add-on):**
- Model: [`knowledgator/gliner-pii-edge-v1.0`](https://huggingface.co/knowledgator/gliner-pii-edge-v1.0) (Apache-2.0), fine-tuned by us for Indian banking text. Labels: person, address, organization (organisation spans are ignored at runtime).
- Training data: 18,000 synthetic examples generated by `ml/gen_data.py` (Hinglish and English support tickets, agent notes, loan remarks, payment narrations, plus general India-first text). It is committed in `ml/dataset/` and regenerates byte-for-byte from fixed seeds. **The organisers' data was never used for training**, only for evaluation.
- Evaluation (`ml/reports/`): held-out bank test F1 0.527 (base) → 0.984 (fine-tuned); hand-written test set F1 0.800 → 0.936; customer names in 918 organiser tickets and remarks: recall 0.911 (rules) → 0.999 (model); counterparty names in 2,883 narrations: 0.630 → 0.942.
- Safety: served on `127.0.0.1` only, weights loaded from safetensors (no pickle, no `trust_remote_code`), model output is treated as untrusted. Spans pass a precision filter (`ml.refine`) and can only *add* masking, never remove it.

**AI coding assistant:** we used Claude Code (Anthropic) throughout the build to write and refactor code, tests and documentation, and to run measurements, under the team's direction. No other LLM API is called at runtime.

## Pre-built and third-party code

- **Written by our team before Day 1, for this hackathon:** `ml/datagen/` (the India-first synthetic text generator and its name and organisation lists) and `ml/golden/` (the hand-written test set). `pardadb/resources/name_words.txt` is built from those name lists.
- **Open-source libraries** (versions pinned in `uv.lock` and `ml/requirements.lock.txt`): sqlglot (MIT), PyMySQL (MIT), pg8000 (BSD-3-Clause), rich (MIT), textual (MIT); for the AI add-on gliner (Apache-2.0), PyTorch (BSD-3-Clause), transformers (Apache-2.0), safetensors (Apache-2.0); for development pytest (MIT) and ruff (MIT).
- **Model:** GLiNER-PII edge by Knowledgator (Apache-2.0).
- **Landing-page assets:** listed with source and licence in `FULL_CODE/pardadb/dashboard/assets/LICENSES.md`.
- **Data:** the Saffron Bank pack is synthetic data from the TalentAP Mega Hackathon 2026 organisers, CC BY 4.0. It is not committed.
- No templates or starter kits were used.

## Limitations

- **Free-text detection is not perfect.** Rules find every identifier with a format, and every written address in the organiser data, but a name with no cue in front of it needs the optional AI model. The model sometimes tags typos as names; that over-masks, it never leaks.
- **The inference guard is deliberately strict.** Some harmless queries are refused (for example a correlated `EXISTS` on `customer_id` for Support), and anything it cannot analyse is refused.
- **Aggregates are not yet limited by group size.** An Analyst can count rows per token; the next step is to suppress groups smaller than 10 rows (k-anonymity).
- **The AI add-on's pinned versions of PyTorch and transformers have published advisories.** The fixes need major upgrades that GLiNER does not yet support. Mitigations: the sidecar listens on localhost only, loads only our own safetensors weights, and never runs model output. The core engine's dependencies have no known vulnerabilities (pip-audit).
- **Roles come from `users.csv`;** there is no login. A deployment would take roles from the bank's identity provider.
- **Next:** a wire-protocol proxy so existing SQL clients connect through PardaDB unchanged, policies compiled into database views, and a smaller AI model.

## License

MIT (see `LICENSE`). The problem pack data is CC BY 4.0 and belongs to the organisers.
