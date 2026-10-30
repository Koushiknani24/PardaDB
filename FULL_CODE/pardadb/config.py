"""Paths, connection settings and feature flags, all overridable by environment variables.

Secrets (database passwords, the HMAC key) are never written here: they come from the
environment or from FULL_CODE/.env (see .env.example), which is not committed.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # FULL_CODE/


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env without overriding variables already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(REPO / ".env")


# The problem pack (sql/, data/users.csv) goes in FULL_CODE/data/, which is git-ignored.
def env_path(name: str, default: Path) -> Path:
    """A path setting from the environment; relative values are taken from FULL_CODE/, not the shell's folder."""
    raw = os.environ.get(name)
    path = Path(raw).expanduser() if raw else default
    return (path if path.is_absolute() else REPO / path).resolve()


DATASET = env_path("PARDADB_DATASET", REPO / "data")
SQLITE_PATH = env_path("PARDADB_SQLITE", DATASET / "sql" / "sqlite" / "saffron_bank.db")
USERS_CSV = env_path("PARDADB_USERS", DATASET / "data" / "users.csv")
STATE_DIR = env_path("PARDADB_STATE", REPO / ".state")
POLICY_PATH = env_path("PARDADB_POLICY", Path(__file__).resolve().parent / "policy.json")

MYSQL = dict(
    host=os.environ.get("PARDADB_MYSQL_HOST", "127.0.0.1"),
    port=int(os.environ.get("PARDADB_MYSQL_PORT", "53306")),
    user=os.environ.get("PARDADB_MYSQL_USER", "saffron"),
    password=os.environ.get("PARDADB_MYSQL_PASSWORD", ""),
    database=os.environ.get("PARDADB_MYSQL_DB", "saffron_bank"),
)
POSTGRES = dict(
    host=os.environ.get("PARDADB_PG_HOST", "127.0.0.1"),
    port=int(os.environ.get("PARDADB_PG_PORT", "55432")),
    user=os.environ.get("PARDADB_PG_USER", "saffron"),
    password=os.environ.get("PARDADB_PG_PASSWORD", ""),
    database=os.environ.get("PARDADB_PG_DB", "saffron_bank"),
)

SAMPLE_SIZE = int(os.environ.get("PARDADB_SAMPLE", "200"))
MAX_ROWS = int(os.environ.get("PARDADB_MAX_ROWS", "1000"))
K_THRESHOLD = int(os.environ.get("PARDADB_K", "10"))
# Host names the web server answers to besides localhost (only needed with `pardadb start --host`).
EXTRA_HOSTS = tuple(h.strip() for h in os.environ.get("PARDADB_ALLOWED_HOSTS", "").split(",") if h.strip())
DB_TIMEOUT_S = int(os.environ.get("PARDADB_DB_TIMEOUT", "10"))

# Optional AI add-on (fine-tuned GLiNER served by ml/serve.py on localhost)
ML_DIR = env_path("PARDADB_ML_DIR", REPO / "ml")
ML_URL = os.environ.get("PARDADB_ML_URL", "http://127.0.0.1:8766").rstrip("/")
if not ML_URL.startswith(("http://", "https://")):
    raise ValueError(f"PARDADB_ML_URL must be an http(s) URL, got {ML_URL!r}")
ML_THRESHOLD = float(os.environ.get("PARDADB_ML_THRESHOLD", "0.5"))
ML_TIMEOUT_S = float(os.environ.get("PARDADB_ML_TIMEOUT", "10"))

# Audit log: rotate the active file past this size; give up waiting for the cross-process lock after this long.
AUDIT_MAX_BYTES = int(os.environ.get("PARDADB_AUDIT_MAX_BYTES", "50000000"))
AUDIT_LOCK_TIMEOUT_S = float(os.environ.get("PARDADB_AUDIT_LOCK_TIMEOUT", "5"))


def state_path(name: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / name
