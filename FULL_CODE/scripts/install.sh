#!/usr/bin/env bash
# PardaDB installer for Linux and macOS.
#
# One command:
#   curl -fsSL https://raw.githubusercontent.com/deadhearth01/pardadb/main/FULL_CODE/scripts/install.sh | bash
# With options (everything after "--" goes to this script):
#   curl -fsSL .../install.sh | bash -s -- --data /path/to/B1_Smart_Data_Masking_PII --with-ai
# From a clone (installs in place, does not pull):
#   ./FULL_CODE/scripts/install.sh
#
# Safe to re-run: it updates the checkout with a fast-forward pull, re-syncs the
# locked dependencies and never overwrites an existing FULL_CODE/.env.
# It never uses sudo; when a system package is missing it prints the command to run.
#
# The whole script is wrapped in main(), called on the last line. When bash reads
# a script from a pipe, a command that reads stdin could otherwise swallow the rest
# of the script, and a cut-off download could run half an installer.

set -euo pipefail

readonly DEFAULT_SOURCE="https://github.com/deadhearth01/pardadb.git"
readonly UV_INSTALLER_URL="https://astral.sh/uv/install.sh"
readonly TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
readonly TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"
readonly DASHBOARD_URL="http://127.0.0.1:8765"

# A temp file to delete on exit, however the script ends.
CLEANUP_FILE=""
trap 'if [[ -n "$CLEANUP_FILE" ]]; then rm -f -- "$CLEANUP_FILE"; fi' EXIT

# Colours only for a terminal, so logs and CI output stay plain. NO_COLOR is the
# common opt-out convention (https://no-color.org).
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_BOLD=$'\033[1m' C_GREEN=$'\033[32m' C_YELLOW=$'\033[33m' C_RED=$'\033[31m' C_RESET=$'\033[0m'
else
  C_BOLD="" C_GREEN="" C_YELLOW="" C_RED="" C_RESET=""
fi

info() { printf '%s==>%s %s\n' "$C_BOLD" "$C_RESET" "$*"; }
ok() { printf '%s ok%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '%swarning:%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die() {
  printf '%serror:%s %s\n' "$C_RED" "$C_RESET" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Install PardaDB (Linux and macOS).

Usage: install.sh [options]

  --dir PATH           Install location (default: $HOME/pardadb)
  --branch NAME        Git branch to install (default: main)
  --data PATH          Link the problem pack (B1_Smart_Data_Masking_PII) into FULL_CODE/data
  --with-ai            Also set up the optional AI add-on in FULL_CODE/ml/.venv (~1 GB download)
  --start-databases    Start the pack's MySQL and PostgreSQL with Docker (needs --data or FULL_CODE/data)
  --source URL|PATH    Git source to clone (default: the GitHub repository)
  --no-uv              Do not use or install uv; use python3 -m venv and pip instead
  -h, --help           Show this help

Every option can also be set with an environment variable:
  PARDADB_INSTALL_DIR, PARDADB_INSTALL_BRANCH, PARDADB_INSTALL_DATA,
  PARDADB_INSTALL_WITH_AI=1, PARDADB_INSTALL_START_DATABASES=1,
  PARDADB_INSTALL_SOURCE, PARDADB_INSTALL_NO_UV=1
EOF
}

# Prints the package-manager command for a missing tool, because the exact
# command differs per OS and we never run sudo ourselves.
install_hint() {
  local tool="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    case "$tool" in
      git) echo "  xcode-select --install     (or: brew install git)" ;;
      python) echo "  brew install python@3.12   (or download it from https://www.python.org/downloads/)" ;;
      venv) echo "  brew install python@3.12" ;;
    esac
  elif command -v apt-get >/dev/null 2>&1; then
    case "$tool" in
      git) echo "  sudo apt-get install -y git" ;;
      python) echo "  sudo apt-get install -y python3 python3-venv   (Ubuntu 22.04+ / Debian 12+ ship 3.10+)" ;;
      venv) echo "  sudo apt-get install -y python3-venv" ;;
    esac
  elif command -v dnf >/dev/null 2>&1; then
    case "$tool" in
      git) echo "  sudo dnf install -y git" ;;
      python | venv) echo "  sudo dnf install -y python3   (Fedora 36+ / RHEL 9 ship 3.10+; else python3.11)" ;;
    esac
  else
    echo "  Install ${tool} with your system's package manager."
  fi
}

python_ok() {
  command -v python3 >/dev/null 2>&1 &&
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

# uv installs into ~/.local/bin (older releases: ~/.cargo/bin). Those folders may
# not be on PATH in this shell yet, so look there too.
find_uv() {
  local candidate
  for candidate in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Downloads a URL to stdout with whichever of curl or wget exists.
fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "$1"
  else
    return 1
  fi
}

# Installs uv with Astral's official installer. Running a downloaded installer is
# the documented way to get uv; we only do it when uv is missing and --no-uv is off.
install_uv() {
  info "Installing uv (Python package manager) from ${UV_INSTALLER_URL}"
  local script
  script="$(fetch "$UV_INSTALLER_URL")" || return 1
  [[ -n "$script" ]] || return 1
  sh -c "$script" </dev/null || return 1
  find_uv >/dev/null
}

# Clones the repo, or fast-forwards an existing clone. A fast-forward pull never
# rewrites local work: if the user changed tracked files, git refuses and we stop.
clone_or_update() {
  local source="$1" branch="$2" dir="$3"
  if [[ -d "$dir/.git" ]]; then
    info "Updating the existing checkout in $dir (branch $branch)"
    git -C "$dir" fetch --quiet origin "$branch" </dev/null ||
      die "Could not fetch branch '$branch' from origin in $dir."
    if git -C "$dir" rev-parse --verify --quiet "refs/heads/$branch" >/dev/null; then
      git -C "$dir" checkout --quiet "$branch" </dev/null ||
        die "Could not switch $dir to branch '$branch' (uncommitted changes?)."
    else
      git -C "$dir" checkout --quiet -b "$branch" --track "origin/$branch" </dev/null ||
        die "Could not create branch '$branch' in $dir."
    fi
    git -C "$dir" pull --ff-only --quiet origin "$branch" </dev/null ||
      die "Could not fast-forward $dir. Commit or discard your local changes there, or choose another --dir."
  elif [[ -e "$dir" && -n "$(ls -A "$dir" 2>/dev/null)" ]]; then
    die "$dir exists, is not empty and is not a git checkout. Choose another --dir or remove it."
  else
    info "Cloning $source (branch $branch) into $dir"
    mkdir -p "$(dirname "$dir")"
    git clone --quiet --branch "$branch" "$source" "$dir" </dev/null ||
      die "git clone failed. Check the network, the --source and that branch '$branch' exists."
  fi
  ok "Source ready: $(git -C "$dir" log -1 --format='%h %s')"
}

# Installs the main package. uv sync --frozen installs exactly what uv.lock lists
# (the reviewed versions) and fails instead of silently re-resolving.
install_main() {
  local code="$1" uv_bin="$2"
  if [[ -n "$uv_bin" ]]; then
    info "Installing PardaDB and its locked dependencies with uv"
    (cd "$code" && "$uv_bin" sync --frozen --extra dev </dev/null) || die "uv sync failed (see the output above)."
  else
    info "Installing PardaDB with python3 -m venv and pip"
    python3 -m venv "$code/.venv" </dev/null || {
      install_hint venv >&2
      die "python3 -m venv failed; install the venv module with the command above and re-run."
    }
    "$code/.venv/bin/python" -m pip install --quiet --upgrade pip </dev/null || die "pip upgrade failed."
    "$code/.venv/bin/python" -m pip install --quiet -e "${code}[dev]" </dev/null || die "pip install failed."
  fi
  ok "PardaDB installed in $code/.venv"
}

# Creates .env from the template once. It is never overwritten, because after the
# first run it holds the user's own passwords and keys.
create_env() {
  local code="$1"
  if [[ -f "$code/.env" ]]; then
    ok "Keeping the existing $code/.env"
  elif [[ -f "$code/.env.example" ]]; then
    cp "$code/.env.example" "$code/.env"
    chmod 600 "$code/.env"
    ok "Created $code/.env from .env.example (put the database passwords from the pack in it)"
  else
    warn "No .env.example found in $code; skipping .env."
  fi
}

# Links the problem pack into FULL_CODE/data, where PardaDB looks by default.
# A real folder already there is left alone so we never hide or delete user data.
link_data() {
  local code="$1" data="$2" target
  [[ -d "$data" ]] || die "--data: '$data' is not a folder."
  target="$(cd "$data" && pwd -P)"
  [[ -f "$target/sql/sqlite/saffron_bank.db" ]] ||
    warn "$target has no sql/sqlite/saffron_bank.db; is this the B1_Smart_Data_Masking_PII folder?"
  if [[ -L "$code/data" || ! -e "$code/data" ]]; then
    ln -sfn "$target" "$code/data"
    ok "Linked $code/data -> $target"
  else
    warn "$code/data already exists as a real folder; leaving it as is (not linking $target)."
  fi
}

# Starts the pack's MySQL and PostgreSQL. We cd into the resolved real folder so
# Docker Compose derives the same project name as a manual "docker compose up" in
# the pack, and reuses those containers instead of starting duplicates.
start_databases() {
  local code="$1" pack
  if [[ ! -f "$code/data/docker-compose.yml" ]]; then
    warn "--start-databases: no docker-compose.yml in $code/data (use --data). Skipping."
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    warn "--start-databases: Docker is not installed (https://docs.docker.com/get-docker/). Skipping."
    return 0
  fi
  pack="$(cd "$code/data" && pwd -P)"
  info "Starting MySQL and PostgreSQL with Docker (first start loads the data, 1-3 minutes)"
  if docker compose version >/dev/null 2>&1; then
    (cd "$pack" && docker compose up -d </dev/null) || warn "docker compose up failed; PardaDB will use SQLite only."
  elif command -v docker-compose >/dev/null 2>&1; then
    (cd "$pack" && docker-compose up -d </dev/null) || warn "docker-compose up failed; PardaDB will use SQLite only."
  else
    warn "Docker Compose is not available; skipping. PardaDB will use SQLite only."
  fi
}

# Sets up the optional AI add-on in its own venv (it needs PyTorch, which the main
# app does not). ml/requirements.lock.txt pins the CUDA 12.8 build of torch; without
# an NVIDIA GPU we install the same torch version as a CPU build and drop the CUDA
# libraries, which saves several GB.
install_ai() {
  local code="$1" uv_bin="$2" ml="$1/ml" lock reqs index=""
  lock="$ml/requirements.lock.txt"
  [[ -f "$lock" ]] || die "--with-ai: $lock not found."
  warn "The AI add-on downloads up to about 1 GB (PyTorch and the model libraries); more with CUDA."

  reqs="$(mktemp "${TMPDIR:-/tmp}/pardadb-ml-reqs.XXXXXX")"
  CLEANUP_FILE="$reqs"
  if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    info "NVIDIA GPU found: installing the CUDA build of PyTorch"
    cp "$lock" "$reqs"
    index="$TORCH_CUDA_INDEX"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    # macOS wheels on PyPI are already CPU (and Apple GPU) builds; there is no +cpu tag.
    info "macOS: installing the standard PyTorch build"
    sed -E -e '/^(nvidia-|triton==)/d' -e 's/^torch==([0-9.]+)\+.*/torch==\1/' "$lock" >"$reqs"
  else
    info "No NVIDIA GPU: installing the CPU build of PyTorch"
    sed -E -e '/^(nvidia-|triton==)/d' -e 's/^torch==([0-9.]+)\+.*/torch==\1+cpu/' "$lock" >"$reqs"
    index="$TORCH_CPU_INDEX"
  fi

  # Extra index arguments as an array; the ${a[@]+...} form keeps bash 3.2
  # (the macOS default) happy with an empty array under set -u.
  local -a index_args=()
  if [[ -n "$index" ]]; then
    index_args=(--extra-index-url "$index")
  fi
  if [[ -n "$uv_bin" ]]; then
    (cd "$ml" && "$uv_bin" venv --quiet --allow-existing --python ">=3.10" .venv </dev/null) || die "uv venv failed in $ml."
    # unsafe-best-match lets the lock's other pins come from PyPI even though the
    # PyTorch index also lists some of those package names.
    "$uv_bin" pip install --python "$ml/.venv/bin/python" --index-strategy unsafe-best-match \
      -r "$reqs" ${index_args[@]+"${index_args[@]}"} </dev/null ||
      die "Installing the AI add-on failed (see the output above)."
  else
    python3 -m venv "$ml/.venv" </dev/null || die "python3 -m venv failed in $ml."
    "$ml/.venv/bin/python" -m pip install --quiet -r "$reqs" ${index_args[@]+"${index_args[@]}"} </dev/null ||
      die "Installing the AI add-on failed (see the output above)."
  fi
  rm -f -- "$reqs"
  ok "AI add-on installed in $ml/.venv (train or copy a model into ml/models, then start with: pardadb --ml)"
}

print_next_steps() {
  local code="$1" uv_bin="$2" run test
  if [[ -n "$uv_bin" ]]; then
    # Show the full path if uv is not on PATH yet (fresh install, old shell).
    local uv_cmd="uv"
    command -v uv >/dev/null 2>&1 || uv_cmd="$uv_bin"
    run="$uv_cmd run pardadb"
    test="$uv_cmd run --extra dev pytest"
  else
    run=".venv/bin/pardadb"
    test=".venv/bin/python -m pytest"
  fi
  printf '\n%s+-- PardaDB is installed ------------------------------------------%s\n' "$C_GREEN" "$C_RESET"
  if [[ -e "$code/data/sql/sqlite/saffron_bank.db" ]]; then
    printf '| 1. Problem pack: found in %s/data\n' "$code"
  else
    printf '| 1. Problem pack: copy or link B1_Smart_Data_Masking_PII to\n'
    printf '|      %s/data\n' "$code"
    printf '|    (re-run with --data /path/to/pack) or set PARDADB_DATASET in .env\n'
  fi
  printf '| 2. Database passwords: edit %s/.env\n' "$code"
  printf '|    (only needed for MySQL / PostgreSQL; SQLite needs nothing)\n'
  printf '| 3. Start:  cd "%s" && %s\n' "$code" "$run"
  printf '|    Dashboard: %s   (terminal UI opens in this window)\n' "$DASHBOARD_URL"
  printf '| 4. Scan only:  cd "%s" && %s scan --db sqlite\n' "$code" "$run"
  printf '| 5. Tests:  cd "%s" && %s\n' "$code" "$test"
  printf '%s+-----------------------------------------------------------------%s\n' "$C_GREEN" "$C_RESET"
}

main() {
  local dir="${PARDADB_INSTALL_DIR:-$HOME/pardadb}"
  local branch="${PARDADB_INSTALL_BRANCH:-main}"
  local data="${PARDADB_INSTALL_DATA:-}"
  local source="${PARDADB_INSTALL_SOURCE:-}"
  local with_ai="${PARDADB_INSTALL_WITH_AI:-0}"
  local start_dbs="${PARDADB_INSTALL_START_DATABASES:-0}"
  local no_uv="${PARDADB_INSTALL_NO_UV:-0}"
  local dir_given=0
  [[ -n "${PARDADB_INSTALL_DIR:-}" ]] && dir_given=1

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dir | --branch | --data | --source)
        [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value (see --help)."
        case "$1" in
          --dir) dir="$2" dir_given=1 ;;
          --branch) branch="$2" ;;
          --data) data="$2" ;;
          --source) source="$2" ;;
        esac
        shift 2
        ;;
      --with-ai) with_ai=1 && shift ;;
      --start-databases) start_dbs=1 && shift ;;
      --no-uv) no_uv=1 && shift ;;
      -h | --help) usage && return 0 ;;
      *) die "Unknown option '$1' (see --help)." ;;
    esac
  done

  # Work with absolute paths, so the summary and links stay right after any cd.
  [[ "$dir" == /* ]] || dir="$PWD/$dir"
  if [[ -n "$data" && "$data" != /* ]]; then
    data="$PWD/$data"
  fi

  # Check --data before any download, so a typo fails in a second, not after the install.
  if [[ -n "$data" && ! -d "$data" ]]; then
    die "--data: '$data' is not a folder."
  fi

  info "PardaDB installer ($(uname -s))"

  # When run as a file from inside a clone, with no --dir or --source, install in
  # place and leave the user's checkout alone. When piped, BASH_SOURCE is empty.
  local self repo=""
  self="${BASH_SOURCE[0]:-}"
  if [[ -n "$self" && -f "$self" && $dir_given -eq 0 && -z "$source" ]]; then
    repo="$(cd "$(dirname "$self")/../.." && pwd -P)"
    [[ -f "$repo/FULL_CODE/pyproject.toml" ]] || repo=""
  fi

  command -v git >/dev/null 2>&1 || {
    printf 'git is required. Install it with:\n' >&2
    install_hint git >&2
    die "git not found."
  }

  # Python 3.10+ is needed. If the system Python is missing or older, uv can
  # still download a suitable one, so we only stop when uv is not an option.
  local have_python=1
  if ! python_ok; then
    have_python=0
    # With --no-uv nothing can provide Python, so stop before cloning anything.
    if [[ "$no_uv" == "1" ]]; then
      printf 'Python 3.10+ is required with --no-uv. Install it with:\n' >&2
      install_hint python >&2
      die "No Python 3.10+ found as python3."
    fi
    warn "Python 3.10 or newer was not found as python3; uv will provide one if it can."
  fi

  if [[ -n "$repo" ]]; then
    dir="$repo"
    info "Using this checkout in place: $dir (not pulling)"
  else
    clone_or_update "${source:-$DEFAULT_SOURCE}" "$branch" "$dir"
  fi
  local code="$dir/FULL_CODE"
  [[ -f "$code/pyproject.toml" ]] || die "$code/pyproject.toml not found; is this the PardaDB repository?"

  local uv_bin=""
  if [[ "$no_uv" != "1" ]]; then
    if uv_bin="$(find_uv)"; then
      ok "uv found: $uv_bin"
    elif install_uv; then
      uv_bin="$(find_uv)"
      ok "uv installed: $uv_bin"
    else
      uv_bin=""
      warn "Could not install uv; falling back to python3 -m venv and pip."
    fi
  fi
  if [[ -z "$uv_bin" && $have_python -eq 0 ]]; then
    printf 'Python 3.10+ is required. Install it with:\n' >&2
    install_hint python >&2
    die "No Python 3.10+ and no uv to provide one."
  fi

  install_main "$code" "$uv_bin"
  create_env "$code"
  if [[ -n "$data" ]]; then
    link_data "$code" "$data"
  fi
  if [[ "$start_dbs" == "1" ]]; then
    start_databases "$code"
  fi
  if [[ "$with_ai" == "1" ]]; then
    install_ai "$code" "$uv_bin"
  fi
  print_next_steps "$code" "$uv_bin"
}

main "$@"
