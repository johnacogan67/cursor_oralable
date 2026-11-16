#!/usr/bin/env bash
# Restore the tested Python 3.13 environment and install the local package.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: ./scripts/bootstrap_python.sh

Creates (or reuses) .venv, installs the fully pinned requirements.lock snapshot,
installs this repository in editable mode, and runs the checkout verifier.
The lock was verified on macOS/Python 3.13. Network access is required when the
required wheels are not already available locally.

Environment overrides:
  PYTHON_BIN=/path/to/python3.13
  VENV_DIR=/path/to/venv
USAGE
  exit 0
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 13):
    raise SystemExit(f"ERROR: Python 3.13 is required by requirements.lock; found {sys.version.split()[0]}")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --require-virtualenv -r "$ROOT/requirements.lock"
"$VENV_DIR/bin/python" -m pip install --no-build-isolation --no-deps -e "$ROOT"
"$VENV_DIR/bin/python" "$ROOT/scripts/verify_research_checkout.py"
