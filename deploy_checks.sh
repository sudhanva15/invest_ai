#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PIP_BIN="${PIP_BIN:-pip}"
PY_BIN=".venv/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  echo "[deploy_checks] Missing .venv or python executable at $PY_BIN"
  echo "Create it with: python -m venv .venv && source .venv/bin/activate"
  exit 1
fi

echo "[deploy_checks] Installing dependencies via $PIP_BIN -r requirements.txt"
$PIP_BIN install -r requirements.txt

echo "[deploy_checks] Running test suite"
PYTHONPATH=. "$PY_BIN" -m pytest tests/ -q

echo "[deploy_checks] Importing Streamlit entrypoint"
PYTHONPATH=. "$PY_BIN" -c "import ui.streamlit_app; print('streamlit_app import OK')"

echo "✅ deploy_checks complete. Push with 'git push origin main' and rerun your Streamlit/Render deploy."
