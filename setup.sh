#!/usr/bin/env bash
# One-step setup for macOS / Linux:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

step() { printf '\n==> %s\n' "$1"; }

step "Checking Python (3.12 or newer)"
PY=""
for candidate in python3.13 python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 \
     && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.12 or newer was not found. Install it and re-run."; exit 1; }

step "Creating virtual environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
VENV_PY=".venv/bin/python"

step "Installing dependencies"
"$VENV_PY" -m pip install --upgrade pip --quiet
"$VENV_PY" -m scripts.setup_env deps | tr '\n' '\0' | xargs -0 "$VENV_PY" -m pip install

step "Configuring your API keys"
"$VENV_PY" -m scripts.setup_env env

step "Applying database migrations"
"$VENV_PY" -m alembic upgrade head

step "Seeding the demo business"
"$VENV_PY" -m scripts.seed_demo

if [ -f frontend/package.json ] && command -v npm >/dev/null 2>&1; then
  step "Installing dashboard packages"
  (cd frontend && npm install)
fi

printf '\nDone. Start the server with:\n  source .venv/bin/activate\n  python -m backend.app.serve\nThen open http://127.0.0.1:8000\n'
