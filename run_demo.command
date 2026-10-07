#!/bin/bash
# Start the PP/PE FTIR web demo on this computer (macOS: double-click, or ./run_demo.command).
# First run creates a local .venv and installs requirements-serve.txt (~1-2 min, needs internet).
cd "$(dirname "$0")"
REQ=requirements-serve.txt; [ -f "$REQ" ] || REQ=requirements.txt
PY=""
for p in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then PY="$p"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10-3.13 not found. Install one, e.g.:  brew install python@3.12   (or from python.org)"; read -p "Press Enter to close"; exit 1
fi
if [ ! -d .venv ]; then
  echo "Creating .venv with $($PY --version) and installing $REQ ..."
  "$PY" -m venv .venv && .venv/bin/pip install --quiet --upgrade pip && .venv/bin/pip install --quiet -r "$REQ" || { echo "Install failed"; read -p "Press Enter to close"; exit 1; }
fi
PORT="${PORT:-8000}"
echo "Starting on http://127.0.0.1:$PORT   (Ctrl+C to stop)"
( sleep 3; (open "http://127.0.0.1:$PORT" || xdg-open "http://127.0.0.1:$PORT") >/dev/null 2>&1 ) &
exec .venv/bin/uvicorn webapp.app:app --host 127.0.0.1 --port "$PORT"
