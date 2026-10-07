#!/bin/bash
# Start the PP/PE FTIR web service (default: localhost:8000). Usage: ./scripts_serve.sh [host] [port]
cd "$(dirname "$0")"
exec .venv/bin/uvicorn webapp.app:app --host "${1:-127.0.0.1}" --port "${2:-8000}"
