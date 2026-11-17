#!/usr/bin/env bash
# Creates a self-contained virtual env in ./.venv and installs everything into it.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Done. Activate with:  source .venv/bin/activate   (optional; run_all.sh uses .venv directly)"
