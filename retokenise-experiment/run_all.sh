#!/usr/bin/env bash
# Fetch human Hindi text from a HF dataset (if texts/hi.txt is missing) and measure drift.
# Change the dataset with:  cd src && ../.venv/bin/python fetch_texts.py --help
#   ./run_all.sh --quick                     smoke test (10 passages)
#   ./run_all.sh                             100 passages of ~200 tokens
#   ./run_all.sh --texts /path/to/IndicCorp_hi_sample.txt --out outputs/indiccorp
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first"; exit 1; }
HERE="$PWD"

ARGS=(); TEXTS=""; OUT="outputs/hi"
while [ $# -gt 0 ]; do
  case "$1" in
    --texts) TEXTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --quick) OUT="outputs/quick"; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
abs() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }
OUT="$(abs "$OUT")"
if [ -z "$TEXTS" ]; then
  TEXTS="$HERE/texts/hi.txt"
  [ -s "$TEXTS" ] || (cd src && ../.venv/bin/python fetch_texts.py --out "$TEXTS")
else
  TEXTS="$(abs "$TEXTS")"
fi
cd src
../.venv/bin/python drift.py --texts "$TEXTS" --out "$OUT" ${ARGS[@]+"${ARGS[@]}"}
