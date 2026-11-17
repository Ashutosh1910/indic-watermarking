#!/usr/bin/env bash
# Fetch human text from a HF dataset (if texts/<lang>.txt is missing) and measure drift.
# Change the dataset with:  cd src && ../.venv/bin/python fetch_texts.py --help
#   ./run_all.sh --quick                     smoke test (10 passages), Hindi
#   ./run_all.sh                             100 passages of ~200 tokens, Hindi
#   ./run_all.sh --lang en                   same for English (English side of the same corpus)
#   ./run_all.sh --lang both                 Hindi and English, then outputs/compare.md side by side
#   ./run_all.sh --texts /path/to/IndicCorp_hi_sample.txt --out outputs/indiccorp
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first"; exit 1; }
HERE="$PWD"

ARGS=(); TEXTS=""; OUT=""; LANG_ARG="hi"; QUICK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --texts) TEXTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --lang) LANG_ARG="$2"; shift 2 ;;
    --quick) QUICK=1; ARGS+=("$1"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
abs() { case "$1" in /*) echo "$1" ;; *) echo "$HERE/$1" ;; esac; }

run_one() {  # $1 = lang, $2 = texts (may be empty), $3 = out dir (may be empty)
  local lang="$1" texts="$2" out="$3"
  if [ -z "$out" ]; then
    if [ -n "$QUICK" ]; then out="outputs/quick-$lang"; else out="outputs/$lang"; fi
  fi
  out="$(abs "$out")"
  if [ -z "$texts" ]; then
    texts="$HERE/texts/$lang.txt"
    [ -s "$texts" ] || (cd src && ../.venv/bin/python fetch_texts.py --lang "$lang" --out "$texts")
  else
    texts="$(abs "$texts")"
  fi
  (cd src && ../.venv/bin/python drift.py --lang "$lang" --texts "$texts" --out "$out" ${ARGS[@]+"${ARGS[@]}"})
}

if [ "$LANG_ARG" = "both" ]; then
  [ -z "$TEXTS" ] && [ -z "$OUT" ] || { echo "--lang both uses texts/<lang>.txt and outputs/<lang>; drop --texts/--out"; exit 1; }
  run_one hi "" ""
  run_one en "" ""
  if [ -n "$QUICK" ]; then HI="outputs/quick-hi"; EN="outputs/quick-en"; CMP="outputs/quick-compare.md"
  else HI="outputs/hi"; EN="outputs/en"; CMP="outputs/compare.md"; fi
  (cd src && ../.venv/bin/python compare.py --a "$HERE/$HI" --b "$HERE/$EN" --out "$HERE/$CMP")
else
  run_one "$LANG_ARG" "$TEXTS" "$OUT"
fi
