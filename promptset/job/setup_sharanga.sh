#!/usr/bin/env bash
# One-time setup on a Sharanga LOGIN node: python, venv, packages and the model weights, all under
# /scratch/$USER/wm (sharanga.env). Downloads only: the CPU tests and the GPU check run as a job afterwards
# (the precheck at the start of the first study job), because code must not run on login nodes.
#
#   cd /scratch/$USER/wm && git clone https://github.com/Ashutosh1910/indic-watermarking && cd indic-watermarking
#   bash promptset/job/setup_sharanga.sh
#
# Environment (optional): PYTHON (a python 3.10+ to use), TORCH_INDEX (e.g. https://download.pytorch.org/whl/cu124
# if the GPU driver is older than the default torch wheel's CUDA), MODEL, anything in sharanga.env.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source promptset/job/sharanga.env
mkdir -p "$WM_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$WM_RESULTS" "$HOME/wm"
chmod 700 "$HOME/wm"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WM_ROOT/cache/conda-pkgs}"   # not ~/.conda: $HOME is nearly full
case "$REPO/" in
  "$WM_ROOT"/*) ;;
  *) echo "WARNING: the repo is at $REPO, outside $WM_ROOT. \$HOME has a 40 GiB quota: clone it under $WM_ROOT." ;;
esac

py_ok() { [[ -n "$1" ]] && command -v "$1" >/dev/null 2>&1 && "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }
PY=""
for c in "${PYTHON:-}" "$WM_ROOT/py311/bin/python" python3.12 python3.11 python3.10 python3; do
  if py_ok "$c"; then PY="$c"; break; fi
done
if [[ -z "$PY" ]]; then
  echo "== no python 3.10+ on PATH: making one with conda in $WM_ROOT/py311"
  if ! command -v conda >/dev/null 2>&1; then
    for m in miniconda3 anaconda3; do
      { module load "$m" 2>/dev/null || spack load "$m" 2>/dev/null; } && break || true
    done
  fi
  command -v conda >/dev/null 2>&1 || { echo "conda not found: 'module spider conda' and load it, or set PYTHON=/path/to/python3.11"; exit 1; }
  conda create -y -q -p "$WM_ROOT/py311" python=3.11
  PY="$WM_ROOT/py311/bin/python"
fi
echo "== python: $PY ($("$PY" --version))"
export PYTHON="$PY"

SKIP_TESTS=1 bash promptset/job/setup_gpu.sh     # venv at $VENV, packages, weights into $HF_HOME (no tests here)
wm_refresh "$REPO"
echo "== disk: $(du -sh "$WM_ROOT" 2>/dev/null | cut -f1) in $WM_ROOT; home quota: run 'usertools quota info'"
echo "== next: WM_TIME=03:00:00 bash promptset/job/submit.sh --dump dump/pilot   (GPU check, then the pilot; log logs/wm-study.<id>.out)"
