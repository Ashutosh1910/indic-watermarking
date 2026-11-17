#!/usr/bin/env bash
# Creates ./.venv and installs requirements.txt into it.
# Uses uv when available (fast, and can fetch Python 3.13 if needed); otherwise python3 -m venv + pip.
# Model weights are NOT downloaded here: they download lazily on first use into the standard
# Hugging Face cache (~/.cache/huggingface, or $HF_HOME if you have set it).
set -euo pipefail
cd "$(dirname "$0")"
PYVER="${PYVER:-3.13}"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYVER" .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  PY="${PYTHON:-python3}"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| mps', torch.backends.mps.is_available(), '| cuda', torch.cuda.is_available())"
echo "Done. Try:  .venv/bin/python -m demo generate --model Qwen/Qwen3-0.6B --prompt 'Write a short story about a lighthouse.'"
