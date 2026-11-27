#!/usr/bin/env bash
# One-time setup of a GPU server for the study (promptset/JOB.md "Quick start").
#
#   bash promptset/job/setup_gpu.sh              # venv + packages + model weights + CPU tests + checks
#   SKIP_TESTS=1 bash promptset/job/setup_gpu.sh # without the CPU tests (about 10 minutes)
#
# Environment (all optional):
#   VENV       virtualenv path (default .venv in the repo)
#   PYTHON     interpreter for the venv (default python3; needs 3.10+)
#   MODEL      model to download (default: model.model_id of promptset/config/generation.yaml)
#   HF_HOME    Hugging Face cache, on a disk with ~35 GB free for Qwen3-14B
#   TORCH_INDEX  e.g. https://download.pytorch.org/whl/cu124 if the default torch wheel does not match the driver
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
VENV="${VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), f"python 3.10+ needed, got {sys.version}"'
[[ -d "$VENV" ]] || "$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
if [[ -n "${TORCH_INDEX:-}" ]]; then python -m pip install -q torch --index-url "$TORCH_INDEX"; fi
python -m pip install -q -r promptset/requirements.txt

MODEL="${MODEL:-$(python -c 'import yaml; print(yaml.safe_load(open("promptset/config/generation.yaml"))["model"]["model_id"])')}"
echo "== downloading $MODEL (once; the job reuses the cache)"
python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*"]))
PY

echo "== checks"
python - <<'PY'
import shutil, torch, transformers
from transformers import SynthIDTextWatermarkingConfig  # noqa: F401  (needs transformers >= 4.46)
print("torch", torch.__version__, "transformers", transformers.__version__)
n = torch.cuda.device_count()
print("cuda", torch.cuda.is_available(), "gpus", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gb = p.total_memory / 2**30
    print(f"  gpu {i}: {p.name} {gb:.0f} GB" + ("" if gb >= 70 else "  (under 80 GB: lower generate.token_budget, see generation.yaml)"))
if n == 0:
    print("WARNING: no CUDA GPU visible")
print("free disk in repo:", shutil.disk_usage(".").free // 2**30, "GB")
PY
for d in data/HindiSamay data/Samalochan data/Medium; do
  [[ -d "$d" ]] && echo "  corpus $d: $(ls "$d" | wc -l) files" || echo "WARNING: $d missing (the edit prompts and human pool need it)"
done

if [[ -z "${SKIP_TESTS:-}" ]]; then
  echo "== CPU tests (tiny random Qwen3)"
  python -m pytest promptset/tests -q -p no:cacheprovider
  python -m pytest synthid/tests -q -p no:cacheprovider
fi
echo "== done. Next: python promptset/scripts/run_study.py --plan --dump dump/pilot   (see promptset/JOB.md)"
