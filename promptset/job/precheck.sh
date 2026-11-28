#!/usr/bin/env bash
# Setup check on a GPU node: GPUs and driver, internet (informational), a short watermarked generation with
# the study model from the offline cache (gpu_check.py) and, with --tests, the CPU test suites.
# Runs inside a job, with the venv active (run_study.sh calls it before the first job of a chain,
# check_sharanga.sh on its own). Exit 0 = all OK.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
TESTS=""; [[ "${1:-}" == "--tests" ]] && TESTS=1
FAIL=0
echo "== precheck: GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || { echo "nvidia-smi failed"; FAIL=1; }
python -c "import torch, sys; n = torch.cuda.device_count(); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', n); sys.exit(n == 0)" || {
  echo "precheck: torch sees no GPU (driver too old for the torch wheel? rerun setup with TORCH_INDEX=https://download.pytorch.org/whl/cu124)"; FAIL=1; }

echo "== precheck: internet from this node"
if curl -sS -m 15 -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/Qwen/Qwen3-14B; then
  echo "internet: yes (the job still runs offline from the cache)"
else
  echo "internet: no (fine: the job runs offline from ${HF_HOME:-the HF cache})"
fi

if [[ $FAIL -eq 0 ]]; then
  echo "== precheck: short watermarked generation (offline, model from the cache)"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python promptset/job/gpu_check.py || FAIL=1
fi

if [[ -n "$TESTS" ]]; then
  echo "== precheck: CPU tests"
  python -m pytest promptset/tests -q -p no:cacheprovider || FAIL=1
  python -m pytest synthid/tests -q -p no:cacheprovider || FAIL=1
fi

if [[ $FAIL -eq 0 ]]; then echo "== PRECHECK OK"; else echo "== PRECHECK FAILED (see above)"; fi
exit $FAIL
