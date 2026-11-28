#!/usr/bin/env bash
# Sharanga: check the setup on a GPU node before the pilot. Runs the CPU test suites, a short watermarked
# generation with the study model (offline, from the cache setup filled), and tests internet access.
#
#   sbatch promptset/job/check_sharanga.sh                                     # H100 partition
#   sbatch -p gpu_a100_8 --qos=qos_gpu_a100 promptset/job/check_sharanga.sh    # another partition
#
#SBATCH -p gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=wm-check
#SBATCH --output=wm-check.%j.out
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
# shellcheck disable=SC1091
source promptset/job/sharanga.env
wm_refresh "$REPO"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi failed"

FAIL=0
echo "== internet from this compute node"
if curl -sS -m 15 -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/Qwen/Qwen3-14B; then
  echo "internet: yes (the job still runs offline from the cache)"
else
  echo "internet: no (fine: the job runs offline from $HF_HOME)"
fi

echo "== GPU check (offline, model from the cache)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python promptset/job/gpu_check.py || FAIL=1

echo "== CPU tests"
python -m pytest promptset/tests -q -p no:cacheprovider || FAIL=1
python -m pytest synthid/tests -q -p no:cacheprovider || FAIL=1

if [[ $FAIL -eq 0 ]]; then
  echo "== ALL OK. Next: bash promptset/job/submit.sh --dump dump/pilot"
else
  echo "== SOME CHECKS FAILED (see above)"
fi
exit $FAIL
