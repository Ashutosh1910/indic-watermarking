#!/usr/bin/env bash
# Sharanga: check the setup on a GPU node on its own (precheck.sh --tests: GPUs, internet, a short watermarked
# generation with the study model from the offline cache, and the CPU test suites). Not needed before the
# pilot: the first job of every submit.sh chain runs the same check (without the tests) before it starts.
#
#   sbatch promptset/job/check_sharanga.sh                                     # H200 partition
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
bash promptset/job/precheck.sh --tests; FAIL=$?
if [[ $FAIL -eq 0 ]]; then
  echo "== ALL OK. Next: bash promptset/job/submit.sh --dump dump/pilot"
else
  echo "== SOME CHECKS FAILED (see above)"
fi
exit $FAIL
