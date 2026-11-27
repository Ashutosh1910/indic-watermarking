#!/usr/bin/env bash
# GPU job: the whole study into one dump (promptset/JOB.md). Works three ways:
#
#   sbatch promptset/job/run_study.sh                         # SLURM (the #SBATCH lines below)
#   nohup bash promptset/job/run_study.sh > job.out 2>&1 &    # plain server
#   bash promptset/job/run_study.sh --dump dump/main --set filters='[split=pilot,main]'   # extra args go to run_study.py
#
# Re-submit the same command to resume after a crash, preemption or time limit: finished work is skipped.
# Environment (all optional):
#   RUNCFG    another run config (wins over study.yaml's run_config). Not needed for keys: the job uses
#             ~/wm/keys.private.yaml (created on the first run, see study.yaml keys)
#   STUDY     job config (default promptset/config/study.yaml)
#   VENV      virtualenv to activate (default .venv in the repo, if present)
#   HF_HOME   Hugging Face cache (model weights)
#
#SBATCH --job-name=wm-study
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=wm-study.%j.out
#SBATCH --signal=B:TERM@300
#SBATCH --requeue
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
VENV="${VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
STUDY="${STUDY:-promptset/config/study.yaml}"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Fewer CUDA allocator fragmentation OOMs on long Hindi batches.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[$(date -u +%FT%TZ)] host $(hostname) repo $REPO commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null || true

# Run in the background and forward TERM/INT, so SLURM's warning signal reaches run_study.py, which stops
# its generation processes cleanly (rows written so far are kept) and exits 130.
python promptset/scripts/run_study.py --study "$STUDY" "$@" &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null' TERM INT
set +e
wait "$PID"; CODE=$?
if [[ $CODE -gt 128 ]]; then wait "$PID"; CODE=$?; fi   # wait again after a trapped signal
set -e
echo "[$(date -u +%FT%TZ)] run_study.py exit $CODE (0 = done, 1 = done with failures, see dump/failures; 130 = interrupted, resubmit to resume)"
exit "$CODE"
