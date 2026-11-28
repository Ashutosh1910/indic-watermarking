#!/usr/bin/env bash
# GPU job: the whole study into one dump (promptset/JOB.md). Works three ways:
#
#   bash promptset/job/submit.sh --dump dump/pilot            # Sharanga: partition, QoS, GPUs, resubmission
#   sbatch promptset/job/run_study.sh                         # other SLURM clusters (the #SBATCH lines below)
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
#   WM_ENV    file to source first (submit.sh sets promptset/job/sharanga.env: paths on /scratch, offline mode)
#   WM_MAX_CHAIN  resubmit after a time limit (exit 130) at most this many times (needs WM_ENV's submit.sh setup)
#
#SBATCH --job-name=wm-study
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:50:00
#SBATCH --output=wm-study.%j.out
#SBATCH --signal=B:TERM@900
#SBATCH --requeue
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
if [[ -n "${WM_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "$WM_ENV"
  wm_refresh "$REPO" "$HOME/wm" || true            # keep /scratch files from the 15-day purge
  # compute nodes may have no internet: use the cache that setup filled (WM_ONLINE=1 turns this off)
  if [[ -z "${WM_ONLINE:-}" ]]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
fi
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

# First job of a chain: check GPUs and the model before the study starts (WM_PRECHECK=0 skips it,
# WM_PRECHECK=tests also runs the CPU test suites). A failed check stops here, with no resubmission.
if [[ -n "${SLURM_JOB_ID:-}" && "${WM_CHAIN:-0}" -eq 0 && "${WM_PRECHECK:-1}" != 0 ]]; then
  if ! bash "$REPO/promptset/job/precheck.sh" $([[ "${WM_PRECHECK:-1}" == tests ]] && echo --tests); then
    echo "[$(date -u +%FT%TZ)] precheck failed: the study did not start (fix, then submit again)"
    exit 2
  fi
fi

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

# Time limit (or scancel/preemption) -> exit 130: queue the same job again; it resumes where this one stopped.
CHAIN="${WM_CHAIN:-0}"
if [[ $CODE -eq 130 && -n "${SLURM_JOB_ID:-}" && -n "${WM_ENV:-}" && -z "${WM_NO_RESUBMIT:-}" ]]; then
  if [[ $CHAIN -lt ${WM_MAX_CHAIN:-8} ]]; then
    if [[ -e "${WM_ROOT:-/nonexistent}/STOP" ]]; then
      echo "not resubmitting: $WM_ROOT/STOP exists (delete it to allow resubmission again)"
    else
      echo "resubmitting (chain $((CHAIN + 1)) of ${WM_MAX_CHAIN:-8}); to stop the chain: touch \$WM_ROOT/STOP"
      WM_CHAIN=$((CHAIN + 1)) WM_DEPENDENCY="afterany:$SLURM_JOB_ID" bash "$REPO/promptset/job/submit.sh" "$@" || true
    fi
  else
    echo "not resubmitting: WM_MAX_CHAIN (${WM_MAX_CHAIN:-8}) reached; submit again by hand to continue"
  fi
fi
exit "$CODE"
