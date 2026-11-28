#!/usr/bin/env bash
# Submit the study job on Sharanga with the right partition, QoS, GPU count and walltime.
#
#   bash promptset/job/submit.sh --dump dump/pilot
#   bash promptset/job/submit.sh --dump dump/main --set filters='[split=pilot,main]'
#   WM_PARTITION=gpu_a100_8 bash promptset/job/submit.sh --dump dump/main --set filters='[split=pilot,main]'
#   WM_TEST_ONLY=1 WM_TIME=03:00:00 bash promptset/job/submit.sh --dump dump/pilot   # estimated start, submits nothing
#
# Every argument goes to run_study.py. Environment (optional): WM_PARTITION (default gpu_h200_8), WM_GPUS (default 2,
# at most the GPUs of one node; the model fits on one 80 GB card, so each GPU runs its own generation process),
# WM_TIME (default 23:50:00, under the 1-day limit of gpu_h200_8), WM_JOBNAME (default ashutosh; log
# logs/<name>.<jobid>.out), WM_QOS, WM_MAIL (address for job mails),
# WM_DEPENDENCY (sbatch --dependency value), WM_MAX_CPUS / WM_MAX_MEM_GB (per-user QoS caps, default 12 / 300),
# WM_TOKEN_BUDGET (H200 only, default 250000), WM_TEST_ONLY=1 (sbatch --test-only). The job resubmits itself after a time limit (run_study.sh,
# WM_MAX_CHAIN times); a finished dump does nothing on a resubmission.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export WM_ENV="${WM_ENV:-$REPO/promptset/job/sharanga.env}"
# shellcheck disable=SC1090
source "$WM_ENV"

P="$WM_PARTITION"
case "$P" in                                     # partition -> QoS (sacctmgr show assoc) and GPUs per node
  gpu_h200_8) QOS_DEF=qos_gpu_h200; G_DEF=8 ;;  # 8x H200 141 GB, max 1 day
  gpu_h100_4) QOS_DEF=qos_gpu_h100; G_DEF=4 ;;  # 4x H100 80 GB, max 4 days
  gpu_a100_8) QOS_DEF=qos_gpu_a100; G_DEF=8 ;;  # 8x A100 80 GB, max 5 days
  gpu_rtx_pro_6000_6_csis_hyd) QOS_DEF=""; G_DEF=6 ;;
  *) QOS_DEF=""; G_DEF=1 ;;
esac
QOS="${WM_QOS-$QOS_DEF}"
GPUS="${WM_GPUS:-2}"
if (( GPUS > G_DEF )); then echo "WM_GPUS=$GPUS: $P has $G_DEF GPUs per node, using $G_DEF" >&2; GPUS=$G_DEF; fi
TIME="${WM_TIME:-23:50:00}"
JOBNAME="${WM_JOBNAME:-ashutosh}"                # shared account: the name shows whose job it is in squeue
# QoS per user (sacctmgr show qos): 2 GPUs, 2 jobs, and on H100/H200 12 CPUs and 300G across all running jobs.
# The GPU request stays generic (gpu:N): the typed H100 name is capped at 1 GPU.
MAX_CPUS="${WM_MAX_CPUS:-12}"; MAX_MEM="${WM_MAX_MEM_GB:-300}"
CPUS=$(( GPUS * 6 )); (( CPUS > MAX_CPUS )) && CPUS=$MAX_CPUS
MEM=$(( GPUS * 48 )); (( MEM > MAX_MEM )) && MEM=$MAX_MEM   # each generation process loads the weights through CPU RAM
MEM="${MEM}G"
extra=()
[[ "$P" == gpu_h200_8 ]] && extra=(--set "generate.token_budget=${WM_TOKEN_BUDGET:-250000}")  # 141 GB cards: bigger batches
mkdir -p "$REPO/logs"
args=(-p "$P" --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --no-requeue
      --job-name="$JOBNAME" --output="$REPO/logs/$JOBNAME.%j.out" --signal=B:TERM@900
      --export="ALL,WM_ENV=$WM_ENV,WM_JOBNAME=$JOBNAME,WM_PARTITION=$P,WM_GPUS=$GPUS,WM_TIME=$TIME,WM_QOS=$QOS,WM_CHAIN=${WM_CHAIN:-0}")
[[ -n "$QOS" ]] && args+=(--qos="$QOS")
[[ -n "${WM_MAIL:-}" ]] && args+=(--mail-user="$WM_MAIL" --mail-type=END,FAIL)
[[ -n "${WM_DEPENDENCY:-}" ]] && args+=(--dependency="$WM_DEPENDENCY")
[[ -n "${WM_TEST_ONLY:-}" ]] && args+=(--test-only)
echo "sbatch ${args[*]} promptset/job/run_study.sh ${extra[*]+${extra[*]}} $*"
sbatch "${args[@]}" promptset/job/run_study.sh ${extra[@]+"${extra[@]}"} "$@"
