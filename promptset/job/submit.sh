#!/usr/bin/env bash
# Submit the study job on Sharanga with the right partition, QoS, GPU count and walltime.
#
#   bash promptset/job/submit.sh --dump dump/pilot
#   bash promptset/job/submit.sh --dump dump/main --set filters='[split=pilot,main]'
#   WM_PARTITION=gpu_a100_8 bash promptset/job/submit.sh --dump dump/main --set filters='[split=pilot,main]'
#
# Every argument goes to run_study.py. Environment (optional): WM_PARTITION (default gpu_h100_4), WM_GPUS,
# WM_TIME (default 23:50:00, under the 1-day limit of gpu_h200_8), WM_QOS, WM_MAIL (address for job mails),
# WM_DEPENDENCY (sbatch --dependency value). The job resubmits itself after a time limit (run_study.sh,
# WM_MAX_CHAIN times); a finished dump does nothing on a resubmission.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export WM_ENV="${WM_ENV:-$REPO/promptset/job/sharanga.env}"
# shellcheck disable=SC1090
source "$WM_ENV"

P="$WM_PARTITION"
case "$P" in                                     # partition -> QoS (sacctmgr show assoc) and GPUs per node
  gpu_h100_4) QOS_DEF=qos_gpu_h100; G_DEF=4 ;;  # 4x H100 80 GB, max 4 days
  gpu_a100_8) QOS_DEF=qos_gpu_a100; G_DEF=8 ;;  # 8x A100 80 GB, max 5 days
  gpu_h200_8) QOS_DEF=qos_gpu_h200; G_DEF=8 ;;  # 8x H200 141 GB, max 1 day
  gpu_rtx_pro_6000_6_csis_hyd) QOS_DEF=""; G_DEF=6 ;;
  *) QOS_DEF=""; G_DEF=1 ;;
esac
QOS="${WM_QOS-$QOS_DEF}"
GPUS="${WM_GPUS:-$G_DEF}"
TIME="${WM_TIME:-23:50:00}"
CPUS=$(( GPUS * 4 < 16 ? 16 : GPUS * 4 ))
MEM=$(( GPUS * 48 ))G                            # each generation process loads the weights through CPU RAM
mkdir -p "$REPO/logs"
args=(-p "$P" --gres="gpu:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --no-requeue
      --job-name=wm-study --output="$REPO/logs/wm-study.%j.out" --signal=B:TERM@900
      --export="ALL,WM_ENV=$WM_ENV,WM_PARTITION=$P,WM_GPUS=$GPUS,WM_TIME=$TIME,WM_QOS=$QOS,WM_CHAIN=${WM_CHAIN:-0}")
[[ -n "$QOS" ]] && args+=(--qos="$QOS")
[[ -n "${WM_MAIL:-}" ]] && args+=(--mail-user="$WM_MAIL" --mail-type=END,FAIL)
[[ -n "${WM_DEPENDENCY:-}" ]] && args+=(--dependency="$WM_DEPENDENCY")
echo "sbatch ${args[*]} promptset/job/run_study.sh $*"
sbatch "${args[@]}" promptset/job/run_study.sh "$@"
