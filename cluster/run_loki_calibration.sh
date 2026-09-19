#!/usr/bin/env bash
# Run from the Mac. Default is a read-only launch check; --launch starts one job.
set -euo pipefail
mode=${1:---check}
case "$mode" in --check|--launch) ;; *) echo 'Usage: bash cluster/run_loki_calibration.sh [--check|--launch]' >&2; exit 2;; esac
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
if [[ -n $(git status --porcelain --untracked-files=no -- xcebra_ibl cluster/run_loki_calibration.sh) ]]; then
  echo 'Commit the training code and launcher before checking or launching remotely.' >&2
  exit 1
fi
revision=$(git rev-parse HEAD)
ssh -o ConnectTimeout=15 mohammadi@100.75.110.13 bash -s -- "$mode" "$revision" <<'REMOTE'
set -euo pipefail
mode=$1
revision=$2
base=/media/hdd/mohammadi/thesis
source "$base/activate.sh"
cd "$base/NEUROAI"
test "$(git rev-parse HEAD)" = "$revision" || { echo 'Remote code revision differs; deploy and test first.' >&2; exit 1; }
test -z "$(git status --porcelain --untracked-files=no -- xcebra_ibl)" || { echo 'Remote training code is modified.' >&2; exit 1; }
gpu=GPU-ddcddfcb-9e9a-2fec-a848-077ca2c870b5
usage=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
echo "Candidate P6000 utilization %, memory MiB: $usage"
awk -F, '{if ($1>10 || $2>1000) exit 1}' <<< "$usage" || { echo 'P6000 is occupied; no job started.' >&2; exit 1; }
python -c 'import torch; assert torch.cuda.is_available(); print("CUDA available; Python:", __import__("sys").executable)'
test -f "$base/data/downloaded/data_1a507308-c63a-4e02-8f32-3239a07dc578.npz"
echo "Checked revision $revision. One session, 3 seeds, dimension 4, 1000 steps, observed + null: up to 48 fits."
if [[ "$mode" != --launch ]]; then
  echo 'Check only: no training started.'
  exit 0
fi
# Exclusive fresh run directory prevents accidental overwriting or duplicate launch.
run="$base/run/calibration1000_${revision:0:12}"
mkdir "$run"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1
nohup bash -c '
  set +e
  python -m xcebra_ibl.experiments --cohort xcebra_ibl/configs/cohort.json --data-dir "$1/data/downloaded" --session-ids 1a507308-c63a-4e02-8f32-3239a07dc578 --seeds 2025 2026 2027 --dimensions 4 --iterations 1000 --device cuda --areas cortical --bootstrap 500 --nulls 1 --regularization 0.01 --output "$2/output"
  result=$?
  printf "%s\n" "$result" > "$2/exit_status.tmp"
  mv "$2/exit_status.tmp" "$2/exit_status"
  exit "$result"
' bash "$base" "$run" > "$run/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run/pid"
echo "Started PID $pid. Log: $run/run.log; exit status: $run/exit_status"
REMOTE
