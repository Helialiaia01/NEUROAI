#!/usr/bin/env bash
# Run from the Mac. Checks or launches the frozen 48-session exploratory study.
set -euo pipefail
mode=${1:---check}
case "$mode" in --check|--launch) ;; *) echo 'Usage: bash cluster/run_loki_thesis.sh [--check|--launch]' >&2; exit 2;; esac
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
if [[ -n $(git status --porcelain --untracked-files=no -- xcebra_ibl cluster/run_loki_thesis.sh) ]]; then
  echo 'Commit training code and launcher before remote check/launch.' >&2
  exit 1
fi
revision=$(git rev-parse HEAD)
ssh -o ConnectTimeout=15 mohammadi@100.75.110.13 bash -s -- "$mode" "$revision" <<'REMOTE'
set -euo pipefail
mode=$1
revision=$2
base=/media/hdd/mohammadi/thesis
repo=$base/NEUROAI
source "$base/activate.sh"
cd "$repo"
test "$(git rev-parse HEAD)" = "$revision" || { echo 'Remote revision differs; deploy first.' >&2; exit 1; }
test -z "$(git status --porcelain --untracked-files=no -- xcebra_ibl cluster/run_loki_thesis.sh)" || { echo 'Remote training code is modified.' >&2; exit 1; }
gpu0=GPU-ddcddfcb-9e9a-2fec-a848-077ca2c870b5
gpu2=GPU-13cc3970-fd10-b6f2-630b-75459933e640
for gpu in "$gpu0" "$gpu2"; do
  usage=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  echo "$gpu utilization %, memory MiB: $usage"
  awk -F, '{if ($1>10 || $2>1000) exit 1}' <<< "$usage" || { echo "$gpu is occupied; no job started." >&2; exit 1; }
done
python -c 'import torch; assert torch.cuda.is_available(); print("CUDA available; Python:", __import__("sys").executable)'
available_kb=$(df --output=avail "$base" | tail -n 1)
(( available_kb >= 100000000 )) || { echo 'Less than 100 GB free; no job started.' >&2; exit 1; }
run=$base/run/thesis48_${revision:0:12}
plan=$run/plan
workers=$run/workers
manifest=$plan/jobs.json
if [[ -e "$run/exit_status" ]]; then
  echo "This run already finished with status $(cat "$run/exit_status")."
  exit 1
fi
if [[ -e "$run/pid" ]] && kill -0 "$(cat "$run/pid")" 2>/dev/null; then
  echo "This run is already active with PID $(cat "$run/pid")."
  exit 1
fi
mkdir -p "$run"
python -m xcebra_ibl.jobs prepare --cohort xcebra_ibl/configs/cohort.json \
  --study xcebra_ibl/configs/thesis_exploratory.json --output "$plan" \
  --phase exploratory --limit 48
python - "$manifest" "$base/data/downloaded" <<'PY'
import json, sys
from pathlib import Path
manifest, data = Path(sys.argv[1]), Path(sys.argv[2])
jobs = json.loads(manifest.read_text())['jobs']
missing = [j['eid'] for j in jobs if not (data/f"data_{j['eid']}.npz").is_file()]
if missing:
    raise SystemExit(f"Missing {len(missing)} inputs, first: {missing[:5]}")
print(f"Verified inputs for {len(jobs)} frozen exploratory sessions.")
PY
echo 'Design: 48 sessions, 3 seeds, dimension 4, 1000 steps, observed + one shuffled null; up to 2304 encoder fits.'
echo 'Measured estimate using the 69.1-minute calibration: about 28 hours on two free P6000s; allow 28-40 hours.'
if [[ "$mode" != --launch ]]; then
  echo 'Check only: no training started.'
  exit 0
fi
export PYTHONUNBUFFERED=1
nohup bash -c '
  set +e
  python -m xcebra_ibl.jobs dispatch --manifest "$1" --data-dir "$2" --root "$3" --gpus "$4" "$5"
  result=$?
  printf "%s\n" "$result" > "$6/exit_status.tmp"
  mv "$6/exit_status.tmp" "$6/exit_status"
  exit "$result"
' bash "$manifest" "$base/data/downloaded" "$workers" "$gpu0" "$gpu2" "$run" > "$run/run.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run/pid"
echo "Started PID $pid. Log: $run/run.log; exit status: $run/exit_status"
REMOTE
