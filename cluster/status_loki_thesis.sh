#!/usr/bin/env bash
# Read-only status for the thesis exploratory run matching the local revision.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
revision=$(git -C "$repo" rev-parse HEAD)
ssh -o ConnectTimeout=15 mohammadi@100.75.110.13 bash -s -- "${revision:0:12}" <<'REMOTE'
set -euo pipefail
tag=$1
run=/media/hdd/mohammadi/thesis/run/thesis48_$tag
test -d "$run" || { echo "Run directory is absent: $run"; exit 1; }
pid=$(cat "$run/pid" 2>/dev/null || true)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "dispatcher: RUNNING (pid $pid)"
else
  echo "dispatcher: STOPPED${pid:+ (last pid $pid)}"
fi
complete=$(find "$run/workers" -name session_complete.json -type f 2>/dev/null | wc -l | tr -d ' ')
started=$(find "$run/workers" -name job_context.json -type f 2>/dev/null | wc -l | tr -d ' ')
echo "sessions: $complete/48 complete; $started started"
if [[ -f "$run/exit_status" ]]; then echo "exit status: $(cat "$run/exit_status")"; fi
tail -n 8 "$run/run.log" 2>/dev/null || true
echo 'GPUs:'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
REMOTE
