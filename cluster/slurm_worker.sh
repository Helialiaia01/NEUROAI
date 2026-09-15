#!/usr/bin/env bash
# Submit with your lab's approved sbatch resource arguments; this file submits nothing.
set -euo pipefail
: "${THESIS_REPO:?Set repository directory}"
: "${THESIS_PYTHON:?Set Python executable in the validated training environment}"
: "${THESIS_JOB_MANIFEST:?Set prepared jobs.json path}"
: "${THESIS_DATA:?Set mounted raw data directory}"
: "${THESIS_OUTPUT:?Set persistent output directory}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array job}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export LOKY_MAX_CPU_COUNT="$OMP_NUM_THREADS"
cd "$THESIS_REPO"
exec "$THESIS_PYTHON" -m xcebra_ibl.jobs run \
  --manifest "$THESIS_JOB_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --data-dir "$THESIS_DATA" --root "$THESIS_OUTPUT"
