#!/usr/bin/env bash
set -euo pipefail

KERNEL="${KAGGLE_KERNEL_ID:-helialiaia/xcebra-ibl-train}"
RESULTS_DIR="${KAGGLE_RESULTS_DIR:-results/kaggle}"

while kaggle kernels status "$KERNEL" | grep -qiE 'queued|running|pulling|starting'; do
  sleep "${KAGGLE_POLL_SECONDS:-60}"
done

echo "== terminal state =="
kaggle kernels status "$KERNEL"
kaggle kernels files "$KERNEL"
mkdir -p "$RESULTS_DIR"
kaggle kernels output "$KERNEL" -p "$RESULTS_DIR" -o
