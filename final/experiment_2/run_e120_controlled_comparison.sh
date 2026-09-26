#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${ROOT}/final/experiment_2/run_sparse_followups.sh"
OUTPUT_ROOT="${E120_OUTPUT_ROOT:-${ROOT}/final/experiment_2/runs/table6_yolo11_10class_extra_ablation_e120_seed42}"
MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-4096}"
UTIL_LIMIT_PERCENT="${GPU_UTIL_LIMIT_PERCENT:-20}"
POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
REQUIRED_QUIET_CHECKS="${GPU_REQUIRED_QUIET_CHECKS:-2}"

quiet_checks=0
while (( quiet_checks < REQUIRED_QUIET_CHECKS )); do
  IFS=, read -r used util < <(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used="${used// /}"
  util="${util// /}"
  if (( used < MEMORY_LIMIT_MIB && util < UTIL_LIMIT_PERCENT )); then
    quiet_checks=$((quiet_checks + 1))
  else
    quiet_checks=0
  fi
  printf '[gpu-wait] memory=%sMiB util=%s%% quiet=%s/%s\n' \
    "${used}" "${util}" "${quiet_checks}" "${REQUIRED_QUIET_CHECKS}"
  if (( quiet_checks < REQUIRED_QUIET_CHECKS )); then
    sleep "${POLL_SECONDS}"
  fi
done

exec env \
  ALLOW_MODEL_TRAINING=1 \
  RELGRAPH_VARIANTS=06,08 \
  GNN_EPOCHS=120 \
  "${RUNNER}" \
  --output_root "${OUTPUT_ROOT}" \
  --eval_every 20
