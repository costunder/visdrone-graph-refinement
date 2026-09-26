#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FINAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

case "${RELGRAPH_EXPERIMENT:-}" in
  1)
    exec "${FINAL_ROOT}/experiment_1/run_experiment_1.sh" "$@"
    ;;
  2)
    exec "${FINAL_ROOT}/experiment_2/run_experiment_2.sh" "$@"
    ;;
  2-followup)
    exec "${FINAL_ROOT}/experiment_2/run_sparse_followups.sh" "$@"
    ;;
  1-pe)
    export GRAPH_PE_VARIANTS="${RELGRAPH_VARIANTS:-06,07}"
    exec "${FINAL_ROOT}/experiment_1/run_graph_pe_ablation.sh" "$@"
    ;;
  2-pe)
    export GRAPH_PE_VARIANTS="${RELGRAPH_VARIANTS:-09,10}"
    exec "${FINAL_ROOT}/experiment_2/run_graph_pe_ablation.sh" "$@"
    ;;
  *)
    echo "Combined experiment execution is disabled." >&2
    echo "Set RELGRAPH_EXPERIMENT=1, 2, 2-followup, 1-pe, or 2-pe." >&2
    exit 2
    ;;
esac
