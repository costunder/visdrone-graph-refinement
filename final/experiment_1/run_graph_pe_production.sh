#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/kim-sanghwa/miniconda3/envs/GNN/bin/python -B \
  "$script_dir/../run_graph_pe_production.py" "$@"
