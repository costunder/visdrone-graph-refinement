#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_MODEL_TRAINING:-0}" != "1" ]]; then
  echo "Graph-PE training is disabled. Set ALLOW_MODEL_TRAINING=1 only after explicit approval." >&2
  exit 2
fi
if [[ "${ALLOW_COCO_EVALUATION:-0}" != "1" ]]; then
  echo "Graph-PE COCO evaluation is disabled. Set ALLOW_COCO_EVALUATION=1 only after explicit approval." >&2
  exit 2
fi

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/kim-sanghwa/miniconda3/envs/GNN/bin/python}"
FINAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_ROOT="$(cd -- "${FINAL_ROOT}/.." && pwd)/Full/data/visdrone_det_yolo_10class"
SHARED_ROOT="${FINAL_ROOT}/runs/table6_yolo11_10class_extra_ablation"
OUT_ROOT="${FINAL_ROOT}/experiment_2/runs/table6_yolo11_10class_extra_ablation"
SELECTED_VARIANTS="$("${PYTHON_BIN}" "${FINAL_ROOT}/scripts/map_local_variants.py" --experiment 2 --scope pe --variants "${GRAPH_PE_VARIANTS:-09,10}")"
EXTRA_ARGS=("$@")
RESUME_ARGS=()
if [[ "${RESUME_GRAPH_PE:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume_train)
fi

"${PYTHON_BIN}" -u "${FINAL_ROOT}/experiment_2/run_graph_pe_ablation.py" \
  --pe_variants "${SELECTED_VARIANTS}" \
  --output_root "${OUT_ROOT}" \
  --common_cache_dir "${SHARED_ROOT}/common_cache" \
  --model_path "${SHARED_ROOT}/detector/weights/best.pt" \
  --train_images "${FULL_ROOT}/images/train" \
  --train_labels "${FULL_ROOT}/labels/train" \
  --eval_images "${FULL_ROOT}/images/val" \
  --eval_labels "${FULL_ROOT}/labels/val" \
  --ground_truth_path "${FULL_ROOT}/annotations/val_coco_gt.json" \
  --disable_large_preserve \
  --class_space visdrone10 \
  --device "${YOLO_DEVICE:-0}" \
  --seed 42 \
  --epochs 40 \
  --gnn_train_steps_per_epoch 1024 \
  --gnn_val_loss_limit 512 \
  --gnn_grad_accum_steps "${GNN_GRAD_ACCUM_STEPS:-1}" \
  --gnn_batch_size "${GNN_BATCH_SIZE:-32}" \
  --require_pyg \
  --cache \
  --rescue_coarse_conf 0.05 \
  --rescue_fine_conf 0.05 \
  --model_iou 0.7 \
  --final_nms_iou 0.4 \
  --max_det 300 \
  --coarse_slice_size 640 \
  --coarse_overlap 0.2 \
  --fine_slice_size 256 \
  --fine_overlap 0.2 \
  --stage1_keep_conf 0.25 \
  --exp2_local_knn 12 \
  --exp2_cross_class_knn 4 \
  --exp2_ppr_knn 8 \
  --exp2_ppr_alpha 0.15 \
  --exp2_ppr_steps 8 \
  --exp2_ppr_frontier 64 \
  --exp2_attention_heads 4 \
  --exp2_dropout 0.10 \
  --eval_best_val_loss \
  --selection_top_k 3 \
  --allow_training \
  --allow_evaluation \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
