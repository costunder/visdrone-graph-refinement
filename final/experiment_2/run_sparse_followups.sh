#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/kim-sanghwa/miniconda3/envs/GNN/bin/python}"
FINAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_ROOT="$(cd -- "${FINAL_ROOT}/.." && pwd)/Full/data/visdrone_det_yolo_10class"
SHARED_ROOT="${FINAL_ROOT}/runs/table6_yolo11_10class_extra_ablation"
OUT_ROOT="${FINAL_ROOT}/experiment_2/runs/table6_yolo11_10class_extra_ablation"
SELECTED_VARIANTS="${RELGRAPH_VARIANTS:-03,04,05}"
EXTRA_ARGS=("$@")
TRAINING_AUTH_ARGS=()
if [[ "${ALLOW_MODEL_TRAINING:-0}" == "1" ]]; then
  TRAINING_AUTH_ARGS+=(--allow_training)
fi

"${PYTHON_BIN}" -u "${FINAL_ROOT}/experiment_2/run_sparse_ppr_ablation.py" \
  --run_pipeline \
  --exp2_variants "${SELECTED_VARIANTS}" \
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
  --epochs "${GNN_EPOCHS:-40}" \
  --gnn_train_steps_per_epoch "${GNN_TRAIN_STEPS_PER_EPOCH:-1024}" \
  --gnn_val_loss_limit "${GNN_VAL_LOSS_LIMIT:-512}" \
  --gnn_grad_accum_steps "${GNN_GRAD_ACCUM_STEPS:-1}" \
  --gnn_batch_size "${GNN_BATCH_SIZE:-32}" \
  --require_pyg \
  --cache \
  --resume_train \
  --rescue_coarse_conf "${RESCUE_COARSE_CONF:-0.05}" \
  --rescue_fine_conf "${RESCUE_FINE_CONF:-0.05}" \
  --model_iou 0.7 \
  --final_nms_iou 0.4 \
  --max_det 300 \
  --coarse_slice_size 640 \
  --coarse_overlap 0.2 \
  --fine_slice_size 256 \
  --fine_overlap 0.2 \
  --stage1_keep_conf 0.25 \
  --fine_infer_batch_size "${FINE_INFER_BATCH_SIZE:-32}" \
  --exp2_local_knn "${EXP2_LOCAL_KNN:-12}" \
  --exp2_cross_class_knn "${EXP2_CROSS_CLASS_KNN:-4}" \
  --exp2_ppr_knn "${EXP2_PPR_KNN:-8}" \
  --exp2_ppr_alpha "${EXP2_PPR_ALPHA:-0.15}" \
  --exp2_ppr_steps "${EXP2_PPR_STEPS:-8}" \
  --exp2_ppr_frontier "${EXP2_PPR_FRONTIER:-64}" \
  --exp2_attention_heads "${EXP2_ATTENTION_HEADS:-4}" \
  --exp2_controlled_hetero_hidden_dim "${EXP2_CONTROLLED_HETERO_HIDDEN_DIM:-56}" \
  --exp2_controlled_reference_hidden_dim "${EXP2_CONTROLLED_REFERENCE_HIDDEN_DIM:-96}" \
  --exp2_hetero_class_loss_weight "${EXP2_HETERO_CLASS_LOSS_WEIGHT:-0.50}" \
  --exp2_hetero_background_weight "${EXP2_HETERO_BACKGROUND_WEIGHT:-0.25}" \
  --exp2_hetero_class_threshold "${EXP2_HETERO_CLASS_THRESHOLD:-0.35}" \
  --exp2_hetero_class_margin "${EXP2_HETERO_CLASS_MARGIN:-0.05}" \
  --exp2_hetero_foreground_gate_alpha "${EXP2_HETERO_FOREGROUND_GATE_ALPHA:-1.0}" \
  --exp2_no_gois_conf "${EXP2_NO_GOIS_CONF:-0.05}" \
  --eval_best_val_loss \
  --selection_top_k "${SELECTION_TOP_K:-3}" \
  "${TRAINING_AUTH_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
