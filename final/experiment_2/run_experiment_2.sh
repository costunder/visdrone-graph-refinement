#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/kim-sanghwa/miniconda3/envs/GNN/bin/python}"
FINAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_ROOT="$(cd -- "${FINAL_ROOT}/.." && pwd)/Full/data/visdrone_det_yolo_10class"
SHARED_ROOT="${FINAL_ROOT}/runs/table6_yolo11_10class_extra_ablation"
OUT_ROOT="${FINAL_ROOT}/experiment_2/runs/table6_yolo11_10class_extra_ablation"
LOCAL_VARIANTS="${RELGRAPH_VARIANTS:-00,01,02}"
INTERNAL_VARIANTS="$(
  "${PYTHON_BIN}" "${FINAL_ROOT}/scripts/map_local_variants.py" \
    --experiment 2 \
    --scope existing \
    --variants "${LOCAL_VARIANTS}"
)"
export RELGRAPH_ARTIFACT_VARIANT_MAP='{"04_gnn_conf_rescue":"00_gnn_conf_rescue","06_size_refinement_conf_rescue":"01_size_refinement_conf_rescue","07_low_conf_cluster_token_hgnn_refinement":"02_low_conf_cluster_token_hgnn_refinement"}'

EXTRA_ARGS=("$@")
TRAINING_AUTH_ARGS=()
if [[ "${ALLOW_MODEL_TRAINING:-0}" == "1" ]]; then
  TRAINING_AUTH_ARGS+=(--allow_training)
fi

"${PYTHON_BIN}" -u "${FINAL_ROOT}/scripts/run_gois_paper_ablation_curves.py" \
  --run_pipeline \
  --pipeline_variants "${INTERNAL_VARIANTS}" \
  --output_root "${OUT_ROOT}" \
  --source_cache_dir "${SHARED_ROOT}/common_cache" \
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
  --full_conf 0.25 \
  --coarse_conf 0.25 \
  --fine_conf 0.25 \
  --rescue_coarse_conf "${RESCUE_COARSE_CONF:-0.05}" \
  --rescue_fine_conf "${RESCUE_FINE_CONF:-0.05}" \
  --hgnn_rescue_coarse_conf "${HGNN_RESCUE_COARSE_CONF:-0.025}" \
  --hgnn_rescue_fine_conf "${HGNN_RESCUE_FINE_CONF:-0.025}" \
  --model_iou 0.7 \
  --final_nms_iou 0.4 \
  --max_det 300 \
  --coarse_slice_size 640 \
  --coarse_overlap 0.2 \
  --fine_slice_size 256 \
  --fine_overlap 0.2 \
  --stage1_keep_conf 0.25 \
  --fine_infer_batch_size "${FINE_INFER_BATCH_SIZE:-32}" \
  --eval_best_val_loss \
  --selection_top_k "${SELECTION_TOP_K:-3}" \
  "${TRAINING_AUTH_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
