#!/usr/bin/env bash
set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/kim-sanghwa/miniconda3/envs/GNN/bin/python}"
FINAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_ROOT="$(cd -- "${FINAL_ROOT}/.." && pwd)/Full/data/visdrone_det_yolo_10class"
SHARED_ROOT="${FINAL_ROOT}/runs/table6_yolo11_10class_extra_ablation"
OUT_ROOT="${FINAL_ROOT}/experiment_1/runs/table6_yolo11_10class_extra_ablation"
LOCAL_VARIANTS="${RELGRAPH_VARIANTS:-00,01,02,03,04,05}"
INTERNAL_VARIANTS="$(
  "${PYTHON_BIN}" "${FINAL_ROOT}/scripts/map_local_variants.py" \
    --experiment 1 \
    --scope existing \
    --variants "${LOCAL_VARIANTS}"
)"
export RELGRAPH_ARTIFACT_VARIANT_MAP='{"00_full_inference":"00_full_inference","01_gois_reimplementation":"01_gois_reimplementation","02_gnn_no_cluster":"02_gnn_no_cluster","03_gnn_dbscan_cluster":"03_gnn_dbscan_cluster","05_size_refinement_gnn":"04_size_refinement_gnn","08_gnn_prune_same_pool":"05_gnn_prune_same_pool"}'

EXTRA_ARGS=("$@")
TRAINING_AUTH_ARGS=()
if [[ "${ALLOW_MODEL_TRAINING:-0}" == "1" ]]; then
  TRAINING_AUTH_ARGS+=(--allow_training)
fi

BASE_VARIANT_CSV="${INTERNAL_VARIANTS}"

COMMON_ARGS=(
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
)

if [[ -n "${BASE_VARIANT_CSV}" ]]; then
  "${PYTHON_BIN}" -u "${FINAL_ROOT}/experiment_1/scripts/run_gois_paper_ablation_curves.py" \
    --run_pipeline \
    --pipeline_variants "${BASE_VARIANT_CSV}" \
    "${COMMON_ARGS[@]}"
fi

