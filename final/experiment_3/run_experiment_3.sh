#!/usr/bin/env bash
set -euo pipefail

echo "Experiment 3 blocked: stage order/handoff design must be confirmed; see final/COMPARISON_REPAIR_SPEC.md." >&2
exit 2

if [[ "${ALLOW_MODEL_TRAINING:-0}" != "1" ]]; then
  echo "Experiment 3 is configured only; no run started."
  echo "After review, set ALLOW_MODEL_TRAINING=1 to authorize training."
  exit 2
fi

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/kim-sanghwa/miniconda3/envs/GNN/bin/python}"
FINAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FULL_ROOT="$(cd -- "${FINAL_ROOT}/.." && pwd)/Full/data/visdrone_det_yolo_10class"
RUN_NAME="table6_yolo11_10class_extra_ablation"
SHARED_ROOT="${FINAL_ROOT}/runs/${RUN_NAME}"
OUTPUT_ROOT="${EXPERIMENT_3_OUTPUT_ROOT:-${FINAL_ROOT}/experiment_3/runs/${RUN_NAME}}"
EXP1_CHECKPOINT="${EXP1_CHECKPOINT:-${FINAL_ROOT}/experiment_1/runs/${RUN_NAME}/pipeline_ablation/02_gnn_no_cluster/checkpoints/epoch_116.pt}"

"${PYTHON_BIN}" -u "${FINAL_ROOT}/experiment_3/run_experiment_3.py" \
  --allow_training \
  --exp3_variants "${RELGRAPH_VARIANTS:-00,01}" \
  --output_root "${OUTPUT_ROOT}" \
  --common_cache_dir "${SHARED_ROOT}/common_cache" \
  --model_path "${SHARED_ROOT}/detector/weights/best.pt" \
  --exp3_exp1_checkpoint "${EXP1_CHECKPOINT}" \
  --train_images "${FULL_ROOT}/images/train" \
  --train_labels "${FULL_ROOT}/labels/train" \
  --eval_images "${FULL_ROOT}/images/val" \
  --eval_labels "${FULL_ROOT}/labels/val" \
  --ground_truth_path "${FULL_ROOT}/annotations/val_coco_gt.json" \
  --disable_large_preserve \
  --class_space visdrone10 \
  --device "${YOLO_DEVICE:-0}" \
  --seed "${EXPERIMENT_SEED:-42}" \
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
  --exp3_core_conf "${EXP3_CORE_CONF:-0.25}" \
  --model_iou 0.7 \
  --final_nms_iou 0.4 \
  --max_det 300 \
  --coarse_slice_size 640 \
  --coarse_overlap 0.2 \
  --fine_slice_size 256 \
  --fine_overlap 0.2 \
  --stage1_keep_conf 0.25 \
  --gnn_score_alpha 1.0 \
  --fine_infer_batch_size "${FINE_INFER_BATCH_SIZE:-32}" \
  --exp2_local_knn "${EXP2_LOCAL_KNN:-12}" \
  --exp2_cross_class_knn "${EXP2_CROSS_CLASS_KNN:-4}" \
  --exp2_ppr_knn "${EXP2_PPR_KNN:-8}" \
  --exp2_ppr_alpha "${EXP2_PPR_ALPHA:-0.15}" \
  --exp2_ppr_steps "${EXP2_PPR_STEPS:-8}" \
  --exp2_ppr_frontier "${EXP2_PPR_FRONTIER:-64}" \
  --exp2_attention_heads "${EXP2_ATTENTION_HEADS:-4}" \
  --eval_best_val_loss \
  --selection_top_k "${SELECTION_TOP_K:-3}" \
  "$@"
