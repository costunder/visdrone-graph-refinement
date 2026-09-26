import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

try:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:
    PyGData = None
    PyGDataLoader = None

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import run_gois_two_stage_gnn_ablation as ab


GOIS_REIMPLEMENTATION_VARIANT = "01_gois_reimplementation"
LEGACY_GOIS_VARIANT = "01_gois_original_repo"
VARIANT_ALIASES = {
    LEGACY_GOIS_VARIANT: GOIS_REIMPLEMENTATION_VARIANT,
}
LEGACY_VARIANT_DIRS = {
    GOIS_REIMPLEMENTATION_VARIANT: LEGACY_GOIS_VARIANT,
}

GNN_NO_CLUSTER_VARIANT = "02_gnn_no_cluster"
GNN_CLUSTER_VARIANT = "03_gnn_dbscan_cluster"
SAME_POOL_RERANK_PRUNE_VARIANT = "08_gnn_prune_same_pool"

PIPELINE_VARIANTS = [
    "00_full_inference",
    GOIS_REIMPLEMENTATION_VARIANT,
    GNN_NO_CLUSTER_VARIANT,
    GNN_CLUSTER_VARIANT,
    "04_gnn_conf_rescue",
    "05_size_refinement_gnn",
    "06_size_refinement_conf_rescue",
    "07_low_conf_cluster_token_hgnn_refinement",
    SAME_POOL_RERANK_PRUNE_VARIANT,
]

TRAINING_VARIANTS = {
    GNN_NO_CLUSTER_VARIANT,
    GNN_CLUSTER_VARIANT,
    "04_gnn_conf_rescue",
    "05_size_refinement_gnn",
    "06_size_refinement_conf_rescue",
    "07_low_conf_cluster_token_hgnn_refinement",
}

METRIC_NAMES = ab.metric_names()
PIPELINE_CODE_VERSION = 47

SAME_POOL_SIZE_REFINEMENT_VARIANT = "05_size_refinement_gnn"
LOW_CONF_SIZE_REFINEMENT_VARIANT = "06_size_refinement_conf_rescue"
LOW_CONF_CLUSTER_HGNN_VARIANT = "07_low_conf_cluster_token_hgnn_refinement"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched multi-scale baselines and relation-aware graph refinement ablations."
    )
    parser.add_argument("--output_root", default="data/gois_detection_gnn_40e_full")
    parser.add_argument("--model_path", default="runs/visdrone_train_full/yolo11n_full_10e/weights/best.pt")
    parser.add_argument("--train_images", default="data/visdrone_yolo_full/images/train")
    parser.add_argument("--train_labels", default="data/visdrone_yolo_full/labels/train")
    parser.add_argument("--eval_images", default="data/visdrone_yolo_full/images/val")
    parser.add_argument("--eval_labels", default="data/visdrone_yolo_full/labels/val")
    parser.add_argument("--ground_truth_path", default="data/ground_truth/visdrone_full_val_coco.json")
    parser.add_argument("--full_predictions_path", default="")
    parser.add_argument(
        "--disable_large_preserve",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable full-image large-object preservation for same-condition ablations.",
    )
    parser.add_argument("--source_cache_dir", default="")
    parser.add_argument("--source_prediction_dir", default="")
    parser.add_argument("--class_space", choices=["visdrone6", "visdrone10"], default="visdrone6")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_images", type=int, default=0, help="0 means full train set.")
    parser.add_argument("--max_eval_images", type=int, default=0, help="0 means full eval set.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--gnn_train_steps_per_epoch",
        type=int,
        default=1024,
        help="Number of non-empty detection graphs sampled per GNN epoch. 0 means a full train pass.",
    )
    parser.add_argument(
        "--gnn_val_loss_limit",
        type=int,
        default=512,
        help="Number of non-empty eval graphs used for fast validation loss. 0 means the full eval split.",
    )
    parser.add_argument(
        "--gnn_grad_accum_steps",
        type=int,
        default=1,
        help="Accumulate gradients over this many PyG mini-batches before optimizer.step().",
    )
    parser.add_argument(
        "--gnn_batch_size",
        type=int,
        default=32,
        help="PyG mini-batch size for detection graphs. 1 disables practical batching.",
    )
    parser.add_argument("--require_pyg", action="store_true", help="Fail instead of silently falling back when PyG is unavailable.")
    parser.add_argument("--cache_pyg_graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gnn_num_workers", type=int, default=0)
    parser.add_argument("--gnn_pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gnn_precompute_graph_pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tensorize the full train/val graph pool once per variant, then sample from that pool each epoch.",
    )
    parser.add_argument("--fine_infer_batch_size", type=int, default=32)
    parser.add_argument("--fine_roi_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fine_roi_cache_flush_every", type=int, default=256)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Run scheduled full COCO evaluation every N epochs. 0 disables scheduled full eval during training.",
    )
    parser.add_argument(
        "--eval_best_val_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, full-evaluate validation-loss-selected checkpoints.",
    )
    parser.add_argument("--selection_top_k", type=int, default=3, help="Number of lowest-val-loss checkpoints to full-evaluate after training.")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Reuse only common detector caches. Variant metrics/checkpoints are fresh unless explicitly requested.",
    )
    parser.add_argument(
        "--reuse_variant_outputs",
        action="store_true",
        help="Allow reusing finished variant metrics/predictions. Off by default for clean ablation runs.",
    )
    parser.add_argument(
        "--resume_train",
        action="store_true",
        help="Resume an interrupted GNN variant from its own checkpoint/history.",
    )
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_predictions", action="store_true")
    parser.add_argument(
        "--allow_training",
        action="store_true",
        help="Explicitly authorize GNN training variants. Required for variants 02 through 07.",
    )
    parser.add_argument("--use_source_predictions", action="store_true")
    parser.add_argument("--run_pipeline", action="store_true")
    parser.add_argument("--pipeline_variants", default="all")

    parser.add_argument("--full_conf", type=float, default=0.001)
    parser.add_argument("--coarse_conf", type=float, default=0.05)
    parser.add_argument("--fine_conf", type=float, default=0.05)
    parser.add_argument(
        "--rescue_coarse_conf",
        type=float,
        default=0.05,
        help="Low detector confidence used by 04 and 06 before GNN filtering.",
    )
    parser.add_argument(
        "--rescue_fine_conf",
        type=float,
        default=0.05,
        help="Low fine-slice confidence used by 04 and 06 before GNN filtering.",
    )
    parser.add_argument(
        "--hgnn_rescue_coarse_conf",
        type=float,
        default=0.025,
        help="Lower coarse confidence used by 07 HGNN stress rerank before GNN filtering.",
    )
    parser.add_argument(
        "--hgnn_rescue_fine_conf",
        type=float,
        default=0.025,
        help="Lower fine-slice confidence used by 07 HGNN stress rerank before GNN filtering.",
    )
    parser.add_argument("--model_iou", type=float, default=0.7)
    parser.add_argument("--final_nms_iou", type=float, default=0.5)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--coarse_slice_size", type=int, default=640)
    parser.add_argument("--coarse_overlap", type=float, default=0.2)
    parser.add_argument("--fine_slice_size", type=int, default=384)
    parser.add_argument("--fine_overlap", type=float, default=0.25)

    parser.add_argument("--dbscan_eps", type=float, default=192.0)
    parser.add_argument("--dbscan_min_samples", type=int, default=2)
    parser.add_argument("--graph_radius", type=float, default=256.0)
    parser.add_argument("--graph_knn", type=int, default=6)
    parser.add_argument("--roi_margin", type=float, default=0.30)
    parser.add_argument("--max_stage2_rois", type=int, default=40)
    parser.add_argument("--stage1_keep_conf", type=float, default=0.25)
    parser.add_argument("--large_keep_conf", type=float, default=0.25)
    parser.add_argument("--heuristic_cluster_score", type=float, default=0.12)
    parser.add_argument("--stage1_prob_threshold", type=float, default=0.45)
    parser.add_argument("--stage2_prob_threshold", type=float, default=0.40)
    parser.add_argument("--gnn_score_alpha", type=float, default=0.25)
    parser.add_argument("--label_iou", type=float, default=0.50)
    parser.add_argument("--roi_label_iou", type=float, default=0.10)
    parser.add_argument("--roi_center_margin", type=float, default=0.75)
    parser.add_argument("--disable_roi_support_target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gnn_small_thr", type=float, default=0.45)
    parser.add_argument("--gnn_large_thr", type=float, default=0.45)
    parser.add_argument("--gnn_object_thr", type=float, default=0.40)
    parser.add_argument("--gnn_roi_thr", type=float, default=0.45)
    parser.add_argument("--hgnn_cluster_roi_thr", type=float, default=0.35)
    parser.add_argument("--hgnn_cluster_keep_thr", type=float, default=0.35)
    parser.add_argument("--hgnn_cluster_suppress_thr", type=float, default=0.20)
    parser.add_argument("--hgnn_class_keep_thr", type=float, default=0.25)
    parser.add_argument("--hgnn_cluster_score_alpha", type=float, default=0.35)
    parser.add_argument("--hgnn_class_score_alpha", type=float, default=0.15)
    parser.add_argument("--no_gnn_large_gate", dest="gnn_large_gate", action="store_false")
    parser.set_defaults(gnn_large_gate=True)
    parser.add_argument("--size_graph_cluster_mode", choices=["none", "hdbscan", "dbscan"], default="none")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=3)
    parser.add_argument("--hdbscan_min_samples", type=int, default=2)
    parser.add_argument("--gnn_hidden_dim", type=int, default=96)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--stage2_source_variant", default=GNN_NO_CLUSTER_VARIANT)
    parser.add_argument("--stage2_checkpoint_epoch", type=int, default=0, help="0 selects the source GNN checkpoint by lowest val loss, then last.")
    parser.add_argument("--stage2_hidden_dim", type=int, default=0, help="0 reuses --gnn_hidden_dim.")
    parser.add_argument("--stage2_layers", type=int, default=2)
    parser.add_argument("--stage2_cleanup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage2_cleanup_min_gnn", type=float, default=0.25)
    parser.add_argument("--stage2_cleanup_high_conf", type=float, default=0.65)
    parser.add_argument("--stage2_cleanup_roi_thr", type=float, default=0.55)
    parser.add_argument("--stage2_cleanup_fine_conf", type=float, default=0.20)
    parser.add_argument("--stage2_fragment_containment_thr", type=float, default=0.30)
    parser.add_argument("--stage2_fragment_iou_thr", type=float, default=0.10)
    parser.add_argument("--hgnn_duplicate_iou_thr", type=float, default=0.50)
    parser.add_argument("--hgnn_duplicate_containment_thr", type=float, default=0.60)
    parser.add_argument("--hgnn_duplicate_norm_dist_thr", type=float, default=1.50)
    parser.add_argument("--hgnn_max_component_members", type=int, default=32)
    parser.add_argument("--hgnn_component_loss_weight", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def select_variants(text, known):
    if text.strip().lower() == "all":
        return known[:]
    selected = []
    for token in [part.strip() for part in text.split(",") if part.strip()]:
        token = VARIANT_ALIASES.get(token, token)
        match = None
        for variant in known:
            if token == variant or token == variant[:2]:
                match = variant
                break
        if match is None:
            raise ValueError(f"Unknown variant {token}. Known: {', '.join(known)}")
        selected.append(match)
    return selected


def assert_training_authorized(selected, allow_training):
    requested_training = sorted(artifact_variant_id(variant) for variant in set(selected) & TRAINING_VARIANTS)
    if requested_training and not allow_training:
        raise RuntimeError(
            "Training variants require explicit --allow_training authorization: "
            + ", ".join(requested_training)
        )


def metric_groups():
    return [
        ("metrics_group_1_ap", ["AP", "AP50", "AP75"]),
        ("metrics_group_2_area_ap", ["AP_small", "AP_medium", "AP_large"]),
        ("metrics_group_3_recall", ["AR_1", "AR_10", "AR_100"]),
        ("metrics_group_4_area_recall", ["AR_small", "AR_medium", "AR_large"]),
    ]


def write_rows(path, rows, fieldnames):
    ab.ensure_dir(Path(path).parent)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json_atomic(path, payload, **dump_kwargs):
    path = Path(path)
    ab.ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, **dump_kwargs)
    os.replace(tmp_path, path)


def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def try_import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"[curves] matplotlib unavailable: {exc}", flush=True)
        return None


def save_variant_curves(variant_dir, rows):
    plt = try_import_pyplot()
    if plt is None:
        return
    ab.ensure_dir(variant_dir)
    rows = sorted(rows, key=lambda item: int(float(item["epoch"])))
    xs = [int(float(row["epoch"])) for row in rows]

    for image_name, metrics in metric_groups():
        plt.figure(figsize=(8.2, 4.8), dpi=160)
        plotted = False
        for metric in metrics:
            points = [
                (int(float(row["epoch"])), float(row[metric]))
                for row in rows
                if str(row.get(metric, "")).strip()
            ]
            if not points:
                continue
            metric_xs, ys = zip(*points)
            plt.plot(metric_xs, ys, marker="o", linewidth=2.0, label=metric)
            plotted = True
        plt.title(f"{Path(variant_dir).name} - {image_name}")
        plt.xlabel("epoch")
        plt.ylabel("COCO metric")
        plt.grid(True, alpha=0.3)
        if plotted:
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No full evaluation points yet.", ha="center", va="center")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(Path(variant_dir) / f"{image_name}.png")
        plt.close()

    plt.figure(figsize=(8.2, 4.8), dpi=160)
    has_loss = any(str(row.get("train_loss", "")).strip() for row in rows)
    if has_loss:
        train_loss = [float(row["train_loss"]) for row in rows]
        val_loss = [float(row["val_loss"]) for row in rows]
        plt.plot(xs, train_loss, marker="o", linewidth=2.0, label="train_loss")
        plt.plot(xs, val_loss, marker="o", linewidth=2.0, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No trainable parameters for this baseline.", ha="center", va="center")
        plt.xticks([])
        plt.yticks([])
    plt.title(f"{Path(variant_dir).name} - train/val loss")
    plt.tight_layout()
    plt.savefig(Path(variant_dir) / "loss_curve.png")
    plt.close()


def save_history(variant_dir, rows):
    fieldnames = ["epoch", "variant", "num_predictions", "train_loss", "val_loss"] + METRIC_NAMES
    write_rows(Path(variant_dir) / "metrics_history.csv", rows, fieldnames)
    loss_rows = [
        {"epoch": row["epoch"], "train_loss": row.get("train_loss", ""), "val_loss": row.get("val_loss", "")}
        for row in rows
    ]
    write_rows(Path(variant_dir) / "loss_history.csv", loss_rows, ["epoch", "train_loss", "val_loss"])
    save_variant_curves(variant_dir, rows)


def record_epoch(variant_dir, variant, epoch, train_loss, val_loss, predictions, gt_path):
    predictions_dir = Path(variant_dir) / "predictions"
    ab.ensure_dir(predictions_dir)
    pred_path = predictions_dir / f"epoch_{int(epoch):03d}.json"
    ab.write_predictions(pred_path, predictions)
    metrics = ab.evaluate_predictions(gt_path, str(pred_path))
    row = {
        "epoch": int(epoch),
        "variant": artifact_variant_id(variant),
        "num_predictions": len(predictions),
        "train_loss": "" if train_loss is None else round(float(train_loss), 6),
        "val_loss": "" if val_loss is None else round(float(val_loss), 6),
    }
    row.update({key: round(float(metrics[key]), 6) for key in METRIC_NAMES})
    return row


def record_loss_only_epoch(variant, epoch, train_loss, val_loss):
    row = {
        "epoch": int(epoch),
        "variant": artifact_variant_id(variant),
        "num_predictions": "",
        "train_loss": "" if train_loss is None else round(float(train_loss), 6),
        "val_loss": "" if val_loss is None else round(float(val_loss), 6),
    }
    row.update({key: "" for key in METRIC_NAMES})
    return row


def has_full_eval(row):
    return str(row.get("AP", "")).strip() != ""


def should_full_eval_epoch(epoch, args):
    if args.eval_every <= 0:
        return False
    return epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0


def best_val_loss_epoch(rows):
    candidates = []
    for row in rows:
        value = str(row.get("val_loss", "")).strip()
        if not value:
            continue
        candidates.append((float(value), epoch_value(row)))
    if not candidates:
        return 0
    return min(candidates)[1]


def best_ap_epoch(rows):
    candidates = []
    for row in rows:
        if not has_full_eval(row):
            continue
        candidates.append((float(row["AP"]), epoch_value(row)))
    if not candidates:
        return completed_epoch_from_rows(rows)
    return max(candidates)[1]


def selection_candidates(rows, args):
    if not rows:
        return []
    candidates = []
    scored = []
    for row in rows:
        value = str(row.get("val_loss", "")).strip()
        if value:
            scored.append((float(value), epoch_value(row)))
    for rank, (_, epoch) in enumerate(sorted(scored)[: max(0, args.selection_top_k)], start=1):
        candidates.append((f"val_loss_top_{rank}", epoch))
    last_epoch = completed_epoch_from_rows(rows)
    if last_epoch:
        candidates.append(("last", last_epoch))
    return candidates


def replace_history_row(rows, updated):
    target_epoch = epoch_value(updated)
    replaced = False
    next_rows = []
    for row in rows:
        if epoch_value(row) == target_epoch:
            merged = dict(row)
            merged.update(updated)
            next_rows.append(merged)
            replaced = True
        else:
            next_rows.append(row)
    if not replaced:
        next_rows.append(updated)
    return sorted(next_rows, key=epoch_value)


def epoch_value(row):
    return int(float(row["epoch"]))


def completed_epoch_from_rows(rows):
    if not rows:
        return 0
    return max(epoch_value(row) for row in rows)


def rows_through_epoch(rows, epoch):
    return [row for row in rows if epoch_value(row) <= epoch]


def checkpoint_epoch(path):
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def checkpoint_at_or_before(ckpt_dir, pattern, epoch):
    candidates = [path for path in Path(ckpt_dir).glob(pattern) if checkpoint_epoch(path) <= epoch]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_epoch)


def normalize_variant_id(variant):
    return VARIANT_ALIASES.get(variant, variant)

def artifact_variant_id(variant):
    canonical = normalize_variant_id(variant)
    raw_mapping = os.environ.get("RELGRAPH_ARTIFACT_VARIANT_MAP", "").strip()
    if not raw_mapping:
        return canonical
    mapping = json.loads(raw_mapping)
    if not isinstance(mapping, dict):
        raise ValueError("RELGRAPH_ARTIFACT_VARIANT_MAP must be a JSON object")
    return str(mapping.get(canonical, canonical))


def existing_variant_dir(table_dir, variant):
    canonical = Path(table_dir) / artifact_variant_id(variant)
    if canonical.exists():
        return canonical
    legacy_name = LEGACY_VARIANT_DIRS.get(normalize_variant_id(variant))
    if legacy_name:
        legacy = Path(table_dir) / legacy_name
        if legacy.exists():
            return legacy
    return canonical


def source_prediction_file(source_dir, variant):
    if source_dir is None:
        return None
    canonical_variant = artifact_variant_id(variant)
    canonical = Path(source_dir) / f"{canonical_variant}.json"
    if canonical.exists():
        return canonical
    legacy_name = LEGACY_VARIANT_DIRS.get(canonical_variant)
    if legacy_name:
        legacy = Path(source_dir) / f"{legacy_name}.json"
        if legacy.exists():
            return legacy
    return canonical


def normalize_result_rows(rows):
    normalized = []
    for row in rows:
        item = dict(row)
        item["variant"] = artifact_variant_id(item.get("variant", ""))
        normalized.append(item)
    return normalized


def gois_baseline_provenance():
    return {
        "canonical_variant": GOIS_REIMPLEMENTATION_VARIANT,
        "legacy_artifact_alias": LEGACY_GOIS_VARIANT,
        "implementation": "local_reimplementation",
        "upstream_gois_entrypoint_invoked": False,
        "exact_paper_table_reproduction": False,
        "claim": "GOIS-Det-inspired matched-condition baseline under the local detector and evaluation protocol recorded in this repository",
        "paper_doi": "10.1016/j.neucom.2025.130327",
        "upstream_repository": "https://github.com/MMUZAMMUL/GOIS",
        "license_review": "Upstream repository is restrictively licensed; see THIRD_PARTY_NOTICES.md before publication or redistribution.",
    }


def summarize_table(table_dir, selected):
    final_rows = []
    best_rows = []
    all_rows = []
    for variant in selected:
        history_path = existing_variant_dir(table_dir, variant) / "metrics_history.csv"
        if not history_path.exists():
            continue
        rows = normalize_result_rows(read_rows(history_path))
        if not rows:
            continue
        all_rows.extend(rows)
        rows_sorted = sorted(rows, key=lambda item: int(float(item["epoch"])))
        eval_rows = [row for row in rows_sorted if str(row.get("AP", "")).strip()]
        if not eval_rows:
            continue
        final_rows.append(eval_rows[-1])
        best_rows.append(max(eval_rows, key=lambda item: float(item["AP"])))
    fieldnames = ["epoch", "variant", "num_predictions", "train_loss", "val_loss"] + METRIC_NAMES
    write_rows(Path(table_dir) / "epoch_metrics_all.csv", all_rows, fieldnames)
    write_rows(Path(table_dir) / "evaluation_results_last_epoch.csv", final_rows, fieldnames)
    write_rows(Path(table_dir) / "evaluation_results_best_ap.csv", best_rows, fieldnames)


def write_manifest(table_dir, table_type, args, selected):
    pyg_available = PyGData is not None and PyGDataLoader is not None
    manifest = {
        "table_type": table_type,
        "baseline_provenance": gois_baseline_provenance(),
        "selected_variants": [artifact_variant_id(variant) for variant in selected],
        "internal_variant_ids": selected,
        "full_data": args.max_train_images == 0 and args.max_eval_images == 0,
        "code": code_signature(),
        "common_conditions": variant_run_config("common", args, "manifest")["args"],
        "run_control": {
            "cache": args.cache,
            "allow_training": args.allow_training,
            "reuse_variant_outputs": args.reuse_variant_outputs,
            "resume_train": args.resume_train,
            "force_train": args.force_train,
            "force_predictions": args.force_predictions,
            "eval_best_val_loss": args.eval_best_val_loss,
            "selection_top_k": args.selection_top_k,
            "eval_gt_path": getattr(args, "eval_gt_path", args.ground_truth_path),
            "require_pyg": args.require_pyg,
        },
        "pyg": {
            "available": pyg_available,
            "enabled": pyg_available and int(getattr(args, "gnn_batch_size", 1)) > 1,
            "batch_size": args.gnn_batch_size,
            "grad_accum_steps": args.gnn_grad_accum_steps,
            "num_workers": args.gnn_num_workers,
            "pin_memory": args.gnn_pin_memory,
            "cache_pyg_graphs": args.cache_pyg_graphs,
        },
        "fine_roi_inference": {
            "batch_size": args.fine_infer_batch_size,
            "cache_enabled": args.fine_roi_cache,
            "cache_flush_every": args.fine_roi_cache_flush_every,
        },
    }
    manifest["controlled_difference"] = (
        "Matched-condition pipeline ablation: 00 local full-image FI-Det control, 01 local GOIS-Det reimplementation "
        "coarse+fine+NMS, 02 legacy 3-head GNN without supervised ROI head and "
        "without cluster, 03 the same 3-head GNN with DBSCAN cluster context. 02-03 use "
        "the same coarse+global-fine+NMS candidate pool as 01, then run with full-image "
        "large preserve disabled. 04 is a separate low-conf rescue experiment: "
        "it widens the YOLO candidate pool before thresholding, then lets GNN rescue/suppress "
        "candidates before final NMS. 05 is a same-pool learned size-only refinement stage "
        "over 02 that preserves the stage1 object/keep score. 06 applies the same size-only "
        "refinement over the 04 low-conf rerank model/pool. 07 uses the same 04 low-conf "
        "candidate pool and cutoff/NMS, but replaces the pairwise GNN reranker with "
        "cluster-token hypergraph context before the 0.25 cutoff. 08 reuses the 02 GNN "
        "and 01 candidate pool, changing only the post-rerank score cutoff so low-ranked "
        "candidates can be pruned. 08 applies the same post-rerank 0.25 pruning rule "
        "to the completed 02 checkpoint."
    )
    ab.ensure_dir(table_dir)
    write_json_atomic(Path(table_dir) / "config_manifest.json", manifest, indent=2)


def file_sha256(path):
    path = Path(path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_signature():
    script_dir = Path(__file__).resolve().parent
    return {
        "runner": file_sha256(script_dir / "run_gois_paper_ablation_curves.py"),
        "ablation": file_sha256(script_dir / "run_gois_two_stage_gnn_ablation.py"),
        "version": PIPELINE_CODE_VERSION,
    }


def variant_run_config(variant, args, mode):
    keys = [
        "model_path",
        "train_images",
        "train_labels",
        "eval_images",
        "eval_labels",
        "ground_truth_path",
        "full_predictions_path",
        "disable_large_preserve",
        "source_cache_dir",
        "source_prediction_dir",
        "class_space",
        "use_source_predictions",
        "max_train_images",
        "max_eval_images",
        "seed",
        "epochs",
        "eval_every",
        "full_conf",
        "coarse_conf",
        "fine_conf",
        "rescue_coarse_conf",
        "rescue_fine_conf",
        "hgnn_rescue_coarse_conf",
        "hgnn_rescue_fine_conf",
        "model_iou",
        "final_nms_iou",
        "max_det",
        "coarse_slice_size",
        "coarse_overlap",
        "fine_slice_size",
        "fine_overlap",
        "dbscan_eps",
        "dbscan_min_samples",
        "graph_radius",
        "graph_knn",
        "roi_margin",
        "max_stage2_rois",
        "stage1_keep_conf",
        "large_keep_conf",
        "heuristic_cluster_score",
        "stage1_prob_threshold",
        "stage2_prob_threshold",
        "gnn_score_alpha",
        "label_iou",
        "roi_label_iou",
        "roi_center_margin",
        "disable_roi_support_target",
        "gnn_small_thr",
        "gnn_large_thr",
        "gnn_object_thr",
        "gnn_roi_thr",
        "hgnn_cluster_roi_thr",
        "hgnn_cluster_keep_thr",
        "hgnn_cluster_suppress_thr",
        "hgnn_class_keep_thr",
        "hgnn_cluster_score_alpha",
        "hgnn_class_score_alpha",
        "gnn_large_gate",
        "size_graph_cluster_mode",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "gnn_hidden_dim",
        "gnn_layers",
        "stage2_source_variant",
        "stage2_checkpoint_epoch",
        "stage2_hidden_dim",
        "stage2_layers",
        "stage2_hypergraph",
        "stage2_cleanup",
        "stage2_cleanup_min_gnn",
        "stage2_cleanup_high_conf",
        "stage2_cleanup_roi_thr",
        "stage2_cleanup_fine_conf",
        "stage2_fragment_containment_thr",
        "stage2_fragment_iou_thr",
        "hgnn_duplicate_iou_thr",
        "hgnn_duplicate_containment_thr",
        "hgnn_duplicate_norm_dist_thr",
        "hgnn_max_component_members",
        "hgnn_component_loss_weight",
        "gnn_train_steps_per_epoch",
        "gnn_val_loss_limit",
        "gnn_grad_accum_steps",
        "gnn_batch_size",
        "cache_pyg_graphs",
        "gnn_num_workers",
        "gnn_pin_memory",
        "gnn_precompute_graph_pool",
        "fine_infer_batch_size",
        "fine_roi_cache",
        "fine_roi_cache_flush_every",
        "lr",
        "weight_decay",
    ]
    values = {key: getattr(args, key, None) for key in keys}
    if variant == GNN_NO_CLUSTER_VARIANT:
        values["slice_candidate_source"] = "same_as_01_candidate_pool_no_cluster_legacy_heads_fusion"
    elif variant == GNN_CLUSTER_VARIANT:
        values["slice_candidate_source"] = "same_as_02_candidate_pool_dbscan_cluster_legacy_heads_fusion"
    elif variant == "04_gnn_conf_rescue":
        values["slice_candidate_source"] = "low_conf_coarse+low_conf_global_fine+final_nms_pre_gnn_rescue"
    elif variant == SAME_POOL_SIZE_REFINEMENT_VARIANT:
        values["slice_candidate_source"] = "coarse+global_fine+final_nms_then_learned_stage2_size_only_refinement"
    elif variant == LOW_CONF_SIZE_REFINEMENT_VARIANT:
        values["slice_candidate_source"] = "04_low_conf_coarse+global_fine+stage1_rerank_then_size_only_refinement_then_gois_nms"
    elif variant == LOW_CONF_CLUSTER_HGNN_VARIANT:
        values["slice_candidate_source"] = "04_low_conf_coarse+global_fine+cluster_token_hgnn_rerank_then_gois_nms"
    elif variant == SAME_POOL_RERANK_PRUNE_VARIANT:
        values["slice_candidate_source"] = "same_as_01_candidate_pool_then_02_gnn_rerank_score_cut"
    else:
        values["slice_candidate_source"] = "coarse+global_fine+final_nms"
    if values.get("disable_large_preserve"):
        values["full_predictions_path"] = ""
        values["large_keep_conf"] = None
    return {
        "variant": artifact_variant_id(variant),
        "mode": mode,
        "baseline_provenance": gois_baseline_provenance() if normalize_variant_id(variant) == GOIS_REIMPLEMENTATION_VARIANT else None,
        "code": code_signature(),
        "args": values,
    }


def variant_config_path(variant_dir):
    return Path(variant_dir) / "run_config.json"


def read_json_or_none(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def variant_config_matches(variant_dir, variant, args, mode):
    expected = variant_run_config(variant, args, mode)
    current = read_json_or_none(variant_config_path(variant_dir))
    return current == expected


def write_variant_config(variant_dir, variant, args, mode):
    path = variant_config_path(variant_dir)
    ab.ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(variant_run_config(variant, args, mode), f, indent=2, sort_keys=True)


def clear_stale_variant_outputs(variant_dir):
    variant_dir = Path(variant_dir)
    for name in [
        "metrics_history.csv",
        "loss_history.csv",
        "selection_evaluations.csv",
        "two_stage_loss_history.csv",
        "run_config.json",
    ]:
        path = variant_dir / name
        if path.exists():
            path.unlink()
    for name in ["predictions", "checkpoints"]:
        path = variant_dir / name
        if path.exists():
            shutil.rmtree(path)
    for path in variant_dir.glob("*.png"):
        path.unlink()


def clone_args_for_ab(args):
    args.stage1_epochs = args.epochs
    return args


def clone_args_with(args, **updates):
    values = vars(args).copy()
    values.update(updates)
    cloned = argparse.Namespace(**values)
    return cloned


def make_cached_prediction_samples(records, cache):
    return [
        {
            "record": record,
            "predictions": cache.get(str(record.image_id), []),
            "nodes": [],
        }
        for record in records
    ]


def fine_prediction_cache_path(cache_dir, split_name, records, args):
    payload = {
        "schema": "global_fine_cache_v1",
        "class_space": getattr(args, "class_space", "visdrone6"),
        "model_path": str(getattr(args, "model_path", "")),
        "split": split_name,
        "conf": float(args.fine_conf),
        "iou": float(args.model_iou),
        "max_det": int(args.max_det),
        "slice_size": int(args.fine_slice_size),
        "overlap": float(args.fine_overlap),
        "records": [
            {
                "image_id": int(record.image_id),
                "path": str(record.path),
                "width": int(record.width),
                "height": int(record.height),
            }
            for record in records
        ],
    }
    signature = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    suffix = (
        f"{split_name}_fine_conf{args.fine_conf:.3f}_s{args.fine_slice_size}"
        f"_o{args.fine_overlap:.2f}_n{len(records) or 'all'}_{signature}.json"
    )
    return Path(cache_dir) / suffix


def generate_or_load_fine_cache(model, records, args, split_name, cache_dir):
    cache_path = fine_prediction_cache_path(cache_dir, split_name, records, args)
    if args.cache and cache_path.exists():
        print(f"[cache] using global fine cache: {cache_path}", flush=True)
        with cache_path.open("r") as f:
            return json.load(f)

    data = {}
    for record in tqdm(records, desc=f"{split_name} global fine GOIS cache"):
        data[str(record.image_id)] = ab.predict_sliced_image(
            model,
            record,
            conf=args.fine_conf,
            iou=args.model_iou,
            max_det=args.max_det,
            device=args.device,
            slice_size=args.fine_slice_size,
            overlap=args.fine_overlap,
            batch_size=getattr(args, "fine_infer_batch_size", 32),
        )
    write_json_atomic(cache_path, data, indent=2)
    return data


def make_global_gois_candidate_cache(records, coarse_cache, fine_cache, args):
    data = {}
    for record in tqdm(records, desc="Build coarse+global fine GOIS candidates"):
        image_id = str(record.image_id)
        predictions = list(coarse_cache.get(image_id, [])) + list(fine_cache.get(image_id, []))
        data[image_id] = ab.classwise_nms(predictions, iou_threshold=args.final_nms_iou, limit=args.max_det)
    return data


def build_gois_candidate_cache_for_args(model, records, args, split_name, cache_dir):
    coarse_cache = ab.generate_or_load_coarse_cache(model, records, args, split_name, cache_dir)
    fine_cache = generate_or_load_fine_cache(model, records, args, split_name, cache_dir)
    return make_global_gois_candidate_cache(records, coarse_cache, fine_cache, args)


def make_pre_gois_candidate_cache(records, coarse_cache, fine_cache):
    data = {}
    for record in tqdm(records, desc="Build pre-GOIS coarse+fine candidates"):
        image_id = str(record.image_id)
        data[image_id] = list(coarse_cache.get(image_id, [])) + list(fine_cache.get(image_id, []))
    return data


def build_pre_gois_candidate_cache_for_args(model, records, args, split_name, cache_dir):
    coarse_cache = ab.generate_or_load_coarse_cache(model, records, args, split_name, cache_dir)
    fine_cache = generate_or_load_fine_cache(model, records, args, split_name, cache_dir)
    return make_pre_gois_candidate_cache(records, coarse_cache, fine_cache)


def load_common_context(
    args,
    table_dir,
    need_train=True,
    need_gois_candidates=True,
    need_base_cache=True,
    need_full_predictions=False,
):
    cache_dir = Path(args.output_root) / "common_cache"
    ab.ensure_dir(cache_dir)
    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    if args.cache and source_cache_dir and source_cache_dir.exists():
        for src in source_cache_dir.glob("*.json"):
            dst = cache_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    model = YOLO(args.model_path)
    train_records = ab.build_image_records(args.train_images, max_images=args.max_train_images) if need_train else []
    eval_records = ab.build_image_records(args.eval_images, args.ground_truth_path, max_images=args.max_eval_images)

    if Path(args.ground_truth_path).exists() and not args.max_eval_images:
        eval_gt_path = args.ground_truth_path
    elif Path(args.ground_truth_path).exists() and args.max_eval_images:
        eval_gt_path = str(Path(table_dir) / "ground_truth_eval_subset.json")
        ab.filter_coco_gt(args.ground_truth_path, eval_records, eval_gt_path)
    else:
        eval_gt_path = str(Path(table_dir) / "ground_truth_eval_from_yolo.json")
        ab.generate_coco_gt_from_yolo(eval_records, args.eval_labels, eval_gt_path)

    train_gt = ab.load_gt_by_image(train_records, args.train_labels) if need_train else {}
    eval_gt = ab.load_coco_gt_by_image(eval_records, eval_gt_path)

    train_cache = {}
    if need_train and need_base_cache:
        train_cache = ab.generate_or_load_coarse_cache(model, train_records, args, "train", cache_dir)
    eval_cache = ab.generate_or_load_coarse_cache(model, eval_records, args, "eval", cache_dir) if need_base_cache else {}
    train_gois_cache = train_cache
    eval_gois_cache = eval_cache
    if need_gois_candidates:
        train_fine_cache = None
        if need_train:
            train_fine_cache = generate_or_load_fine_cache(model, train_records, args, "train", cache_dir)
            train_gois_cache = make_global_gois_candidate_cache(train_records, train_cache, train_fine_cache, args)
        eval_fine_cache = generate_or_load_fine_cache(model, eval_records, args, "eval", cache_dir)
        eval_gois_cache = make_global_gois_candidate_cache(eval_records, eval_cache, eval_fine_cache, args)

    eval_image_ids = {record.image_id for record in eval_records}
    local_full_prediction_path = (
        Path(args.output_root)
        / "pipeline_ablation"
        / "00_full_inference"
        / "predictions"
        / "epoch_000.json"
    )
    local_full_variant_dir = local_full_prediction_path.parent.parent
    if getattr(args, "disable_large_preserve", False) and not need_full_predictions:
        full_predictions = []
    elif args.full_predictions_path:
        if not Path(args.full_predictions_path).exists():
            raise FileNotFoundError(f"--full_predictions_path not found: {args.full_predictions_path}")
        full_predictions = ab.filter_predictions_to_images(ab.read_prediction_file(args.full_predictions_path), eval_image_ids)
    elif (
        args.reuse_variant_outputs
        and local_full_prediction_path.exists()
        and variant_config_matches(local_full_variant_dir, "00_full_inference", args, mode="static")
    ):
        print(f"[full_predictions] using config-matched local predictions: {local_full_prediction_path}", flush=True)
        full_predictions = ab.filter_predictions_to_images(ab.read_prediction_file(local_full_prediction_path), eval_image_ids)
    else:
        full_predictions = ab.run_full_baseline(model, eval_records, args)
    full_by_image = ab.group_predictions_by_image(full_predictions)
    return {
        "model": model,
        "train_records": train_records,
        "eval_records": eval_records,
        "eval_gt_path": eval_gt_path,
        "train_gt": train_gt,
        "eval_gt": eval_gt,
        "train_cache": train_cache,
        "eval_cache": eval_cache,
        "train_gois_cache": train_gois_cache,
        "eval_gois_cache": eval_gois_cache,
        "full_by_image": full_by_image,
    }


def run_full_reference(eval_records, full_by_image):
    predictions = []
    for record in eval_records:
        predictions.extend(full_by_image.get(record.image_id, []))
    return predictions


def run_gois_reimplementation(eval_samples, args):
    predictions = []
    for sample in tqdm(eval_samples, desc="GOIS-Det local reimplementation condition"):
        keep = [
            pred
            for pred in sample["predictions"]
            if float(pred.get("score", 0.0)) >= float(args.stage1_keep_conf)
        ]
        predictions.extend(ab.classwise_nms(keep, iou_threshold=args.final_nms_iou))
    return predictions

def run_static_variant(table_dir, variant, epoch, predict_fn, gt_path, args, source_pred_path=None, eval_image_ids=None):
    variant_dir = Path(table_dir) / artifact_variant_id(variant)
    history_path = variant_dir / "metrics_history.csv"
    config_ok = variant_config_matches(variant_dir, variant, args, mode="static")
    if args.reuse_variant_outputs and history_path.exists() and config_ok and not args.force_predictions:
        print(f"[{variant}] reuse static history: {history_path}", flush=True)
        return
    if history_path.exists() and not config_ok:
        print(f"[{variant}] stale static cache ignored because code/config changed", flush=True)
        clear_stale_variant_outputs(variant_dir)
    elif history_path.exists() and not args.reuse_variant_outputs:
        print(f"[{variant}] fresh static run clears previous generated summaries", flush=True)
        clear_stale_variant_outputs(variant_dir)
    print(f"[{variant}] evaluating static baseline", flush=True)
    if (
        source_pred_path
        and Path(source_pred_path).exists()
        and args.use_source_predictions
        and not args.force_predictions
    ):
        print(f"[{variant}] using source predictions: {source_pred_path}", flush=True)
        predictions = ab.read_prediction_file(source_pred_path)
    else:
        predictions = predict_fn()
    if eval_image_ids is not None:
        predictions = ab.filter_predictions_to_images(predictions, eval_image_ids)
    row = record_epoch(variant_dir, variant, int(epoch), None, None, predictions, gt_path)
    save_history(variant_dir, [row])
    write_variant_config(variant_dir, variant, args, mode="static")


def load_single_stage_resume_state(scorer, optimizer, ckpt_dir, completed_epoch, device):
    if completed_epoch <= 0:
        return 0
    model_path = checkpoint_at_or_before(ckpt_dir, "epoch_*.pt", completed_epoch)
    if model_path is not None:
        scorer.load_state_dict(torch.load(model_path, map_location=device))
        loaded_epoch = checkpoint_epoch(model_path)
        print(f"[resume] loaded model weights {model_path}", flush=True)
        return loaded_epoch
    return 0


def load_single_stage_model_at_epoch(scorer, ckpt_dir, epoch, device):
    model_path = Path(ckpt_dir) / f"epoch_{epoch:03d}.pt"
    if model_path.exists():
        scorer.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[selection] loaded best-val-loss weights {model_path}", flush=True)
        return True
    return False


def resolve_stage2_source_epoch(source_dir, args):
    requested = int(getattr(args, "stage2_checkpoint_epoch", 0) or 0)
    if requested > 0:
        return requested

    history_path = Path(source_dir) / "metrics_history.csv"
    if history_path.exists():
        rows = read_rows(history_path)
        scored = [
            (float(row["val_loss"]), epoch_value(row))
            for row in rows
            if str(row.get("val_loss", "")).strip()
        ]
        if scored:
            return min(scored)[1]
        completed = completed_epoch_from_rows(rows)
        if completed:
            return completed

    ckpt_dir = Path(source_dir) / "checkpoints"
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"), key=checkpoint_epoch)
    return checkpoint_epoch(candidates[-1]) if candidates else 0


def load_stage2_source_model(table_dir, args, device):
    source_variant = getattr(args, "stage2_source_variant", GNN_NO_CLUSTER_VARIANT)
    source_dir = Path(table_dir) / source_variant
    ckpt_dir = source_dir / "checkpoints"
    source_epoch = resolve_stage2_source_epoch(source_dir, args)
    if source_epoch <= 0:
        raise FileNotFoundError(
            f"{source_variant} checkpoint is required before running two-stage variants. "
            f"Run --pipeline_variants 02 first or provide a completed {source_variant} first."
        )
    output_dim = 3 if stage2_source_uses_legacy_heads(source_variant) else ab.SIZE_AWARE_OUTPUT_DIM
    model = ab.SizeAwareGraphGNN(
        ab.SIZE_AWARE_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        args.gnn_hidden_dim,
        args.gnn_layers,
        output_dim=output_dim,
    ).to(device)
    if not load_single_stage_model_at_epoch(model, ckpt_dir, source_epoch, device):
        raise FileNotFoundError(f"Could not load {source_variant} checkpoint at epoch {source_epoch} from {ckpt_dir}")
    print(f"[stage2] source={source_variant} epoch={source_epoch}", flush=True)
    return model, source_variant, source_epoch


def stage2_source_uses_legacy_heads(source_variant):
    return source_variant in {
        GNN_NO_CLUSTER_VARIANT,
        GNN_CLUSTER_VARIANT,
        "04_gnn_conf_rescue",
    }


def attach_stage2_source_scores(samples, model, args, device, source_variant):
    if stage2_source_uses_legacy_heads(source_variant):
        attach_legacy_scores_for_eval(samples, model, args, device, hypergraph=False)
    else:
        attach_size_aware_scores_for_eval(samples, model, args, device)


def nonempty_graph_samples(samples):
    return [sample for sample in samples if sample["nodes"]]


def select_limited_samples(samples, limit, seed):
    valid = nonempty_graph_samples(samples)
    return select_limited_items(valid, limit, seed)


def select_limited_items(items, limit, seed):
    limit = int(limit or 0)
    if limit <= 0 or limit >= len(items):
        selected = list(items)
        random.Random(seed).shuffle(selected)
        return selected
    return random.Random(seed).sample(list(items), limit)


def use_pyg_batching(args):
    return PyGData is not None and PyGDataLoader is not None and int(getattr(args, "gnn_batch_size", 1)) > 1


def make_pyg_loader(data_list, args, shuffle):
    workers = max(0, int(getattr(args, "gnn_num_workers", 0)))
    kwargs = {
        "batch_size": max(1, int(args.gnn_batch_size)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(getattr(args, "gnn_pin_memory", False)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return PyGDataLoader(data_list, **kwargs)


def sample_to_pyg_data(sample, args):
    if bool(getattr(args, "cache_pyg_graphs", True)) and sample.get("_pyg_data") is not None:
        cached = sample["_pyg_data"]
        return cached
    x, edge_index, edge_attr, y, weights = ab.build_size_aware_tensors(sample["nodes"], sample["record"], args)
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, weights=weights)
    if bool(getattr(args, "cache_pyg_graphs", True)):
        sample["_pyg_data"] = data
    return data


def selected_graph_data(samples, limit, seed, args, desc):
    selected = select_limited_samples(samples, limit, seed)
    if not use_pyg_batching(args):
        return selected
    return [sample_to_pyg_data(sample, args) for sample in tqdm(selected, desc=desc, leave=False)]


def is_pyg_graph_pool(pool):
    return bool(pool) and PyGData is not None and isinstance(pool[0], PyGData)


def prepare_graph_pool(samples, args, desc):
    valid = nonempty_graph_samples(samples)
    if not use_pyg_batching(args) or not bool(getattr(args, "gnn_precompute_graph_pool", True)):
        return valid
    return [sample_to_pyg_data(sample, args) for sample in tqdm(valid, desc=desc, leave=False)]


def compute_global_size_aware_pos_weight(samples, head_count=ab.SIZE_AWARE_OUTPUT_DIM):
    pos = torch.zeros(head_count, dtype=torch.float32)
    neg = torch.zeros(head_count, dtype=torch.float32)
    for sample in nonempty_graph_samples(samples):
        for node in sample["nodes"]:
            values = torch.tensor(
                [node.target_obj, node.target_small, node.target_large, node.target_roi],
                dtype=torch.float32,
            )[:head_count]
            positive = values > 0.5
            pos += positive.float()
            neg += (~positive).float()
    head_names = ["obj", "small", "large", "roi"]
    zero_heads = [head_names[index] for index, value in enumerate(pos.tolist()) if value <= 0]
    if zero_heads:
        print(f"[WARN] zero positives for GNN heads: {', '.join(zero_heads)}", flush=True)
    return (neg / pos.clamp_min(1.0)).clamp(1.0, 25.0)


def compute_global_stage2_action_loss_weight(samples):
    action_counts = torch.zeros(ab.STAGE2_ACTION_DIM, dtype=torch.float32)
    aux_pos = torch.zeros(ab.STAGE2_AUX_DIM, dtype=torch.float32)
    aux_neg = torch.zeros(ab.STAGE2_AUX_DIM, dtype=torch.float32)
    for sample in nonempty_graph_samples(samples):
        for node in sample["nodes"]:
            values = torch.tensor(ab.stage2_action_values(node), dtype=torch.float32)
            action = int(values[0].item())
            if 0 <= action < ab.STAGE2_ACTION_DIM:
                action_counts[action] += 1.0
            aux = values[1:3] > 0.5
            aux_pos += aux.float()
            aux_neg += (~aux).float()
    total = action_counts.sum().clamp_min(1.0)
    class_weight = (total / action_counts.clamp_min(1.0)).clamp(0.1, 25.0)
    aux_pos_weight = (aux_neg / aux_pos.clamp_min(1.0)).clamp(1.0, 25.0)
    action_names = ["keep", "duplicate", "fragment", "background"]
    zero_actions = [action_names[index] for index, value in enumerate(action_counts.tolist()) if value <= 0]
    if zero_actions:
        print(f"[WARN] zero samples for stage2 actions: {', '.join(zero_actions)}", flush=True)
    zero_aux = [name for name, value in zip(["roi_refine", "large_preserve"], aux_pos.tolist()) if value <= 0]
    if zero_aux:
        print(f"[WARN] zero positives for stage2 aux heads: {', '.join(zero_aux)}", flush=True)
    return torch.cat([class_weight, aux_pos_weight], dim=0)


def compute_global_stage2_size_loss_weight(samples):
    pos = torch.zeros(ab.STAGE2_SIZE_TARGET_DIM, dtype=torch.float32)
    neg = torch.zeros(ab.STAGE2_SIZE_TARGET_DIM, dtype=torch.float32)
    for sample in nonempty_graph_samples(samples):
        for node in sample["nodes"]:
            values = torch.tensor([node.target_small, node.target_large], dtype=torch.float32)
            positive = values > 0.5
            pos += positive.float()
            neg += (~positive).float()
    zero_heads = [name for name, value in zip(["small", "large"], pos.tolist()) if value <= 0]
    if zero_heads:
        print(f"[WARN] zero positives for stage2 size heads: {', '.join(zero_heads)}", flush=True)
    return (neg / pos.clamp_min(1.0)).clamp(1.0, 25.0)


def print_effective_gnn_schedule(train_pool, args, pos_weight):
    graphs = len(select_limited_items(train_pool, args.gnn_train_steps_per_epoch, args.seed + 1009))
    batch_size = max(1, int(args.gnn_batch_size)) if use_pyg_batching(args) else 1
    batches = int(math.ceil(graphs / batch_size)) if graphs else 0
    updates = int(math.ceil(batches / max(1, int(args.gnn_grad_accum_steps)))) if batches else 0
    pos_text = ", ".join(f"{float(value):.2f}" for value in pos_weight.tolist())
    print(
        f"[GNN schedule] pool={len(train_pool)} graphs/epoch={graphs} batch_size={batch_size} "
        f"batches/epoch={batches} optimizer_updates/epoch={updates} "
        f"fixed_pos_weight=[{pos_text}]",
        flush=True,
    )


def derived_roi_from_heads(obj_prob, small_prob, large_prob, node):
    if float(getattr(node, "union_area", 0.0)) >= ab.SMALL_MEDIUM_AREA_THR:
        return 0.0
    return float(max(0.0, min(1.0, obj_prob * small_prob * (1.0 - 0.5 * large_prob))))


def assign_size_aware_probs(node, prob):
    values = [float(value) for value in prob]
    if len(values) >= 4:
        obj_prob, small_prob, large_prob, roi_prob = values[:4]
    else:
        obj_prob, small_prob, large_prob = values[:3]
        roi_prob = derived_roi_from_heads(obj_prob, small_prob, large_prob, node)
    node.gnn_obj = obj_prob
    node.gnn_small = small_prob
    node.gnn_large = large_prob
    node.gnn_roi = roi_prob
    size_prob = large_prob if node.union_area >= ab.SMALL_MEDIUM_AREA_THR else small_prob
    node.stage2_roi_prob = roi_prob
    node.stage2_prob = float(max(obj_prob, size_prob, roi_prob))


def legacy3head_roi_like(obj_prob, small_prob):
    return float(max(float(small_prob), 0.7 * float(obj_prob) + 0.3 * float(small_prob)))


def assign_legacy3head_probs(node, prob):
    obj_prob, small_prob, large_prob = [float(value) for value in prob[:3]]
    roi_prob = legacy3head_roi_like(obj_prob, small_prob)
    node.gnn_obj = obj_prob
    node.gnn_small = small_prob
    node.gnn_large = large_prob
    node.gnn_roi = roi_prob
    node.stage2_roi_prob = roi_prob
    node.stage2_prob = float(max(obj_prob, small_prob, large_prob))


def attach_size_aware_scores_batched(samples, model, args, device, assign_fn=assign_size_aware_probs):
    valid = nonempty_graph_samples(samples)
    data_list = [sample_to_pyg_data(sample, args) for sample in tqdm(valid, desc="tensorize score graphs", leave=False)]
    loader = make_pyg_loader(data_list, args, shuffle=False)
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="score size-aware detection graph batches", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.sigmoid(logits[:, : ab.SIZE_AWARE_OUTPUT_DIM]).detach().cpu()
            counts = torch.bincount(batch.batch.cpu(), minlength=batch.num_graphs).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[sample_offset + local_index]
                graph_probs = probs[cursor : cursor + count].tolist()
                for node, prob in zip(sample["nodes"], graph_probs):
                    assign_fn(node, prob)
                cursor += count
            sample_offset += int(batch.num_graphs)


def attach_size_aware_scores_for_eval(samples, model, args, device, assign_fn=assign_size_aware_probs):
    if use_pyg_batching(args):
        attach_size_aware_scores_batched(samples, model, args, device, assign_fn=assign_fn)
    elif (
        assign_fn is assign_size_aware_probs
        and int(getattr(model, "size_aware_output_dim", ab.SIZE_AWARE_OUTPUT_DIM)) >= ab.SIZE_AWARE_OUTPUT_DIM
    ):
        ab.attach_size_aware_scores(samples, model, args, device)
    else:
        model.eval()
        with torch.no_grad():
            for sample in tqdm(nonempty_graph_samples(samples), desc="score no-ROI-head detection graphs", leave=False):
                x, edge_index, edge_attr, _, _ = ab.build_size_aware_tensors(sample["nodes"], sample["record"], args)
                probs = torch.sigmoid(model(x.to(device), edge_index.to(device), edge_attr.to(device))).detach().cpu().tolist()
                for node, prob in zip(sample["nodes"], probs):
                    assign_fn(node, prob)


def freeze_stage1_scores(samples):
    for sample in samples:
        for node in sample.get("nodes", []):
            node.stage1_obj = float(node.gnn_obj)
            node.stage1_small = float(node.gnn_small)
            node.stage1_large = float(node.gnn_large)
            node.stage1_roi = float(node.gnn_roi)


LEGACY_HYPER_NODE_TYPE_DIM = 3
LEGACY_HYPER_EDGE_TYPE_DIM = 5
LEGACY_HYPER_EDGE_SELF = 0
LEGACY_HYPER_EDGE_DET_DET = 1
LEGACY_HYPER_EDGE_DET_CLUSTER = 2
LEGACY_HYPER_EDGE_CLUSTER_CLASS = 3
LEGACY_HYPER_EDGE_CLASS_CLASS = 4
LEGACY_MAX_HYPER_CLUSTERS_PER_CLASS = 64
LEGACY_MAX_HYPER_MEMBERS_PER_CLUSTER = 48


def legacy_hyper_node_dim():
    return ab.SIZE_AWARE_NODE_DIM + LEGACY_HYPER_NODE_TYPE_DIM


def legacy_hyper_edge_dim():
    return ab.SIZE_AWARE_EDGE_DIM + LEGACY_HYPER_EDGE_TYPE_DIM


def legacy06_tensors(sample, args):
    x, edge_index, edge_attr, y, weights = ab.build_size_aware_tensors(sample["nodes"], sample["record"], args)
    return x, edge_index, edge_attr, y[:, :3].contiguous(), weights


def legacy_typed_node_feature(base, kind):
    if kind == "detection":
        suffix = [1.0, 0.0, 0.0]
    elif kind == "cluster":
        suffix = [0.0, 1.0, 0.0]
    elif kind == "class":
        suffix = [0.0, 0.0, 1.0]
    else:
        raise ValueError(kind)
    return torch.cat([base.float(), torch.tensor(suffix, dtype=torch.float32)], dim=0)


def legacy_typed_edge_attr(base, edge_type):
    if not torch.is_tensor(base):
        base = torch.tensor(base, dtype=torch.float32)
    suffix = torch.zeros((LEGACY_HYPER_EDGE_TYPE_DIM,), dtype=torch.float32)
    suffix[int(edge_type)] = 1.0
    return torch.cat([base.float(), suffix], dim=0)


def legacy_zero_edge_base():
    return torch.zeros((ab.SIZE_AWARE_EDGE_DIM,), dtype=torch.float32)


def legacy_class_base_feature(category_id, det_x, indices):
    if indices:
        base = det_x[indices].mean(dim=0).clone()
    else:
        base = torch.zeros((ab.SIZE_AWARE_NODE_DIM,), dtype=torch.float32)
    one_hot_start = ab.SIZE_AWARE_NODE_DIM - ab.NUM_CLASSES
    class_index = ab.VISDRONE_TO_CLASS_INDEX.get(int(category_id), 0)
    base[one_hot_start : one_hot_start + ab.NUM_CLASSES] = 0.0
    base[one_hot_start + class_index] = 1.0
    if base.numel() > 12:
        base[12] = min(1.0, len(indices) / 32.0)
    return base


def legacy_cluster_edge_base(node, cluster, args, is_member):
    node_center = np.asarray(ab.bbox_center(node.bbox), dtype=np.float32)
    cluster_center = np.asarray(ab.bbox_center(cluster["bbox"]), dtype=np.float32)
    dist = float(np.linalg.norm(node_center - cluster_center))
    dist_score = max(0.0, 1.0 - dist / max(1.0, float(args.graph_radius) * 1.5))
    node_area = max(1.0, float(node.union_area))
    cluster_area = max(1.0, float(cluster["area"]))
    area_ratio = math.log(node_area / cluster_area)
    scale_sim = max(0.0, 1.0 - abs(area_ratio) / 3.0)
    iou = ab.iou_xywh(node.bbox, cluster["bbox"])
    contain_nc = ab.containment(node.bbox, cluster["bbox"])
    contain_cn = ab.containment(cluster["bbox"], node.bbox)
    both_slice = 1.0 if node.source == ab.SOURCE_SLICE else 0.0
    both_full = 1.0 if node.source == ab.SOURCE_FULL else 0.0
    mean_score = 0.5 * (float(node.max_score) + float(cluster["score"]))
    node_w = max(1.0, float(node.bbox[2]))
    node_h = max(1.0, float(node.bbox[3]))
    cluster_w = max(1.0, float(cluster["bbox"][2]))
    cluster_h = max(1.0, float(cluster["bbox"][3]))
    dx = math.tanh((cluster_center[0] - node_center[0]) / math.sqrt(node_w * cluster_w))
    dy = math.tanh((cluster_center[1] - node_center[1]) / math.sqrt(node_h * cluster_h))
    return torch.tensor(
        [
            1.0,
            dist_score,
            scale_sim,
            iou,
            contain_nc,
            contain_cn,
            1.0 if is_member else 0.0,
            0.0,
            both_slice,
            both_full,
            mean_score,
            dx,
            dy,
            max(-3.0, min(3.0, area_ratio)) / 3.0,
            max(-3.0, min(3.0, math.log(node_w / cluster_w))) / 3.0,
            max(-3.0, min(3.0, math.log(node_h / cluster_h))) / 3.0,
        ],
        dtype=torch.float32,
    )


def legacy_build_hyper_clusters(nodes, det_x, args):
    groups = defaultdict(list)
    for index, node in enumerate(nodes):
        if node.source == ab.SOURCE_SLICE and int(node.dbscan_label) >= 0:
            groups[(int(node.category_id), f"dbscan_{int(node.dbscan_label)}")].append(index)
    clustered = {index for members in groups.values() for index in members}
    score_thr = float(getattr(args, "heuristic_cluster_score", 0.12))
    for index, node in enumerate(nodes):
        if index in clustered:
            continue
        small_like = float(node.union_area) <= ab.SMALL_MEDIUM_AREA_THR * 2.5
        if small_like and float(node.max_score) >= score_thr:
            groups[(int(node.category_id), f"single_{index}")].append(index)

    clusters = []
    for (category_id, key), members in groups.items():
        members = sorted(set(members))
        if not members:
            continue
        bbox = ab.union_xywh([nodes[index].bbox for index in members])
        base = det_x[members].mean(dim=0).clone()
        max_score = max(float(nodes[index].max_score) for index in members)
        if base.numel() > 0:
            base[0] = max_score
        if base.numel() > 11:
            base[11] = 1.0
        if base.numel() > 12:
            base[12] = min(1.0, len(members) / 16.0)
        score = min(1.0, max_score + 0.02 * min(len(members), 10))
        clusters.append(
            {
                "cluster_id": len(clusters),
                "category_id": int(category_id),
                "key": str(key),
                "members": members,
                "members_set": set(members),
                "bbox": bbox,
                "area": ab.bbox_area(bbox),
                "score": float(score),
                "feature_base": base,
                "target_obj": max(float(nodes[index].target_obj) for index in members),
                "target_small": max(float(nodes[index].target_small) for index in members),
                "target_large": max(float(nodes[index].target_large) for index in members),
            }
        )
    filtered = []
    for category_id in sorted(ab.YOLO_TO_VISDRONE.values()):
        class_clusters = [item for item in clusters if int(item["category_id"]) == int(category_id)]
        class_clusters.sort(key=lambda item: (float(item["score"]), len(item["members"])), reverse=True)
        filtered.extend(class_clusters[:LEGACY_MAX_HYPER_CLUSTERS_PER_CLASS])
    for index, cluster in enumerate(filtered):
        cluster["cluster_id"] = index
    return filtered


def legacy07_hypergraph_tensors(sample, args, training=True):
    nodes = sample["nodes"]
    if not nodes:
        empty_x = torch.empty((0, legacy_hyper_node_dim()), dtype=torch.float32)
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        empty_ea = torch.empty((0, legacy_hyper_edge_dim()), dtype=torch.float32)
        empty_y = torch.empty((0, 3), dtype=torch.float32)
        empty_w = torch.empty((0,), dtype=torch.float32)
        return empty_x, empty_ei, empty_ea, empty_y, empty_w, 0

    record = sample["record"]
    det_x, det_edge_index, det_edge_attr, det_targets, det_weights = ab.build_size_aware_tensors(nodes, record, args)
    features = [legacy_typed_node_feature(det_x[index], "detection") for index in range(det_x.shape[0])]
    targets = [det_targets[index, :3].float() for index in range(det_targets.shape[0])]
    weights = [det_weights[index].float() for index in range(det_weights.shape[0])]
    edge_pairs = []
    edge_attrs = []
    seen_edges = set()

    def add_edge(src, dst, base, edge_type):
        key = (int(src), int(dst), int(edge_type))
        if key in seen_edges:
            return
        seen_edges.add(key)
        edge_pairs.append((int(src), int(dst)))
        edge_attrs.append(legacy_typed_edge_attr(base, edge_type))

    for edge_pos in range(det_edge_index.shape[1]):
        src = int(det_edge_index[0, edge_pos])
        dst = int(det_edge_index[1, edge_pos])
        edge_type = LEGACY_HYPER_EDGE_SELF if src == dst else LEGACY_HYPER_EDGE_DET_DET
        add_edge(src, dst, det_edge_attr[edge_pos], edge_type)

    indices_by_class = defaultdict(list)
    for index, node in enumerate(nodes):
        indices_by_class[int(node.category_id)].append(index)

    clusters = legacy_build_hyper_clusters(nodes, det_x, args)
    for cluster in clusters:
        cluster_node_index = len(features)
        cluster["node_index"] = cluster_node_index
        features.append(legacy_typed_node_feature(cluster["feature_base"], "cluster"))
        targets.append(
            torch.tensor(
                [cluster["target_obj"], cluster["target_small"], cluster["target_large"]],
                dtype=torch.float32,
            )
        )
        weights.append(torch.tensor(2.0 if float(cluster["target_obj"]) > 0.5 else 0.75, dtype=torch.float32))
        add_edge(cluster_node_index, cluster_node_index, legacy_zero_edge_base(), LEGACY_HYPER_EDGE_SELF)

    class_node_by_category = {}
    for category_id in sorted(ab.YOLO_TO_VISDRONE.values()):
        class_node_index = len(features)
        class_node_by_category[int(category_id)] = class_node_index
        class_indices = indices_by_class.get(int(category_id), [])
        features.append(legacy_typed_node_feature(legacy_class_base_feature(category_id, det_x, class_indices), "class"))
        class_nodes = [nodes[index] for index in class_indices]
        target_obj = max([float(node.target_obj) for node in class_nodes], default=0.0)
        target_small = max([float(node.target_small) for node in class_nodes], default=0.0)
        target_large = max([float(node.target_large) for node in class_nodes], default=0.0)
        targets.append(torch.tensor([target_obj, target_small, target_large], dtype=torch.float32))
        weights.append(torch.tensor(1.0 if target_obj > 0.5 else 0.25, dtype=torch.float32))
        add_edge(class_node_index, class_node_index, legacy_zero_edge_base(), LEGACY_HYPER_EDGE_SELF)

    for cluster in clusters:
        cluster_index = int(cluster["node_index"])
        class_index = int(class_node_by_category[int(cluster["category_id"])])
        class_base = legacy_zero_edge_base()
        class_base[0] = 1.0
        class_base[10] = float(cluster["score"])
        add_edge(cluster_index, class_index, class_base, LEGACY_HYPER_EDGE_CLUSTER_CLASS)
        add_edge(class_index, cluster_index, class_base, LEGACY_HYPER_EDGE_CLUSTER_CLASS)

        same_class = indices_by_class.get(int(cluster["category_id"]), [])
        cluster_center = np.asarray(ab.bbox_center(cluster["bbox"]), dtype=np.float32)
        candidates = []
        for det_index in same_class:
            node = nodes[det_index]
            node_center = np.asarray(ab.bbox_center(node.bbox), dtype=np.float32)
            dist = float(np.linalg.norm(node_center - cluster_center))
            is_member = det_index in cluster["members_set"]
            near = dist <= float(args.graph_radius) * 1.5
            overlap = ab.iou_xywh(node.bbox, cluster["bbox"]) > 0.0 or ab.containment(node.bbox, cluster["bbox"]) > 0.05
            if is_member or near or overlap:
                dist_score = max(0.0, 1.0 - dist / max(1.0, float(args.graph_radius) * 1.5))
                candidates.append((det_index, is_member, dist_score, float(node.max_score)))
        candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        for det_index, is_member, _, _ in candidates[:LEGACY_MAX_HYPER_MEMBERS_PER_CLUSTER]:
            base = legacy_cluster_edge_base(nodes[det_index], cluster, args, is_member)
            add_edge(det_index, cluster_index, base, LEGACY_HYPER_EDGE_DET_CLUSTER)
            add_edge(cluster_index, det_index, base, LEGACY_HYPER_EDGE_DET_CLUSTER)

    class_edge_base = legacy_zero_edge_base()
    class_categories = sorted(class_node_by_category)
    for src_category in class_categories:
        for dst_category in class_categories:
            if src_category == dst_category:
                continue
            add_edge(
                class_node_by_category[src_category],
                class_node_by_category[dst_category],
                class_edge_base,
                LEGACY_HYPER_EDGE_CLASS_CLASS,
            )

    sample["_legacy07_clusters"] = [
        {
            "cluster_id": int(cluster["cluster_id"]),
            "category_id": int(cluster["category_id"]),
            "key": str(cluster["key"]),
            "members": [int(index) for index in cluster["members"]],
            "members_set": set(int(index) for index in cluster["members"]),
            "bbox": [float(value) for value in cluster["bbox"]],
            "area": float(cluster["area"]),
            "score": float(cluster["score"]),
            "node_index": int(cluster["node_index"]),
        }
        for cluster in clusters
    ]
    sample["_legacy07_class_node_by_category"] = {
        int(category_id): int(node_index)
        for category_id, node_index in class_node_by_category.items()
    }

    x = torch.stack(features, dim=0)
    y = torch.stack(targets, dim=0)
    weights_tensor = torch.stack(weights, dim=0)
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, legacy_hyper_edge_dim()), dtype=torch.float32)
    return x, edge_index, edge_attr, y, weights_tensor, len(nodes)


def legacy_sample_to_pyg_data(sample, args, hypergraph=False):
    key = "_legacy07_hyper_pyg_data" if hypergraph else "_legacy06_pyg_data"
    if bool(getattr(args, "cache_pyg_graphs", True)) and sample.get(key) is not None:
        return sample[key]
    if hypergraph:
        x, edge_index, edge_attr, y, weights, det_count = legacy07_hypergraph_tensors(sample, args, training=True)
    else:
        x, edge_index, edge_attr, y, weights = legacy06_tensors(sample, args)
        det_count = int(y.shape[0])
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, weights=weights)
    data.det_count = int(det_count)
    if bool(getattr(args, "cache_pyg_graphs", True)):
        sample[key] = data
    return data


def selected_legacy_graph_data(samples, limit, seed, args, desc, hypergraph=False):
    selected = select_limited_samples(samples, limit, seed)
    if not use_pyg_batching(args):
        return selected
    return [
        legacy_sample_to_pyg_data(sample, args, hypergraph=hypergraph)
        for sample in tqdm(selected, desc=desc, leave=False)
    ]


def prepare_legacy_graph_pool(samples, args, desc, hypergraph=False):
    valid = nonempty_graph_samples(samples)
    if not use_pyg_batching(args) or not bool(getattr(args, "gnn_precompute_graph_pool", True)):
        return valid
    return [
        legacy_sample_to_pyg_data(sample, args, hypergraph=hypergraph)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def legacy_loss(logits, targets, weights):
    return ab.weighted_size_aware_bce_loss(logits, targets, weights, pos_weight=None)


def legacy_loss_mean(model, graph_pool, args, device, hypergraph=False):
    if is_pyg_graph_pool(graph_pool):
        selected = select_limited_items(graph_pool, args.gnn_val_loss_limit, args.seed + 9301)
    else:
        selected = selected_legacy_graph_data(
            graph_pool,
            args.gnn_val_loss_limit,
            args.seed + 9301,
            args,
            "tensorize legacy val graphs",
            hypergraph=hypergraph,
        )
    model.eval()
    losses = []
    with torch.no_grad():
        if is_pyg_graph_pool(selected):
            loader = make_pyg_loader(selected, args, shuffle=False)
            total_loss = 0.0
            total_graphs = 0
            for batch in tqdm(loader, desc="legacy GNN val batches", leave=False):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = legacy_loss(logits, batch.y, batch.weights)
                graph_count = int(batch.num_graphs)
                total_loss += float(loss.detach().cpu()) * graph_count
                total_graphs += graph_count
            return total_loss / total_graphs if total_graphs else 0.0
        for sample in tqdm(selected, desc="legacy GNN val-loss", leave=False):
            if hypergraph:
                x, edge_index, edge_attr, y, weights, _ = legacy07_hypergraph_tensors(sample, args, training=True)
            else:
                x, edge_index, edge_attr, y, weights = legacy06_tensors(sample, args)
            logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            loss = legacy_loss(logits, y.to(device), weights.to(device))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def train_legacy_one_epoch(model, graph_pool, args, optimizer, device, epoch, hypergraph=False):
    model.train()
    if is_pyg_graph_pool(graph_pool):
        order = select_limited_items(graph_pool, args.gnn_train_steps_per_epoch, args.seed + epoch * 1009)
    else:
        order = selected_legacy_graph_data(
            graph_pool,
            args.gnn_train_steps_per_epoch,
            args.seed + epoch * 1009,
            args,
            "tensorize legacy train graphs",
            hypergraph=hypergraph,
        )
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    if is_pyg_graph_pool(order):
        loader = make_pyg_loader(order, args, shuffle=True)
        batch_index = 0
        for batch in tqdm(loader, desc="legacy GNN batch epoch", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = legacy_loss(logits, batch.y, batch.weights)
            (loss / accum_steps).backward()
            batch_index += 1
            if batch_index % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if losses and batch_index % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else 0.0
    for sample in tqdm(order, desc="legacy GNN epoch", leave=False):
        if hypergraph:
            x, edge_index, edge_attr, y, weights, _ = legacy07_hypergraph_tensors(sample, args, training=True)
        else:
            x, edge_index, edge_attr, y, weights = legacy06_tensors(sample, args)
        logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        loss = legacy_loss(logits, y.to(device), weights.to(device))
        (loss / accum_steps).backward()
        if (len(losses) + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def legacy07_apply_hyper_scores(sample, graph_probs, det_count):
    nodes = sample["nodes"]
    clusters = sample.get("_legacy07_clusters", [])
    class_node_by_category = sample.get("_legacy07_class_node_by_category", {})

    for node in nodes:
        node.hgnn_cluster_obj = 0.0
        node.hgnn_cluster_small = 0.0
        node.hgnn_cluster_large = 0.0
        node.hgnn_cluster_score = 0.0
        node.hgnn_cluster_roi = 0.0
        node.hgnn_class_obj = 0.0
        node.hgnn_class_small = 0.0
        node.hgnn_class_large = 0.0
        node.hgnn_class_score = 0.0

    scored_clusters = []
    for cluster in clusters:
        node_index = int(cluster.get("node_index", -1))
        if node_index < 0 or node_index >= len(graph_probs):
            continue
        prob = graph_probs[node_index]
        obj = float(prob[0])
        small = float(prob[1])
        large = float(prob[2])
        score = max(obj, small, large)
        roi_score = max(small, 0.7 * obj + 0.3 * small)
        cluster = dict(cluster)
        cluster.update(
            {
                "hgnn_obj": obj,
                "hgnn_small": small,
                "hgnn_large": large,
                "hgnn_score": score,
                "hgnn_roi": roi_score,
            }
        )
        scored_clusters.append(cluster)
        for det_index in cluster.get("members", []):
            if det_index < 0 or det_index >= len(nodes):
                continue
            node = nodes[int(det_index)]
            node.hgnn_cluster_obj = max(float(node.hgnn_cluster_obj), obj)
            node.hgnn_cluster_small = max(float(node.hgnn_cluster_small), small)
            node.hgnn_cluster_large = max(float(node.hgnn_cluster_large), large)
            node.hgnn_cluster_score = max(float(node.hgnn_cluster_score), score)
            node.hgnn_cluster_roi = max(float(node.hgnn_cluster_roi), roi_score)

    class_scores = {}
    for category_id, node_index in class_node_by_category.items():
        node_index = int(node_index)
        if node_index < 0 or node_index >= len(graph_probs):
            continue
        prob = graph_probs[node_index]
        obj = float(prob[0])
        small = float(prob[1])
        large = float(prob[2])
        class_scores[int(category_id)] = {
            "obj": obj,
            "small": small,
            "large": large,
            "score": max(obj, small, large),
        }

    for node in nodes:
        score = class_scores.get(int(node.category_id), {})
        node.hgnn_class_obj = float(score.get("obj", 0.0))
        node.hgnn_class_small = float(score.get("small", 0.0))
        node.hgnn_class_large = float(score.get("large", 0.0))
        node.hgnn_class_score = float(score.get("score", 0.0))

    sample["_legacy07_scored_clusters"] = scored_clusters
    sample["_legacy07_class_scores"] = class_scores


def legacy07_direct_node_scores(node, args):
    large_like = float(node.union_area) >= ab.SMALL_MEDIUM_AREA_THR
    node_size = float(node.gnn_large) if large_like else float(node.gnn_small)
    node_score = max(float(node.gnn_obj), node_size)

    cluster_size = float(getattr(node, "hgnn_cluster_large", 0.0)) if large_like else float(getattr(node, "hgnn_cluster_small", 0.0))
    cluster_score = max(float(getattr(node, "hgnn_cluster_obj", 0.0)), cluster_size)

    class_size = float(getattr(node, "hgnn_class_large", 0.0)) if large_like else float(getattr(node, "hgnn_class_small", 0.0))
    class_score = max(float(getattr(node, "hgnn_class_obj", 0.0)), class_size)

    fused_score = node_score
    cluster_alpha = float(getattr(args, "hgnn_cluster_score_alpha", 0.35))
    class_alpha = float(getattr(args, "hgnn_class_score_alpha", 0.15))
    if cluster_score >= float(getattr(args, "hgnn_cluster_keep_thr", 0.35)):
        fused_score = max(fused_score, cluster_score, (1.0 - cluster_alpha) * node_score + cluster_alpha * cluster_score)
    if class_score >= float(getattr(args, "hgnn_class_keep_thr", 0.25)):
        fused_score = max(fused_score, (1.0 - class_alpha) * fused_score + class_alpha * class_score)

    roi_score = max(
        float(getattr(node, "gnn_roi", 0.0)),
        float(getattr(node, "hgnn_cluster_roi", 0.0)),
        (1.0 - class_alpha) * float(getattr(node, "gnn_roi", 0.0)) + class_alpha * float(getattr(node, "hgnn_class_small", 0.0)),
    )
    large_score = max(
        float(node.gnn_large),
        (1.0 - cluster_alpha) * float(node.gnn_large) + cluster_alpha * float(getattr(node, "hgnn_cluster_large", 0.0)),
        (1.0 - class_alpha) * float(node.gnn_large) + class_alpha * float(getattr(node, "hgnn_class_large", 0.0)),
    )
    return {
        "node_score": float(node_score),
        "cluster_score": float(cluster_score),
        "class_score": float(class_score),
        "fused_score": float(fused_score),
        "roi_score": float(roi_score),
        "large_score": float(large_score),
    }


def attach_legacy_scores_batched(samples, model, args, device, hypergraph=False):
    valid = nonempty_graph_samples(samples)
    data_list = [
        legacy_sample_to_pyg_data(sample, args, hypergraph=hypergraph)
        for sample in tqdm(valid, desc="tensorize legacy score graphs", leave=False)
    ]
    loader = make_pyg_loader(data_list, args, shuffle=False)
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="score legacy graph batches", leave=False):
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).detach().cpu()
            counts = torch.bincount(batch.batch.cpu(), minlength=batch.num_graphs).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[sample_offset + local_index]
                det_count = len(sample["nodes"]) if not hypergraph else int(getattr(data_list[sample_offset + local_index], "det_count", len(sample["nodes"])))
                graph_probs = probs[cursor : cursor + int(count)].tolist()
                for node, prob in zip(sample["nodes"], graph_probs[:det_count]):
                    node.gnn_obj = float(prob[0])
                    node.gnn_small = float(prob[1])
                    node.gnn_large = float(prob[2])
                    node.gnn_roi = max(float(prob[1]), 0.7 * float(prob[0]) + 0.3 * float(prob[1]))
                    node.stage2_prob = max(float(prob[0]), float(prob[1]), float(prob[2]))
                    node.stage2_roi_prob = float(node.gnn_roi)
                if hypergraph:
                    legacy07_apply_hyper_scores(sample, graph_probs, det_count)
                cursor += count
            sample_offset += int(batch.num_graphs)


def attach_legacy_scores_for_eval(samples, model, args, device, hypergraph=False):
    if use_pyg_batching(args):
        attach_legacy_scores_batched(samples, model, args, device, hypergraph=hypergraph)
        return
    model.eval()
    with torch.no_grad():
        for sample in tqdm(nonempty_graph_samples(samples), desc="score legacy graphs", leave=False):
            if hypergraph:
                x, edge_index, edge_attr, _, _, det_count = legacy07_hypergraph_tensors(sample, args, training=False)
            else:
                x, edge_index, edge_attr, _, weights = legacy06_tensors(sample, args)
                det_count = int(weights.shape[0])
            probs = torch.sigmoid(model(x.to(device), edge_index.to(device), edge_attr.to(device))).detach().cpu().tolist()
            for node, prob in zip(sample["nodes"], probs[:det_count]):
                node.gnn_obj = float(prob[0])
                node.gnn_small = float(prob[1])
                node.gnn_large = float(prob[2])
                node.gnn_roi = max(float(prob[1]), 0.7 * float(prob[0]) + 0.3 * float(prob[1]))
                node.stage2_prob = max(float(prob[0]), float(prob[1]), float(prob[2]))
                node.stage2_roi_prob = float(node.gnn_roi)
            if hypergraph:
                legacy07_apply_hyper_scores(sample, probs, det_count)


def legacy06_rois(nodes, record, args):
    candidates = []
    for node in nodes:
        small_like = float(node.union_area) <= ab.SMALL_MEDIUM_AREA_THR * 2.5
        score = max(float(node.gnn_small), 0.7 * float(node.gnn_obj) + 0.3 * float(node.gnn_small))
        if small_like and score >= float(args.gnn_small_thr):
            candidates.append((ab.expand_xywh(node.bbox, record.width, record.height, args.roi_margin), score))
    return ab.merge_rois(candidates)[: int(args.max_stage2_rois)]


def legacy06_final_fusion(record, full_predictions, coarse_predictions, nodes, rois, args, fine_roi_predictions=None):
    final = []
    full_node_by_index = {node.det_index: node for node in nodes if node.source == ab.SOURCE_FULL}
    for index, pred in enumerate(full_predictions):
        node = full_node_by_index.get(index)
        gnn_score = 0.0 if node is None else float(node.gnn_large)
        if ab.bbox_area(pred["bbox"]) >= ab.SMALL_MEDIUM_AREA_THR and (
            float(pred["score"]) >= float(args.large_keep_conf)
            or gnn_score >= float(args.gnn_large_thr)
        ):
            adjusted = dict(pred)
            if gnn_score > 0.0:
                adjusted["score"] = float(max(float(adjusted["score"]), 0.55 * float(adjusted["score"]) + 0.45 * gnn_score))
            final.append(adjusted)

    slice_node_by_index = {node.det_index: node for node in nodes if node.source == ab.SOURCE_SLICE}
    for index, pred in enumerate(coarse_predictions):
        node = slice_node_by_index.get(index)
        if node is None:
            node_score = 0.0
        else:
            size_score = float(node.gnn_large) if float(node.union_area) >= ab.SMALL_MEDIUM_AREA_THR else float(node.gnn_small)
            node_score = max(float(node.gnn_obj), size_score)
        small_like = ab.bbox_area(pred["bbox"]) <= ab.SMALL_MEDIUM_AREA_THR * 2.5
        if float(pred["score"]) >= float(args.stage1_keep_conf) or (
            small_like and node_score >= float(args.gnn_object_thr)
        ):
            adjusted = dict(pred)
            if node_score > 0.0:
                adjusted["score"] = float(max(float(adjusted["score"]), 0.5 * float(adjusted["score"]) + 0.5 * node_score))
            final.append(adjusted)

    if fine_roi_predictions is None:
        fine_roi_predictions = []
    for pred in fine_roi_predictions:
        if ab.bbox_area(pred["bbox"]) <= ab.SMALL_MEDIUM_AREA_THR * 2.5:
            adjusted = dict(pred)
            adjusted.pop("_roi_score", None)
            final.append(adjusted)
    return ab.classwise_nms(final, iou_threshold=args.final_nms_iou)


def legacy07_hyper_rois(sample, record, args):
    candidates = []
    for node in sample["nodes"]:
        small_like = float(node.union_area) <= ab.SMALL_MEDIUM_AREA_THR * 2.5
        scores = legacy07_direct_node_scores(node, args)
        if small_like and scores["roi_score"] >= float(args.gnn_small_thr):
            candidates.append((ab.expand_xywh(node.bbox, record.width, record.height, args.roi_margin), scores["roi_score"]))

    for cluster in sample.get("_legacy07_scored_clusters", []):
        small_like = float(cluster.get("area", 0.0)) <= ab.SMALL_MEDIUM_AREA_THR * 3.0
        roi_score = float(cluster.get("hgnn_roi", 0.0))
        if small_like and roi_score >= float(getattr(args, "hgnn_cluster_roi_thr", 0.35)):
            candidates.append((ab.expand_xywh(cluster["bbox"], record.width, record.height, args.roi_margin), roi_score))

    return ab.merge_rois(candidates)[: int(args.max_stage2_rois)]


def legacy07_hyper_final_fusion(record, full_predictions, coarse_predictions, nodes, rois, args, fine_roi_predictions=None):
    final = []
    full_node_by_index = {node.det_index: node for node in nodes if node.source == ab.SOURCE_FULL}
    for index, pred in enumerate(full_predictions):
        node = full_node_by_index.get(index)
        large_score = 0.0 if node is None else legacy07_direct_node_scores(node, args)["large_score"]
        if ab.bbox_area(pred["bbox"]) >= ab.SMALL_MEDIUM_AREA_THR and (
            float(pred["score"]) >= float(args.large_keep_conf)
            or large_score >= float(args.gnn_large_thr)
        ):
            adjusted = dict(pred)
            if large_score > 0.0:
                adjusted["score"] = float(max(float(adjusted["score"]), 0.50 * float(adjusted["score"]) + 0.50 * large_score))
            final.append(adjusted)

    slice_node_by_index = {node.det_index: node for node in nodes if node.source == ab.SOURCE_SLICE}
    for index, pred in enumerate(coarse_predictions):
        node = slice_node_by_index.get(index)
        if node is None:
            scores = {"fused_score": 0.0, "node_score": 0.0, "cluster_score": 0.0, "class_score": 0.0}
        else:
            scores = legacy07_direct_node_scores(node, args)
        small_like = ab.bbox_area(pred["bbox"]) <= ab.SMALL_MEDIUM_AREA_THR * 2.5
        weak_cluster = (
            node is not None
            and small_like
            and float(getattr(node, "hgnn_cluster_score", 0.0)) > 0.0
            and float(getattr(node, "hgnn_cluster_score", 0.0)) < float(getattr(args, "hgnn_cluster_suppress_thr", 0.20))
            and float(pred["score"]) < float(args.stage1_keep_conf)
        )
        keep_by_hyper = small_like and (
            scores["fused_score"] >= float(args.gnn_object_thr)
            or scores["cluster_score"] >= float(getattr(args, "hgnn_cluster_keep_thr", 0.35))
            or scores["class_score"] >= float(getattr(args, "hgnn_class_keep_thr", 0.25))
        )
        if not weak_cluster and (float(pred["score"]) >= float(args.stage1_keep_conf) or keep_by_hyper):
            adjusted = dict(pred)
            if scores["fused_score"] > 0.0:
                adjusted["score"] = float(max(float(adjusted["score"]), 0.45 * float(adjusted["score"]) + 0.55 * scores["fused_score"]))
            final.append(adjusted)

    if fine_roi_predictions is None:
        fine_roi_predictions = []
    for pred in fine_roi_predictions:
        if ab.bbox_area(pred["bbox"]) <= ab.SMALL_MEDIUM_AREA_THR * 2.5:
            adjusted = dict(pred)
            roi_score = float(adjusted.pop("_roi_score", 0.0))
            if roi_score > 0.0:
                adjusted["score"] = float(max(float(adjusted["score"]), 0.45 * float(adjusted["score"]) + 0.55 * roi_score))
            final.append(adjusted)
    return ab.classwise_nms(final, iou_threshold=args.final_nms_iou)


def run_legacy06_full_aware_variant(model_for_fine, eval_samples, full_by_image, args):
    all_predictions = []
    plans = []
    for sample in tqdm(eval_samples, desc="prepare GNN fusion plans"):
        record = sample["record"]
        plans.append(
            {
                "record": record,
                "coarse_predictions": sample["predictions"],
                "full_predictions": full_by_image.get(record.image_id, []),
                "nodes": sample["nodes"],
                "rois": legacy06_rois(sample["nodes"], record, args),
                "fine_roi_keys": [],
            }
        )
    ab.prefetch_fine_roi_predictions(model_for_fine, plans, args)
    for plan in tqdm(plans, desc="GNN fusion"):
        all_predictions.extend(
            legacy06_final_fusion(
                plan["record"],
                plan["full_predictions"],
                plan["coarse_predictions"],
                plan["nodes"],
                plan["rois"],
                args,
                fine_roi_predictions=ab.fine_roi_predictions_for_plan(plan, args),
            )
        )
    ab.flush_fine_roi_cache(args, force=True)
    return all_predictions


def run_legacy07_hyper_full_aware_variant(model_for_fine, eval_samples, full_by_image, args):
    all_predictions = []
    plans = []
    for sample in tqdm(eval_samples, desc="prepare HyperGNN fusion plans"):
        record = sample["record"]
        plans.append(
            {
                "record": record,
                "coarse_predictions": sample["predictions"],
                "full_predictions": full_by_image.get(record.image_id, []),
                "nodes": sample["nodes"],
                "rois": legacy07_hyper_rois(sample, record, args),
                "fine_roi_keys": [],
            }
        )
    ab.prefetch_fine_roi_predictions(model_for_fine, plans, args)
    for plan in tqdm(plans, desc="HyperGNN fusion"):
        all_predictions.extend(
            legacy07_hyper_final_fusion(
                plan["record"],
                plan["full_predictions"],
                plan["coarse_predictions"],
                plan["nodes"],
                plan["rois"],
                args,
                fine_roi_predictions=ab.fine_roi_predictions_for_plan(plan, args),
            )
        )
    ab.flush_fine_roi_cache(args, force=True)
    return all_predictions


def legacy_node_rerank_score(node, args):
    if bool(getattr(node, "stage2_size_scored", False)):
        obj_score = float(getattr(node, "stage1_obj", node.gnn_obj))
        small_score = float(getattr(node, "stage2_small", getattr(node, "stage1_small", node.gnn_small)))
        large_score = float(getattr(node, "stage2_large", getattr(node, "stage1_large", node.gnn_large)))
        size_score = large_score if float(node.union_area) >= ab.SMALL_MEDIUM_AREA_THR else small_score
        return max(obj_score, size_score)
    if bool(getattr(node, "stage2_action_scored", False)):
        keep = float(getattr(node, "stage2_keep", 0.0))
        background = float(getattr(node, "stage2_background", 0.0))
        return max(0.0, min(1.0, keep * (1.0 - background)))
    size_score = float(node.gnn_large) if float(node.union_area) >= ab.SMALL_MEDIUM_AREA_THR else float(node.gnn_small)
    return max(float(node.gnn_obj), size_score)


def run_legacy_gnn_rerank_gois_variant(eval_samples, args):
    all_predictions = []
    threshold = float(args.stage1_keep_conf)
    alpha = min(1.0, max(0.0, float(getattr(args, "gnn_score_alpha", 1.0))))
    for sample in tqdm(eval_samples, desc="GNN rerank -> 0.25 cut -> GOIS NMS"):
        source_predictions = sample.get("predictions", [])
        image_predictions = []
        for node in sample["nodes"]:
            if node.source != ab.SOURCE_SLICE:
                continue
            index = int(node.det_index)
            if index < 0 or index >= len(source_predictions):
                continue
            pred = source_predictions[index]
            detector_score = float(pred.get("score", 0.0))
            gnn_score = legacy_node_rerank_score(node, args)
            reranked_score = (1.0 - alpha) * detector_score + alpha * gnn_score
            reranked_score = max(0.0, min(1.0, float(reranked_score)))
            if reranked_score < threshold:
                continue
            adjusted = dict(pred)
            adjusted["score"] = reranked_score
            image_predictions.append(adjusted)
        all_predictions.extend(
            ab.classwise_nms(image_predictions, iou_threshold=args.final_nms_iou, limit=args.max_det)
        )
    return all_predictions


def run_size_aware_fixed_candidate_variant(eval_samples, args, prune_after_rerank=False):
    all_predictions = []
    alpha = min(1.0, max(0.0, float(getattr(args, "gnn_score_alpha", 0.25))))
    threshold = float(getattr(args, "stage1_keep_conf", 0.25))
    desc = "fixed-candidate GNN rerank prune" if prune_after_rerank else "fixed-candidate GNN score fusion"
    for sample in tqdm(eval_samples, desc=desc):
        node_scores = ab.size_aware_detection_node_scores(sample["nodes"], prefix="gnn")
        for index, pred in enumerate(sample.get("predictions", [])):
            adjusted = dict(pred)
            node_score = float(node_scores.get(index, 0.0))
            detector_score = float(adjusted.get("score", 0.0))
            fused_score = detector_score
            if node_score > 0.0 and alpha > 0.0:
                fused_score = (1.0 - alpha) * detector_score + alpha * node_score
            if prune_after_rerank:
                if fused_score < threshold:
                    continue
                adjusted["score"] = float(max(0.0, min(1.0, fused_score)))
            elif node_score > 0.0 and alpha > 0.0:
                adjusted["score"] = float(max(detector_score, fused_score))
            all_predictions.append(adjusted)
    return all_predictions


def legacy3head_stage_score(node):
    return float(
        max(
            float(getattr(node, "gnn_obj", 0.0)),
            float(getattr(node, "gnn_small", 0.0)),
            float(getattr(node, "gnn_large", 0.0)),
        )
    )


def run_legacy3head_fixed_candidate_variant(eval_samples, args):
    all_predictions = []
    for sample in tqdm(eval_samples, desc="fixed-candidate legacy 3-head fusion"):
        slice_node_by_index = {
            int(node.det_index): node
            for node in sample.get("nodes", [])
            if node.source == ab.SOURCE_SLICE
        }
        for index, pred in enumerate(sample.get("predictions", [])):
            adjusted = dict(pred)
            node = slice_node_by_index.get(int(index))
            if node is not None:
                node_score = legacy3head_stage_score(node)
                detector_score = float(adjusted.get("score", 0.0))
                adjusted["score"] = float(max(detector_score, 0.5 * detector_score + 0.5 * node_score))
            all_predictions.append(adjusted)
    return all_predictions


def run_legacy3head_rerank_prune_fixed_candidate_variant(eval_samples, args):
    all_predictions = []
    alpha = min(1.0, max(0.0, float(getattr(args, "gnn_score_alpha", 0.25))))
    threshold = float(getattr(args, "stage1_keep_conf", 0.25))
    for sample in tqdm(eval_samples, desc="09 legacy 3-head rerank prune"):
        slice_node_by_index = {
            int(node.det_index): node
            for node in sample.get("nodes", [])
            if node.source == ab.SOURCE_SLICE
        }
        for index, pred in enumerate(sample.get("predictions", [])):
            adjusted = dict(pred)
            node = slice_node_by_index.get(int(index))
            detector_score = float(adjusted.get("score", 0.0))
            node_score = legacy3head_stage_score(node) if node is not None else 0.0
            fused_score = detector_score
            if node_score > 0.0 and alpha > 0.0:
                fused_score = (1.0 - alpha) * detector_score + alpha * node_score
            if fused_score < threshold:
                continue
            adjusted["score"] = float(max(0.0, min(1.0, fused_score)))
            all_predictions.append(adjusted)
    return all_predictions


def run_legacy_eval_predictions(model_for_fine, eval_samples, full_by_image, args, hypergraph=False, prediction_mode="fusion"):
    if prediction_mode == "rerank_gois":
        return run_legacy_gnn_rerank_gois_variant(eval_samples, args)
    if prediction_mode == "size_aware_fusion":
        return ab.run_size_aware_gnn_large_preserve_variant(model_for_fine, eval_samples, full_by_image, args)
    if hypergraph:
        return run_legacy07_hyper_full_aware_variant(model_for_fine, eval_samples, full_by_image, args)
    return run_legacy06_full_aware_variant(model_for_fine, eval_samples, full_by_image, args)


def sample_to_stage2_pyg_data(sample, args, hypergraph):
    key = "_stage2_hgnn_data" if hypergraph else "_stage2_gnn_data"
    if bool(getattr(args, "cache_pyg_graphs", True)) and sample.get(key) is not None:
        return sample[key]
    x, edge_index, edge_attr, y, weights = ab.build_stage2_tensors(
        sample["nodes"],
        sample["record"],
        args,
        hypergraph=hypergraph,
    )
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, weights=weights)
    if bool(getattr(args, "cache_pyg_graphs", True)):
        sample[key] = data
    return data


def selected_stage2_graph_data(samples, limit, seed, args, desc, hypergraph):
    selected = select_limited_samples(samples, limit, seed)
    if not use_pyg_batching(args):
        return selected
    return [
        sample_to_stage2_pyg_data(sample, args, hypergraph)
        for sample in tqdm(selected, desc=desc, leave=False)
    ]


def prepare_stage2_graph_pool(samples, args, desc, hypergraph):
    valid = nonempty_graph_samples(samples)
    if not use_pyg_batching(args) or not bool(getattr(args, "gnn_precompute_graph_pool", True)):
        return valid
    return [
        sample_to_stage2_pyg_data(sample, args, hypergraph)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def sample_to_stage2_size_pyg_data(sample, args, hypergraph):
    key = "_stage2_size_hgnn_data" if hypergraph else "_stage2_size_gnn_data"
    if bool(getattr(args, "cache_pyg_graphs", True)) and sample.get(key) is not None:
        return sample[key]
    x, edge_index, edge_attr, y, weights = ab.build_stage2_size_tensors(
        sample["nodes"],
        sample["record"],
        args,
        hypergraph=hypergraph,
    )
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, weights=weights)
    if bool(getattr(args, "cache_pyg_graphs", True)):
        sample[key] = data
    return data


def selected_stage2_size_graph_data(samples, limit, seed, args, desc, hypergraph):
    selected = select_limited_samples(samples, limit, seed)
    if not use_pyg_batching(args):
        return selected
    return [
        sample_to_stage2_size_pyg_data(sample, args, hypergraph)
        for sample in tqdm(selected, desc=desc, leave=False)
    ]


def prepare_stage2_size_graph_pool(samples, args, desc, hypergraph):
    valid = nonempty_graph_samples(samples)
    if not use_pyg_batching(args) or not bool(getattr(args, "gnn_precompute_graph_pool", True)):
        return valid
    return [
        sample_to_stage2_size_pyg_data(sample, args, hypergraph)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def train_stage2_one_epoch(model, graph_pool, args, optimizer, device, epoch, pos_weight, hypergraph):
    model.train()
    if is_pyg_graph_pool(graph_pool):
        order = select_limited_items(graph_pool, args.gnn_train_steps_per_epoch, args.seed + epoch * 2003)
    else:
        order = selected_stage2_graph_data(
            graph_pool,
            args.gnn_train_steps_per_epoch,
            args.seed + epoch * 2003,
            args,
            "tensorize stage2 train graphs",
            hypergraph,
        )
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    if is_pyg_graph_pool(order):
        loader = make_pyg_loader(order, args, shuffle=True)
        batch_index = 0
        for batch in tqdm(loader, desc="stage2 GNN batch epoch", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = ab.stage2_action_refiner_loss(logits, batch.y, batch.weights, loss_weight=pos_weight.to(device))
            (loss / accum_steps).backward()
            batch_index += 1
            if batch_index % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if losses and batch_index % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else 0.0

    for sample in tqdm(order, desc="stage2 GNN epoch", leave=False):
        x, edge_index, edge_attr, y, weights = ab.build_stage2_tensors(
            sample["nodes"],
            sample["record"],
            args,
            hypergraph=hypergraph,
        )
        logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        loss = ab.stage2_action_refiner_loss(logits, y.to(device), weights.to(device), loss_weight=pos_weight.to(device))
        (loss / accum_steps).backward()
        if (len(losses) + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def stage2_loss_mean(model, graph_pool, args, device, pos_weight, hypergraph):
    losses = []
    if is_pyg_graph_pool(graph_pool):
        selected = select_limited_items(graph_pool, args.gnn_val_loss_limit, args.seed + 19301)
    else:
        selected = selected_stage2_graph_data(
            graph_pool,
            args.gnn_val_loss_limit,
            args.seed + 19301,
            args,
            "tensorize stage2 val graphs",
            hypergraph,
        )
    model.eval()
    with torch.no_grad():
        if is_pyg_graph_pool(selected):
            loader = make_pyg_loader(selected, args, shuffle=False)
            total_loss = 0.0
            total_graphs = 0
            for batch in tqdm(loader, desc="stage2 GNN val batches", leave=False):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = ab.stage2_action_refiner_loss(logits, batch.y, batch.weights, loss_weight=pos_weight.to(device))
                graph_count = int(batch.num_graphs)
                total_loss += float(loss.detach().cpu()) * graph_count
                total_graphs += graph_count
            return total_loss / total_graphs if total_graphs else 0.0
        for sample in tqdm(selected, desc="stage2 GNN val-loss", leave=False):
            x, edge_index, edge_attr, y, weights = ab.build_stage2_tensors(
                sample["nodes"],
                sample["record"],
                args,
                hypergraph=hypergraph,
            )
            logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            loss = ab.stage2_action_refiner_loss(logits, y.to(device), weights.to(device), loss_weight=pos_weight.to(device))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def train_stage2_size_one_epoch(model, graph_pool, args, optimizer, device, epoch, pos_weight, hypergraph):
    model.train()
    if is_pyg_graph_pool(graph_pool):
        order = select_limited_items(graph_pool, args.gnn_train_steps_per_epoch, args.seed + epoch * 3001)
    else:
        order = selected_stage2_size_graph_data(
            graph_pool,
            args.gnn_train_steps_per_epoch,
            args.seed + epoch * 3001,
            args,
            "tensorize stage2 size train graphs",
            hypergraph,
        )
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    if is_pyg_graph_pool(order):
        loader = make_pyg_loader(order, args, shuffle=True)
        batch_index = 0
        for batch in tqdm(loader, desc="stage2 size GNN batch epoch", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = ab.stage2_size_refiner_loss(logits, batch.y, batch.weights, pos_weight=pos_weight.to(device))
            (loss / accum_steps).backward()
            batch_index += 1
            if batch_index % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if losses and batch_index % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else 0.0

    for sample in tqdm(order, desc="stage2 size GNN epoch", leave=False):
        x, edge_index, edge_attr, y, weights = ab.build_stage2_size_tensors(
            sample["nodes"],
            sample["record"],
            args,
            hypergraph=hypergraph,
        )
        logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        loss = ab.stage2_size_refiner_loss(logits, y.to(device), weights.to(device), pos_weight=pos_weight.to(device))
        (loss / accum_steps).backward()
        if (len(losses) + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def stage2_size_loss_mean(model, graph_pool, args, device, pos_weight, hypergraph):
    losses = []
    if is_pyg_graph_pool(graph_pool):
        selected = select_limited_items(graph_pool, args.gnn_val_loss_limit, args.seed + 29301)
    else:
        selected = selected_stage2_size_graph_data(
            graph_pool,
            args.gnn_val_loss_limit,
            args.seed + 29301,
            args,
            "tensorize stage2 size val graphs",
            hypergraph,
        )
    model.eval()
    with torch.no_grad():
        if is_pyg_graph_pool(selected):
            loader = make_pyg_loader(selected, args, shuffle=False)
            total_loss = 0.0
            total_graphs = 0
            for batch in tqdm(loader, desc="stage2 size GNN val batches", leave=False):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = ab.stage2_size_refiner_loss(logits, batch.y, batch.weights, pos_weight=pos_weight.to(device))
                graph_count = int(batch.num_graphs)
                total_loss += float(loss.detach().cpu()) * graph_count
                total_graphs += graph_count
            return total_loss / total_graphs if total_graphs else 0.0
        for sample in tqdm(selected, desc="stage2 size GNN val-loss", leave=False):
            x, edge_index, edge_attr, y, weights = ab.build_stage2_size_tensors(
                sample["nodes"],
                sample["record"],
                args,
                hypergraph=hypergraph,
            )
            logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            loss = ab.stage2_size_refiner_loss(logits, y.to(device), weights.to(device), pos_weight=pos_weight.to(device))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def apply_stage2_size_probs(node, prob):
    small, large = [float(value) for value in prob[: ab.STAGE2_SIZE_OUTPUT_DIM]]
    node.stage2_small = small
    node.stage2_large = large
    node.stage2_size_scored = True
    node.stage2_action_scored = False
    node.gnn_obj = float(getattr(node, "stage1_obj", node.gnn_obj))
    node.gnn_small = small
    node.gnn_large = large
    node.gnn_roi = float(getattr(node, "stage1_roi", node.gnn_roi))
    size_prob = large if float(node.union_area) >= ab.SMALL_MEDIUM_AREA_THR else small
    node.stage2_roi_prob = float(node.gnn_roi)
    node.stage2_prob = float(max(node.gnn_obj, size_prob, node.gnn_roi))


def attach_stage2_size_scores_batched(samples, model, args, device, hypergraph):
    valid = nonempty_graph_samples(samples)
    data_list = [
        sample_to_stage2_size_pyg_data(sample, args, hypergraph)
        for sample in tqdm(valid, desc="tensorize stage2 size score graphs", leave=False)
    ]
    loader = make_pyg_loader(data_list, args, shuffle=False)
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="score stage2 size graph batches", leave=False):
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).detach().cpu()
            counts = torch.bincount(batch.batch.cpu(), minlength=batch.num_graphs).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[sample_offset + local_index]
                node_count = len(sample["nodes"])
                graph_probs = probs[cursor : cursor + node_count].tolist()
                for node, prob in zip(sample["nodes"], graph_probs):
                    apply_stage2_size_probs(node, prob)
                cursor += count
            sample_offset += int(batch.num_graphs)


def attach_stage2_size_scores_for_eval(samples, model, args, device, hypergraph):
    if use_pyg_batching(args):
        attach_stage2_size_scores_batched(samples, model, args, device, hypergraph)
        return
    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="score stage2 size graphs"):
            nodes = sample["nodes"]
            if not nodes:
                continue
            x, edge_index, edge_attr, _, _ = ab.build_stage2_size_tensors(
                nodes,
                sample["record"],
                args,
                hypergraph=hypergraph,
            )
            probs = torch.sigmoid(model(x.to(device), edge_index.to(device), edge_attr.to(device))).detach().cpu()
            for node, prob in zip(nodes, probs[: len(nodes)].tolist()):
                apply_stage2_size_probs(node, prob)


def attach_stage2_scores_batched(samples, model, args, device, hypergraph):
    valid = nonempty_graph_samples(samples)
    data_list = [
        sample_to_stage2_pyg_data(sample, args, hypergraph)
        for sample in tqdm(valid, desc="tensorize stage2 score graphs", leave=False)
    ]
    loader = make_pyg_loader(data_list, args, shuffle=False)
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="score stage2 graph batches", leave=False):
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr)).detach().cpu()
            counts = torch.bincount(batch.batch.cpu(), minlength=batch.num_graphs).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[sample_offset + local_index]
                node_count = len(sample["nodes"])
                graph_probs = probs[cursor : cursor + node_count].tolist()
                for node, prob in zip(sample["nodes"], graph_probs):
                    keep, duplicate, fragment, background, roi_refine, large_preserve = [float(value) for value in prob]
                    node.stage2_keep = keep
                    node.stage2_duplicate = duplicate
                    node.stage2_fragment = fragment
                    node.stage2_background = background
                    node.stage2_roi_refine = roi_refine
                    node.stage2_large_preserve = large_preserve
                    node.stage2_action_scored = True
                    node.gnn_obj = keep
                    node.gnn_small = keep * (1.0 - background)
                    node.gnn_large = large_preserve
                    node.gnn_roi = roi_refine
                    node.stage2_roi_prob = roi_refine
                    node.stage2_prob = float(max(node.gnn_small, roi_refine, large_preserve))
                cursor += count
            sample_offset += int(batch.num_graphs)


def attach_stage2_scores_for_eval(samples, model, args, device, hypergraph):
    if use_pyg_batching(args):
        attach_stage2_scores_batched(samples, model, args, device, hypergraph)
        return
    model.eval()
    with torch.no_grad():
        for sample in tqdm(samples, desc="score stage2 graphs"):
            nodes = sample["nodes"]
            if not nodes:
                continue
            x, edge_index, edge_attr, _, _ = ab.build_stage2_tensors(
                nodes,
                sample["record"],
                args,
                hypergraph=hypergraph,
            )
            logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            action_probs = torch.softmax(logits[:, : ab.STAGE2_ACTION_DIM], dim=-1)
            aux_probs = torch.sigmoid(logits[:, ab.STAGE2_ACTION_DIM : ab.STAGE2_ACTION_DIM + ab.STAGE2_AUX_DIM])
            probs = torch.cat([action_probs, aux_probs], dim=-1).detach().cpu()
            for node, prob in zip(nodes, probs[: len(nodes)].tolist()):
                keep, duplicate, fragment, background, roi_refine, large_preserve = [float(value) for value in prob]
                node.stage2_keep = keep
                node.stage2_duplicate = duplicate
                node.stage2_fragment = fragment
                node.stage2_background = background
                node.stage2_roi_refine = roi_refine
                node.stage2_large_preserve = large_preserve
                node.stage2_action_scored = True
                node.gnn_obj = keep
                node.gnn_small = keep * (1.0 - background)
                node.gnn_large = large_preserve
                node.gnn_roi = roi_refine
                node.stage2_roi_prob = roi_refine
                node.stage2_prob = float(max(node.gnn_small, roi_refine, large_preserve))


def size_aware_loss_mean(model, graph_pool, args, device, pos_weight, head_count=ab.SIZE_AWARE_OUTPUT_DIM):
    losses = []
    if is_pyg_graph_pool(graph_pool):
        selected = select_limited_items(graph_pool, args.gnn_val_loss_limit, args.seed + 9301)
    else:
        selected = selected_graph_data(
            graph_pool,
            args.gnn_val_loss_limit,
            args.seed + 9301,
            args,
            "tensorize val graphs",
        )
    model.eval()
    with torch.no_grad():
        if is_pyg_graph_pool(selected):
            loader = make_pyg_loader(selected, args, shuffle=False)
            total_loss = 0.0
            total_graphs = 0
            for batch in tqdm(loader, desc="size-aware GNN val batches", leave=False):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = ab.weighted_size_aware_bce_loss(
                    logits[:, :head_count],
                    batch.y[:, :head_count],
                    batch.weights,
                    pos_weight=pos_weight.to(device),
                )
                graph_count = int(batch.num_graphs)
                total_loss += float(loss.detach().cpu()) * graph_count
                total_graphs += graph_count
            return total_loss / total_graphs if total_graphs else 0.0
        for sample in tqdm(selected, desc="size-aware GNN val-loss", leave=False):
            x, edge_index, edge_attr, y, weights = ab.build_size_aware_tensors(sample["nodes"], sample["record"], args)
            logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
            loss = ab.weighted_size_aware_bce_loss(
                logits[:, :head_count],
                y[:, :head_count].to(device),
                weights.to(device),
                pos_weight=pos_weight.to(device),
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def train_size_aware_one_epoch(model, graph_pool, args, optimizer, device, epoch, pos_weight, head_count=ab.SIZE_AWARE_OUTPUT_DIM):
    model.train()
    if is_pyg_graph_pool(graph_pool):
        order = select_limited_items(graph_pool, args.gnn_train_steps_per_epoch, args.seed + epoch * 1009)
    else:
        order = selected_graph_data(
            graph_pool,
            args.gnn_train_steps_per_epoch,
            args.seed + epoch * 1009,
            args,
            "tensorize train graphs",
        )
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    if is_pyg_graph_pool(order):
        loader = make_pyg_loader(order, args, shuffle=True)
        batch_index = 0
        for batch in tqdm(loader, desc="size-aware GNN batch epoch", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = ab.weighted_size_aware_bce_loss(
                logits[:, :head_count],
                batch.y[:, :head_count],
                batch.weights,
                pos_weight=pos_weight.to(device),
            )
            (loss / accum_steps).backward()
            batch_index += 1
            if batch_index % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if losses and batch_index % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else 0.0
    for sample in tqdm(order, desc="size-aware GNN epoch", leave=False):
        x, edge_index, edge_attr, y, weights = ab.build_size_aware_tensors(sample["nodes"], sample["record"], args)
        logits = model(x.to(device), edge_index.to(device), edge_attr.to(device))
        loss = ab.weighted_size_aware_bce_loss(
            logits[:, :head_count],
            y[:, :head_count].to(device),
            weights.to(device),
            pos_weight=pos_weight.to(device),
        )
        (loss / accum_steps).backward()
        if (len(losses) + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def run_size_aware_curve(
    table_dir,
    variant,
    train_samples,
    eval_samples,
    full_by_image,
    model_for_fine,
    args,
    device,
    output_dim=ab.SIZE_AWARE_OUTPUT_DIM,
    score_assign_fn=assign_size_aware_probs,
    prediction_fn=run_size_aware_fixed_candidate_variant,
    mode="size_aware_detection_gnn",
):
    variant_dir = Path(table_dir) / artifact_variant_id(variant)
    history_path = variant_dir / "metrics_history.csv"
    ckpt_dir = variant_dir / "checkpoints"
    rows = []
    completed_epoch = 0
    config_ok = variant_config_matches(variant_dir, variant, args, mode=mode)
    may_reuse_variant_state = args.reuse_variant_outputs or args.resume_train
    if may_reuse_variant_state and history_path.exists() and config_ok and not args.force_train and not args.force_predictions:
        rows = read_rows(history_path)
        completed_epoch = completed_epoch_from_rows(rows)
        selection_path = variant_dir / "selection_evaluations.csv"
        if args.reuse_variant_outputs and completed_epoch >= args.epochs and (not args.eval_best_val_loss or selection_path.exists()):
            print(f"[{variant}] reuse history: {history_path}", flush=True)
            return
        if args.reuse_variant_outputs and completed_epoch < args.epochs and not args.resume_train:
            print(
                f"[{variant}] incomplete history found, but --resume_train is off. Restarting fresh.",
                flush=True,
            )
            clear_stale_variant_outputs(variant_dir)
            rows = []
            completed_epoch = 0
    elif history_path.exists() and not config_ok:
        print(f"[{variant}] stale size-aware cache ignored because code/config changed", flush=True)
        clear_stale_variant_outputs(variant_dir)
    elif history_path.exists() and not may_reuse_variant_state:
        print(f"[{variant}] fresh size-aware run clears previous generated summaries", flush=True)
        clear_stale_variant_outputs(variant_dir)
    write_variant_config(variant_dir, variant, args, mode=mode)

    model = ab.SizeAwareGraphGNN(
        ab.SIZE_AWARE_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        args.gnn_hidden_dim,
        args.gnn_layers,
        output_dim=output_dim,
    ).to(device)
    model.size_aware_output_dim = int(output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_global_size_aware_pos_weight(train_samples, head_count=output_dim)
    train_graph_pool = prepare_graph_pool(train_samples, args, "tensorize train graph pool")
    eval_graph_pool = prepare_graph_pool(eval_samples, args, "tensorize val graph pool")
    print_effective_gnn_schedule(train_graph_pool, args, pos_weight)
    if completed_epoch and args.resume_train and not args.force_train:
        loaded_epoch = load_single_stage_resume_state(model, optimizer, ckpt_dir, completed_epoch, device)
        if loaded_epoch < completed_epoch:
            rows = rows_through_epoch(rows, loaded_epoch)
            completed_epoch = loaded_epoch
            print(f"[{variant}] trimmed history to epoch {completed_epoch} for checkpoint match", flush=True)

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_loss = train_size_aware_one_epoch(
            model,
            train_graph_pool,
            args,
            optimizer,
            device,
            epoch,
            pos_weight,
            head_count=output_dim,
        )
        val_loss = size_aware_loss_mean(model, eval_graph_pool, args, device, pos_weight, head_count=output_dim)
        if should_full_eval_epoch(epoch, args):
            attach_size_aware_scores_for_eval(eval_samples, model, args, device, assign_fn=score_assign_fn)
            predictions = prediction_fn(eval_samples, args)
            row = record_epoch(variant_dir, variant, epoch, train_loss, val_loss, predictions, args.eval_gt_path)
        else:
            row = record_loss_only_epoch(variant, epoch, train_loss, val_loss)
        rows.append(row)
        save_history(variant_dir, rows)
        ab.ensure_dir(ckpt_dir)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"AP={float(row['AP']):.4f} " if has_full_eval(row) else f"[{variant}] epoch {epoch}/{args.epochs} AP=skip ",
            end="",
            flush=True,
        )
        print(f"loss={train_loss:.4f}/{val_loss:.4f}", flush=True)

    selection_items = selection_candidates(rows, args) if args.eval_best_val_loss else []
    if selection_items:
        for selected_epoch in sorted({epoch for _, epoch in selection_items}):
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is None or has_full_eval(selected_row):
                continue
            print(f"[{variant}] full-evaluating selected checkpoint at epoch {selected_epoch}", flush=True)
            if load_single_stage_model_at_epoch(model, ckpt_dir, selected_epoch, device):
                attach_size_aware_scores_for_eval(eval_samples, model, args, device, assign_fn=score_assign_fn)
                predictions = prediction_fn(eval_samples, args)
                evaluated = record_epoch(
                    variant_dir,
                    variant,
                    selected_epoch,
                    selected_row.get("train_loss"),
                    selected_row.get("val_loss"),
                    predictions,
                    args.eval_gt_path,
                )
                rows = replace_history_row(rows, evaluated)
                save_history(variant_dir, rows)
        selection_rows = []
        for label, selected_epoch in selection_items:
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is not None:
                selection_rows.append(dict(selected_row, selection=label))
        if selection_rows:
            write_rows(
                variant_dir / "selection_evaluations.csv",
                selection_rows,
                ["selection", "epoch", "variant", "num_predictions", "train_loss", "val_loss"] + METRIC_NAMES,
            )


def predictions_to_string_image_cache(predictions):
    cache = defaultdict(list)
    for pred in predictions:
        cache[str(int(pred["image_id"]))].append(pred)
    return cache


def run_same_pool_rerank_prune_curve(table_dir, eval_records, eval_gt, args, device):
    variant = SAME_POOL_RERANK_PRUNE_VARIANT
    source_variant = GNN_NO_CLUSTER_VARIANT
    source_dir = Path(table_dir) / source_variant
    source_history_path = source_dir / "metrics_history.csv"
    if not source_history_path.exists():
        raise FileNotFoundError(f"{variant} requires completed {source_variant}: {source_history_path}")
    source_rows = read_rows(source_history_path)
    source_epoch = int(getattr(args, "stage2_checkpoint_epoch", 0) or 0) or best_ap_epoch(source_rows)
    variant_args = clone_args_with(
        args,
        size_graph_cluster_mode="none",
        stage2_source_variant=source_variant,
        stage2_checkpoint_epoch=source_epoch,
    )

    variant_dir = Path(table_dir) / artifact_variant_id(variant)
    history_path = variant_dir / "metrics_history.csv"
    config_ok = variant_config_matches(
        variant_dir,
        variant,
        variant_args,
        mode="size_aware_detection_gnn_legacy_heads_rerank_prune_from_02",
    )
    if args.reuse_variant_outputs and history_path.exists() and config_ok and not args.force_predictions:
        print(f"[{variant}] reuse history: {history_path}", flush=True)
        return
    if history_path.exists() and not config_ok:
        print(f"[{variant}] stale rerank-prune cache ignored because code/config changed", flush=True)
        clear_stale_variant_outputs(variant_dir)
    elif history_path.exists() and not args.reuse_variant_outputs:
        print(f"[{variant}] fresh rerank-prune run clears previous generated summaries", flush=True)
        clear_stale_variant_outputs(variant_dir)

    source_prediction_path = existing_variant_dir(table_dir, GOIS_REIMPLEMENTATION_VARIANT) / "predictions" / "epoch_000.json"
    if not source_prediction_path.exists():
        raise FileNotFoundError(f"{variant} requires 01 candidate predictions: {source_prediction_path}")
    eval_image_ids = {record.image_id for record in eval_records}
    source_predictions = ab.filter_predictions_to_images(ab.read_prediction_file(source_prediction_path), eval_image_ids)
    eval_candidate_cache = predictions_to_string_image_cache(source_predictions)
    eval_samples = ab.make_size_aware_detection_samples(eval_records, {}, eval_candidate_cache, eval_gt, variant_args)

    model = ab.SizeAwareGraphGNN(
        ab.SIZE_AWARE_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        variant_args.gnn_hidden_dim,
        variant_args.gnn_layers,
        output_dim=3,
    ).to(device)
    model.size_aware_output_dim = 3
    ckpt_dir = source_dir / "checkpoints"
    if not load_single_stage_model_at_epoch(model, ckpt_dir, source_epoch, device):
        raise FileNotFoundError(f"{variant} could not load {source_variant} checkpoint epoch {source_epoch}: {ckpt_dir}")
    attach_size_aware_scores_for_eval(eval_samples, model, variant_args, device, assign_fn=assign_legacy3head_probs)
    predictions = run_legacy3head_rerank_prune_fixed_candidate_variant(eval_samples, variant_args)
    row = record_epoch(
        variant_dir,
        variant,
        source_epoch,
        None,
        None,
        predictions,
        args.eval_gt_path,
    )
    save_history(variant_dir, [row])
    write_variant_config(
        variant_dir,
        variant,
        variant_args,
        mode="size_aware_detection_gnn_legacy_heads_rerank_prune_from_02",
    )


def run_size_only_stage2_curve(
    table_dir,
    variant,
    train_samples,
    eval_samples,
    full_by_image,
    model_for_fine,
    args,
    device,
    hypergraph=False,
    prediction_mode="fusion",
):
    variant_dir = Path(table_dir) / artifact_variant_id(variant)
    history_path = variant_dir / "metrics_history.csv"
    ckpt_dir = variant_dir / "checkpoints"
    rows = []
    completed_epoch = 0
    base_mode = "stage2_size_only_hgnn" if hypergraph else "stage2_size_only_gnn"
    mode = f"{base_mode}_{prediction_mode}"
    config_ok = variant_config_matches(variant_dir, variant, args, mode=mode)
    may_reuse_variant_state = args.reuse_variant_outputs or args.resume_train
    if may_reuse_variant_state and history_path.exists() and config_ok and not args.force_train and not args.force_predictions:
        rows = read_rows(history_path)
        completed_epoch = completed_epoch_from_rows(rows)
        selection_path = variant_dir / "selection_evaluations.csv"
        if args.reuse_variant_outputs and completed_epoch >= args.epochs and (not args.eval_best_val_loss or selection_path.exists()):
            print(f"[{variant}] reuse history: {history_path}", flush=True)
            return
        if args.reuse_variant_outputs and completed_epoch < args.epochs and not args.resume_train:
            print(
                f"[{variant}] incomplete history found, but --resume_train is off. Restarting fresh.",
                flush=True,
            )
            clear_stale_variant_outputs(variant_dir)
            rows = []
            completed_epoch = 0
    elif history_path.exists() and not config_ok:
        print(f"[{variant}] stale size-only stage2 cache ignored because code/config changed", flush=True)
        clear_stale_variant_outputs(variant_dir)
    elif history_path.exists() and not may_reuse_variant_state:
        print(f"[{variant}] fresh size-only stage2 run clears previous generated summaries", flush=True)
        clear_stale_variant_outputs(variant_dir)
    write_variant_config(variant_dir, variant, args, mode=mode)

    source_model, source_variant, source_epoch = load_stage2_source_model(table_dir, args, device)
    attach_stage2_source_scores(train_samples, source_model, args, device, source_variant)
    attach_stage2_source_scores(eval_samples, source_model, args, device, source_variant)
    freeze_stage1_scores(train_samples)
    freeze_stage1_scores(eval_samples)
    with (variant_dir / "stage1_source.json").open("w") as f:
        json.dump(
            {
                "variant": source_variant,
                "epoch": source_epoch,
                "refinement": "size_only",
                "object_keep_score": "preserved_from_stage1",
            },
            f,
            indent=2,
        )

    hidden_dim = int(args.stage2_hidden_dim) if int(args.stage2_hidden_dim or 0) > 0 else int(args.gnn_hidden_dim)
    model = ab.SizeAwareGraphGNN(
        ab.STAGE2_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        hidden_dim,
        args.stage2_layers,
        output_dim=ab.STAGE2_SIZE_OUTPUT_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_global_stage2_size_loss_weight(train_samples)
    train_graph_pool = prepare_stage2_size_graph_pool(
        train_samples,
        args,
        "tensorize stage2 size train graph pool",
        hypergraph,
    )
    eval_graph_pool = prepare_stage2_size_graph_pool(
        eval_samples,
        args,
        "tensorize stage2 size val graph pool",
        hypergraph,
    )
    print_effective_gnn_schedule(train_graph_pool, args, pos_weight)
    if completed_epoch and args.resume_train and not args.force_train:
        loaded_epoch = load_single_stage_resume_state(model, optimizer, ckpt_dir, completed_epoch, device)
        if loaded_epoch < completed_epoch:
            rows = rows_through_epoch(rows, loaded_epoch)
            completed_epoch = loaded_epoch
            print(f"[{variant}] trimmed history to epoch {completed_epoch} for checkpoint match", flush=True)

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_loss = train_stage2_size_one_epoch(
            model,
            train_graph_pool,
            args,
            optimizer,
            device,
            epoch,
            pos_weight,
            hypergraph,
        )
        val_loss = stage2_size_loss_mean(model, eval_graph_pool, args, device, pos_weight, hypergraph)
        if should_full_eval_epoch(epoch, args):
            attach_stage2_size_scores_for_eval(eval_samples, model, args, device, hypergraph)
            if prediction_mode == "rerank_gois":
                predictions = run_legacy_gnn_rerank_gois_variant(eval_samples, args)
            else:
                predictions = ab.run_size_aware_gnn_large_preserve_variant(model_for_fine, eval_samples, full_by_image, args)
            row = record_epoch(variant_dir, variant, epoch, train_loss, val_loss, predictions, args.eval_gt_path)
        else:
            row = record_loss_only_epoch(variant, epoch, train_loss, val_loss)
        rows.append(row)
        save_history(variant_dir, rows)
        ab.ensure_dir(ckpt_dir)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"AP={float(row['AP']):.4f} " if has_full_eval(row) else f"[{variant}] epoch {epoch}/{args.epochs} AP=skip ",
            end="",
            flush=True,
        )
        print(f"loss={train_loss:.4f}/{val_loss:.4f}", flush=True)

    selection_items = selection_candidates(rows, args) if args.eval_best_val_loss else []
    if selection_items:
        for selected_epoch in sorted({epoch for _, epoch in selection_items}):
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is None or has_full_eval(selected_row):
                continue
            print(f"[{variant}] full-evaluating selected checkpoint at epoch {selected_epoch}", flush=True)
            if load_single_stage_model_at_epoch(model, ckpt_dir, selected_epoch, device):
                attach_stage2_size_scores_for_eval(eval_samples, model, args, device, hypergraph)
                if prediction_mode == "rerank_gois":
                    predictions = run_legacy_gnn_rerank_gois_variant(eval_samples, args)
                else:
                    predictions = ab.run_size_aware_gnn_large_preserve_variant(model_for_fine, eval_samples, full_by_image, args)
                evaluated = record_epoch(
                    variant_dir,
                    variant,
                    selected_epoch,
                    selected_row.get("train_loss"),
                    selected_row.get("val_loss"),
                    predictions,
                    args.eval_gt_path,
                )
                rows = replace_history_row(rows, evaluated)
                save_history(variant_dir, rows)
        selection_rows = []
        for label, selected_epoch in selection_items:
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is not None:
                selection_rows.append(dict(selected_row, selection=label))
        if selection_rows:
            write_rows(
                variant_dir / "selection_evaluations.csv",
                selection_rows,
                ["selection", "epoch", "variant", "num_predictions", "train_loss", "val_loss"] + METRIC_NAMES,
            )


def run_two_stage_curve(
    table_dir,
    variant,
    train_samples,
    eval_samples,
    full_by_image,
    model_for_fine,
    args,
    device,
    hypergraph=False,
    prediction_mode="fusion",
):
    variant_dir = Path(table_dir) / artifact_variant_id(variant)
    history_path = variant_dir / "metrics_history.csv"
    ckpt_dir = variant_dir / "checkpoints"
    rows = []
    completed_epoch = 0
    base_mode = "two_stage_hgnn" if hypergraph else "two_stage_gnn"
    mode = f"{base_mode}_{prediction_mode}"
    config_ok = variant_config_matches(variant_dir, variant, args, mode=mode)
    may_reuse_variant_state = args.reuse_variant_outputs or args.resume_train
    if may_reuse_variant_state and history_path.exists() and config_ok and not args.force_train and not args.force_predictions:
        rows = read_rows(history_path)
        completed_epoch = completed_epoch_from_rows(rows)
        selection_path = variant_dir / "selection_evaluations.csv"
        if args.reuse_variant_outputs and completed_epoch >= args.epochs and (not args.eval_best_val_loss or selection_path.exists()):
            print(f"[{variant}] reuse history: {history_path}", flush=True)
            return
        if args.reuse_variant_outputs and completed_epoch < args.epochs and not args.resume_train:
            print(
                f"[{variant}] incomplete history found, but --resume_train is off. Restarting fresh.",
                flush=True,
            )
            clear_stale_variant_outputs(variant_dir)
            rows = []
            completed_epoch = 0
    elif history_path.exists() and not config_ok:
        print(f"[{variant}] stale two-stage cache ignored because code/config changed", flush=True)
        clear_stale_variant_outputs(variant_dir)
    elif history_path.exists() and not may_reuse_variant_state:
        print(f"[{variant}] fresh two-stage run clears previous generated summaries", flush=True)
        clear_stale_variant_outputs(variant_dir)
    write_variant_config(variant_dir, variant, args, mode=mode)

    source_model, source_variant, source_epoch = load_stage2_source_model(table_dir, args, device)
    attach_stage2_source_scores(train_samples, source_model, args, device, source_variant)
    attach_stage2_source_scores(eval_samples, source_model, args, device, source_variant)
    freeze_stage1_scores(train_samples)
    freeze_stage1_scores(eval_samples)
    with (variant_dir / "stage1_source.json").open("w") as f:
        json.dump({"variant": source_variant, "epoch": source_epoch}, f, indent=2)

    hidden_dim = int(args.stage2_hidden_dim) if int(args.stage2_hidden_dim or 0) > 0 else int(args.gnn_hidden_dim)
    model = ab.SizeAwareGraphGNN(
        ab.STAGE2_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        hidden_dim,
        args.stage2_layers,
        output_dim=ab.STAGE2_ACTION_OUTPUT_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = compute_global_stage2_action_loss_weight(train_samples)
    train_graph_pool = prepare_stage2_graph_pool(
        train_samples,
        args,
        "tensorize stage2 train graph pool",
        hypergraph,
    )
    eval_graph_pool = prepare_stage2_graph_pool(
        eval_samples,
        args,
        "tensorize stage2 val graph pool",
        hypergraph,
    )
    print_effective_gnn_schedule(train_graph_pool, args, pos_weight)
    if completed_epoch and args.resume_train and not args.force_train:
        loaded_epoch = load_single_stage_resume_state(model, optimizer, ckpt_dir, completed_epoch, device)
        if loaded_epoch < completed_epoch:
            rows = rows_through_epoch(rows, loaded_epoch)
            completed_epoch = loaded_epoch
            print(f"[{variant}] trimmed history to epoch {completed_epoch} for checkpoint match", flush=True)

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_loss = train_stage2_one_epoch(
            model,
            train_graph_pool,
            args,
            optimizer,
            device,
            epoch,
            pos_weight,
            hypergraph,
        )
        val_loss = stage2_loss_mean(model, eval_graph_pool, args, device, pos_weight, hypergraph)
        if should_full_eval_epoch(epoch, args):
            attach_stage2_scores_for_eval(eval_samples, model, args, device, hypergraph)
            if prediction_mode == "rerank_gois":
                predictions = run_legacy_gnn_rerank_gois_variant(eval_samples, args)
            else:
                predictions = ab.run_size_aware_gnn_large_preserve_variant(model_for_fine, eval_samples, full_by_image, args)
            row = record_epoch(variant_dir, variant, epoch, train_loss, val_loss, predictions, args.eval_gt_path)
        else:
            row = record_loss_only_epoch(variant, epoch, train_loss, val_loss)
        rows.append(row)
        save_history(variant_dir, rows)
        ab.ensure_dir(ckpt_dir)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"AP={float(row['AP']):.4f} " if has_full_eval(row) else f"[{variant}] epoch {epoch}/{args.epochs} AP=skip ",
            end="",
            flush=True,
        )
        print(f"loss={train_loss:.4f}/{val_loss:.4f}", flush=True)

    selection_items = selection_candidates(rows, args) if args.eval_best_val_loss else []
    if selection_items:
        for selected_epoch in sorted({epoch for _, epoch in selection_items}):
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is None or has_full_eval(selected_row):
                continue
            print(f"[{variant}] full-evaluating selected checkpoint at epoch {selected_epoch}", flush=True)
            if load_single_stage_model_at_epoch(model, ckpt_dir, selected_epoch, device):
                attach_stage2_scores_for_eval(eval_samples, model, args, device, hypergraph)
                if prediction_mode == "rerank_gois":
                    predictions = run_legacy_gnn_rerank_gois_variant(eval_samples, args)
                else:
                    predictions = ab.run_size_aware_gnn_large_preserve_variant(model_for_fine, eval_samples, full_by_image, args)
                evaluated = record_epoch(
                    variant_dir,
                    variant,
                    selected_epoch,
                    selected_row.get("train_loss"),
                    selected_row.get("val_loss"),
                    predictions,
                    args.eval_gt_path,
                )
                rows = replace_history_row(rows, evaluated)
                save_history(variant_dir, rows)
        selection_rows = []
        for label, selected_epoch in selection_items:
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is not None:
                selection_rows.append(dict(selected_row, selection=label))
        if selection_rows:
            write_rows(
                variant_dir / "selection_evaluations.csv",
                selection_rows,
                ["selection", "epoch", "variant", "num_predictions", "train_loss", "val_loss"] + METRIC_NAMES,
            )


def run_legacy_repro_curve(
    table_dir,
    variant,
    train_samples,
    eval_samples,
    full_by_image,
    model_for_fine,
    args,
    device,
    hypergraph=False,
    prediction_mode="fusion",
):
    variant_dir = Path(table_dir) / artifact_variant_id(variant)
    history_path = variant_dir / "metrics_history.csv"
    ckpt_dir = variant_dir / "checkpoints"
    rows = []
    completed_epoch = 0
    base_mode = "hypergnn_no_large_preserve" if hypergraph else "size_aware_gnn_no_large_preserve"
    mode = f"{base_mode}_{prediction_mode}"
    config_ok = variant_config_matches(variant_dir, variant, args, mode=mode)
    may_reuse_variant_state = args.reuse_variant_outputs or args.resume_train
    if may_reuse_variant_state and history_path.exists() and config_ok and not args.force_train and not args.force_predictions:
        rows = read_rows(history_path)
        completed_epoch = completed_epoch_from_rows(rows)
        selection_path = variant_dir / "selection_evaluations.csv"
        if args.reuse_variant_outputs and completed_epoch >= args.epochs and (not args.eval_best_val_loss or selection_path.exists()):
            print(f"[{variant}] reuse history: {history_path}", flush=True)
            return
        if args.reuse_variant_outputs and completed_epoch < args.epochs and not args.resume_train:
            print(f"[{variant}] incomplete history found, but --resume_train is off. Restarting fresh.", flush=True)
            clear_stale_variant_outputs(variant_dir)
            rows = []
            completed_epoch = 0
    elif history_path.exists() and not config_ok:
        print(f"[{variant}] stale legacy-repro cache ignored because code/config changed", flush=True)
        clear_stale_variant_outputs(variant_dir)
    elif history_path.exists() and not may_reuse_variant_state:
        print(f"[{variant}] fresh legacy-repro run clears previous generated summaries", flush=True)
        clear_stale_variant_outputs(variant_dir)
    write_variant_config(variant_dir, variant, args, mode=mode)

    input_dim = legacy_hyper_node_dim() if hypergraph else ab.SIZE_AWARE_NODE_DIM
    edge_dim = legacy_hyper_edge_dim() if hypergraph else ab.SIZE_AWARE_EDGE_DIM
    model = ab.SizeAwareGraphGNN(
        input_dim,
        edge_dim,
        args.gnn_hidden_dim,
        args.gnn_layers,
        output_dim=3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_graph_pool = prepare_legacy_graph_pool(
        train_samples,
        args,
        "tensorize HyperGNN train hypergraphs" if hypergraph else "tensorize GNN train graphs",
        hypergraph=hypergraph,
    )
    eval_graph_pool = prepare_legacy_graph_pool(
        eval_samples,
        args,
        "tensorize HyperGNN val hypergraphs" if hypergraph else "tensorize GNN val graphs",
        hypergraph=hypergraph,
    )
    print(
        f"[{variant} schedule] pool={len(train_graph_pool)} "
        f"graphs/epoch={len(select_limited_items(train_graph_pool, args.gnn_train_steps_per_epoch, args.seed + 1009))} "
        f"batch_size={max(1, int(args.gnn_batch_size)) if use_pyg_batching(args) else 1}",
        flush=True,
    )
    if completed_epoch and args.resume_train and not args.force_train:
        loaded_epoch = load_single_stage_resume_state(model, optimizer, ckpt_dir, completed_epoch, device)
        if loaded_epoch < completed_epoch:
            rows = rows_through_epoch(rows, loaded_epoch)
            completed_epoch = loaded_epoch
            print(f"[{variant}] trimmed history to epoch {completed_epoch} for checkpoint match", flush=True)

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_loss = train_legacy_one_epoch(model, train_graph_pool, args, optimizer, device, epoch, hypergraph=hypergraph)
        val_loss = legacy_loss_mean(model, eval_graph_pool, args, device, hypergraph=hypergraph)
        if should_full_eval_epoch(epoch, args):
            attach_legacy_scores_for_eval(eval_samples, model, args, device, hypergraph=hypergraph)
            predictions = run_legacy_eval_predictions(
                model_for_fine,
                eval_samples,
                full_by_image,
                args,
                hypergraph=hypergraph,
                prediction_mode=prediction_mode,
            )
            row = record_epoch(variant_dir, variant, epoch, train_loss, val_loss, predictions, args.eval_gt_path)
        else:
            row = record_loss_only_epoch(variant, epoch, train_loss, val_loss)
        rows.append(row)
        save_history(variant_dir, rows)
        ab.ensure_dir(ckpt_dir)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"AP={float(row['AP']):.4f} " if has_full_eval(row) else f"[{variant}] epoch {epoch}/{args.epochs} AP=skip ",
            end="",
            flush=True,
        )
        print(f"loss={train_loss:.4f}/{val_loss:.4f}", flush=True)

    selection_items = selection_candidates(rows, args) if args.eval_best_val_loss else []
    if selection_items:
        for selected_epoch in sorted({epoch for _, epoch in selection_items}):
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is None or has_full_eval(selected_row):
                continue
            print(f"[{variant}] full-evaluating selected checkpoint at epoch {selected_epoch}", flush=True)
            if load_single_stage_model_at_epoch(model, ckpt_dir, selected_epoch, device):
                attach_legacy_scores_for_eval(eval_samples, model, args, device, hypergraph=hypergraph)
                predictions = run_legacy_eval_predictions(
                    model_for_fine,
                    eval_samples,
                    full_by_image,
                    args,
                    hypergraph=hypergraph,
                    prediction_mode=prediction_mode,
                )
                evaluated = record_epoch(
                    variant_dir,
                    variant,
                    selected_epoch,
                    selected_row.get("train_loss"),
                    selected_row.get("val_loss"),
                    predictions,
                    args.eval_gt_path,
                )
                rows = replace_history_row(rows, evaluated)
                save_history(variant_dir, rows)
        selection_rows = []
        for label, selected_epoch in selection_items:
            selected_row = next((row for row in rows if epoch_value(row) == selected_epoch), None)
            if selected_row is not None:
                selection_rows.append(dict(selected_row, selection=label))
        if selection_rows:
            write_rows(
                variant_dir / "selection_evaluations.csv",
                selection_rows,
                ["selection", "epoch", "variant", "num_predictions", "train_loss", "val_loss"] + METRIC_NAMES,
            )


def run_pipeline_ablation(args, device):
    selected = select_variants(args.pipeline_variants, PIPELINE_VARIANTS)
    assert_training_authorized(selected, args.allow_training)
    table_dir = Path(args.output_root) / "pipeline_ablation"
    source_pred_dir = Path(args.source_prediction_dir) if args.source_prediction_dir else None
    print(f"\n[pipeline_ablation] variants: {', '.join(selected)}", flush=True)
    detection_gnn_variants = {
        GNN_NO_CLUSTER_VARIANT,
        GNN_CLUSTER_VARIANT,
        "04_gnn_conf_rescue",
        SAME_POOL_SIZE_REFINEMENT_VARIANT,
        LOW_CONF_SIZE_REFINEMENT_VARIANT,
        LOW_CONF_CLUSTER_HGNN_VARIANT,
    }
    needs_detection_gnn = any(variant in selected for variant in detection_gnn_variants)
    needs_same_condition_gois = any(
        variant in selected
        for variant in {
            GOIS_REIMPLEMENTATION_VARIANT,
            GNN_NO_CLUSTER_VARIANT,
            GNN_CLUSTER_VARIANT,
            SAME_POOL_SIZE_REFINEMENT_VARIANT,
        }
    )
    needs_gois_candidates = needs_same_condition_gois
    ctx = load_common_context(
        args,
        table_dir,
        need_train=needs_detection_gnn,
        need_gois_candidates=needs_gois_candidates,
        need_base_cache=needs_same_condition_gois,
        need_full_predictions=("00_full_inference" in selected),
    )
    args.eval_gt_path = ctx["eval_gt_path"]
    write_manifest(table_dir, "pipeline_ablation", args, selected)

    train_candidate_cache = ctx["train_gois_cache"] if needs_gois_candidates else ctx["train_cache"]
    eval_candidate_cache = ctx["eval_gois_cache"] if needs_gois_candidates else ctx["eval_cache"]
    eval_gois_samples = make_cached_prediction_samples(ctx["eval_records"], eval_candidate_cache)
    train_full_cache = {}
    eval_full_cache = {}
    if needs_detection_gnn and not args.disable_large_preserve:
        train_full_cache = ab.generate_or_load_full_cache(ctx["model"], ctx["train_records"], args, "train", Path(args.output_root) / "common_cache")
        eval_full_cache = {str(record.image_id): ctx["full_by_image"].get(record.image_id, []) for record in ctx["eval_records"]}
    active_full_by_image = {} if args.disable_large_preserve else ctx["full_by_image"]
    eval_image_ids = {record.image_id for record in ctx["eval_records"]}

    if "00_full_inference" in selected:
        run_static_variant(
            table_dir,
            "00_full_inference",
            0,
            lambda: run_full_reference(ctx["eval_records"], ctx["full_by_image"]),
            ctx["eval_gt_path"],
            args,
            (source_pred_dir / "00_full_inference.json") if source_pred_dir else None,
            eval_image_ids=eval_image_ids,
        )
    if GOIS_REIMPLEMENTATION_VARIANT in selected:
        run_static_variant(
            table_dir,
            GOIS_REIMPLEMENTATION_VARIANT,
            0,
            lambda: run_gois_reimplementation(eval_gois_samples, args),
            ctx["eval_gt_path"],
            args,
            source_prediction_file(source_pred_dir, GOIS_REIMPLEMENTATION_VARIANT),
            eval_image_ids=eval_image_ids,
        )

    if GNN_NO_CLUSTER_VARIANT in selected:
        variant_args = clone_args_with(args, size_graph_cluster_mode="none")
        train_samples = ab.make_size_aware_detection_samples(
            ctx["train_records"],
            train_full_cache,
            train_candidate_cache,
            ctx["train_gt"],
            variant_args,
        )
        eval_samples = ab.make_size_aware_detection_samples(
            ctx["eval_records"],
            eval_full_cache,
            eval_candidate_cache,
            ctx["eval_gt"],
            variant_args,
        )
        run_size_aware_curve(
            table_dir,
            GNN_NO_CLUSTER_VARIANT,
            train_samples,
            eval_samples,
            active_full_by_image,
            ctx["model"],
            variant_args,
            device,
            output_dim=3,
            score_assign_fn=assign_legacy3head_probs,
            prediction_fn=run_legacy3head_fixed_candidate_variant,
            mode="size_aware_detection_gnn_legacy_heads_fixed_pool",
        )

    if SAME_POOL_RERANK_PRUNE_VARIANT in selected:
        run_same_pool_rerank_prune_curve(table_dir, ctx["eval_records"], ctx["eval_gt"], args, device)

    if GNN_CLUSTER_VARIANT in selected:
        variant_args = clone_args_with(args, size_graph_cluster_mode="dbscan")
        train_samples = ab.make_size_aware_detection_samples(
            ctx["train_records"],
            train_full_cache,
            train_candidate_cache,
            ctx["train_gt"],
            variant_args,
        )
        eval_samples = ab.make_size_aware_detection_samples(
            ctx["eval_records"],
            eval_full_cache,
            eval_candidate_cache,
            ctx["eval_gt"],
            variant_args,
        )
        run_size_aware_curve(
            table_dir,
            GNN_CLUSTER_VARIANT,
            train_samples,
            eval_samples,
            active_full_by_image,
            ctx["model"],
            variant_args,
            device,
            output_dim=3,
            score_assign_fn=assign_legacy3head_probs,
            prediction_fn=run_legacy3head_fixed_candidate_variant,
            mode="size_aware_detection_gnn_legacy_heads_cluster_fixed_pool",
        )

    if "04_gnn_conf_rescue" in selected:
        variant_args = clone_args_with(
            args,
            coarse_conf=float(args.rescue_coarse_conf),
            fine_conf=float(args.rescue_fine_conf),
            size_graph_cluster_mode="dbscan",
            disable_large_preserve=True,
            gnn_score_alpha=1.0,
        )
        cache_dir = Path(args.output_root) / "common_cache"
        train_rescue_cache = build_pre_gois_candidate_cache_for_args(
            ctx["model"],
            ctx["train_records"],
            variant_args,
            "train",
            cache_dir,
        )
        eval_rescue_cache = build_pre_gois_candidate_cache_for_args(
            ctx["model"],
            ctx["eval_records"],
            variant_args,
            "eval",
            cache_dir,
        )
        train_samples = ab.make_size_aware_detection_samples(
            ctx["train_records"],
            {},
            train_rescue_cache,
            ctx["train_gt"],
            variant_args,
        )
        eval_samples = ab.make_size_aware_detection_samples(
            ctx["eval_records"],
            {},
            eval_rescue_cache,
            ctx["eval_gt"],
            variant_args,
        )
        run_legacy_repro_curve(
            table_dir,
            "04_gnn_conf_rescue",
            train_samples,
            eval_samples,
            {},
            ctx["model"],
            variant_args,
            device,
            hypergraph=False,
            prediction_mode="rerank_gois",
        )

    if SAME_POOL_SIZE_REFINEMENT_VARIANT in selected:
        variant_args = clone_args_with(
            args,
            size_graph_cluster_mode="none",
            stage2_source_variant=GNN_NO_CLUSTER_VARIANT,
        )
        train_samples = ab.make_size_aware_detection_samples(
            ctx["train_records"],
            train_full_cache,
            train_candidate_cache,
            ctx["train_gt"],
            variant_args,
        )
        eval_samples = ab.make_size_aware_detection_samples(
            ctx["eval_records"],
            eval_full_cache,
            eval_candidate_cache,
            ctx["eval_gt"],
            variant_args,
        )
        run_size_only_stage2_curve(
            table_dir,
            SAME_POOL_SIZE_REFINEMENT_VARIANT,
            train_samples,
            eval_samples,
            active_full_by_image,
            ctx["model"],
            variant_args,
            device,
        )

    if LOW_CONF_SIZE_REFINEMENT_VARIANT in selected:
        variant_args = clone_args_with(
            args,
            coarse_conf=float(args.rescue_coarse_conf),
            fine_conf=float(args.rescue_fine_conf),
            size_graph_cluster_mode="dbscan",
            disable_large_preserve=True,
            gnn_score_alpha=1.0,
            stage2_source_variant="04_gnn_conf_rescue",
        )
        cache_dir = Path(args.output_root) / "common_cache"
        train_rescue_cache = build_pre_gois_candidate_cache_for_args(
            ctx["model"],
            ctx["train_records"],
            variant_args,
            "train",
            cache_dir,
        )
        eval_rescue_cache = build_pre_gois_candidate_cache_for_args(
            ctx["model"],
            ctx["eval_records"],
            variant_args,
            "eval",
            cache_dir,
        )
        train_samples = ab.make_size_aware_detection_samples(
            ctx["train_records"],
            {},
            train_rescue_cache,
            ctx["train_gt"],
            variant_args,
        )
        eval_samples = ab.make_size_aware_detection_samples(
            ctx["eval_records"],
            {},
            eval_rescue_cache,
            ctx["eval_gt"],
            variant_args,
        )
        run_size_only_stage2_curve(
            table_dir,
            LOW_CONF_SIZE_REFINEMENT_VARIANT,
            train_samples,
            eval_samples,
            {},
            ctx["model"],
            variant_args,
            device,
            hypergraph=False,
            prediction_mode="rerank_gois",
        )

    if LOW_CONF_CLUSTER_HGNN_VARIANT in selected:
        variant_args = clone_args_with(
            args,
            coarse_conf=float(args.rescue_coarse_conf),
            fine_conf=float(args.rescue_fine_conf),
            size_graph_cluster_mode="dbscan",
            disable_large_preserve=True,
            gnn_score_alpha=1.0,
        )
        cache_dir = Path(args.output_root) / "common_cache"
        train_rescue_cache = build_pre_gois_candidate_cache_for_args(
            ctx["model"],
            ctx["train_records"],
            variant_args,
            "train",
            cache_dir,
        )
        eval_rescue_cache = build_pre_gois_candidate_cache_for_args(
            ctx["model"],
            ctx["eval_records"],
            variant_args,
            "eval",
            cache_dir,
        )
        train_samples = ab.make_size_aware_detection_samples(
            ctx["train_records"],
            {},
            train_rescue_cache,
            ctx["train_gt"],
            variant_args,
        )
        eval_samples = ab.make_size_aware_detection_samples(
            ctx["eval_records"],
            {},
            eval_rescue_cache,
            ctx["eval_gt"],
            variant_args,
        )
        run_legacy_repro_curve(
            table_dir,
            LOW_CONF_CLUSTER_HGNN_VARIANT,
            train_samples,
            eval_samples,
            {},
            ctx["model"],
            variant_args,
            device,
            hypergraph=True,
            prediction_mode="rerank_gois",
        )

    summary_selected = [
        variant
        for variant in PIPELINE_VARIANTS
        if variant in selected or (existing_variant_dir(table_dir, variant) / "metrics_history.csv").exists()
    ]
    summarize_table(table_dir, summary_selected)
    print(f"[pipeline_ablation] saved: {table_dir}", flush=True)


def main():
    args = clone_args_for_ab(parse_args())
    selected = select_variants(args.pipeline_variants, PIPELINE_VARIANTS)
    assert_training_authorized(selected, args.allow_training)
    ab.configure_class_space(args.class_space)
    if not args.run_pipeline:
        args.run_pipeline = True
    if args.eval_every <= 0 and not args.eval_best_val_loss:
        raise ValueError("No full evaluation will be run: set --eval_every > 0 or keep --eval_best_val_loss enabled.")
    ab.set_seed(args.seed)
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = ab.resolve_torch_device(args.device)
    if args.require_pyg and (PyGData is None or PyGDataLoader is None):
        raise RuntimeError("torch_geometric is required by --require_pyg but is unavailable.")
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Device: infer={args.device}, train={device}", flush=True)
    print(f"Full data: train_limit={args.max_train_images or 'all'}, eval_limit={args.max_eval_images or 'all'}", flush=True)
    if use_pyg_batching(args):
        print(
            f"[PyG] enabled: batch_size={args.gnn_batch_size} "
            f"workers={args.gnn_num_workers} pin_memory={args.gnn_pin_memory}",
            flush=True,
        )
    else:
        print("[PyG] disabled or unavailable; using per-graph fallback", flush=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    if args.run_pipeline:
        run_pipeline_ablation(args, device)


if __name__ == "__main__":
    main()
