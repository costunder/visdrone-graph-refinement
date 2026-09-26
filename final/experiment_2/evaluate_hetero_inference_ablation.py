#!/usr/bin/env python3
"""Evaluate hetero checkpoint scoring policies without retraining or detector inference."""

import argparse
import csv
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
FINAL_DIR = HERE.parent
SCRIPT_DIR = FINAL_DIR / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(HERE))

import hetero_detection_class_view as hetero
import hetero_training
import run_gois_paper_ablation_curves as base
import run_gois_two_stage_gnn_ablation as ab
import run_sparse_ppr_ablation as runner


DEFAULT_VARIANT_DIR = (
    HERE
    / "runs/table6_yolo11_10class_extra_ablation/pipeline_ablation"
    / "07_hetero_detection_class_view_gatv2"
)
POLICIES = (
    ("gate1_correction", 1.0, False),
    ("no_gate_no_correction", 0.0, True),
    ("correction_only", 0.0, False),
    ("gate025_correction", 0.25, False),
    ("gate050_correction", 0.50, False),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--checkpoint-epoch", type=int, default=40)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_reference_ap(table_dir, variant):
    path = Path(table_dir) / "evaluation_results_best_ap.csv"
    with path.open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["variant"] == variant)
    return float(row["AP"])


def write_csv(path, rows):
    fields = [
        "policy",
        "foreground_gate_alpha",
        "class_correction_enabled",
        "num_predictions",
        "applied_class_corrections",
        "correction_rate",
        *ab.metric_names(),
        "delta_ap_vs_gate1_correction",
        "delta_ap_vs_06",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli = parse_args()
    variant_dir = cli.variant_dir.resolve()
    config = json.loads((variant_dir / "run_config.json").read_text())
    if config.get("status") != "completed":
        raise RuntimeError(f"Variant is not completed: {variant_dir}")
    args = Namespace(**config["args"])
    ab.configure_class_space(args.class_space)
    ab.set_seed(args.seed)
    device = ab.resolve_torch_device(args.device)

    eval_records = ab.build_image_records(
        args.eval_images,
        args.ground_truth_path,
        max_images=args.max_eval_images,
    )
    eval_gt = ab.load_coco_gt_by_image(eval_records, args.eval_gt_path)
    candidate_cache = runner.tagged_low_conf_cache(
        None,
        eval_records,
        args,
        "eval",
    )
    eval_samples = runner.build_samples(
        eval_records,
        eval_gt,
        candidate_cache,
        args,
        "low_conf_multiscale",
    )
    hetero_training.prepare_pool(
        eval_samples,
        args,
        include_ppr=True,
        desc="tensorize hetero inference-ablation graphs",
    )

    checkpoint_path = variant_dir / "checkpoints" / f"epoch_{cli.checkpoint_epoch:03d}.pt"
    model = hetero.build_model(ab, args).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    hetero_training.attach_scores(eval_samples, model, args, device, include_ppr=True)

    output_dir = variant_dir / "inference_policy_ablation" / f"epoch_{cli.checkpoint_epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_ap = read_reference_ap(variant_dir.parent, "06_class_relation_ppr_gatv2_sage")
    rows = []
    for policy, gate_alpha, disable_correction in POLICIES:
        policy_args = base.clone_args_with(
            args,
            exp2_hetero_foreground_gate_alpha=gate_alpha,
            disable_exp2_hetero_class_correction=disable_correction,
        )
        predictions = hetero_training.refined_predictions(eval_samples, policy_args)
        prediction_path = output_dir / f"{policy}.json"
        ab.write_predictions(prediction_path, predictions)
        metrics = ab.evaluate_predictions(args.eval_gt_path, str(prediction_path))
        correction = hetero_training.correction_summary(eval_samples, policy_args)
        row = {
            "policy": policy,
            "foreground_gate_alpha": gate_alpha,
            "class_correction_enabled": not disable_correction,
            "num_predictions": len(predictions),
            "applied_class_corrections": correction["applied_class_corrections"],
            "correction_rate": correction["correction_rate"],
            **{name: round(float(metrics[name]), 6) for name in ab.metric_names()},
            "delta_ap_vs_gate1_correction": 0.0,
            "delta_ap_vs_06": round(float(metrics["AP"]) - reference_ap, 6),
        }
        rows.append(row)

    current_ap = float(rows[0]["AP"])
    for row in rows:
        row["delta_ap_vs_gate1_correction"] = round(float(row["AP"]) - current_ap, 6)
    write_csv(output_dir / "metrics.csv", rows)
    metadata = {
        "variant": config["variant"],
        "checkpoint_epoch": cli.checkpoint_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "detector_inference_performed": False,
        "model_training_performed": False,
        "shared_detector_cache": str(args.common_cache_dir),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
