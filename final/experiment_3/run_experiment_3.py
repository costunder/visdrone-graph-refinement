#!/usr/bin/env python3
"""Train the controlled sequential experiment-1 -> experiment-2 pipeline.

This runner never performs detector inference.  It requires existing raw
coarse/fine caches and refuses to start model training without --allow_training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

import sequential_exp1_exp2 as sequential


base = sequential.base
ab = sequential.ab
exp2 = sequential.exp2

VARIANTS = {
    "00_zero_prior_control": {
        "prior_mode": sequential.PRIOR_ZERO,
        "role": "capacity-matched experiment-2 control with five zero prior channels",
    },
    "01_exp1_graph_prior_exp2_ppr": {
        "prior_mode": sequential.PRIOR_EXP1,
        "role": "frozen experiment-1 graph probabilities passed into experiment-2 PPR refinement",
    },
}
CODE_VERSION = 1
METRIC_NAMES = base.METRIC_NAMES


def require_confirmed_stage_design():
    """No flags can authorize the user-rejected exp1-prior -> exp2 design."""
    raise RuntimeError(
        "Experiment 3 execution blocked: stage order and handoff semantics are unconfirmed. "
        "See final/COMPARISON_REPAIR_SPEC.md. New design approval and conformance required."
    )


def parse_args():
    experiment_dir = Path(__file__).resolve().parent
    final_dir = experiment_dir.parent
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--exp3_variants", default="00,01")
    extra.add_argument(
        "--common_cache_dir",
        type=Path,
        default=final_dir / "runs/table6_yolo11_10class_extra_ablation/common_cache",
    )
    extra.add_argument("--exp3_core_conf", type=float, default=0.25)
    extra.add_argument(
        "--exp3_exp1_checkpoint",
        type=Path,
        default=(
            final_dir
            / "experiment_1/runs/table6_yolo11_10class_extra_ablation"
            / "pipeline_ablation/02_gnn_no_cluster/checkpoints/epoch_116.pt"
        ),
    )
    extra.add_argument("--exp3_exp1_hidden_dim", type=int, default=96)
    extra.add_argument("--exp3_exp1_layers", type=int, default=3)
    extra.add_argument("--exp2_local_knn", type=int, default=12)
    extra.add_argument("--exp2_cross_class_knn", type=int, default=4)
    extra.add_argument("--exp2_spatial_cell_size", type=float, default=0.0)
    extra.add_argument("--exp2_ppr_knn", type=int, default=8)
    extra.add_argument("--exp2_ppr_alpha", type=float, default=0.15)
    extra.add_argument("--exp2_ppr_steps", type=int, default=8)
    extra.add_argument("--exp2_ppr_frontier", type=int, default=64)
    extra.add_argument("--exp2_attention_heads", type=int, default=4)
    extra.add_argument("--exp2_dropout", type=float, default=0.10)
    exp3_args, remaining = extra.parse_known_args()

    original = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = base.clone_args_for_ab(base.parse_args())
    finally:
        sys.argv = original
    for key, value in vars(exp3_args).items():
        setattr(args, key, value)
    return args


def select_variants(text):
    selected = []
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        match = next(
            (name for name in VARIANTS if token in {name, name[:2]}),
            None,
        )
        if match is None:
            raise ValueError(
                f"Unknown experiment-3 variant {token}. Known: {', '.join(VARIANTS)}"
            )
        if match not in selected:
            selected.append(match)
    if not selected:
        raise ValueError("No experiment-3 variants selected")
    return selected


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_json_cache(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 3 never runs detector inference; required {label} cache is missing: {path}"
        )
    with path.open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} cache must contain an image-id mapping: {path}")
    print(f"[cache] using required {label}: {path}", flush=True)
    return payload, path


def load_required_candidate_cache(records, args, split_name):
    cache_args = base.clone_args_with(
        args,
        coarse_conf=float(args.rescue_coarse_conf),
        fine_conf=float(args.rescue_fine_conf),
    )
    cache_root = Path(args.common_cache_dir).resolve()
    coarse_path = ab.prediction_cache_path(
        cache_root,
        split_name,
        records,
        cache_args,
    )
    fine_path = base.fine_prediction_cache_path(
        cache_root,
        split_name,
        records,
        cache_args,
    )
    coarse, coarse_path = read_required_json_cache(
        coarse_path,
        f"{split_name} coarse low-confidence",
    )
    fine, fine_path = read_required_json_cache(
        fine_path,
        f"{split_name} fine low-confidence",
    )
    return (
        sequential.tag_shared_candidate_cache(records, coarse, fine),
        {"coarse": str(coarse_path), "fine": str(fine_path)},
    )


def load_records_and_targets(args, table_dir):
    ground_truth = Path(args.ground_truth_path)
    if not ground_truth.exists():
        raise FileNotFoundError(
            f"Experiment 3 requires an existing COCO ground truth file: {ground_truth}"
        )
    train_records = ab.build_image_records(
        args.train_images,
        max_images=args.max_train_images,
    )
    eval_records = ab.build_image_records(
        args.eval_images,
        args.ground_truth_path,
        max_images=args.max_eval_images,
    )
    if args.max_eval_images:
        eval_gt_path = Path(table_dir) / "ground_truth_eval_subset.json"
        ab.filter_coco_gt(args.ground_truth_path, eval_records, eval_gt_path)
    else:
        eval_gt_path = ground_truth
    return {
        "train_records": train_records,
        "eval_records": eval_records,
        "train_gt": ab.load_gt_by_image(train_records, args.train_labels),
        "eval_gt": ab.load_coco_gt_by_image(eval_records, eval_gt_path),
        "eval_gt_path": str(eval_gt_path.resolve()),
    }


def load_frozen_exp1_model(args, device):
    checkpoint = Path(args.exp3_exp1_checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_dir = checkpoint.parent.parent
    base.require_current_checkpoint_schema(source_dir, "02_gnn_no_cluster")
    model = ab.SizeAwareGraphGNN(
        ab.SIZE_AWARE_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        int(args.exp3_exp1_hidden_dim),
        int(args.exp3_exp1_layers),
        output_dim=3,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.size_aware_output_dim = 3
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"[experiment-1] frozen checkpoint: {checkpoint}", flush=True)
    return model, checkpoint


def sample_to_pyg(sample, args, prior_mode):
    key = f"_exp3_pyg_v{sequential.GRAPH_SCHEMA_VERSION}_{prior_mode}"
    if bool(args.cache_pyg_graphs) and sample.get(key) is not None:
        return sample[key]
    x, edge_index, edge_attr, targets, weights = sequential.build_sequential_tensors(
        sample,
        args,
        prior_mode,
    )
    data = base.PyGData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=targets,
        weights=weights,
    )
    if bool(args.cache_pyg_graphs):
        sample[key] = data
    return data


def prepare_graph_pool(samples, args, prior_mode, desc):
    valid = base.nonempty_graph_samples(samples)
    if not base.use_pyg_batching(args) or not bool(args.gnn_precompute_graph_pool):
        return valid
    return [
        sample_to_pyg(sample, args, prior_mode)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def selected_graphs(pool, args, limit, seed, prior_mode, desc):
    selected = base.select_limited_items(pool, limit, seed)
    if base.is_pyg_graph_pool(selected) or not base.use_pyg_batching(args):
        return selected
    return [
        sample_to_pyg(sample, args, prior_mode)
        for sample in tqdm(selected, desc=desc, leave=False)
    ]


def graph_loss(logits, targets, weights):
    return ab.weighted_size_aware_bce_loss(
        logits,
        targets,
        weights,
        pos_weight=None,
    )


def train_epoch(model, pool, args, optimizer, device, epoch, prior_mode):
    require_confirmed_stage_design()
    model.train()
    order = selected_graphs(
        pool,
        args,
        args.gnn_train_steps_per_epoch,
        args.seed + epoch * 1009,
        prior_mode,
        f"tensorize experiment-3 {prior_mode} train graphs",
    )
    losses = []
    accumulation = max(1, int(args.gnn_grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    if base.is_pyg_graph_pool(order):
        batches = base.make_pyg_loader(order, args, shuffle=True)
        for batch_index, batch in enumerate(
            tqdm(batches, desc=f"experiment-3 {prior_mode} batches", leave=False),
            start=1,
        ):
            batch = batch.to(device)
            loss = graph_loss(
                model(batch.x, batch.edge_index, batch.edge_attr),
                batch.y,
                batch.weights,
            )
            (loss / accumulation).backward()
            if batch_index % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if losses and len(losses) % accumulation:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return sum(losses) / len(losses) if losses else 0.0

    for sample_index, sample in enumerate(
        tqdm(order, desc=f"experiment-3 {prior_mode} epoch", leave=False),
        start=1,
    ):
        x, edge_index, edge_attr, targets, weights = sequential.build_sequential_tensors(
            sample,
            args,
            prior_mode,
        )
        loss = graph_loss(
            model(x.to(device), edge_index.to(device), edge_attr.to(device)),
            targets.to(device),
            weights.to(device),
        )
        (loss / accumulation).backward()
        if sample_index % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses) if losses else 0.0


def validation_loss(model, pool, args, device, prior_mode):
    selected = selected_graphs(
        pool,
        args,
        args.gnn_val_loss_limit,
        args.seed + 9301,
        prior_mode,
        f"tensorize experiment-3 {prior_mode} validation graphs",
    )
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        if base.is_pyg_graph_pool(selected):
            for batch in base.make_pyg_loader(selected, args, shuffle=False):
                batch = batch.to(device)
                loss = graph_loss(
                    model(batch.x, batch.edge_index, batch.edge_attr),
                    batch.y,
                    batch.weights,
                )
                total += float(loss.detach().cpu()) * int(batch.num_graphs)
                count += int(batch.num_graphs)
        else:
            for sample in selected:
                x, edge_index, edge_attr, targets, weights = sequential.build_sequential_tensors(
                    sample,
                    args,
                    prior_mode,
                )
                loss = graph_loss(
                    model(x.to(device), edge_index.to(device), edge_attr.to(device)),
                    targets.to(device),
                    weights.to(device),
                )
                total += float(loss.detach().cpu())
                count += 1
    return total / count if count else 0.0


def attach_scores(samples, model, args, device, prior_mode):
    valid = base.nonempty_graph_samples(samples)
    model.eval()
    with torch.no_grad():
        if base.use_pyg_batching(args):
            data_list = [sample_to_pyg(sample, args, prior_mode) for sample in valid]
            offset = 0
            for batch in base.make_pyg_loader(data_list, args, shuffle=False):
                batch = batch.to(device)
                probabilities = torch.sigmoid(
                    model(batch.x, batch.edge_index, batch.edge_attr)
                ).detach().cpu()
                counts = torch.bincount(
                    batch.batch.cpu(), minlength=batch.num_graphs
                ).tolist()
                cursor = 0
                for local_index, node_count in enumerate(counts):
                    sample = valid[offset + local_index]
                    graph_probabilities = probabilities[
                        cursor : cursor + int(node_count)
                    ].tolist()
                    for node, probability in zip(
                        sample["nodes"], graph_probabilities
                    ):
                        sequential.assign_stage2_probabilities(node, probability)
                    cursor += int(node_count)
                offset += int(batch.num_graphs)
            return

        for sample in valid:
            x, edge_index, edge_attr, _, _ = sequential.build_sequential_tensors(
                sample,
                args,
                prior_mode,
            )
            probabilities = torch.sigmoid(
                model(x.to(device), edge_index.to(device), edge_attr.to(device))
            ).detach().cpu().tolist()
            for node, probability in zip(sample["nodes"], probabilities):
                sequential.assign_stage2_probabilities(node, probability)


def serializable_args(args):
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
    return payload


def expected_run_config(variant, spec, args, source_metadata, status):
    return {
        "experiment": 3,
        "variant": variant,
        "status": status,
        "code_version": CODE_VERSION,
        "graph_schema_version": sequential.GRAPH_SCHEMA_VERSION,
        "architecture": {
            "stage_1": "frozen_experiment_1_typed_relation_gnn_3head",
            "stage_2": "experiment_2_class_relation_ppr_gatv2_sage",
            "prior_mode": spec["prior_mode"],
            "prior_channels": list(sequential.PRIOR_NAMES),
            "single_final_nms": True,
        },
        "source": source_metadata,
        "args": serializable_args(args),
        "controlled_contract": (
            "00 and 01 use identical candidates, edges, targets, loss, model "
            "capacity, scoring, and NMS; only zero versus frozen experiment-1 "
            "prior values differ."
        ),
    }


def write_run_config(variant_dir, payload):
    variant_dir.mkdir(parents=True, exist_ok=True)
    base.write_json_atomic(variant_dir / "run_config.json", payload, indent=2)


def config_matches(variant_dir, expected):
    path = Path(variant_dir) / "run_config.json"
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    current.pop("status", None)
    expected = dict(expected)
    expected.pop("status", None)
    return current == expected


def run_variant(
    table_dir,
    variant,
    spec,
    train_samples,
    eval_samples,
    args,
    device,
    source_metadata,
):
    require_confirmed_stage_design()
    ab.set_seed(args.seed)
    prior_mode = spec["prior_mode"]
    variant_dir = Path(table_dir) / variant
    checkpoint_dir = variant_dir / "checkpoints"
    history_path = variant_dir / "metrics_history.csv"
    expected = expected_run_config(
        variant,
        spec,
        args,
        source_metadata,
        status="training_in_progress",
    )
    rows = []
    completed_epoch = 0
    if args.resume_train and config_matches(variant_dir, expected) and history_path.exists():
        rows = base.read_rows(history_path)
        completed_epoch = base.completed_epoch_from_rows(rows)
    write_run_config(variant_dir, expected)

    model = sequential.build_stage2_model(args).to(device)
    capacity = sequential.model_capacity(model)
    capacity["variant"] = variant
    base.write_json_atomic(variant_dir / "model_capacity.json", capacity, indent=2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_pool = prepare_graph_pool(
        train_samples,
        args,
        prior_mode,
        f"tensorize {variant} train graph pool",
    )
    eval_pool = prepare_graph_pool(
        eval_samples,
        args,
        prior_mode,
        f"tensorize {variant} validation graph pool",
    )

    if completed_epoch:
        loaded = base.load_single_stage_resume_state(
            model,
            optimizer,
            checkpoint_dir,
            completed_epoch,
            device,
        )
        completed_epoch = min(completed_epoch, loaded)
        rows = base.rows_through_epoch(rows, completed_epoch)

    for epoch in range(completed_epoch + 1, int(args.epochs) + 1):
        train_value = train_epoch(
            model,
            train_pool,
            args,
            optimizer,
            device,
            epoch,
            prior_mode,
        )
        val_value = validation_loss(
            model,
            eval_pool,
            args,
            device,
            prior_mode,
        )
        if base.should_full_eval_epoch(epoch, args):
            attach_scores(eval_samples, model, args, device, prior_mode)
            row = base.record_epoch(
                variant_dir,
                variant,
                epoch,
                train_value,
                val_value,
                sequential.refined_predictions(eval_samples, args),
                args.eval_gt_path,
            )
        else:
            row = base.record_loss_only_epoch(
                variant,
                epoch,
                train_value,
                val_value,
            )
        rows.append(row)
        base.save_history(variant_dir, rows)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"loss={train_value:.4f}/{val_value:.4f}",
            flush=True,
        )

    selections = base.selection_candidates(rows, args) if args.eval_best_val_loss else []
    for _, selected_epoch in selections:
        selected_row = next(
            (row for row in rows if base.epoch_value(row) == selected_epoch),
            None,
        )
        if selected_row is None or base.has_full_eval(selected_row):
            continue
        if not base.load_single_stage_model_at_epoch(
            model,
            checkpoint_dir,
            selected_epoch,
            device,
        ):
            continue
        attach_scores(eval_samples, model, args, device, prior_mode)
        evaluated = base.record_epoch(
            variant_dir,
            variant,
            selected_epoch,
            selected_row.get("train_loss"),
            selected_row.get("val_loss"),
            sequential.refined_predictions(eval_samples, args),
            args.eval_gt_path,
        )
        rows = base.replace_history_row(rows, evaluated)
        base.save_history(variant_dir, rows)

    completed = expected_run_config(
        variant,
        spec,
        args,
        source_metadata,
        status="completed",
    )
    write_run_config(variant_dir, completed)


def write_experiment_manifest(
    table_dir,
    selected,
    args,
    source_metadata,
    cache_metadata,
    prior_audit,
    status,
):
    payload = {
        "experiment": 3,
        "experiment_name": "sequential_exp1_graph_prior_to_exp2_rescue",
        "status": str(status),
        "code_version": CODE_VERSION,
        "graph_schema_version": sequential.GRAPH_SCHEMA_VERSION,
        "selected_variants": selected,
        "variants": {name: VARIANTS[name] for name in selected},
        "source": source_metadata,
        "detector_caches": cache_metadata,
        "prior_transfer_audit": prior_audit,
        "training_authorized": bool(args.allow_training),
        "detector_inference_allowed": False,
        "single_final_nms": True,
        "controlled_contract": (
            "00 and 01 differ only in the values of five appended prior channels."
        ),
    }
    Path(table_dir).mkdir(parents=True, exist_ok=True)
    base.write_json_atomic(
        Path(table_dir) / "experiment_3_manifest.json",
        payload,
        indent=2,
    )


def main():
    args = parse_args()
    require_confirmed_stage_design()
    selected = select_variants(args.exp3_variants)
    if not args.allow_training:
        raise RuntimeError(
            "Experiment 3 is configured but training was not authorized. "
            "Set ALLOW_MODEL_TRAINING=1 in run_experiment_3.sh only after review."
        )

    ab.configure_class_space(args.class_space)
    if args.class_space != "visdrone10":
        raise ValueError("Experiment 3 is specified for the VisDrone 10-class setup")
    if not 0.0 < float(args.exp3_core_conf) <= 1.0:
        raise ValueError("--exp3_core_conf must be in (0, 1]")
    args.disable_large_preserve = True
    args.size_graph_cluster_mode = "none"
    args.gnn_score_alpha = 1.0
    args.coarse_conf = float(args.rescue_coarse_conf)
    args.fine_conf = float(args.rescue_fine_conf)
    args.output_root = str(Path(args.output_root).resolve())
    table_dir = Path(args.output_root) / "pipeline_ablation"
    if "experiment_3" not in str(table_dir):
        raise ValueError(
            f"Experiment-3 output_root must be inside final/experiment_3: {table_dir}"
        )

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = ab.resolve_torch_device(args.device)
    if args.require_pyg and (base.PyGData is None or base.PyGDataLoader is None):
        raise RuntimeError("torch_geometric is required but unavailable")

    context = load_records_and_targets(args, table_dir)
    args.eval_gt_path = context["eval_gt_path"]
    train_cache, train_cache_paths = load_required_candidate_cache(
        context["train_records"], args, "train"
    )
    eval_cache, eval_cache_paths = load_required_candidate_cache(
        context["eval_records"], args, "eval"
    )
    train_core_cache = sequential.core_candidate_cache(
        context["train_records"], train_cache, args.exp3_core_conf
    )
    eval_core_cache = sequential.core_candidate_cache(
        context["eval_records"], eval_cache, args.exp3_core_conf
    )
    train_samples = sequential.build_samples(
        context["train_records"], context["train_gt"], train_cache, args
    )
    eval_samples = sequential.build_samples(
        context["eval_records"], context["eval_gt"], eval_cache, args
    )

    checkpoint = Path(args.exp3_exp1_checkpoint).resolve()
    source_metadata = {
        "experiment_1_variant": "02_gnn_no_cluster",
        "experiment_1_dependent_variant": "05_gnn_prune_same_pool",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "core_confidence": float(args.exp3_core_conf),
        "stage_1_frozen": True,
    }
    if any(VARIANTS[name]["prior_mode"] == sequential.PRIOR_EXP1 for name in selected):
        train_core_samples = sequential.build_samples(
            context["train_records"],
            context["train_gt"],
            train_core_cache,
            args,
        )
        eval_core_samples = sequential.build_samples(
            context["eval_records"],
            context["eval_gt"],
            eval_core_cache,
            args,
        )
        stage1_model, _ = load_frozen_exp1_model(args, device)
        prior_audit = {
            "train": sequential.score_and_transfer_exp1(
                train_core_samples,
                train_samples,
                stage1_model,
                args,
                device,
            ),
            "eval": sequential.score_and_transfer_exp1(
                eval_core_samples,
                eval_samples,
                stage1_model,
                args,
                device,
            ),
        }
        del stage1_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        sequential.initialize_exp1_priors(train_samples)
        sequential.initialize_exp1_priors(eval_samples)
        prior_audit = {"train": "not_required", "eval": "not_required"}

    cache_metadata = {
        "train": train_cache_paths,
        "eval": eval_cache_paths,
        "detector_inference_performed": False,
    }
    write_experiment_manifest(
        table_dir,
        selected,
        args,
        source_metadata,
        cache_metadata,
        prior_audit,
        status="training_in_progress",
    )

    for variant in selected:
        run_variant(
            table_dir,
            variant,
            VARIANTS[variant],
            train_samples,
            eval_samples,
            args,
            device,
            source_metadata,
        )
    base.summarize_table(table_dir, selected)
    write_experiment_manifest(
        table_dir,
        selected,
        args,
        source_metadata,
        cache_metadata,
        prior_audit,
        status="completed",
    )


if __name__ == "__main__":
    main()
