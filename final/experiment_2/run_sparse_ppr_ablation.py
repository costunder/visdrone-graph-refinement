#!/usr/bin/env python3
"""Run experiment-2 sparse GraphSAGE/PPR ablations.

This entrypoint is intentionally separate from experiment 1. It never accepts
the fixed-pool 00/01/02/03/05/08 variants and retains the explicit training
authorization guard used by the main runner.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
FINAL_DIR = EXPERIMENT_DIR.parent
SCRIPT_DIR = FINAL_DIR / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

import run_gois_paper_ablation_curves as base
import run_gois_two_stage_gnn_ablation as ab
import controlled_hetero_training
import hetero_training
import sparse_ppr_sage as exp2


VARIANTS = {
    "03_sparse_edge_sage": {
        "architecture": "edge_sage",
        "include_ppr": False,
        "candidate_mode": "low_conf_multiscale",
    },
    "04_ppr_gatv2_sage": {
        "architecture": "ppr_gatv2_sage",
        "include_ppr": True,
        "candidate_mode": "low_conf_multiscale",
    },
    "05_ppr_gatv2_sage_no_gois": {
        "architecture": "ppr_gatv2_sage",
        "include_ppr": True,
        "candidate_mode": "full_image_only",
    },
    "06_class_relation_ppr_gatv2_sage": {
        "architecture": "class_relation_ppr_gatv2_sage",
        "include_ppr": True,
        "candidate_mode": "low_conf_multiscale",
    },
    "07_hetero_detection_class_view_gatv2": {
        "architecture": "hetero_detection_class_view_gatv2",
        "include_ppr": True,
        "candidate_mode": "low_conf_multiscale",
        "heterogeneous": True,
        "class_correction": True,
        "node_types": ["detection", "class", "view"],
    },
    "08_controlled_hetero_ppr_gatv2_sage": {
        "architecture": "controlled_hetero_ppr_gatv2_sage",
        "include_ppr": True,
        "candidate_mode": "low_conf_multiscale",
        "controlled_heterogeneous": True,
        "controlled_against": "06_class_relation_ppr_gatv2_sage",
        "class_correction": False,
        "node_types": ["detection", "class", "view"],
    },
}
CODE_VERSION = 7
CONTROLLED_HETERO_CODE_VERSION = 1


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--exp2_variants", default="03,04,05")
    extra.add_argument("--common_cache_dir", default="")
    extra.add_argument("--exp2_local_knn", type=int, default=12)
    extra.add_argument("--exp2_cross_class_knn", type=int, default=4)
    extra.add_argument("--exp2_spatial_cell_size", type=float, default=0.0)
    extra.add_argument("--exp2_ppr_knn", type=int, default=8)
    extra.add_argument("--exp2_ppr_alpha", type=float, default=0.15)
    extra.add_argument("--exp2_ppr_steps", type=int, default=8)
    extra.add_argument("--exp2_ppr_frontier", type=int, default=64)
    extra.add_argument("--exp2_attention_heads", type=int, default=4)
    extra.add_argument("--exp2_dropout", type=float, default=0.10)
    extra.add_argument("--exp2_controlled_hetero_hidden_dim", type=int, default=56)
    extra.add_argument("--exp2_controlled_reference_hidden_dim", type=int, default=96)
    extra.add_argument("--exp2_no_gois_conf", type=float, default=0.05)
    extra.add_argument("--exp2_hetero_class_loss_weight", type=float, default=0.50)
    extra.add_argument("--exp2_hetero_background_weight", type=float, default=0.25)
    extra.add_argument("--exp2_hetero_class_threshold", type=float, default=0.35)
    extra.add_argument("--exp2_hetero_class_margin", type=float, default=0.05)
    extra.add_argument(
        "--exp2_hetero_foreground_gate_alpha", type=float, default=1.0
    )
    extra.add_argument(
        "--disable_exp2_hetero_class_correction", action="store_true"
    )
    exp2_args, remaining = extra.parse_known_args()
    original = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = base.clone_args_for_ab(base.parse_args())
    finally:
        sys.argv = original
    for key, value in vars(exp2_args).items():
        setattr(args, key, value)
    return args


def select_variants(text):
    selected = []
    for token in [part.strip() for part in text.split(",") if part.strip()]:
        match = next((name for name in VARIANTS if token in {name, name[:2]}), None)
        if match is None:
            raise ValueError(f"Unknown experiment-2 follow-up {token}. Known: {', '.join(VARIANTS)}")
        if match not in selected:
            selected.append(match)
    return selected


def cache_dir(args):
    if str(args.common_cache_dir).strip():
        return Path(args.common_cache_dir)
    return Path(args.output_root) / "common_cache"


def tagged_low_conf_cache(model, records, args, split_name):
    coarse = ab.generate_or_load_coarse_cache(model, records, args, split_name, cache_dir(args))
    fine = base.generate_or_load_fine_cache(model, records, args, split_name, cache_dir(args))
    combined = {}
    for record in records:
        image_id = str(record.image_id)
        combined[image_id] = [
            dict(prediction, _candidate_source="coarse")
            for prediction in coarse.get(image_id, [])
        ] + [
            dict(prediction, _candidate_source="fine")
            for prediction in fine.get(image_id, [])
        ]
    return combined


def tagged_full_cache(model, records, args, split_name):
    raw = ab.generate_or_load_full_cache(model, records, args, split_name, cache_dir(args))
    return {
        image_id: [dict(prediction, _candidate_source="full") for prediction in predictions]
        for image_id, predictions in raw.items()
    }


def build_samples(records, gt_by_image, candidate_cache, args, candidate_mode):
    if candidate_mode == "full_image_only":
        samples = ab.make_size_aware_detection_samples(
            records, candidate_cache, {}, gt_by_image, args
        )
    else:
        samples = ab.make_size_aware_detection_samples(
            records, {}, candidate_cache, gt_by_image, args
        )
    for sample in samples:
        sample["gt_boxes"] = list(gt_by_image.get(sample["record"].image_id, []))
    return samples


def graph_cache_key(args, include_ppr):
    values = (
        int(exp2.GRAPH_SCHEMA_VERSION),
        int(include_ppr),
        int(args.exp2_local_knn),
        int(args.exp2_cross_class_knn),
        float(args.graph_radius),
        int(args.exp2_ppr_knn),
        float(args.exp2_ppr_alpha),
        int(args.exp2_ppr_steps),
        int(args.exp2_ppr_frontier),
    )
    return "_exp2_sparse_pyg_" + "_".join(str(value).replace(".", "p") for value in values)


def tensors(sample, args, include_ppr):
    return exp2.build_sparse_graph_tensors(ab, sample, args, include_ppr=include_ppr)


def sample_to_pyg(sample, args, include_ppr):
    key = graph_cache_key(args, include_ppr)
    if bool(args.cache_pyg_graphs) and sample.get(key) is not None:
        return sample[key]
    x, edge_index, edge_attr, y, weights = tensors(sample, args, include_ppr)
    data = base.PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, weights=weights)
    if bool(args.cache_pyg_graphs):
        sample[key] = data
    return data


def prepare_graph_pool(samples, args, include_ppr, desc):
    valid = base.nonempty_graph_samples(samples)
    if not base.use_pyg_batching(args) or not bool(args.gnn_precompute_graph_pool):
        return valid
    return [
        sample_to_pyg(sample, args, include_ppr)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def selected_graphs(pool, args, limit, seed, include_ppr, desc):
    selected = base.select_limited_items(pool, limit, seed)
    if base.is_pyg_graph_pool(selected) or not base.use_pyg_batching(args):
        return selected
    return [
        sample_to_pyg(sample, args, include_ppr)
        for sample in tqdm(selected, desc=desc, leave=False)
    ]


def graph_loss(logits, targets, weights):
    return ab.weighted_size_aware_bce_loss(logits, targets, weights, pos_weight=None)


def train_epoch(model, pool, args, optimizer, device, epoch, include_ppr):
    model.train()
    order = selected_graphs(
        pool,
        args,
        args.gnn_train_steps_per_epoch,
        args.seed + epoch * 1009,
        include_ppr,
        "tensorize experiment-2 train graphs",
    )
    losses = []
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    if base.is_pyg_graph_pool(order):
        batches = base.make_pyg_loader(order, args, shuffle=True)
        for batch_index, batch in enumerate(
            tqdm(batches, desc="experiment-2 sparse GNN batches", leave=False),
            start=1,
        ):
            batch = batch.to(device)
            loss = graph_loss(model(batch.x, batch.edge_index, batch.edge_attr), batch.y, batch.weights)
            (loss / accum_steps).backward()
            if batch_index % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        if losses and len(losses) % accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return float(np.mean(losses)) if losses else 0.0

    for sample_index, sample in enumerate(
        tqdm(order, desc="experiment-2 sparse GNN epoch", leave=False),
        start=1,
    ):
        x, edge_index, edge_attr, y, weights = tensors(sample, args, include_ppr)
        loss = graph_loss(
            model(x.to(device), edge_index.to(device), edge_attr.to(device)),
            y.to(device),
            weights.to(device),
        )
        (loss / accum_steps).backward()
        if sample_index % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accum_steps:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def validation_loss(model, pool, args, device, include_ppr):
    selected = selected_graphs(
        pool,
        args,
        args.gnn_val_loss_limit,
        args.seed + 9301,
        include_ppr,
        "tensorize experiment-2 validation graphs",
    )
    model.eval()
    losses = []
    with torch.no_grad():
        if base.is_pyg_graph_pool(selected):
            total = 0.0
            graph_count = 0
            for batch in tqdm(
                base.make_pyg_loader(selected, args, shuffle=False),
                desc="experiment-2 sparse validation batches",
                leave=False,
            ):
                batch = batch.to(device)
                loss = graph_loss(
                    model(batch.x, batch.edge_index, batch.edge_attr),
                    batch.y,
                    batch.weights,
                )
                total += float(loss.detach().cpu()) * int(batch.num_graphs)
                graph_count += int(batch.num_graphs)
            return total / graph_count if graph_count else 0.0
        for sample in selected:
            x, edge_index, edge_attr, y, weights = tensors(sample, args, include_ppr)
            loss = graph_loss(
                model(x.to(device), edge_index.to(device), edge_attr.to(device)),
                y.to(device),
                weights.to(device),
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def assign_probabilities(node, probability):
    node.gnn_obj = float(probability[0])
    node.gnn_small = float(probability[1])
    node.gnn_large = float(probability[2])
    node.gnn_roi = max(node.gnn_small, 0.7 * node.gnn_obj + 0.3 * node.gnn_small)
    node.stage2_prob = max(node.gnn_obj, node.gnn_small, node.gnn_large)
    node.stage2_roi_prob = node.gnn_roi


def attach_scores(samples, model, args, device, include_ppr):
    valid = base.nonempty_graph_samples(samples)
    model.eval()
    with torch.no_grad():
        if base.use_pyg_batching(args):
            data_list = [sample_to_pyg(sample, args, include_ppr) for sample in valid]
            offset = 0
            for batch in tqdm(
                base.make_pyg_loader(data_list, args, shuffle=False),
                desc="score experiment-2 sparse graphs",
                leave=False,
            ):
                batch = batch.to(device)
                probabilities = torch.sigmoid(
                    model(batch.x, batch.edge_index, batch.edge_attr)
                ).detach().cpu()
                counts = torch.bincount(batch.batch.cpu(), minlength=batch.num_graphs).tolist()
                cursor = 0
                for local_index, count in enumerate(counts):
                    sample = valid[offset + local_index]
                    for node, probability in zip(
                        sample["nodes"],
                        probabilities[cursor : cursor + len(sample["nodes"])].tolist(),
                    ):
                        assign_probabilities(node, probability)
                    cursor += int(count)
                offset += int(batch.num_graphs)
            return

        for sample in tqdm(valid, desc="score experiment-2 sparse graphs", leave=False):
            x, edge_index, edge_attr, _, _ = tensors(sample, args, include_ppr)
            probabilities = torch.sigmoid(
                model(x.to(device), edge_index.to(device), edge_attr.to(device))
            ).detach().cpu().tolist()
            for node, probability in zip(sample["nodes"], probabilities):
                assign_probabilities(node, probability)


def refined_predictions(samples, args):
    predictions = []
    threshold = float(args.stage1_keep_conf)
    alpha = min(1.0, max(0.0, float(args.gnn_score_alpha)))
    for sample in samples:
        image_predictions = []
        for node in sample["nodes"]:
            source = sample["full_predictions"] if node.source == ab.SOURCE_FULL else sample["predictions"]
            index = int(node.det_index)
            if not 0 <= index < len(source):
                continue
            prediction = source[index]
            detector_score = float(prediction.get("score", 0.0))
            size_score = node.gnn_large if node.union_area >= ab.SMALL_MEDIUM_AREA_THR else node.gnn_small
            graph_score = max(float(node.gnn_obj), float(size_score))
            score = (1.0 - alpha) * detector_score + alpha * graph_score
            if score < threshold:
                continue
            adjusted = dict(prediction)
            adjusted.pop("_candidate_source", None)
            adjusted["score"] = max(0.0, min(1.0, float(score)))
            image_predictions.append(adjusted)
        predictions.extend(
            ab.classwise_nms(
                image_predictions,
                iou_threshold=float(args.final_nms_iou),
                limit=int(args.max_det),
            )
        )
    return predictions


def serializable_args(args):
    values = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            values[key] = value
    return values


def write_run_config(variant_dir, variant, args, spec, status):
    payload = {
        "experiment": 2,
        "variant": variant,
        "status": status,
        "code_version": CODE_VERSION,
        "architecture": spec,
        "args": serializable_args(args),
        "gois_policy": (
            "raw low-confidence coarse+fine candidates followed by learned rerank and standard NMS"
            if spec["candidate_mode"] == "low_conf_multiscale"
            else "strict no-GOIS dependency test using full-image candidates only and standard NMS"
        ),
    }
    variant_dir.mkdir(parents=True, exist_ok=True)
    with (variant_dir / "run_config.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def config_matches(variant_dir, expected):
    path = variant_dir / "run_config.json"
    if not path.exists():
        return False
    try:
        with path.open("r") as handle:
            current = json.load(handle)
        reference = dict(expected)
        current.pop("status", None)
        reference.pop("status", None)
        return current == reference
    except Exception:
        return False


def run_variant(table_dir, variant, spec, train_samples, eval_samples, args, device):
    if spec.get("controlled_heterogeneous"):
        return controlled_hetero_training.run_variant(
            table_dir,
            variant,
            spec,
            train_samples,
            eval_samples,
            args,
            device,
            CONTROLLED_HETERO_CODE_VERSION,
            assign_probabilities,
            refined_predictions,
        )
    if spec.get("heterogeneous"):
        return hetero_training.run_variant(
            table_dir,
            variant,
            spec,
            train_samples,
            eval_samples,
            args,
            device,
            CODE_VERSION,
        )

    variant_dir = table_dir / variant
    ckpt_dir = variant_dir / "checkpoints"
    history_path = variant_dir / "metrics_history.csv"
    expected = {
        "experiment": 2,
        "variant": variant,
        "status": "training_in_progress",
        "code_version": CODE_VERSION,
        "architecture": spec,
        "args": serializable_args(args),
        "gois_policy": (
            "raw low-confidence coarse+fine candidates followed by learned rerank and standard NMS"
            if spec["candidate_mode"] == "low_conf_multiscale"
            else "strict no-GOIS dependency test using full-image candidates only and standard NMS"
        ),
    }
    rows = []
    completed_epoch = 0
    if args.resume_train and config_matches(variant_dir, expected) and history_path.exists():
        rows = base.read_rows(history_path)
        completed_epoch = base.completed_epoch_from_rows(rows)
    write_run_config(variant_dir, variant, args, spec, status="training_in_progress")

    model = exp2.build_model(ab, args, spec["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_pool = prepare_graph_pool(
        train_samples,
        args,
        spec["include_ppr"],
        f"tensorize {variant} train graph pool",
    )
    eval_pool = prepare_graph_pool(
        eval_samples,
        args,
        spec["include_ppr"],
        f"tensorize {variant} validation graph pool",
    )
    if completed_epoch:
        loaded = base.load_single_stage_resume_state(model, optimizer, ckpt_dir, completed_epoch, device)
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
            spec["include_ppr"],
        )
        val_value = validation_loss(model, eval_pool, args, device, spec["include_ppr"])
        if base.should_full_eval_epoch(epoch, args):
            attach_scores(eval_samples, model, args, device, spec["include_ppr"])
            row = base.record_epoch(
                variant_dir,
                variant,
                epoch,
                train_value,
                val_value,
                refined_predictions(eval_samples, args),
                args.eval_gt_path,
            )
        else:
            row = base.record_loss_only_epoch(variant, epoch, train_value, val_value)
        rows.append(row)
        base.save_history(variant_dir, rows)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        print(f"[{variant}] epoch {epoch}/{args.epochs} loss={train_value:.4f}/{val_value:.4f}", flush=True)

    selections = base.selection_candidates(rows, args) if args.eval_best_val_loss else []
    for _, selected_epoch in selections:
        selected_row = next((row for row in rows if base.epoch_value(row) == selected_epoch), None)
        if selected_row is None or base.has_full_eval(selected_row):
            continue
        if not base.load_single_stage_model_at_epoch(model, ckpt_dir, selected_epoch, device):
            continue
        attach_scores(eval_samples, model, args, device, spec["include_ppr"])
        evaluated = base.record_epoch(
            variant_dir,
            variant,
            selected_epoch,
            selected_row.get("train_loss"),
            selected_row.get("val_loss"),
            refined_predictions(eval_samples, args),
            args.eval_gt_path,
        )
        rows = base.replace_history_row(rows, evaluated)
        base.save_history(variant_dir, rows)
    write_run_config(variant_dir, variant, args, spec, status="completed")


def write_experiment_manifest(table_dir, selected, args):
    payload = {
        "experiment": 2,
        "experiment_name": "low_confidence_candidate_rescue",
        "selected_variants": selected,
        "shared_detector_cache": str(cache_dir(args).resolve()),
        "output_root": str(table_dir.resolve()),
        "training_authorized": bool(args.allow_training),
        "code_version": CODE_VERSION,
        "variants": {variant: VARIANTS[variant] for variant in selected},
        "separation_policy": "Experiment 1 variants are rejected by this entrypoint.",
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "experiment_2_followup_manifest.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    selected = select_variants(args.exp2_variants)
    if not args.allow_training:
        raise RuntimeError(
            "Experiment-2 follow-ups require training, but training was not authorized. "
            "Review the configuration, then pass --allow_training explicitly."
        )
    ab.configure_class_space(args.class_space)
    ab.set_seed(args.seed)
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = ab.resolve_torch_device(args.device)
    if args.require_pyg and (base.PyGData is None or base.PyGDataLoader is None):
        raise RuntimeError("torch_geometric is required but unavailable")

    args.output_root = str(Path(args.output_root).resolve())
    table_dir = Path(args.output_root) / "pipeline_ablation"
    if "experiment_2" not in str(table_dir):
        raise ValueError(f"Experiment-2 output_root must be inside final/experiment_2: {table_dir}")

    context_args = base.clone_args_with(args, source_cache_dir="")
    context = base.load_common_context(
        context_args,
        table_dir,
        need_train=True,
        need_gois_candidates=False,
        need_base_cache=False,
        need_full_predictions=False,
    )
    args.eval_gt_path = context["eval_gt_path"]
    write_experiment_manifest(table_dir, selected, args)

    low_conf_specs = [VARIANTS[name] for name in selected if VARIANTS[name]["candidate_mode"] == "low_conf_multiscale"]
    low_train_cache = low_eval_cache = None
    if low_conf_specs:
        low_args = base.clone_args_with(
            args,
            coarse_conf=float(args.rescue_coarse_conf),
            fine_conf=float(args.rescue_fine_conf),
            size_graph_cluster_mode="none",
            disable_large_preserve=True,
            gnn_score_alpha=1.0,
        )
        low_train_cache = tagged_low_conf_cache(context["model"], context["train_records"], low_args, "train")
        low_eval_cache = tagged_low_conf_cache(context["model"], context["eval_records"], low_args, "eval")

    full_train_cache = full_eval_cache = None
    if any(VARIANTS[name]["candidate_mode"] == "full_image_only" for name in selected):
        full_args = base.clone_args_with(
            args,
            full_conf=float(args.exp2_no_gois_conf),
            size_graph_cluster_mode="none",
            disable_large_preserve=True,
            gnn_score_alpha=1.0,
        )
        full_train_cache = tagged_full_cache(context["model"], context["train_records"], full_args, "train")
        full_eval_cache = tagged_full_cache(context["model"], context["eval_records"], full_args, "eval")

    for variant in selected:
        spec = VARIANTS[variant]
        variant_args = base.clone_args_with(
            args,
            coarse_conf=float(args.rescue_coarse_conf),
            fine_conf=float(args.rescue_fine_conf),
            full_conf=float(args.exp2_no_gois_conf),
            size_graph_cluster_mode="none",
            disable_large_preserve=True,
            gnn_score_alpha=1.0,
        )
        if spec.get("controlled_heterogeneous"):
            variant_args.gnn_hidden_dim = int(
                args.exp2_controlled_hetero_hidden_dim
            )
        if spec["candidate_mode"] == "full_image_only":
            train_cache, eval_cache = full_train_cache, full_eval_cache
        else:
            train_cache, eval_cache = low_train_cache, low_eval_cache
        train_samples = build_samples(
            context["train_records"],
            context["train_gt"],
            train_cache,
            variant_args,
            spec["candidate_mode"],
        )
        eval_samples = build_samples(
            context["eval_records"],
            context["eval_gt"],
            eval_cache,
            variant_args,
            spec["candidate_mode"],
        )
        run_variant(
            table_dir,
            variant,
            spec,
            train_samples,
            eval_samples,
            variant_args,
            device,
        )

    existing = [
        "00_gnn_conf_rescue",
        "01_size_refinement_conf_rescue",
        "02_low_conf_cluster_token_hgnn_refinement",
    ]
    completed_followups = [
        variant
        for variant in VARIANTS
        if (table_dir / variant / "metrics_history.csv").exists()
    ]
    base.summarize_table(table_dir, existing + completed_followups)


if __name__ == "__main__":
    main()
