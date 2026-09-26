"""Training and evaluation loop for the true heterogeneous experiment-2 model."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import hetero_detection_class_view as hetero
import run_gois_paper_ablation_curves as base
import run_gois_two_stage_gnn_ablation as ab


def serializable_args(args):
    values = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            values[key] = value
    return values


def graph_cache_key(args, include_ppr):
    values = (
        int(hetero.HETERO_GRAPH_SCHEMA_VERSION),
        int(include_ppr),
        int(args.exp2_local_knn),
        int(args.exp2_cross_class_knn),
        float(args.graph_radius),
        int(args.exp2_ppr_knn),
        float(args.exp2_ppr_alpha),
        int(args.exp2_ppr_steps),
        int(args.exp2_ppr_frontier),
    )
    return "_exp2_hetero_pyg_" + "_".join(
        str(value).replace(".", "p") for value in values
    )


def sample_to_data(sample, args, include_ppr):
    key = graph_cache_key(args, include_ppr)
    if bool(args.cache_pyg_graphs) and sample.get(key) is not None:
        return sample[key]
    data = hetero.build_hetero_graph_data(ab, sample, args, include_ppr=include_ppr)
    if bool(args.cache_pyg_graphs):
        sample[key] = data
    return data


def prepare_pool(samples, args, include_ppr, desc):
    valid = base.nonempty_graph_samples(samples)
    if not bool(args.gnn_precompute_graph_pool):
        return valid
    return [
        sample_to_data(sample, args, include_ppr)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def selected_data(pool, args, limit, seed, include_ppr, desc):
    selected = base.select_limited_items(pool, limit, seed)
    if not selected:
        return []
    if hasattr(selected[0], "x_dict"):
        return selected
    return [
        sample_to_data(sample, args, include_ppr)
        for sample in tqdm(selected, desc=desc, leave=False)
    ]


def hetero_loss(size_logits, class_logits, batch, args):
    detection_store = batch["detection"]
    size_loss = ab.weighted_size_aware_bce_loss(
        size_logits,
        detection_store.y_size,
        detection_store.weights,
        pos_weight=None,
    )
    class_target = detection_store.y_class
    class_loss_per_node = F.cross_entropy(class_logits, class_target, reduction="none")
    positive = class_target < int(ab.NUM_CLASSES)
    background_weight = float(getattr(args, "exp2_hetero_background_weight", 0.25))
    class_weights = torch.where(
        positive,
        detection_store.weights,
        torch.full_like(detection_store.weights, background_weight),
    )
    class_loss = (class_loss_per_node * class_weights).sum() / class_weights.sum().clamp_min(1.0)
    total = size_loss + float(getattr(args, "exp2_hetero_class_loss_weight", 0.50)) * class_loss
    return total, size_loss, class_loss


def _mean_loss_stats(rows):
    if not rows:
        return {"total": 0.0, "size": 0.0, "class": 0.0}
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("total", "size", "class")
    }


def train_epoch(model, pool, args, optimizer, device, epoch, include_ppr):
    model.train()
    selected = selected_data(
        pool,
        args,
        args.gnn_train_steps_per_epoch,
        args.seed + epoch * 1009,
        include_ppr,
        "tensorize hetero train graphs",
    )
    if not selected:
        return _mean_loss_stats([])
    batches = base.make_pyg_loader(selected, args, shuffle=True)
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for batch_index, batch in enumerate(
        tqdm(batches, desc="experiment-2 hetero GNN batches", leave=False),
        start=1,
    ):
        batch = batch.to(device)
        size_logits, class_logits = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
        )
        total, size_loss, class_loss = hetero_loss(size_logits, class_logits, batch, args)
        (total / accum_steps).backward()
        if batch_index % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(
            {
                "total": float(total.detach().cpu()),
                "size": float(size_loss.detach().cpu()),
                "class": float(class_loss.detach().cpu()),
            }
        )
    if losses and len(losses) % accum_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return _mean_loss_stats(losses)


def validation_loss(model, pool, args, device, include_ppr):
    selected = selected_data(
        pool,
        args,
        args.gnn_val_loss_limit,
        args.seed + 9301,
        include_ppr,
        "tensorize hetero validation graphs",
    )
    if not selected:
        return _mean_loss_stats([])
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(
            base.make_pyg_loader(selected, args, shuffle=False),
            desc="experiment-2 hetero validation batches",
            leave=False,
        ):
            batch = batch.to(device)
            size_logits, class_logits = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
            )
            total, size_loss, class_loss = hetero_loss(size_logits, class_logits, batch, args)
            losses.append(
                {
                    "total": float(total.detach().cpu()),
                    "size": float(size_loss.detach().cpu()),
                    "class": float(class_loss.detach().cpu()),
                }
            )
    return _mean_loss_stats(losses)


def _class_index_to_category():
    return {int(index): int(category) for category, index in ab.VISDRONE_TO_CLASS_INDEX.items()}


def assign_probabilities(node, size_probability, class_probability):
    node.gnn_obj = float(size_probability[0])
    node.gnn_small = float(size_probability[1])
    node.gnn_large = float(size_probability[2])
    node.gnn_roi = max(node.gnn_small, 0.7 * node.gnn_obj + 0.3 * node.gnn_small)
    node.stage2_prob = max(node.gnn_obj, node.gnn_small, node.gnn_large)
    node.stage2_roi_prob = node.gnn_roi

    foreground = class_probability[: int(ab.NUM_CLASSES)]
    best_class = int(np.argmax(foreground))
    original_class = int(ab.VISDRONE_TO_CLASS_INDEX.get(int(node.category_id), 0))
    node.gnn_class_probabilities = [float(value) for value in class_probability]
    node.gnn_background_probability = float(class_probability[int(ab.NUM_CLASSES)])
    node.gnn_class_index = best_class
    node.gnn_category_id = _class_index_to_category()[best_class]
    node.gnn_class_confidence = float(foreground[best_class])
    node.gnn_class_margin = float(foreground[best_class] - foreground[original_class])


def attach_scores(samples, model, args, device, include_ppr):
    valid = base.nonempty_graph_samples(samples)
    data_list = [sample_to_data(sample, args, include_ppr) for sample in valid]
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in tqdm(
            base.make_pyg_loader(data_list, args, shuffle=False),
            desc="score experiment-2 hetero graphs",
            leave=False,
        ):
            batch = batch.to(device)
            size_logits, class_logits = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
            )
            size_probabilities = torch.sigmoid(size_logits).detach().cpu()
            class_probabilities = torch.softmax(class_logits, dim=-1).detach().cpu()
            counts = torch.bincount(
                batch["detection"].batch.cpu(),
                minlength=batch.num_graphs,
            ).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[offset + local_index]
                node_count = len(sample["nodes"])
                for node, size_probability, class_probability in zip(
                    sample["nodes"],
                    size_probabilities[cursor : cursor + node_count].tolist(),
                    class_probabilities[cursor : cursor + node_count].tolist(),
                ):
                    assign_probabilities(node, size_probability, class_probability)
                cursor += int(count)
            offset += int(batch.num_graphs)


def _apply_class_correction(node, adjusted, args):
    corrected_category = int(getattr(node, "gnn_category_id", node.category_id))
    if corrected_category == int(node.category_id):
        return False
    confidence = float(getattr(node, "gnn_class_confidence", 0.0))
    margin = float(getattr(node, "gnn_class_margin", 0.0))
    if confidence < float(getattr(args, "exp2_hetero_class_threshold", 0.35)):
        return False
    if margin < float(getattr(args, "exp2_hetero_class_margin", 0.05)):
        return False
    adjusted["category_id"] = corrected_category
    return True


def refined_predictions(samples, args):
    predictions = []
    threshold = float(args.stage1_keep_conf)
    alpha = min(1.0, max(0.0, float(args.gnn_score_alpha)))
    for sample in samples:
        image_predictions = []
        for node in sample["nodes"]:
            source = (
                sample["full_predictions"]
                if node.source == ab.SOURCE_FULL
                else sample["predictions"]
            )
            index = int(node.det_index)
            if not 0 <= index < len(source):
                continue
            prediction = source[index]
            detector_score = float(prediction.get("score", 0.0))
            size_score = (
                node.gnn_large
                if node.union_area >= ab.SMALL_MEDIUM_AREA_THR
                else node.gnn_small
            )
            foreground_probability = 1.0 - float(
                getattr(node, "gnn_background_probability", 0.0)
            )
            foreground_probability = max(0.0, min(1.0, foreground_probability))
            gate_alpha = max(
                0.0,
                min(
                    1.0,
                    float(getattr(args, "exp2_hetero_foreground_gate_alpha", 1.0)),
                ),
            )
            graph_score = max(float(node.gnn_obj), float(size_score))
            graph_score *= (1.0 - gate_alpha) + gate_alpha * foreground_probability
            score = (1.0 - alpha) * detector_score + alpha * graph_score
            if score < threshold:
                continue
            adjusted = {
                key: value
                for key, value in prediction.items()
                if not str(key).startswith("_")
            }
            adjusted["score"] = max(0.0, min(1.0, float(score)))
            if not bool(getattr(args, "disable_exp2_hetero_class_correction", False)):
                _apply_class_correction(node, adjusted, args)
            image_predictions.append(adjusted)
        predictions.extend(
            ab.classwise_nms(
                image_predictions,
                iou_threshold=float(args.final_nms_iou),
                limit=int(args.max_det),
            )
        )
    return predictions


def correction_summary(samples, args):
    total = 0
    changed = 0
    background_sum = 0.0
    for sample in samples:
        for node in sample.get("nodes", []):
            total += 1
            background_sum += float(getattr(node, "gnn_background_probability", 0.0))
            adjusted = {"category_id": int(node.category_id)}
            if _apply_class_correction(node, adjusted, args):
                changed += 1
    correction_enabled = not bool(
        getattr(args, "disable_exp2_hetero_class_correction", False)
    )
    if not correction_enabled:
        changed = 0
    return {
        "detection_nodes": total,
        "class_correction_enabled": correction_enabled,
        "foreground_gate_alpha": float(
            getattr(args, "exp2_hetero_foreground_gate_alpha", 1.0)
        ),
        "applied_class_corrections": changed,
        "correction_rate": changed / max(1, total),
        "mean_background_probability": background_sum / max(1, total),
        "class_threshold": float(getattr(args, "exp2_hetero_class_threshold", 0.35)),
        "class_margin": float(getattr(args, "exp2_hetero_class_margin", 0.05)),
    }


def _write_loss_components(variant_dir, rows):
    path = Path(variant_dir) / "hetero_loss_components.csv"
    fields = [
        "epoch",
        "train_total",
        "train_size",
        "train_class",
        "val_total",
        "val_size",
        "val_class",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_config_payload(variant, args, spec, code_version, status):
    return {
        "experiment": 2,
        "variant": variant,
        "status": status,
        "code_version": int(code_version),
        "architecture": spec,
        "hetero_graph_schema_version": hetero.HETERO_GRAPH_SCHEMA_VERSION,
        "node_types": list(hetero.NODE_TYPES),
        "edge_types": [list(edge_type) for edge_type in hetero.EDGE_TYPES],
        "args": serializable_args(args),
        "gois_policy": (
            "raw low-confidence coarse+fine candidates followed by true heterogeneous "
            "detection/class/view message passing, class correction, and standard NMS"
        ),
    }


def _write_run_config(variant_dir, variant, args, spec, code_version, status):
    payload = _run_config_payload(variant, args, spec, code_version, status)
    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    with (variant_dir / "run_config.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def _config_matches(variant_dir, expected):
    path = Path(variant_dir) / "run_config.json"
    if not path.exists():
        return False
    try:
        with path.open("r") as handle:
            current = json.load(handle)
        current.pop("status", None)
        reference = dict(expected)
        reference.pop("status", None)
        return current == reference
    except Exception:
        return False


def _evaluate_epoch(variant_dir, variant, epoch, train_loss, val_loss, eval_samples, model, args, device, include_ppr):
    attach_scores(eval_samples, model, args, device, include_ppr)
    summary = correction_summary(eval_samples, args)
    summary["epoch"] = int(epoch)
    summary_dir = Path(variant_dir) / "class_correction"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / f"epoch_{int(epoch):03d}.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return base.record_epoch(
        variant_dir,
        variant,
        int(epoch),
        train_loss,
        val_loss,
        refined_predictions(eval_samples, args),
        args.eval_gt_path,
    )


def run_variant(table_dir, variant, spec, train_samples, eval_samples, args, device, code_version):
    variant_dir = Path(table_dir) / variant
    checkpoint_dir = variant_dir / "checkpoints"
    history_path = variant_dir / "metrics_history.csv"
    expected = _run_config_payload(
        variant,
        args,
        spec,
        code_version,
        status="training_in_progress",
    )
    rows = []
    component_rows = []
    completed_epoch = 0
    if args.resume_train and _config_matches(variant_dir, expected) and history_path.exists():
        rows = base.read_rows(history_path)
        completed_epoch = base.completed_epoch_from_rows(rows)
        component_path = variant_dir / "hetero_loss_components.csv"
        if component_path.exists():
            component_rows = [
                row
                for row in base.read_rows(component_path)
                if int(float(row.get("epoch", 0))) <= completed_epoch
            ]
    _write_run_config(
        variant_dir,
        variant,
        args,
        spec,
        code_version,
        status="training_in_progress",
    )

    model = hetero.build_model(ab, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_pool = prepare_pool(
        train_samples,
        args,
        spec["include_ppr"],
        f"tensorize {variant} train hetero graph pool",
    )
    eval_pool = prepare_pool(
        eval_samples,
        args,
        spec["include_ppr"],
        f"tensorize {variant} validation hetero graph pool",
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
        component_rows = [
            row for row in component_rows if int(float(row["epoch"])) <= completed_epoch
        ]

    for epoch in range(completed_epoch + 1, int(args.epochs) + 1):
        train_stats = train_epoch(
            model,
            train_pool,
            args,
            optimizer,
            device,
            epoch,
            spec["include_ppr"],
        )
        val_stats = validation_loss(
            model,
            eval_pool,
            args,
            device,
            spec["include_ppr"],
        )
        if base.should_full_eval_epoch(epoch, args):
            row = _evaluate_epoch(
                variant_dir,
                variant,
                epoch,
                train_stats["total"],
                val_stats["total"],
                eval_samples,
                model,
                args,
                device,
                spec["include_ppr"],
            )
        else:
            row = base.record_loss_only_epoch(
                variant,
                epoch,
                train_stats["total"],
                val_stats["total"],
            )
        rows.append(row)
        component_rows.append(
            {
                "epoch": epoch,
                "train_total": train_stats["total"],
                "train_size": train_stats["size"],
                "train_class": train_stats["class"],
                "val_total": val_stats["total"],
                "val_size": val_stats["size"],
                "val_class": val_stats["class"],
            }
        )
        base.save_history(variant_dir, rows)
        _write_loss_components(variant_dir, component_rows)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch:03d}.pt")
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"loss={train_stats['total']:.4f}/{val_stats['total']:.4f} "
            f"size={train_stats['size']:.4f}/{val_stats['size']:.4f} "
            f"class={train_stats['class']:.4f}/{val_stats['class']:.4f}",
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
        evaluated = _evaluate_epoch(
            variant_dir,
            variant,
            selected_epoch,
            selected_row.get("train_loss"),
            selected_row.get("val_loss"),
            eval_samples,
            model,
            args,
            device,
            spec["include_ppr"],
        )
        rows = base.replace_history_row(rows, evaluated)
        base.save_history(variant_dir, rows)

    _write_run_config(
        variant_dir,
        variant,
        args,
        spec,
        code_version,
        status="completed",
    )
