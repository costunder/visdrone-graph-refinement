"""Training loop for the controlled experiment-2 heterogeneous ablation."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import controlled_hetero_graph as controlled_hetero
import run_gois_paper_ablation_curves as base
import run_gois_two_stage_gnn_ablation as ab
import sparse_ppr_sage as sparse_graph


CONTROL_VARIANT = "06_class_relation_ppr_gatv2_sage"


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
        int(controlled_hetero.CONTROLLED_HETERO_GRAPH_SCHEMA_VERSION),
        int(sparse_graph.GRAPH_SCHEMA_VERSION),
        int(include_ppr),
        int(args.exp2_local_knn),
        int(args.exp2_cross_class_knn),
        float(args.graph_radius),
        int(args.exp2_ppr_knn),
        float(args.exp2_ppr_alpha),
        int(args.exp2_ppr_steps),
        int(args.exp2_ppr_frontier),
    )
    return "_exp2_controlled_hetero_pyg_" + "_".join(
        str(value).replace(".", "p")
        for value in values
    )


def sample_to_data(sample, args, include_ppr):
    key = graph_cache_key(args, include_ppr)
    if bool(args.cache_pyg_graphs) and sample.get(key) is not None:
        return sample[key]
    data = controlled_hetero.build_controlled_hetero_graph_data(
        ab,
        sample,
        args,
        include_ppr=include_ppr,
    )
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


def controlled_loss(logits, batch):
    detection_store = batch["detection"]
    return ab.weighted_size_aware_bce_loss(
        logits,
        detection_store.y,
        detection_store.weights,
        pos_weight=None,
    )


def train_epoch(model, pool, args, optimizer, device, epoch, include_ppr):
    model.train()
    selected = selected_data(
        pool,
        args,
        args.gnn_train_steps_per_epoch,
        args.seed + epoch * 1009,
        include_ppr,
        "tensorize controlled hetero train graphs",
    )
    if not selected:
        return 0.0

    batches = base.make_pyg_loader(selected, args, shuffle=True)
    accum_steps = max(1, int(args.gnn_grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for batch_index, batch in enumerate(
        tqdm(
            batches,
            desc="experiment-2 controlled hetero batches",
            leave=False,
        ),
        start=1,
    ):
        batch = batch.to(device)
        logits = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
        )
        loss = controlled_loss(logits, batch)
        (loss / accum_steps).backward()
        if batch_index % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accum_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def validation_loss(model, pool, args, device, include_ppr):
    selected = selected_data(
        pool,
        args,
        args.gnn_val_loss_limit,
        args.seed + 9301,
        include_ppr,
        "tensorize controlled hetero validation graphs",
    )
    if not selected:
        return 0.0

    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(
            base.make_pyg_loader(selected, args, shuffle=False),
            desc="experiment-2 controlled hetero validation",
            leave=False,
        ):
            batch = batch.to(device)
            logits = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
            )
            losses.append(float(controlled_loss(logits, batch).detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def attach_scores(
    samples,
    model,
    args,
    device,
    include_ppr,
    assign_probability,
):
    valid = base.nonempty_graph_samples(samples)
    data_list = [
        sample_to_data(sample, args, include_ppr)
        for sample in valid
    ]
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in tqdm(
            base.make_pyg_loader(data_list, args, shuffle=False),
            desc="score experiment-2 controlled hetero graphs",
            leave=False,
        ):
            batch = batch.to(device)
            probabilities = torch.sigmoid(
                model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch.edge_attr_dict,
                )
            ).detach().cpu()
            counts = torch.bincount(
                batch["detection"].batch.cpu(),
                minlength=batch.num_graphs,
            ).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[offset + local_index]
                node_count = len(sample["nodes"])
                for node, probability in zip(
                    sample["nodes"],
                    probabilities[cursor : cursor + node_count].tolist(),
                ):
                    assign_probability(node, probability)
                cursor += int(count)
            offset += int(batch.num_graphs)


def _run_config_payload(variant, args, spec, code_version, status):
    return {
        "experiment": 2,
        "variant": variant,
        "status": status,
        "code_version": int(code_version),
        "architecture": spec,
        "controlled_against": CONTROL_VARIANT,
        "hetero_graph_schema_version": (
            controlled_hetero.CONTROLLED_HETERO_GRAPH_SCHEMA_VERSION
        ),
        "detection_graph_schema_version": sparse_graph.GRAPH_SCHEMA_VERSION,
        "node_types": list(controlled_hetero.NODE_TYPES),
        "edge_types": [
            list(edge_type)
            for edge_type in controlled_hetero.EDGE_TYPES
        ],
        "fixed_protocol": {
            "candidate_pool": "identical_to_06_low_conf_multiscale",
            "detection_features": "identical_to_06",
            "detection_spatial_ppr_edges": "identical_to_06",
            "targets": "identical_to_06_three_channel_size_targets",
            "sample_weights": "identical_to_06",
            "loss": "identical_to_06_weighted_size_aware_bce",
            "score_fusion": "identical_to_06",
            "threshold_and_nms": "identical_to_06",
        },
        "changed_factor": (
            "homogeneous class-pair backbone replaced by sparse relation-specific "
            "detection/class/view heterogeneous message passing"
        ),
        "args": serializable_args(args),
        "gois_policy": (
            "raw low-confidence coarse+fine candidates followed by controlled "
            "heterogeneous reranking and the unchanged variant-06 NMS policy"
        ),
    }


def _write_run_config(
    variant_dir,
    variant,
    args,
    spec,
    code_version,
    status,
):
    payload = _run_config_payload(
        variant,
        args,
        spec,
        code_version,
        status,
    )
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


def _parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _write_model_capacity(variant_dir, model, args):
    reference_args = base.clone_args_with(
        args,
        gnn_hidden_dim=int(args.exp2_controlled_reference_hidden_dim),
    )
    reference_model = sparse_graph.build_model(
        ab,
        reference_args,
        "class_relation_ppr_gatv2_sage",
    )
    variant_parameters = _parameter_count(model)
    reference_parameters = _parameter_count(reference_model)
    payload = {
        "variant": "08_controlled_hetero_ppr_gatv2_sage",
        "reference": CONTROL_VARIANT,
        "variant_trainable_parameters": variant_parameters,
        "reference_trainable_parameters": reference_parameters,
        "parameter_ratio": variant_parameters / max(1, reference_parameters),
        "relative_difference": (
            variant_parameters - reference_parameters
        ) / max(1, reference_parameters),
        "variant_hidden_dim": int(args.gnn_hidden_dim),
        "reference_hidden_dim": int(
            args.exp2_controlled_reference_hidden_dim
        ),
        "layers": int(args.gnn_layers),
        "attention_heads": int(args.exp2_attention_heads),
    }
    with (Path(variant_dir) / "model_capacity.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _relation_mixture(model):
    rows = []
    for layer_index, layer in enumerate(model.layers, start=1):
        by_target = {node_type: [] for node_type in controlled_hetero.NODE_TYPES}
        for edge_type in layer.edge_types:
            key = controlled_hetero._relation_key(edge_type)
            by_target[edge_type[2]].append(
                (
                    edge_type,
                    layer.relation_logits[key].detach().cpu(),
                )
            )
        layer_values = {"layer": layer_index, "targets": {}}
        for target_type, entries in by_target.items():
            if not entries:
                continue
            probabilities = torch.softmax(
                torch.stack([value for _, value in entries]),
                dim=0,
            ).tolist()
            layer_values["targets"][target_type] = {
                "__".join(edge_type): float(probability)
                for (edge_type, _), probability in zip(entries, probabilities)
            }
        rows.append(layer_values)
    return rows


def _write_relation_mixture(variant_dir, model):
    payload = {
        "description": (
            "Learned softmax mixture over relation-specific messages for each "
            "target node type."
        ),
        "layers": _relation_mixture(model),
    }
    with (Path(variant_dir) / "learned_relation_mixture.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _evaluate_epoch(
    variant_dir,
    variant,
    epoch,
    train_loss,
    val_loss,
    eval_samples,
    model,
    args,
    device,
    include_ppr,
    assign_probability,
    refined_predictions,
):
    attach_scores(
        eval_samples,
        model,
        args,
        device,
        include_ppr,
        assign_probability,
    )
    return base.record_epoch(
        variant_dir,
        variant,
        int(epoch),
        train_loss,
        val_loss,
        refined_predictions(eval_samples, args),
        args.eval_gt_path,
    )


def run_variant(
    table_dir,
    variant,
    spec,
    train_samples,
    eval_samples,
    args,
    device,
    code_version,
    assign_probability,
    refined_predictions,
):
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
    completed_epoch = 0
    if (
        args.resume_train
        and _config_matches(variant_dir, expected)
        and history_path.exists()
    ):
        rows = base.read_rows(history_path)
        completed_epoch = base.completed_epoch_from_rows(rows)
    _write_run_config(
        variant_dir,
        variant,
        args,
        spec,
        code_version,
        status="training_in_progress",
    )

    model = controlled_hetero.build_model(ab, args)
    _write_model_capacity(variant_dir, model, args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_pool = prepare_pool(
        train_samples,
        args,
        spec["include_ppr"],
        f"tensorize {variant} train controlled hetero graph pool",
    )
    eval_pool = prepare_pool(
        eval_samples,
        args,
        spec["include_ppr"],
        f"tensorize {variant} validation controlled hetero graph pool",
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
            spec["include_ppr"],
        )
        val_value = validation_loss(
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
                train_value,
                val_value,
                eval_samples,
                model,
                args,
                device,
                spec["include_ppr"],
                assign_probability,
                refined_predictions,
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
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"epoch_{epoch:03d}.pt",
        )
        _write_relation_mixture(variant_dir, model)
        print(
            f"[{variant}] epoch {epoch}/{args.epochs} "
            f"loss={train_value:.4f}/{val_value:.4f}",
            flush=True,
        )

    selections = (
        base.selection_candidates(rows, args)
        if args.eval_best_val_loss
        else []
    )
    for _, selected_epoch in selections:
        selected_row = next(
            (
                row
                for row in rows
                if base.epoch_value(row) == selected_epoch
            ),
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
            assign_probability,
            refined_predictions,
        )
        rows = base.replace_history_row(rows, evaluated)
        base.save_history(variant_dir, rows)

    _write_relation_mixture(variant_dir, model)
    _write_run_config(
        variant_dir,
        variant,
        args,
        spec,
        code_version,
        status="completed",
    )
