#!/usr/bin/env python3
"""Legacy Experiment-2 graph-PE entrypoint; production execution is blocked.

The entrypoint preserves variant 06's low-confidence sparse/PPR contract and
fails on detector-cache misses instead of invoking detector inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
FINAL_DIR = EXPERIMENT_DIR.parent
PROJECT_ROOT = FINAL_DIR.parent
SCRIPT_DIR = FINAL_DIR / "scripts"
for path in (FINAL_DIR, EXPERIMENT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import graph_pe_model
import graph_pe_run_support as run_support
import graph_positional_encoding as graph_pe
import run_gois_paper_ablation_curves as base
import run_gois_two_stage_gnn_ablation as ab
import run_sparse_ppr_ablation as sparse_runner
import sparse_ppr_sage as exp2


RUNNER_CODE_VERSION = 3
PE_VARIANTS = {
    "09_class_relation_ppr_zero_pe": graph_pe.ZERO_MODE,
    "10_class_relation_ppr_signnet_pe": graph_pe.SIGNNET_MODE,
    "11_class_relation_ppr_rpearl_pe": graph_pe.RPEARL_MODE,
}


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--pe_variants", default="09,10,11")
    extra.add_argument("--graph_radius_ratio", type=float, default=0.12)
    extra.add_argument(
        "--allow_evaluation",
        action="store_true",
        help="Explicitly authorize COCO evaluation as well as training.",
    )
    pe_args, remaining = extra.parse_known_args()
    original = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = sparse_runner.parse_args()
    finally:
        sys.argv = original
    args.pe_variants = pe_args.pe_variants
    args.graph_radius_ratio = float(pe_args.graph_radius_ratio)
    args.allow_evaluation = bool(pe_args.allow_evaluation)
    return args


def select_variants(text):
    selected = []
    for token in [part.strip() for part in str(text).split(",") if part.strip()]:
        match = next(
            (name for name in PE_VARIANTS if token in {name, name[:2]}),
            None,
        )
        if match is None:
            raise ValueError(
                f"Unknown Experiment-2 PE variant {token!r}; known: {', '.join(PE_VARIANTS)}"
            )
        if match not in selected:
            selected.append(match)
    if not selected:
        raise ValueError("At least one Experiment-2 PE variant must be selected")
    return selected


def _assert_equal(name, actual, expected) -> None:
    if isinstance(expected, float):
        matches = abs(float(actual) - expected) <= 1e-12
    else:
        matches = actual == expected
    if not matches:
        raise RuntimeError(f"Experiment-2 PE contract fixes {name}={expected!r}, got {actual!r}")


def assert_legacy_configuration_contract(args) -> None:
    expected_paths = {
        "output_root": FINAL_DIR
        / "experiment_2/runs/table6_yolo11_10class_extra_ablation",
        "common_cache_dir": FINAL_DIR
        / "runs/table6_yolo11_10class_extra_ablation/common_cache",
        "model_path": FINAL_DIR
        / "runs/table6_yolo11_10class_extra_ablation/detector/weights/best.pt",
        "train_images": PROJECT_ROOT / "Full/data/visdrone_det_yolo_10class/images/train",
        "train_labels": PROJECT_ROOT / "Full/data/visdrone_det_yolo_10class/labels/train",
        "eval_images": PROJECT_ROOT / "Full/data/visdrone_det_yolo_10class/images/val",
        "eval_labels": PROJECT_ROOT / "Full/data/visdrone_det_yolo_10class/labels/val",
        "ground_truth_path": PROJECT_ROOT
        / "Full/data/visdrone_det_yolo_10class/annotations/val_coco_gt.json",
    }
    for name, expected in expected_paths.items():
        actual = Path(getattr(args, name)).resolve()
        if actual != expected.resolve():
            raise RuntimeError(
                f"Experiment-2 PE contract fixes {name}={expected.resolve()}, got {actual}"
            )
    fixed_values = {
        "class_space": "visdrone10",
        "seed": 42,
        "max_train_images": 0,
        "max_eval_images": 0,
        "epochs": 40,
        "gnn_train_steps_per_epoch": 1024,
        "gnn_val_loss_limit": 512,
        "gnn_hidden_dim": 96,
        "gnn_layers": 3,
        "rescue_coarse_conf": 0.05,
        "rescue_fine_conf": 0.05,
        "model_iou": 0.7,
        "final_nms_iou": 0.4,
        "max_det": 300,
        "coarse_slice_size": 640,
        "coarse_overlap": 0.2,
        "fine_slice_size": 256,
        "fine_overlap": 0.2,
        "stage1_keep_conf": 0.25,
        "exp2_local_knn": 12,
        "exp2_cross_class_knn": 4,
        "exp2_ppr_knn": 8,
        "exp2_ppr_alpha": 0.15,
        "exp2_ppr_steps": 8,
        "exp2_ppr_frontier": 64,
        "exp2_attention_heads": 4,
        "exp2_dropout": 0.10,
        "graph_radius": 256.0,
        "graph_radius_ratio": 0.12,
        "exp2_spatial_cell_size": 0.0,
        "label_iou": 0.50,
        "roi_label_iou": 0.10,
        "roi_center_margin": 0.75,
        "disable_roi_support_target": False,
        "eval_every": 0,
        "eval_best_val_loss": True,
        "selection_top_k": 3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }
    for name, expected in fixed_values.items():
        _assert_equal(name, getattr(args, name), expected)
    if not bool(args.disable_large_preserve):
        raise RuntimeError("Experiment-2 PE requires --disable_large_preserve")
    if not bool(args.cache):
        raise RuntimeError("Experiment-2 PE is cache-only and requires --cache")
    if not bool(args.require_pyg) or base.PyGData is None or base.PyGDataLoader is None:
        raise RuntimeError("Experiment-2 PE requires PyG batching and --require_pyg")
    if int(args.gnn_batch_size) <= 1:
        raise RuntimeError("Experiment-2 legacy PE configuration requires gnn_batch_size > 1")


def preflight_detector_caches(args):
    train_records = ab.build_image_records(args.train_images, max_images=0)
    eval_records = ab.build_image_records(
        args.eval_images,
        args.ground_truth_path,
        max_images=0,
    )
    cache_root = Path(args.common_cache_dir)
    low_args = base.clone_args_with(
        args,
        coarse_conf=float(args.rescue_coarse_conf),
        fine_conf=float(args.rescue_fine_conf),
    )
    cache_specs = [
        (
            "train_coarse",
            ab.prediction_cache_path(cache_root, "train", train_records, low_args),
            train_records,
        ),
        (
            "train_fine",
            base.fine_prediction_cache_path(cache_root, "train", train_records, low_args),
            train_records,
        ),
        (
            "eval_coarse",
            ab.prediction_cache_path(cache_root, "eval", eval_records, low_args),
            eval_records,
        ),
        (
            "eval_fine",
            base.fine_prediction_cache_path(cache_root, "eval", eval_records, low_args),
            eval_records,
        ),
    ]
    audits = {}
    for label, path, records in cache_specs:
        audits[label] = run_support.validate_prediction_cache(path, records)
    audits["detector_weight"] = {
        "path": str(Path(args.model_path).resolve()),
        "sha256": run_support.file_sha256(args.model_path),
    }
    return audits


def tensor_cache_key(args):
    values = (
        graph_pe.GRAPH_PE_SCHEMA_VERSION,
        exp2.GRAPH_SCHEMA_VERSION,
        int(args.seed),
        int(args.exp2_local_knn),
        int(args.exp2_cross_class_knn),
        int(args.exp2_ppr_knn),
        float(args.exp2_ppr_alpha),
        int(args.exp2_ppr_steps),
        int(args.exp2_ppr_frontier),
    )
    return "_experiment2_graph_pe_" + "_".join(
        str(value).replace(".", "p") for value in values
    )


def tensors(sample, args):
    key = tensor_cache_key(args)
    cached = sample.get(key)
    if cached is not None:
        return cached
    x, edge_index, edge_attr, targets, weights = exp2.build_sparse_graph_tensors(
        ab,
        sample,
        args,
        include_ppr=True,
    )
    augmented_x, pe_audit = graph_pe.append_raw_graph_pe(
        x,
        edge_index,
        base_seed=int(args.seed),
        graph_key=("experiment_2", int(sample["record"].image_id)),
    )
    result = (augmented_x, edge_index, edge_attr, targets, weights)
    sample[key] = result
    sample[f"{key}_audit"] = pe_audit
    return result


def sample_to_pyg(sample, args):
    key = f"{tensor_cache_key(args)}_pyg"
    if sample.get(key) is not None:
        return sample[key]
    x, edge_index, edge_attr, targets, weights = tensors(sample, args)
    data = base.PyGData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=targets,
        weights=weights,
    )
    sample[key] = data
    return data


def prepare_graph_pool(samples, args, desc):
    valid = base.nonempty_graph_samples(samples)
    return [
        sample_to_pyg(sample, args)
        for sample in tqdm(valid, desc=desc, leave=False)
    ]


def train_epoch(model, pool, args, optimizer, device, epoch):
    run_support.require_approved_production_design()
    epoch_seed = int(args.seed) + int(epoch) * 1_000_003
    torch.manual_seed(epoch_seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(epoch_seed)
    model.train()
    order = base.select_limited_items(
        pool,
        args.gnn_train_steps_per_epoch,
        args.seed + epoch * 1009,
    )
    loader = base.make_pyg_loader(order, args, shuffle=True)
    accumulation = max(1, int(args.gnn_grad_accum_steps))
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(
        tqdm(loader, desc="Experiment-2 graph-PE train batches", leave=False),
        start=1,
    ):
        batch = batch.to(device)
        loss = sparse_runner.graph_loss(
            model(batch.x, batch.edge_index, batch.edge_attr),
            batch.y,
            batch.weights,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Experiment-2 graph-PE training loss is non-finite")
        (loss / accumulation).backward()
        if batch_index % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if losses and len(losses) % accumulation:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses)) if losses else 0.0


def validation_loss(model, pool, args, device):
    selected = base.select_limited_items(
        pool,
        args.gnn_val_loss_limit,
        args.seed + 9301,
    )
    loader = base.make_pyg_loader(selected, args, shuffle=False)
    total = 0.0
    graph_count = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Experiment-2 graph-PE validation batches",
            leave=False,
        ):
            batch = batch.to(device)
            loss = sparse_runner.graph_loss(
                model(batch.x, batch.edge_index, batch.edge_attr),
                batch.y,
                batch.weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Experiment-2 graph-PE validation loss is non-finite")
            count = int(batch.num_graphs)
            total += float(loss.detach().cpu()) * count
            graph_count += count
    return total / graph_count if graph_count else 0.0


def attach_scores(samples, model, args, device):
    valid = base.nonempty_graph_samples(samples)
    data_list = [sample_to_pyg(sample, args) for sample in valid]
    loader = base.make_pyg_loader(data_list, args, shuffle=False)
    sample_offset = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc="Experiment-2 graph-PE score batches",
            leave=False,
        ):
            batch = batch.to(device)
            probabilities = torch.sigmoid(
                model(batch.x, batch.edge_index, batch.edge_attr)
            ).detach().cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Experiment-2 graph-PE scores are non-finite")
            counts = torch.bincount(
                batch.batch.cpu(),
                minlength=batch.num_graphs,
            ).tolist()
            cursor = 0
            for local_index, count in enumerate(counts):
                sample = valid[sample_offset + local_index]
                for node, probability in zip(
                    sample["nodes"],
                    probabilities[cursor : cursor + int(count)].tolist(),
                ):
                    sparse_runner.assign_probabilities(node, probability)
                cursor += int(count)
            sample_offset += int(batch.num_graphs)


def serializable_args(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, Path)) or value is None
    }


def code_provenance():
    paths = {
        "runner": Path(__file__),
        "model": EXPERIMENT_DIR / "graph_pe_model.py",
        "pe": FINAL_DIR / "graph_positional_encoding.py",
        "run_support": FINAL_DIR / "graph_pe_run_support.py",
        "base_runner": SCRIPT_DIR / "run_gois_paper_ablation_curves.py",
        "base_graph": SCRIPT_DIR / "run_gois_two_stage_gnn_ablation.py",
        "sparse_runner": EXPERIMENT_DIR / "run_sparse_ppr_ablation.py",
        "sparse_graph": EXPERIMENT_DIR / "sparse_ppr_sage.py",
    }
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": run_support.file_sha256(path),
        }
        for name, path in paths.items()
    }


def run_config(variant, mode, args, capacity, cache_audit, status):
    return {
        "experiment": 2,
        "variant": variant,
        "status": status,
        "runner_code_version": RUNNER_CODE_VERSION,
        "checkpoint_schema": run_support.CHECKPOINT_SCHEMA,
        "selection_policy": run_support.SELECTION_POLICY,
        "base_variant": "06_class_relation_ppr_gatv2_sage",
        "controlled_intervention": "graph_pe_activation_mask_only",
        "activation_mode": mode,
        "activation_mask": list(graph_pe.activation_mask(mode)),
        "graph_pe": graph_pe.config_dict(),
        "capacity": capacity,
        "candidate_policy": "same_raw_low_confidence_coarse_plus_fine_pool_as_06",
        "graph_policy": "same_sparse_local_plus_truncated_ppr_detection_graph_as_06",
        "score_policy": "graph_score_alpha_1_cutoff_0.25_then_one_final_nms",
        "args": serializable_args(args),
        "detector_caches": cache_audit,
        "code": code_provenance(),
        "command": run_support.command_provenance(),
    }


def run_variant(
    table_dir,
    variant,
    mode,
    train_samples,
    eval_samples,
    args,
    device,
    cache_audit,
):
    run_support.require_run_authorization(args)
    run_support.require_approved_production_design()
    ab.set_seed(int(args.seed))
    model = graph_pe_model.build_model(ab, args, mode).to(device)
    capacity = model.capacity_inventory()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    variant_dir = Path(table_dir) / variant
    config_path = variant_dir / "run_config.json"
    history_path = variant_dir / "metrics_history.csv"
    checkpoint_dir = variant_dir / "checkpoints"
    expected = run_config(
        variant,
        mode,
        args,
        capacity,
        cache_audit,
        status="training_in_progress",
    )

    with run_support.exclusive_variant_lock(variant_dir):
        rows = []
        completed_epoch = 0
        if args.resume_train:
            if not run_support.config_matches(config_path, expected):
                raise RuntimeError(
                    f"Cannot resume {variant}: run_config does not exactly match the PE contract"
                )
            if not history_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume {variant}: missing {history_path}"
                )
            rows = base.read_rows(history_path)
            completed_epoch = base.completed_epoch_from_rows(rows)
        elif config_path.exists() or history_path.exists() or checkpoint_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing graph-PE artifacts for {variant}; use a new "
                "run directory or an exact --resume_train"
            )

        expected["started_at"] = run_support.utc_now()
        run_support.write_json_atomic(config_path, expected)
        run_support.write_json_atomic(variant_dir / "model_capacity.json", capacity)
        run_support.write_json_atomic(
            variant_dir / "progress.json",
            {
                "status": "tensorizing",
                "variant": variant,
                "completed_epoch": completed_epoch,
                "updated_at": run_support.utc_now(),
            },
        )
        try:
            train_pool = prepare_graph_pool(
                train_samples,
                args,
                f"tensorize {variant} train graphs",
            )
            eval_pool = prepare_graph_pool(
                eval_samples,
                args,
                f"tensorize {variant} validation graphs",
            )
            if completed_epoch:
                run_support.load_training_checkpoint(
                    model,
                    optimizer,
                    checkpoint_dir / f"epoch_{completed_epoch:03d}.pt",
                    device,
                    completed_epoch,
                )
                rows = base.rows_through_epoch(rows, completed_epoch)

            for epoch in range(completed_epoch + 1, int(args.epochs) + 1):
                train_value = train_epoch(
                    model,
                    train_pool,
                    args,
                    optimizer,
                    device,
                    epoch,
                )
                val_value = validation_loss(model, eval_pool, args, device)
                row = run_support.loss_only_row(
                    variant, epoch, train_value, val_value, base.METRIC_NAMES,
                )
                rows.append(row)
                completed_epoch = epoch
                run_support.save_training_checkpoint_atomic(
                    model,
                    optimizer,
                    epoch,
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                )
                run_support.save_history(variant_dir, rows, base.METRIC_NAMES)
                run_support.write_json_atomic(
                    variant_dir / "progress.json",
                    {
                        "status": "training_in_progress",
                        "variant": variant,
                        "completed_epoch": epoch,
                        "updated_at": run_support.utc_now(),
                    },
                )
                run_support.append_jsonl(
                    variant_dir / "resource_usage.jsonl",
                    {
                        "epoch": epoch,
                        **run_support.resource_snapshot(device),
                    },
                )

            selection_plan = run_support.freeze_checkpoint_selection(
                variant_dir, rows, args.selection_top_k,
            )
            for selection in selection_plan["checkpoints"]:
                selected_epoch = int(selection["epoch"])
                run_support.verify_selected_checkpoint(selection_plan, selected_epoch)
                selected_row = next(
                    (row for row in rows if base.epoch_value(row) == selected_epoch),
                    None,
                )
                if selected_row is None or base.has_full_eval(selected_row):
                    continue
                run_support.load_training_checkpoint(
                    model,
                    None,
                    checkpoint_dir / f"epoch_{selected_epoch:03d}.pt",
                    device,
                    selected_epoch,
                )
                attach_scores(eval_samples, model, args, device)
                evaluated = run_support.record_evaluation(
                    base,
                    variant_dir,
                    variant,
                    selected_epoch,
                    selected_row.get("train_loss"),
                    selected_row.get("val_loss"),
                    sparse_runner.refined_predictions(eval_samples, args),
                    args.eval_gt_path,
                    selection_plan,
                )
                rows = base.replace_history_row(rows, evaluated)
                run_support.save_history(variant_dir, rows, base.METRIC_NAMES)

            run_support.write_primary_result(variant_dir, rows, selection_plan)
            completed = dict(expected)
            completed["status"] = "completed"
            completed["completed_at"] = run_support.utc_now()
            run_support.write_json_atomic(config_path, completed)
            run_support.write_json_atomic(
                variant_dir / "progress.json",
                {
                    "status": "completed",
                    "variant": variant,
                    "completed_epoch": int(args.epochs),
                    "updated_at": run_support.utc_now(),
                },
            )
        except Exception as exc:
            failed = dict(expected)
            failed["status"] = "failed"
            failed["failure"] = f"{type(exc).__name__}: {exc}"
            failed["completed_at"] = run_support.utc_now()
            run_support.write_json_atomic(config_path, failed)
            failure_resource = run_support.resource_snapshot(device)
            failure_text = str(exc).lower()
            failure_resource.update(
                {
                    "event": "failure",
                    "oom": "out of memory" in failure_text or "oom" in failure_text,
                    "cuda_error": "cuda" in failure_text,
                    "non_finite": "non-finite" in failure_text or "nan" in failure_text or "inf" in failure_text,
                }
            )
            run_support.append_jsonl(variant_dir / "resource_usage.jsonl", failure_resource)
            run_support.write_json_atomic(
                variant_dir / "progress.json",
                {
                    "status": "failed",
                    "variant": variant,
                    "completed_epoch": completed_epoch,
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": run_support.utc_now(),
                },
            )
            raise


def main():
    args = parse_args()
    selected = select_variants(args.pe_variants)
    run_support.require_run_authorization(args)
    run_support.require_approved_production_design()
    assert_legacy_configuration_contract(args)
    ab.configure_class_space(args.class_space)
    ab.set_seed(args.seed)
    cache_audit = preflight_detector_caches(args)
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = ab.resolve_torch_device(args.device)

    table_dir = Path(args.output_root) / "pipeline_ablation"
    context = run_support.load_cache_only_context(ab, args, cache_audit, tag_sources=True)
    args.eval_gt_path = context["eval_gt_path"]
    variant_args = base.clone_args_with(
        args,
        coarse_conf=float(args.rescue_coarse_conf),
        fine_conf=float(args.rescue_fine_conf),
        size_graph_cluster_mode="none",
        disable_large_preserve=True,
        gnn_score_alpha=1.0,
    )
    train_cache = context["train_candidate_cache"]
    eval_cache = context["eval_candidate_cache"]
    train_samples = sparse_runner.build_samples(
        context["train_records"],
        context["train_gt"],
        train_cache,
        variant_args,
        "low_conf_multiscale",
    )
    eval_samples = sparse_runner.build_samples(
        context["eval_records"],
        context["eval_gt"],
        eval_cache,
        variant_args,
        "low_conf_multiscale",
    )
    if not base.nonempty_graph_samples(train_samples) or not base.nonempty_graph_samples(eval_samples):
        raise RuntimeError("Experiment-2 graph-PE candidate cache produced an empty graph split")

    for variant in selected:
        run_variant(
            table_dir,
            variant,
            PE_VARIANTS[variant],
            train_samples,
            eval_samples,
            variant_args,
            device,
            cache_audit,
        )


if __name__ == "__main__":
    main()
