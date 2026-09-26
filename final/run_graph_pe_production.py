#!/usr/bin/env python3
"""Foreground full-data H256/L6 Experiment-1 production runner.

No detector invocation, architecture-size switches, daemon, scheduler, or
automatic restart. The process owns preparation -> conformance -> training ->
three fixed COCO evaluations. Resume requires an explicit completed epoch.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import graph_pe_production as model_code
import graph_pe_production_data as data
import graph_pe_run_support as support

FINAL = Path(__file__).resolve().parent
APPROVAL = FINAL / "manifests/graph_pe_production_proposal.json"
MATCHED_APPROVAL = FINAL / "manifests/matched_pe_execution_20260923.json"


def approval_path(args):
    return MATCHED_APPROVAL if getattr(args, "study", "graph_pe_production_v1") == "matched_pe_v2" else APPROVAL


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(model_code.EXPECTED_E1), required=True)
    parser.add_argument("--study", choices=("graph_pe_production_v1", "matched_pe_v2"), default="graph_pe_production_v1")
    parser.add_argument("--alignment", choices=("aligned", "node_permuted_control"), default="aligned")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--allow_training", action="store_true")
    parser.add_argument("--allow_evaluation", action="store_true")
    parser.add_argument("--resume_epoch", type=int)
    return parser.parse_args()


def require_authorization(args):
    support.require_run_authorization(args)
    approval = json.loads(approval_path(args).read_text())
    if not all(approval.get(key) is True for key in ("implementation_authorized", "training_authorized", "gpu_validation_authorized", "evaluation_authorized")):
        raise RuntimeError("Explicit production implementation/GPU/training/evaluation approval missing")
    if "experiment_1" not in approval.get("execution_authorized_experiments", []):
        raise RuntimeError("Experiment-1 execution not approved")
    if getattr(args, "study", "graph_pe_production_v1") == "matched_pe_v2":
        for attr, key in (("mode", "allowed_modes"), ("seed", "allowed_seeds"), ("alignment", "allowed_alignments")):
            if getattr(args, attr) not in approval.get(key, []):
                raise RuntimeError(f"Matched study {attr} not authorized")
    if args.mode == "no_pe" and args.alignment != "aligned":
        raise ValueError("No-PE has no alignment-control profile")
    if str(args.device) != "cuda:0":
        raise RuntimeError("Production training requires the approved CUDA device; no CPU fallback")


def run_directory(args):
    if getattr(args, "study", "graph_pe_production_v1") == "matched_pe_v2":
        name = {"no_pe": "02_gnn_no_cluster", "signnet": "06_gnn_no_cluster_signnet_pe", "rpearl": "07_gnn_no_cluster_rpearl_pe"}[args.mode]
        return FINAL / "experiment_1/runs/matched_pe_v2" / name / f"seed_{args.seed}"
    local_id = {"no_pe": "09", "signnet": "10", "rpearl": "11"}[args.mode]
    profile = "main" if args.mode == "no_pe" else args.alignment
    return FINAL / "experiment_1/runs/graph_pe_production_v1" / local_id / args.mode / profile / f"seed_{args.seed}"


def code_identity():
    paths = [Path(__file__), Path(model_code.__file__), Path(data.__file__), Path(support.__file__),
             Path(model_code.legacy_ops.__file__), Path(data.ab.__file__), Path(data.base.__file__),
             FINAL / "test_graph_pe_production_contract.py", FINAL / "test_graph_pe_production_runner.py",
             FINAL / "GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md",
             FINAL / "MATCHED_PE_EXECUTION_20260923.md", FINAL / "scripts/check_pe_comparison.py",
             FINAL / "test_matched_pe_execution.py"]
    return {str(path.relative_to(FINAL)): support.file_sha256(path) for path in paths}


def configuration(args, manifest):
    return {"study": getattr(args, "study", "graph_pe_production_v1"), "experiment": "experiment_1", "mode": args.mode,
            "explicit_user_approval": json.loads(approval_path(args).read_text())["approval"],
            "alignment": args.alignment, "seed": args.seed, "device": args.device,
            "checkpoint_schema": model_code.CHECKPOINT_SCHEMA,
            "code": code_identity(), "input_identity_digest": manifest["identity_digest"],
            "input_manifest_sha256": support.file_sha256(data.CACHE_ROOT / "manifest.json"),
            "epochs": 120, "virtual_batch_graphs": 32, "microbatch_graphs": 1,
            "backbone_hidden": 256, "backbone_layers": 6,
            "pe_hidden": None if args.mode == "no_pe" else 128, "pe_gin_layers": 0 if args.mode == "no_pe" else 8,
            "signnet_eigenvectors": 32 if args.mode == "signnet" else None,
            "rpearl_samples": 120 if args.mode == "rpearl" else None, "rpearl_filter_terms": 12 if args.mode == "rpearl" else None,
            "signal_chunk": 8, "edge_chunk": 4096, "attention_node_chunk": 256,
            "activation_checkpointing": True, "dtype": "float32", "tf32": False,
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8},
            "scheduler": None, "gradient_clipping": None, "early_stopping": False,
            "train_coverage": "all_nonempty_graphs_once_per_epoch", "val_coverage": "full_split",
            "checkpoint_selection": "three_lowest_unrounded_val_loss_then_highest_AP_earliest_epoch_ties",
            "cache_only": True, "historical_results_reused": False}


def rng_state():
    numpy = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "python": random.getstate(),
            "numpy": {"name": numpy[0], "keys": numpy[1].tolist(), "position": numpy[2], "has_gauss": numpy[3], "cached_gaussian": numpy[4]}}


def restore_rng(state):
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG checkpoint requires CUDA; no CPU resume fallback")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))


def save_checkpoint(path, model, optimizer, epoch, config, history):
    validate_model_identity(model, config)
    if Path(path).exists():
        raise RuntimeError(f"Checkpoint already exists; refusing overwrite: {path}")
    support.save_torch_atomic({"schema": model_code.CHECKPOINT_SCHEMA, "config": config,
                               "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "rng": rng_state(), "history": history}, path)


def load_checkpoint(path, model, optimizer, config, epoch, *, restore_random=True):
    validate_model_identity(model, config)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != model_code.CHECKPOINT_SCHEMA or saved.get("config") != config:
        raise RuntimeError("Checkpoint architecture/mode/seed/code/cache/design mismatch")
    if saved.get("epoch") != epoch or not 1 <= epoch <= 120:
        raise RuntimeError("Only the requested completed epoch can be resumed")
    if len(saved.get("history", [])) != epoch or [row["epoch"] for row in saved["history"]] != list(range(1, epoch+1)):
        raise RuntimeError("Checkpoint does not contain contiguous completed-epoch history")
    if any(key not in saved for key in ("model", "optimizer", "rng")):
        raise RuntimeError("Incomplete portable training checkpoint")
    for group in saved["optimizer"]["param_groups"]:
        if group["lr"] != .001 or group["weight_decay"] != .0001 or tuple(group["betas"]) != (.9, .999) or group["eps"] != 1e-8:
            raise RuntimeError("Checkpoint optimizer differs from approved AdamW contract")
    model.load_state_dict(saved["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
    if restore_random:
        restore_rng(saved["rng"])
    return saved["history"]


def validate_model_identity(model, config):
    model.assert_contract()
    if any(getattr(model, key) != config.get(key) for key in ("mode", "seed", "alignment")):
        raise RuntimeError("Constructed model mode/seed/alignment differs from checkpoint configuration")


def read_graph(entry, manifest):
    path = data.CACHE_ROOT / entry["path"]
    if "sha256" in entry and support.file_sha256(path) != entry["sha256"]:
        raise RuntimeError("Graph shard changed since input conformance")
    graph = torch.load(path, map_location="cpu", weights_only=True)
    if graph["input_identity_digest"] != manifest["identity_digest"] or graph["candidate_identity_sha256"] != entry["candidate_identity_sha256"]:
        raise RuntimeError("Graph input provenance mismatch")
    if graph["raw_pe_schema"] != model_code.RAW_SCHEMA:
        raise RuntimeError("Legacy raw PE schema rejected")
    return graph


def graph_arguments(graph, model, device, epoch):
    tensors = tuple(graph[key].to(device) for key in ("x", "edge_index", "edge_attr"))
    extra = {}
    if model.mode != "no_pe":
        extra["support"] = graph["support"].to(device)
        if model.mode == "signnet":
            extra.update(u=graph["u"].to(device), mask=graph["mask"].to(device))
        else:
            extra["w"] = model_code.probes(len(graph["x"]), seed=model.seed, experiment="experiment_1", split=graph["split"], image_id=graph["image_id"], epoch=epoch).to(device)
        if model.alignment == "node_permuted_control":
            extra["permutation"] = model_code.alignment_permutation(graph["node_identity"], seed=model.seed, experiment="experiment_1", split=graph["split"], image_id=graph["image_id"]).to(device)
    return tensors, extra


def finite_parameters(model, gradients=False):
    values = [p.grad if gradients else p for p in model.parameters()]
    values = [value for value in values if value is not None]
    return bool(torch.stack([torch.isfinite(value).all() for value in values]).all())


def cpu_acceptance():
    reports = []
    for name in ("test_graph_pe_production_contract.py", "test_graph_pe_production_runner.py", "test_matched_pe_execution.py"):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", str(FINAL / name)], env=env, text=True, capture_output=True)
        reports.append({"test": name, "sha256": support.file_sha256(FINAL / name), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(f"CPU conformance failed: {name}\n{result.stdout}\n{result.stderr}")
    from test_graph_pe_production_runner import variance_audit
    variance = variance_audit()
    return {"passed": True, "reports": reports, "rpearl_variance_diagnostic": variance, "time": support.utc_now()}


def gpu_acceptance(args, manifest, *, diagnostic_entries=None):
    """Full-size real-graph forward/backward; weights never reused for training."""
    require_authorization(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Approved GPU unavailable; refusing CPU training fallback")
    device = torch.device(args.device)
    torch.cuda.set_device(device)  # Initialize allocator/device before peak-stat reset.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    entries = [entry for entry in manifest["splits"]["train"] if entry["nodes"]]
    chosen = {}
    if diagnostic_entries is None:
        for field in ("nodes", "edges"):
            ordered = sorted(entries, key=lambda entry: entry[field])
            for quantile in (.5, .95, 1.0):
                entry = ordered[min(len(ordered)-1, int((len(ordered)-1)*quantile))]
                chosen[entry["image_id"]] = entry
    else:
        chosen = {entry["image_id"]: entry for entry in diagnostic_entries}
    reports = []
    for entry in chosen.values():
        torch.cuda.reset_peak_memory_stats(device)
        cuda_rng_before = torch.cuda.get_rng_state(device)
        model = model_code.Experiment1Production(data.ab, args.mode, args.seed, args.alignment).to(device)
        if not torch.equal(cuda_rng_before, torch.cuda.get_rng_state(device)):
            raise RuntimeError("Model initialization modified the independent CUDA RNG stream")
        if model.mode != "no_pe":
            model.gate.data.fill_(.2)  # Audit an open PE path, never a training initialization.
        graph = read_graph(entry, manifest)
        tensors, extra = graph_arguments(graph, model, device, epoch=1)
        counts = {}
        def hook(name):
            def count(*_):
                counts[name] = counts.get(name, 0) + 1
            return count
        handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()
                   if name in {"backbone.node_type_projection", "backbone.head", "encoder.rho"}
                   or name.startswith("backbone.layers.") and (name.count(".") == 2 or "relation_messages." in name)
                   or name.startswith("encoder.layers.") and name.count(".") == 2]
        started = time.monotonic()
        output = model(*tensors, **extra)
        numerator = model_code.bce_numerator(output, graph["targets"].to(device), graph["weights"].to(device), output.new_tensor(manifest["global_pos_weight"]))
        loss = numerator / graph["weights"].sum().to(device)
        if not torch.isfinite(loss):
            raise RuntimeError("GPU conformance loss is nonfinite")
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        if not finite_parameters(model, gradients=True):
            raise RuntimeError("GPU conformance gradient is nonfinite")
        gradients = {name: {"present": parameter.grad is not None,
                             "nonzero": int(torch.count_nonzero(parameter.grad)) if parameter.grad is not None else 0}
                     for name, parameter in model.named_parameters()}
        if model.mode != "no_pe":
            for layer in range(8):
                prefix = f"encoder.layers.{layer}."
                if not any(value["nonzero"] for name, value in gradients.items() if name.startswith(prefix)):
                    raise RuntimeError(f"No real GPU gradient through PE layer {layer}")
        reports.append({"image_id": entry["image_id"], "nodes": entry["nodes"], "edges": entry["edges"],
                        "diagnostic_loss_not_experiment_result": float(loss.detach()), "seconds_forward_backward": elapsed,
                        "registered_parameters": model.assert_contract(), "parameter_gradients": gradients,
                        "operator_calls_including_recomputation": counts, "resources": support.resource_snapshot(device)})
        print(json.dumps({"phase": "gpu_conformance", "image_id": entry["image_id"], "nodes": entry["nodes"], "seconds": elapsed, "peak_allocated": reports[-1]["resources"]["torch_cuda_max_memory_allocated"]}), flush=True)
        for handle in handles:
            handle.remove()
        del model, graph, tensors, extra, output, loss, numerator
        gc.collect()
        torch.cuda.empty_cache()
    return {"passed": True, "mode": args.mode, "scope": "full_split_quantile_graphs" if diagnostic_entries is None else "partial_input_gpu_diagnostic_only",
            "full_input_statistics": manifest.get("statistics"), "graphs": reports, "time": support.utc_now(), "performance_claim": False}


def run_epoch(args, model, manifest, optimizer, epoch, directory, train):
    require_authorization(args)
    device = torch.device(args.device)
    split = "train" if train else "eval"
    entries = [entry for entry in manifest["splits"][split] if entry["nodes"]]
    if train:
        random.Random(args.seed + epoch * 1009).shuffle(entries)
    model.train(train)
    pos_weight = torch.tensor(manifest["global_pos_weight"], dtype=torch.float32, device=device)
    numerator_sum, denominator_sum = 0.0, 0.0
    started = time.monotonic()
    for start in range(0, len(entries), 32):
        graphs = [read_graph(entry, manifest) for entry in entries[start:start + 32]]
        denominator, _ = model_code.virtual_batch_statistics(graphs, pos_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
        for graph in graphs:
            tensors, extra = graph_arguments(graph, model, device, epoch)
            with torch.set_grad_enabled(train):
                logits = model(*tensors, **extra)
                numerator = model_code.bce_numerator(logits, graph["targets"].to(device), graph["weights"].to(device), pos_weight)
                loss = numerator / denominator
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {split} loss: epoch {epoch}, image {graph['image_id']}")
                if train:
                    loss.backward()
                numerator_sum += float(numerator.detach())
            del logits, numerator, loss, tensors, extra
        if train:
            if not finite_parameters(model, gradients=True):
                raise RuntimeError("Nonfinite training gradient; optimizer not advanced")
            optimizer.step()
            if not finite_parameters(model):
                raise RuntimeError("Nonfinite model state after optimizer step")
        denominator_sum += denominator
        processed = min(start + 32, len(entries))
        status = {"status": "training" if train else "validation_loss", "epoch": epoch, "epochs": 120,
                  "graphs_processed": processed, "graphs_total": len(entries), "virtual_batches_processed": start // 32 + 1,
                  "loss_so_far": numerator_sum / denominator_sum, "seconds": time.monotonic()-started,
                  "gate": float(model.gate.detach()) if hasattr(model, "gate") else None,
                  "time": support.utc_now(), "pid": os.getpid()}
        support.write_json_atomic(directory / "progress.json", status)
        if start == 0 or (start // 32 + 1) % 10 == 0 or processed == len(entries):
            print(json.dumps(status), flush=True)
            support.append_jsonl(directory / "resources.jsonl", {**support.resource_snapshot(device), "epoch": epoch, "split": split, "graphs_processed": processed})
        del graphs
    return numerator_sum / denominator_sum


def predictions(model, manifest, args):
    require_authorization(args)
    model.eval()
    all_predictions = []
    builder_args = data.approved_args()
    with torch.no_grad():
        for entry in manifest["splits"]["eval"]:
            graph = read_graph(entry, manifest)
            if len(graph["x"]):
                tensors, extra = graph_arguments(graph, model, args.device, epoch=None)
                scores = model(*tensors, **extra).sigmoid().amax(-1).cpu().tolist()
            else:
                scores = []
            adjusted = []
            for candidate, score in zip(graph["predictions"], scores, strict=True):
                candidate = dict(candidate)
                candidate["score"] = max(candidate["score"], .5 * candidate["score"] + .5 * score)
                adjusted.append(candidate)
            # Exactly the approved E1 fusion and ONE post-graph classwise NMS.
            all_predictions.extend(data.ab.classwise_nms(adjusted, iou_threshold=builder_args.final_nms_iou, limit=builder_args.max_det))
    return all_predictions


def run(args):
    require_authorization(args)  # Before model/cache/device/lock/output work.
    torch.set_num_threads(2)
    cpu_audit = cpu_acceptance()
    print(json.dumps({"phase": "cpu_conformance", "passed": cpu_audit["passed"], "test_suites": len(cpu_audit["reports"]), "variance_rows": len(cpu_audit["rpearl_variance_diagnostic"]["rows"])}), flush=True)
    directory = run_directory(args)
    with support.exclusive_variant_lock(directory):
        config_path = directory / "run_config.json"
        if args.resume_epoch is None and config_path.exists():
            raise RuntimeError("Existing run requires explicit --resume_epoch; no overwrite or automatic restart")
        support.write_json_atomic(directory / "progress.json", {"status": "preparing_inputs", "pid": os.getpid(), "time": support.utc_now()})
        try:
            manifest = data.prepare_inputs()
        except BaseException as error:
            support.write_json_atomic(directory / "progress.json", {"status": "input_preparation_failed", "error": f"{type(error).__name__}: {error}", "time": support.utc_now()})
            raise
        config = configuration(args, manifest)
        if config["study"] == "matched_pe_v2":
            from types import SimpleNamespace
            from scripts.check_pe_comparison import compare_configs
            pair_checks = []
            for other_mode in model_code.EXPECTED_E1:
                if other_mode != args.mode:
                    other = configuration(SimpleNamespace(**{**vars(args), "mode": other_mode}), manifest)
                    result = compare_configs(config, other)
                    if not result["config_comparable"]:
                        raise RuntimeError(f"Matched comparison contract failed: {result}")
                    pair_checks.append({"mode": other_mode, **result})
            support.write_json_atomic(directory / "comparison_config_audit.json", pair_checks)
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise RuntimeError("Existing immutable run configuration differs")
        elif args.resume_epoch is not None:
            raise RuntimeError("Cannot resume a missing run")
        else:
            support.write_json_atomic(config_path, config)
            support.write_json_atomic(directory / "command.json", support.command_provenance())
            support.write_json_atomic(directory / "cpu_conformance.json", cpu_audit)
        try:
            gpu_audit = gpu_acceptance(args, manifest)
            audit_path = directory / ("gpu_conformance.json" if args.resume_epoch is None else f"gpu_resume_{args.resume_epoch:03d}_conformance.json")
            if audit_path.exists():
                raise RuntimeError("Existing GPU audit is immutable; inspect partial run before another launch")
            support.write_json_atomic(audit_path, gpu_audit)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)
            model = model_code.Experiment1Production(data.ab, args.mode, args.seed, args.alignment).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8)
            if not (directory / "model_capacity.json").exists():
                support.write_json_atomic(directory / "model_capacity.json", model.capacity_inventory())
            history, first_epoch = [], 1
            if args.resume_epoch is not None:
                history = load_checkpoint(directory / "checkpoints" / f"epoch_{args.resume_epoch:03d}.pt", model, optimizer, config, args.resume_epoch)
                first_epoch = args.resume_epoch + 1
                future = [path for path in (directory / "checkpoints").glob("epoch_*.pt") if int(path.stem.split("_")[-1]) >= first_epoch]
                if future:
                    raise RuntimeError("Later completed checkpoints exist; refuse to overwrite them")
            # Preserve every completed epoch's portable model+optimizer+RNG checkpoint.
            estimated_bytes = (121 - first_epoch) * model.assert_contract() * 4 * 3 + 512 * 1024**2
            if shutil.disk_usage(directory).free < estimated_bytes:
                raise RuntimeError(f"Insufficient disk for full checkpoint retention: need {estimated_bytes} bytes; no shrinking or deletion fallback")
            for epoch in range(first_epoch, 121):
                train_loss = run_epoch(args, model, manifest, optimizer, epoch, directory, True)
                val_loss = run_epoch(args, model, manifest, optimizer, epoch, directory, False)
                row = support.loss_only_row(f"experiment_1/{args.mode}/{args.alignment}/seed_{args.seed}", epoch, train_loss, val_loss, data.base.METRIC_NAMES)
                history.append(row)
                save_checkpoint(directory / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, config, history)
                support.save_history(directory, history, data.base.METRIC_NAMES)
                support.append_jsonl(directory / "epoch_events.jsonl", {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "completed_at": support.utc_now()})
            plan = support.freeze_checkpoint_selection(directory, history)
            for selected in plan["checkpoints"]:
                epoch = selected["epoch"]
                support.verify_selected_checkpoint(plan, epoch)
                load_checkpoint(selected["checkpoint"], model, None, config, epoch, restore_random=False)
                output = predictions(model, manifest, args)
                row = history[epoch - 1]
                history[epoch - 1] = support.record_evaluation(data.base, directory, row["variant"], epoch, row["train_loss"], row["val_loss"], output, data.approved_args().ground_truth_path, plan)
                support.save_history(directory, history, data.base.METRIC_NAMES)
            support.write_primary_result(directory, history, plan)
            support.write_json_atomic(directory / "progress.json", {"status": "completed", "epochs": 120, "evaluated_checkpoints": 3, "time": support.utc_now()})
        except BaseException as error:
            failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": f"{type(error).__name__}: {error}", "time": support.utc_now(), "pid": os.getpid()}
            support.write_json_atomic(directory / "progress.json", failure)
            snapshot = support.resource_snapshot(args.device)
            snapshot.update(failure, oom=isinstance(error, torch.OutOfMemoryError), non_finite="Nonfinite" in str(error), cuda_error="CUDA" in str(error))
            support.append_jsonl(directory / "resources.jsonl", snapshot)
            raise


if __name__ == "__main__":
    run(parse_args())
