"""Run-control helpers shared by the two graph-PE experiment entrypoints."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping

import torch


CHECKPOINT_SCHEMA = "graph_pe_training_v2"
SELECTION_POLICY = "exact_top_3_by_unrounded_validation_loss_then_best_ap_earliest_epoch_ties"


def require_approved_production_design() -> None:
    """Keep the rejected 96x3 masked-branch runners out of production.

    Deliberately no CLI/environment bypass: approval of a new design must be
    followed by its implementation and conformance audit, not by unlocking
    these legacy models. CPU model/fixture regressions remain available.
    """
    raise RuntimeError(
        "Legacy graph-PE 96x3 execution is disabled. The legacy 96x3 "
        "masked-branch runner is disabled, including with execution flags set. "
        "See final/GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md. Approving that "
        "proposal does not unlock this legacy runner; use the separately "
        "approved production implementation and its conformance gates."
    )


def require_run_authorization(args) -> None:
    for flag, variable in (
        ("allow_training", "ALLOW_MODEL_TRAINING"),
        ("allow_evaluation", "ALLOW_COCO_EVALUATION"),
    ):
        if not bool(getattr(args, flag, False)) or os.environ.get(variable) != "1":
            raise RuntimeError(
                f"Graph-PE execution requires both --{flag} and {variable}=1 after explicit approval"
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_sha256(path) -> str:
    path = Path(path)
    with path.open("r") as handle:
        payload = json.load(handle)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_json_atomic(path, payload, *, sort_keys: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=sort_keys, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def save_torch_atomic(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_checkpoint_contract(model) -> Dict[str, object]:
    contract = {"model_class": type(model).__name__}
    if hasattr(model, "graph_pe"):
        import graph_positional_encoding as graph_pe

        graph_pe.assert_default_capacity(model.graph_pe)
        contract["graph_pe"] = {
            "mode": model.graph_pe.mode,
            "config": graph_pe.config_dict(model.graph_pe.config),
        }
    return contract


def save_training_checkpoint_atomic(model, optimizer, epoch: int, path) -> None:
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "model_contract": model_checkpoint_contract(model),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    save_torch_atomic(payload, path)


def load_training_checkpoint(model, optimizer, path, device, expected_epoch: int) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Portable graph-PE checkpoint is missing: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"Unsupported graph-PE checkpoint schema: {path}")
    if payload.get("model_contract") != model_checkpoint_contract(model):
        raise RuntimeError(f"Graph-PE checkpoint model/code/config/mode mismatch: {path}")
    epoch = int(payload.get("epoch", -1))
    if epoch != int(expected_epoch):
        raise RuntimeError(
            f"Graph-PE checkpoint epoch mismatch: expected {expected_epoch}, found {epoch}"
        )
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise RuntimeError(f"Incomplete graph-PE checkpoint payload: {path}")
    if hasattr(model, "graph_pe"):
        saved_mask = payload["model_state_dict"].get("graph_pe.activation_mask")
        if saved_mask is None or not torch.equal(
            saved_mask.cpu(), model.graph_pe.activation_mask.detach().cpu()
        ):
            raise RuntimeError(f"Graph-PE checkpoint activation-mask mismatch: {path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return epoch


def validate_prediction_cache(path, records: Iterable[object]) -> Dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required detector cache is missing; detector inference is disabled: {path}"
        )
    with path.open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Detector cache must be a JSON object: {path}")
    expected = [str(int(record.image_id)) for record in records]
    missing = [image_id for image_id in expected if image_id not in payload]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"Detector cache {path} misses {len(missing)} image IDs (first: {preview}); "
            "detector inference will not be used as a fallback"
        )
    invalid = [image_id for image_id in expected if not isinstance(payload[image_id], list)]
    if invalid:
        raise ValueError(
            f"Detector cache entries must be lists; invalid image IDs: {invalid[:8]}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "images": len(expected),
        "candidates": sum(len(payload[image_id]) for image_id in expected),
    }


def read_approved_cache(audit, records):
    """Read verified input only; there is deliberately no detector argument."""
    path = Path(audit["path"])
    if file_sha256(path) != audit["sha256"]:
        raise RuntimeError(f"Detector cache changed after preflight: {path}")
    with path.open("r") as handle:
        payload = json.load(handle)
    if file_sha256(path) != audit["sha256"]:
        raise RuntimeError(f"Detector cache changed while reading: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Detector cache must be an object: {path}")
    for record in records:
        image_id = str(int(record.image_id))
        if image_id not in payload or not isinstance(payload[image_id], list):
            raise ValueError(f"Missing/invalid cached image {image_id}: {path}")
    return payload


def load_cache_only_context(ab, args, cache_audit, *, tag_sources):
    """Keep the original coarse-then-fine order without any NMS or inference."""
    gt_path = Path(args.ground_truth_path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Approved COCO ground truth is missing: {gt_path}")
    train_records = ab.build_image_records(args.train_images, max_images=0)
    eval_records = ab.build_image_records(args.eval_images, args.ground_truth_path, max_images=0)
    context = {
        "train_records": train_records,
        "eval_records": eval_records,
        "train_gt": ab.load_gt_by_image(train_records, args.train_labels),
        "eval_gt": ab.load_coco_gt_by_image(eval_records, str(gt_path)),
        "eval_gt_path": str(gt_path),
    }
    for split, records in (("train", train_records), ("eval", eval_records)):
        coarse = read_approved_cache(cache_audit[f"{split}_coarse"], records)
        fine = read_approved_cache(cache_audit[f"{split}_fine"], records)
        combined = {}
        for record in records:
            image_id = str(int(record.image_id))
            if tag_sources:
                combined[image_id] = [dict(p, _candidate_source="coarse") for p in coarse[image_id]] + [
                    dict(p, _candidate_source="fine") for p in fine[image_id]
                ]
            else:
                combined[image_id] = list(coarse[image_id]) + list(fine[image_id])
        context[f"{split}_candidate_cache"] = combined
    return context


def finite_float(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Graph-PE {name} must be finite")
    return value


def loss_only_row(variant, epoch, train_loss, val_loss, metric_names):
    return {
        "epoch": int(epoch),
        "variant": variant,
        "num_predictions": "",
        "train_loss": finite_float(train_loss, "train_loss"),
        "val_loss": finite_float(val_loss, "val_loss"),
        **{key: "" for key in metric_names},
    }


def write_csv_atomic(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_history(variant_dir, rows, metric_names):
    """Preserve float precision and do not generate empty metric plots."""
    fields = ["epoch", "variant", "num_predictions", "train_loss", "val_loss"] + list(metric_names)
    write_csv_atomic(Path(variant_dir) / "metrics_history.csv", rows, fields)
    loss_fields = ["epoch", "train_loss", "val_loss"]
    write_csv_atomic(
        Path(variant_dir) / "loss_history.csv",
        [{key: row[key] for key in loss_fields} for row in rows],
        loss_fields,
    )


def selection_candidates(rows, top_k=3):
    if int(top_k) != 3:
        raise ValueError("Graph-PE selection requires exactly three checkpoints")
    scored = []
    seen = set()
    for row in rows:
        epoch = int(row["epoch"])
        if epoch <= 0 or epoch in seen:
            raise ValueError("Graph-PE history contains a duplicate/invalid epoch")
        seen.add(epoch)
        scored.append((finite_float(row["val_loss"], "val_loss"), epoch))
    if len(scored) < 3:
        raise ValueError("Graph-PE needs three validation-loss checkpoints before evaluation")
    return [(f"val_loss_top_{rank}", epoch) for rank, (_, epoch) in enumerate(sorted(scored)[:3], 1)]


def freeze_checkpoint_selection(variant_dir, rows, top_k=3):
    selections = selection_candidates(rows, top_k)
    epochs = {epoch for _, epoch in selections}
    evaluated = {int(row["epoch"]) for row in rows if str(row.get("AP", "")).strip()}
    if not evaluated.issubset(epochs):
        raise RuntimeError("Graph-PE history contains evaluation outside the fixed top three")
    by_epoch = {int(row["epoch"]): row for row in rows}
    selected = []
    for reason, epoch in selections:
        checkpoint = Path(variant_dir) / "checkpoints" / f"epoch_{epoch:03d}.pt"
        selected.append({
            "reason": reason, "epoch": epoch,
            "val_loss": finite_float(by_epoch[epoch]["val_loss"], "val_loss"),
            "checkpoint": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint),
        })
    plan = {"policy": SELECTION_POLICY, "checkpoints": selected}
    path = Path(variant_dir) / "checkpoint_selection.json"
    if path.exists():
        with path.open("r") as handle:
            if json.load(handle) != plan:
                raise RuntimeError("Graph-PE checkpoint selection changed on resume")
    else:
        write_json_atomic(path, plan)
    return plan


def verify_selected_checkpoint(plan, epoch):
    selected = next((entry for entry in plan["checkpoints"] if entry["epoch"] == int(epoch)), None)
    if selected is None:
        raise RuntimeError(f"Epoch {epoch} is outside the fixed graph-PE evaluation plan")
    if file_sha256(selected["checkpoint"]) != selected["sha256"]:
        raise RuntimeError(f"Selected graph-PE checkpoint changed: epoch {epoch}")
    return selected


def record_evaluation(base, variant_dir, variant, epoch, train_loss, val_loss, predictions, gt_path, plan):
    selected = verify_selected_checkpoint(plan, epoch)
    pred_path = Path(variant_dir) / "predictions" / f"epoch_{int(epoch):03d}.json"
    write_json_atomic(pred_path, predictions)
    metrics = base.ab.evaluate_predictions(gt_path, str(pred_path))
    metrics = {key: finite_float(metrics[key], key) for key in base.METRIC_NAMES}
    row = loss_only_row(variant, epoch, train_loss, val_loss, base.METRIC_NAMES)
    row.update(metrics)
    row["num_predictions"] = len(predictions)
    write_json_atomic(Path(variant_dir) / "metrics" / f"epoch_{int(epoch):03d}.json", {
        "checkpoint": selected, "predictions_sha256": file_sha256(pred_path),
        "metrics": metrics,
    })
    return row


def write_primary_result(variant_dir, rows, plan):
    epochs = {entry["epoch"] for entry in plan["checkpoints"]}
    evaluated = [row for row in rows if str(row.get("AP", "")).strip()]
    if len(evaluated) != 3 or {int(row["epoch"]) for row in evaluated} != epochs:
        raise RuntimeError("Graph-PE primary result requires exactly the three selected evaluations")
    for row in evaluated:
        epoch = int(row["epoch"])
        entry = verify_selected_checkpoint(plan, epoch)
        with (Path(variant_dir) / "metrics" / f"epoch_{epoch:03d}.json").open("r") as handle:
            artifact = json.load(handle)
        pred_path = Path(variant_dir) / "predictions" / f"epoch_{epoch:03d}.json"
        if artifact["checkpoint"] != entry or artifact["predictions_sha256"] != file_sha256(pred_path):
            raise RuntimeError(f"Graph-PE evaluation provenance mismatch: epoch {epoch}")
        if any(finite_float(row[key], key) != value for key, value in artifact["metrics"].items()):
            raise RuntimeError(f"Graph-PE metric/history mismatch: epoch {epoch}")
    best = min(evaluated, key=lambda row: (-finite_float(row["AP"], "AP"), int(row["epoch"])))
    selected = verify_selected_checkpoint(plan, best["epoch"])
    best = {
        key: value if key == "variant" else int(value) if key in {"epoch", "num_predictions"}
        else finite_float(value, key)
        for key, value in best.items()
    }
    result = {"policy": SELECTION_POLICY, "checkpoint": selected, "row": best}
    write_json_atomic(Path(variant_dir) / "primary_result.json", result)
    return result


def normalized_config(payload: Mapping[str, object]) -> Dict[str, object]:
    normalized = dict(payload)
    normalized.pop("status", None)
    normalized.pop("started_at", None)
    normalized.pop("completed_at", None)
    normalized.pop("failure", None)
    normalized.pop("command", None)
    if isinstance(normalized.get("args"), dict):
        arguments = dict(normalized["args"])
        for key in (
            "allow_training",
            "allow_evaluation",
            "resume_train",
            "force_train",
            "force_predictions",
            "reuse_variant_outputs",
        ):
            arguments.pop(key, None)
        normalized["args"] = arguments
    return normalized


def config_matches(path, expected: Mapping[str, object]) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with path.open("r") as handle:
            current = json.load(handle)
    except Exception:
        return False
    return normalized_config(current) == normalized_config(expected)


@contextlib.contextmanager
def exclusive_variant_lock(variant_dir) -> Iterator[Path]:
    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    lock_path = variant_dir / ".run.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds the graph-PE run lock: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def resource_snapshot(device) -> Dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    payload: Dict[str, object] = {
        "time": utc_now(),
        "pid": os.getpid(),
        "max_rss_kib": int(usage.ru_maxrss),
        "involuntary_context_switches": int(usage.ru_nivcsw),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "cuda_available": bool(torch.cuda.is_available()),
        "oom": False,
        "cuda_error": False,
        "non_finite": False,
    }
    try:
        status_values = {}
        with Path("/proc/self/status").open("r") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmSwap:")):
                    key, value = line.split(":", 1)
                    status_values[key] = int(value.strip().split()[0])
        payload["process_rss_kib"] = int(status_values.get("VmRSS", 0))
        payload["process_swap_kib"] = int(status_values.get("VmSwap", 0))
    except Exception as exc:
        payload["process_memory_error"] = f"{type(exc).__name__}: {exc}"
    device_text = str(device)
    if torch.cuda.is_available() and device_text.startswith("cuda"):
        payload.update(
            {
                "torch_cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
                "torch_cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
        )
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            payload["nvidia_smi"] = [
                line.strip() for line in result.stdout.splitlines() if line.strip()
            ]
        except Exception as exc:  # informational snapshot only
            payload["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def append_jsonl(path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def command_provenance() -> Dict[str, object]:
    return {
        "argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "pid": os.getpid(),
        "python": sys.executable,
    }
