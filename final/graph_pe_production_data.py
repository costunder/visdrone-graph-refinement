"""Strict, detector-free Experiment-1 graph preparation, one image at a time."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

FINAL = Path(__file__).resolve().parent
ROOT = FINAL.parent
SCRIPTS = FINAL / "experiment_1/scripts"
sys.path.insert(0, str(SCRIPTS))
import run_gois_paper_ablation_curves as base
import run_gois_two_stage_gnn_ablation as ab
import graph_pe_production as production
import graph_pe_run_support as support

ab.configure_class_space("visdrone10")
CACHE_ROOT = FINAL / "experiment_1/runs/graph_pe_production_v1/input_graphs_v1"
SOURCE_CACHE = FINAL / "experiment_1/runs/table6_yolo11_10class_extra_ablation/common_cache"
EXPECTED_SHA = {
    "train_coarse": "839f0d7d136650221054a697f1e2b6ab257092c5e6e740959c119d8e28b6bc4c",
    "train_fine": "43749b5538ebc88869571eac3efd9bedb6e98cf6ef3da236c5560b67e748cbb8",
    "eval_coarse": "7068ab981838222d587ed26cebba89d4e65a75e559d9894148709406948c4a19",
    "eval_fine": "02d74ea2a66ba0f3e263529d98416c547b29800e9f57fd964621bcc5ad8273b2",
}
EXPECTED_COUNTS = {"train": (6471, 1378319), "eval": (548, 156763)}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def approved_args():
    dataset = ROOT / "Full/data/visdrone_det_yolo_10class"
    shared = FINAL / "runs/table6_yolo11_10class_extra_ablation"
    argv = ["graph_pe_production", "--output_root", str(CACHE_ROOT.parent),
            "--model_path", str(shared / "detector/weights/best.pt"),
            "--train_images", str(dataset / "images/train"), "--train_labels", str(dataset / "labels/train"),
            "--eval_images", str(dataset / "images/val"), "--eval_labels", str(dataset / "labels/val"),
            "--ground_truth_path", str(dataset / "annotations/val_coco_gt.json"),
            "--class_space", "visdrone10", "--device", "cpu", "--disable_large_preserve",
            "--cache", "--epochs", "120", "--coarse_conf", ".25", "--fine_conf", ".25",
            "--fine_slice_size", "256", "--fine_overlap", ".2", "--final_nms_iou", ".4",
            "--stage1_keep_conf", ".25", "--source_cache_dir", str(SOURCE_CACHE)]
    original = sys.argv
    try:
        sys.argv = argv
        args = base.clone_args_for_ab(base.parse_args())
    finally:
        sys.argv = original
    args.gnn_hidden_dim, args.gnn_layers = 256, 6
    args.gnn_batch_size, args.gnn_train_steps_per_epoch, args.gnn_val_loss_limit = 32, 0, 0
    fixed = {"class_space": "visdrone10", "coarse_conf": .25, "fine_conf": .25,
             "model_iou": .7, "final_nms_iou": .4, "max_det": 300,
             "coarse_slice_size": 640, "coarse_overlap": .2, "fine_slice_size": 256, "fine_overlap": .2,
             "graph_radius": 256.0, "graph_radius_ratio": .12, "graph_cross_class_radius_ratio": .08,
             "graph_knn": 6, "graph_cross_class_knn": 2, "label_iou": .5,
             "roi_label_iou": .1, "roi_center_margin": .75, "disable_roi_support_target": False,
             "size_graph_cluster_mode": "none", "disable_large_preserve": True}
    for key, expected in fixed.items():
        if getattr(args, key) != expected:
            raise RuntimeError(f"Approved builder setting changed: {key}")
    if base.PIPELINE_CODE_VERSION != 48:
        raise RuntimeError("Input pipeline version must remain 48")
    return args


def iter_json_object(path, chunk_size=1024 * 1024):
    """Stream a top-level object without loading its 300+ MB JSON text at once."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer, position, eof = "", 0, False

        def refill():
            nonlocal buffer, position, eof
            chunk = handle.read(chunk_size)
            buffer, position = buffer[position:] + chunk, 0
            eof = not chunk

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        def token():
            nonlocal position
            whitespace()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()

        def literal(expected):
            nonlocal position
            whitespace()
            if position >= len(buffer) or buffer[position] != expected:
                raise ValueError(f"Invalid JSON object delimiter in {path}: expected {expected}")
            position += 1

        literal("{")
        seen = set()
        whitespace()
        while position < len(buffer) and buffer[position] != "}":
            key = token()
            if not isinstance(key, str) or key in seen:
                raise ValueError(f"Invalid or duplicate JSON key: {key}")
            seen.add(key)
            literal(":")
            value = token()
            yield key, value
            whitespace()
            if position < len(buffer) and buffer[position] == "}":
                break
            literal(",")
        literal("}")
        whitespace()
        if position != len(buffer):
            raise ValueError(f"Trailing JSON data in {path}")


def validate_predictions(predictions, record, source):
    if not isinstance(predictions, list):
        raise ValueError("Each image cache entry must be a list")
    view_type = 2 if source == "coarse" else 3
    for pred in predictions:
        if int(pred["image_id"]) != record.image_id or int(pred["category_id"]) not in range(1, 11):
            raise ValueError("Candidate image/class mapping mismatch")
        if not math.isfinite(pred["score"]) or not .25 <= pred["score"] <= 1:
            raise ValueError("Candidate score outside approved raw .25 pool")
        if pred.get("_view_type") != view_type or not isinstance(pred.get("_view_id"), str):
            raise ValueError("Actual crop/view metadata required; legacy fallback is forbidden")
        if not pred["_view_id"].startswith(f"{source}:{record.image_id}:"):
            raise ValueError("Crop ID/source/image mismatch")
        if pred.get("_candidate_source") != source:
            raise ValueError("Candidate source mismatch")
        for key in ("bbox", "_view_bbox"):
            box = pred.get(key)
            if not isinstance(box, list) or len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError(f"Missing/nonfinite geometry: {key}")
            x, y, width, height = box
            if width <= 0 or height <= 0 or x < -.001 or y < -.001 or x + width > record.width + .01 or y + height > record.height + .01:
                raise ValueError(f"Invalid image-relative geometry: {key}, {box}")


def source_identity(args):
    records = {"train": ab.build_image_records(args.train_images, max_images=0),
               "eval": ab.build_image_records(args.eval_images, args.ground_truth_path, max_images=0)}
    caches = {}
    for split, items in records.items():
        if len(items) != EXPECTED_COUNTS[split][0]:
            raise RuntimeError(f"Full {split} split size mismatch")
        for source in ("coarse", "fine"):
            fn = ab.prediction_cache_path if source == "coarse" else base.fine_prediction_cache_path
            path = fn(SOURCE_CACHE, split, items, args)
            digest = support.file_sha256(path)  # Missing means failure, never inference.
            if digest != EXPECTED_SHA[f"{split}_{source}"]:
                raise RuntimeError(f"Audited detector cache SHA mismatch: {path}")
            caches[f"{split}_{source}"] = {"path": str(path), "sha256": digest}
    labels = [{"path": str(path.relative_to(ROOT)), "sha256": support.file_sha256(path)}
              for path in sorted(Path(args.train_labels).glob("*.txt"))]
    identity = {"schema": "experiment_1_production_graph_cache_v1", "raw_pe_schema": production.RAW_SCHEMA,
                "cache_inputs": caches, "train_labels_digest": canonical_digest(labels),
                "ground_truth_sha256": support.file_sha256(args.ground_truth_path),
                "detector_sha256": support.file_sha256(args.model_path),
                "records_digest": canonical_digest({split: [[r.image_id, r.file_name, r.width, r.height] for r in items] for split, items in records.items()}),
                "builder_sha256": support.file_sha256(ab.__file__),
                "base_runner_sha256": support.file_sha256(base.__file__),
                "eigensolver_sha256": support.file_sha256(production.legacy_ops.__file__),
                "raw_encoder_code_digest": canonical_digest([inspect.getsource(function) for function in (production.eigen_features, production.adjacency, production.sparse_apply)]),
                "data_module_sha256": support.file_sha256(__file__),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    return records, identity


def build_graph(record, candidates, gt, args):
    nodes = ab.build_size_aware_detection_nodes(record, [], candidates, gt, args)
    if len(nodes) != len(candidates) or any(node.det_index != i for i, node in enumerate(nodes)):
        raise RuntimeError("Candidate count/order changed in graph builder")
    x, edges, attrs, targets, weights = ab.build_size_aware_tensors(nodes, record, args)
    for tensor in (x, attrs, targets, weights):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("Nonfinite graph tensor")
    u, mask, pe_support, eig_audit = production.eigen_features(edges, len(nodes))
    return {"x": x, "edge_index": edges, "edge_attr": attrs, "targets": targets[:, :3].contiguous(),
            "weights": weights, "u": u, "mask": mask, "support": pe_support,
            "node_identity": torch.arange(len(nodes), dtype=torch.long),
            "image_id": record.image_id, "candidate_identity_sha256": canonical_digest(candidates),
            "eigensolver_audit": eig_audit, "raw_pe_schema": production.RAW_SCHEMA}


def prepare_inputs():
    args = approved_args()
    records, identity = source_identity(args)
    identity_digest = canonical_digest(identity)
    path = CACHE_ROOT / "manifest.json"
    with support.exclusive_variant_lock(CACHE_ROOT):
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest["identity"] != identity or manifest["status"] != "completed":
                raise RuntimeError("Existing graph-cache identity mismatch; refusing overwrite")
            for entries in manifest["splits"].values():
                for entry in entries:
                    if support.file_sha256(CACHE_ROOT / entry["path"]) != entry["sha256"]:
                        raise RuntimeError("Graph shard changed after construction")
            return manifest
        splits, statistics = {}, {}
        positive, total_nodes = torch.zeros(3), 0
        for split, items in records.items():
            by_id = {str(record.image_id): record for record in items}
            gt = ab.load_gt_by_image(items, args.train_labels) if split == "train" else ab.load_coco_gt_by_image(items, args.ground_truth_path)
            print(f"Preparing {split}: reading verified coarse cache", flush=True)
            coarse = dict(iter_json_object(identity["cache_inputs"][f"{split}_coarse"]["path"]))
            if set(coarse) != set(by_id):
                raise RuntimeError("Coarse cache image IDs differ from full split")
            entries = []
            for key, fine in iter_json_object(identity["cache_inputs"][f"{split}_fine"]["path"]):
                if key not in coarse or key not in by_id:
                    raise RuntimeError("Fine cache image IDs differ from full split")
                record = by_id[key]
                first = coarse.pop(key)
                validate_predictions(first, record, "coarse")
                validate_predictions(fine, record, "fine")
                candidates = first + fine  # no NMS, cutoff, deduplication, or cap
                graph = build_graph(record, candidates, gt.get(record.image_id, []), args)
                graph.update(split=split, input_identity_digest=identity_digest)
                if split == "eval":
                    graph["predictions"] = candidates
                relative = f"{split}/{record.image_id:06d}.pt"
                shard = CACHE_ROOT / relative
                if shard.exists():
                    previous = torch.load(shard, map_location="cpu", weights_only=True)
                    if previous["input_identity_digest"] != identity_digest or previous["candidate_identity_sha256"] != graph["candidate_identity_sha256"]:
                        raise RuntimeError(f"Partial cache has conflicting provenance: {shard}")
                    for tensor_name in ("x", "edge_index", "edge_attr", "targets", "weights", "u", "mask", "support"):
                        if not torch.equal(previous[tensor_name], graph[tensor_name]):
                            raise RuntimeError(f"Partial cache tensor changed: {shard}/{tensor_name}")
                else:
                    support.save_torch_atomic(graph, shard)
                n, e = len(graph["x"]), graph["edge_index"].shape[1]
                entries.append({"image_id": record.image_id, "nodes": n, "edges": e, "path": relative,
                                "sha256": support.file_sha256(shard), "candidate_identity_sha256": graph["candidate_identity_sha256"],
                                "weight_sum": float(graph["weights"].double().sum())})
                if split == "train":
                    positive += (graph["targets"] > .5).sum(0)
                    total_nodes += n
                if len(entries) % 25 == 0 or len(entries) == len(items):
                    progress = {"status": "preparing_cpu_graphs", "split": split, "images": len(entries), "total_images": len(items), "last_image_id": record.image_id, "time": support.utc_now()}
                    support.write_json_atomic(CACHE_ROOT / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            if coarse or len(entries) != len(items) or sum(e["nodes"] for e in entries) != EXPECTED_COUNTS[split][1]:
                raise RuntimeError("Full candidate/image coverage mismatch")
            splits[split] = sorted(entries, key=lambda entry: entry["image_id"])
            statistics[split] = {field: {"median": float(np.median([entry[field] for entry in entries])),
                                        "p95": float(np.percentile([entry[field] for entry in entries], 95)),
                                        "max": max(entry[field] for entry in entries),
                                        "sum": sum(entry[field] for entry in entries)} for field in ("nodes", "edges")}
            statistics[split]["empty_image_ids"] = [e["image_id"] for e in entries if e["nodes"] == 0]
            del coarse, gt
            gc.collect()
        manifest = {"status": "completed", "identity": identity, "identity_digest": identity_digest,
                    "splits": splits, "statistics": statistics,
                    "global_pos_weight": ((total_nodes - positive) / positive.clamp_min(1)).clamp(1, 25).tolist(),
                    "completed_at": support.utc_now(), "detector_inference_performed": False}
        support.write_json_atomic(path, manifest)
        support.write_json_atomic(CACHE_ROOT / "progress.json", {"status": "completed", "time": support.utc_now()})
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(2)
    manifest = prepare_inputs()
    print(json.dumps(manifest["statistics"], indent=2), flush=True)
