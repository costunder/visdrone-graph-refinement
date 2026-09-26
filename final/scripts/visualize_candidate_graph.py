#!/usr/bin/env python3
"""Visualize one experiment-2 candidate graph without training a model."""

import argparse
import importlib.util
import json
import math
import time
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


CLASS_COLORS = {
    1: "#f94144",
    2: "#f3722c",
    3: "#f9c74f",
    4: "#43aa8b",
    5: "#90be6d",
    6: "#00b4d8",
    7: "#577590",
    8: "#9b5de5",
    9: "#f15bb5",
    10: "#ffffff",
}


def parse_args():
    workspace = Path(__file__).resolve().parents[2]
    run_root = workspace / "final/runs/table6_yolo11_10class_extra_ablation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", type=int, default=218)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=workspace / "Full/data/visdrone_det_yolo_10class/images/val",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=workspace / "Full/data/visdrone_det_yolo_10class/annotations/val_coco_gt.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=run_root / "detector/weights/best.pt")
    parser.add_argument("--cache-dir", type=Path, default=run_root / "common_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "final/experiment_2/reports/graph_visualization",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--force-inference", action="store_true")
    return parser.parse_args()


def load_ablation_module():
    path = Path(__file__).with_name("run_gois_two_stage_gnn_ablation.py")
    spec = importlib.util.spec_from_file_location("gois_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_class_space("visdrone10")
    return module


def load_json(path):
    with path.open("r") as handle:
        return json.load(handle)


def find_single_cache(cache_dir, pattern):
    matches = sorted(cache_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cache matching {pattern} in {cache_dir}")
    return matches[-1]


def find_record(ab, args):
    records = ab.build_image_records(args.images_dir, args.ground_truth)
    for record in records:
        if record.image_id == args.image_id:
            return record
    raise ValueError(f"Image id {args.image_id} is not present in {args.ground_truth}")


def graph_args():
    return Namespace(
        dbscan_eps=192.0,
        dbscan_min_samples=2,
        size_graph_cluster_mode="dbscan",
        label_iou=0.5,
        roi_label_iou=0.1,
        roi_center_margin=0.75,
        disable_roi_support_target=False,
        stage1_keep_conf=0.25,
        stage2_fragment_containment_thr=0.3,
        stage2_fragment_iou_thr=0.1,
        disable_large_preserve=True,
        large_keep_conf=0.25,
        graph_radius=256.0,
        graph_knn=6,
    )


def run_low_conf_inference(ab, args, record, output_dir):
    cache_path = output_dir / f"image_{record.image_id:04d}_conf005_candidates.json"
    if cache_path.exists() and not args.force_inference:
        payload = load_json(cache_path)
        return payload["coarse"], payload["fine"], payload.get("timing_seconds", {})

    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    timings = {}
    start = time.perf_counter()
    coarse = ab.predict_sliced_image(
        model,
        record,
        conf=0.05,
        iou=0.7,
        max_det=300,
        device=args.device,
        slice_size=640,
        overlap=0.2,
        batch_size=32,
    )
    timings["coarse"] = time.perf_counter() - start

    start = time.perf_counter()
    fine = ab.predict_sliced_image(
        model,
        record,
        conf=0.05,
        iou=0.7,
        max_det=300,
        device=args.device,
        slice_size=256,
        overlap=0.2,
        batch_size=32,
    )
    timings["fine"] = time.perf_counter() - start

    payload = {
        "image_id": record.image_id,
        "file_name": record.file_name,
        "confidence": 0.05,
        "coarse": coarse,
        "fine": fine,
        "timing_seconds": timings,
    }
    with cache_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return coarse, fine, timings


def build_graph(ab, record, predictions, gt_boxes, args):
    start = time.perf_counter()
    nodes = ab.build_size_aware_detection_nodes(record, [], predictions, gt_boxes, args)
    _, edge_index, _, _, _ = ab.build_size_aware_tensors(nodes, record, args)
    elapsed = time.perf_counter() - start
    directed_edges = [
        (int(src), int(dst))
        for src, dst in edge_index.t().tolist()
        if int(src) != int(dst)
    ]
    visual_edges = sorted({tuple(sorted((src, dst))) for src, dst in directed_edges})
    return nodes, directed_edges, visual_edges, elapsed


def densest_crop(nodes, width, height):
    crop_width = min(width, max(420.0, width * 0.44))
    crop_height = min(height, max(300.0, height * 0.52))
    centers = np.asarray([node.center for node in nodes], dtype=np.float32)
    if len(centers) == 0:
        return (0.0, 0.0, float(width), float(height))

    best_count = -1
    best_center = (width / 2.0, height / 2.0)
    for cx, cy in centers:
        count = np.sum(
            (np.abs(centers[:, 0] - cx) <= crop_width / 2.0)
            & (np.abs(centers[:, 1] - cy) <= crop_height / 2.0)
        )
        if int(count) > best_count:
            best_count = int(count)
            best_center = (float(cx), float(cy))

    x1 = min(max(0.0, best_center[0] - crop_width / 2.0), width - crop_width)
    y1 = min(max(0.0, best_center[1] - crop_height / 2.0), height - crop_height)
    return (x1, y1, x1 + crop_width, y1 + crop_height)


def edge_segments(nodes, edges, crop=None):
    regular = []
    clustered = []
    for src, dst in edges:
        first = nodes[src]
        second = nodes[dst]
        if crop is not None:
            x1, y1, x2, y2 = crop
            if not (
                x1 <= first.center[0] <= x2
                and y1 <= first.center[1] <= y2
                and x1 <= second.center[0] <= x2
                and y1 <= second.center[1] <= y2
            ):
                continue
        segment = [first.center, second.center]
        if first.dbscan_label >= 0 and first.dbscan_label == second.dbscan_label:
            clustered.append(segment)
        else:
            regular.append(segment)
    return regular, clustered


def draw_graph(ax, image, nodes, edges, crop=None):
    ax.imshow(image)
    regular, clustered = edge_segments(nodes, edges, crop)
    if regular:
        ax.add_collection(LineCollection(regular, colors="#d9e1e8", linewidths=0.55, alpha=0.24, zorder=2))
    if clustered:
        ax.add_collection(LineCollection(clustered, colors="#00e5ff", linewidths=0.9, alpha=0.55, zorder=3))

    for node in nodes:
        cx, cy = node.center
        if crop is not None:
            x1, y1, x2, y2 = crop
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
        x, y, width, height = node.bbox
        color = CLASS_COLORS.get(node.category_id, "#ffffff")
        low_conf = node.max_score < 0.25
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=0.55 if low_conf else 0.9,
                alpha=0.38 if low_conf else 0.72,
                linestyle="--" if low_conf else "-",
                zorder=4,
            )
        )
        size = 10.0 + 30.0 * math.sqrt(max(0.0, min(1.0, node.max_score)))
        ax.scatter(
            [cx],
            [cy],
            s=size,
            facecolors="none" if low_conf else color,
            edgecolors=color if low_conf else "#101418",
            linewidths=1.1 if low_conf else 0.55,
            alpha=0.95,
            zorder=5,
        )

    if crop is None:
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
    else:
        x1, y1, x2, y2 = crop
        ax.set_xlim(x1, x2)
        ax.set_ylim(y2, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#3d4852")
        spine.set_linewidth(1.0)


def stats_for(nodes, directed_edges, visual_edges, graph_seconds, coarse_count, fine_count):
    return {
        "coarse_candidates": coarse_count,
        "fine_candidates": fine_count,
        "nodes": len(nodes),
        "low_conf_nodes": sum(node.max_score < 0.25 for node in nodes),
        "directed_edges_excluding_self_loops": len(directed_edges),
        "visual_undirected_edges": len(visual_edges),
        "pairwise_slots": len(nodes) ** 2,
        "graph_build_seconds": graph_seconds,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ab = load_ablation_module()
    record = find_record(ab, args)
    gt_by_image = ab.load_coco_gt_by_image([record], args.ground_truth)
    gt_boxes = gt_by_image.get(record.image_id, [])

    coarse_cache_path = find_single_cache(args.cache_dir, "eval_coarse_conf0.250_*.json")
    fine_cache_path = find_single_cache(args.cache_dir, "eval_fine_conf0.250_*.json")
    coarse_cache = load_json(coarse_cache_path)
    fine_cache = load_json(fine_cache_path)
    base_coarse = coarse_cache.get(str(record.image_id), [])
    base_fine = fine_cache.get(str(record.image_id), [])
    low_coarse, low_fine, inference_timings = run_low_conf_inference(ab, args, record, args.output_dir)

    settings = graph_args()
    base_nodes, base_directed, base_edges, base_seconds = build_graph(
        ab, record, list(base_coarse) + list(base_fine), gt_boxes, settings
    )
    low_nodes, low_directed, low_edges, low_seconds = build_graph(
        ab, record, list(low_coarse) + list(low_fine), gt_boxes, settings
    )

    base_stats = stats_for(
        base_nodes, base_directed, base_edges, base_seconds, len(base_coarse), len(base_fine)
    )
    low_stats = stats_for(
        low_nodes, low_directed, low_edges, low_seconds, len(low_coarse), len(low_fine)
    )
    crop = densest_crop(low_nodes, record.width, record.height)
    image = Image.open(record.path).convert("RGB")

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
    figure.patch.set_facecolor("#11161b")
    figure.suptitle(
        f"Experiment 2 candidate graph growth | image {record.image_id}: {record.file_name}",
        fontsize=18,
        fontweight="bold",
        color="#f4f7f9",
    )

    draw_graph(axes[0, 0], image, base_nodes, base_edges)
    axes[0, 0].set_title(
        f"Baseline pool, conf=0.25 | N={len(base_nodes):,}, E={len(base_directed):,}, N^2={len(base_nodes) ** 2:,}",
        fontsize=13,
        pad=9,
    )
    draw_graph(axes[0, 1], image, low_nodes, low_edges)
    pair_ratio = (len(low_nodes) / max(1, len(base_nodes))) ** 2
    axes[0, 1].set_title(
        f"Experiment 2 pool, conf=0.05 | N={len(low_nodes):,}, E={len(low_directed):,}, N^2={len(low_nodes) ** 2:,} ({pair_ratio:.2f}x)",
        fontsize=13,
        pad=9,
    )
    draw_graph(axes[1, 0], image, base_nodes, base_edges, crop=crop)
    axes[1, 0].set_title("Baseline pool | densest-region zoom", fontsize=13, pad=9)
    draw_graph(axes[1, 1], image, low_nodes, low_edges, crop=crop)
    axes[1, 1].set_title("Experiment 2 pool | same-region zoom", fontsize=13, pad=9)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f94144", markeredgecolor="#101418", label="score >= 0.25", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#f9c74f", label="0.05 <= score < 0.25", markersize=8),
        Line2D([0], [0], color="#d9e1e8", linewidth=1.2, alpha=0.7, label="spatial / overlap edge"),
        Line2D([0], [0], color="#00e5ff", linewidth=1.5, label="same DBSCAN-cluster edge"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=11)
    comparison_path = args.output_dir / f"image_{record.image_id:04d}_graph_conf025_vs_conf005.png"
    figure.savefig(comparison_path, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "image": {
            "id": record.image_id,
            "file_name": record.file_name,
            "width": record.width,
            "height": record.height,
            "ground_truth_objects": len(gt_boxes),
        },
        "graph_settings": {
            "candidate_pool": "coarse-640 + global-fine-256 before cross-scale final NMS",
            "dbscan_eps": settings.dbscan_eps,
            "dbscan_min_samples": settings.dbscan_min_samples,
            "graph_radius": settings.graph_radius,
            "graph_knn": settings.graph_knn,
        },
        "conf_0.25": base_stats,
        "conf_0.05": low_stats,
        "low_conf_inference_seconds": inference_timings,
        "growth": {
            "node_ratio": len(low_nodes) / max(1, len(base_nodes)),
            "pairwise_ratio": (len(low_nodes) / max(1, len(base_nodes))) ** 2,
            "directed_edge_ratio": len(low_directed) / max(1, len(base_directed)),
        },
        "outputs": {"comparison": str(comparison_path)},
    }
    metadata_path = args.output_dir / f"image_{record.image_id:04d}_graph_stats.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
