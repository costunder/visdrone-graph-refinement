#!/usr/bin/env python3
"""Render one real Experiment-1 typed relation graph from cached candidates."""

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

EDGE_STYLES = {
    "spatial": ("#d9e1e8", 0.8, 0.34),
    "cross_class_context": ("#d66efd", 1.15, 0.58),
    "same_cluster": ("#00e5ff", 1.45, 0.68),
    "overlap_or_containment": ("#ff9f1c", 1.8, 0.78),
}


def parse_args():
    workspace = Path(__file__).resolve().parents[3]
    output_dir = workspace / "final/experiment_1/reports/graph_visualization"
    cache_dir = (
        workspace
        / "final/experiment_1/runs/table6_yolo11_10class_extra_ablation/common_cache"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", type=int, default=59)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=workspace / "Full/data/visdrone_det_yolo_10class/images/val",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=workspace
        / "Full/data/visdrone_det_yolo_10class/annotations/val_coco_gt.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=cache_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    return parser.parse_args()


def load_visualization_helper():
    path = Path(__file__).with_name("visualize_candidate_graph.py")
    spec = importlib.util.spec_from_file_location("candidate_graph_visualizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def newest_cache(cache_dir, pattern):
    matches = list(cache_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cache matching {pattern} in {cache_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def find_record(ab, images_dir, ground_truth, image_id):
    records = ab.build_image_records(images_dir, ground_truth)
    for record in records:
        if record.image_id == image_id:
            return record
    raise ValueError(f"Image id {image_id} is not present in {ground_truth}")


def relation_dict(ab, first, second, record, settings):
    flags = ab.edge_relation_flags(first, second, record, settings)
    return {
        name: bool(value)
        for name, value in zip(ab.EDGE_RELATION_NAMES, flags)
    }


def visual_relation(relations):
    # One color per line keeps the drawing readable. The JSON retains every
    # non-exclusive relation flag for each edge.
    if relations["overlap"] or relations["containment"]:
        return "overlap_or_containment"
    if relations["same_cluster"]:
        return "same_cluster"
    if relations["cross_class_context"]:
        return "cross_class_context"
    return "spatial"


def draw_base_image(ax, image, title, darken=False):
    ax.imshow(image)
    if darken:
        overlay = np.zeros((image.height, image.width, 4), dtype=np.float32)
        overlay[..., 3] = 0.34
        ax.imshow(overlay)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, pad=10, color="#f4f7f9", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#52606d")
        spine.set_linewidth(1.0)


def draw_nodes(ax, nodes, class_colors, label_nodes):
    for node in nodes:
        color = class_colors.get(node.category_id, "#ffffff")
        x, y, width, height = node.bbox
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=1.15,
                alpha=0.88,
                zorder=5,
            )
        )
        marker = "s" if node.view_type == 2 else "o" if node.view_type == 3 else "D"
        size = 28.0 + 55.0 * math.sqrt(max(0.0, min(1.0, node.max_score)))
        ax.scatter(
            [node.center[0]],
            [node.center[1]],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor="#11161b",
            linewidth=0.8,
            alpha=0.96,
            zorder=7,
        )
        if label_nodes:
            text = ax.text(
                node.center[0] + 5,
                node.center[1] - 5,
                f"n{node.node_id}",
                fontsize=7.4,
                color="#ffffff",
                zorder=8,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=2.2, foreground="#000000")]
            )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    helper = load_visualization_helper()
    ab = helper.load_ablation_module()
    settings = helper.graph_args()
    record = find_record(ab, args.images_dir, args.ground_truth, args.image_id)
    gt_by_image = ab.load_coco_gt_by_image([record], args.ground_truth)

    coarse_path = newest_cache(args.cache_dir, "eval_coarse_conf0.250_*.json")
    fine_path = newest_cache(args.cache_dir, "eval_fine_conf0.250_*.json")
    coarse = load_json(coarse_path).get(str(record.image_id), [])
    fine = load_json(fine_path).get(str(record.image_id), [])
    predictions = list(coarse) + list(fine)
    nodes, directed_nonself, visual_edges, graph_seconds = helper.build_graph(
        ab,
        record,
        predictions,
        gt_by_image.get(record.image_id, []),
        settings,
    )
    directed_with_self = ab.build_size_aware_edge_pairs(nodes, record, settings)

    segments = defaultdict(list)
    relation_counts = Counter()
    edge_payload = []
    duplicate_candidates = []
    for src, dst in visual_edges:
        first = nodes[src]
        second = nodes[dst]
        relations = relation_dict(ab, first, second, record, settings)
        for name, active in relations.items():
            if active:
                relation_counts[name] += 1
        relation_name = visual_relation(relations)
        segments[relation_name].append([first.center, second.center])
        overlap = float(ab.iou_xywh(first.bbox, second.bbox))
        if (
            overlap > 0.0
            and first.category_id == second.category_id
            and relations["cross_view"]
        ):
            duplicate_candidates.append((overlap, src, dst))
        edge_payload.append(
            {
                "source": src,
                "target": dst,
                "display_relation": relation_name,
                "relations": relations,
                "iou": overlap,
            }
        )

    highlighted = max(duplicate_candidates, default=None)
    image = Image.open(record.path).convert("RGB")
    class_colors = helper.CLASS_COLORS
    plt.style.use("dark_background")
    figure, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    figure.patch.set_facecolor("#11161b")
    figure.suptitle(
        f"Experiment 1 typed relation graph | image {record.image_id}: {record.file_name}",
        fontsize=18,
        fontweight="bold",
        color="#f4f7f9",
    )

    draw_base_image(
        axes[0],
        image,
        f"Pre-NMS candidates become nodes | coarse={len(coarse)}, fine={len(fine)}",
    )
    draw_nodes(axes[0], nodes, class_colors, label_nodes=True)

    draw_base_image(
        axes[1],
        image,
        (
            f"Sparse typed graph | N={len(nodes)}, "
            f"directed E={len(directed_with_self)} (self-loops included)"
        ),
        darken=True,
    )
    for relation_name in (
        "spatial",
        "cross_class_context",
        "same_cluster",
        "overlap_or_containment",
    ):
        if not segments[relation_name]:
            continue
        color, width, alpha = EDGE_STYLES[relation_name]
        axes[1].add_collection(
            LineCollection(
                segments[relation_name],
                colors=color,
                linewidths=width,
                alpha=alpha,
                zorder=3,
            )
        )
    draw_nodes(axes[1], nodes, class_colors, label_nodes=True)

    highlighted_payload = None
    if highlighted is not None:
        overlap, src, dst = highlighted
        first = nodes[src]
        second = nodes[dst]
        axes[1].plot(
            [first.center[0], second.center[0]],
            [first.center[1], second.center[1]],
            color="#fff200",
            linewidth=3.2,
            alpha=0.98,
            zorder=6,
        )
        for node in (first, second):
            x, y, width, height = node.bbox
            axes[1].add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=False,
                    edgecolor="#fff200",
                    linewidth=2.6,
                    zorder=8,
                )
            )
        axes[1].text(
            0.02,
            0.025,
            (
                f"Highlighted duplicate-like pair: n{src} ↔ n{dst} | "
                f"same class, cross-view, IoU={overlap:.2f}"
            ),
            transform=axes[1].transAxes,
            fontsize=10.5,
            color="#fff200",
            bbox={"facecolor": "#11161b", "edgecolor": "#fff200", "alpha": 0.88},
            zorder=10,
        )
        highlighted_payload = {
            "source": src,
            "target": dst,
            "iou": overlap,
            "category_id": first.category_id,
            "category_name": CLASS_NAMES[first.category_id],
            "source_view": ab.VIEW_TYPE_NAMES[first.view_type],
            "target_view": ab.VIEW_TYPE_NAMES[second.view_type],
            "source_score": first.max_score,
            "target_score": second.max_score,
        }

    present_classes = sorted({node.category_id for node in nodes})
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=class_colors[category_id],
            markeredgecolor="#11161b",
            label=f"{category_id}: {CLASS_NAMES[category_id]}",
            markersize=8,
        )
        for category_id in present_classes
    ]
    legend.extend(
        [
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#ffffff", label="coarse-view node", markersize=8),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffffff", label="fine-view node", markersize=8),
            Line2D([0], [0], color=EDGE_STYLES["spatial"][0], linewidth=2, label="spatial / kNN"),
            Line2D([0], [0], color=EDGE_STYLES["cross_class_context"][0], linewidth=2, label="cross-class context"),
            Line2D([0], [0], color=EDGE_STYLES["same_cluster"][0], linewidth=2, label="same DBSCAN cluster"),
            Line2D([0], [0], color=EDGE_STYLES["overlap_or_containment"][0], linewidth=2, label="overlap / containment"),
            Line2D([0], [0], color="#fff200", linewidth=3, label="highlighted duplicate-like pair"),
        ]
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=9.4,
    )
    figure.text(
        0.5,
        0.058,
        (
            "Node: 37-D confidence + normalized geometry + view type/geometry + class type   |   "
            "Edge: 24-D geometry (16) + relation mask (8)   |   self-loops omitted visually"
        ),
        ha="center",
        va="center",
        fontsize=10.2,
        color="#cbd5df",
    )

    output_path = args.output_dir / f"image_{record.image_id:04d}_experiment1_typed_graph.png"
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)

    node_payload = [
        {
            "node_id": node.node_id,
            "category_id": node.category_id,
            "category_name": CLASS_NAMES[node.category_id],
            "view_type": ab.VIEW_TYPE_NAMES[node.view_type],
            "score": node.max_score,
            "bbox_xywh": list(node.bbox),
            "dbscan_label": node.dbscan_label,
        }
        for node in nodes
    ]
    metadata = {
        "image": {
            "id": record.image_id,
            "file_name": record.file_name,
            "width": record.width,
            "height": record.height,
        },
        "candidate_pool": {
            "description": "coarse-640 + global-fine-256 before cross-view final NMS",
            "coarse": len(coarse),
            "fine": len(fine),
            "total": len(predictions),
            "coarse_cache": str(coarse_path),
            "fine_cache": str(fine_path),
        },
        "graph": {
            "variant_example": "03_gnn_dbscan_cluster",
            "nodes": len(nodes),
            "directed_edges_including_self_loops": len(directed_with_self),
            "directed_edges_excluding_self_loops": len(directed_nonself),
            "visual_undirected_edges": len(visual_edges),
            "relation_counts_nonexclusive_undirected": dict(relation_counts),
            "node_feature_dim": ab.SIZE_AWARE_NODE_DIM,
            "edge_feature_dim": ab.SIZE_AWARE_EDGE_DIM,
            "same_class_knn_limit": settings.graph_knn,
            "cross_class_knn_limit": settings.graph_cross_class_knn,
            "build_seconds": graph_seconds,
        },
        "highlighted_duplicate_like_pair": highlighted_payload,
        "nodes": node_payload,
        "edges": edge_payload,
        "output": str(output_path),
    }
    metadata_path = args.output_dir / f"image_{record.image_id:04d}_experiment1_typed_graph.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
