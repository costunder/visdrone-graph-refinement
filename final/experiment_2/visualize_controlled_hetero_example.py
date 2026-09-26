#!/usr/bin/env python3
"""Draw one real variant-08 heterogeneous graph without detector inference."""

import argparse
import json
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "final/experiment_2"
SCRIPT_DIR = ROOT / "final/scripts"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import controlled_hetero_graph as hetero  # noqa: E402
import run_gois_two_stage_gnn_ablation as ab  # noqa: E402
import run_sparse_ppr_ablation as runner  # noqa: E402
import sparse_ppr_sage as sparse_graph  # noqa: E402


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
CLASS_COLORS = {
    1: "#e63946",
    2: "#f3722c",
    3: "#f9c74f",
    4: "#277da1",
    5: "#43aa8b",
    6: "#577590",
    7: "#9b5de5",
    8: "#f15bb5",
    9: "#6a994e",
    10: "#ef476f",
}
VIEW_NAMES = {
    0: "unknown",
    1: "full",
    2: "coarse-640",
    3: "fine-256",
    4: "ROI",
}
CLASS_SHORT_NAMES = {
    **CLASS_NAMES,
    8: "awning tri.",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", type=int, default=218)
    parser.add_argument("--max-detections", type=int, default=14)
    parser.add_argument("--max-det-edges", type=int, default=28)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=(
            ROOT
            / "final/experiment_2/runs"
            / "table6_yolo11_10class_extra_ablation_e120_seed42"
            / "pipeline_ablation/08_controlled_hetero_ppr_gatv2_sage"
            / "run_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "final/experiment_2/reports/graph_visualization",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r") as handle:
        return json.load(handle)


def find_cache(cache_dir, pattern):
    matches = sorted(Path(cache_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cache matching {pattern} in {cache_dir}")
    return matches[-1]


def find_record(records, image_id):
    for record in records:
        if int(record.image_id) == int(image_id):
            return record
    raise ValueError(f"Image id {image_id} is absent from the evaluation split")


def densest_crop(nodes, width, height):
    crop_width = min(float(width), max(460.0, float(width) * 0.46))
    crop_height = min(float(height), max(330.0, float(height) * 0.56))
    centers = np.asarray([node.center for node in nodes], dtype=np.float32)
    best_count = -1
    best_center = (float(width) / 2.0, float(height) / 2.0)
    for cx, cy in centers:
        count = int(
            np.sum(
                (np.abs(centers[:, 0] - cx) <= crop_width / 2.0)
                & (np.abs(centers[:, 1] - cy) <= crop_height / 2.0)
            )
        )
        if count > best_count:
            best_count = count
            best_center = (float(cx), float(cy))
    x1 = min(max(0.0, best_center[0] - crop_width / 2.0), width - crop_width)
    y1 = min(max(0.0, best_center[1] - crop_height / 2.0), height - crop_height)
    return (x1, y1, x1 + crop_width, y1 + crop_height)


def choose_detection_indices(sample, crop, limit):
    nodes = sample["nodes"]
    x1, y1, x2, y2 = crop
    candidates = [
        index
        for index, node in enumerate(nodes)
        if x1 <= node.center[0] <= x2 and y1 <= node.center[1] <= y2
    ]
    ranked = sorted(candidates, key=lambda index: float(nodes[index].max_score), reverse=True)
    low_ranked = [index for index in ranked if float(nodes[index].max_score) < 0.25]
    source_tags = {
        index: sparse_graph._candidate_source_tag(ab, sample, nodes[index])
        for index in candidates
    }

    selected = []

    def add(index):
        if index not in selected and len(selected) < limit:
            selected.append(index)

    by_class = defaultdict(list)
    by_source = defaultdict(list)
    for index in ranked:
        by_class[int(nodes[index].category_id)].append(index)
        by_source[source_tags[index]].append(index)
    for category_id in sorted(by_class):
        add(by_class[category_id][0])
    for source in ("coarse", "fine", "full"):
        if by_source[source]:
            add(by_source[source][0])
    for index in ranked[: max(1, limit // 2)]:
        add(index)
    for index in low_ranked:
        add(index)
    for index in ranked:
        add(index)
    return selected


def selected_detection_edges(data, selected, max_edges):
    selected_set = set(selected)
    edge_index = data[hetero.DET_DET].edge_index.cpu().numpy()
    edge_attr = data[hetero.DET_DET].edge_attr.cpu().numpy()
    ppr_column = int(ab.SIZE_AWARE_EDGE_DIM) + sparse_graph.PPR_EXTRA_OFFSET
    pair_weight = {}
    for offset, (source, target) in enumerate(edge_index.T):
        source = int(source)
        target = int(target)
        if source == target or source not in selected_set or target not in selected_set:
            continue
        ppr_weight = float(edge_attr[offset, ppr_column])
        if ppr_weight <= 0.0:
            continue
        pair = tuple(sorted((source, target)))
        pair_weight[pair] = max(
            pair_weight.get(pair, 0.0),
            ppr_weight,
        )
    return sorted(pair_weight.items(), key=lambda item: item[1], reverse=True)[:max_edges]


def view_metadata(sample):
    record = sample["record"]
    metadata = []
    id_to_local = {}
    detection_to_view = []
    for node in sample["nodes"]:
        source = sparse_graph._candidate_source_tag(ab, sample, node)
        item = sparse_graph._node_view_metadata(ab, sample, node, record, source)
        view_type, view_id, view_bbox = item
        if view_id not in id_to_local:
            id_to_local[view_id] = len(metadata)
            metadata.append(
                {
                    "type": int(view_type),
                    "id": str(view_id),
                    "bbox": tuple(float(value) for value in view_bbox),
                }
            )
        detection_to_view.append(id_to_local[view_id])
    return metadata, detection_to_view


def draw_image_graph(ax, image, sample, selected, edges, crop, display_ids):
    nodes = sample["nodes"]
    segments = [
        [nodes[source].center, nodes[target].center]
        for (source, target), _ in edges
    ]
    if segments:
        weights = np.asarray([weight for _, weight in edges], dtype=np.float32)
        scale = weights / max(1e-6, float(weights.max()))
        ax.add_collection(
            LineCollection(
                segments,
                colors=[(0.0, 0.72, 0.95, 0.22 + 0.58 * value) for value in scale],
                linewidths=[0.7 + 2.1 * value for value in scale],
                zorder=2,
            )
        )
    ax.imshow(image)
    for index in selected:
        node = nodes[index]
        x, y, width, height = [float(value) for value in node.bbox]
        cx, cy = node.center
        color = CLASS_COLORS.get(int(node.category_id), "#ffffff")
        low_confidence = float(node.max_score) < 0.25
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=1.7,
                linestyle="--" if low_confidence else "-",
                alpha=0.95,
                zorder=3,
            )
        )
        ax.scatter(
            [cx],
            [cy],
            s=88,
            c=[color],
            edgecolors="#111827",
            linewidths=1.0,
            zorder=4,
        )
        ax.annotate(
            display_ids[index],
            (cx, cy),
            xytext=(5, -6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "fc": "#111827", "ec": "none", "alpha": 0.86},
            zorder=5,
        )
    x1, y1, x2, y2 = crop
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Image-space detection graph\nboxes = nodes, cyan = strongest spatial/PPR edges",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )


def add_relation(ax, start, end, color, rad=0.0, width=1.1, alpha=0.5, arrows="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrows,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=8,
        linewidth=width,
        color=color,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(patch)


def draw_typed_graph(
    ax,
    sample,
    data,
    selected,
    edges,
    display_ids,
    view_info,
    detection_to_view,
):
    nodes = sample["nodes"]
    selected_classes = sorted({int(nodes[index].category_id) for index in selected})
    selected_views = sorted({int(detection_to_view[index]) for index in selected})
    class_y = {
        category_id: value
        for category_id, value in zip(
            selected_classes,
            np.linspace(0.87, 0.13, len(selected_classes)),
        )
    }
    view_y = {
        view_id: value
        for view_id, value in zip(
            selected_views,
            np.linspace(0.72, 0.28, len(selected_views)),
        )
    }
    detections_by_class = defaultdict(list)
    for index in selected:
        detections_by_class[int(nodes[index].category_id)].append(index)
    detection_position = {}
    for category_id, members in detections_by_class.items():
        members = sorted(members, key=lambda index: float(nodes[index].center[0]))
        xs = np.linspace(0.38, 0.69, len(members)) if len(members) > 1 else [0.535]
        offsets = np.linspace(-0.035, 0.035, len(members)) if len(members) > 1 else [0.0]
        for index, x_value, offset in zip(members, xs, offsets):
            detection_position[index] = (float(x_value), float(class_y[category_id] + offset))
    class_position = {category_id: (0.11, y_value) for category_id, y_value in class_y.items()}
    view_position = {view_id: (0.91, y_value) for view_id, y_value in view_y.items()}

    for edge_rank, ((source, target), weight) in enumerate(edges):
        add_relation(
            ax,
            detection_position[source],
            detection_position[target],
            "#6b7280",
            rad=0.10 if edge_rank % 2 == 0 else -0.10,
            width=0.7 + 2.0 * weight,
            alpha=0.30 + min(0.45, weight),
            arrows="->",
        )
    for index in selected:
        category_id = int(nodes[index].category_id)
        add_relation(
            ax,
            class_position[category_id],
            detection_position[index],
            "#f59e0b",
            width=1.15,
            alpha=0.52,
            arrows="<->",
        )
        add_relation(
            ax,
            detection_position[index],
            view_position[int(detection_to_view[index])],
            "#16a34a",
            width=1.05,
            alpha=0.45,
            arrows="<->",
        )

    global_to_category = {
        int(global_id): int(global_id) + 1
        for global_id in data["class"].global_id.tolist()
    }
    local_to_global = {
        local_id: int(global_id)
        for local_id, global_id in enumerate(data["class"].global_id.tolist())
    }
    class_relations = []
    for offset, (source, target) in enumerate(
        data[hetero.CLASS_CLASS].edge_index.t().tolist()
    ):
        source_global = local_to_global[int(source)]
        target_global = local_to_global[int(target)]
        source_category = global_to_category[source_global]
        target_category = global_to_category[target_global]
        if source_category == target_category:
            continue
        if source_category not in class_position or target_category not in class_position:
            continue
        strength = float(data[hetero.CLASS_CLASS].edge_attr[offset, 0])
        class_relations.append((strength, source_category, target_category))
    for relation_rank, (_, source_category, target_category) in enumerate(
        sorted(class_relations, reverse=True)[:8]
    ):
        add_relation(
            ax,
            class_position[source_category],
            class_position[target_category],
            "#c026d3",
            rad=0.17 if relation_rank % 2 == 0 else -0.17,
            width=1.2,
            alpha=0.60,
            arrows="->",
        )

    for category_id, position in class_position.items():
        ax.scatter(
            [position[0]],
            [position[1]],
            s=620,
            marker="h",
            c=["#fff3bf"],
            edgecolors="#b45309",
            linewidths=1.8,
            zorder=4,
        )
        ax.text(
            position[0],
            position[1],
            f"C{category_id}\n{CLASS_SHORT_NAMES[category_id]}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#78350f",
            zorder=5,
        )
    for view_id, position in view_position.items():
        view_type = int(view_info[view_id]["type"])
        ax.scatter(
            [position[0]],
            [position[1]],
            s=760,
            marker="s",
            c=["#dcfce7"],
            edgecolors="#15803d",
            linewidths=1.8,
            zorder=4,
        )
        ax.text(
            position[0],
            position[1],
            f"V{view_id}\n{VIEW_NAMES.get(view_type, 'unknown')}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#14532d",
            zorder=5,
        )
    for index, position in detection_position.items():
        category_id = int(nodes[index].category_id)
        color = CLASS_COLORS.get(category_id, "#64748b")
        ax.scatter(
            [position[0]],
            [position[1]],
            s=360,
            marker="o",
            c=[color],
            edgecolors="#111827",
            linewidths=1.2,
            zorder=4,
        )
        ax.text(
            position[0],
            position[1],
            f"{display_ids[index]}\n{float(nodes[index].max_score):.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="white",
            zorder=5,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.05, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8fafc")
    ax.set_title(
        "Same nodes as a typed heterogeneous graph\nclass and view context return messages to detections",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )


def main():
    cli = parse_args()
    payload = load_json(cli.run_config)
    args = Namespace(**payload["args"])
    ab.configure_class_space(str(args.class_space))

    records = ab.build_image_records(Path(args.eval_images), Path(args.ground_truth_path))
    record = find_record(records, cli.image_id)
    gt_by_image = ab.load_coco_gt_by_image([record], Path(args.ground_truth_path))
    cache_dir = Path(args.common_cache_dir)
    coarse_cache = load_json(find_cache(cache_dir, "eval_coarse_conf0.050_s640_*.json"))
    fine_cache = load_json(find_cache(cache_dir, "eval_fine_conf0.050_s256_*.json"))
    image_key = str(record.image_id)
    candidates = {
        image_key: [
            dict(prediction, _candidate_source="coarse")
            for prediction in coarse_cache.get(image_key, [])
        ]
        + [
            dict(prediction, _candidate_source="fine")
            for prediction in fine_cache.get(image_key, [])
        ]
    }
    sample = runner.build_samples(
        [record],
        gt_by_image,
        candidates,
        args,
        "low_conf_multiscale",
    )[0]
    data = hetero.build_controlled_hetero_graph_data(
        ab,
        sample,
        args,
        include_ppr=True,
    )
    crop = densest_crop(sample["nodes"], record.width, record.height)
    selected = choose_detection_indices(sample, crop, cli.max_detections)
    display_ids = {index: f"d{offset}" for offset, index in enumerate(selected)}
    edges = selected_detection_edges(data, selected, cli.max_det_edges)
    view_info, detection_to_view = view_metadata(sample)

    image = Image.open(record.path).convert("RGB")
    figure, axes = plt.subplots(1, 2, figsize=(19, 9.8))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        f"Variant 08 graph example | image {record.image_id}: {record.file_name}",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    draw_image_graph(
        axes[0],
        image,
        sample,
        selected,
        edges,
        crop,
        display_ids,
    )
    draw_typed_graph(
        axes[1],
        sample,
        data,
        selected,
        edges,
        display_ids,
        view_info,
        detection_to_view,
    )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#277da1", markeredgecolor="#111827", label="detection node"),
        Line2D([0], [0], marker="h", color="none", markerfacecolor="#fff3bf", markeredgecolor="#b45309", label="class node"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#dcfce7", markeredgecolor="#15803d", label="view node"),
        Line2D([0], [0], color="#6b7280", linewidth=2, label="detection → detection (spatial/PPR)"),
        Line2D([0], [0], color="#f59e0b", linewidth=2, label="detection ↔ class"),
        Line2D([0], [0], color="#16a34a", linewidth=2, label="detection ↔ view"),
        Line2D([0], [0], color="#c026d3", linewidth=2, label="class → class co-occurrence"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.045),
    )
    graph_stats = sample["_exp2_controlled_hetero_graph_stats"]
    node_counts = graph_stats["node_counts"]
    edge_counts = graph_stats["edge_counts"]
    figure.text(
        0.5,
        0.018,
        (
            f"Full graph: {node_counts['detection']} detections + "
            f"{node_counts['class']} active classes + {node_counts['view']} views | "
            f"E(det→det)={edge_counts['detection_spatial_ppr_detection']}, "
            f"E(det↔class)={2 * edge_counts['detection_predicted_as_class']}, "
            f"E(class→class)={edge_counts['class_cooccurs_class']}, "
            f"E(det↔view)={2 * edge_counts['detection_observed_in_view']} | "
            f"displayed subset: {len(selected)} detections, {len(edges)} PPR edges"
        ),
        ha="center",
        fontsize=10,
        color="#334155",
    )
    figure.tight_layout(rect=(0.02, 0.10, 0.98, 0.93))

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = cli.output_dir / f"image_{record.image_id:04d}_controlled_hetero_graph_example.png"
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    selected_payload = []
    for index in selected:
        node = sample["nodes"][index]
        selected_payload.append(
            {
                "display_id": display_ids[index],
                "full_detection_index": index,
                "category_id": int(node.category_id),
                "category_name": CLASS_NAMES[int(node.category_id)],
                "score": float(node.max_score),
                "center": [float(value) for value in node.center],
                "view_id": int(detection_to_view[index]),
            }
        )
    metadata = {
        "image": {
            "id": int(record.image_id),
            "file_name": record.file_name,
            "width": int(record.width),
            "height": int(record.height),
        },
        "full_graph": graph_stats,
        "display": {
            "crop_xyxy": [float(value) for value in crop],
            "detections": selected_payload,
            "ppr_edges": [
                {
                    "source": display_ids[source],
                    "target": display_ids[target],
                    "weight": float(weight),
                }
                for (source, target), weight in edges
            ],
        },
        "output": str(output_path.resolve()),
        "detector_inference_performed": False,
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
