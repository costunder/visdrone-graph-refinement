"""Sparse, inductive graph refinement for experiment 2.

The graph builder deliberately avoids allocating an N x N relation matrix. It
uses a spatial hash for local candidates, keeps a bounded number of same-class
and cross-class neighbors, and optionally adds truncated personalized PageRank
neighbors. The models retain the existing geometric edge features instead of
reducing the experiment to a vanilla feature-only GraphSAGE baseline.
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


NODE_SOURCE_DIM = 3
EDGE_EXTRA_DIM = 3
PPR_EXTRA_OFFSET = 0
GRAPH_SCHEMA_VERSION = 6


def node_dim(ab):
    return int(ab.SIZE_AWARE_NODE_DIM) + NODE_SOURCE_DIM


def edge_dim(ab):
    return int(ab.SIZE_AWARE_EDGE_DIM) + EDGE_EXTRA_DIM


def _candidate_source_tag(ab, sample, node):
    if int(node.source) == int(ab.SOURCE_FULL):
        return "full"
    predictions = sample.get("predictions", [])
    index = int(node.det_index)
    if 0 <= index < len(predictions):
        return str(predictions[index].get("_candidate_source", "slice"))
    return "slice"


def _source_one_hot(source):
    if source == "coarse":
        return [1.0, 0.0, 0.0]
    if source == "fine":
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _graph_radius_ratio(record, args):
    ratio = float(getattr(args, "graph_radius_ratio", 0.0) or 0.0)
    if ratio > 0.0:
        return ratio
    diagonal = math.hypot(float(record.width), float(record.height))
    pixel_radius = float(getattr(args, "graph_radius", 0.0) or 0.0)
    if pixel_radius > 0.0 and diagonal > 0.0:
        return pixel_radius / diagonal
    return 0.12


def _node_prediction(ab, sample, node):
    predictions = (
        sample.get("full_predictions", [])
        if int(node.source) == int(ab.SOURCE_FULL)
        else sample.get("predictions", [])
    )
    index = int(node.det_index)
    return predictions[index] if 0 <= index < len(predictions) else {}


def _node_view_metadata(ab, sample, node, record, source_tag):
    prediction = _node_prediction(ab, sample, node)
    view_unknown = int(getattr(ab, "VIEW_UNKNOWN", 0))
    view_full = int(getattr(ab, "VIEW_FULL", 1))
    view_coarse = int(getattr(ab, "VIEW_COARSE", 2))
    view_fine = int(getattr(ab, "VIEW_FINE", 3))
    view_roi = int(getattr(ab, "VIEW_ROI", 4))
    valid_view_types = {
        view_unknown,
        view_full,
        view_coarse,
        view_fine,
        view_roi,
    }
    try:
        view_type = int(prediction.get("_view_type", view_unknown))
    except (TypeError, ValueError):
        view_type = view_unknown
    if view_type not in valid_view_types:
        view_type = view_unknown
    if view_type == view_unknown:
        view_type = {
            "full": view_full,
            "coarse": view_coarse,
            "fine": view_fine,
            "roi": view_roi,
        }.get(str(source_tag).lower(), view_unknown)

    raw_bbox = prediction.get("_view_bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raw_bbox = [0.0, 0.0, float(record.width), float(record.height)]
    view_bbox = tuple(ab.clip_xywh(raw_bbox, record.width, record.height))
    view_id = str(prediction.get("_view_id", ""))
    if not view_id:
        view_id = f"legacy:{source_tag}:{record.image_id}"
    return view_type, view_id, view_bbox


def _view_geometry(ab, node, record, view_bbox):
    vx, vy, vw, vh = ab.clip_xywh(view_bbox, record.width, record.height)
    width = max(1.0, float(record.width))
    height = max(1.0, float(record.height))
    rel_x = (float(node.center[0]) - vx) / max(1.0, float(vw))
    rel_y = (float(node.center[1]) - vy) / max(1.0, float(vh))
    return [
        vx / width,
        vy / height,
        vw / width,
        vh / height,
        max(0.0, min(1.0, rel_x)),
        max(0.0, min(1.0, rel_y)),
    ]


def _node_feature(ab, node, record, neighbor_density, source_tag, view_metadata):
    x, y, width, height = node.bbox
    center_x, center_y = node.center
    image_width = max(1.0, float(record.width))
    image_height = max(1.0, float(record.height))
    image_area = max(1.0, image_width * image_height)
    area = max(1.0, float(node.union_area))

    view_type, _, view_bbox = view_metadata
    view_one_hot = [0.0] * int(ab.NUM_VIEW_TYPES)
    if 0 <= int(view_type) < len(view_one_hot):
        view_one_hot[int(view_type)] = 1.0
    class_one_hot = [0.0] * int(ab.NUM_CLASSES)
    class_index = int(ab.VISDRONE_TO_CLASS_INDEX.get(int(node.category_id), 0))
    class_one_hot[class_index] = 1.0

    feature = [
        float(node.max_score),
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
        math.sqrt(area / image_area),
        math.log((width + 1.0) / (height + 1.0)),
        1.0 if area < ab.SMALL_MEDIUM_AREA_THR else 0.0,
        1.0 if area >= ab.SMALL_MEDIUM_AREA_THR else 0.0,
        1.0 if int(node.source) == int(ab.SOURCE_FULL) else 0.0,
        1.0 if int(node.source) == int(ab.SOURCE_SLICE) else 0.0,
        1.0 if int(node.dbscan_label) >= 0 else 0.0,
        min(1.0, float(neighbor_density) / 16.0),
        math.log1p(area) / math.log1p(image_area),
        x / image_width,
        y / image_height,
        *view_one_hot,
        *_view_geometry(ab, node, record, view_bbox),
        *class_one_hot,
        *_source_one_hot(source_tag),
    ]
    if len(feature) != node_dim(ab):
        raise RuntimeError(f"node feature dim {len(feature)} != {node_dim(ab)}")
    return feature


def _grid_key(center, cell_size):
    return int(math.floor(center[0] / cell_size)), int(math.floor(center[1] / cell_size))


def _pair_affinity(ab, src, dst, radius, same_source):
    dx = float(src.center[0]) - float(dst.center[0])
    dy = float(src.center[1]) - float(dst.center[1])
    distance = math.sqrt(dx * dx + dy * dy)
    iou = float(ab.iou_xywh(src.bbox, dst.bbox))
    contain = max(
        float(ab.containment(src.bbox, dst.bbox)),
        float(ab.containment(dst.bbox, src.bbox)),
    )
    if distance > radius and iou <= 0.0 and contain <= 0.05:
        return None
    distance_score = max(0.0, 1.0 - distance / max(1.0, radius))
    same_class = int(src.category_id) == int(dst.category_id)
    score = (
        distance_score
        + 1.50 * iou
        + 0.75 * contain
        + (0.35 if same_class else 0.0)
        + (0.10 if same_source else 0.0)
    )
    return max(score, 1e-6)


def _sparse_node_arrays(nodes, source_tags):
    return {
        "centers": np.asarray([node.center for node in nodes], dtype=np.float64),
        "boxes": np.asarray([node.bbox for node in nodes], dtype=np.float64),
        "categories": np.asarray([int(node.category_id) for node in nodes], dtype=np.int64),
        "sources": np.asarray(source_tags, dtype=object),
    }


def _pair_affinities(nodes, src_index, candidate_indices, radius, source_tags, arrays=None):
    """Vectorized equivalent of _pair_affinity for one source node."""
    if not candidate_indices:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)

    arrays = arrays or _sparse_node_arrays(nodes, source_tags)
    candidates = np.fromiter(candidate_indices, dtype=np.int64)
    centers = arrays["centers"]
    boxes = arrays["boxes"]
    categories = arrays["categories"]
    sources = arrays["sources"]
    delta = centers[candidates] - centers[src_index]
    distance = np.sqrt(np.sum(delta * delta, axis=1))

    src_box = boxes[src_index]
    dst_boxes = boxes[candidates]
    src_x2 = src_box[0] + src_box[2]
    src_y2 = src_box[1] + src_box[3]
    dst_x2 = dst_boxes[:, 0] + dst_boxes[:, 2]
    dst_y2 = dst_boxes[:, 1] + dst_boxes[:, 3]
    inter_width = np.maximum(
        0.0,
        np.minimum(src_x2, dst_x2) - np.maximum(src_box[0], dst_boxes[:, 0]),
    )
    inter_height = np.maximum(
        0.0,
        np.minimum(src_y2, dst_y2) - np.maximum(src_box[1], dst_boxes[:, 1]),
    )
    intersection = inter_width * inter_height
    src_area = max(0.0, float(src_box[2])) * max(0.0, float(src_box[3]))
    dst_area = np.maximum(0.0, dst_boxes[:, 2]) * np.maximum(0.0, dst_boxes[:, 3])
    union = src_area + dst_area - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)
    min_area = np.minimum(src_area, dst_area)
    contain = np.divide(
        intersection,
        min_area,
        out=np.zeros_like(intersection),
        where=min_area > 0.0,
    )

    same_class = categories[candidates] == categories[src_index]
    same_source = sources[candidates] == sources[src_index]
    valid = (distance <= radius) | (iou > 0.0) | (contain > 0.05)
    distance_score = np.maximum(0.0, 1.0 - distance / max(1.0, radius))
    affinity = (
        distance_score
        + 1.50 * iou
        + 0.75 * contain
        + 0.35 * same_class.astype(np.float64)
        + 0.10 * same_source.astype(np.float64)
    )
    return candidates[valid], np.maximum(affinity[valid], 1e-6)


def build_sparse_local_edges(ab, sample, args):
    nodes = sample.get("nodes", [])
    if not nodes:
        return {}, []

    radius = max(1.0, float(getattr(args, "graph_radius", 256.0)))
    cell_size = float(getattr(args, "exp2_spatial_cell_size", 0.0) or radius)
    cell_size = max(1.0, cell_size)
    same_class_knn = max(0, int(getattr(args, "exp2_local_knn", 12)))
    cross_class_knn = max(0, int(getattr(args, "exp2_cross_class_knn", 4)))
    cell_hops = max(1, int(math.ceil(radius / cell_size)))
    source_tags = [_candidate_source_tag(ab, sample, node) for node in nodes]
    node_arrays = _sparse_node_arrays(nodes, source_tags)

    grid = defaultdict(list)
    large_indices = []
    for index, node in enumerate(nodes):
        grid[_grid_key(node.center, cell_size)].append(index)
        if float(node.bbox[2]) > radius or float(node.bbox[3]) > radius:
            large_indices.append(index)

    directed = {}
    for src_index, src in enumerate(nodes):
        gx, gy = _grid_key(src.center, cell_size)
        candidate_indices = set(large_indices)
        for oy in range(-cell_hops, cell_hops + 1):
            for ox in range(-cell_hops, cell_hops + 1):
                candidate_indices.update(grid.get((gx + ox, gy + oy), ()))
        candidate_indices.discard(src_index)

        same_class = []
        cross_class = []
        valid_indices, affinities = _pair_affinities(
            nodes,
            src_index,
            candidate_indices,
            radius,
            source_tags,
            node_arrays,
        )
        for dst_index, affinity in zip(valid_indices.tolist(), affinities.tolist()):
            dst = nodes[dst_index]
            item = (float(affinity), int(dst_index))
            if int(src.category_id) == int(dst.category_id):
                same_class.append(item)
            else:
                cross_class.append(item)

        same_class.sort(reverse=True)
        cross_class.sort(reverse=True)
        selected = same_class[:same_class_knn] + cross_class[:cross_class_knn]
        for affinity, dst_index in selected:
            directed[(src_index, dst_index)] = max(
                float(affinity),
                directed.get((src_index, dst_index), 0.0),
            )
            directed[(dst_index, src_index)] = max(
                float(affinity),
                directed.get((dst_index, src_index), 0.0),
            )
    return directed, source_tags


def _normalized_adjacency(node_count, weighted_edges):
    rows = [defaultdict(float) for _ in range(node_count)]
    for (src, dst), weight in weighted_edges.items():
        if src == dst:
            continue
        rows[int(src)][int(dst)] = max(rows[int(src)][int(dst)], float(weight))
    normalized = []
    for index, row in enumerate(rows):
        if not row:
            normalized.append([(index, 1.0)])
            continue
        total = sum(row.values())
        normalized.append([(dst, weight / total) for dst, weight in row.items()])
    return normalized


def _truncated_topk_ppr_scalar(node_count, weighted_edges, args):
    top_k = max(0, int(getattr(args, "exp2_ppr_knn", 8)))
    if node_count == 0 or top_k == 0:
        return {}

    alpha = min(0.95, max(0.01, float(getattr(args, "exp2_ppr_alpha", 0.15))))
    steps = max(1, int(getattr(args, "exp2_ppr_steps", 8)))
    frontier_cap = max(top_k, int(getattr(args, "exp2_ppr_frontier", 64)))
    adjacency = _normalized_adjacency(node_count, weighted_edges)
    ppr_edges = {}

    for target in range(node_count):
        walk = {target: 1.0}
        scores = defaultdict(float)
        decay = 1.0
        for _ in range(steps + 1):
            for node_index, mass in walk.items():
                scores[node_index] += alpha * decay * mass
            next_walk = defaultdict(float)
            for node_index, mass in walk.items():
                for neighbor, probability in adjacency[node_index]:
                    next_walk[neighbor] += mass * probability
            if len(next_walk) > frontier_cap:
                next_walk = defaultdict(
                    float,
                    sorted(next_walk.items(), key=lambda item: item[1], reverse=True)[:frontier_cap],
                )
            walk = next_walk
            decay *= 1.0 - alpha

        ranked = sorted(
            ((score, neighbor) for neighbor, score in scores.items() if neighbor != target),
            reverse=True,
        )[:top_k]
        for score, neighbor in ranked:
            # edge_index is source -> destination; pi_target(neighbor) is the
            # structural prior for a message from neighbor into target.
            ppr_edges[(int(neighbor), int(target))] = float(score)
    return ppr_edges


def truncated_topk_ppr(node_count, weighted_edges, args):
    top_k = max(0, int(getattr(args, "exp2_ppr_knn", 8)))
    if node_count == 0 or top_k == 0:
        return {}

    alpha = min(0.95, max(0.01, float(getattr(args, "exp2_ppr_alpha", 0.15))))
    steps = max(1, int(getattr(args, "exp2_ppr_steps", 8)))
    frontier_cap = max(top_k, int(getattr(args, "exp2_ppr_frontier", 64)))
    adjacency = _normalized_adjacency(node_count, weighted_edges)

    rows = []
    columns = []
    probabilities = []
    for src_index, neighbors in enumerate(adjacency):
        for dst_index, probability in neighbors:
            rows.append(src_index)
            columns.append(int(dst_index))
            probabilities.append(float(probability))
    transition = sparse.csr_matrix(
        (probabilities, (rows, columns)),
        shape=(node_count, node_count),
        dtype=np.float64,
    )

    # Keep memory O(batch_size * N), while sharing the same sparse transition
    # multiplication across several personalized walks.
    batch_size = min(64, node_count)
    ppr_edges = {}
    for batch_start in range(0, node_count, batch_size):
        targets = np.arange(
            batch_start,
            min(node_count, batch_start + batch_size),
            dtype=np.int64,
        )
        batch_count = int(targets.size)
        walk = np.zeros((batch_count, node_count), dtype=np.float64)
        walk[np.arange(batch_count), targets] = 1.0
        scores = np.zeros_like(walk)
        decay = 1.0

        for _ in range(steps + 1):
            scores += alpha * decay * walk
            next_walk = transition.transpose().dot(walk.transpose()).transpose()
            if node_count > frontier_cap:
                pruned = np.zeros_like(next_walk)
                for row_index in range(batch_count):
                    nonzero = np.flatnonzero(next_walk[row_index])
                    if nonzero.size <= frontier_cap:
                        pruned[row_index, nonzero] = next_walk[row_index, nonzero]
                        continue
                    values = next_walk[row_index, nonzero]
                    keep = np.argpartition(values, -frontier_cap)[-frontier_cap:]
                    kept_nodes = nonzero[keep]
                    pruned[row_index, kept_nodes] = next_walk[row_index, kept_nodes]
                next_walk = pruned
            walk = next_walk
            decay *= 1.0 - alpha

        for row_index, target in enumerate(targets.tolist()):
            candidates = np.flatnonzero(scores[row_index])
            candidates = candidates[candidates != target]
            if candidates.size == 0:
                continue
            values = scores[row_index, candidates]
            order = np.lexsort((-candidates, -values))[:top_k]
            for candidate_index in order:
                neighbor = int(candidates[candidate_index])
                ppr_edges[(neighbor, int(target))] = float(values[candidate_index])
    return ppr_edges


def _edge_feature_matrix(
    ab,
    nodes,
    edge_pairs,
    source_tags,
    view_metadata,
    ppr_scores,
    record,
    args,
):
    pairs = np.asarray(edge_pairs, dtype=np.int64)
    src_indices = pairs[:, 0]
    dst_indices = pairs[:, 1]
    centers = np.asarray([node.center for node in nodes], dtype=np.float64)
    boxes = np.asarray([node.bbox for node in nodes], dtype=np.float64)
    categories = np.asarray([int(node.category_id) for node in nodes], dtype=np.int64)
    cluster_ids = np.asarray([int(node.dbscan_label) for node in nodes], dtype=np.int64)
    sources = np.asarray([int(node.source) for node in nodes], dtype=np.int64)
    scores = np.asarray([float(node.max_score) for node in nodes], dtype=np.float64)
    areas = np.asarray([float(node.union_area) for node in nodes], dtype=np.float64)
    node_ids = np.asarray([int(node.node_id) for node in nodes], dtype=np.int64)
    view_ids = np.asarray([str(metadata[1]) for metadata in view_metadata], dtype=object)
    source_tags_array = np.asarray(source_tags, dtype=object)

    src_boxes = boxes[src_indices]
    dst_boxes = boxes[dst_indices]
    src_centers = centers[src_indices]
    dst_centers = centers[dst_indices]
    diagonal = max(1.0, math.hypot(float(record.width), float(record.height)))
    distance = np.sqrt(np.sum((src_centers - dst_centers) ** 2, axis=1)) / diagonal
    radius = _graph_radius_ratio(record, args)
    dist_score = np.maximum(0.0, 1.0 - distance / max(1e-6, radius))

    src_width = np.maximum(1.0, src_boxes[:, 2])
    src_height = np.maximum(1.0, src_boxes[:, 3])
    dst_width = np.maximum(1.0, dst_boxes[:, 2])
    dst_height = np.maximum(1.0, dst_boxes[:, 3])
    dx_norm = (dst_centers[:, 0] - src_centers[:, 0]) / np.sqrt(src_width * dst_width)
    dy_norm = (dst_centers[:, 1] - src_centers[:, 1]) / np.sqrt(src_height * dst_height)
    log_area_ratio = np.log((areas[src_indices] + 1.0) / (areas[dst_indices] + 1.0))
    log_width_ratio = np.log(src_width / dst_width)
    log_height_ratio = np.log(src_height / dst_height)
    scale_similarity = np.maximum(0.0, 1.0 - np.abs(log_area_ratio) / 3.0)

    src_x2 = src_boxes[:, 0] + src_boxes[:, 2]
    src_y2 = src_boxes[:, 1] + src_boxes[:, 3]
    dst_x2 = dst_boxes[:, 0] + dst_boxes[:, 2]
    dst_y2 = dst_boxes[:, 1] + dst_boxes[:, 3]
    intersection = (
        np.maximum(0.0, np.minimum(src_x2, dst_x2) - np.maximum(src_boxes[:, 0], dst_boxes[:, 0]))
        * np.maximum(0.0, np.minimum(src_y2, dst_y2) - np.maximum(src_boxes[:, 1], dst_boxes[:, 1]))
    )
    src_area = np.maximum(0.0, src_boxes[:, 2]) * np.maximum(0.0, src_boxes[:, 3])
    dst_area = np.maximum(0.0, dst_boxes[:, 2]) * np.maximum(0.0, dst_boxes[:, 3])
    union = src_area + dst_area - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)
    contain_src_dst = np.divide(
        intersection,
        src_area,
        out=np.zeros_like(intersection),
        where=src_area > 0.0,
    )
    contain_dst_src = np.divide(
        intersection,
        dst_area,
        out=np.zeros_like(intersection),
        where=dst_area > 0.0,
    )

    same_class = categories[src_indices] == categories[dst_indices]
    same_cluster = (cluster_ids[src_indices] >= 0) & (
        cluster_ids[src_indices] == cluster_ids[dst_indices]
    )
    self_edge = node_ids[src_indices] == node_ids[dst_indices]
    src_views = view_ids[src_indices]
    dst_views = view_ids[dst_indices]
    src_has_view = src_views != ""
    dst_has_view = dst_views != ""
    same_view = src_has_view & (src_views == dst_views)
    cross_view = src_has_view & dst_has_view & (src_views != dst_views)
    max_containment = np.maximum(contain_src_dst, contain_dst_src)

    features = np.column_stack(
        [
            same_class,
            dist_score,
            scale_similarity,
            overlap,
            contain_src_dst,
            contain_dst_src,
            same_cluster,
            sources[src_indices] != sources[dst_indices],
            (sources[src_indices] == int(ab.SOURCE_SLICE))
            & (sources[dst_indices] == int(ab.SOURCE_SLICE)),
            (sources[src_indices] == int(ab.SOURCE_FULL))
            & (sources[dst_indices] == int(ab.SOURCE_FULL)),
            0.5 * (scores[src_indices] + scores[dst_indices]),
            np.tanh(dx_norm),
            np.tanh(dy_norm),
            np.clip(log_area_ratio, -3.0, 3.0) / 3.0,
            np.clip(log_width_ratio, -3.0, 3.0) / 3.0,
            np.clip(log_height_ratio, -3.0, 3.0) / 3.0,
            self_edge,
            (~self_edge) & (distance <= radius),
            (~self_edge) & (overlap > 0.0),
            (~self_edge) & (max_containment > 0.10),
            (~self_edge) & same_class & same_cluster,
            same_view,
            (~self_edge) & cross_view,
            (~self_edge) & (~same_class),
            np.fromiter(
                (float(ppr_scores.get((int(src), int(dst)), 0.0)) for src, dst in edge_pairs),
                dtype=np.float64,
            ),
            source_tags_array[src_indices] == source_tags_array[dst_indices],
            ~same_class,
        ]
    )
    if features.shape[1] != edge_dim(ab):
        raise RuntimeError(f"edge feature dim {features.shape[1]} != {edge_dim(ab)}")
    return features.astype(np.float32, copy=False)


def build_sparse_graph_tensors(ab, sample, args, include_ppr):
    nodes = sample.get("nodes", [])
    if not nodes:
        return (
            torch.empty((0, node_dim(ab)), dtype=torch.float32),
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, edge_dim(ab)), dtype=torch.float32),
            torch.empty((0, 3), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    record = sample["record"]
    local_edges, source_tags = build_sparse_local_edges(ab, sample, args)
    view_metadata = [
        _node_view_metadata(ab, sample, node, record, source_tags[index])
        for index, node in enumerate(nodes)
    ]
    ppr_scores = truncated_topk_ppr(len(nodes), local_edges, args) if include_ppr else {}
    edge_pairs = set(local_edges)
    edge_pairs.update(ppr_scores)
    edge_pairs.update((index, index) for index in range(len(nodes)))
    edge_pairs = sorted(edge_pairs)

    density = defaultdict(int)
    for src, dst in edge_pairs:
        if src != dst:
            density[src] += 1
            density[dst] += 1

    node_features = []
    for index, node in enumerate(nodes):
        node_features.append(
            _node_feature(
                ab,
                node,
                record,
                density[index],
                source_tags[index],
                view_metadata[index],
            )
        )

    edge_features = _edge_feature_matrix(
        ab,
        nodes,
        edge_pairs,
        source_tags,
        view_metadata,
        ppr_scores,
        record,
        args,
    )

    targets = [
        [float(node.target_obj), float(node.target_small), float(node.target_large)]
        for node in nodes
    ]
    weights = [float(node.weight) for node in nodes]
    sample["_exp2_sparse_graph_stats"] = {
        "nodes": len(nodes),
        "local_directed_edges": len(local_edges),
        "ppr_directed_edges": len(ppr_scores),
        "total_edges_with_self_loops": len(edge_pairs),
        "cross_class_edges": sum(
            int(nodes[src].category_id) != int(nodes[dst].category_id)
            for src, dst in edge_pairs
            if src != dst
        ),
        "dense_pair_slots_avoided": len(nodes) * len(nodes),
    }
    return (
        torch.tensor(node_features, dtype=torch.float32),
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


class EdgeAwareSAGELayer(nn.Module):
    def __init__(self, hidden_dim, edge_feature_dim, dropout):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        messages = self.message(torch.cat([h[src], edge_attr], dim=-1))
        gates = self.gate(torch.cat([h[dst], h[src], edge_attr], dim=-1))
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, dst, messages * gates)
        denominator = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        denominator.index_add_(0, dst, gates)
        aggregate = aggregate / denominator.clamp_min(1e-6)
        update = F.relu(self.update(torch.cat([h, aggregate], dim=-1)))
        return self.norm(h + self.dropout(update))


class PPRGATv2SAGELayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        edge_feature_dim,
        heads,
        ppr_index,
        dropout,
        num_classes=0,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.head_dim = hidden_dim // self.heads
        self.ppr_index = int(ppr_index)
        self.num_classes = int(num_classes)
        self.src_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_projection = nn.Linear(edge_feature_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.ppr_beta_raw = nn.Parameter(torch.full((self.heads,), -2.0))
        if self.num_classes > 0:
            relation_count = self.num_classes * self.num_classes
            self.class_pair_key = nn.Embedding(relation_count, hidden_dim)
            self.class_pair_value = nn.Embedding(relation_count, hidden_dim)
            self.class_pair_bias = nn.Embedding(relation_count, self.heads)
            nn.init.xavier_uniform_(self.class_pair_key.weight)
            nn.init.xavier_uniform_(self.class_pair_value.weight)
            nn.init.zeros_(self.class_pair_bias.weight)
        else:
            self.class_pair_key = None
            self.class_pair_value = None
            self.class_pair_bias = None
        self.update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attention)

    @staticmethod
    def _segment_softmax(scores, dst, node_count):
        index = dst[:, None].expand(-1, scores.shape[1])
        maximum = torch.full(
            (node_count, scores.shape[1]),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        maximum.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        exponent = torch.exp(scores - maximum[dst])
        denominator = torch.zeros_like(maximum)
        denominator.scatter_add_(0, index, exponent)
        return exponent / denominator[dst].clamp_min(1e-12)

    def forward(self, h, edge_index, edge_attr, class_ids=None):
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        src_key = self.src_projection(h)[src].view(-1, self.heads, self.head_dim)
        dst_query = self.dst_projection(h)[dst].view(-1, self.heads, self.head_dim)
        edge_key = self.edge_projection(edge_attr).view(-1, self.heads, self.head_dim)
        pair_key = src_key + dst_query + edge_key
        relation_ids = None
        if self.num_classes > 0:
            if class_ids is None:
                raise ValueError("class_ids are required for class-relation attention")
            relation_ids = class_ids[src] * self.num_classes + class_ids[dst]
            pair_key = pair_key + self.class_pair_key(relation_ids).view(
                -1,
                self.heads,
                self.head_dim,
            )
        dynamic_pair = F.leaky_relu(pair_key, negative_slope=0.2)
        scores = (dynamic_pair * self.attention[None, :, :]).sum(dim=-1) / math.sqrt(self.head_dim)
        if relation_ids is not None:
            scores = scores + self.class_pair_bias(relation_ids)

        ppr = edge_attr[:, self.ppr_index].clamp_min(0.0)
        ppr_prior = torch.log1p(ppr / 1e-6)[:, None]
        scores = scores + F.softplus(self.ppr_beta_raw)[None, :] * ppr_prior
        coefficients = self._segment_softmax(scores, dst, h.shape[0])

        values = self.value_projection(h)[src]
        if relation_ids is not None:
            values = values + self.class_pair_value(relation_ids)
        values = values.view(-1, self.heads, self.head_dim)
        messages = values * coefficients[:, :, None]
        aggregate = torch.zeros(
            (h.shape[0], self.heads, self.head_dim),
            dtype=h.dtype,
            device=h.device,
        )
        aggregate.index_add_(0, dst, messages)
        aggregate = aggregate.reshape(h.shape[0], -1)
        update = F.relu(self.update(torch.cat([h, aggregate], dim=-1)))
        return self.norm(h + self.dropout(update))


class Experiment2GraphRefiner(nn.Module):
    def __init__(
        self,
        input_dim,
        edge_feature_dim,
        hidden_dim,
        num_layers,
        output_dim=3,
        architecture="edge_sage",
        attention_heads=4,
        dropout=0.10,
        ppr_index=0,
        class_offset=0,
        num_classes=0,
    ):
        super().__init__()
        self.architecture = str(architecture)
        self.class_offset = int(class_offset)
        self.num_classes = int(num_classes)
        self.relation_aware = self.architecture == "class_relation_ppr_gatv2_sage"
        if self.relation_aware and self.num_classes <= 0:
            raise ValueError("num_classes must be positive for class-relation attention")
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        layers = []
        for _ in range(int(num_layers)):
            if self.architecture == "edge_sage":
                layer = EdgeAwareSAGELayer(hidden_dim, edge_feature_dim, dropout)
            elif self.architecture in {
                "ppr_gatv2_sage",
                "class_relation_ppr_gatv2_sage",
            }:
                layer = PPRGATv2SAGELayer(
                    hidden_dim,
                    edge_feature_dim,
                    attention_heads,
                    ppr_index,
                    dropout,
                    num_classes=self.num_classes if self.relation_aware else 0,
                )
            else:
                raise ValueError(f"Unknown experiment-2 architecture: {self.architecture}")
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        if self.relation_aware:
            self.class_context = nn.Sequential(
                nn.Linear(2 * self.num_classes, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.class_context_gate = nn.Sequential(
                nn.Linear(2 * self.num_classes, hidden_dim),
                nn.Sigmoid(),
            )
            self.class_context_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.input(x)
        class_ids = None
        if self.relation_aware:
            class_end = self.class_offset + self.num_classes
            if class_end > x.shape[1]:
                raise ValueError(
                    f"class feature slice [{self.class_offset}:{class_end}] "
                    f"exceeds input dim {x.shape[1]}"
                )
            class_probabilities = x[:, self.class_offset:class_end].clamp_min(0.0)
            class_probabilities = class_probabilities / class_probabilities.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            class_ids = class_probabilities.argmax(dim=-1)
            neighbor_context = torch.zeros_like(class_probabilities)
            neighbor_count = torch.zeros(
                (x.shape[0], 1),
                dtype=x.dtype,
                device=x.device,
            )
            if edge_index.numel() > 0:
                src, dst = edge_index
                non_self = src != dst
                src = src[non_self]
                dst = dst[non_self]
                if src.numel() > 0:
                    neighbor_context.index_add_(0, dst, class_probabilities[src])
                    neighbor_count.index_add_(
                        0,
                        dst,
                        torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device),
                    )
            neighbor_context = neighbor_context / neighbor_count.clamp_min(1.0)
            context_input = torch.cat([class_probabilities, neighbor_context], dim=-1)
            context_update = self.class_context(context_input)
            context_gate = self.class_context_gate(context_input)
            h = self.class_context_norm(h + context_gate * context_update)
        for layer in self.layers:
            if self.relation_aware:
                h = layer(h, edge_index, edge_attr, class_ids=class_ids)
            else:
                h = layer(h, edge_index, edge_attr)
        return self.head(torch.cat([h, x], dim=-1))


def build_model(ab, args, architecture):
    return Experiment2GraphRefiner(
        input_dim=node_dim(ab),
        edge_feature_dim=edge_dim(ab),
        hidden_dim=int(args.gnn_hidden_dim),
        num_layers=int(args.gnn_layers),
        output_dim=3,
        architecture=architecture,
        attention_heads=int(getattr(args, "exp2_attention_heads", 4)),
        dropout=float(getattr(args, "exp2_dropout", 0.10)),
        ppr_index=int(ab.SIZE_AWARE_EDGE_DIM) + PPR_EXTRA_OFFSET,
        class_offset=int(ab.NODE_CLASS_OFFSET),
        num_classes=int(ab.NUM_CLASSES),
    )
