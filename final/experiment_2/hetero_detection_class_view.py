"""True heterogeneous detection/class/view graph for experiment 2.

Detection candidates, semantic class prototypes, and detector view/source nodes
are represented as distinct PyG node types. Relation-specific GATv2 operators
preserve sparse spatial/PPR edge attributes while allowing class correction.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

import sparse_ppr_sage as sparse_graph


HETERO_GRAPH_SCHEMA_VERSION = 1
HETERO_CLASS_EXTRA_DIM = 6
HETERO_VIEW_EXTRA_DIM = 7
DET_CLASS_EDGE_DIM = 4
CLASS_CLASS_EDGE_DIM = 6
DET_VIEW_EDGE_DIM = 6

DET_DET = ("detection", "spatial_ppr", "detection")
DET_CLASS = ("detection", "predicted_as", "class")
CLASS_DET = ("class", "supports", "detection")
CLASS_CLASS = ("class", "cooccurs", "class")
DET_VIEW = ("detection", "observed_in", "view")
VIEW_DET = ("view", "contains", "detection")
EDGE_TYPES = (DET_DET, DET_CLASS, CLASS_DET, CLASS_CLASS, DET_VIEW, VIEW_DET)
NODE_TYPES = ("detection", "class", "view")


def class_node_dim(ab):
    return int(ab.NUM_CLASSES) + HETERO_CLASS_EXTRA_DIM


def view_node_dim(ab):
    return int(ab.NUM_VIEW_TYPES) + HETERO_VIEW_EXTRA_DIM + int(ab.NUM_CLASSES)


def relation_edge_dims(ab):
    return {
        DET_DET: sparse_graph.edge_dim(ab),
        DET_CLASS: DET_CLASS_EDGE_DIM,
        CLASS_DET: DET_CLASS_EDGE_DIM,
        CLASS_CLASS: CLASS_CLASS_EDGE_DIM,
        DET_VIEW: DET_VIEW_EDGE_DIM,
        VIEW_DET: DET_VIEW_EDGE_DIM,
    }


def _class_index(ab, category_id):
    return int(ab.VISDRONE_TO_CLASS_INDEX.get(int(category_id), 0))


def _all_class_targets(ab, sample, args):
    """Greedy COCO-style targets that permit correction of a wrong hard class."""
    nodes = sample.get("nodes", [])
    gt_boxes = sample.get("gt_boxes", [])
    count = len(nodes)
    background_index = int(ab.NUM_CLASSES)
    size_targets = torch.zeros((count, 3), dtype=torch.float32)
    class_targets = torch.full((count,), background_index, dtype=torch.long)
    weights = torch.ones((count,), dtype=torch.float32)
    label_iou = float(getattr(args, "label_iou", 0.50))

    matches = {}
    used_gt = set()
    order = sorted(range(count), key=lambda index: float(nodes[index].max_score), reverse=True)
    for node_index in order:
        node = nodes[node_index]
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in used_gt:
                continue
            if int(gt.get("category_id", -1)) not in ab.VISDRONE_TO_CLASS_INDEX:
                continue
            overlap = float(ab.iou_xywh(node.bbox, gt["bbox"]))
            if overlap > best_iou:
                best_gt_index = gt_index
                best_iou = overlap
        if best_gt_index is not None and best_iou >= label_iou:
            used_gt.add(best_gt_index)
            matches[node_index] = (gt_boxes[best_gt_index], best_iou)

    corrected_class_targets = 0
    for node_index, node in enumerate(nodes):
        matched = matches.get(node_index)
        if matched is None:
            if float(node.max_score) >= float(getattr(args, "stage1_keep_conf", 0.25)):
                weights[node_index] = 1.5
            continue
        gt, _ = matched
        gt_area = float(gt.get("area", ab.bbox_area(gt["bbox"])))
        is_small = gt_area < float(ab.SMALL_MEDIUM_AREA_THR)
        size_targets[node_index, 0] = 1.0
        size_targets[node_index, 1 if is_small else 2] = 1.0
        class_targets[node_index] = _class_index(ab, gt["category_id"])
        weights[node_index] = 3.0 if is_small else 2.5
        if int(gt["category_id"]) != int(node.category_id):
            corrected_class_targets += 1

    stats = {
        "matched_detections": len(matches),
        "background_detections": count - len(matches),
        "cross_class_correction_targets": corrected_class_targets,
    }
    return size_targets, class_targets, weights, stats


def _class_node_features(ab, nodes, record):
    num_classes = int(ab.NUM_CLASSES)
    image_area = max(1.0, float(record.width) * float(record.height))
    class_ids = np.asarray([_class_index(ab, node.category_id) for node in nodes], dtype=np.int64)
    scores = np.asarray([float(node.max_score) for node in nodes], dtype=np.float32)
    areas = np.asarray(
        [math.sqrt(max(1.0, float(node.union_area)) / image_area) for node in nodes],
        dtype=np.float32,
    )
    centers_x = np.asarray(
        [float(node.center[0]) / max(1.0, float(record.width)) for node in nodes],
        dtype=np.float32,
    )
    centers_y = np.asarray(
        [float(node.center[1]) / max(1.0, float(record.height)) for node in nodes],
        dtype=np.float32,
    )
    features = []
    for class_id in range(num_classes):
        mask = class_ids == class_id
        identity = [0.0] * num_classes
        identity[class_id] = 1.0
        if np.any(mask):
            extras = [
                float(mask.mean()),
                float(scores[mask].mean()),
                float(areas[mask].mean()),
                float(centers_x[mask].mean()),
                float(centers_y[mask].mean()),
                1.0,
            ]
        else:
            extras = [0.0] * HETERO_CLASS_EXTRA_DIM
        features.append(identity + extras)
    return torch.tensor(features, dtype=torch.float32), class_ids


def _view_nodes(ab, sample, source_tags, view_metadata):
    nodes = sample["nodes"]
    record = sample["record"]
    num_classes = int(ab.NUM_CLASSES)
    image_area = max(1.0, float(record.width) * float(record.height))
    view_to_index = {}
    view_members = []
    for node_index, metadata in enumerate(view_metadata):
        view_id = str(metadata[1])
        if view_id not in view_to_index:
            view_to_index[view_id] = len(view_members)
            view_members.append([])
        view_members[view_to_index[view_id]].append(node_index)

    features = []
    detection_view_indices = [0] * len(nodes)
    for view_index, members in enumerate(view_members):
        first_metadata = view_metadata[members[0]]
        view_type, _, view_bbox = first_metadata
        vx, vy, vw, vh = ab.clip_xywh(view_bbox, record.width, record.height)
        type_one_hot = [0.0] * int(ab.NUM_VIEW_TYPES)
        if 0 <= int(view_type) < len(type_one_hot):
            type_one_hot[int(view_type)] = 1.0
        class_histogram = [0.0] * num_classes
        for node_index in members:
            detection_view_indices[node_index] = view_index
            class_histogram[_class_index(ab, nodes[node_index].category_id)] += 1.0
        member_count = max(1, len(members))
        class_histogram = [value / member_count for value in class_histogram]
        mean_score = sum(float(nodes[index].max_score) for index in members) / member_count
        mean_area = sum(
            math.sqrt(max(1.0, float(nodes[index].union_area)) / image_area)
            for index in members
        ) / member_count
        features.append(
            type_one_hot
            + [
                vx / max(1.0, float(record.width)),
                vy / max(1.0, float(record.height)),
                vw / max(1.0, float(record.width)),
                vh / max(1.0, float(record.height)),
                member_count / max(1, len(nodes)),
                mean_score,
                mean_area,
            ]
            + class_histogram
        )
    return torch.tensor(features, dtype=torch.float32), detection_view_indices


def _predicted_class_edges(ab, nodes, class_ids, class_frequencies, record):
    image_area = max(1.0, float(record.width) * float(record.height))
    src = torch.arange(len(nodes), dtype=torch.long)
    dst = torch.tensor(class_ids, dtype=torch.long)
    attributes = []
    for node, class_id in zip(nodes, class_ids.tolist()):
        attributes.append(
            [
                float(node.max_score),
                math.sqrt(max(1.0, float(node.union_area)) / image_area),
                float(class_frequencies[class_id]),
                1.0,
            ]
        )
    return torch.stack([src, dst]), torch.tensor(attributes, dtype=torch.float32)


def _class_support_edges(ab, nodes, class_ids, class_frequencies):
    num_classes = int(ab.NUM_CLASSES)
    source = []
    destination = []
    attributes = []
    for node_index, (node, predicted_class) in enumerate(zip(nodes, class_ids.tolist())):
        score = min(1.0, max(0.0, float(node.max_score)))
        residual = (1.0 - score) / max(1, num_classes - 1)
        for class_id in range(num_classes):
            is_predicted = class_id == predicted_class
            source.append(class_id)
            destination.append(node_index)
            attributes.append(
                [
                    score if is_predicted else residual,
                    float(is_predicted),
                    score,
                    float(class_frequencies[class_id]),
                ]
            )
    return (
        torch.tensor([source, destination], dtype=torch.long),
        torch.tensor(attributes, dtype=torch.float32),
    )


def _class_cooccurrence_edges(ab, class_ids, detection_edge_index, detection_edge_attr):
    num_classes = int(ab.NUM_CLASSES)
    pair_strength = np.zeros((num_classes, num_classes), dtype=np.float64)
    if detection_edge_index.numel() > 0:
        ppr_index = int(ab.SIZE_AWARE_EDGE_DIM) + sparse_graph.PPR_EXTRA_OFFSET
        for edge_offset, (src, dst) in enumerate(detection_edge_index.t().tolist()):
            if int(src) == int(dst):
                continue
            source_class = int(class_ids[int(src)])
            target_class = int(class_ids[int(dst)])
            ppr = float(detection_edge_attr[edge_offset, ppr_index])
            pair_strength[source_class, target_class] += 1.0 + ppr

    frequencies = np.bincount(class_ids, minlength=num_classes).astype(np.float64)
    frequencies /= max(1.0, frequencies.sum())
    total = max(1.0, pair_strength.sum())
    outgoing = np.maximum(1.0, pair_strength.sum(axis=1))
    incoming = np.maximum(1.0, pair_strength.sum(axis=0))
    sources = []
    destinations = []
    attributes = []
    for source_class in range(num_classes):
        for target_class in range(num_classes):
            value = pair_strength[source_class, target_class]
            sources.append(source_class)
            destinations.append(target_class)
            attributes.append(
                [
                    value / total,
                    value / outgoing[source_class],
                    value / incoming[target_class],
                    frequencies[source_class],
                    frequencies[target_class],
                    float(source_class == target_class),
                ]
            )
    return (
        torch.tensor([sources, destinations], dtype=torch.long),
        torch.tensor(attributes, dtype=torch.float32),
    )


def _detection_view_edges(ab, sample, detection_view_indices, view_metadata):
    nodes = sample["nodes"]
    record = sample["record"]
    image_area = max(1.0, float(record.width) * float(record.height))
    source = torch.arange(len(nodes), dtype=torch.long)
    destination = torch.tensor(detection_view_indices, dtype=torch.long)
    attributes = []
    for node, metadata in zip(nodes, view_metadata):
        geometry = sparse_graph._view_geometry(ab, node, record, metadata[2])
        attributes.append(
            [
                float(node.max_score),
                math.sqrt(max(1.0, float(node.union_area)) / image_area),
                float(geometry[4]),
                float(geometry[5]),
                float(int(node.source) == int(ab.SOURCE_SLICE)),
                1.0,
            ]
        )
    edge_index = torch.stack([source, destination])
    edge_attr = torch.tensor(attributes, dtype=torch.float32)
    return edge_index, edge_attr


def build_hetero_graph_data(ab, sample, args, include_ppr=True):
    nodes = sample.get("nodes", [])
    if not nodes:
        raise ValueError("Heterogeneous graph construction requires at least one detection node")

    detection_x, detection_edge_index, detection_edge_attr, _, _ = (
        sparse_graph.build_sparse_graph_tensors(ab, sample, args, include_ppr=include_ppr)
    )
    record = sample["record"]
    source_tags = [
        sparse_graph._candidate_source_tag(ab, sample, node)
        for node in nodes
    ]
    view_metadata = [
        sparse_graph._node_view_metadata(ab, sample, node, record, source_tags[index])
        for index, node in enumerate(nodes)
    ]
    class_x, class_ids = _class_node_features(ab, nodes, record)
    class_frequencies = np.bincount(class_ids, minlength=int(ab.NUM_CLASSES)).astype(np.float64)
    class_frequencies /= max(1.0, class_frequencies.sum())
    view_x, detection_view_indices = _view_nodes(ab, sample, source_tags, view_metadata)
    size_targets, class_targets, weights, target_stats = _all_class_targets(ab, sample, args)

    predicted_edge_index, predicted_edge_attr = _predicted_class_edges(
        ab, nodes, class_ids, class_frequencies, record
    )
    support_edge_index, support_edge_attr = _class_support_edges(
        ab, nodes, class_ids, class_frequencies
    )
    class_edge_index, class_edge_attr = _class_cooccurrence_edges(
        ab, class_ids, detection_edge_index, detection_edge_attr
    )
    view_edge_index, view_edge_attr = _detection_view_edges(
        ab, sample, detection_view_indices, view_metadata
    )

    data = HeteroData()
    data["detection"].x = detection_x
    data["detection"].y_size = size_targets
    data["detection"].y_class = class_targets
    data["detection"].weights = weights
    data["class"].x = class_x
    data["view"].x = view_x
    data[DET_DET].edge_index = detection_edge_index
    data[DET_DET].edge_attr = detection_edge_attr
    data[DET_CLASS].edge_index = predicted_edge_index
    data[DET_CLASS].edge_attr = predicted_edge_attr
    data[CLASS_DET].edge_index = support_edge_index
    data[CLASS_DET].edge_attr = support_edge_attr
    data[CLASS_CLASS].edge_index = class_edge_index
    data[CLASS_CLASS].edge_attr = class_edge_attr
    data[DET_VIEW].edge_index = view_edge_index
    data[DET_VIEW].edge_attr = view_edge_attr
    data[VIEW_DET].edge_index = view_edge_index.flip(0)
    data[VIEW_DET].edge_attr = view_edge_attr.clone()

    sample["_exp2_hetero_graph_stats"] = {
        "node_counts": {
            "detection": len(nodes),
            "class": int(ab.NUM_CLASSES),
            "view": int(view_x.shape[0]),
        },
        "edge_counts": {
            "detection_spatial_ppr_detection": int(detection_edge_index.shape[1]),
            "detection_predicted_as_class": int(predicted_edge_index.shape[1]),
            "class_supports_detection": int(support_edge_index.shape[1]),
            "class_cooccurs_class": int(class_edge_index.shape[1]),
            "detection_observed_in_view": int(view_edge_index.shape[1]),
            "view_contains_detection": int(view_edge_index.shape[1]),
        },
        **target_stats,
    }
    return data


class Experiment2HeteroRefiner(nn.Module):
    """Relation-specific GATv2 over three explicit PyG node types."""

    def __init__(self, ab, args):
        super().__init__()
        hidden_dim = int(args.gnn_hidden_dim)
        heads = int(getattr(args, "exp2_attention_heads", 4))
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        dropout = float(getattr(args, "exp2_dropout", 0.10))
        self.num_classes = int(ab.NUM_CLASSES)
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.ModuleDict(
            {
                "detection": nn.Linear(sparse_graph.node_dim(ab), hidden_dim),
                "class": nn.Linear(class_node_dim(ab), hidden_dim),
                "view": nn.Linear(view_node_dim(ab), hidden_dim),
            }
        )
        self.input_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES}
        )

        edge_dims = relation_edge_dims(ab)
        self.convs = nn.ModuleList()
        self.updates = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(args.gnn_layers)):
            operators = {}
            for edge_type, edge_feature_dim in edge_dims.items():
                source_type, _, target_type = edge_type
                in_channels = hidden_dim if source_type == target_type else (hidden_dim, hidden_dim)
                operators[edge_type] = GATv2Conv(
                    in_channels,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=edge_feature_dim,
                )
            self.convs.append(HeteroConv(operators, aggr="mean"))
            self.updates.append(
                nn.ModuleDict(
                    {
                        node_type: nn.Sequential(
                            nn.Linear(2 * hidden_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, hidden_dim),
                        )
                        for node_type in NODE_TYPES
                    }
                )
            )
            self.norms.append(
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES}
                )
            )

        detection_input_dim = sparse_graph.node_dim(ab)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim + detection_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim + detection_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes + 1),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        hidden = {
            node_type: self.input_norm[node_type](
                F.relu(self.input_projection[node_type](x_dict[node_type]))
            )
            for node_type in NODE_TYPES
        }
        for convolution, updates, norms in zip(self.convs, self.updates, self.norms):
            aggregated = convolution(
                hidden,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
            hidden = {
                node_type: norms[node_type](
                    hidden[node_type]
                    + self.dropout(
                        updates[node_type](
                            torch.cat([hidden[node_type], aggregated[node_type]], dim=-1)
                        )
                    )
                )
                for node_type in NODE_TYPES
            }
        detection_state = torch.cat([hidden["detection"], x_dict["detection"]], dim=-1)
        return self.size_head(detection_state), self.class_head(detection_state)


def build_model(ab, args):
    return Experiment2HeteroRefiner(ab, args)
