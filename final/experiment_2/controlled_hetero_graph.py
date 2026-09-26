"""Controlled heterogeneous backbone for experiment 2.

This variant keeps the detection tensors, targets, weights, sparse PPR graph,
loss, and inference policy from variant 06. Only the message-passing backbone
changes: detection, active class, and detector-view entities have distinct
states and relation-specific GATv2 operators.
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv

import sparse_ppr_sage as sparse_graph


CONTROLLED_HETERO_GRAPH_SCHEMA_VERSION = 1
CLASS_EXTRA_DIM = 8
VIEW_EXTRA_DIM = 8
DET_CLASS_EDGE_DIM = 5
CLASS_CLASS_EDGE_DIM = 9
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
    return int(ab.NUM_CLASSES) + CLASS_EXTRA_DIM


def view_node_dim(ab):
    return int(ab.NUM_VIEW_TYPES) + VIEW_EXTRA_DIM + int(ab.NUM_CLASSES)


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


def _active_class_nodes(ab, nodes, source_tags, record):
    global_ids = np.asarray(
        [_class_index(ab, node.category_id) for node in nodes],
        dtype=np.int64,
    )
    active_global_ids = sorted(set(global_ids.tolist()))
    global_to_local = {
        global_id: local_id
        for local_id, global_id in enumerate(active_global_ids)
    }
    local_ids = np.asarray(
        [global_to_local[int(global_id)] for global_id in global_ids],
        dtype=np.int64,
    )

    image_area = max(1.0, float(record.width) * float(record.height))
    scores = np.asarray([float(node.max_score) for node in nodes], dtype=np.float32)
    areas = np.asarray(
        [
            math.sqrt(max(1.0, float(node.union_area)) / image_area)
            for node in nodes
        ],
        dtype=np.float32,
    )
    centers_x = np.asarray(
        [
            float(node.center[0]) / max(1.0, float(record.width))
            for node in nodes
        ],
        dtype=np.float32,
    )
    centers_y = np.asarray(
        [
            float(node.center[1]) / max(1.0, float(record.height))
            for node in nodes
        ],
        dtype=np.float32,
    )
    source_tags = np.asarray(source_tags, dtype=object)

    features = []
    frequencies = np.zeros((len(active_global_ids),), dtype=np.float32)
    for local_id, global_id in enumerate(active_global_ids):
        mask = local_ids == local_id
        member_count = int(mask.sum())
        frequencies[local_id] = member_count / max(1, len(nodes))
        identity = [0.0] * int(ab.NUM_CLASSES)
        identity[global_id] = 1.0
        features.append(
            identity
            + [
                float(frequencies[local_id]),
                float(scores[mask].mean()),
                float(scores[mask].max()),
                float(areas[mask].mean()),
                float(centers_x[mask].mean()),
                float(centers_y[mask].mean()),
                float(np.mean(source_tags[mask] == "coarse")),
                float(np.mean(source_tags[mask] == "fine")),
            ]
        )
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(active_global_ids, dtype=torch.long),
        local_ids,
        frequencies,
    )


def _view_nodes(ab, sample, source_tags, view_metadata):
    nodes = sample["nodes"]
    record = sample["record"]
    image_area = max(1.0, float(record.width) * float(record.height))
    view_to_local = {}
    view_members = []
    for node_index, metadata in enumerate(view_metadata):
        view_id = str(metadata[1])
        if view_id not in view_to_local:
            view_to_local[view_id] = len(view_members)
            view_members.append([])
        view_members[view_to_local[view_id]].append(node_index)

    features = []
    detection_view_ids = [0] * len(nodes)
    for view_index, members in enumerate(view_members):
        view_type, _, view_bbox = view_metadata[members[0]]
        vx, vy, vw, vh = ab.clip_xywh(
            view_bbox,
            record.width,
            record.height,
        )
        view_one_hot = [0.0] * int(ab.NUM_VIEW_TYPES)
        if 0 <= int(view_type) < len(view_one_hot):
            view_one_hot[int(view_type)] = 1.0

        class_histogram = [0.0] * int(ab.NUM_CLASSES)
        for node_index in members:
            detection_view_ids[node_index] = view_index
            class_histogram[_class_index(ab, nodes[node_index].category_id)] += 1.0
        member_count = max(1, len(members))
        class_histogram = [value / member_count for value in class_histogram]
        mean_score = sum(float(nodes[index].max_score) for index in members) / member_count
        mean_area = sum(
            math.sqrt(max(1.0, float(nodes[index].union_area)) / image_area)
            for index in members
        ) / member_count
        slice_fraction = sum(
            int(nodes[index].source) == int(ab.SOURCE_SLICE)
            for index in members
        ) / member_count
        features.append(
            view_one_hot
            + [
                vx / max(1.0, float(record.width)),
                vy / max(1.0, float(record.height)),
                vw / max(1.0, float(record.width)),
                vh / max(1.0, float(record.height)),
                member_count / max(1, len(nodes)),
                mean_score,
                mean_area,
                slice_fraction,
            ]
            + class_histogram
        )
    return torch.tensor(features, dtype=torch.float32), detection_view_ids


def _detection_class_edges(nodes, local_class_ids, class_frequencies, record):
    image_area = max(1.0, float(record.width) * float(record.height))
    source = torch.arange(len(nodes), dtype=torch.long)
    destination = torch.tensor(local_class_ids, dtype=torch.long)
    attributes = []
    for node, local_class_id in zip(nodes, local_class_ids.tolist()):
        score = max(0.0, min(1.0, float(node.max_score)))
        attributes.append(
            [
                score,
                1.0 - score,
                float(class_frequencies[local_class_id]),
                math.sqrt(max(1.0, float(node.union_area)) / image_area),
                1.0,
            ]
        )
    return torch.stack([source, destination]), torch.tensor(
        attributes,
        dtype=torch.float32,
    )


def _class_cooccurrence_edges(
    ab,
    local_class_ids,
    class_frequencies,
    detection_edge_index,
    detection_edge_attr,
):
    pair_stats = defaultdict(
        lambda: {
            "count": 0,
            "ppr_sum": 0.0,
            "ppr_max": 0.0,
            "proximity_sum": 0.0,
            "overlap_sum": 0.0,
        }
    )
    ppr_index = int(ab.SIZE_AWARE_EDGE_DIM) + sparse_graph.PPR_EXTRA_OFFSET
    nonself_count = 0
    for edge_offset, (source, destination) in enumerate(
        detection_edge_index.t().tolist()
    ):
        if int(source) == int(destination):
            continue
        source_class = int(local_class_ids[int(source)])
        target_class = int(local_class_ids[int(destination)])
        stats = pair_stats[(source_class, target_class)]
        ppr = float(detection_edge_attr[edge_offset, ppr_index])
        stats["count"] += 1
        stats["ppr_sum"] += ppr
        stats["ppr_max"] = max(stats["ppr_max"], ppr)
        stats["proximity_sum"] += float(detection_edge_attr[edge_offset, 1])
        stats["overlap_sum"] += float(detection_edge_attr[edge_offset, 3])
        nonself_count += 1

    active_class_count = len(class_frequencies)
    for local_class_id in range(active_class_count):
        pair_stats[(local_class_id, local_class_id)]

    outgoing = defaultdict(int)
    for (source_class, _), stats in pair_stats.items():
        outgoing[source_class] += int(stats["count"])

    sources = []
    destinations = []
    attributes = []
    for source_class, target_class in sorted(pair_stats):
        stats = pair_stats[(source_class, target_class)]
        count = int(stats["count"])
        divisor = max(1, count)
        sources.append(source_class)
        destinations.append(target_class)
        attributes.append(
            [
                count / max(1, nonself_count),
                count / max(1, outgoing[source_class]),
                stats["ppr_sum"] / divisor,
                stats["ppr_max"],
                stats["proximity_sum"] / divisor,
                stats["overlap_sum"] / divisor,
                float(class_frequencies[source_class]),
                float(class_frequencies[target_class]),
                float(source_class == target_class),
            ]
        )
    return (
        torch.tensor([sources, destinations], dtype=torch.long),
        torch.tensor(attributes, dtype=torch.float32),
    )


def _detection_view_edges(ab, sample, detection_view_ids, view_metadata):
    nodes = sample["nodes"]
    record = sample["record"]
    image_area = max(1.0, float(record.width) * float(record.height))
    source = torch.arange(len(nodes), dtype=torch.long)
    destination = torch.tensor(detection_view_ids, dtype=torch.long)
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
    return torch.stack([source, destination]), torch.tensor(
        attributes,
        dtype=torch.float32,
    )


def build_controlled_hetero_graph_data(ab, sample, args, include_ppr=True):
    nodes = sample.get("nodes", [])
    if not nodes:
        raise ValueError("Controlled heterogeneous graph requires detection nodes")

    detection_x, detection_edge_index, detection_edge_attr, targets, weights = (
        sparse_graph.build_sparse_graph_tensors(
            ab,
            sample,
            args,
            include_ppr=include_ppr,
        )
    )
    record = sample["record"]
    source_tags = [
        sparse_graph._candidate_source_tag(ab, sample, node)
        for node in nodes
    ]
    view_metadata = [
        sparse_graph._node_view_metadata(
            ab,
            sample,
            node,
            record,
            source_tags[index],
        )
        for index, node in enumerate(nodes)
    ]
    class_x, active_global_ids, local_class_ids, class_frequencies = (
        _active_class_nodes(ab, nodes, source_tags, record)
    )
    view_x, detection_view_ids = _view_nodes(
        ab,
        sample,
        source_tags,
        view_metadata,
    )
    detection_class_index, detection_class_attr = _detection_class_edges(
        nodes,
        local_class_ids,
        class_frequencies,
        record,
    )
    class_class_index, class_class_attr = _class_cooccurrence_edges(
        ab,
        local_class_ids,
        class_frequencies,
        detection_edge_index,
        detection_edge_attr,
    )
    detection_view_index, detection_view_attr = _detection_view_edges(
        ab,
        sample,
        detection_view_ids,
        view_metadata,
    )

    data = HeteroData()
    data["detection"].x = detection_x
    data["detection"].y = targets
    data["detection"].weights = weights
    data["detection"].class_global_id = torch.tensor(
        [_class_index(ab, node.category_id) for node in nodes],
        dtype=torch.long,
    )
    data["class"].x = class_x
    data["class"].global_id = active_global_ids
    data["view"].x = view_x

    data[DET_DET].edge_index = detection_edge_index
    data[DET_DET].edge_attr = detection_edge_attr
    data[DET_CLASS].edge_index = detection_class_index
    data[DET_CLASS].edge_attr = detection_class_attr
    data[CLASS_DET].edge_index = detection_class_index.flip(0)
    data[CLASS_DET].edge_attr = detection_class_attr.clone()
    data[CLASS_CLASS].edge_index = class_class_index
    data[CLASS_CLASS].edge_attr = class_class_attr
    data[DET_VIEW].edge_index = detection_view_index
    data[DET_VIEW].edge_attr = detection_view_attr
    data[VIEW_DET].edge_index = detection_view_index.flip(0)
    data[VIEW_DET].edge_attr = detection_view_attr.clone()

    sample["_exp2_controlled_hetero_graph_stats"] = {
        "node_counts": {
            "detection": len(nodes),
            "class": int(class_x.shape[0]),
            "view": int(view_x.shape[0]),
        },
        "edge_counts": {
            "detection_spatial_ppr_detection": int(detection_edge_index.shape[1]),
            "detection_predicted_as_class": int(detection_class_index.shape[1]),
            "class_supports_detection": int(detection_class_index.shape[1]),
            "class_cooccurs_class": int(class_class_index.shape[1]),
            "detection_observed_in_view": int(detection_view_index.shape[1]),
            "view_contains_detection": int(detection_view_index.shape[1]),
        },
        "active_global_class_ids": active_global_ids.tolist(),
        "control_invariants": {
            "detection_features": "identical_to_variant_06",
            "detection_edges": "identical_to_variant_06",
            "targets_and_weights": "identical_to_variant_06",
        },
    }
    return data


def _relation_key(edge_type):
    return "__".join(edge_type)


class ControlledHeteroLayer(nn.Module):
    """Relation-specific attention followed by a learned relation mixture."""

    def __init__(self, hidden_dim, heads, dropout, edge_dims):
        super().__init__()
        self.edge_types = tuple(edge_dims)
        self.convs = nn.ModuleDict()
        self.relation_logits = nn.ParameterDict()
        for edge_type, feature_dim in edge_dims.items():
            key = _relation_key(edge_type)
            self.convs[key] = GATv2Conv(
                (hidden_dim, hidden_dim),
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=feature_dim,
            )
            initial_logit = 1.0 if edge_type == DET_DET else 0.0
            self.relation_logits[key] = nn.Parameter(
                torch.tensor(initial_logit, dtype=torch.float32)
            )
        self.updates = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for node_type in NODE_TYPES
            }
        )
        self.norms = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in NODE_TYPES
            }
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, edge_index_dict, edge_attr_dict):
        messages = {node_type: [] for node_type in NODE_TYPES}
        logits = {node_type: [] for node_type in NODE_TYPES}
        for edge_type in self.edge_types:
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue
            source_type, _, target_type = edge_type
            key = _relation_key(edge_type)
            relation_message = self.convs[key](
                (hidden[source_type], hidden[target_type]),
                edge_index,
                edge_attr_dict[edge_type],
            )
            messages[target_type].append(relation_message)
            logits[target_type].append(self.relation_logits[key])

        updated = {}
        for node_type in NODE_TYPES:
            if messages[node_type]:
                coefficients = torch.softmax(
                    torch.stack(logits[node_type]),
                    dim=0,
                )
                aggregate = sum(
                    coefficient * message
                    for coefficient, message in zip(
                        coefficients,
                        messages[node_type],
                    )
                )
            else:
                aggregate = torch.zeros_like(hidden[node_type])
            delta = self.updates[node_type](
                torch.cat([hidden[node_type], aggregate], dim=-1)
            )
            updated[node_type] = self.norms[node_type](
                hidden[node_type] + self.dropout(delta)
            )
        return updated


class ControlledHeteroRefiner(nn.Module):
    """Heterogeneous replacement for variant 06's homogeneous backbone."""

    def __init__(self, ab, args):
        super().__init__()
        hidden_dim = int(args.gnn_hidden_dim)
        heads = int(getattr(args, "exp2_attention_heads", 4))
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by heads={heads}"
            )
        dropout = float(getattr(args, "exp2_dropout", 0.10))
        self.input_projection = nn.ModuleDict(
            {
                "detection": nn.Linear(
                    sparse_graph.node_dim(ab),
                    hidden_dim,
                ),
                "class": nn.Linear(class_node_dim(ab), hidden_dim),
                "view": nn.Linear(view_node_dim(ab), hidden_dim),
            }
        )
        self.input_norm = nn.ModuleDict(
            {
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in NODE_TYPES
            }
        )
        edge_dims = relation_edge_dims(ab)
        self.layers = nn.ModuleList(
            [
                ControlledHeteroLayer(
                    hidden_dim,
                    heads,
                    dropout,
                    edge_dims,
                )
                for _ in range(int(args.gnn_layers))
            ]
        )
        detection_input_dim = sparse_graph.node_dim(ab)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + detection_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        hidden = {
            node_type: self.input_norm[node_type](
                F.relu(self.input_projection[node_type](x_dict[node_type]))
            )
            for node_type in NODE_TYPES
        }
        for layer in self.layers:
            hidden = layer(hidden, edge_index_dict, edge_attr_dict)
        detection_state = torch.cat(
            [hidden["detection"], x_dict["detection"]],
            dim=-1,
        )
        return self.head(detection_state)


def build_model(ab, args):
    return ControlledHeteroRefiner(ab, args)
