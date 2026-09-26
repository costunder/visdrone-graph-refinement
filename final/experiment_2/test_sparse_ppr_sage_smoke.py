#!/usr/bin/env python3
"""Forward/backward checks for homogeneous and heterogeneous experiment-2 graphs."""

import math
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import run_gois_two_stage_gnn_ablation as ab
import controlled_hetero_graph as controlled_hetero
import controlled_hetero_training
import hetero_detection_class_view as hetero
import hetero_training
import sparse_ppr_sage as exp2


def make_args():
    return Namespace(
        dbscan_eps=192.0,
        dbscan_min_samples=2,
        size_graph_cluster_mode="none",
        label_iou=0.5,
        roi_label_iou=0.1,
        roi_center_margin=0.75,
        disable_roi_support_target=True,
        stage1_keep_conf=0.25,
        stage2_fragment_containment_thr=0.3,
        stage2_fragment_iou_thr=0.1,
        disable_large_preserve=True,
        large_keep_conf=0.25,
        graph_radius=80.0,
        graph_knn=6,
        exp2_spatial_cell_size=80.0,
        exp2_local_knn=2,
        exp2_cross_class_knn=1,
        exp2_ppr_knn=6,
        exp2_ppr_alpha=0.15,
        exp2_ppr_steps=6,
        exp2_ppr_frontier=24,
        exp2_attention_heads=4,
        exp2_dropout=0.0,
        exp2_hetero_class_loss_weight=0.5,
        exp2_hetero_background_weight=0.25,
        gnn_hidden_dim=32,
        gnn_layers=2,
    )


def make_chain_predictions():
    predictions = []
    for class_offset, category_id in enumerate((1, 2)):
        y = 90.0 + class_offset * 48.0
        for column in range(6):
            predictions.append(
                {
                    "image_id": 1,
                    "category_id": category_id,
                    "bbox": [45.0 + 65.0 * column, y, 28.0, 36.0],
                    "score": 0.06 + 0.025 * ((column + class_offset) % 5),
                    "_candidate_source": "coarse" if column % 2 == 0 else "fine",
                }
            )
    return predictions


def edge_pairs(edge_index):
    return {tuple(pair) for pair in edge_index.t().tolist()}


def center_distance(first, second):
    return math.hypot(
        float(first.center[0]) - float(second.center[0]),
        float(first.center[1]) - float(second.center[1]),
    )


def main():
    ab.configure_class_space("visdrone10")
    args = make_args()
    record = ab.ImageRecord(
        image_id=1,
        file_name="synthetic.jpg",
        path=Path("synthetic.jpg"),
        width=480,
        height=280,
    )
    predictions = make_chain_predictions()
    gt_boxes = [
        {
            "image_id": 1,
            "category_id": 2,
            "bbox": list(predictions[0]["bbox"]),
            "area": predictions[0]["bbox"][2] * predictions[0]["bbox"][3],
        }
    ]
    nodes = ab.build_size_aware_detection_nodes(record, [], predictions, [], args)
    sample = {
        "record": record,
        "full_predictions": [],
        "predictions": predictions,
        "nodes": nodes,
        "gt_boxes": gt_boxes,
    }

    sage_tensors = exp2.build_sparse_graph_tensors(ab, sample, args, include_ppr=False)
    ppr_tensors = exp2.build_sparse_graph_tensors(ab, sample, args, include_ppr=True)
    sage_x, sage_edge_index, sage_edge_attr, _, _ = sage_tensors
    ppr_x, ppr_edge_index, ppr_edge_attr, ppr_y, ppr_weights = ppr_tensors

    assert sage_x.shape == (len(nodes), exp2.node_dim(ab))
    assert ppr_edge_attr.shape[1] == exp2.edge_dim(ab)
    assert ppr_edge_index.shape[1] > sage_edge_index.shape[1]
    assert ppr_edge_index.shape[1] < len(nodes) * len(nodes)

    ppr_column = int(ab.SIZE_AWARE_EDGE_DIM)
    cross_class_column = ppr_column + 2
    assert torch.any(ppr_edge_attr[:, ppr_column] > 0.0)
    assert torch.any(ppr_edge_attr[:, cross_class_column] > 0.0)

    local_pairs = edge_pairs(sage_edge_index)
    high_order_pairs = []
    for edge_offset, (src, dst) in enumerate(ppr_edge_index.t().tolist()):
        pair = (int(src), int(dst))
        if pair in local_pairs or float(ppr_edge_attr[edge_offset, ppr_column]) <= 0.0:
            continue
        if center_distance(nodes[src], nodes[dst]) > args.graph_radius:
            high_order_pairs.append(pair)
    assert high_order_pairs, "PPR must add at least one edge beyond the local spatial radius"

    sage = exp2.build_model(ab, args, "edge_sage")
    ppr_gat = exp2.build_model(ab, args, "ppr_gatv2_sage")
    relation_gat = exp2.build_model(ab, args, "class_relation_ppr_gatv2_sage")
    with torch.no_grad():
        sage_logits = sage(sage_x, sage_edge_index, sage_edge_attr)
        ppr_logits = ppr_gat(ppr_x, ppr_edge_index, ppr_edge_attr)
        relation_logits = relation_gat(ppr_x, ppr_edge_index, ppr_edge_attr)
    assert sage_logits.shape == (len(nodes), 3)
    assert ppr_logits.shape == (len(nodes), 3)
    assert relation_logits.shape == (len(nodes), 3)
    assert torch.isfinite(sage_logits).all()
    assert torch.isfinite(ppr_logits).all()
    assert torch.isfinite(relation_logits).all()
    assert ppr_gat.layers[0].class_pair_key is None
    assert relation_gat.layers[0].class_pair_key.num_embeddings == ab.NUM_CLASSES ** 2

    hetero_data = hetero.build_hetero_graph_data(ab, sample, args, include_ppr=True)
    assert set(hetero_data.node_types) == set(hetero.NODE_TYPES)
    assert set(hetero_data.edge_types) == set(hetero.EDGE_TYPES)
    assert hetero_data["detection"].num_nodes == len(nodes)
    assert hetero_data["class"].num_nodes == ab.NUM_CLASSES
    assert hetero_data["view"].num_nodes >= 2
    assert hetero_data[hetero.CLASS_DET].num_edges == len(nodes) * ab.NUM_CLASSES
    assert hetero_data[hetero.CLASS_CLASS].num_edges == ab.NUM_CLASSES ** 2
    assert sample["_exp2_hetero_graph_stats"]["cross_class_correction_targets"] == 1
    assert int(hetero_data["detection"].y_class[0]) == ab.VISDRONE_TO_CLASS_INDEX[2]

    hetero_model = hetero.build_model(ab, args)
    batch = next(iter(DataLoader([hetero_data, hetero_data.clone()], batch_size=2)))
    size_logits, class_logits = hetero_model(
        batch.x_dict,
        batch.edge_index_dict,
        batch.edge_attr_dict,
    )
    assert size_logits.shape == (2 * len(nodes), 3)
    assert class_logits.shape == (2 * len(nodes), ab.NUM_CLASSES + 1)
    assert torch.isfinite(size_logits).all()
    assert torch.isfinite(class_logits).all()
    total_loss, size_loss, class_loss = hetero_training.hetero_loss(
        size_logits,
        class_logits,
        batch,
        args,
    )
    total_loss.backward()
    assert torch.isfinite(total_loss)
    assert torch.isfinite(size_loss)
    assert torch.isfinite(class_loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in hetero_model.parameters()
    )

    controlled_data = controlled_hetero.build_controlled_hetero_graph_data(
        ab,
        sample,
        args,
        include_ppr=True,
    )
    assert set(controlled_data.node_types) == set(controlled_hetero.NODE_TYPES)
    assert set(controlled_data.edge_types) == set(controlled_hetero.EDGE_TYPES)
    assert torch.equal(controlled_data["detection"].x, ppr_x)
    assert torch.equal(controlled_data["detection"].y, ppr_y)
    assert torch.equal(controlled_data["detection"].weights, ppr_weights)
    assert torch.equal(
        controlled_data[controlled_hetero.DET_DET].edge_index,
        ppr_edge_index,
    )
    assert torch.equal(
        controlled_data[controlled_hetero.DET_DET].edge_attr,
        ppr_edge_attr,
    )
    active_class_count = len({node.category_id for node in nodes})
    assert controlled_data["class"].num_nodes == active_class_count
    assert controlled_data[controlled_hetero.CLASS_DET].num_edges == len(nodes)
    assert controlled_data[controlled_hetero.DET_CLASS].num_edges == len(nodes)
    assert (
        controlled_data[controlled_hetero.CLASS_CLASS].num_edges
        <= active_class_count ** 2
    )
    assert (
        controlled_data[controlled_hetero.CLASS_CLASS].num_edges
        >= active_class_count
    )

    controlled_model = controlled_hetero.build_model(ab, args)
    controlled_batch = next(
        iter(DataLoader([controlled_data, controlled_data.clone()], batch_size=2))
    )
    controlled_logits = controlled_model(
        controlled_batch.x_dict,
        controlled_batch.edge_index_dict,
        controlled_batch.edge_attr_dict,
    )
    assert controlled_logits.shape == (2 * len(nodes), 3)
    assert torch.isfinite(controlled_logits).all()
    controlled_loss = controlled_hetero_training.controlled_loss(
        controlled_logits,
        controlled_batch,
    )
    controlled_loss.backward()
    assert torch.isfinite(controlled_loss)
    for node_type in ("class", "view"):
        gradients = [
            parameter.grad
            for name, parameter in controlled_model.named_parameters()
            if name.startswith(f"input_projection.{node_type}")
        ]
        assert gradients
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )

    class_ids = ppr_x[
        :, ab.NODE_CLASS_OFFSET : ab.NODE_CLASS_OFFSET + ab.NUM_CLASSES
    ].argmax(dim=-1)
    src, dst = ppr_edge_index
    relation_ids = class_ids[src] * ab.NUM_CLASSES + class_ids[dst]
    directed_relation_types = torch.unique(relation_ids).numel()
    assert directed_relation_types >= 4

    print(
        {
            "nodes": len(nodes),
            "sage_edges": int(sage_edge_index.shape[1]),
            "ppr_edges": int(ppr_edge_index.shape[1]),
            "nonlocal_ppr_edges": len(high_order_pairs),
            "cross_class_edges": int(ppr_edge_attr[:, cross_class_column].sum().item()),
            "directed_class_relation_types": int(directed_relation_types),
            "hetero_node_counts": {
                node_type: int(hetero_data[node_type].num_nodes)
                for node_type in hetero.NODE_TYPES
            },
            "hetero_edge_counts": {
                str(edge_type): int(hetero_data[edge_type].num_edges)
                for edge_type in hetero.EDGE_TYPES
            },
            "controlled_hetero_node_counts": {
                node_type: int(controlled_data[node_type].num_nodes)
                for node_type in controlled_hetero.NODE_TYPES
            },
            "controlled_hetero_edge_counts": {
                str(edge_type): int(controlled_data[edge_type].num_edges)
                for edge_type in controlled_hetero.EDGE_TYPES
            },
            "status": "homogeneous_and_heterogeneous_forward_backward_passed",
        }
    )


if __name__ == "__main__":
    main()
