#!/usr/bin/env python3
"""Invariant checks for the Experiment-1 relation graph schema."""

import inspect
import math
import sys
from argparse import Namespace
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))

import run_gois_two_stage_gnn_ablation as ab
import run_gois_paper_ablation_curves as pipeline


def graph_args():
    return Namespace(
        dbscan_eps=192.0,
        dbscan_eps_ratio=0.08,
        dbscan_min_samples=2,
        size_graph_cluster_mode="dbscan",
        label_iou=0.5,
        roi_label_iou=0.1,
        roi_center_margin=0.75,
        disable_roi_support_target=True,
        stage1_keep_conf=0.25,
        stage2_fragment_containment_thr=0.3,
        stage2_fragment_iou_thr=0.1,
        disable_large_preserve=True,
        large_keep_conf=0.25,
        graph_radius=256.0,
        graph_radius_ratio=0.12,
        graph_cross_class_radius_ratio=0.08,
        graph_knn=4,
        graph_cross_class_knn=2,
        final_nms_iou=0.4,
        max_det=300,
        gnn_score_alpha=0.25,
    )


def prediction(image_id, category_id, bbox, score, view_type, view_id, view_bbox):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(value) for value in bbox],
        "score": float(score),
        "_view_type": int(view_type),
        "_view_id": str(view_id),
        "_view_bbox": [float(value) for value in view_bbox],
        "_candidate_source": ab.VIEW_TYPE_NAMES[int(view_type)],
    }


def candidate_pool(scale=1.0):
    def box(values):
        return [scale * float(value) for value in values]

    return [
        prediction(1, 1, box([100, 100, 80, 44]), 0.72, ab.VIEW_COARSE, "coarse:0", box([0, 0, 640, 500])),
        prediction(1, 1, box([105, 102, 76, 42]), 0.68, ab.VIEW_FINE, "fine:0", box([48, 48, 256, 256])),
        prediction(1, 2, box([108, 101, 78, 43]), 0.51, ab.VIEW_COARSE, "coarse:1", box([0, 0, 640, 500])),
        prediction(1, 1, box([760, 350, 52, 36]), 0.60, ab.VIEW_FINE, "fine:1", box([700, 250, 256, 250])),
    ]


def build(scale=1.0):
    width = int(1000 * scale)
    height = int(500 * scale)
    record = ab.ImageRecord(1, "synthetic.jpg", Path("synthetic.jpg"), width, height)
    predictions = candidate_pool(scale)
    nodes = ab.build_size_aware_detection_nodes(record, [], predictions, [], graph_args())
    tensors = ab.build_size_aware_tensors(nodes, record, graph_args())
    return record, predictions, nodes, tensors


def edge_map(edge_index, edge_attr):
    return {
        (int(src), int(dst)): edge_attr[index]
        for index, (src, dst) in enumerate(edge_index.t().tolist())
    }


def main():
    ab.configure_class_space("visdrone10")
    pipeline_source = inspect.getsource(pipeline.run_pipeline_ablation)
    local_execution_markers = [
        "if GNN_NO_CLUSTER_VARIANT in selected:",
        "if GNN_CLUSTER_VARIANT in selected:",
        "if SAME_POOL_SIZE_REFINEMENT_VARIANT in selected:",
        "if SAME_POOL_RERANK_PRUNE_VARIANT in selected:",
    ]
    marker_positions = [pipeline_source.index(marker) for marker in local_execution_markers]
    assert marker_positions == sorted(marker_positions), "local variants 02-05 must execute in local order"

    record, predictions, nodes, (x, edge_index, edge_attr, targets, weights) = build()

    coarse_cache = {"1": [predictions[0], predictions[2]]}
    fine_cache = {"1": [predictions[1], predictions[3]]}
    pre_nms = pipeline.make_pre_gois_candidate_cache([record], coarse_cache, fine_cache)
    post_nms = pipeline.make_global_gois_candidate_cache([record], coarse_cache, fine_cache, graph_args())
    assert len(pre_nms["1"]) == 4, "graph pool must retain overlapping cross-view candidates"
    assert len(post_nms["1"]) == 3, "baseline output pool must apply cross-view NMS"

    assert len(nodes) == len(predictions), "pre-NMS overlapping candidates must remain separate graph nodes"
    assert [node.view_type for node in nodes[:3]] == [ab.VIEW_COARSE, ab.VIEW_FINE, ab.VIEW_COARSE]
    assert nodes[0].dbscan_label >= 0 and nodes[0].dbscan_label == nodes[1].dbscan_label
    assert x.shape == (len(nodes), ab.SIZE_AWARE_NODE_DIM)
    assert edge_attr.shape[1] == ab.SIZE_AWARE_EDGE_DIM
    assert targets.shape == (len(nodes), ab.SIZE_AWARE_OUTPUT_DIM)
    assert weights.shape == (len(nodes),)
    assert torch.isfinite(x).all() and torch.isfinite(edge_attr).all()

    coarse_column = ab.NODE_VIEW_TYPE_OFFSET + ab.VIEW_COARSE
    fine_column = ab.NODE_VIEW_TYPE_OFFSET + ab.VIEW_FINE
    assert float(x[0, coarse_column]) == 1.0
    assert float(x[1, fine_column]) == 1.0
    assert int(torch.argmax(x[0, ab.NODE_CLASS_OFFSET : ab.NODE_CLASS_OFFSET + ab.NUM_CLASSES])) == 0
    assert int(torch.argmax(x[2, ab.NODE_CLASS_OFFSET : ab.NODE_CLASS_OFFSET + ab.NUM_CLASSES])) == 1

    edges = edge_map(edge_index, edge_attr)
    for src, dst in edges:
        if src != dst:
            assert (dst, src) in edges, f"missing reverse edge for {(src, dst)}"
    for index in range(len(nodes)):
        self_attr = edges[(index, index)]
        assert float(self_attr[ab.EDGE_RELATION_OFFSET + ab.EDGE_RELATION_INDEX["self"]]) == 1.0

    overlap_attr = edges[(0, 1)]
    assert float(overlap_attr[ab.EDGE_RELATION_OFFSET + ab.EDGE_RELATION_INDEX["overlap"]]) == 1.0
    assert float(overlap_attr[ab.EDGE_RELATION_OFFSET + ab.EDGE_RELATION_INDEX["cross_view"]]) == 1.0
    cross_class_attr = edges[(0, 2)]
    assert float(cross_class_attr[ab.EDGE_RELATION_OFFSET + ab.EDGE_RELATION_INDEX["cross_class_context"]]) == 1.0

    if pipeline.PyGData is not None:
        pyg_data = pipeline.sample_to_pyg_data(
            {"record": record, "predictions": predictions, "nodes": nodes},
            graph_args(),
        )
        assert pyg_data.node_class_type.tolist()[:3] == [0, 0, 1]
        assert pyg_data.node_view_type.tolist()[:3] == [ab.VIEW_COARSE, ab.VIEW_FINE, ab.VIEW_COARSE]
        assert pyg_data.edge_relation_mask.shape == (edge_attr.shape[0], ab.EDGE_RELATION_DIM)

    _, _, _, (_, scaled_edges, _, _, _) = build(scale=2.0)
    assert set(edge_map(edge_index, edge_attr)) == set(tuple(pair) for pair in scaled_edges.t().tolist())

    model = ab.SizeAwareGraphGNN(
        ab.SIZE_AWARE_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        hidden_dim=32,
        num_layers=2,
        output_dim=3,
    )
    logits = model(x, edge_index, edge_attr)
    assert logits.shape == (len(nodes), 3)
    assert torch.isfinite(logits).all()

    for node in nodes:
        node.stage1_obj = 0.5
        node.stage1_small = 0.4
        node.stage1_large = 0.3
        node.stage1_roi = 0.2
    stage_x, stage_edges, stage_attr, stage_targets, _ = ab.build_stage2_size_tensors(
        nodes,
        record,
        graph_args(),
        hypergraph=False,
    )
    assert stage_x.shape == (len(nodes), ab.STAGE2_NODE_DIM)
    assert stage_attr.shape[1] == ab.SIZE_AWARE_EDGE_DIM
    assert stage_targets.shape == (len(nodes), ab.STAGE2_SIZE_TARGET_DIM)
    stage_model = ab.SizeAwareGraphGNN(
        ab.STAGE2_NODE_DIM,
        ab.SIZE_AWARE_EDGE_DIM,
        hidden_dim=32,
        num_layers=2,
        output_dim=ab.STAGE2_SIZE_OUTPUT_DIM,
    )
    stage_logits = stage_model(stage_x, stage_edges, stage_attr)
    assert stage_logits.shape == (len(nodes), ab.STAGE2_SIZE_OUTPUT_DIM)
    assert torch.isfinite(stage_logits).all()

    fixed_predictions = pipeline.run_legacy3head_fixed_candidate_variant(
        [{"record": record, "predictions": predictions, "nodes": nodes}],
        graph_args(),
    )
    assert len(fixed_predictions) == 3, "GNN output must run final NMS after reranking"

    public = ab.add_annotation_ids(predictions)
    assert all(not key.startswith("_") for item in public for key in item)
    print(
        {
            "nodes": len(nodes),
            "directed_edges_with_self": int(edge_index.shape[1]),
            "node_dim": int(x.shape[1]),
            "edge_dim": int(edge_attr.shape[1]),
            "normalized_scale_invariant": True,
            "forward_finite": bool(torch.isfinite(logits).all()),
        }
    )


if __name__ == "__main__":
    main()
