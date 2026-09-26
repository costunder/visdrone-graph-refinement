#!/usr/bin/env python3
"""Sequential experiment-1 prior -> experiment-2 rescue components.

The experiment-1 graph is evaluated only on the high-confidence core of one
shared raw low-confidence candidate pool.  Its frozen probabilities are
attached to the corresponding nodes in the full pool.  Experiment 2 then
performs sparse PPR/class-relation refinement and the only final NMS.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
FINAL_DIR = EXPERIMENT_DIR.parent
EXP1_SCRIPT_DIR = FINAL_DIR / "experiment_1/scripts"
EXP2_DIR = FINAL_DIR / "experiment_2"

sys.path.insert(0, str(EXP1_SCRIPT_DIR))
import run_gois_paper_ablation_curves as base  # noqa: E402
import run_gois_two_stage_gnn_ablation as ab  # noqa: E402

sys.path.insert(0, str(EXP2_DIR))
import sparse_ppr_sage as exp2  # noqa: E402


GRAPH_SCHEMA_VERSION = 1
PRIOR_NAMES = (
    "exp1_core_mask",
    "exp1_object_probability",
    "exp1_small_probability",
    "exp1_large_probability",
    "exp1_roi_probability",
)
PRIOR_DIM = len(PRIOR_NAMES)
PRIOR_ZERO = "zero"
PRIOR_EXP1 = "exp1"


def tag_shared_candidate_cache(records, coarse_cache, fine_cache):
    """Combine raw coarse/fine caches while assigning one stable node index."""

    combined = {}
    for record in records:
        image_id = str(record.image_id)
        predictions = []
        for source, source_predictions in (
            ("coarse", coarse_cache.get(image_id, [])),
            ("fine", fine_cache.get(image_id, [])),
        ):
            for prediction in source_predictions:
                tagged = dict(prediction)
                tagged["_candidate_source"] = source
                tagged["_exp3_full_index"] = len(predictions)
                predictions.append(tagged)
        combined[image_id] = predictions
    return combined


def core_candidate_cache(records, shared_cache, core_confidence):
    """Take the experiment-1 core as a subset of the shared raw pool."""

    threshold = float(core_confidence)
    return {
        str(record.image_id): [
            dict(prediction)
            for prediction in shared_cache.get(str(record.image_id), [])
            if float(prediction.get("score", 0.0)) >= threshold
        ]
        for record in records
    }


def build_samples(records, gt_by_image, candidate_cache, args):
    samples = ab.make_size_aware_detection_samples(
        records,
        {},
        candidate_cache,
        gt_by_image,
        args,
    )
    for sample in samples:
        sample["gt_boxes"] = list(
            gt_by_image.get(sample["record"].image_id, [])
        )
    return samples


def initialize_exp1_priors(samples):
    for sample in samples:
        for node in sample.get("nodes", []):
            node.exp1_core_mask = 0.0
            node.exp1_obj = 0.0
            node.exp1_small = 0.0
            node.exp1_large = 0.0
            node.exp1_roi = 0.0


def transfer_exp1_priors(core_samples, full_samples):
    """Map frozen experiment-1 node probabilities back to the full pool."""

    initialize_exp1_priors(full_samples)
    full_by_image = {
        int(sample["record"].image_id): sample for sample in full_samples
    }
    transferred = 0
    core_nodes = 0
    seen = set()

    for core_sample in core_samples:
        image_id = int(core_sample["record"].image_id)
        if image_id not in full_by_image:
            raise ValueError(f"Missing full-pool sample for image {image_id}")
        full_sample = full_by_image[image_id]
        full_nodes = {
            int(node.det_index): node
            for node in full_sample.get("nodes", [])
            if int(node.source) == int(ab.SOURCE_SLICE)
        }
        core_predictions = core_sample.get("predictions", [])
        for core_node in core_sample.get("nodes", []):
            if int(core_node.source) != int(ab.SOURCE_SLICE):
                continue
            core_nodes += 1
            core_index = int(core_node.det_index)
            if not 0 <= core_index < len(core_predictions):
                raise IndexError(
                    f"Core node index {core_index} is outside image {image_id} cache"
                )
            prediction = core_predictions[core_index]
            if "_exp3_full_index" not in prediction:
                raise KeyError("Core candidate is missing _exp3_full_index")
            full_index = int(prediction["_exp3_full_index"])
            identity = (image_id, full_index)
            if identity in seen:
                raise ValueError(f"Duplicate core mapping for {identity}")
            seen.add(identity)
            if full_index not in full_nodes:
                raise KeyError(
                    f"Full-pool node {full_index} is missing for image {image_id}"
                )
            full_node = full_nodes[full_index]
            full_node.exp1_core_mask = 1.0
            full_node.exp1_obj = float(core_node.gnn_obj)
            full_node.exp1_small = float(core_node.gnn_small)
            full_node.exp1_large = float(core_node.gnn_large)
            full_node.exp1_roi = float(core_node.gnn_roi)
            transferred += 1

    if transferred != core_nodes:
        raise RuntimeError(
            f"Transferred {transferred} experiment-1 priors for {core_nodes} core nodes"
        )
    return {
        "core_nodes": core_nodes,
        "transferred_nodes": transferred,
        "full_nodes": sum(len(sample.get("nodes", [])) for sample in full_samples),
    }


def score_and_transfer_exp1(core_samples, full_samples, model, args, device):
    base.attach_size_aware_scores_for_eval(
        core_samples,
        model,
        args,
        device,
        assign_fn=base.assign_legacy3head_probs,
    )
    return transfer_exp1_priors(core_samples, full_samples)


def prior_feature_matrix(nodes, mode):
    if mode not in {PRIOR_ZERO, PRIOR_EXP1}:
        raise ValueError(f"Unknown experiment-3 prior mode: {mode}")
    if not nodes:
        return torch.empty((0, PRIOR_DIM), dtype=torch.float32)
    if mode == PRIOR_ZERO:
        return torch.zeros((len(nodes), PRIOR_DIM), dtype=torch.float32)
    return torch.tensor(
        [
            [
                float(getattr(node, "exp1_core_mask", 0.0)),
                float(getattr(node, "exp1_obj", 0.0)),
                float(getattr(node, "exp1_small", 0.0)),
                float(getattr(node, "exp1_large", 0.0)),
                float(getattr(node, "exp1_roi", 0.0)),
            ]
            for node in nodes
        ],
        dtype=torch.float32,
    )


def append_prior_features(base_features, nodes, mode):
    priors = prior_feature_matrix(nodes, mode)
    if base_features.shape[0] != priors.shape[0]:
        raise ValueError(
            f"Base/prior node count mismatch: {base_features.shape[0]} vs {priors.shape[0]}"
        )
    return torch.cat([base_features, priors], dim=-1)


def node_dim():
    return int(exp2.node_dim(ab)) + PRIOR_DIM


def edge_dim():
    return int(exp2.edge_dim(ab))


def build_sequential_tensors(sample, args, prior_mode):
    x, edge_index, edge_attr, targets, weights = exp2.build_sparse_graph_tensors(
        ab,
        sample,
        args,
        include_ppr=True,
    )
    x = append_prior_features(x, sample.get("nodes", []), prior_mode)
    if x.shape[1] != node_dim():
        raise RuntimeError(f"Sequential node dim {x.shape[1]} != {node_dim()}")
    sample[f"_exp3_graph_stats_{prior_mode}"] = {
        "nodes": int(x.shape[0]),
        "core_nodes": int(
            sum(
                float(getattr(node, "exp1_core_mask", 0.0)) > 0.5
                for node in sample.get("nodes", [])
            )
        ),
        "base_node_dim": int(exp2.node_dim(ab)),
        "prior_dim": PRIOR_DIM,
        "edge_dim": edge_dim(),
        "prior_mode": prior_mode,
    }
    return x, edge_index, edge_attr, targets, weights


def build_stage2_model(args):
    return exp2.Experiment2GraphRefiner(
        input_dim=node_dim(),
        edge_feature_dim=edge_dim(),
        hidden_dim=int(args.gnn_hidden_dim),
        num_layers=int(args.gnn_layers),
        output_dim=3,
        architecture="class_relation_ppr_gatv2_sage",
        attention_heads=int(getattr(args, "exp2_attention_heads", 4)),
        dropout=float(getattr(args, "exp2_dropout", 0.10)),
        ppr_index=int(ab.SIZE_AWARE_EDGE_DIM) + exp2.PPR_EXTRA_OFFSET,
        class_offset=int(ab.NODE_CLASS_OFFSET),
        num_classes=int(ab.NUM_CLASSES),
    )


def assign_stage2_probabilities(node, probability):
    node.gnn_obj = float(probability[0])
    node.gnn_small = float(probability[1])
    node.gnn_large = float(probability[2])
    node.gnn_roi = max(
        node.gnn_small,
        0.7 * node.gnn_obj + 0.3 * node.gnn_small,
    )
    node.stage2_prob = max(node.gnn_obj, node.gnn_small, node.gnn_large)
    node.stage2_roi_prob = node.gnn_roi


def refined_predictions(samples, args):
    """Use the experiment-2 scoring contract and perform the only final NMS."""

    predictions = []
    threshold = float(args.stage1_keep_conf)
    alpha = min(1.0, max(0.0, float(args.gnn_score_alpha)))
    for sample in samples:
        image_predictions = []
        for node in sample.get("nodes", []):
            source = (
                sample.get("full_predictions", [])
                if int(node.source) == int(ab.SOURCE_FULL)
                else sample.get("predictions", [])
            )
            index = int(node.det_index)
            if not 0 <= index < len(source):
                continue
            prediction = source[index]
            detector_score = float(prediction.get("score", 0.0))
            size_score = (
                node.gnn_large
                if node.union_area >= ab.SMALL_MEDIUM_AREA_THR
                else node.gnn_small
            )
            graph_score = max(float(node.gnn_obj), float(size_score))
            score = (1.0 - alpha) * detector_score + alpha * graph_score
            if score < threshold:
                continue
            adjusted = {
                key: value
                for key, value in prediction.items()
                if not str(key).startswith("_")
            }
            adjusted["score"] = max(0.0, min(1.0, float(score)))
            image_predictions.append(adjusted)
        predictions.extend(
            ab.classwise_nms(
                image_predictions,
                iou_threshold=float(args.final_nms_iou),
                limit=int(args.max_det),
            )
        )
    return predictions


def model_capacity(model):
    return {
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "node_dim": node_dim(),
        "edge_dim": edge_dim(),
        "prior_dim": PRIOR_DIM,
    }
