"""Experiment-2 class-relation PPR backbone with a graph PE residual."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
FINAL_DIR = EXPERIMENT_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import graph_positional_encoding as graph_pe
import sparse_ppr_sage as exp2


EXPECTED_BASE_PARAMETERS = 237_087
EXPECTED_FULL_PARAMETERS = 416_710
BASE_ARCHITECTURE = "class_relation_ppr_gatv2_sage"


class Experiment2GraphPEModel(nn.Module):
    """Preserve variant 06 and inject PE before its three PPR graph layers."""

    def __init__(
        self,
        ab,
        mode: str,
        hidden_dim: int = 96,
        num_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.base_input_dim = int(exp2.node_dim(ab))
        self.base_edge_dim = int(exp2.edge_dim(ab))
        self.backbone = exp2.Experiment2GraphRefiner(
            input_dim=self.base_input_dim,
            edge_feature_dim=self.base_edge_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            output_dim=3,
            architecture=BASE_ARCHITECTURE,
            attention_heads=int(attention_heads),
            dropout=float(dropout),
            ppr_index=int(ab.SIZE_AWARE_EDGE_DIM) + exp2.PPR_EXTRA_OFFSET,
            class_offset=int(ab.NODE_CLASS_OFFSET),
            num_classes=int(ab.NUM_CLASSES),
        )
        self.graph_pe = graph_pe.BranchMaskedGraphPE(mode)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        base_x, eigenvectors, eigen_mask, probes = graph_pe.split_augmented_features(
            x,
            self.base_input_dim,
        )
        if edge_attr.ndim != 2 or int(edge_attr.shape[1]) != self.base_edge_dim:
            raise ValueError(
                f"Experiment-2 edge_attr must have {self.base_edge_dim} columns, "
                f"got {tuple(edge_attr.shape)}"
            )
        backbone = self.backbone
        h = backbone.input(base_x)
        class_end = backbone.class_offset + backbone.num_classes
        if class_end > int(base_x.shape[1]):
            raise ValueError(
                f"Experiment-2 class feature slice [{backbone.class_offset}:{class_end}] "
                f"exceeds base input dim {base_x.shape[1]}"
            )
        class_probabilities = base_x[:, backbone.class_offset:class_end].clamp_min(0.0)
        class_probabilities = class_probabilities / class_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        class_ids = class_probabilities.argmax(dim=-1)
        neighbor_context = torch.zeros_like(class_probabilities)
        neighbor_count = torch.zeros(
            (base_x.shape[0], 1),
            dtype=base_x.dtype,
            device=base_x.device,
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
                    torch.ones(
                        (dst.shape[0], 1),
                        dtype=base_x.dtype,
                        device=base_x.device,
                    ),
                )
        neighbor_context = neighbor_context / neighbor_count.clamp_min(1.0)
        context_input = torch.cat(
            [class_probabilities, neighbor_context],
            dim=-1,
        )
        context_update = backbone.class_context(context_input)
        context_gate = backbone.class_context_gate(context_input)
        h = backbone.class_context_norm(h + context_gate * context_update)

        positional_residual = self.graph_pe(
            eigenvectors,
            eigen_mask,
            probes,
            edge_index,
        )
        if positional_residual.shape != h.shape:
            raise RuntimeError(
                f"Experiment-2 PE residual shape {tuple(positional_residual.shape)} "
                f"does not match base hidden state {tuple(h.shape)}"
            )
        h = h + positional_residual
        for layer in backbone.layers:
            h = layer(h, edge_index, edge_attr, class_ids=class_ids)
        return backbone.head(torch.cat([h, base_x], dim=-1))

    def capacity_inventory(self) -> Dict[str, object]:
        base_parameters = sum(
            parameter.numel() for parameter in self.backbone.parameters() if parameter.requires_grad
        )
        pe_inventory = self.graph_pe.parameter_inventory()
        return {
            "base_trainable_parameters": int(base_parameters),
            "pe_trainable_parameters": int(pe_inventory["total"]),
            "total_trainable_parameters": int(base_parameters + pe_inventory["total"]),
            "pe_modules": pe_inventory,
            "pe_loss_path": self.graph_pe.loss_path_inventory(),
            "base_graph_layer_calls": len(self.backbone.layers),
            "pe_operator_calls": self.graph_pe.operation_inventory(),
        }

    def assert_production_contract(self) -> None:
        graph_pe.assert_default_capacity(self.graph_pe)
        inventory = self.capacity_inventory()
        if self.base_input_dim != 40 or self.base_edge_dim != 27:
            raise RuntimeError(
                f"Experiment-2 PE requires node/edge dims 40/27, got "
                f"{self.base_input_dim}/{self.base_edge_dim}"
            )
        if self.backbone.architecture != BASE_ARCHITECTURE:
            raise RuntimeError("Experiment-2 PE backbone architecture changed")
        if len(self.backbone.layers) != 3:
            raise RuntimeError("Experiment-2 PE requires the three-layer base backbone")
        if int(self.backbone.input[0].out_features) != 96:
            raise RuntimeError("Experiment-2 PE requires base hidden width 96")
        if any(int(layer.heads) != 4 for layer in self.backbone.layers):
            raise RuntimeError("Experiment-2 PE requires four attention heads per layer")
        if int(inventory["base_trainable_parameters"]) != EXPECTED_BASE_PARAMETERS:
            raise RuntimeError(
                f"Experiment-2 base capacity mismatch: {inventory}"
            )
        if int(inventory["total_trainable_parameters"]) != EXPECTED_FULL_PARAMETERS:
            raise RuntimeError(
                f"Experiment-2 full capacity mismatch: {inventory}"
            )


def build_model(ab, args, mode: str) -> Experiment2GraphPEModel:
    if int(args.gnn_hidden_dim) != 96 or int(args.gnn_layers) != 3:
        raise RuntimeError(
            "Experiment-2 PE contract fixes the base model at hidden=96, layers=3"
        )
    if int(args.exp2_attention_heads) != 4:
        raise RuntimeError("Experiment-2 PE contract fixes attention_heads=4")
    if abs(float(args.exp2_dropout) - 0.10) > 1e-12:
        raise RuntimeError("Experiment-2 PE contract fixes dropout=0.10")
    model = Experiment2GraphPEModel(
        ab,
        mode,
        hidden_dim=int(args.gnn_hidden_dim),
        num_layers=int(args.gnn_layers),
        attention_heads=int(args.exp2_attention_heads),
        dropout=float(args.exp2_dropout),
    )
    model.assert_production_contract()
    return model
