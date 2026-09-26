"""Experiment-1 fixed-pool backbone with a branch-masked graph PE residual."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn


FINAL_DIR = Path(__file__).resolve().parent.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

import graph_positional_encoding as graph_pe


EXPECTED_BASE_PARAMETERS = 449_187
EXPECTED_FULL_PARAMETERS = 628_810


class Experiment1GraphPEModel(nn.Module):
    """Preserve `02_gnn_no_cluster` and inject PE before its graph layers."""

    def __init__(
        self,
        ab,
        mode: str,
        hidden_dim: int = 96,
        num_layers: int = 3,
        output_dim: int = 3,
    ):
        super().__init__()
        self.ab = ab
        self.base_input_dim = int(ab.SIZE_AWARE_NODE_DIM)
        self.base_edge_dim = int(ab.SIZE_AWARE_EDGE_DIM)
        self.backbone = ab.SizeAwareGraphGNN(
            self.base_input_dim,
            self.base_edge_dim,
            int(hidden_dim),
            int(num_layers),
            output_dim=int(output_dim),
        )
        self.graph_pe = graph_pe.BranchMaskedGraphPE(mode)
        self.size_aware_output_dim = int(output_dim)

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
                f"Experiment-1 edge_attr must have {self.base_edge_dim} columns, "
                f"got {tuple(edge_attr.shape)}"
            )
        h = self.backbone.input(base_x)
        class_end = self.ab.NODE_CLASS_OFFSET + self.ab.NUM_CLASSES
        if int(base_x.shape[1]) < int(class_end):
            raise ValueError(
                f"Experiment-1 class feature slice ends at {class_end}, "
                f"but base input has {base_x.shape[1]} columns"
            )
        node_types = torch.cat(
            [
                base_x[
                    :,
                    self.ab.NODE_VIEW_TYPE_OFFSET : self.ab.NODE_VIEW_GEOMETRY_OFFSET,
                ],
                base_x[:, self.ab.NODE_CLASS_OFFSET:class_end],
            ],
            dim=-1,
        )
        h = self.backbone.node_type_norm(
            h + self.backbone.node_type_projection(node_types)
        )
        positional_residual = self.graph_pe(
            eigenvectors,
            eigen_mask,
            probes,
            edge_index,
        )
        if positional_residual.shape != h.shape:
            raise RuntimeError(
                f"Experiment-1 PE residual shape {tuple(positional_residual.shape)} "
                f"does not match base hidden state {tuple(h.shape)}"
            )
        h = h + positional_residual
        for layer in self.backbone.layers:
            h = layer(h, edge_index, edge_attr)
        return self.backbone.head(torch.cat([h, base_x], dim=-1))

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
        if self.base_input_dim != 37 or self.base_edge_dim != 24:
            raise RuntimeError(
                f"Experiment-1 PE requires node/edge dims 37/24, got "
                f"{self.base_input_dim}/{self.base_edge_dim}"
            )
        if len(self.backbone.layers) != 3:
            raise RuntimeError("Experiment-1 PE requires the three-layer base backbone")
        if int(self.backbone.input[0].out_features) != 96:
            raise RuntimeError("Experiment-1 PE requires base hidden width 96")
        if int(inventory["base_trainable_parameters"]) != EXPECTED_BASE_PARAMETERS:
            raise RuntimeError(
                f"Experiment-1 base capacity mismatch: {inventory}"
            )
        if int(inventory["total_trainable_parameters"]) != EXPECTED_FULL_PARAMETERS:
            raise RuntimeError(
                f"Experiment-1 full capacity mismatch: {inventory}"
            )


def build_model(ab, args, mode: str) -> Experiment1GraphPEModel:
    if int(args.gnn_hidden_dim) != 96 or int(args.gnn_layers) != 3:
        raise RuntimeError(
            "Experiment-1 PE contract fixes the base model at hidden=96, layers=3"
        )
    model = Experiment1GraphPEModel(
        ab,
        mode,
        hidden_dim=int(args.gnn_hidden_dim),
        num_layers=int(args.gnn_layers),
        output_dim=3,
    )
    model.assert_production_contract()
    return model
