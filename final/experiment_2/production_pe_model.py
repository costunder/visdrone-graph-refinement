"""H256/L6 E2 backbone + approved H128/L8 PE; no training entrypoint.

See PRODUCTION_PE_CONFORMANCE_SPEC.md. Existing sparse PPR equations are
preserved; chunking does not remove edges or normalize attention per chunk.
"""
import math
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import graph_pe_production as pe
from experiment_2 import sparse_ppr_sage as sparse

EXPECTED = {"no_pe": 2488187, "signnet": 2903172, "rpearl": 2838916}


def chunked_ppr_forward(layer, h, edge_index, edge_attr, class_ids=None):
    if not edge_index.numel():
        return h
    if class_ids is None:
        raise ValueError("Class-relation production layer requires hard class IDs")
    keys = layer.src_projection(h)
    queries = layer.dst_projection(h)
    values = layer.value_projection(h)
    blocks = []
    for start in range(0, edge_index.shape[1], layer.production_edge_chunk):
        src, dst = edge_index[:, start:start + layer.production_edge_chunk]
        attrs = edge_attr[start:start + layer.production_edge_chunk]
        relation = class_ids[src] * layer.num_classes + class_ids[dst]
        pair = (keys[src].view(-1, layer.heads, layer.head_dim)
                + queries[dst].view(-1, layer.heads, layer.head_dim)
                + layer.edge_projection(attrs).view(-1, layer.heads, layer.head_dim))
        pair = pair + layer.class_pair_key(relation).view(-1, layer.heads, layer.head_dim)
        score = (F.leaky_relu(pair, negative_slope=.2) * layer.attention[None]).sum(-1) / math.sqrt(layer.head_dim)
        score = score + layer.class_pair_bias(relation)
        prior = torch.log1p(attrs[:, layer.ppr_index].clamp_min(0) / 1e-6)[:, None]
        blocks.append(score + F.softplus(layer.ppr_beta_raw)[None] * prior)
    coefficients = layer._segment_softmax(torch.cat(blocks), edge_index[1], len(h))
    aggregate = h.new_zeros((len(h), layer.heads, layer.head_dim))
    for start in range(0, edge_index.shape[1], layer.production_edge_chunk):
        src, dst = edge_index[:, start:start + layer.production_edge_chunk]
        relation = class_ids[src] * layer.num_classes + class_ids[dst]
        message = (values[src] + layer.class_pair_value(relation)).view(-1, layer.heads, layer.head_dim)
        aggregate.index_add_(0, dst, message * coefficients[start:start + layer.production_edge_chunk, :, None])
    update = F.relu(layer.update(torch.cat((h, aggregate.reshape(len(h), -1)), -1)))
    return layer.norm(h + layer.dropout(update))


class Experiment2Production(nn.Module):
    def __init__(self, ab, mode, seed=42, *, recompute=True, edge_chunk=4096, signal_chunk=8, node_chunk=256):
        super().__init__()
        if mode not in EXPECTED or seed not in (42, 43, 44):
            raise ValueError("Unknown approved mode/seed")
        if (sparse.node_dim(ab), sparse.edge_dim(ab), ab.NUM_CLASSES) != (40, 27, 10):
            raise ValueError("Expected VisDrone10 node40/edge27")
        if min(edge_chunk, signal_chunk, node_chunk) < 1:
            raise ValueError("Chunk sizes must be positive")
        self.mode, self.seed, self.alignment = mode, seed, "aligned"
        self.recompute, self.signal_chunk, self.node_chunk = recompute, signal_chunk, node_chunk
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(pe.stable_seed("backbone", seed=seed, experiment="experiment_2"))
            self.backbone = sparse.build_model(ab, SimpleNamespace(gnn_hidden_dim=256, gnn_layers=6,
                                                exp2_attention_heads=4, exp2_dropout=.10),
                                               "class_relation_ppr_gatv2_sage")
        for layer in self.backbone.layers:
            layer.production_edge_chunk = edge_chunk
            layer.forward = MethodType(chunked_ppr_forward, layer)
        if mode != "no_pe":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(pe.stable_seed(f"{mode}_init", seed=seed, experiment="experiment_2"))
                self.encoder = pe.SignNetK32() if mode == "signnet" else pe.RPEARLM120()
                self.projection = nn.Linear(128, 256, bias=False)
            self.projection_norm = nn.LayerNorm(256, elementwise_affine=False, eps=1e-5)
            self.gate = nn.Parameter(torch.zeros(()))
        self.assert_contract()

    def forward(self, x, edge_index, edge_attr, *, support=None, u=None, mask=None, w=None):
        if x.ndim != 2 or x.shape[1] != 40 or edge_attr.shape != (edge_index.shape[1], 27):
            raise ValueError("Expected X[N,40], edge_attr[E,27]")
        if edge_index.dtype != torch.long or edge_index.shape[0] != 2:
            raise ValueError("Expected int64 edge_index[2,E]")
        backbone = self.backbone
        h = backbone.input(x)
        classes = x[:, backbone.class_offset:backbone.class_offset + 10].clamp_min(0)
        classes = classes / classes.sum(-1, keepdim=True).clamp_min(1e-12)
        ids = classes.argmax(-1)
        context = torch.zeros_like(classes)
        count = x.new_zeros((len(x), 1))
        if edge_index.numel():
            src, dst = edge_index
            keep = src != dst
            src, dst = src[keep], dst[keep]
            context.index_add_(0, dst, classes[src])
            count.index_add_(0, dst, x.new_ones((len(dst), 1)))
        context = torch.cat((classes, context / count.clamp_min(1)), -1)
        h = backbone.class_context_norm(h + backbone.class_context_gate(context) * backbone.class_context(context))
        if self.mode != "no_pe":
            if support is None:
                raise ValueError("Explicit PE support required")
            if self.mode == "signnet":
                if u is None or mask is None:
                    raise ValueError("SignNet eigenvectors/mask required")
                encoded = self.encoder(u, mask, support, self.signal_chunk, self.node_chunk, self.recompute)
            else:
                if w is None:
                    raise ValueError("R-PEARL probes required")
                encoded = self.encoder(w, support, self.signal_chunk, self.recompute)
            h = h + self.gate.tanh() * self.projection_norm(self.projection(encoded))
        for layer in backbone.layers:
            h = (checkpoint(layer, h, edge_index, edge_attr, ids, use_reentrant=False)
                 if self.recompute and torch.is_grad_enabled() else layer(h, edge_index, edge_attr, ids))
        return backbone.head(torch.cat((h, x), -1))

    def assert_contract(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count != EXPECTED[self.mode] or len(self.backbone.layers) != 6:
            raise RuntimeError(f"E2 production capacity mismatch: {self.mode}={count}")
        if self.backbone.input[0].out_features != 256 or any(layer.heads != 4 for layer in self.backbone.layers):
            raise RuntimeError("E2 requires H256/L6/4 heads")
        if self.mode != "no_pe" and len({layer.mlp[0].weight.data_ptr() for layer in self.encoder.layers}) != 8:
            raise RuntimeError("PE must have eight independently parameterized layers")
        return count
