"""Approved pe_production_v1 operators; no legacy PE modules or fallback sizes.

Project-specific SignNet-k32 / R-PEARL-M120 adaptations, not exact paper
reproductions. See GRAPH_PE_PRODUCTION_DESIGN_PROPOSAL.md for provenance.
All graph aggregation is sparse. Attention is over 32 signals within a node.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import graph_positional_encoding as legacy_ops

CODE_VERSION = 1
RAW_SCHEMA = "graph_pe_production_raw_v1"
CHECKPOINT_SCHEMA = "graph_pe_production_training_v1"
EXPECTED_E1 = {"no_pe": 6073347, "signnet": 6488332, "rpearl": 6424076}
EIGEN_CONFIG = legacy_ops.GraphPEConfig(
    eigenvectors=32, random_samples=120, filter_order=11, hidden_dim=128, gin_layers=8,
)


def stable_seed(namespace, **key):
    payload = json.dumps({"namespace": namespace, **key}, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") & (2**63 - 1)


def probes(num_nodes, *, seed, experiment, split, image_id, epoch=None, samples=120):
    key = dict(seed=seed, experiment=experiment, split=split, image_id=image_id)
    key["epoch" if split == "train" else "evaluation"] = epoch if split == "train" else "eval_probe_v1"
    if split == "train" and (epoch is None or epoch < 1):
        raise ValueError("Training probes require a positive epoch")
    generator = torch.Generator(device="cpu").manual_seed(stable_seed("probe", **key))
    return torch.randn(num_nodes, samples, generator=generator, dtype=torch.float32)


def alignment_permutation(node_identity, *, seed, experiment, split, image_id):
    identity = node_identity.detach().cpu().tolist()
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(
        "alignment", seed=seed, experiment=experiment, split=split,
        image_id=image_id, node_identity=identity))
    n = len(identity)
    if n < 2:
        return torch.arange(n)
    while True:
        permutation = torch.randperm(n, generator=generator)
        if not torch.any(permutation == torch.arange(n)):
            return permutation


def adjacency(support, n, dtype, *, normalized=False):
    src, dst = support
    values = torch.ones(src.numel(), device=support.device, dtype=dtype)
    if normalized and src.numel():
        degree = torch.zeros(n, dtype=dtype, device=support.device).index_add_(0, dst, values)
        inverse = degree.clamp_min(1).rsqrt()
        values = inverse[src] * inverse[dst]
    return torch.sparse_coo_tensor(torch.stack((dst, src)), values, (n, n)).coalesce()


def sparse_apply(matrix, signal):
    if signal.shape[0] == 0:
        return torch.zeros_like(signal)
    return torch.sparse.mm(matrix, signal.reshape(signal.shape[0], -1)).reshape_as(signal)


def eigen_features(edge_index, num_nodes):
    """Existing solver and node-order convention; additionally audit eigenvalues."""
    u, mask, audit = legacy_ops.laplacian_eigenvectors(edge_index.cpu(), num_nodes, EIGEN_CONFIG)
    support = legacy_ops.coalesced_undirected_support(edge_index.cpu(), num_nodes)
    normalized = adjacency(support, num_nodes, torch.float64, normalized=True)
    double_u = u.double()
    rayleigh = (double_u * (double_u - sparse_apply(normalized, double_u))).sum(0)
    count = int(audit["valid_eigenvectors"])
    audit.update(schema=RAW_SCHEMA, eigenvalues_rayleigh=rayleigh[:count].tolist(),
                 eigenvalue_audit="Rayleigh quotient of returned float32 vectors",
                 repeated_eigenspace_rotation_invariance=False)
    return u, mask.bool(), support, audit


class SparseResidualGIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.norm = nn.LayerNorm(128, eps=1e-5)

    def forward(self, h, support_matrix):
        return self.norm(h + self.mlp((1 + self.eps) * h + sparse_apply(support_matrix, h)))


class SignalSetTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.q = nn.Linear(128, 128, bias=False)
        self.k = nn.Linear(128, 128, bias=False)
        self.v = nn.Linear(128, 128, bias=False)
        self.out = nn.Linear(128, 128, bias=False)
        self.attention_norm = nn.LayerNorm(128, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.ffn_norm = nn.LayerNorm(128, eps=1e-6)

    def forward(self, h, mask):
        n, signals, _ = h.shape
        if not n:
            return h
        q, k, v = [projection(h).reshape(n, signals, 4, 32).transpose(1, 2)
                   for projection in (self.q, self.k, self.v)]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(32)
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        # An all-padding row is not allowed to softmax(-inf, ..., -inf).
        scores = torch.where(mask.any(-1)[:, None, None, None], scores, torch.zeros_like(scores))
        coefficients = scores.softmax(-1) * mask[:, None, None, :]
        attended = (coefficients @ v).transpose(1, 2).reshape(n, signals, 128)
        h = self.attention_norm(h + self.out(attended))
        h = self.ffn_norm(h + self.ffn(h))
        return h * mask.unsqueeze(-1)


class SignNetK32(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1, 128)
        self.layers = nn.ModuleList(SparseResidualGIN() for _ in range(8))
        self.rho = SignalSetTransformer()
        self.output = nn.Sequential(nn.Linear(128, 128, bias=False), nn.LayerNorm(128, eps=1e-5))

    def phi(self, u, mask, matrix):
        h = F.relu(self.input(u.unsqueeze(-1))) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, matrix) * mask.unsqueeze(-1)
        return h

    def symmetric_phi(self, u, mask, matrix):
        return self.phi(u, mask, matrix) + self.phi(-u, mask, matrix)

    def forward(self, u, mask, support, signal_chunk=8, node_chunk=256, recompute=True):
        if u.ndim != 2 or u.shape[1] != 32 or mask.shape != u.shape or mask.dtype != torch.bool:
            raise ValueError("SignNet requires float U[N,32] and bool mask[N,32]")
        n = u.shape[0]
        if n == 0:
            return u.new_empty((0, 128))
        matrix = adjacency(support, n, u.dtype)
        encoded = []
        for start in range(0, 32, signal_chunk):
            values = (u[:, start:start + signal_chunk], mask[:, start:start + signal_chunk], matrix)
            h = checkpoint(self.symmetric_phi, *values, use_reentrant=False) if recompute and torch.is_grad_enabled() else self.symmetric_phi(*values)
            encoded.append(h)
        h = torch.cat(encoded, dim=1)
        output = []
        for start in range(0, n, node_chunk):
            selected = mask[start:start + node_chunk]
            mixed = self.rho(h[start:start + node_chunk], selected)
            output.append(self.output(mixed.sum(1)) * selected.any(1, keepdim=True))
        return torch.cat(output, dim=0)


class RPEARLM120(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128), nn.LayerNorm(128, eps=1e-5))
        self.layers = nn.ModuleList(SparseResidualGIN() for _ in range(8))
        self.rho = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.LayerNorm(128, eps=1e-5))

    def sample_sum(self, w, matrix, normalized):
        powers = [w]
        for _ in range(11):
            powers.append(sparse_apply(normalized, powers[-1]))
        h = self.filter(torch.stack(powers, dim=-1))
        for layer in self.layers:
            h = layer(h, matrix)
        return self.rho(h).sum(1)  # rho BEFORE sample pooling, never after it

    def forward(self, w, support, signal_chunk=8, recompute=True, variance_diagnostic=False):
        allowed = (120, 240, 480) if variance_diagnostic else (120,)
        if w.ndim != 2 or w.shape[1] not in allowed:
            raise ValueError("Production R-PEARL requires all 120 random probes")
        n = w.shape[0]
        if not n:
            return w.new_empty((0, 128))
        matrix = adjacency(support, n, w.dtype)
        normalized = adjacency(support, n, w.dtype, normalized=True)
        total = w.new_zeros((n, 128))
        for start in range(0, w.shape[1], signal_chunk):
            values = (w[:, start:start + signal_chunk], matrix, normalized)
            result = checkpoint(self.sample_sum, *values, use_reentrant=False) if recompute and torch.is_grad_enabled() else self.sample_sum(*values)
            total = total + result
        return total / w.shape[1]


def edge_gated_chunk_forward(layer, h, edge_index, edge_attr):
    """Same EdgeGatedLayer parameters and equation, bounded E*8*H activation."""
    if not edge_index.numel():
        return layer.norm(F.relu(layer.self_linear(h)) + h)
    aggregate = torch.zeros_like(h)
    degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
    for start in range(0, edge_index.shape[1], layer.production_edge_chunk):
        src, dst = edge_index[:, start:start + layer.production_edge_chunk]
        attrs = edge_attr[start:start + layer.production_edge_chunk]
        pair = torch.cat((h[src], h[dst], attrs), dim=-1)
        messages = layer.message(pair)
        weights = attrs[:, 16:16 + layer.relation_count].clamp_min(0)
        typed = torch.stack([transform(h[src]) for transform in layer.relation_messages], dim=1)
        typed = (typed * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        messages = (messages + typed) * layer.gate(pair)
        aggregate.index_add_(0, dst, messages)
        degree.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=h.dtype, device=h.device))
    return layer.norm(F.relu(layer.self_linear(h) + aggregate / degree.clamp_min(1)) + h)


class Experiment1Production(nn.Module):
    """Only full H256/L6 main architectures; no hidden-width CLI fallback."""
    def __init__(self, ab, mode, seed=42, alignment="aligned", *, recompute=True,
                 signal_chunk=8, edge_chunk=4096, node_chunk=256):
        super().__init__()
        if mode not in EXPECTED_E1 or seed not in (42, 43, 44):
            raise ValueError("Unknown approved production mode or seed")
        if alignment not in ("aligned", "node_permuted_control") or (mode == "no_pe" and alignment != "aligned"):
            raise ValueError("Invalid alignment control")
        if min(signal_chunk, edge_chunk, node_chunk) < 1:
            raise ValueError("Chunk sizes must be positive; they never change model capacity")
        if (ab.SIZE_AWARE_NODE_DIM, ab.SIZE_AWARE_EDGE_DIM, ab.NUM_CLASSES) != (37, 24, 10):
            raise RuntimeError("Experiment-1 requires VisDrone10 / node37 / edge24")
        self.mode, self.seed, self.alignment = mode, seed, alignment
        self.recompute, self.signal_chunk = recompute, signal_chunk
        self.edge_chunk, self.node_chunk = edge_chunk, node_chunk
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(stable_seed("backbone", seed=seed, experiment="experiment_1"))
            self.backbone = ab.SizeAwareGraphGNN(37, 24, 256, 6, output_dim=3)
        for layer in self.backbone.layers:
            layer.production_edge_chunk = edge_chunk
            layer.forward = MethodType(edge_gated_chunk_forward, layer)
        if mode != "no_pe":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(stable_seed(f"{mode}_init", seed=seed, experiment="experiment_1"))
                self.encoder = SignNetK32() if mode == "signnet" else RPEARLM120()
                self.projection = nn.Linear(128, 256, bias=False)
            self.projection_norm = nn.LayerNorm(256, elementwise_affine=False, eps=1e-5)
            self.gate = nn.Parameter(torch.zeros(()))
        self.assert_contract()

    def forward(self, x, edge_index, edge_attr, *, support=None, u=None, mask=None, w=None, permutation=None):
        if x.ndim != 2 or x.shape[1] != 37 or edge_attr.shape != (edge_index.shape[1], 24):
            raise ValueError("Production E1 base tensor shape mismatch")
        if edge_index.dtype != torch.long or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be int64[2,E]")
        h = self.backbone.input(x)
        h = self.backbone.node_type_norm(h + self.backbone.node_type_projection(torch.cat((x[:, 16:21], x[:, 27:37]), -1)))
        if self.mode != "no_pe":
            if support is None:
                raise ValueError("Explicit PE support required; no graph fallback")
            if self.mode == "signnet":
                if u is None or mask is None:
                    raise ValueError("SignNet raw tensors missing")
                pe = self.encoder(u, mask, support, self.signal_chunk, self.node_chunk, self.recompute)
            else:
                if w is None:
                    raise ValueError("R-PEARL probes missing")
                pe = self.encoder(w, support, self.signal_chunk, self.recompute)
            if self.alignment == "node_permuted_control":
                if permutation is None or permutation.shape != (len(x),):
                    raise ValueError("Control requires explicit node permutation")
                pe = pe[permutation]
            elif permutation is not None:
                raise ValueError("Aligned treatment must not permute PE nodes")
            h = h + self.gate.tanh() * self.projection_norm(self.projection(pe))
        for layer in self.backbone.layers:
            h = checkpoint(layer, h, edge_index, edge_attr, use_reentrant=False) if self.recompute and torch.is_grad_enabled() else layer(h, edge_index, edge_attr)
        return self.backbone.head(torch.cat((h, x), -1))

    def assert_contract(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if count != EXPECTED_E1[self.mode] or len(self.backbone.layers) != 6 or self.backbone.input[0].out_features != 256:
            raise RuntimeError(f"Constructed model does not conform: {self.mode}, {count}")
        if self.mode == "no_pe":
            if hasattr(self, "encoder") or hasattr(self, "gate"):
                raise RuntimeError("No-PE must not register inactive PE modules")
        else:
            if len(self.encoder.layers) != 8 or len({p.mlp[0].weight.data_ptr() for p in self.encoder.layers}) != 8:
                raise RuntimeError("PE requires eight distinct parameterized GIN layers")
            if self.encoder.layers[0].mlp[0].in_features != 128:
                raise RuntimeError("PE hidden width must be 128")
        return count

    def capacity_inventory(self):
        return {"mode": self.mode, "alignment": self.alignment, "seed": self.seed,
                "backbone_hidden": 256, "backbone_layers": 6,
                "pe_hidden": None if self.mode == "no_pe" else 128,
                "pe_gin_layers": 0 if self.mode == "no_pe" else 8,
                "total_trainable_parameters": self.assert_contract(),
                "named_parameters": {name: {"shape": list(p.shape), "numel": p.numel(), "trainable": p.requires_grad}
                                     for name, p in self.named_parameters()},
                "claim": "Registered capacity; observed gradients and relation activation audited separately"}


def bce_numerator(logits, targets, weights, pos_weight):
    per_entry = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    return (per_entry * weights[:, None] * logits.new_tensor([1.0, 2.0, 1.5])).sum()


def virtual_batch_statistics(graphs, global_pos_weight=None):
    total_weight = sum(float(graph["weights"].double().sum()) for graph in graphs)
    if total_weight <= 0:
        raise ValueError("Nonempty virtual batch requires positive total sample weight")
    if global_pos_weight is None:
        positive = sum((graph["targets"] > 0.5).sum(0) for graph in graphs).float()
        total = sum(len(graph["targets"]) for graph in graphs)
        global_pos_weight = ((total - positive).clamp_min(1) / positive.clamp_min(1)).clamp(1, 25)
    return total_weight, global_pos_weight
