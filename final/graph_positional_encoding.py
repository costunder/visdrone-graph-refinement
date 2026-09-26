"""Branch-masked SignNet/R-PEARL adaptations for graph PE ablations.

This module is deliberately independent from the experiment runners.  It does
not load detector data, start training, or write result artifacts.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, Hashable, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    import scipy.sparse as scipy_sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh
except Exception as scipy_import_error:  # pragma: no cover - exercised by fail-closed path
    scipy_sparse = None
    connected_components = None
    eigsh = None
    ArpackNoConvergence = RuntimeError
    SCIPY_IMPORT_ERROR = scipy_import_error
else:
    SCIPY_IMPORT_ERROR = None


GRAPH_PE_SCHEMA_VERSION = 1
GRAPH_PE_CODE_VERSION = 2
SIGNNET_MODE = "signnet"
RPEARL_MODE = "rpearl"
ZERO_MODE = "zero"
PE_MODES = (ZERO_MODE, SIGNNET_MODE, RPEARL_MODE)


@dataclass(frozen=True)
class GraphPEConfig:
    eigenvectors: int = 8
    random_samples: int = 16
    filter_order: int = 8
    hidden_dim: int = 96
    gin_layers: int = 3
    dense_eigh_max_component_nodes: int = 64
    eigenvalue_tolerance: float = 1e-6
    eigsh_tolerance: float = 1e-6
    eigsh_max_iterations_per_node: int = 20

    @property
    def raw_dim(self) -> int:
        return 2 * int(self.eigenvectors) + int(self.random_samples)

    def validate(self) -> None:
        integer_fields = {
            "eigenvectors": self.eigenvectors,
            "random_samples": self.random_samples,
            "filter_order": self.filter_order,
            "hidden_dim": self.hidden_dim,
            "gin_layers": self.gin_layers,
            "dense_eigh_max_component_nodes": self.dense_eigh_max_component_nodes,
            "eigsh_max_iterations_per_node": self.eigsh_max_iterations_per_node,
        }
        invalid = [name for name, value in integer_fields.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"Graph PE integer values must be positive: {', '.join(invalid)}")
        if float(self.eigenvalue_tolerance) <= 0 or float(self.eigsh_tolerance) <= 0:
            raise ValueError("Graph PE eigenvalue tolerances must be positive")


DEFAULT_GRAPH_PE_CONFIG = GraphPEConfig()
DEFAULT_GRAPH_PE_CONFIG.validate()


def config_dict(config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG) -> Dict[str, object]:
    config.validate()
    return {
        "schema_version": GRAPH_PE_SCHEMA_VERSION,
        "code_version": GRAPH_PE_CODE_VERSION,
        **asdict(config),
        "raw_dim": config.raw_dim,
        "laplacian": "symmetric_normalized_binary_undirected_nonself_support",
        "signnet_pool": "masked_sum_then_deepsets_rho",
        "rpearl_filter": "W_to_S_power_8_W_then_filter_mlp_then_gin_then_rho_then_sample_sum",
        "probe_order_contract": "fixed_candidate_order_seeded_draw_not_exact_topology_only_equivariance",
        "fusion": "bias_free_linear_then_affine_free_layernorm_then_rezero",
    }


def activation_mask(mode: str) -> Tuple[float, float]:
    mode = str(mode).strip().lower()
    if mode == ZERO_MODE:
        return 0.0, 0.0
    if mode == SIGNNET_MODE:
        return 1.0, 0.0
    if mode == RPEARL_MODE:
        return 0.0, 1.0
    raise ValueError(f"Unknown graph PE mode {mode!r}; expected one of {PE_MODES}")


def _require_scipy() -> None:
    if scipy_sparse is None or connected_components is None or eigsh is None:
        raise RuntimeError(
            "SciPy sparse linear algebra is required for graph PE construction; "
            f"import failed with: {SCIPY_IMPORT_ERROR}"
        )


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    if not torch.is_tensor(edge_index) or edge_index.dtype != torch.long:
        raise TypeError("edge_index must be a torch.int64 tensor")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if int(num_nodes) < 0:
        raise ValueError("num_nodes must be non-negative")
    if edge_index.numel() == 0:
        return
    minimum = int(edge_index.min().item())
    maximum = int(edge_index.max().item())
    if minimum < 0 or maximum >= int(num_nodes):
        raise ValueError(
            f"edge_index range [{minimum}, {maximum}] is invalid for {num_nodes} nodes"
        )


def coalesced_undirected_support(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return sorted, unique, non-self, bidirectional support in O(E) storage."""

    _validate_edge_index(edge_index, num_nodes)
    device = edge_index.device
    if int(num_nodes) == 0 or edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src, dst = edge_index
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    encoded = torch.cat(
        [src * int(num_nodes) + dst, dst * int(num_nodes) + src],
        dim=0,
    )
    encoded = torch.unique(encoded, sorted=True)
    return torch.stack(
        [torch.div(encoded, int(num_nodes), rounding_mode="floor"), encoded % int(num_nodes)],
        dim=0,
    )


def _scipy_adjacency(edge_index: torch.Tensor, num_nodes: int):
    _require_scipy()
    support = coalesced_undirected_support(edge_index.detach().cpu(), num_nodes)
    if support.numel() == 0:
        return scipy_sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    rows = support[0].numpy()
    cols = support[1].numpy()
    adjacency = scipy_sparse.coo_matrix(
        (np.ones(rows.shape[0], dtype=np.float64), (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data.fill(1.0)
    return adjacency


def _component_laplacian(adjacency):
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        raise RuntimeError("Non-singleton spectral component contains a zero-degree node")
    inverse_sqrt = 1.0 / np.sqrt(degree)
    scale = scipy_sparse.diags(inverse_sqrt, format="csr")
    normalized_adjacency = scale @ adjacency @ scale
    identity = scipy_sparse.identity(adjacency.shape[0], dtype=np.float64, format="csr")
    return identity - normalized_adjacency


def laplacian_eigenvectors(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG,
):
    """Compute component-local non-trivial normalized-Laplacian eigenvectors.

    Returns `(eigenvectors, mask, audit)`, where the first two tensors are CPU
    float32 tensors with shape `[N, K]`.
    """

    config.validate()
    _require_scipy()
    num_nodes = int(num_nodes)
    _validate_edge_index(edge_index, num_nodes)
    eigenvectors = np.zeros((num_nodes, int(config.eigenvectors)), dtype=np.float32)
    valid_mask = np.zeros((int(config.eigenvectors),), dtype=np.float32)
    audit = {
        "num_nodes": num_nodes,
        "support_edges_directed": 0,
        "components": num_nodes,
        "dense_eigh_components": 0,
        "sparse_eigsh_components": 0,
        "valid_eigenvectors": 0,
    }
    if num_nodes == 0:
        return torch.from_numpy(eigenvectors), torch.from_numpy(eigenvectors.copy()), audit

    adjacency = _scipy_adjacency(edge_index, num_nodes)
    audit["support_edges_directed"] = int(adjacency.nnz)
    if adjacency.nnz == 0:
        repeated_mask = np.repeat(valid_mask[None, :], num_nodes, axis=0)
        return torch.from_numpy(eigenvectors), torch.from_numpy(repeated_mask), audit

    component_count, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    audit["components"] = int(component_count)
    candidates = []
    for component_id in range(int(component_count)):
        node_ids = np.flatnonzero(labels == component_id).astype(np.int64, copy=False)
        component_nodes = int(node_ids.shape[0])
        if component_nodes <= 1:
            continue
        component_adjacency = adjacency[node_ids][:, node_ids].tocsr()
        laplacian = _component_laplacian(component_adjacency)
        if component_nodes <= int(config.dense_eigh_max_component_nodes):
            values, vectors = np.linalg.eigh(laplacian.toarray())
            audit["dense_eigh_components"] += 1
        else:
            requested = min(component_nodes - 1, int(config.eigenvectors) + 1)
            initial = np.linspace(1.0, 2.0, component_nodes, dtype=np.float64)
            initial /= np.linalg.norm(initial)
            try:
                values, vectors = eigsh(
                    laplacian,
                    k=requested,
                    which="SM",
                    tol=float(config.eigsh_tolerance),
                    maxiter=max(
                        1000,
                        int(config.eigsh_max_iterations_per_node) * component_nodes,
                    ),
                    v0=initial,
                )
            except ArpackNoConvergence as exc:
                raise RuntimeError(
                    f"Sparse graph PE eigensolver did not converge for component "
                    f"{component_id} with {component_nodes} nodes"
                ) from exc
            audit["sparse_eigsh_components"] += 1
        order = np.argsort(values, kind="stable")
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise RuntimeError("Non-finite value produced by graph PE eigensolver")
        local_rank = 0
        for index in order.tolist():
            eigenvalue = float(values[index])
            if eigenvalue <= float(config.eigenvalue_tolerance):
                continue
            vector = np.asarray(vectors[:, index], dtype=np.float64)
            if not np.isfinite(vector).all() or not math.isfinite(eigenvalue):
                raise RuntimeError("Non-finite value produced by graph PE eigensolver")
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise RuntimeError("Graph PE eigensolver returned a zero-norm eigenvector")
            candidates.append(
                (
                    eigenvalue,
                    int(node_ids.min()),
                    local_rank,
                    node_ids,
                    (vector / norm).astype(np.float32, copy=False),
                )
            )
            local_rank += 1
            if local_rank >= int(config.eigenvectors):
                break

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for output_index, (_, _, _, node_ids, vector) in enumerate(
        candidates[: int(config.eigenvectors)]
    ):
        eigenvectors[node_ids, output_index] = vector
        valid_mask[output_index] = 1.0
    audit["valid_eigenvectors"] = int(valid_mask.sum())
    repeated_mask = np.repeat(valid_mask[None, :], num_nodes, axis=0)
    return torch.from_numpy(eigenvectors), torch.from_numpy(repeated_mask), audit


def _stable_random_seed(base_seed: int, graph_key: Hashable) -> int:
    payload = f"graph-pe-v{GRAPH_PE_SCHEMA_VERSION}|{int(base_seed)}|{graph_key!r}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def random_probes(
    num_nodes: int,
    base_seed: int,
    graph_key: Hashable,
    config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG,
) -> torch.Tensor:
    config.validate()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_random_seed(base_seed, graph_key))
    return torch.randn(
        (int(num_nodes), int(config.random_samples)),
        dtype=torch.float32,
        generator=generator,
    )


def raw_graph_pe_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    base_seed: int,
    graph_key: Hashable,
    config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG,
):
    eigenvectors, eigen_mask, audit = laplacian_eigenvectors(
        edge_index,
        num_nodes,
        config,
    )
    probes = random_probes(num_nodes, base_seed, graph_key, config)
    raw = torch.cat([eigenvectors, eigen_mask, probes], dim=-1)
    if tuple(raw.shape) != (int(num_nodes), int(config.raw_dim)):
        raise RuntimeError(
            f"Graph PE raw shape mismatch: got {tuple(raw.shape)}, "
            f"expected {(int(num_nodes), int(config.raw_dim))}"
        )
    if not torch.isfinite(raw).all():
        raise RuntimeError("Graph PE raw tensor contains non-finite values")
    audit = dict(audit)
    audit.update(
        {
            "schema_version": GRAPH_PE_SCHEMA_VERSION,
            "raw_dim": int(config.raw_dim),
            "random_seed": _stable_random_seed(base_seed, graph_key),
        }
    )
    return raw, audit


def append_raw_graph_pe(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    base_seed: int,
    graph_key: Hashable,
    config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG,
):
    if x.ndim != 2:
        raise ValueError(f"Node tensor must have shape [N, D], got {tuple(x.shape)}")
    raw, audit = raw_graph_pe_features(
        edge_index.detach().cpu(),
        int(x.shape[0]),
        base_seed,
        graph_key,
        config,
    )
    raw = raw.to(device=x.device, dtype=x.dtype)
    return torch.cat([x, raw], dim=-1), audit


def split_augmented_features(
    x: torch.Tensor,
    base_dim: int,
    config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG,
):
    config.validate()
    expected = int(base_dim) + int(config.raw_dim)
    if x.ndim != 2 or int(x.shape[1]) != expected:
        raise ValueError(
            f"Augmented node feature shape must be [N, {expected}], got {tuple(x.shape)}"
        )
    start = int(base_dim)
    eigen_end = start + int(config.eigenvectors)
    mask_end = eigen_end + int(config.eigenvectors)
    base_x = x[:, :start]
    eigenvectors = x[:, start:eigen_end]
    eigen_mask = x[:, eigen_end:mask_end]
    probes = x[:, mask_end:]
    if not torch.isfinite(x).all():
        raise ValueError("Augmented graph PE input contains non-finite values")
    if torch.any((eigen_mask != 0) & (eigen_mask != 1)):
        raise ValueError("Graph PE eigenvector mask must be binary")
    return base_x, eigenvectors, eigen_mask, probes


class SparseGINLayer(nn.Module):
    """GIN-style sparse update supporting a parallel signal/sample axis."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, support_edge_index: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"SparseGINLayer expects [N, S, H], got {tuple(h.shape)}")
        aggregate = torch.zeros_like(h)
        if support_edge_index.numel() > 0:
            src, dst = support_edge_index
            aggregate.index_add_(0, dst, h[src])
        update = self.mlp((1.0 + self.eps) * h + aggregate)
        return self.norm(h + update)


class SignNetAdaptation(nn.Module):
    def __init__(self, config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG):
        super().__init__()
        config.validate()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.input_projection = nn.Linear(1, hidden_dim)
        self.phi_layers = nn.ModuleList(
            SparseGINLayer(hidden_dim) for _ in range(int(config.gin_layers))
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _phi(self, signal: torch.Tensor, support_edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_projection(signal.unsqueeze(-1)))
        for layer in self.phi_layers:
            h = layer(h, support_edge_index)
        return h

    def forward(
        self,
        eigenvectors: torch.Tensor,
        eigen_mask: torch.Tensor,
        support_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        expected = int(self.config.eigenvectors)
        if eigenvectors.shape != eigen_mask.shape or int(eigenvectors.shape[1]) != expected:
            raise ValueError("SignNet eigenvector/mask shape mismatch")
        encoded = self._phi(eigenvectors, support_edge_index)
        encoded = encoded + self._phi(-eigenvectors, support_edge_index)
        encoded = encoded * eigen_mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1)
        valid = (eigen_mask.sum(dim=1, keepdim=True) > 0).to(dtype=pooled.dtype)
        return self.rho(pooled) * valid


def normalized_adjacency_apply(
    signal: torch.Tensor,
    support_edge_index: torch.Tensor,
) -> torch.Tensor:
    if signal.ndim < 2:
        raise ValueError("Normalized adjacency signal must have at least two dimensions")
    num_nodes = int(signal.shape[0])
    output = torch.zeros_like(signal)
    if num_nodes == 0 or support_edge_index.numel() == 0:
        return output
    src, dst = support_edge_index
    degree = torch.zeros((num_nodes,), dtype=signal.dtype, device=signal.device)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=signal.dtype))
    inverse_sqrt = degree.clamp_min(1.0).rsqrt()
    scale_shape = (src.shape[0],) + (1,) * (signal.ndim - 1)
    scale = (inverse_sqrt[src] * inverse_sqrt[dst]).reshape(scale_shape)
    output.index_add_(0, dst, signal[src] * scale)
    return output


class RPEARLAdaptation(nn.Module):
    def __init__(self, config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG):
        super().__init__()
        config.validate()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.filter_mlp = nn.Sequential(
            nn.Linear(int(config.filter_order) + 1, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.sample_layers = nn.ModuleList(
            SparseGINLayer(hidden_dim) for _ in range(int(config.gin_layers))
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        probes: torch.Tensor,
        support_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if probes.ndim != 2 or int(probes.shape[1]) != int(self.config.random_samples):
            raise ValueError("R-PEARL random probe shape mismatch")
        powers = [probes]
        current = probes
        for _ in range(int(self.config.filter_order)):
            current = normalized_adjacency_apply(current, support_edge_index)
            powers.append(current)
        filter_bank = torch.stack(powers, dim=-1)
        h = self.filter_mlp(filter_bank)
        for layer in self.sample_layers:
            h = layer(h, support_edge_index)
        # The declared adaptation encodes EACH sample before summing samples.
        return self.rho(h).sum(dim=1)


class BranchMaskedGraphPE(nn.Module):
    """Match registered parameters/forward calls, not active learning capacity."""

    def __init__(
        self,
        mode: str,
        config: GraphPEConfig = DEFAULT_GRAPH_PE_CONFIG,
    ):
        super().__init__()
        config.validate()
        self.mode = str(mode).strip().lower()
        self.config = config
        mask = activation_mask(self.mode)
        self.signnet = SignNetAdaptation(config)
        self.rpearl = RPEARLAdaptation(config)
        self.fusion = nn.Linear(2 * int(config.hidden_dim), int(config.hidden_dim), bias=False)
        self.fusion_norm = nn.LayerNorm(int(config.hidden_dim), elementwise_affine=False)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "activation_mask",
            torch.tensor(mask, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        eigenvectors: torch.Tensor,
        eigen_mask: torch.Tensor,
        probes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(eigenvectors.shape[0])
        support = coalesced_undirected_support(edge_index, num_nodes)
        signnet_output = self.signnet(eigenvectors, eigen_mask, support)
        rpearl_output = self.rpearl(probes, support)
        mask = self.activation_mask.to(device=signnet_output.device, dtype=signnet_output.dtype)
        combined = torch.cat(
            [mask[0] * signnet_output, mask[1] * rpearl_output],
            dim=-1,
        )
        projected = self.fusion_norm(self.fusion(combined))
        return torch.tanh(self.residual_gate) * projected

    def operation_inventory(self) -> Dict[str, int]:
        return {
            "signnet_sparse_gin_calls": 2 * int(self.config.gin_layers),
            "rpearl_normalized_adjacency_calls": int(self.config.filter_order),
            "rpearl_sparse_gin_calls": int(self.config.gin_layers),
            "total_pe_sparse_operator_calls": (
                3 * int(self.config.gin_layers) + int(self.config.filter_order)
            ),
        }

    def parameter_inventory(self) -> Dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        signnet = count(self.signnet)
        rpearl = count(self.rpearl)
        fusion_and_gate = count(self.fusion) + int(self.residual_gate.numel())
        return {
            "signnet": signnet,
            "rpearl": rpearl,
            "fusion_and_rezero": fusion_and_gate,
            "total": signnet + rpearl + fusion_and_gate,
        }

    def loss_path_inventory(self) -> Dict[str, object]:
        """Structural upper bounds after the gate opens, NOT measured gradients."""
        registered = self.parameter_inventory()
        active = self.mode != ZERO_MODE
        encoder = registered[self.mode] if active else 0
        fusion = int(self.fusion.weight.numel()) // 2 if active else 0
        gate = int(self.residual_gate.numel()) if active else 0
        return {
            "capacity_matching_scope": "registered_parameters_and_executed_operators_only",
            "active_capacity_matched": False,
            "active_encoder": self.mode if active else None,
            "encoder_parameters": encoder,
            "fusion_parameters": fusion,
            "gate_parameters": gate,
            "parameters_after_gate_opens": encoder + fusion + gate,
            "executed_sparse_operator_calls": 17,
            "signal_carrying_sparse_operator_calls": (
                6 if self.mode == SIGNNET_MODE else 11 if self.mode == RPEARL_MODE else 0
            ),
            "encoder_gradient_at_zero_gate": "zero_until_gate_opens",
        }


EXPECTED_PE_PARAMETER_INVENTORY = {
    "signnet": 75_459,
    "rpearl": 85_731,
    "fusion_and_rezero": 18_433,
    "total": 179_623,
}


def assert_default_capacity(module: BranchMaskedGraphPE) -> None:
    if module.config != DEFAULT_GRAPH_PE_CONFIG:
        raise RuntimeError("Production graph PE capacity audit only accepts the frozen default config")
    expected_mask = torch.tensor(activation_mask(module.mode), device=module.activation_mask.device)
    if not torch.equal(module.activation_mask, expected_mask):
        raise RuntimeError("Graph PE activation mask does not match the declared mode")
    if any(not parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("Graph PE registered trainable scope was changed")
    actual = module.parameter_inventory()
    if actual != EXPECTED_PE_PARAMETER_INVENTORY:
        raise RuntimeError(
            f"Graph PE parameter inventory mismatch: actual={actual}, "
            f"expected={EXPECTED_PE_PARAMETER_INVENTORY}"
        )
    expected_operations = {
        "signnet_sparse_gin_calls": 6,
        "rpearl_normalized_adjacency_calls": 8,
        "rpearl_sparse_gin_calls": 3,
        "total_pe_sparse_operator_calls": 17,
    }
    actual_operations = module.operation_inventory()
    if actual_operations != expected_operations:
        raise RuntimeError(
            f"Graph PE operator inventory mismatch: actual={actual_operations}, "
            f"expected={expected_operations}"
        )
