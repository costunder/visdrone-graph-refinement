#!/usr/bin/env python3
"""CPU-only contract tests for Experiment-1/2 graph positional encodings."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch


FINAL_DIR = Path(__file__).resolve().parent
EXPERIMENT_1_DIR = FINAL_DIR / "experiment_1"
EXPERIMENT_2_DIR = FINAL_DIR / "experiment_2"
SCRIPT_DIR = EXPERIMENT_1_DIR / "scripts"
for path in (FINAL_DIR, EXPERIMENT_1_DIR, EXPERIMENT_2_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import graph_positional_encoding as graph_pe
import graph_pe_run_support as run_support
import run_gois_two_stage_gnn_ablation as ab
import sparse_ppr_sage as exp2


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment_1_model = load_module(
    "experiment_1_graph_pe_model_contract_test",
    EXPERIMENT_1_DIR / "graph_pe_model.py",
)
experiment_2_model = load_module(
    "experiment_2_graph_pe_model_contract_test",
    EXPERIMENT_2_DIR / "graph_pe_model.py",
)


def ring_graph(num_nodes):
    pairs = []
    for node in range(num_nodes):
        pairs.append((node, node))
        neighbor = (node + 1) % num_nodes
        pairs.append((node, neighbor))
        pairs.append((neighbor, node))
    return torch.tensor(sorted(set(pairs)), dtype=torch.long).t().contiguous()


def permute_graph(x, edge_index, permutation):
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    return x[permutation], inverse[edge_index]


def gradient_sum(module):
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def assert_capacity_and_initialization(models, expected_total, expected_base):
    reference_parameters = {
        name: parameter.detach().clone()
        for name, parameter in models[graph_pe.ZERO_MODE].named_parameters()
    }
    for mode, model in models.items():
        model.assert_production_contract()
        inventory = model.capacity_inventory()
        assert inventory["base_trainable_parameters"] == expected_base
        assert inventory["pe_trainable_parameters"] == 179_623
        assert inventory["total_trainable_parameters"] == expected_total
        assert inventory["base_graph_layer_calls"] == 3
        assert inventory["pe_operator_calls"]["total_pe_sparse_operator_calls"] == 17
        loss_path = inventory["pe_loss_path"]
        assert loss_path["active_capacity_matched"] is False
        assert loss_path["parameters_after_gate_opens"] == {
            graph_pe.ZERO_MODE: 0, graph_pe.SIGNNET_MODE: 84_676, graph_pe.RPEARL_MODE: 94_948,
        }[mode]
        parameters = dict(model.named_parameters())
        assert parameters.keys() == reference_parameters.keys()
        for name, parameter in parameters.items():
            assert torch.equal(parameter.detach(), reference_parameters[name]), (mode, name)
    assert torch.equal(
        models[graph_pe.ZERO_MODE].graph_pe.activation_mask,
        torch.tensor([0.0, 0.0]),
    )
    assert torch.equal(
        models[graph_pe.SIGNNET_MODE].graph_pe.activation_mask,
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(
        models[graph_pe.RPEARL_MODE].graph_pe.activation_mask,
        torch.tensor([0.0, 1.0]),
    )


def build_models(builder, args):
    models = {}
    for mode in graph_pe.PE_MODES:
        torch.manual_seed(12345)
        models[mode] = builder(ab, args, mode)
    return models


def assert_zero_identity(model, augmented_x, edge_index, edge_attr):
    model.eval()
    with torch.no_grad():
        wrapped = model(augmented_x, edge_index, edge_attr)
        base_x = augmented_x[:, : model.base_input_dim]
        standalone = model.backbone(base_x, edge_index, edge_attr)
    assert torch.equal(wrapped, standalone), (wrapped - standalone).abs().max().item()


def assert_branch_gradients(model, augmented_x, edge_index, edge_attr, active):
    model.train()
    model.zero_grad(set_to_none=True)
    model.graph_pe.residual_gate.data.fill_(0.5)
    output = model(augmented_x, edge_index, edge_attr)
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    signnet_gradient = gradient_sum(model.graph_pe.signnet)
    rpearl_gradient = gradient_sum(model.graph_pe.rpearl)
    if active == graph_pe.ZERO_MODE:
        assert signnet_gradient == 0.0
        assert rpearl_gradient == 0.0
    elif active == graph_pe.SIGNNET_MODE:
        assert signnet_gradient > 0.0
        assert rpearl_gradient == 0.0
        assert torch.count_nonzero(model.graph_pe.fusion.weight.grad[:, 96:]) == 0
        assert torch.count_nonzero(model.graph_pe.fusion.weight.grad[:, :96]) > 0
    elif active == graph_pe.RPEARL_MODE:
        assert signnet_gradient == 0.0
        assert rpearl_gradient > 0.0
        assert torch.count_nonzero(model.graph_pe.fusion.weight.grad[:, :96]) == 0
        assert torch.count_nonzero(model.graph_pe.fusion.weight.grad[:, 96:]) > 0
    else:
        raise AssertionError(active)


def assert_permutation_consistency(model, augmented_x, edge_index, edge_attr):
    """Conditional equivariance: precomputed PE travels WITH each node."""
    model.eval()
    model.graph_pe.residual_gate.data.fill_(0.25)
    generator = torch.Generator().manual_seed(91)
    permutation = torch.randperm(augmented_x.shape[0], generator=generator)
    permuted_x, permuted_edges = permute_graph(augmented_x, edge_index, permutation)
    edge_attr_permuted = edge_attr.clone()
    with torch.no_grad():
        original = model(augmented_x, edge_index, edge_attr)
        permuted = model(permuted_x, permuted_edges, edge_attr_permuted)
    assert torch.allclose(permuted, original[permutation], atol=2e-5, rtol=2e-5)


def assert_runtime_operator_calls(model, x, edges, attrs):
    counts = Counter()
    handles = []
    for name, layer in model.named_modules():
        if isinstance(layer, graph_pe.SparseGINLayer):
            branch = "signnet" if ".signnet." in name else "rpearl"
            handles.append(layer.register_forward_hook(
                lambda module, inputs, output, branch=branch: counts.update([branch])
            ))
    for layer in model.backbone.layers:
        handles.append(layer.register_forward_hook(lambda module, inputs, output: counts.update(["base"])))
    try:
        with torch.no_grad(), patch.object(
            graph_pe, "normalized_adjacency_apply", wraps=graph_pe.normalized_adjacency_apply,
        ) as adjacency:
            model(x, edges, attrs)
            assert adjacency.call_count == 8
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"signnet": 6, "rpearl": 3, "base": 3}, counts


def assert_initial_gate_learns(model, x, edges, attrs, mode):
    model.eval()
    model.zero_grad(set_to_none=True)
    assert model.graph_pe.residual_gate.item() == 0.0
    output = model(x, edges, attrs)
    weights = torch.randn(output.shape, generator=torch.Generator().manual_seed(710))
    (output * weights).sum().backward()
    assert gradient_sum(model.graph_pe.signnet) == 0.0
    assert gradient_sum(model.graph_pe.rpearl) == 0.0
    gate_gradient = model.graph_pe.residual_gate.grad.abs().item()
    assert (gate_gradient == 0.0) if mode == graph_pe.ZERO_MODE else (gate_gradient > 0.0)


def assert_rpearl_pool_order():
    edges = graph_pe.coalesced_undirected_support(ring_graph(12), 12)
    probes = graph_pe.random_probes(12, 42, ("rho-order", 1))
    torch.manual_seed(222)
    model = graph_pe.RPEARLAdaptation().eval()
    rho_shapes = []
    hook = model.rho.register_forward_pre_hook(lambda module, inputs: rho_shapes.append(tuple(inputs[0].shape)))
    try:
        with torch.no_grad():
            actual = model(probes, edges)
    finally:
        hook.remove()
    assert rho_shapes == [(12, 16, 96)], rho_shapes
    with torch.no_grad():
        powers = [probes]
        for _ in range(8):
            powers.append(graph_pe.normalized_adjacency_apply(powers[-1], edges))
        h = model.filter_mlp(torch.stack(powers, dim=-1))
        for layer in model.sample_layers:
            h = layer(h, edges)
        expected = model.rho(h).sum(dim=1)
        wrong_order = model.rho(h.sum(dim=1))
    assert torch.equal(actual, expected)
    assert not torch.allclose(actual, wrong_order)


def assert_rebuilt_pe_permutation_scope(models, base_dim, edge_dim):
    # Irregular simple-spectrum graph avoids claiming BasisNet invariance.
    pairs = set(map(tuple, ring_graph(12).t().tolist()))
    for src, dst in [(0, 2), (0, 5), (1, 6), (4, 8), (7, 10)]:
        pairs.update([(src, dst), (dst, src)])
    edges = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    generator = torch.Generator().manual_seed(735)
    x = torch.randn((12, base_dim), generator=generator)
    attrs = torch.randn((edges.shape[1], edge_dim), generator=generator)
    original_x, _ = graph_pe.append_raw_graph_pe(x, edges, 42, ("rebuild", base_dim))
    permutation = torch.randperm(12, generator=torch.Generator().manual_seed(91))
    permuted_x, permuted_edges = permute_graph(x, edges, permutation)
    rebuilt_x, _ = graph_pe.append_raw_graph_pe(permuted_x, permuted_edges, 42, ("rebuild", base_dim))
    # A new seeded draw attaches probes to row indices, NOT stable candidate IDs.
    assert torch.equal(rebuilt_x[:, -16:], original_x[:, -16:])
    assert not torch.equal(rebuilt_x[:, -16:], original_x[permutation, -16:])
    coupled_x = rebuilt_x.clone()
    coupled_x[:, -16:] = original_x[permutation, -16:]
    for mode in (graph_pe.SIGNNET_MODE, graph_pe.RPEARL_MODE):
        model = models[mode].eval()
        with torch.no_grad():
            model.graph_pe.residual_gate.fill_(0.25)
            expected = model(original_x, edges, attrs)[permutation]
            rebuilt = model(rebuilt_x, permuted_edges, attrs)
            coupled = model(coupled_x, permuted_edges, attrs)
        assert torch.allclose(coupled, expected, atol=2e-5, rtol=2e-5)
        if mode == graph_pe.SIGNNET_MODE:
            assert torch.allclose(rebuilt, expected, atol=2e-5, rtol=2e-5)
        else:
            # Deliberate limitation regression, NOT a claim of exact equivariance.
            assert not torch.allclose(rebuilt, expected, atol=2e-5, rtol=2e-5)


def assert_pe_checkpoint_resume(model, other_mode_model, x, edges, attrs):
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    def step():
        optimizer.zero_grad(set_to_none=True)
        model(x, edges, attrs).square().mean().backward()
        optimizer.step()
    step()  # CPU synthetic optimizer-state test only.
    with tempfile.TemporaryDirectory(prefix="pe_real_model_checkpoint_") as directory:
        path = Path(directory) / "epoch_001.pt"
        run_support.save_training_checkpoint_atomic(model, optimizer, 1, path)
        step()
        uninterrupted = {key: value.detach().clone() for key, value in model.state_dict().items()}
        run_support.load_training_checkpoint(model, optimizer, path, "cpu", 1)
        step()
        for key, expected in uninterrupted.items():
            assert torch.equal(model.state_dict()[key], expected), key
        try:
            run_support.load_training_checkpoint(other_mode_model, None, path, "cpu", 1)
        except RuntimeError as exc:
            assert "mismatch" in str(exc)
        else:
            raise AssertionError("Checkpoint from another PE mode was accepted")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["model_contract"]["graph_pe"]["mode"] = other_mode_model.graph_pe.mode
        tampered = Path(directory) / "tampered_mask.pt"
        run_support.save_torch_atomic(payload, tampered)
        try:
            run_support.load_training_checkpoint(other_mode_model, None, tampered, "cpu", 1)
        except RuntimeError as exc:
            assert "activation-mask mismatch" in str(exc)
        else:
            raise AssertionError("Tampered checkpoint activation mask was accepted")


def assert_empty_and_isolated_inputs():
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    for count in (0, 1, 4):
        raw, audit = graph_pe.raw_graph_pe_features(empty_edges, count, 42, ("isolated", count))
        assert raw.shape == (count, 32) and torch.isfinite(raw).all()
        assert torch.count_nonzero(raw[:, :16]) == 0
        assert audit["valid_eigenvectors"] == 0
        for mode in graph_pe.PE_MODES:
            model = graph_pe.BranchMaskedGraphPE(mode)
            output = model(raw[:, :8], raw[:, 8:16], raw[:, 16:], empty_edges)
            assert output.shape == (count, 96) and torch.isfinite(output).all()


def assert_disjoint_batch_consistency(
    model,
    first_x,
    first_edge_index,
    first_edge_attr,
    base_dim,
    edge_dim,
    graph_key,
):
    second_nodes = 7
    second_edge_index = ring_graph(second_nodes)
    generator = torch.Generator().manual_seed(1701 + int(base_dim))
    second_base_x = torch.randn((second_nodes, int(base_dim)), generator=generator)
    second_x, _ = graph_pe.append_raw_graph_pe(
        second_base_x,
        second_edge_index,
        base_seed=42,
        graph_key=graph_key,
    )
    second_edge_attr = torch.randn(
        (second_edge_index.shape[1], int(edge_dim)),
        generator=generator,
    )
    batched_x = torch.cat([first_x, second_x], dim=0)
    batched_edge_index = torch.cat(
        [
            first_edge_index,
            second_edge_index + int(first_x.shape[0]),
        ],
        dim=1,
    )
    batched_edge_attr = torch.cat([first_edge_attr, second_edge_attr], dim=0)
    model.eval()
    model.graph_pe.residual_gate.data.fill_(0.25)
    with torch.no_grad():
        first_output = model(first_x, first_edge_index, first_edge_attr)
        second_output = model(second_x, second_edge_index, second_edge_attr)
        batched_output = model(batched_x, batched_edge_index, batched_edge_attr)
    expected = torch.cat([first_output, second_output], dim=0)
    assert torch.allclose(batched_output, expected, atol=2e-5, rtol=2e-5)


def assert_portable_checkpoint_round_trip():
    torch.manual_seed(314)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(5, 3)
    model(inputs).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    expected_state_entries = len(optimizer.state)
    with tempfile.TemporaryDirectory(prefix="graph_pe_checkpoint_") as directory:
        checkpoint = Path(directory) / "epoch_003.pt"
        run_support.save_training_checkpoint_atomic(
            model,
            optimizer,
            3,
            checkpoint,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        optimizer.state.clear()
        loaded_epoch = run_support.load_training_checkpoint(
            model,
            optimizer,
            checkpoint,
            torch.device("cpu"),
            3,
        )
    assert loaded_epoch == 3
    assert len(optimizer.state) == expected_state_entries
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), expected[name])


def main():
    torch.set_num_threads(1)
    assert_portable_checkpoint_round_trip()
    assert_rpearl_pool_order()
    assert_empty_and_isolated_inputs()
    ab.configure_class_space("visdrone10")
    assert ab.SIZE_AWARE_NODE_DIM == 37
    assert exp2.node_dim(ab) == 40
    assert exp2.edge_dim(ab) == 27

    num_nodes = 12
    edge_index = ring_graph(num_nodes)
    generator = torch.Generator().manual_seed(7)
    experiment_1_x = torch.randn(
        (num_nodes, ab.SIZE_AWARE_NODE_DIM),
        generator=generator,
    )
    experiment_2_x = torch.randn(
        (num_nodes, exp2.node_dim(ab)),
        generator=generator,
    )
    experiment_1_edge_attr = torch.randn(
        (edge_index.shape[1], ab.SIZE_AWARE_EDGE_DIM),
        generator=generator,
    )
    experiment_2_edge_attr = torch.randn(
        (edge_index.shape[1], exp2.edge_dim(ab)),
        generator=generator,
    )
    experiment_2_edge_attr[:, ab.SIZE_AWARE_EDGE_DIM + exp2.PPR_EXTRA_OFFSET].abs_()

    augmented_1, audit_1 = graph_pe.append_raw_graph_pe(
        experiment_1_x,
        edge_index,
        base_seed=42,
        graph_key=("contract", 1),
    )
    augmented_1_repeat, audit_1_repeat = graph_pe.append_raw_graph_pe(
        experiment_1_x,
        edge_index,
        base_seed=42,
        graph_key=("contract", 1),
    )
    augmented_2, _ = graph_pe.append_raw_graph_pe(
        experiment_2_x,
        edge_index,
        base_seed=42,
        graph_key=("contract", 2),
    )
    assert augmented_1.shape == (num_nodes, 69)
    assert augmented_2.shape == (num_nodes, 72)
    assert torch.equal(augmented_1, augmented_1_repeat)
    assert audit_1 == audit_1_repeat
    assert audit_1["valid_eigenvectors"] == 8
    assert torch.isfinite(augmented_1).all() and torch.isfinite(augmented_2).all()

    _, eigenvectors, eigen_mask, _ = graph_pe.split_augmented_features(
        augmented_1,
        ab.SIZE_AWARE_NODE_DIM,
    )
    torch.manual_seed(222)
    signnet = graph_pe.SignNetAdaptation().eval()
    support = graph_pe.coalesced_undirected_support(edge_index, num_nodes)
    with torch.no_grad():
        positive = signnet(eigenvectors, eigen_mask, support)
        negative = signnet(-eigenvectors, eigen_mask, support)
        signs = torch.tensor([1, -1, 1, -1, -1, 1, -1, 1], dtype=eigenvectors.dtype)
        independent_signs = signnet(eigenvectors * signs, eigen_mask, support)
    assert torch.equal(positive, negative)
    assert torch.equal(positive, independent_signs)

    args_1 = argparse.Namespace(gnn_hidden_dim=96, gnn_layers=3)
    models_1 = build_models(experiment_1_model.build_model, args_1)
    assert_capacity_and_initialization(models_1, 628_810, 449_187)
    for model in models_1.values():
        assert_zero_identity(
            model,
            augmented_1,
            edge_index,
            experiment_1_edge_attr,
        )
    for mode, model in models_1.items():
        assert_initial_gate_learns(model, augmented_1, edge_index, experiment_1_edge_attr, mode)
        assert_runtime_operator_calls(model, augmented_1, edge_index, experiment_1_edge_attr)
        assert_branch_gradients(
            model,
            augmented_1,
            edge_index,
            experiment_1_edge_attr,
            mode,
        )
    assert_permutation_consistency(
        models_1[graph_pe.SIGNNET_MODE],
        augmented_1,
        edge_index,
        experiment_1_edge_attr,
    )
    assert_permutation_consistency(
        models_1[graph_pe.RPEARL_MODE],
        augmented_1,
        edge_index,
        experiment_1_edge_attr,
    )
    for mode in (graph_pe.SIGNNET_MODE, graph_pe.RPEARL_MODE):
        assert_disjoint_batch_consistency(
            models_1[mode],
            augmented_1,
            edge_index,
            experiment_1_edge_attr,
            ab.SIZE_AWARE_NODE_DIM,
            ab.SIZE_AWARE_EDGE_DIM,
            ("batch", "experiment_1", mode),
        )

    args_2 = argparse.Namespace(
        gnn_hidden_dim=96,
        gnn_layers=3,
        exp2_attention_heads=4,
        exp2_dropout=0.10,
    )
    models_2 = build_models(experiment_2_model.build_model, args_2)
    assert_capacity_and_initialization(models_2, 416_710, 237_087)
    for model in models_2.values():
        assert_zero_identity(
            model,
            augmented_2,
            edge_index,
            experiment_2_edge_attr,
        )
    for mode, model in models_2.items():
        assert_initial_gate_learns(model, augmented_2, edge_index, experiment_2_edge_attr, mode)
        assert_runtime_operator_calls(model, augmented_2, edge_index, experiment_2_edge_attr)
        assert_branch_gradients(
            model,
            augmented_2,
            edge_index,
            experiment_2_edge_attr,
            mode,
        )
    for mode in (graph_pe.SIGNNET_MODE, graph_pe.RPEARL_MODE):
        assert_disjoint_batch_consistency(
            models_2[mode],
            augmented_2,
            edge_index,
            experiment_2_edge_attr,
            exp2.node_dim(ab),
            exp2.edge_dim(ab),
            ("batch", "experiment_2", mode),
        )

    for mode in (graph_pe.SIGNNET_MODE, graph_pe.RPEARL_MODE):
        assert_permutation_consistency(models_2[mode], augmented_2, edge_index, experiment_2_edge_attr)
    assert_rebuilt_pe_permutation_scope(models_1, 37, 24)
    assert_rebuilt_pe_permutation_scope(models_2, 40, 27)
    for mode, other in ((graph_pe.SIGNNET_MODE, graph_pe.RPEARL_MODE), (graph_pe.RPEARL_MODE, graph_pe.ZERO_MODE)):
        assert_pe_checkpoint_resume(models_1[mode], models_1[other], augmented_1, edge_index, experiment_1_edge_attr)
        assert_pe_checkpoint_resume(models_2[mode], models_2[other], augmented_2, edge_index, experiment_2_edge_attr)

    large_edge_index = ring_graph(70)
    _, _, large_audit = graph_pe.laplacian_eigenvectors(large_edge_index, 70)
    assert large_audit["dense_eigh_components"] == 0
    assert large_audit["sparse_eigsh_components"] == 1
    assert large_audit["valid_eigenvectors"] == 8

    print("graph PE contract: PASS")
    print("experiment 1 parameters: 628810")
    print("experiment 2 parameters: 416710")
    print("PE sparse operator calls per forward: 17 (measured using hooks)")
    print("rho-before-pool / raw-PE reconstruction scope / PE optimizer resume: PASS")


if __name__ == "__main__":
    main()
