"""Deterministic CPU acceptance tests of full-size production models, not runs."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import graph_pe_production as p
import graph_pe_production_data as data

torch.set_num_threads(2)


def fixture(n=7, dtype=torch.float32):
    generator = torch.Generator().manual_seed(9001)
    x = torch.randn(n, 37, generator=generator, dtype=dtype)
    x[:, 16:21] = 0
    x[:, 18] = 1
    x[:, 27:37] = 0
    x[torch.arange(n), 27 + torch.arange(n) % 10] = 1
    edges = torch.tensor([(i, j) for i in range(n) for j in range(n) if abs(i-j) <= 2], dtype=torch.long).T.contiguous()
    if n == 0:
        edges = torch.empty((2, 0), dtype=torch.long)
    attrs = torch.rand(edges.shape[1], 24, generator=generator, dtype=dtype)
    support = p.legacy_ops.coalesced_undirected_support(edges, n)
    u = torch.randn(n, 32, generator=generator, dtype=dtype)
    mask = torch.ones(n, 32, dtype=torch.bool)
    w = torch.randn(n, 120, generator=generator, dtype=dtype)
    return x, edges, attrs, dict(u=u, mask=mask, w=w, support=support)


def outputs_and_gradients(model, f):
    model.zero_grad(set_to_none=True)
    result = model(*f[:3], **f[3])
    (result.square().sum() + result.sum() / 3).backward()
    return result.detach(), {name: None if param.grad is None else param.grad.detach().clone() for name, param in model.named_parameters()}


class ProductionContractTests(unittest.TestCase):
    def test_constructed_capacity_and_base_initialization(self):
        reference = p.Experiment1Production(data.ab, "no_pe")
        for mode, expected in p.EXPECTED_E1.items():
            with self.subTest(mode=mode):
                model = p.Experiment1Production(data.ab, mode)
                self.assertEqual(model.assert_contract(), expected)
                for key, value in model.backbone.state_dict().items():
                    self.assertTrue(torch.equal(value, reference.backbone.state_dict()[key]), key)
                if mode != "no_pe":
                    self.assertEqual(len({layer.mlp[0].weight.data_ptr() for layer in model.encoder.layers}), 8)
                    self.assertFalse(any("activation_mask" in key for key in model.state_dict()))
                else:
                    self.assertFalse(hasattr(model, "encoder"))
        with self.assertRaises(ValueError):
            p.Experiment1Production(data.ab, "zero")
        # CPU initialization substreams must not reseed CUDA dropout streams.
        with patch.object(torch.cuda, "manual_seed_all", side_effect=AssertionError("Global CUDA RNG was modified")):
            for mode in p.EXPECTED_E1:
                p.Experiment1Production(data.ab, mode)

    def test_gate_identity_then_all_pe_parameter_gradients(self):
        f = fixture()
        reference = p.Experiment1Production(data.ab, "no_pe", recompute=False)
        expected = reference(*f[:3])
        for mode in ("signnet", "rpearl"):
            model = p.Experiment1Production(data.ab, mode, recompute=False)
            self.assertTrue(torch.equal(model(*f[:3], **f[3]), expected), mode)
            _, gradients = outputs_and_gradients(model, f)
            self.assertGreater(float(gradients["gate"].abs()), 0)
            self.assertTrue(all(not torch.any(g) for name, g in gradients.items() if name.startswith("encoder.")))
            model.gate.data.fill_(.2)
            _, gradients = outputs_and_gradients(model, f)
            for name, gradient in gradients.items():
                self.assertIsNotNone(gradient, name)
                self.assertTrue(torch.isfinite(gradient).all(), name)
                self.assertGreater(int(torch.count_nonzero(gradient)), 0, name)

    def test_chunked_forward_backward_equivalence(self):
        for dtype, atol, rtol in ((torch.float64, 1e-9, 1e-7), (torch.float32, 1e-5, 1e-4)):
            for mode in p.EXPECTED_E1:
                with self.subTest(dtype=dtype, mode=mode):
                    full = p.Experiment1Production(data.ab, mode, recompute=False, signal_chunk=120, edge_chunk=100000, node_chunk=10000).to(dtype)
                    chunked = p.Experiment1Production(data.ab, mode, recompute=True, signal_chunk=8, edge_chunk=5, node_chunk=2).to(dtype)
                    if mode != "no_pe":
                        full.gate.data.fill_(.2)
                        chunked.gate.data.fill_(.2)
                    f = fixture(dtype=dtype)
                    a, ga = outputs_and_gradients(full, f)
                    b, gb = outputs_and_gradients(chunked, f)
                    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
                    for name in ga:
                        torch.testing.assert_close(ga[name], gb[name], atol=atol, rtol=rtol, msg=lambda message: f"{mode}/{name}: {message}")

    def test_original_backbone_equivalence_and_hooks(self):
        f = fixture(dtype=torch.float64)
        model = p.Experiment1Production(data.ab, "no_pe", recompute=False).double()
        original = data.ab.SizeAwareGraphGNN(37, 24, 256, 6, output_dim=3).double()
        original.load_state_dict(model.backbone.state_dict(), strict=True)
        actual = model(*f[:3])
        torch.testing.assert_close(actual, original(*f[:3]), atol=1e-9, rtol=1e-7)
        counts = {}
        def hook(name):
            def count(*_):
                counts[name] = counts.get(name, 0) + 1
            return count
        handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()
                   if name in {"backbone.node_type_projection", "backbone.head"} or name.startswith("backbone.layers.") and (name.count(".") == 2 or "relation_messages." in name)]
        model(*f[:3])
        for handle in handles:
            handle.remove()
        self.assertEqual(len(counts), 56)  # 6 layers + 48 typed transforms + input/head skips
        self.assertTrue(all(value == 1 for value in counts.values()))

    def test_sign_symmetry_padding_and_permutation(self):
        _, _, _, raw = fixture()
        encoder = p.SignNetK32().eval()
        with torch.no_grad():
            actual = encoder(raw["u"], raw["mask"], raw["support"])
            signs = torch.where(torch.arange(32) % 2 == 0, -1., 1.)
            flipped = encoder(raw["u"] * signs, raw["mask"], raw["support"])
            torch.testing.assert_close(actual, flipped, atol=1e-5, rtol=1e-4)
            masked = encoder(raw["u"], torch.zeros_like(raw["mask"]), raw["support"])
            self.assertEqual(int(torch.count_nonzero(masked)), 0)
            perm = torch.tensor([2, 0, 6, 1, 4, 3, 5])
            inverse = torch.argsort(perm)
            changed = encoder(raw["u"][perm], raw["mask"][perm], inverse[raw["support"]])
            torch.testing.assert_close(changed, actual[perm], atol=1e-5, rtol=1e-4)
            for n in (0, 1):
                u, mask, support, _ = p.eigen_features(torch.empty((2, 0), dtype=torch.long), n)
                self.assertEqual(int(torch.count_nonzero(encoder(u, mask, support))), 0)
        edges = torch.tensor([(i, i+1) for i in range(69)]).T
        u, mask, support, audit = p.eigen_features(edges, 70)
        self.assertEqual(audit["sparse_eigsh_components"], 1)
        self.assertEqual(audit["valid_eigenvectors"], 32)
        self.assertTrue(mask.all())
        # Recompute the PE from a permuted simple-spectrum topology, not only
        # permute an already supplied U. Repeated eigenspaces remain excluded.
        edges = torch.tensor([(i, i+1) for i in range(8)]).T
        perm = torch.tensor([2, 8, 0, 5, 1, 7, 3, 6, 4])
        inverse = torch.argsort(perm)
        u, mask, support, _ = p.eigen_features(edges, 9)
        pu, pmask, psupport, _ = p.eigen_features(inverse[edges], 9)
        with torch.no_grad():
            torch.testing.assert_close(encoder(pu, pmask, psupport), encoder(u, mask, support)[perm], atol=1e-5, rtol=1e-4)

    def test_rpearl_rho_before_mean_and_conditional_permutation(self):
        raw = fixture()[3]
        encoder = p.RPEARLM120().eval()
        seen = []
        handle = encoder.rho.register_forward_pre_hook(lambda module, inputs: seen.append(tuple(inputs[0].shape)))
        with torch.no_grad():
            actual = encoder(raw["w"], raw["support"])
            handle.remove()
            self.assertEqual(seen, [(7, 8, 128)] * 15)
            perm = torch.tensor([2, 0, 6, 1, 4, 3, 5])
            inverse = torch.argsort(perm)
            changed = encoder(raw["w"][perm], inverse[raw["support"]])
            torch.testing.assert_close(changed, actual[perm], atol=1e-5, rtol=1e-4)
            with self.assertRaises(ValueError):
                encoder(raw["w"][:, :16], raw["support"])

    def test_separate_probe_and_alignment_rng(self):
        before = torch.get_rng_state()
        options = dict(seed=42, experiment="experiment_1", image_id=10)
        first = p.probes(7, split="train", epoch=1, **options)
        self.assertTrue(torch.equal(first, p.probes(7, split="train", epoch=1, **options)))
        self.assertFalse(torch.equal(first, p.probes(7, split="train", epoch=2, **options)))
        self.assertTrue(torch.equal(p.probes(7, split="eval", epoch=1, **options), p.probes(7, split="eval", epoch=120, **options)))
        permutation = p.alignment_permutation(torch.arange(7), split="train", **options)
        self.assertFalse(torch.any(permutation == torch.arange(7)))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_loss_accumulation_matches_original_weighted_batch(self):
        torch.manual_seed(4)
        logits = torch.randn(19, 3, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, 2, (19, 3)).double()
        weights = torch.rand(19, dtype=torch.float64) + .1
        pos = torch.tensor([2., 3., 4.], dtype=torch.float64)
        original = data.ab.weighted_size_aware_bce_loss(logits, targets, weights, pos)
        graphs = [dict(targets=targets[:2], weights=weights[:2]), dict(targets=targets[2:], weights=weights[2:])]
        denominator, shared_pos = p.virtual_batch_statistics(graphs, pos)
        accumulated = sum(p.bce_numerator(l, y, w, shared_pos) / denominator for l, y, w in
                          ((logits[:2], targets[:2], weights[:2]), (logits[2:], targets[2:], weights[2:])))
        torch.testing.assert_close(original, accumulated, atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(torch.autograd.grad(original, logits, retain_graph=True)[0], torch.autograd.grad(accumulated, logits)[0], atol=1e-9, rtol=1e-7)

    def test_streaming_cache_and_missing_view_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="production_contract_") as directory:
            path = Path(directory) / "cache.json"
            data.support.write_json_atomic(path, {"10": [], "2": [{"text": "가나다", "value": 1.23456}]})
            self.assertEqual(dict(data.iter_json_object(path, chunk_size=3)), json.loads(path.read_text()))
        record = data.ab.ImageRecord(1, "synthetic.jpg", Path("synthetic.jpg"), 100, 100)
        candidate = {"image_id": 1, "category_id": 1, "score": .5, "bbox": [1, 2, 10, 10]}
        with self.assertRaises(ValueError):
            data.validate_predictions([candidate], record, "coarse")

    def test_builder_invariance_no_gt_features_and_single_final_nms(self):
        args = data.approved_args()
        record = data.ab.ImageRecord(1, "synthetic.jpg", Path("synthetic.jpg"), 1000, 500)
        predictions = [dict(image_id=1, category_id=1, score=.7 - .01*i, bbox=[100+i, 100, 80, 44],
                            _view_type=2 if i % 2 == 0 else 3, _view_id=f"crop:{i}", _view_bbox=[0., 0., 640., 500.]) for i in range(4)]
        gt = [{"category_id": 1, "bbox": [100, 100, 80, 44], "area": 3520.0}]
        with patch.object(data.ab, "classwise_nms", wraps=data.ab.classwise_nms) as nms:
            first = data.build_graph(record, predictions, gt, args)
            second = data.build_graph(record, predictions, [], args)
            nms.assert_not_called()
            for field in ("x", "edge_index", "edge_attr", "support", "u", "mask"):
                self.assertTrue(torch.equal(first[field], second[field]), field)
            self.assertFalse(torch.equal(first["targets"], second["targets"]))
            nodes = data.ab.build_size_aware_detection_nodes(record, [], predictions, gt, args)
            for node in nodes:
                data.base.assign_legacy3head_probs(node, [.6, .7, .4])
            result = data.base.run_legacy3head_fixed_candidate_variant([dict(record=record, nodes=nodes, predictions=predictions)], args)
            self.assertEqual(nms.call_count, 1)
            self.assertEqual(len(first["x"]), 4)
            self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
