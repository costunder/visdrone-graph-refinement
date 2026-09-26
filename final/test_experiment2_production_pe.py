"""Full-capacity CPU-only model conformance; not an E2 training run."""
import copy
from types import MethodType
import unittest

import torch
import graph_pe_production_data as data
from experiment_2 import production_pe_model as model
from test_graph_pe_production_contract import fixture

torch.set_num_threads(2)


def inputs(n=7, dtype=torch.float32):
    if n == 0:
        return (torch.empty(0, 40, dtype=dtype), torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, 27, dtype=dtype),
                dict(u=torch.empty(0, 32, dtype=dtype), mask=torch.empty(0, 32, dtype=torch.bool),
                     w=torch.empty(0, 120, dtype=dtype), support=torch.empty(2, 0, dtype=torch.long)))
    x, edges, attrs, extra = fixture(n, dtype)
    return torch.cat((x, x.new_zeros((n, 3))), -1), edges, torch.cat((attrs, attrs.new_full((len(attrs), 3), .1)), -1), extra


class E2ModelTests(unittest.TestCase):
    def test_full_capacity_shared_initialization_and_gate_identity(self):
        x, edges, attrs, extra = inputs()
        base = model.Experiment2Production(data.ab, "no_pe", recompute=False).eval()
        reference = base(x, edges, attrs)
        for mode, count in model.EXPECTED.items():
            candidate = model.Experiment2Production(data.ab, mode, recompute=False).eval()
            self.assertEqual(candidate.assert_contract(), count)
            for name, value in base.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, candidate.backbone.state_dict()[name]), name)
            actual = candidate(x, edges, attrs, **extra) if mode != "no_pe" else candidate(x, edges, attrs)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_chunking_matches_original_forward_and_parameter_gradients(self):
        x, edges, attrs, _ = inputs(dtype=torch.float64)
        candidate = model.Experiment2Production(data.ab, "no_pe", recompute=False, edge_chunk=3).double().eval()
        original = copy.deepcopy(candidate.backbone)
        for layer in original.layers:
            layer.forward = MethodType(model.sparse.PPRGATv2SAGELayer.forward, layer)
        left, right = candidate(x, edges, attrs), original(x, edges, attrs)
        torch.testing.assert_close(left, right, rtol=1e-7, atol=1e-9)
        left.square().sum().backward()
        right.square().sum().backward()
        for (name, param), (other, reference) in zip(candidate.backbone.named_parameters(), original.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(param.grad, reference.grad, rtol=1e-7, atol=1e-9, msg=name)

    def test_active_gradients_through_every_pe_and_backbone_layer(self):
        x, edges, attrs, extra = inputs()
        for mode in ("signnet", "rpearl"):
            candidate = model.Experiment2Production(data.ab, mode, recompute=True, edge_chunk=3)
            candidate.gate.data.fill_(.2)
            candidate(x, edges, attrs, **extra).square().sum().backward()
            for name, layer in [(f"pe{i}", layer) for i, layer in enumerate(candidate.encoder.layers)] + [(f"base{i}", layer) for i, layer in enumerate(candidate.backbone.layers)]:
                gradients = [p.grad for p in layer.parameters() if p.grad is not None]
                self.assertTrue(gradients, name)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
                self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients), name)

    def test_empty_graph_and_bad_inputs(self):
        x, edges, attrs, extra = inputs(n=0)
        for mode in model.EXPECTED:
            candidate = model.Experiment2Production(data.ab, mode, recompute=False).eval()
            output = candidate(x, edges, attrs, **extra)
            self.assertEqual(tuple(output.shape), (0, 3))
        with self.assertRaises(ValueError):
            model.Experiment2Production(data.ab, "signnet")(x, edges, attrs)

    def test_recomputation_preserves_dropout_rng_and_gradients(self):
        x, edges, attrs, _ = inputs()
        original = model.Experiment2Production(data.ab, "no_pe", recompute=False)
        candidate = model.Experiment2Production(data.ab, "no_pe", recompute=True)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(7654)
            expected = original(x, edges, attrs)
            expected.square().sum().backward()
            torch.default_generator.manual_seed(7654)
            actual = candidate(x, edges, attrs)
            actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (name, left), (_, right) in zip(original.named_parameters(), candidate.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-5, msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
