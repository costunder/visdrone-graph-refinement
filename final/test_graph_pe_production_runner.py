"""CPU-only production runner guards, strict resume, scoring and variance audit."""
from __future__ import annotations

import copy
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import graph_pe_production as p
import graph_pe_production_data as data
import run_graph_pe_production as runner
from test_graph_pe_production_contract import fixture

torch.set_num_threads(2)


def args(**overrides):
    values = dict(mode="signnet", seed=42, alignment="aligned", device="cuda:0", allow_training=True, allow_evaluation=True, resume_epoch=None)
    return SimpleNamespace(**{**values, **overrides})


class RunnerTests(unittest.TestCase):
    def test_all_authorization_combinations_and_before_side_effects(self):
        # Current execution is on hold. Test historical positive authorization
        # with an explicit fixture, never by reopening the live approval file.
        approval_fixture = {"implementation_authorized": True, "training_authorized": True,
                            "gpu_validation_authorized": True, "evaluation_authorized": True,
                            "execution_authorized_experiments": ["experiment_1"]}
        for train_flag, eval_flag, train_env, eval_env in itertools.product((False, True), repeat=4):
            options = args(allow_training=train_flag, allow_evaluation=eval_flag)
            with patch.dict(os.environ, {"ALLOW_MODEL_TRAINING": str(int(train_env)), "ALLOW_COCO_EVALUATION": str(int(eval_env))}):
                if all((train_flag, eval_flag, train_env, eval_env)):
                    with patch.object(runner.APPROVAL.__class__, "read_text", return_value=json.dumps(approval_fixture)):
                        runner.require_authorization(options)
                else:
                    with patch.object(runner, "cpu_acceptance") as cpu, patch.object(data, "prepare_inputs") as inputs, patch.object(torch.cuda, "is_available") as cuda:
                        with self.assertRaises(RuntimeError):
                            runner.run(options)
                        cpu.assert_not_called()
                        inputs.assert_not_called()
                        cuda.assert_not_called()
        with patch.dict(os.environ, {"ALLOW_MODEL_TRAINING": "1", "ALLOW_COCO_EVALUATION": "1"}):
            with self.assertRaises(RuntimeError):
                runner.require_authorization(args(device="cpu"))
        with patch.object(runner.APPROVAL.__class__, "read_text", return_value=json.dumps({"implementation_authorized": False})):
            with patch.dict(os.environ, {"ALLOW_MODEL_TRAINING": "1", "ALLOW_COCO_EVALUATION": "1"}):
                with self.assertRaises(RuntimeError):
                    runner.require_authorization(args())

    def test_strict_full_model_optimizer_rng_resume_same_next_update(self):
        f = fixture(n=4)
        model = p.Experiment1Production(data.ab, "signnet", recompute=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        config = {"mode": "signnet", "seed": 42, "alignment": "aligned", "design": "synthetic_contract_only", "cache": "fixture", "code": "fixture"}
        def update(m, o):
            o.zero_grad(set_to_none=True)
            result = m(*f[:3], **f[3])
            # Exercise stochastic RNG restoration independently of E1's dropout=0.
            result = torch.nn.functional.dropout(result, p=.25, training=True)
            result.square().sum().backward()
            o.step()
        update(model, optimizer)
        with tempfile.TemporaryDirectory(prefix="production_resume_") as directory:
            path = Path(directory) / "epoch_001.pt"
            history = [dict(epoch=1, train_loss=.1, val_loss=.2)]
            runner.save_checkpoint(path, model, optimizer, 1, config, history)
            with self.assertRaises(RuntimeError):
                runner.save_checkpoint(path, model, optimizer, 1, config, history)
            update(model, optimizer)
            expected = copy.deepcopy(model.state_dict())
            state_expected = copy.deepcopy(optimizer.state_dict())
            fresh = p.Experiment1Production(data.ab, "signnet", recompute=False)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=.001, weight_decay=.0001)
            self.assertEqual(runner.load_checkpoint(path, fresh, fresh_optimizer, config, 1), history)
            update(fresh, fresh_optimizer)
            for name, value in fresh.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            for key, state in fresh_optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, state_expected["state"][key][name]), name)
            for field in ("mode", "seed", "design", "cache", "code"):
                changed = dict(config, **{field: "wrong"})
                with self.assertRaises(RuntimeError):
                    runner.load_checkpoint(path, fresh, fresh_optimizer, changed, 1)
            wrong_seed_model = p.Experiment1Production(data.ab, "signnet", seed=43, recompute=False)
            with self.assertRaises(RuntimeError):
                runner.load_checkpoint(path, wrong_seed_model, None, config, 1)
            with self.assertRaises(RuntimeError):
                runner.load_checkpoint(path, fresh, fresh_optimizer, config, 2)
            old = Path(directory) / "old.pt"
            data.support.save_torch_atomic({"schema": "graph_pe_training_v2"}, old)
            with self.assertRaises(RuntimeError):
                runner.load_checkpoint(old, fresh, fresh_optimizer, config, 1)

    def test_production_scoring_matches_original_and_one_nms(self):
        options = args()
        record = data.ab.ImageRecord(1, "fixture.jpg", Path("fixture.jpg"), 1000, 500)
        candidates = [dict(image_id=1, category_id=1, score=.3 + .1*i, bbox=[100+i, 100, 30, 40],
                           _view_type=2, _view_id=f"coarse:1:{i}", _view_bbox=[0., 0., 640., 500.]) for i in range(3)]
        builder = data.approved_args()
        graph = data.build_graph(record, candidates, [], builder)
        graph.update(predictions=candidates, split="eval")
        model = p.Experiment1Production(data.ab, "signnet", recompute=False).eval()
        with torch.no_grad():
            tensors, extra = runner.graph_arguments(graph, model, "cpu", None)
            probabilities = model(*tensors, **extra).sigmoid().tolist()
        nodes = data.ab.build_size_aware_detection_nodes(record, [], candidates, [], builder)
        for node, probability in zip(nodes, probabilities):
            data.base.assign_legacy3head_probs(node, probability)
        expected = data.base.run_legacy3head_fixed_candidate_variant([dict(predictions=candidates, nodes=nodes)], builder)
        # Only the compute device is CPU in this synthetic unit test, never a run.
        options.device = "cpu"
        with patch.object(runner, "require_authorization"), patch.object(runner, "read_graph", return_value=graph), patch.object(data.ab, "classwise_nms", wraps=data.ab.classwise_nms) as nms:
            actual = runner.predictions(model, {"splits": {"eval": [None]}}, options)
            self.assertEqual(nms.call_count, 1)
        self.assertEqual(actual, expected)

    def test_permuted_control_uses_same_parameters_and_active_encoder(self):
        f = fixture()
        aligned = p.Experiment1Production(data.ab, "signnet", recompute=False)
        control = p.Experiment1Production(data.ab, "signnet", alignment="node_permuted_control", recompute=False)
        self.assertEqual(aligned.assert_contract(), control.assert_contract())
        for key, value in aligned.state_dict().items():
            self.assertTrue(torch.equal(value, control.state_dict()[key]))
        control.gate.data.fill_(.2)
        permutation = p.alignment_permutation(torch.arange(7), seed=42, experiment="experiment_1", split="train", image_id=1)
        output = control(*f[:3], **f[3], permutation=permutation)
        output.square().sum().backward()
        for i, layer in enumerate(control.encoder.layers):
            self.assertGreater(int(torch.count_nonzero(layer.mlp[0].weight.grad)), 0, i)
        with self.assertRaises(ValueError):
            control(*f[:3], **f[3])


def variance_audit():
    """Full PE128/L8, 16 graphs, 8 independent banks, M120/240/480 diagnostics."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(p.stable_seed("rpearl_init", seed=42, experiment="experiment_1"))
        encoder = p.RPEARLM120().eval()
    rows = []
    with torch.no_grad():
        for index in range(16):
            n = 8 + index
            edges = torch.tensor([(i, (i+1) % n) for i in range(n)] + [(i, (i+3) % n) for i in range(0, n, 3)]).T
            support = p.legacy_ops.coalesced_undirected_support(edges, n)
            for samples in (120, 240, 480):
                outputs = []
                for bank in range(8):
                    w = p.probes(n, seed=42, experiment="variance_diagnostic", split="train", image_id=index, epoch=bank+1, samples=samples)
                    outputs.append(encoder(w, support, recompute=False, variance_diagnostic=True))
                stacked = torch.stack(outputs)
                rows.append({"graph": index, "nodes": n, "samples": samples, "independent_banks": 8,
                             "mean_element_variance": float(stacked.var(0, unbiased=True).mean()),
                             "max_element_variance": float(stacked.var(0, unbiased=True).max()),
                             "mean_output_l2": float(stacked.norm(dim=-1).mean())})
    return {"scope": "synthetic_variance_diagnostic_not_detection_performance", "rows": rows,
            "production_samples_unchanged": 120, "exact_topology_invariance_claim": False,
            "stability_claim": "withheld; finite-M variability is measured, not declared eliminated"}


if __name__ == "__main__":
    unittest.main(verbosity=2)
