#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import torch

import sequential_exp1_exp2 as sequential


class SequentialExperimentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sequential.ab.configure_class_space("visdrone10")

    def test_shared_cache_assigns_stable_full_indices(self):
        records = [SimpleNamespace(image_id=1)]
        coarse = {"1": [{"score": 0.8}, {"score": 0.1}]}
        fine = {"1": [{"score": 0.5}]}

        combined = sequential.tag_shared_candidate_cache(records, coarse, fine)

        self.assertEqual(
            [item["_exp3_full_index"] for item in combined["1"]],
            [0, 1, 2],
        )
        self.assertEqual(
            [item["_candidate_source"] for item in combined["1"]],
            ["coarse", "coarse", "fine"],
        )

    def test_core_is_a_subset_of_the_same_pool(self):
        records = [SimpleNamespace(image_id=1)]
        shared = {
            "1": [
                {"score": 0.8, "_exp3_full_index": 0},
                {"score": 0.24, "_exp3_full_index": 1},
                {"score": 0.25, "_exp3_full_index": 2},
            ]
        }

        core = sequential.core_candidate_cache(records, shared, 0.25)

        self.assertEqual(
            [item["_exp3_full_index"] for item in core["1"]],
            [0, 2],
        )

    def test_prior_transfer_preserves_full_node_identity(self):
        full_nodes = [
            SimpleNamespace(source=1, det_index=0),
            SimpleNamespace(source=1, det_index=1),
        ]
        core_node = SimpleNamespace(
            source=1,
            det_index=0,
            gnn_obj=0.9,
            gnn_small=0.8,
            gnn_large=0.7,
            gnn_roi=0.85,
        )
        record = SimpleNamespace(image_id=1)
        full_samples = [{"record": record, "nodes": full_nodes}]
        core_samples = [
            {
                "record": record,
                "nodes": [core_node],
                "predictions": [{"_exp3_full_index": 1}],
            }
        ]

        audit = sequential.transfer_exp1_priors(core_samples, full_samples)

        self.assertEqual(audit["transferred_nodes"], 1)
        self.assertEqual(full_nodes[0].exp1_core_mask, 0.0)
        self.assertEqual(full_nodes[1].exp1_core_mask, 1.0)
        self.assertAlmostEqual(full_nodes[1].exp1_obj, 0.9)

    def test_zero_control_and_actual_prior_have_identical_shape(self):
        nodes = [
            SimpleNamespace(
                exp1_core_mask=1.0,
                exp1_obj=0.9,
                exp1_small=0.8,
                exp1_large=0.7,
                exp1_roi=0.85,
            ),
            SimpleNamespace(),
        ]
        base_features = torch.randn(2, sequential.exp2.node_dim(sequential.ab))

        zero = sequential.append_prior_features(
            base_features, nodes, sequential.PRIOR_ZERO
        )
        actual = sequential.append_prior_features(
            base_features, nodes, sequential.PRIOR_EXP1
        )

        self.assertEqual(zero.shape, actual.shape)
        self.assertEqual(zero.shape[1], sequential.node_dim())
        self.assertTrue(torch.equal(zero[:, :-sequential.PRIOR_DIM], base_features))
        self.assertTrue(torch.equal(actual[:, :-sequential.PRIOR_DIM], base_features))
        self.assertTrue(torch.count_nonzero(zero[:, -sequential.PRIOR_DIM:]) == 0)
        self.assertGreater(
            int(torch.count_nonzero(actual[:, -sequential.PRIOR_DIM:])), 0
        )

    def test_capacity_matched_stage2_model_forward(self):
        args = SimpleNamespace(
            gnn_hidden_dim=16,
            gnn_layers=2,
            exp2_attention_heads=4,
            exp2_dropout=0.0,
        )
        model = sequential.build_stage2_model(args)
        x = torch.zeros((3, sequential.node_dim()), dtype=torch.float32)
        class_offset = sequential.ab.NODE_CLASS_OFFSET
        x[0, class_offset + 0] = 1.0
        x[1, class_offset + 1] = 1.0
        x[2, class_offset + 0] = 1.0
        edge_index = torch.tensor(
            [[0, 1, 2, 0, 1], [0, 1, 2, 1, 2]], dtype=torch.long
        )
        edge_attr = torch.zeros(
            (edge_index.shape[1], sequential.edge_dim()), dtype=torch.float32
        )

        output = model(x, edge_index, edge_attr)

        self.assertEqual(tuple(output.shape), (3, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
