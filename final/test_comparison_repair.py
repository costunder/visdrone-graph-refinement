"""CPU/read-only checks for numbering, config rejection and execution holds."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FINAL = Path(__file__).resolve().parent
ROOT = FINAL.parent
spec = importlib.util.spec_from_file_location("comparison_check", FINAL / "scripts/check_pe_comparison.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
RUN = FINAL / "experiment_1/runs/graph_pe_production_v1/10/signnet/aligned/seed_42"


class ComparisonRepairTests(unittest.TestCase):
    def setUp(self):
        self.signnet = json.loads((RUN / "run_config.json").read_text())
        self.baseline = dict(self.signnet, mode="no_pe", pe_hidden=None,
                             pe_gin_layers=0, signnet_eigenvectors=None)

    def test_public_numbers_contiguous_without_extra_baseline(self):
        catalog = json.loads((FINAL / "manifests/current_experiments.json").read_text())
        for name, count in (("experiment_1", 8), ("experiment_2", 11)):
            rows = catalog["experiments"][name]
            self.assertEqual([row["id"] for row in rows], [f"{n:02d}" for n in range(count)])
            self.assertFalse(any("zero_pe" in row["name"] or "no_pe" in row["name"] for row in rows))
        self.assertEqual(catalog["experiments"]["experiment_3"], [])
        self.assertFalse(catalog["execution_authorized"])

    def test_cli_mapping_no_number_collision(self):
        mapper = str(FINAL / "scripts/map_local_variants.py")
        for experiment, ids, expected in (
            ("1", "06,07", "10_gnn_no_cluster_signnet_pe,11_gnn_no_cluster_rpearl_pe"),
            ("2", "09,10", "10_class_relation_ppr_signnet_pe,11_class_relation_ppr_rpearl_pe"),
        ):
            result = subprocess.run([sys.executable, "-B", mapper, "--experiment", experiment,
                                     "--scope", "pe", "--variants", ids], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)
        for experiment, token in (("1", "09"), ("1", "09_gnn_no_cluster_zero_pe"),
                                  ("2", "09_class_relation_ppr_zero_pe")):
            result = subprocess.run([sys.executable, "-B", mapper, "--experiment", experiment,
                                     "--scope", "pe", "--variants", token], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_only_encoder_intervention_allowed(self):
        result = check.compare_configs(self.baseline, self.signnet)
        self.assertTrue(result["config_comparable"], result)
        self.assertFalse(result["execution_authorized"])
        self.assertFalse(result["equal_capacity_claim"])
        for key, value in (("backbone_hidden", 96), ("backbone_layers", 3), ("epochs", 40),
                           ("seed", 43), ("train_coverage", "1024_graphs"),
                           ("val_coverage", "512_graphs"), ("input_identity_digest", "other"),
                           ("input_manifest_sha256", "other"), ("code", {"runner": "other"}),
                           ("checkpoint_selection", "best_of_every_epoch"),
                           ("virtual_batch_graphs", 16)):
            with self.subTest(field=key):
                changed = dict(self.baseline, **{key: value})
                self.assertFalse(check.compare_configs(changed, self.signnet)["config_comparable"])
        changed = copy.deepcopy(self.baseline)
        changed["optimizer"]["lr"] = .002
        self.assertFalse(check.compare_configs(changed, self.signnet)["config_comparable"])
        changed = dict(self.signnet, pe_gin_layers=3)
        self.assertFalse(check.compare_configs(self.baseline, changed)["config_comparable"])

    def test_historical_config_and_missing_fields_rejected(self):
        old = json.loads((FINAL / "experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation/02_gnn_no_cluster/run_config.json").read_text())
        self.assertFalse(check.compare_configs(old, self.signnet)["config_comparable"])
        for field in check.REQUIRED:
            changed = dict(self.baseline)
            changed.pop(field, None)
            self.assertFalse(check.compare_configs(changed, self.signnet)["config_comparable"], field)

    def test_completed_run_status_not_comparison_completion(self):
        progress = json.loads((RUN / "progress.json").read_text())
        selection = json.loads((RUN / "checkpoint_selection.json").read_text())
        primary = json.loads((RUN / "primary_result.json").read_text())
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["epochs"], 120)
        self.assertEqual(progress["evaluated_checkpoints"], 3)
        self.assertEqual(len(selection["checkpoints"]), 3)
        self.assertIn(primary["checkpoint"], selection["checkpoints"])
        approval = json.loads((FINAL / "manifests/graph_pe_production_proposal.json").read_text())
        self.assertTrue(approval["training_performed"])
        self.assertTrue(approval["evaluation_performed"])
        self.assertFalse(approval["training_authorized"])
        self.assertFalse(approval["evaluation_authorized"])

    def test_production_hold_before_any_work(self):
        import run_graph_pe_production as runner
        options = SimpleNamespace(mode="signnet", alignment="aligned", seed=42, device="cuda:0",
                                  allow_training=True, allow_evaluation=True, resume_epoch=None)
        with patch.dict(os.environ, {"ALLOW_MODEL_TRAINING": "1", "ALLOW_COCO_EVALUATION": "1"}):
            with patch.object(runner, "cpu_acceptance") as cpu, patch.object(runner.data, "prepare_inputs") as inputs:
                with self.assertRaises(RuntimeError):
                    runner.run(options)
                cpu.assert_not_called()
                inputs.assert_not_called()

    def test_experiment3_direct_entrypoints_fail_before_side_effects(self):
        sys.path.insert(0, str(FINAL / "experiment_3"))
        import run_experiment_3 as runner
        with patch.object(runner, "parse_args", return_value=SimpleNamespace(allow_training=True)):
            with self.assertRaisesRegex(RuntimeError, "stage order"):
                runner.main()
        with self.assertRaisesRegex(RuntimeError, "stage order"):
            runner.run_variant(*([None] * 8))
        with self.assertRaisesRegex(RuntimeError, "stage order"):
            runner.train_epoch(*([None] * 7))
        result = subprocess.run(["bash", str(FINAL / "experiment_3/run_experiment_3.sh")],
                                env={**os.environ, "ALLOW_MODEL_TRAINING": "1"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("blocked", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
