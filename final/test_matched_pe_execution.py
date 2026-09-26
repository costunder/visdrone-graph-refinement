"""CPU-only checks for matched revision routing and approval boundaries."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import os

import run_graph_pe_production as runner
from scripts.check_pe_comparison import compare_configs


def options(mode="no_pe", **kwargs):
    return SimpleNamespace(**{**dict(mode=mode, study="matched_pe_v2", alignment="aligned", seed=42,
                                   device="cuda:0", allow_training=True, allow_evaluation=True, resume_epoch=None), **kwargs})


class MatchedTests(unittest.TestCase):
    def test_public_baseline_path_and_no_overwrite(self):
        for mode, prefix in (("no_pe", "02_"), ("signnet", "06_"), ("rpearl", "07_")):
            path = runner.run_directory(options(mode))
            self.assertIn("matched_pe_v2", path.parts)
            self.assertTrue(path.parent.name.startswith(prefix))
            self.assertNotIn("09", path.parts)

    def test_reapproval_is_scoped(self):
        with patch.dict(os.environ, {"ALLOW_MODEL_TRAINING": "1", "ALLOW_COCO_EVALUATION": "1"}):
            runner.require_authorization(options())
            for opts in (options(seed=43), options(alignment="node_permuted_control"),
                         options(study="graph_pe_production_v1")):
                with self.assertRaises(RuntimeError):
                    runner.require_authorization(opts)

    def test_real_configurations_differ_only_by_encoder(self):
        manifest = json.loads((runner.data.CACHE_ROOT / "manifest.json").read_text())
        configs = {mode: runner.configuration(options(mode), manifest) for mode in ("no_pe", "signnet", "rpearl")}
        for mode in ("signnet", "rpearl"):
            result = compare_configs(configs["no_pe"], configs[mode])
            self.assertTrue(result["config_comparable"], result)
        self.assertEqual(configs["no_pe"]["backbone_hidden"], 256)
        self.assertEqual(configs["no_pe"]["backbone_layers"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
