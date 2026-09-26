#!/usr/bin/env python3
"""CPU-only PE runner regressions. All artifact fixtures live in temporary dirs.

Detector inference, experiment training and COCO evaluation are never invoked.
Each runner is imported in its own subprocess to exercise its REAL module paths.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

FINAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINAL_DIR))
import graph_pe_run_support as support


class SupportTests(unittest.TestCase):
    def test_all_four_authorization_conditions(self):
        for train_flag, eval_flag, train_env, eval_env in itertools.product((False, True), repeat=4):
            args = SimpleNamespace(allow_training=train_flag, allow_evaluation=eval_flag)
            env = {"ALLOW_MODEL_TRAINING": str(int(train_env)), "ALLOW_COCO_EVALUATION": str(int(eval_env))}
            with patch.dict(os.environ, env):
                if all((train_flag, eval_flag, train_env, eval_env)):
                    support.require_run_authorization(args)
                else:
                    with self.assertRaises(RuntimeError):
                        support.require_run_authorization(args)

    def test_exact_top_three_and_unrounded_loss(self):
        rows = [{"epoch": e, "val_loss": v} for e, v in [(1, .100000049), (2, .1), (3, .2), (4, .9)]]
        self.assertEqual([e for _, e in support.selection_candidates(rows)], [2, 1, 3])
        for invalid in (rows[:2], rows + [rows[0]], [{"epoch": 1, "val_loss": float("nan")}] + rows[1:]):
            with self.assertRaises(ValueError):
                support.selection_candidates(invalid)
        with self.assertRaises(ValueError):
            support.selection_candidates(rows, 4)

    def test_selection_precision_ties_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="pe_selection_test_") as directory:
            root = Path(directory)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            rows = [support.loss_only_row("synthetic_only", e, .765432109876543, .123456789012345 if e < 4 else .9, ["AP", "AP50"]) for e in range(1, 5)]
            for row in rows:
                support.save_training_checkpoint_atomic(model, optimizer, row["epoch"], root / "checkpoints" / f"epoch_{row['epoch']:03d}.pt")
            plan = support.freeze_checkpoint_selection(root, rows)
            self.assertEqual([p["epoch"] for p in plan["checkpoints"]], [1, 2, 3])
            metrics = {"AP": .123456789012345, "AP50": .234567890123456}
            evaluator = Mock(return_value=metrics)
            fake_base = SimpleNamespace(ab=SimpleNamespace(evaluate_predictions=evaluator), METRIC_NAMES=["AP", "AP50"])
            for entry in plan["checkpoints"]:
                e = entry["epoch"]
                rows[e - 1] = support.record_evaluation(
                    fake_base, root, "synthetic_only", e, rows[e-1]["train_loss"], rows[e-1]["val_loss"],
                    [], "unused_mock_gt", plan,
                )
            self.assertEqual(evaluator.call_count, 3)
            support.save_history(root, rows, fake_base.METRIC_NAMES)
            with (root / "metrics_history.csv").open(newline="") as handle:
                reloaded = list(csv.DictReader(handle))
            self.assertEqual(float(reloaded[0]["AP"]), metrics["AP"])
            self.assertEqual(float(reloaded[0]["val_loss"]), rows[0]["val_loss"])
            self.assertEqual(reloaded[3]["AP"], "")
            self.assertFalse(list(root.glob("*.png")))
            result = support.write_primary_result(root, reloaded, plan)
            self.assertEqual(result["checkpoint"]["epoch"], 1)  # earliest AP tie
            self.assertEqual(result["row"]["AP"], metrics["AP"])
            tampered = copy.deepcopy(reloaded)
            tampered[0]["AP"] = .9
            with self.assertRaisesRegex(RuntimeError, "metric/history mismatch"):
                support.write_primary_result(root, tampered, plan)
            with (root / "metrics" / "epoch_001.json").open() as handle:
                self.assertEqual(json.load(handle)["metrics"], metrics)
            self.assertEqual(support.freeze_checkpoint_selection(root, reloaded), plan)
            with self.assertRaises(RuntimeError):
                support.record_evaluation(fake_base, root, "synthetic_only", 4, .1, .1, [], "unused", plan)
            self.assertEqual(evaluator.call_count, 3)
            contaminated = copy.deepcopy(rows)
            contaminated[3]["AP"] = .99
            with self.assertRaises(RuntimeError):
                support.freeze_checkpoint_selection(root, contaminated)
            with self.assertRaises(RuntimeError):
                support.write_primary_result(root, rows[:2], plan)
            changed = copy.deepcopy(rows)
            changed[0]["val_loss"] = .99
            with self.assertRaises(RuntimeError):
                support.freeze_checkpoint_selection(root, changed)
            support.save_torch_atomic({"changed": True}, root / "checkpoints" / "epoch_001.pt")
            with self.assertRaises(RuntimeError):
                support.verify_selected_checkpoint(plan, 1)

    def test_cache_only_preserves_order_and_detects_missing_or_changed_cache(self):
        with tempfile.TemporaryDirectory(prefix="pe_cache_test_") as directory:
            root = Path(directory)
            records = [SimpleNamespace(image_id=1)]
            audits = {}
            for split in ("train", "eval"):
                for source, score in (("coarse", .8), ("fine", .7)):
                    path = root / f"{split}_{source}.json"
                    support.write_json_atomic(path, {"1": [{"score": score, "bbox": [1, 2, 3, 4], "category_id": 1}]})
                    audits[f"{split}_{source}"] = support.validate_prediction_cache(path, records)
            gt = root / "gt.json"
            support.write_json_atomic(gt, {})
            args = SimpleNamespace(ground_truth_path=str(gt), train_images="unused", eval_images="unused", train_labels="unused")
            fake_ab = SimpleNamespace(
                build_image_records=Mock(return_value=records), load_gt_by_image=Mock(return_value={}),
                load_coco_gt_by_image=Mock(return_value={}),
                generate_or_load_coarse_cache=Mock(side_effect=AssertionError("inference fallback")),
            )
            for tagged in (False, True):
                context = support.load_cache_only_context(fake_ab, args, audits, tag_sources=tagged)
                for split in ("train", "eval"):
                    candidates = context[f"{split}_candidate_cache"]["1"]
                    self.assertEqual([p["score"] for p in candidates], [.8, .7])
                    self.assertEqual(len(candidates), 2)  # overlapping candidates must survive
                    if tagged:
                        self.assertEqual([p["_candidate_source"] for p in candidates], ["coarse", "fine"])
                    else:
                        self.assertTrue(all("_candidate_source" not in p for p in candidates))
            fake_ab.generate_or_load_coarse_cache.assert_not_called()
            with self.assertRaises(FileNotFoundError):
                support.validate_prediction_cache(root / "missing.json", records)
            bad = root / "bad.json"
            support.write_json_atomic(bad, {"2": []})
            with self.assertRaises(ValueError):
                support.validate_prediction_cache(bad, records)
            support.write_json_atomic(root / "train_coarse.json", {"1": []})
            with self.assertRaises(RuntimeError):
                support.load_cache_only_context(fake_ab, args, audits, tag_sources=False)

    def test_old_checkpoint_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pe_schema_test_") as directory:
            model = torch.nn.Linear(2, 1)
            path = Path(directory) / "old.pt"
            support.save_torch_atomic({"checkpoint_schema": "graph_pe_training_v1"}, path)
            with self.assertRaises(RuntimeError):
                support.load_training_checkpoint(model, None, path, "cpu", 1)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                support.loss_only_row("synthetic_only", 1, .1, value, ["AP"])


def import_runner(experiment):
    directory = FINAL_DIR / f"experiment_{experiment}"
    sys.path.insert(0, str(directory))
    path = directory / "run_graph_pe_ablation.py"
    spec = importlib.util.spec_from_file_location(f"pe_runner_test_{experiment}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_configuration_args(runner, experiment):
    # Parse the same options supplied by the shell entrypoint without executing it.
    dataset = FINAL_DIR.parent / "Full/data/visdrone_det_yolo_10class"
    shared = FINAL_DIR / "runs/table6_yolo11_10class_extra_ablation"
    argv = [
        "test", "--output_root", str(FINAL_DIR / f"experiment_{experiment}/runs/table6_yolo11_10class_extra_ablation"),
        "--model_path", str(shared / "detector/weights/best.pt"),
        "--train_images", str(dataset / "images/train"), "--train_labels", str(dataset / "labels/train"),
        "--eval_images", str(dataset / "images/val"), "--eval_labels", str(dataset / "labels/val"),
        "--ground_truth_path", str(dataset / "annotations/val_coco_gt.json"),
        "--class_space", "visdrone10", "--device", "cpu", "--disable_large_preserve",
        "--require_pyg", "--cache", "--epochs", str(120 if experiment == 1 else 40),
        "--coarse_conf", ".25" if experiment == 1 else ".05",
        "--fine_conf", ".25" if experiment == 1 else ".05",
        "--fine_slice_size", "256", "--fine_overlap", ".2", "--final_nms_iou", ".4",
        "--stage1_keep_conf", ".25",
        "--source_cache_dir" if experiment == 1 else "--common_cache_dir", str(shared / "common_cache"),
    ]
    with patch.object(sys, "argv", argv):
        return runner.parse_args()


def runner_tests(experiment):
    runner = import_runner(experiment)
    expected_scripts = FINAL_DIR / "experiment_1/scripts" if experiment == 1 else FINAL_DIR / "scripts"
    assert Path(runner.ab.__file__).parent == expected_scripts
    assert Path(runner.base.__file__).parent == expected_scripts
    args = legacy_configuration_args(runner, experiment)
    runner.assert_legacy_configuration_contract(args)
    runner.ab.configure_class_space("visdrone10")

    class RunnerTests(unittest.TestCase):
        def test_guards_precede_model_cache_or_gpu_work(self):
            minimal = SimpleNamespace(pe_variants="09", allow_training=True, allow_evaluation=True)
            with patch.object(runner, "parse_args", return_value=minimal), patch.dict(
                os.environ, {"ALLOW_MODEL_TRAINING": "0", "ALLOW_COCO_EVALUATION": "0"}
            ), patch.object(runner, "preflight_detector_caches") as preflight, patch.object(runner.graph_pe_model, "build_model") as build:
                with self.assertRaises(RuntimeError):
                    runner.main()
                with self.assertRaises(RuntimeError):
                    runner.run_variant(None, "09", "zero", [], [], minimal, "cpu", {})
                preflight.assert_not_called()
                build.assert_not_called()

        def test_legacy_configuration_rejects_shrink_or_policy_changes(self):
            for key, value in (("gnn_hidden_dim", 32), ("gnn_layers", 1), ("selection_top_k", 4), ("eval_every", 1), ("final_nms_iou", .5)):
                altered = copy.copy(args)
                setattr(altered, key, value)
                with self.assertRaises(RuntimeError):
                    runner.assert_legacy_configuration_contract(altered)

        def test_legacy_runner_stays_blocked_after_separate_production_approval(self):
            minimal = SimpleNamespace(pe_variants="09,10,11", allow_training=True, allow_evaluation=True)
            with patch.object(runner, "parse_args", return_value=minimal), patch.dict(
                os.environ, {"ALLOW_MODEL_TRAINING": "1", "ALLOW_COCO_EVALUATION": "1"}
            ), patch.object(runner, "preflight_detector_caches") as preflight, patch.object(
                runner.graph_pe_model, "build_model"
            ) as build, patch.object(runner.ab, "set_seed") as seed, patch.object(
                torch.cuda, "is_available"
            ) as cuda_probe, patch.object(runner.run_support, "exclusive_variant_lock") as lock:
                # No real experiment starts: every path must raise before any
                # model, input, device, optimizer or output-directory work.
                with self.assertRaisesRegex(RuntimeError, "Legacy graph-PE 96x3 execution is disabled"):
                    runner.main()
                for variant, mode in runner.PE_VARIANTS.items():
                    with self.assertRaisesRegex(RuntimeError, "Legacy graph-PE 96x3 execution is disabled"):
                        runner.run_variant(None, variant, mode, [], [], minimal, "cpu", {})
                with self.assertRaisesRegex(RuntimeError, "Legacy graph-PE 96x3 execution is disabled"):
                    if experiment == 1:
                        runner.train_epoch(None, [], minimal, None, "cpu", 1, None)
                    else:
                        runner.train_epoch(None, [], minimal, None, "cpu", 1)
                preflight.assert_not_called()
                build.assert_not_called()
                seed.assert_not_called()
                cuda_probe.assert_not_called()
                lock.assert_not_called()

        def test_real_tensor_path_and_one_final_nms(self):
            ab = runner.ab
            record = ab.ImageRecord(1, "synthetic.jpg", Path("synthetic.jpg"), 1000, 500)
            predictions = []
            for i, (view, bbox, score) in enumerate([
                (ab.VIEW_COARSE, [100, 100, 80, 44], .72),
                (ab.VIEW_FINE, [105, 102, 76, 42], .68),
                (ab.VIEW_COARSE, [108, 101, 78, 43], .51),
                (ab.VIEW_FINE, [760, 350, 52, 36], .60),
            ]):
                predictions.append({
                    "image_id": 1, "category_id": 2 if i == 2 else 1, "bbox": bbox, "score": score,
                    "_view_type": view, "_view_id": f"synthetic:{i}",
                    "_view_bbox": [0, 0, 1000, 500],
                    "_candidate_source": "coarse" if view == ab.VIEW_COARSE else "fine",
                })
            sample_args = copy.copy(args)
            sample_args.size_graph_cluster_mode = "none"
            if experiment == 2:
                sample_args.gnn_score_alpha = 1.0
                samples = runner.sparse_runner.build_samples([record], {1: []}, {"1": predictions}, sample_args, "low_conf_multiscale")
            else:
                samples = ab.make_size_aware_detection_samples([record], {}, {"1": predictions}, {1: []}, sample_args)
            sample = samples[0]
            self.assertEqual(len(sample["nodes"]), len(predictions))
            self.assertEqual([n.det_index for n in sample["nodes"]], list(range(4)))
            if experiment == 1:
                baseline = list(ab.build_size_aware_tensors(sample["nodes"], record, sample_args))
                baseline[3] = baseline[3][:, :3].contiguous()
            else:
                baseline = runner.exp2.build_sparse_graph_tensors(ab, sample, sample_args, include_ppr=True)
            reference = None
            for variant, mode in runner.PE_VARIANTS.items():
                local_samples = copy.deepcopy(samples)
                augmented = runner.tensors(local_samples[0], sample_args)
                self.assertTrue(torch.equal(augmented[0][:, :baseline[0].shape[1]], baseline[0]))
                for actual, expected in zip(augmented[1:], baseline[1:]):
                    self.assertTrue(torch.equal(actual, expected))
                if reference is not None:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(augmented, reference)))
                reference = augmented
                torch.manual_seed(42)
                model = runner.graph_pe_model.build_model(ab, sample_args, mode).eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(*augmented[:3]), model.backbone(*baseline[:3])))
                with patch.object(ab, "classwise_nms", wraps=ab.classwise_nms) as nms:
                    runner.attach_scores(local_samples, model, sample_args, torch.device("cpu"))
                    nms.assert_not_called()
                    if experiment == 1:
                        runner.base.run_legacy3head_fixed_candidate_variant(local_samples, sample_args)
                    else:
                        runner.sparse_runner.refined_predictions(local_samples, sample_args)
                    self.assertEqual(nms.call_count, 1)
            self.assertTrue(all(not getattr(args, flag) for flag in ("allow_training", "allow_evaluation")))

    return unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.experiment:
        suite = runner_tests(args.experiment)
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SupportTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.experiment is None:
        for experiment in (1, 2):
            subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--experiment", str(experiment)],
                check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60,
            )
        print("graph PE runner contract: PASS (support + isolated Experiment 1/2)")
    else:
        print(f"Experiment {args.experiment} actual-import runner contract: PASS")


if __name__ == "__main__":
    main()
