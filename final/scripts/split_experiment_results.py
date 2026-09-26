#!/usr/bin/env python3
"""Physically split legacy combined outputs into experiment 1 and 2 roots."""

import csv
import json
import shutil
from pathlib import Path
from renumber_experiment_variants import main as renumber_variants


WORKSPACE = Path(__file__).resolve().parents[2]
RUN_NAME = "table6_yolo11_10class_extra_ablation"
LEGACY_TABLE = WORKSPACE / "final/runs" / RUN_NAME / "pipeline_ablation"
GROUPS = {
    "experiment_1": {
        "title": "fixed_candidate_pool",
        "variants": [
            "00_full_inference",
            "01_gois_original_repo",
            "02_gnn_no_cluster",
            "03_gnn_dbscan_cluster",
            "05_size_refinement_gnn",
            "08_gnn_prune_same_pool",
        ],
        "extras": ["visual", "visual.zip"],
    },
    "experiment_2": {
        "title": "low_confidence_candidate_rescue",
        "variants": [
            "04_gnn_conf_rescue",
            "06_size_refinement_conf_rescue",
            "07_low_conf_cluster_token_hgnn_refinement",
            "09_exp2_sparse_edge_sage",
            "10_exp2_ppr_gatv2_sage",
            "11_exp2_ppr_gatv2_sage_no_gois",
        ],
        "extras": [],
    },
}
SUMMARY_FILES = [
    "epoch_metrics_all.csv",
    "evaluation_results_last_epoch.csv",
    "evaluation_results_best_ap.csv",
]


def group_table(group):
    return WORKSPACE / "final" / group / "runs" / RUN_NAME / "pipeline_ablation"


def normalize_variant(value):
    if value == "01_gois_reimplementation":
        return "01_gois_original_repo"
    return value


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r") as handle:
        return json.load(handle)


def move_artifacts():
    for group, spec in GROUPS.items():
        destination = group_table(group)
        destination.mkdir(parents=True, exist_ok=True)
        for name in spec["variants"] + spec["extras"]:
            source = LEGACY_TABLE / name
            target = destination / name
            if source.exists() and not target.exists():
                shutil.move(str(source), str(target))


def split_csv(source_path):
    if not source_path.exists():
        return
    with source_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for group, spec in GROUPS.items():
        allowed = set(spec["variants"])
        selected = [row for row in rows if normalize_variant(row.get("variant", "")) in allowed]
        destination = group_table(group) / source_path.name
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected)


def split_manifest():
    legacy = load_json(LEGACY_TABLE / "config_manifest.json")
    for group, spec in GROUPS.items():
        destination = group_table(group)
        destination_manifest = destination / "config_manifest.json"
        # On later runs the legacy manifest is gone. Preserve the already split
        # experiment settings instead of replacing them with a minimal payload.
        manifest = dict(legacy or load_json(destination_manifest))
        existing = [name for name in spec["variants"] if (destination / name).exists()]
        manifest.update(
            {
                "table_type": spec["title"],
                "experiment_group": group,
                "selected_variants": existing,
                "artifact_root": str(destination),
                "shared_detector_and_cache_root": str(WORKSPACE / "final/runs" / RUN_NAME),
                "separation_policy": (
                    "Experiment outputs are physically isolated. Only detector weights and raw "
                    "detector caches remain shared outside the experiment roots."
                ),
            }
        )
        destination.mkdir(parents=True, exist_ok=True)
        with destination_manifest.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")


def remove_legacy_combined_summaries():
    for name in SUMMARY_FILES + ["config_manifest.json"]:
        path = LEGACY_TABLE / name
        if path.exists():
            path.unlink()
    try:
        LEGACY_TABLE.rmdir()
    except OSError:
        pass


def main():
    if not LEGACY_TABLE.exists():
        renumber_variants()
        for group in GROUPS:
            print(f"{group}: {group_table(group)}")
        return
    move_artifacts()
    for name in SUMMARY_FILES:
        split_csv(LEGACY_TABLE / name)
    split_manifest()
    remove_legacy_combined_summaries()
    renumber_variants()
    for group in GROUPS:
        print(f"{group}: {group_table(group)}")


if __name__ == "__main__":
    main()
