#!/usr/bin/env python3
"""Export the Experiment-1 publication summary without model execution."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


VARIANT_ALIASES = {"01_gois_original_repo": "01_gois_reimplementation"}
GRAPH_SCHEMA_V2_MIN_CODE_VERSION = 48
EXPERIMENT_1_GRAPH_VARIANTS = {
    "02_gnn_no_cluster",
    "03_gnn_dbscan_cluster",
    "04_size_refinement_gnn",
    "05_gnn_prune_same_pool",
}

EXPERIMENT_VARIANTS = {
    1: {
        "00_full_inference": ("local_full_image_control", False),
        "01_gois_reimplementation": ("local_multiscale_reimplementation", True),
        "02_gnn_no_cluster": ("same_pool_graph_refinement", True),
        "03_gnn_dbscan_cluster": ("same_pool_graph_refinement", True),
        "04_size_refinement_gnn": ("same_pool_graph_refinement", True),
        "05_gnn_prune_same_pool": ("same_pool_graph_refinement", True),
    },
    2: {
        "00_gnn_conf_rescue": ("low_confidence_candidate_pool", False),
        "01_size_refinement_conf_rescue": ("low_confidence_candidate_pool", False),
        "02_low_conf_cluster_token_hgnn_refinement": (
            "lower_confidence_hypergraph_pool",
            False,
        ),
        "03_sparse_edge_sage": ("low_confidence_sparse_graph_pool", False),
        "04_ppr_gatv2_sage": ("low_confidence_sparse_graph_pool", False),
        "05_ppr_gatv2_sage_no_gois": ("full_image_no_gois_ablation", False),
    },
}


def parse_args():
    final_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("1",), default="1")
    parser.add_argument("--final-root", type=Path, default=final_root)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(final_root, experiment):
    return (
        final_root
        / f"experiment_{experiment}"
        / "runs/table6_yolo11_10class_extra_ablation/pipeline_ablation"
        / "evaluation_results_best_ap.csv"
    )


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def experiment_1_graph_schema(final_root):
    table_dir = (
        final_root
        / "experiment_1/runs/table6_yolo11_10class_extra_ablation/pipeline_ablation"
    )
    versions = {}
    for variant in sorted(EXPERIMENT_1_GRAPH_VARIANTS):
        config_path = table_dir / variant / "run_config.json"
        version = None
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text())
                version = int(payload.get("code", {}).get("version", 0))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                version = None
        versions[variant] = version
    current = all(
        version is not None and version >= GRAPH_SCHEMA_V2_MIN_CODE_VERSION
        for version in versions.values()
    )
    return {
        "name": "v2_pre_nms_typed_relations" if current else "v1_post_nms_historical",
        "current": current,
        "minimum_code_version": GRAPH_SCHEMA_V2_MIN_CODE_VERSION,
        "variant_code_versions": versions,
    }


def export_experiment(final_root, experiment):
    source = source_path(final_root, experiment)
    if not source.exists():
        raise FileNotFoundError(source)
    rows = read_rows(source)
    allowed = EXPERIMENT_VARIANTS[experiment]
    graph_schema = experiment_1_graph_schema(final_root) if experiment == 1 else None
    normalized = []
    for row in rows:
        item = dict(row)
        legacy_variant = item["variant"]
        variant = VARIANT_ALIASES.get(legacy_variant, legacy_variant)
        if variant not in allowed:
            raise ValueError(f"Experiment {experiment} source contains foreign variant: {variant}")
        group, directly_comparable = allowed[variant]
        is_graph_variant = experiment == 1 and variant in EXPERIMENT_1_GRAPH_VARIANTS
        if is_graph_variant:
            result_current = bool(graph_schema["current"])
            result_schema = graph_schema["name"]
        else:
            result_current = True
            result_schema = "not_applicable"
        item["variant"] = variant
        item["experiment"] = str(experiment)
        item["legacy_variant"] = legacy_variant if legacy_variant != variant else ""
        item["comparison_group"] = group
        item["graph_schema"] = result_schema
        item["publication_status"] = (
            "current" if result_current else "historical_retraining_required"
        )
        item["directly_comparable_to_01"] = str(
            directly_comparable and result_current
        ).lower()
        normalized.append(item)

    baseline = None
    if experiment == 1:
        baseline = next(
            (row for row in normalized if row["variant"] == "01_gois_reimplementation"),
            None,
        )
        if baseline is None:
            raise ValueError("Experiment 1 is missing 01_gois_reimplementation")
    baseline_ap = float(baseline["AP"]) if baseline is not None else None
    for row in normalized:
        row["delta_ap_vs_01"] = (
            f"{float(row['AP']) - baseline_ap:.6f}"
            if baseline_ap is not None and row["directly_comparable_to_01"] == "true"
            else ""
        )

    output_dir = final_root / f"experiment_{experiment}" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "publication_results_best_ap.csv"
    source_fields = list(rows[0].keys()) if rows else []
    extra_fields = [
        "experiment",
        "legacy_variant",
        "comparison_group",
        "graph_schema",
        "publication_status",
        "directly_comparable_to_01",
        "delta_ap_vs_01",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields + extra_fields)
        writer.writeheader()
        writer.writerows(normalized)

    metadata = {
        "experiment": experiment,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output_csv.resolve()),
        "gois_baseline_claim": "local_reimplementation" if experiment == 1 else None,
        "exact_paper_table_reproduction": False,
        "model_training_or_inference_performed": False,
        "graph_schema": graph_schema,
        "publication_ready": bool(graph_schema["current"]),
        "comparison_note": "Experiment 1 rows marked directly comparable use the fixed candidate pool.",
    }
    with (output_dir / "publication_results_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output_csv


def main():
    args = parse_args()
    selected = (1, 2) if args.experiment == "all" else (int(args.experiment),)
    for experiment in selected:
        print(export_experiment(args.final_root.resolve(), experiment))


if __name__ == "__main__":
    main()
