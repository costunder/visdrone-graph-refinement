#!/usr/bin/env python3
"""Apply experiment-local variant numbering to existing artifacts."""

import csv
import json
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "table6_yolo11_10class_extra_ablation"
MAPPING_PATH = FINAL_ROOT / "manifests/experiment_variant_ids.json"


def load_mapping():
    with MAPPING_PATH.open("r") as handle:
        return json.load(handle)


def experiment_table(experiment):
    return (
        FINAL_ROOT
        / f"experiment_{experiment}"
        / "runs"
        / RUN_NAME
        / "pipeline_ablation"
    )


def replacement_map(config):
    replacements = dict(config["internal_to_local"])
    replacements.update(config.get("legacy_artifact_to_local", {}))
    return replacements


def replace_string(value, replacements):
    if value in replacements:
        return replacements[value]
    result = value
    for legacy, local in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(legacy, local)
    return result


def replace_json_value(value, replacements):
    if isinstance(value, dict):
        return {key: replace_json_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_json_value(item, replacements) for item in value]
    if isinstance(value, str):
        return replace_string(value, replacements)
    return value


def rename_variant_directories(table, replacements):
    for legacy, local in replacements.items():
        if legacy == local:
            continue
        source = table / legacy
        target = table / local
        if source.exists() and target.exists():
            raise FileExistsError(f"Cannot merge {source} into existing {target}")
        if source.exists():
            source.rename(target)


def rewrite_csv(path, replacements):
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        return
    rewritten = [
        {key: replace_string(value, replacements) for key, value in row.items()}
        for row in rows
    ]
    if rewritten == rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rewritten)


def rewrite_json(path, replacements):
    with path.open("r") as handle:
        payload = json.load(handle)
    rewritten = replace_json_value(payload, replacements)
    if rewritten == payload:
        return
    with path.open("w") as handle:
        json.dump(rewritten, handle, indent=2)
        handle.write("\n")


def annotate_group_manifest(table, experiment, config):
    path = table / "config_manifest.json"
    payload = {}
    if path.exists():
        with path.open("r") as handle:
            payload = json.load(handle)
    payload.update(
        {
            "experiment_group": f"experiment_{experiment}",
            "numbering_scope": "experiment_local",
            "local_variant_order": config["local_order"],
            "internal_to_local_variant": config["internal_to_local"],
        }
    )
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def renumber_experiment(experiment, config):
    table = experiment_table(experiment)
    table.mkdir(parents=True, exist_ok=True)
    replacements = replacement_map(config)
    rename_variant_directories(table, replacements)
    for path in sorted(table.rglob("*.csv")):
        rewrite_csv(path, replacements)
    for path in sorted(table.rglob("*.json")):
        rewrite_json(path, replacements)
    annotate_group_manifest(table, experiment, config)
    print(f"experiment_{experiment}: {', '.join(config['local_order'])}")


def main():
    mapping = load_mapping()
    renumber_experiment(1, mapping["experiment_1"])
    renumber_experiment(2, mapping["experiment_2"])


if __name__ == "__main__":
    main()
