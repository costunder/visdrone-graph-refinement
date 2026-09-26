#!/usr/bin/env python3
"""Read-only, fail-closed PE config comparison. No model or CUDA imports."""
import argparse
import json
from pathlib import Path

# Only the declared encoder intervention may differ. All other stored fields,
# including input identity, seed, code, backbone, budget and selection, must match.
PE_FIELDS = {"mode", "pe_hidden", "pe_gin_layers", "signnet_eigenvectors",
             "rpearl_samples", "rpearl_filter_terms"}
REQUIRED = {"study", "experiment", "mode", "seed", "code", "checkpoint_schema",
            "input_identity_digest", "input_manifest_sha256", "backbone_hidden",
            "backbone_layers", "epochs", "optimizer", "virtual_batch_graphs",
            "microbatch_graphs", "train_coverage", "val_coverage",
            "checkpoint_selection", "alignment", "dtype", "scheduler",
            "early_stopping", "gradient_clipping", "cache_only"} | PE_FIELDS
ENCODERS = {
    "no_pe": (None, 0, None, None, None),
    "signnet": (128, 8, 32, None, None),
    "rpearl": (128, 8, None, 120, 12),
}


def compare_configs(left, right):
    errors = []
    for side, config in (("left", left), ("right", right)):
        missing = sorted(REQUIRED - config.keys())
        if missing:
            errors.append({"side": side, "missing_fields": missing})
        mode = config.get("mode")
        actual = tuple(config.get(key) for key in (
            "pe_hidden", "pe_gin_layers", "signnet_eigenvectors", "rpearl_samples", "rpearl_filter_terms"))
        if mode not in ENCODERS or actual != ENCODERS.get(mode):
            errors.append({"side": side, "invalid_encoder_contract": mode})
        if any(config.get(key) != value for key, value in {
            "backbone_hidden": 256, "backbone_layers": 6, "epochs": 120,
            "virtual_batch_graphs": 32, "train_coverage": "all_nonempty_graphs_once_per_epoch",
            "val_coverage": "full_split", "alignment": "aligned", "cache_only": True,
            "dtype": "float32", "scheduler": None, "early_stopping": False,
            "gradient_clipping": None,
            "checkpoint_schema": "graph_pe_production_training_v1",
            "checkpoint_selection": "three_lowest_unrounded_val_loss_then_highest_AP_earliest_epoch_ties",
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001,
                          "betas": [.9, .999], "eps": 1e-8},
        }.items()):
            errors.append({"side": side, "invalid_common_production_contract": True})
        if config.get("seed") not in (42, 43, 44):
            errors.append({"side": side, "invalid_seed": config.get("seed")})
        if not config.get("code") or not config.get("input_identity_digest") or not config.get("input_manifest_sha256"):
            errors.append({"side": side, "missing_provenance_identity": True})
    if left.get("mode") == right.get("mode"):
        errors.append({"identical_modes_not_a_pe_comparison": True})
    for key in sorted((left.keys() | right.keys()) - PE_FIELDS):
        if key not in left or key not in right or left[key] != right[key]:
            errors.append({"field": key, "left": left.get(key), "right": right.get(key)})
    return {"config_comparable": not errors, "differences": errors,
            "scope": "stored_config_only_not_tensor_or_completion_audit",
            "execution_authorized": False, "equal_capacity_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare_configs(json.loads(args.left.read_text()), json.loads(args.right.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["config_comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
