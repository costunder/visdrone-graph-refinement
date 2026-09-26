#!/usr/bin/env python3
"""Map experiment-local variant selections to shared runner internal IDs."""

import argparse
import json
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[1] / "manifests/experiment_variant_ids.json"
PUBLIC_CATALOG = MANIFEST.with_name("current_experiments.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("1", "2"), required=True)
    parser.add_argument("--scope", choices=("existing", "followup", "pe"), default="existing")
    parser.add_argument("--variants", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with MANIFEST.open("r") as handle:
        config = json.load(handle)[f"experiment_{args.experiment}"]

    if args.scope == "pe":
        rows = json.loads(PUBLIC_CATALOG.read_text())["experiments"][f"experiment_{args.experiment}"]
        aliases = {}
        for row in rows:
            if "mode" not in row:
                continue
            for token in (row["id"], row["id"] + "_" + row["name"], row["legacy_variant"]):
                aliases[token] = row["legacy_variant"]
        selected = []
        for token in (part.strip() for part in args.variants.split(",")):
            if not token:
                continue
            if token not in aliases:
                raise ValueError(f"Unknown public PE variant {token!r}; see {PUBLIC_CATALOG}. Separate no-PE IDs are removed.")
            if aliases[token] not in selected:
                selected.append(aliases[token])
        if not selected:
            raise ValueError("No variants selected")
        print(",".join(selected))
        return

    internal_to_local = config["internal_to_local"]
    local_to_internal = {local: internal for internal, local in internal_to_local.items()}
    local_by_id = {name[:2]: name for name in config["local_order"]}
    allowed_ids = set(config["runner_scopes"].get(args.scope, []))
    selected = []
    for raw_token in args.variants.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token in local_by_id:
            local_name = local_by_id[token]
        elif token in local_to_internal:
            local_name = token
        elif token in internal_to_local:
            local_name = internal_to_local[token]
        else:
            raise ValueError(f"Unknown experiment-{args.experiment} local variant: {token}")
        local_id = local_name[:2]
        if local_id not in allowed_ids:
            raise ValueError(
                f"Variant {local_name} belongs to a different runner scope; "
                f"allowed {args.scope} IDs: {', '.join(sorted(allowed_ids))}"
            )
        internal = local_to_internal[local_name]
        if internal not in selected:
            selected.append(internal)
    if not selected:
        raise ValueError("No variants selected")
    print(",".join(selected))


if __name__ == "__main__":
    main()
