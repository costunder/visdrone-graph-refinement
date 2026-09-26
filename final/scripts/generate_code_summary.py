#!/usr/bin/env python3
"""Regenerate code_summary.md as repeated absolute-path/source blocks."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "code_summary.md"
FILES = (
    "final/experiment_1/run_experiment_1.sh",
    "final/scripts/map_local_variants.py",
    "final/manifests/experiment_variant_ids.json",
    "final/manifests/variant_order.json",
    "final/experiment_1/scripts/run_gois_paper_ablation_curves.py",
    "final/experiment_1/scripts/run_gois_two_stage_gnn_ablation.py",
    "final/scripts/my_package/visdrone_categories.py",
    "final/experiment_1/scripts/export_publication_results.py",
    "final/experiment_1/scripts/visualize_experiment1_graph_example.py",
    "final/scripts/generate_code_summary.py",
)


def main():
    blocks = []
    for relative in FILES:
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        blocks.append(f"# {path}\n{path.read_text().rstrip()}\n")
    OUTPUT.write_text("\n".join(blocks))
    print(f"wrote {OUTPUT} with {len(FILES)} files")


if __name__ == "__main__":
    main()
