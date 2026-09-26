#!/usr/bin/env python3
"""Reconstruct the 6,471-image VisDrone 10-class train split."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


CLASS_TO_INDEX = {
    "pedestrians": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning_tricycle": 7,
    "bus": 8,
    "motor": 9,
}
IGNORED_LABELS = {"ignore_regions", "others"}
SOURCE_URL = "https://huggingface.co/datasets/Voxel51/VisDrone2019-DET"


def parse_args():
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples-json",
        type=Path,
        default=workspace / "Full/raw/VisDrone2019-DET-samples.json",
    )
    parser.add_argument(
        "--source-images",
        type=Path,
        action="append",
        default=None,
        help="Repeat for every directory containing a portion of official train images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace / "Full/data/visdrone_det_yolo_10class",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_index(directories):
    indexed = {}
    for directory in directories:
        for path in directory.iterdir():
            if not path.is_file():
                continue
            previous = indexed.get(path.name)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(f"Duplicate image filename: {path.name}")
            indexed[path.name] = path.resolve()
    return indexed


def clipped_yolo_box(box):
    x, y, width, height = [float(value) for value in box]
    x1 = min(1.0, max(0.0, x))
    y1 = min(1.0, max(0.0, y))
    x2 = min(1.0, max(0.0, x + width))
    y2 = min(1.0, max(0.0, y + height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        x2 - x1,
        y2 - y1,
    )


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def main():
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    source_dirs = args.source_images or [
        workspace / "GOIS/data/visdrone_yolo_full/images/train",
        workspace / "GOIS/data/visdrone_yolo_full/images/val",
    ]
    for directory in source_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not args.samples_json.is_file():
        raise FileNotFoundError(args.samples_json)

    with args.samples_json.open("r") as handle:
        samples = json.load(handle).get("samples", [])
    train_samples = [sample for sample in samples if "train" in sample.get("tags", [])]
    if len(train_samples) != 6471:
        raise ValueError(f"Expected 6471 official train samples, found {len(train_samples)}")

    available_images = image_index(source_dirs)
    expected_names = {Path(sample["filepath"]).name for sample in train_samples}
    missing = sorted(expected_names - set(available_images))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} train images; first: {missing[:5]}")

    image_output = args.output_root / "images/train"
    label_output = args.output_root / "labels/train"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()
    ignored_counts = Counter()
    clipped_or_invalid = 0
    for sample in train_samples:
        image_name = Path(sample["filepath"]).name
        source = available_images[image_name]
        image_link = image_output / image_name
        if image_link.exists() or image_link.is_symlink():
            if image_link.resolve() != source:
                raise FileExistsError(f"Unexpected existing image link: {image_link}")
        else:
            image_link.symlink_to(os.path.relpath(source, image_output))

        lines = []
        detections = sample.get("ground_truth", {}).get("detections", [])
        for detection in detections:
            label = str(detection.get("label", ""))
            if label in IGNORED_LABELS:
                ignored_counts[label] += 1
                continue
            if label not in CLASS_TO_INDEX:
                raise ValueError(f"Unknown VisDrone label: {label}")
            box = clipped_yolo_box(detection.get("bounding_box", []))
            if box is None:
                clipped_or_invalid += 1
                continue
            class_index = CLASS_TO_INDEX[label]
            class_counts[class_index] += 1
            lines.append(
                f"{class_index} " + " ".join(f"{value:.8f}" for value in box)
            )
        atomic_text(label_output / f"{Path(image_name).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))

    provenance = {
        "source": SOURCE_URL,
        "samples_json": str(args.samples_json.resolve()),
        "samples_json_sha256": sha256(args.samples_json),
        "train_samples": len(train_samples),
        "linked_images": len(expected_names),
        "class_to_index": CLASS_TO_INDEX,
        "class_counts": {str(index): class_counts[index] for index in range(10)},
        "ignored_counts": dict(ignored_counts),
        "clipped_or_invalid_boxes_skipped": clipped_or_invalid,
        "source_image_directories": [str(path.resolve()) for path in source_dirs],
    }
    provenance_path = args.output_root / "annotations/train_10class_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = provenance_path.with_suffix(provenance_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, provenance_path)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
