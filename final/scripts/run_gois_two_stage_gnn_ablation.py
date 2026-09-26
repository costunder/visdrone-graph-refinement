import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_package.visdrone_categories import (
    VISDRONE_6_CATEGORIES,
    VISDRONE_10_CATEGORIES,
    VISDRONE_EVAL_CATEGORIES,
    model_class_to_visdrone,
)


YOLO_TO_VISDRONE = {
    0: 1,   # pedestrian
    1: 3,   # bicycle
    2: 4,   # car
    3: 6,   # truck
    4: 9,   # bus
    5: 10,  # motor
}
CLASS_SPACES = {
    "visdrone6": YOLO_TO_VISDRONE,
    "visdrone10": {index: index + 1 for index in range(10)},
}
VISDRONE_TO_CLASS_INDEX = {category_id: index for index, category_id in YOLO_TO_VISDRONE.items()}
NUM_CLASSES = len(YOLO_TO_VISDRONE)
SMALL_MEDIUM_AREA_THR = 96.0 * 96.0
SOURCE_FULL = 0
SOURCE_SLICE = 1
VIEW_UNKNOWN = 0
VIEW_FULL = 1
VIEW_COARSE = 2
VIEW_FINE = 3
VIEW_ROI = 4
NUM_VIEW_TYPES = 5
NODE_LEGACY_FEATURE_DIM = 16
NODE_VIEW_TYPE_OFFSET = NODE_LEGACY_FEATURE_DIM
NODE_VIEW_GEOMETRY_OFFSET = NODE_VIEW_TYPE_OFFSET + NUM_VIEW_TYPES
NODE_VIEW_GEOMETRY_DIM = 6
NODE_CLASS_OFFSET = NODE_VIEW_GEOMETRY_OFFSET + NODE_VIEW_GEOMETRY_DIM
SIZE_AWARE_NODE_DIM = NODE_CLASS_OFFSET + NUM_CLASSES
EDGE_GEOMETRY_DIM = 16
EDGE_RELATION_OFFSET = EDGE_GEOMETRY_DIM
EDGE_RELATION_NAMES = (
    "self",
    "spatial",
    "overlap",
    "containment",
    "same_cluster",
    "same_view",
    "cross_view",
    "cross_class_context",
)
EDGE_RELATION_DIM = len(EDGE_RELATION_NAMES)
SIZE_AWARE_EDGE_DIM = EDGE_RELATION_OFFSET + EDGE_RELATION_DIM
SIZE_AWARE_OUTPUT_DIM = 4
STAGE2_ACTION_DIM = 4
STAGE2_AUX_DIM = 2
STAGE2_ACTION_OUTPUT_DIM = STAGE2_ACTION_DIM + STAGE2_AUX_DIM
STAGE2_ACTION_TARGET_DIM = 1 + STAGE2_AUX_DIM
STAGE2_SIZE_OUTPUT_DIM = 2
STAGE2_SIZE_TARGET_DIM = 2
STAGE2_COMPONENT_TYPE_DIM = 6
STAGE2_EXTRA_NODE_DIM = SIZE_AWARE_OUTPUT_DIM + 4 + STAGE2_COMPONENT_TYPE_DIM
STAGE2_NODE_DIM = SIZE_AWARE_NODE_DIM + STAGE2_EXTRA_NODE_DIM
STAGE2_KEEP = 0
STAGE2_DUPLICATE = 1
STAGE2_FRAGMENT = 2
STAGE2_BACKGROUND = 3
CACHE_SCHEMA_VERSION = "gois_graph_schema_v4"
_HDBSCAN_FALLBACK_WARNED = False


def configure_class_space(class_space):
    global YOLO_TO_VISDRONE
    global VISDRONE_TO_CLASS_INDEX
    global NUM_CLASSES
    global SIZE_AWARE_NODE_DIM
    global STAGE2_NODE_DIM
    global VISDRONE_EVAL_CATEGORIES

    if class_space not in CLASS_SPACES:
        raise ValueError(f"Unknown class_space {class_space}. Known: {', '.join(sorted(CLASS_SPACES))}")
    YOLO_TO_VISDRONE = dict(CLASS_SPACES[class_space])
    VISDRONE_TO_CLASS_INDEX = {category_id: index for index, category_id in YOLO_TO_VISDRONE.items()}
    NUM_CLASSES = len(YOLO_TO_VISDRONE)
    SIZE_AWARE_NODE_DIM = NODE_CLASS_OFFSET + NUM_CLASSES
    STAGE2_NODE_DIM = SIZE_AWARE_NODE_DIM + STAGE2_EXTRA_NODE_DIM
    VISDRONE_EVAL_CATEGORIES = VISDRONE_10_CATEGORIES if class_space == "visdrone10" else VISDRONE_6_CATEGORIES


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int


@dataclass
class ClusterNode:
    node_id: int
    category_id: int
    detection_indices: list
    bbox: list
    center: tuple
    count: int
    mean_score: float
    max_score: float
    mean_det_area: float
    union_area: float
    density: float
    dbscan_clustered: float
    target: float = 0.0
    weight: float = 0.25
    stage1_prob: float = 0.0
    stage2_prob: float = 0.0
    stage2_obj_prob: float = 0.0
    stage2_small_prob: float = 0.0
    stage2_large_prob: float = 0.0
    stage2_roi_prob: float = 0.0
    source: int = SOURCE_SLICE
    det_index: int = 0
    dbscan_label: int = -1
    target_obj: float = 0.0
    target_small: float = 0.0
    target_large: float = 0.0
    target_roi: float = 0.0
    gnn_obj: float = 0.0
    gnn_small: float = 0.0
    gnn_large: float = 0.0
    gnn_roi: float = 0.0
    best_match_iou: float = 0.0
    best_match_area: float = 0.0
    stage2_target_keep: float = 0.0
    stage2_target_duplicate: float = 0.0
    stage2_target_fragment: float = 0.0
    stage2_target_roi_refine: float = 0.0
    stage2_target_large_preserve: float = 0.0
    stage2_action_weight: float = 1.0
    stage2_keep: float = 0.0
    stage2_duplicate: float = 0.0
    stage2_fragment: float = 0.0
    stage2_background: float = 0.0
    stage2_roi_refine: float = 0.0
    stage2_large_preserve: float = 0.0
    stage2_small: float = 0.0
    stage2_large: float = 0.0



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(device_arg):
    if device_arg is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_text = str(device_arg)
    if device_text.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{device_text}")
    if device_text.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_text)
    return torch.device("cpu")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def image_files_from_dir(images_dir):
    return sorted(
        path for path in Path(images_dir).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def build_image_records(images_dir, gt_json_path=None, max_images=0):
    image_paths = {path.name: path for path in image_files_from_dir(images_dir)}
    records = []

    if gt_json_path and Path(gt_json_path).exists():
        with open(gt_json_path, "r") as f:
            gt = json.load(f)
        for image_info in sorted(gt["images"], key=lambda item: item["id"]):
            file_name = image_info["file_name"]
            if file_name not in image_paths:
                continue
            records.append(
                ImageRecord(
                    image_id=int(image_info["id"]),
                    file_name=file_name,
                    path=image_paths[file_name],
                    width=int(image_info["width"]),
                    height=int(image_info["height"]),
                )
            )
    else:
        for index, image_path in enumerate(sorted(image_paths.values()), start=1):
            with Image.open(image_path) as image:
                width, height = image.size
            records.append(
                ImageRecord(
                    image_id=index,
                    file_name=image_path.name,
                    path=image_path,
                    width=width,
                    height=height,
                )
            )

    if max_images and max_images > 0:
        records = records[:max_images]
    return records


def load_yolo_gt_for_record(record, labels_dir):
    label_path = Path(labels_dir) / f"{Path(record.file_name).stem}.txt"
    gt = []
    if not label_path.exists():
        return gt
    with label_path.open("r") as f:
        for raw_line in f:
            values = raw_line.strip().split()
            if len(values) < 5:
                continue
            yolo_class = int(float(values[0]))
            if yolo_class not in YOLO_TO_VISDRONE:
                continue
            x_center = float(values[1]) * record.width
            y_center = float(values[2]) * record.height
            width = float(values[3]) * record.width
            height = float(values[4]) * record.height
            x_min = x_center - width / 2.0
            y_min = y_center - height / 2.0
            gt.append(
                {
                    "category_id": YOLO_TO_VISDRONE[yolo_class],
                    "bbox": clip_xywh([x_min, y_min, width, height], record.width, record.height),
                    "area": max(0.0, width) * max(0.0, height),
                }
            )
    return gt


def generate_coco_gt_from_yolo(records, labels_dir, output_path):
    annotations = []
    annotation_id = 1
    for record in records:
        for gt in load_yolo_gt_for_record(record, labels_dir):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": record.image_id,
                    "category_id": gt["category_id"],
                    "bbox": gt["bbox"],
                    "area": gt["area"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    data = {
        "images": [
            {
                "id": record.image_id,
                "file_name": record.file_name,
                "width": record.width,
                "height": record.height,
            }
            for record in records
        ],
        "annotations": annotations,
        "categories": VISDRONE_EVAL_CATEGORIES,
    }
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return output_path


def filter_coco_gt(gt_path, records, output_path):
    record_ids = {record.image_id for record in records}
    with open(gt_path, "r") as f:
        data = json.load(f)
    filtered = {
        "images": [image for image in data["images"] if image["id"] in record_ids],
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in record_ids],
        "categories": data.get("categories", VISDRONE_EVAL_CATEGORIES),
    }
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
    return output_path


def load_gt_by_image(records, labels_dir):
    return {record.image_id: load_yolo_gt_for_record(record, labels_dir) for record in records}


def load_coco_gt_by_image(records, gt_path):
    record_by_id = {int(record.image_id): record for record in records}
    gt_by_image = {int(record.image_id): [] for record in records}
    with open(gt_path, "r") as f:
        data = json.load(f)
    valid_categories = {int(category["id"]) for category in data.get("categories", VISDRONE_EVAL_CATEGORIES)}
    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id not in record_by_id:
            continue
        if int(ann.get("iscrowd", 0)):
            continue
        category_id = int(ann.get("category_id", -1))
        if category_id not in valid_categories:
            continue
        record = record_by_id[image_id]
        bbox = clip_xywh(ann.get("bbox", [0, 0, 0, 0]), record.width, record.height)
        area = bbox_area(bbox)
        if area <= 0:
            continue
        gt_by_image[image_id].append(
            {
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
            }
        )
    return gt_by_image


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_xywh(box, width, height):
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    x2 = min(max(0.0, x2), float(width))
    y2 = min(max(0.0, y2), float(height))
    return xyxy_to_xywh([x1, y1, x2, y2])


def bbox_area(box):
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def bbox_center(box):
    return (float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0)


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def containment(inner, outer):
    ix1, iy1, ix2, iy2 = xywh_to_xyxy(inner)
    ox1, oy1, ox2, oy2 = xywh_to_xyxy(outer)
    cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
    cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    area = bbox_area(inner)
    if area <= 0:
        return 0.0
    return inter / area


def point_in_xywh(point, box):
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def expand_xywh(box, width, height, margin):
    x, y, w, h = box
    margin = float(margin)
    if margin > 1.0:
        pad_x = margin
        pad_y = margin
    else:
        pad_x = w * margin
        pad_y = h * margin
    return clip_xywh([x - pad_x, y - pad_y, w + 2.0 * pad_x, h + 2.0 * pad_y], width, height)


def union_xywh(boxes):
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    xyxy = [xywh_to_xyxy(box) for box in boxes]
    x1 = min(box[0] for box in xyxy)
    y1 = min(box[1] for box in xyxy)
    x2 = max(box[2] for box in xyxy)
    y2 = max(box[3] for box in xyxy)
    return xyxy_to_xywh([x1, y1, x2, y2])


def classwise_nms(predictions, iou_threshold=0.5, limit=None):
    if not predictions:
        return []
    kept = []
    for category_id in sorted({pred["category_id"] for pred in predictions}):
        class_preds = [pred for pred in predictions if pred["category_id"] == category_id]
        class_preds.sort(key=lambda pred: pred.get("score", 0.0), reverse=True)
        while class_preds:
            best = class_preds.pop(0)
            kept.append(best)
            class_preds = [
                pred for pred in class_preds
                if iou_xywh(best["bbox"], pred["bbox"]) <= iou_threshold
            ]
    kept.sort(key=lambda pred: pred.get("score", 0.0), reverse=True)
    if limit is not None:
        kept = kept[:limit]
    return kept


def convert_yolo_results_to_predictions(results, model, image_id, x_offset=0.0, y_offset=0.0, image_width=None, image_height=None):
    predictions = []
    for box in results.boxes:
        x_min, y_min, x_max, y_max = [float(v) for v in box.xyxy[0].tolist()]
        yolo_class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        category_id = YOLO_TO_VISDRONE.get(yolo_class_id)
        if category_id is None:
            category = model_class_to_visdrone(model.names, yolo_class_id)
            if category is not None:
                category_id = int(category["id"])
        if category_id is None:
            continue
        bbox = [x_min + x_offset, y_min + y_offset, x_max - x_min, y_max - y_min]
        if image_width is not None and image_height is not None:
            bbox = clip_xywh(bbox, image_width, image_height)
        if bbox_area(bbox) <= 0:
            continue
        predictions.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox_area(bbox),
                "iscrowd": 0,
                "score": confidence,
            }
        )
    return predictions


def predict_full_image(model, record, conf, iou, max_det, device):
    results = model.predict(
        str(record.path),
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )[0]
    return convert_yolo_results_to_predictions(
        results,
        model,
        record.image_id,
        image_width=record.width,
        image_height=record.height,
    )


def iter_slices(width, height, slice_size, overlap):
    step = max(1, int(slice_size * (1.0 - overlap)))
    def starts(length):
        if length <= slice_size:
            return [0]
        values = list(range(0, length - slice_size + 1, step))
        final_start = length - slice_size
        if values[-1] != final_start:
            values.append(final_start)
        return sorted(set(values))

    x_starts = starts(width)
    y_starts = starts(height)
    for y in y_starts:
        for x in x_starts:
            x_end = min(x + slice_size, width)
            y_end = min(y + slice_size, height)
            if x_end > x and y_end > y:
                yield int(x), int(y), int(x_end), int(y_end)


def predict_sliced_image(model, record, conf, iou, max_det, device, slice_size, overlap, batch_size=32):
    predictions = []
    crops = []
    offsets = []
    with Image.open(record.path).convert("RGB") as image:
        for x1, y1, x2, y2 in iter_slices(record.width, record.height, slice_size, overlap):
            crops.append(image.crop((x1, y1, x2, y2)))
            offsets.append((x1, y1))

        batch_size = max(1, int(batch_size))
        for index in range(0, len(crops), batch_size):
            crop_batch = crops[index : index + batch_size]
            offset_batch = offsets[index : index + batch_size]
            results_list = model.predict(
                crop_batch,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )
            for results, (x1, y1) in zip(results_list, offset_batch):
                predictions.extend(
                    convert_yolo_results_to_predictions(
                        results,
                        model,
                        record.image_id,
                        x_offset=x1,
                        y_offset=y1,
                        image_width=record.width,
                        image_height=record.height,
                    )
                )
    return classwise_nms(predictions, iou_threshold=iou, limit=max_det)


def predict_roi_sliced(model, record, roi, conf, iou, max_det, device, slice_size, overlap):
    roi = clip_xywh(roi, record.width, record.height)
    x, y, w, h = roi
    if w <= 1 or h <= 1:
        return []
    predictions = []
    with Image.open(record.path).convert("RGB") as image:
        crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
        crop_width, crop_height = crop.size
        if crop_width <= slice_size and crop_height <= slice_size:
            results = model.predict(
                crop,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )[0]
            predictions.extend(
                convert_yolo_results_to_predictions(
                    results,
                    model,
                    record.image_id,
                    x_offset=x,
                    y_offset=y,
                    image_width=record.width,
                    image_height=record.height,
                )
            )
        else:
            for sx1, sy1, sx2, sy2 in iter_slices(crop_width, crop_height, slice_size, overlap):
                sub_crop = crop.crop((sx1, sy1, sx2, sy2))
                results = model.predict(
                    sub_crop,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=device,
                    verbose=False,
                )[0]
                predictions.extend(
                    convert_yolo_results_to_predictions(
                        results,
                        model,
                        record.image_id,
                        x_offset=x + sx1,
                        y_offset=y + sy1,
                        image_width=record.width,
                        image_height=record.height,
                    )
                )
    return classwise_nms(predictions, iou_threshold=iou, limit=max_det)


def file_light_signature(path):
    path = Path(path)
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    except OSError:
        return {"path": str(path), "missing": True}


def fine_roi_cache_path(args):
    root = Path(getattr(args, "output_root", "data/gois_detection_gnn_40e_full"))
    model_sig = file_light_signature(getattr(args, "model_path", ""))
    signature_payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "class_space": getattr(args, "class_space", "visdrone6"),
        "class_map": YOLO_TO_VISDRONE,
        "model": model_sig,
        "fine_conf": float(getattr(args, "fine_conf", 0.0)),
        "model_iou": float(getattr(args, "model_iou", 0.0)),
        "max_det": int(getattr(args, "max_det", 0)),
        "slice_size": int(getattr(args, "fine_slice_size", 0)),
        "overlap": float(getattr(args, "fine_overlap", 0.0)),
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return root / "common_cache" / f"fine_roi_predictions_v1_{signature}.json"


def fine_roi_cache_enabled(args):
    return bool(getattr(args, "fine_roi_cache", True))


def fine_roi_cache_key(record, roi, args):
    clipped = clip_xywh(roi, record.width, record.height)
    rounded_roi = [int(round(float(value))) for value in clipped]
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "class_space": getattr(args, "class_space", "visdrone6"),
        "class_map": YOLO_TO_VISDRONE,
        "image_id": int(record.image_id),
        "file": str(record.file_name),
        "path": str(record.path),
        "width": int(record.width),
        "height": int(record.height),
        "image_file": file_light_signature(record.path),
        "roi": rounded_roi,
        "model_path": str(getattr(args, "model_path", "")),
        "model_file": file_light_signature(getattr(args, "model_path", "")),
        "conf": float(getattr(args, "fine_conf", 0.0)),
        "iou": float(getattr(args, "model_iou", 0.0)),
        "max_det": int(getattr(args, "max_det", 0)),
        "slice_size": int(getattr(args, "fine_slice_size", 0)),
        "overlap": float(getattr(args, "fine_overlap", 0.0)),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def get_fine_roi_cache(args):
    if not fine_roi_cache_enabled(args):
        return {}
    cache = getattr(args, "_fine_roi_prediction_cache", None)
    if cache is not None:
        return cache
    path = fine_roi_cache_path(args)
    cache = {}
    if getattr(args, "cache", False) and path.exists() and not getattr(args, "force_predictions", False):
        try:
            with path.open("r") as f:
                cache = json.load(f)
            print(f"[cache] using fine ROI cache: {path}", flush=True)
        except Exception as exc:
            print(f"[WARN] could not load fine ROI cache {path}: {exc}", flush=True)
            cache = {}
    setattr(args, "_fine_roi_prediction_cache", cache)
    setattr(args, "_fine_roi_prediction_cache_dirty", 0)
    return cache


def flush_fine_roi_cache(args, force=False):
    if not fine_roi_cache_enabled(args):
        return
    dirty = int(getattr(args, "_fine_roi_prediction_cache_dirty", 0))
    if dirty <= 0 and not force:
        return
    cache = getattr(args, "_fine_roi_prediction_cache", None)
    if cache is None:
        return
    path = fine_roi_cache_path(args)
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(cache, f)
    tmp_path.replace(path)
    setattr(args, "_fine_roi_prediction_cache_dirty", 0)


def _append_roi_crop(crops, metas, key, record, crop, x_offset, y_offset):
    crop = crop.copy()
    crops.append(crop)
    metas.append((key, record, float(x_offset), float(y_offset)))


def _flush_roi_crop_batch(model, crops, metas, predictions_by_key, args, progress):
    if not crops:
        return
    results_list = model.predict(
        crops,
        conf=args.fine_conf,
        iou=args.model_iou,
        max_det=args.max_det,
        device=args.device,
        batch=len(crops),
        verbose=False,
    )
    for results, (key, record, x_offset, y_offset) in zip(results_list, metas):
        predictions_by_key[key].extend(
            convert_yolo_results_to_predictions(
                results,
                model,
                record.image_id,
                x_offset=x_offset,
                y_offset=y_offset,
                image_width=record.width,
                image_height=record.height,
            )
        )
    progress.update(1)
    crops.clear()
    metas.clear()


def predict_roi_sliced_batch(model, roi_items, args):
    predictions_by_key = {key: [] for key, _, _ in roi_items}
    crops = []
    metas = []
    items_by_path = defaultdict(list)
    for key, record, roi in roi_items:
        items_by_path[str(record.path)].append((key, record, roi))

    batch_size = max(1, int(getattr(args, "fine_infer_batch_size", 1)))
    with tqdm(desc="fine ROI batch inference", leave=False, unit="batch") as progress:
        for path, items in items_by_path.items():
            with Image.open(path).convert("RGB") as image:
                for key, record, roi in items:
                    roi = clip_xywh(roi, record.width, record.height)
                    x, y, w, h = roi
                    if w <= 1 or h <= 1:
                        continue
                    crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
                    crop_width, crop_height = crop.size
                    if crop_width <= args.fine_slice_size and crop_height <= args.fine_slice_size:
                        _append_roi_crop(crops, metas, key, record, crop, x, y)
                        if len(crops) >= batch_size:
                            _flush_roi_crop_batch(model, crops, metas, predictions_by_key, args, progress)
                    else:
                        for sx1, sy1, sx2, sy2 in iter_slices(crop_width, crop_height, args.fine_slice_size, args.fine_overlap):
                            sub_crop = crop.crop((sx1, sy1, sx2, sy2))
                            _append_roi_crop(crops, metas, key, record, sub_crop, x + sx1, y + sy1)
                            if len(crops) >= batch_size:
                                _flush_roi_crop_batch(model, crops, metas, predictions_by_key, args, progress)
        _flush_roi_crop_batch(model, crops, metas, predictions_by_key, args, progress)

    for key, predictions in list(predictions_by_key.items()):
        predictions_by_key[key] = classwise_nms(predictions, iou_threshold=args.model_iou, limit=args.max_det)
    return predictions_by_key


def prefetch_fine_roi_predictions(model, plans, args):
    cache = get_fine_roi_cache(args)
    missing = []
    seen = set()
    for plan in plans:
        record = plan["record"]
        plan.setdefault("fine_roi_key_scores", {})
        for roi, roi_score in plan["rois"]:
            key = fine_roi_cache_key(record, roi, args)
            plan["fine_roi_keys"].append(key)
            plan["fine_roi_key_scores"][key] = max(
                float(plan["fine_roi_key_scores"].get(key, 0.0)),
                float(roi_score),
            )
            if key in cache or key in seen:
                continue
            seen.add(key)
            missing.append((key, record, roi))
    if missing:
        print(f"[fine ROI] cache_miss={len(missing)} batch_size={getattr(args, 'fine_infer_batch_size', 1)}", flush=True)
        predictions_by_key = predict_roi_sliced_batch(model, missing, args)
        cache.update(predictions_by_key)
        dirty = int(getattr(args, "_fine_roi_prediction_cache_dirty", 0)) + len(predictions_by_key)
        setattr(args, "_fine_roi_prediction_cache_dirty", dirty)
        flush_every = int(getattr(args, "fine_roi_cache_flush_every", 256))
        if flush_every > 0 and dirty >= flush_every:
            flush_fine_roi_cache(args)
    else:
        print("[fine ROI] cache_hit=all", flush=True)


def fine_roi_predictions_for_plan(plan, args):
    cache = get_fine_roi_cache(args)
    predictions = []
    key_scores = plan.get("fine_roi_key_scores", {})
    for key in plan["fine_roi_keys"]:
        roi_score = float(key_scores.get(key, 0.0))
        for pred in cache.get(key, []):
            adjusted = dict(pred)
            adjusted["_roi_score"] = roi_score
            predictions.append(adjusted)
    return predictions


def record_cache_signature(records, args, mode):
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "class_space": getattr(args, "class_space", "visdrone6"),
        "class_map": YOLO_TO_VISDRONE,
        "mode": mode,
        "model_path": str(getattr(args, "model_path", "")),
        "model_file": file_light_signature(getattr(args, "model_path", "")),
        "model_iou": float(getattr(args, "model_iou", 0.0)),
        "max_det": int(getattr(args, "max_det", 0)),
        "image_ids": [int(record.image_id) for record in records],
        "files": [str(record.file_name) for record in records],
        "sizes": [[int(record.width), int(record.height)] for record in records],
        "image_files": [file_light_signature(record.path) for record in records],
    }
    if mode == "coarse":
        payload.update(
            {
                "conf": float(getattr(args, "coarse_conf", 0.0)),
                "slice_size": int(getattr(args, "coarse_slice_size", 0)),
                "overlap": float(getattr(args, "coarse_overlap", 0.0)),
            }
        )
    elif mode == "full":
        payload.update({"conf": float(getattr(args, "full_conf", 0.0))})
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:12]


def prediction_cache_path(cache_dir, split_name, records, args):
    signature = record_cache_signature(records, args, "coarse")
    suffix = (
        f"{split_name}_coarse_conf{args.coarse_conf:.3f}_s{args.coarse_slice_size}"
        f"_o{args.coarse_overlap:.2f}_n{len(records) or 'all'}_{signature}.json"
    )
    return Path(cache_dir) / suffix


def generate_or_load_coarse_cache(model, records, args, split_name, cache_dir):
    cache_path = prediction_cache_path(cache_dir, split_name, records, args)
    if args.cache and cache_path.exists():
        print(f"[cache] using coarse cache: {cache_path}", flush=True)
        for attempt in range(30):
            try:
                with cache_path.open("r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                if attempt == 29:
                    raise
                import time

                time.sleep(2.0)

    data = {}
    for record in tqdm(records, desc=f"{split_name} coarse detector cache"):
        data[str(record.image_id)] = predict_sliced_image(
            model,
            record,
            conf=args.coarse_conf,
            iou=args.model_iou,
            max_det=args.max_det,
            device=args.device,
            slice_size=args.coarse_slice_size,
            overlap=args.coarse_overlap,
            batch_size=getattr(args, "fine_infer_batch_size", 32),
        )
    ensure_dir(cache_path.parent)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, cache_path)
    return data


def full_prediction_cache_path(cache_dir, split_name, records, args):
    signature = record_cache_signature(records, args, "full")
    suffix = f"{split_name}_full_conf{args.full_conf:.3f}_n{len(records) or 'all'}_{signature}.json"
    return Path(cache_dir) / suffix


def generate_or_load_full_cache(model, records, args, split_name, cache_dir):
    cache_path = full_prediction_cache_path(cache_dir, split_name, records, args)
    if args.cache and cache_path.exists():
        print(f"[cache] using full cache: {cache_path}", flush=True)
        with cache_path.open("r") as f:
            return json.load(f)

    data = {}
    for record in tqdm(records, desc=f"{split_name} full-image cache"):
        data[str(record.image_id)] = predict_full_image(
            model,
            record,
            conf=args.full_conf,
            iou=args.model_iou,
            max_det=args.max_det,
            device=args.device,
        )
    ensure_dir(cache_path.parent)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, cache_path)
    return data




def dbscan_labels(points, eps, min_samples):
    if len(points) == 0:
        return []
    labels = [-99] * len(points)
    cluster_id = 0

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    for index in range(len(points)):
        if labels[index] != -99:
            continue
        neighbors = np.where(distances[index] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[index] = -1
            continue

        labels[index] = cluster_id
        queue = deque(neighbors)
        while queue:
            neighbor = queue.popleft()
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            if labels[neighbor] != -99:
                continue
            labels[neighbor] = cluster_id
            neighbor_neighbors = np.where(distances[neighbor] <= eps)[0].tolist()
            if len(neighbor_neighbors) >= min_samples:
                queue.extend(neighbor_neighbors)
        cluster_id += 1

    return labels


def classwise_dbscan_detection_groups(predictions, eps, min_samples):
    groups = []
    next_cluster_id = 0
    by_class = defaultdict(list)
    for index, pred in enumerate(predictions):
        by_class[int(pred["category_id"])].append(index)

    for category_id, indices in by_class.items():
        points = np.array(
            [bbox_center(predictions[index]["bbox"]) for index in indices],
            dtype=np.float32,
        )
        labels = dbscan_labels(points, eps=eps, min_samples=min_samples)
        clustered = defaultdict(list)

        for local_index, label in enumerate(labels):
            global_index = indices[local_index]
            if label < 0:
                groups.append((category_id, -1, [global_index]))
            else:
                clustered[int(label)].append(global_index)

        for group_indices in clustered.values():
            groups.append((category_id, next_cluster_id, group_indices))
            next_cluster_id += 1

    return groups




def dbscan_labels_for_predictions(predictions, args):
    labels = [-1] * len(predictions)
    for _, cluster_id, indices in classwise_dbscan_detection_groups(
        predictions,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    ):
        if cluster_id < 0:
            continue
        for index in indices:
            labels[index] = int(cluster_id)
    return labels


def hdbscan_point_labels(points, args):
    global _HDBSCAN_FALLBACK_WARNED
    if len(points) == 0:
        return []
    min_cluster_size = max(2, int(getattr(args, "hdbscan_min_cluster_size", 3)))
    min_samples = max(1, int(getattr(args, "hdbscan_min_samples", 2)))
    if len(points) < min_cluster_size:
        return [-1] * len(points)
    try:
        from sklearn.cluster import HDBSCAN

        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                copy=False,
            )
        except TypeError:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
        return clusterer.fit_predict(points).astype(int).tolist()
    except Exception as exc:
        if not _HDBSCAN_FALLBACK_WARNED:
            print(f"[WARN] HDBSCAN unavailable; falling back to DBSCAN: {exc}", flush=True)
            _HDBSCAN_FALLBACK_WARNED = True
        return dbscan_labels(points, eps=args.dbscan_eps, min_samples=min_samples)


def hdbscan_labels_for_predictions(predictions, args):
    labels = [-1] * len(predictions)
    next_cluster_id = 0
    by_class = defaultdict(list)
    for index, pred in enumerate(predictions):
        by_class[pred["category_id"]].append(index)

    for _, indices in by_class.items():
        points = np.array([bbox_center(predictions[index]["bbox"]) for index in indices], dtype=np.float32)
        local_labels = hdbscan_point_labels(points, args)
        local_to_global = {}
        for local_index, local_label in enumerate(local_labels):
            if local_label < 0:
                continue
            if local_label not in local_to_global:
                local_to_global[local_label] = next_cluster_id
                next_cluster_id += 1
            labels[indices[local_index]] = local_to_global[local_label]
    return labels


def best_gt_match(prediction, gt_boxes):
    best_gt = None
    best_iou = 0.0
    pred_category = int(prediction["category_id"])
    for gt in gt_boxes:
        if int(gt["category_id"]) != pred_category:
            continue
        iou = iou_xywh(prediction["bbox"], gt["bbox"])
        if iou > best_iou:
            best_gt = gt
            best_iou = iou
    return best_gt, best_iou


def small_roi_support_match(bbox, category_id, gt_boxes, record, args):
    if bool(getattr(args, "disable_roi_support_target", False)):
        return None, 0.0
    roi_iou = float(getattr(args, "roi_label_iou", 0.10))
    center_margin = float(getattr(args, "roi_center_margin", 0.75))
    best_gt = None
    best_iou = 0.0
    pred_center = bbox_center(bbox)
    for gt in gt_boxes:
        if int(gt["category_id"]) != int(category_id):
            continue
        if float(gt.get("area", bbox_area(gt["bbox"]))) >= SMALL_MEDIUM_AREA_THR:
            continue
        iou = iou_xywh(bbox, gt["bbox"])
        expanded_gt = expand_xywh(gt["bbox"], record.width, record.height, center_margin)
        expanded_pred = expand_xywh(bbox, record.width, record.height, center_margin)
        gt_center = bbox_center(gt["bbox"])
        supports_roi = (
            iou >= roi_iou
            or point_in_xywh(pred_center, expanded_gt)
            or point_in_xywh(gt_center, expanded_pred)
        )
        if supports_roi and iou >= best_iou:
            best_gt = gt
            best_iou = iou
    return best_gt, best_iou


def build_size_aware_detection_nodes(record, full_predictions, slice_predictions, gt_boxes, args):
    nodes = []
    cluster_mode = getattr(args, "size_graph_cluster_mode", "none")
    if cluster_mode == "hdbscan":
        slice_labels = hdbscan_labels_for_predictions(slice_predictions, args)
    elif cluster_mode == "dbscan":
        slice_labels = dbscan_labels_for_predictions(slice_predictions, args)
    else:
        slice_labels = [-1] * len(slice_predictions)
    label_iou = float(getattr(args, "label_iou", 0.50))

    entries = []
    for index, prediction in enumerate(full_predictions):
        entries.append((SOURCE_FULL, index, prediction, -1))
    for index, prediction in enumerate(slice_predictions):
        entries.append((SOURCE_SLICE, index, prediction, slice_labels[index]))

    # COCO AP treats duplicate detections for the same GT as false positives.
    # Use a greedy one-to-one training target so duplicated boxes are not all
    # labeled positive just because they overlap the same object.
    match_by_entry = {}
    used_gt_indices = set()
    scored_entries = sorted(
        enumerate(entries),
        key=lambda item: float(item[1][2].get("score", 0.0)),
        reverse=True,
    )
    for entry_index, (_, _, prediction, _) in scored_entries:
        bbox = clip_xywh(prediction["bbox"], record.width, record.height)
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in used_gt_indices:
                continue
            if int(gt["category_id"]) != int(prediction["category_id"]):
                continue
            iou = iou_xywh(bbox, gt["bbox"])
            if iou > best_iou:
                best_gt_index = gt_index
                best_iou = iou
        if best_gt_index is not None and best_iou >= label_iou:
            used_gt_indices.add(best_gt_index)
            match_by_entry[entry_index] = (gt_boxes[best_gt_index], best_iou)

    def add_node(entry_index, source, det_index, prediction, dbscan_label=-1):
        bbox = clip_xywh(prediction["bbox"], record.width, record.height)
        area = max(1.0, bbox_area(bbox))
        score = float(prediction.get("score", 0.0))
        gt, match_iou = match_by_entry.get(entry_index, (None, 0.0))
        best_gt, best_iou = best_gt_match({"category_id": prediction["category_id"], "bbox": bbox}, gt_boxes)
        is_positive = gt is not None and match_iou >= label_iou
        target_area = float(gt["area"]) if gt is not None else area
        is_small = target_area < SMALL_MEDIUM_AREA_THR
        is_large = target_area >= SMALL_MEDIUM_AREA_THR
        roi_gt, _ = small_roi_support_match(bbox, int(prediction["category_id"]), gt_boxes, record, args)
        is_roi_support = roi_gt is not None

        weight = 1.0
        if is_positive and is_small:
            weight = 3.0
        elif is_positive and is_large:
            weight = 2.5
        elif is_roi_support:
            weight = 2.0
        elif score >= args.stage1_keep_conf:
            weight = 1.5

        node = ClusterNode(
            node_id=len(nodes),
            category_id=int(prediction["category_id"]),
            detection_indices=[det_index],
            bbox=bbox,
            center=bbox_center(bbox),
            count=1,
            mean_score=score,
            max_score=score,
            mean_det_area=area,
            union_area=area,
            density=1.0 / area,
            dbscan_clustered=1.0 if dbscan_label >= 0 else 0.0,
            target=1.0 if is_positive else 0.0,
            weight=weight,
            source=source,
            det_index=det_index,
            dbscan_label=int(dbscan_label),
            target_obj=1.0 if is_positive else 0.0,
            target_small=1.0 if is_positive and is_small else 0.0,
            target_large=1.0 if is_positive and is_large else 0.0,
            target_roi=1.0 if is_roi_support else 0.0,
            best_match_iou=float(best_iou),
            best_match_area=float(best_gt["area"]) if best_gt is not None else 0.0,
        )
        nodes.append(node)

    for entry_index, (source, det_index, prediction, dbscan_label) in enumerate(entries):
        add_node(entry_index, source, det_index, prediction, dbscan_label)
    assign_stage2_action_targets(nodes, args)
    return nodes


def assign_stage2_action_targets(nodes, args):
    label_iou = float(getattr(args, "label_iou", 0.50))
    fragment_containment_thr = float(getattr(args, "stage2_fragment_containment_thr", 0.30))
    fragment_iou_thr = float(getattr(args, "stage2_fragment_iou_thr", 0.10))

    for node in nodes:
        node.stage2_target_keep = float(node.target_obj)
        node.stage2_target_duplicate = 1.0 if node.best_match_iou >= label_iou and node.target_obj < 0.5 else 0.0
        node.stage2_target_fragment = 0.0
        node.stage2_target_roi_refine = float(node.target_roi)
        node.stage2_target_large_preserve = 1.0 if node.source == SOURCE_FULL and node.target_large > 0.5 else 0.0
        node.stage2_action_weight = 1.0
        if node.stage2_target_keep > 0.5:
            node.stage2_action_weight = 3.0
        elif node.stage2_target_duplicate > 0.5:
            node.stage2_action_weight = 2.0
        elif node.stage2_target_roi_refine > 0.5:
            node.stage2_action_weight = 2.0

    if bool(getattr(args, "disable_large_preserve", False)):
        full_large_nodes = [
            node
            for node in nodes
            if node.source == SOURCE_FULL
            and node.union_area >= SMALL_MEDIUM_AREA_THR
            and node.target_large > 0.5
        ]
    else:
        full_large_nodes = [
            node
            for node in nodes
            if node.source == SOURCE_FULL
            and node.union_area >= SMALL_MEDIUM_AREA_THR
            and (node.target_large > 0.5 or node.max_score >= float(getattr(args, "large_keep_conf", 0.25)))
        ]
    for node in nodes:
        if node.source != SOURCE_SLICE:
            continue
        for full_node in full_large_nodes:
            if node.category_id != full_node.category_id:
                continue
            if (
                containment(node.bbox, full_node.bbox) >= fragment_containment_thr
                or iou_xywh(node.bbox, full_node.bbox) >= fragment_iou_thr
            ):
                node.stage2_target_fragment = 1.0
                node.stage2_action_weight = max(float(node.stage2_action_weight), 2.5)
                break


def size_aware_node_feature(node, record, neighbor_density):
    x, y, w, h = node.bbox
    cx, cy = node.center
    image_area = max(1.0, float(record.width * record.height))
    area = max(1.0, float(node.union_area))
    one_hot = [0.0] * NUM_CLASSES
    class_index = VISDRONE_TO_CLASS_INDEX.get(node.category_id, 0)
    one_hot[class_index] = 1.0
    return [
        float(node.max_score),
        cx / max(1.0, float(record.width)),
        cy / max(1.0, float(record.height)),
        w / max(1.0, float(record.width)),
        h / max(1.0, float(record.height)),
        math.sqrt(area / image_area),
        math.log((w + 1.0) / (h + 1.0)),
        1.0 if area < SMALL_MEDIUM_AREA_THR else 0.0,
        1.0 if area >= SMALL_MEDIUM_AREA_THR else 0.0,
        1.0 if node.source == SOURCE_FULL else 0.0,
        1.0 if node.source == SOURCE_SLICE else 0.0,
        1.0 if node.dbscan_label >= 0 else 0.0,
        min(1.0, float(neighbor_density) / 16.0),
        math.log1p(area) / math.log1p(image_area),
        x / max(1.0, float(record.width)),
        y / max(1.0, float(record.height)),
        *one_hot,
    ]


def size_aware_edge_feature(src, dst, record, args):
    src_center = np.array(src.center, dtype=np.float32)
    dst_center = np.array(dst.center, dtype=np.float32)
    dist = float(np.linalg.norm(src_center - dst_center))
    dist_score = max(0.0, 1.0 - dist / max(1.0, float(args.graph_radius)))
    src_w = max(1.0, float(src.bbox[2]))
    src_h = max(1.0, float(src.bbox[3]))
    dst_w = max(1.0, float(dst.bbox[2]))
    dst_h = max(1.0, float(dst.bbox[3]))
    dx_norm = (float(dst.center[0]) - float(src.center[0])) / math.sqrt(src_w * dst_w)
    dy_norm = (float(dst.center[1]) - float(src.center[1])) / math.sqrt(src_h * dst_h)
    log_area_ratio = math.log((src.union_area + 1.0) / (dst.union_area + 1.0))
    log_w_ratio = math.log(src_w / dst_w)
    log_h_ratio = math.log(src_h / dst_h)
    scale_sim = max(0.0, 1.0 - abs(log_area_ratio) / 3.0)
    same_class = 1.0 if src.category_id == dst.category_id else 0.0
    same_cluster = 1.0 if src.dbscan_label >= 0 and src.dbscan_label == dst.dbscan_label else 0.0
    overlap = iou_xywh(src.bbox, dst.bbox)
    contain_sd = containment(src.bbox, dst.bbox)
    contain_ds = containment(dst.bbox, src.bbox)
    cross_source = 1.0 if src.source != dst.source else 0.0
    both_slice = 1.0 if src.source == SOURCE_SLICE and dst.source == SOURCE_SLICE else 0.0
    both_full = 1.0 if src.source == SOURCE_FULL and dst.source == SOURCE_FULL else 0.0
    mean_score = 0.5 * (src.max_score + dst.max_score)
    return [
        same_class,
        dist_score,
        scale_sim,
        overlap,
        contain_sd,
        contain_ds,
        same_cluster,
        cross_source,
        both_slice,
        both_full,
        mean_score,
        math.tanh(dx_norm),
        math.tanh(dy_norm),
        max(-3.0, min(3.0, log_area_ratio)) / 3.0,
        max(-3.0, min(3.0, log_w_ratio)) / 3.0,
        max(-3.0, min(3.0, log_h_ratio)) / 3.0,
    ]


def should_connect_size_nodes(src, dst, args):
    if src.node_id == dst.node_id:
        return False
    if src.category_id != dst.category_id:
        return False
    src_center = np.array(src.center, dtype=np.float32)
    dst_center = np.array(dst.center, dtype=np.float32)
    dist = float(np.linalg.norm(src_center - dst_center))
    if dist <= args.graph_radius:
        return True
    if iou_xywh(src.bbox, dst.bbox) > 0.0:
        return True
    if src.source != dst.source and (
        containment(src.bbox, dst.bbox) > 0.1
        or containment(dst.bbox, src.bbox) > 0.1
    ):
        return True
    if src.dbscan_label >= 0 and src.dbscan_label == dst.dbscan_label:
        return True
    return False


def build_size_aware_edge_pairs(nodes, args):
    edge_pairs = set()
    for i, src in enumerate(nodes):
        scored = []
        for j, dst in enumerate(nodes):
            if not should_connect_size_nodes(src, dst, args):
                continue
            src_center = np.array(src.center, dtype=np.float32)
            dst_center = np.array(dst.center, dtype=np.float32)
            dist = float(np.linalg.norm(src_center - dst_center))
            score = max(0.0, 1.0 - dist / max(1.0, float(args.graph_radius))) + iou_xywh(src.bbox, dst.bbox)
            if src.dbscan_label >= 0 and src.dbscan_label == dst.dbscan_label:
                score += 0.5
            if src.source != dst.source:
                score += 0.25
            scored.append((j, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        for j, _ in scored[: args.graph_knn]:
            edge_pairs.add((i, j))
    for i in range(len(nodes)):
        edge_pairs.add((i, i))
    return sorted(edge_pairs)


def build_size_aware_tensors_vectorized(nodes, record, args):
    n = len(nodes)
    boxes = np.asarray([node.bbox for node in nodes], dtype=np.float32)
    centers = np.asarray([node.center for node in nodes], dtype=np.float32)
    categories = np.asarray([node.category_id for node in nodes], dtype=np.int32)
    sources = np.asarray([node.source for node in nodes], dtype=np.int32)
    labels = np.asarray([node.dbscan_label for node in nodes], dtype=np.int32)
    scores = np.asarray([node.max_score for node in nodes], dtype=np.float32)
    union_areas = np.asarray([max(1.0, float(node.union_area)) for node in nodes], dtype=np.float32)
    widths = np.maximum(1.0, boxes[:, 2])
    heights = np.maximum(1.0, boxes[:, 3])
    image_area = max(1.0, float(record.width * record.height))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    inter_w = np.maximum(0.0, np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]))
    inter_h = np.maximum(0.0, np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]))
    inter = inter_w * inter_h
    bbox_areas = np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])
    union = bbox_areas[:, None] + bbox_areas[None, :] - inter
    iou = np.divide(inter, np.maximum(union, 1e-6), out=np.zeros_like(inter), where=union > 0.0)
    contain_src_dst = np.divide(
        inter,
        np.maximum(bbox_areas[:, None], 1e-6),
        out=np.zeros_like(inter),
        where=bbox_areas[:, None] > 0.0,
    )
    contain_dst_src = np.divide(
        inter,
        np.maximum(bbox_areas[None, :], 1e-6),
        out=np.zeros_like(inter),
        where=bbox_areas[None, :] > 0.0,
    )

    delta = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    same_class = categories[:, None] == categories[None, :]
    not_self = ~np.eye(n, dtype=bool)
    cross_source = sources[:, None] != sources[None, :]
    same_cluster = (labels[:, None] >= 0) & (labels[:, None] == labels[None, :])
    connect = (
        not_self
        & same_class
        & (
            (dist <= float(args.graph_radius))
            | (iou > 0.0)
            | (cross_source & ((contain_src_dst > 0.1) | (contain_dst_src > 0.1)))
            | same_cluster
        )
    )

    dist_score = np.maximum(0.0, 1.0 - dist / max(1.0, float(args.graph_radius)))
    edge_score = dist_score + iou + 0.5 * same_cluster.astype(np.float32) + 0.25 * cross_source.astype(np.float32)
    edge_pairs = set()
    top_k = max(0, int(args.graph_knn))
    for src_index in range(n):
        candidate_indices = np.flatnonzero(connect[src_index])
        if candidate_indices.size == 0 or top_k <= 0:
            continue
        candidate_scores = edge_score[src_index, candidate_indices]
        if candidate_indices.size > top_k:
            local = np.argpartition(candidate_scores, -top_k)[-top_k:]
            candidate_indices = candidate_indices[local]
            candidate_scores = candidate_scores[local]
        order = np.argsort(candidate_scores)[::-1]
        for dst_index in candidate_indices[order]:
            edge_pairs.add((src_index, int(dst_index)))
    for index in range(n):
        edge_pairs.add((index, index))
    edge_pairs = sorted(edge_pairs)

    src_idx = np.asarray([src for src, _ in edge_pairs], dtype=np.int64)
    dst_idx = np.asarray([dst for _, dst in edge_pairs], dtype=np.int64)
    density = np.zeros(n, dtype=np.int32)
    non_self = src_idx != dst_idx
    np.add.at(density, src_idx[non_self], 1)
    np.add.at(density, dst_idx[non_self], 1)

    one_hot = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    for row, category_id in enumerate(categories.tolist()):
        one_hot[row, VISDRONE_TO_CLASS_INDEX.get(int(category_id), 0)] = 1.0
    node_features = np.column_stack(
        [
            scores,
            centers[:, 0] / max(1.0, float(record.width)),
            centers[:, 1] / max(1.0, float(record.height)),
            boxes[:, 2] / max(1.0, float(record.width)),
            boxes[:, 3] / max(1.0, float(record.height)),
            np.sqrt(union_areas / image_area),
            np.log((widths + 1.0) / (heights + 1.0)),
            (union_areas < SMALL_MEDIUM_AREA_THR).astype(np.float32),
            (union_areas >= SMALL_MEDIUM_AREA_THR).astype(np.float32),
            (sources == SOURCE_FULL).astype(np.float32),
            (sources == SOURCE_SLICE).astype(np.float32),
            (labels >= 0).astype(np.float32),
            np.minimum(1.0, density.astype(np.float32) / 16.0),
            np.log1p(union_areas) / math.log1p(image_area),
            boxes[:, 0] / max(1.0, float(record.width)),
            boxes[:, 1] / max(1.0, float(record.height)),
            one_hot,
        ]
    ).astype(np.float32)

    src_w = widths[src_idx]
    src_h = heights[src_idx]
    dst_w = widths[dst_idx]
    dst_h = heights[dst_idx]
    log_area_ratio = np.log((union_areas[src_idx] + 1.0) / (union_areas[dst_idx] + 1.0))
    log_w_ratio = np.log(src_w / dst_w)
    log_h_ratio = np.log(src_h / dst_h)
    edge_features = np.column_stack(
        [
            (categories[src_idx] == categories[dst_idx]).astype(np.float32),
            dist_score[src_idx, dst_idx],
            np.maximum(0.0, 1.0 - np.abs(log_area_ratio) / 3.0),
            iou[src_idx, dst_idx],
            contain_src_dst[src_idx, dst_idx],
            contain_dst_src[src_idx, dst_idx],
            ((labels[src_idx] >= 0) & (labels[src_idx] == labels[dst_idx])).astype(np.float32),
            (sources[src_idx] != sources[dst_idx]).astype(np.float32),
            ((sources[src_idx] == SOURCE_SLICE) & (sources[dst_idx] == SOURCE_SLICE)).astype(np.float32),
            ((sources[src_idx] == SOURCE_FULL) & (sources[dst_idx] == SOURCE_FULL)).astype(np.float32),
            0.5 * (scores[src_idx] + scores[dst_idx]),
            np.tanh((centers[dst_idx, 0] - centers[src_idx, 0]) / np.sqrt(src_w * dst_w)),
            np.tanh((centers[dst_idx, 1] - centers[src_idx, 1]) / np.sqrt(src_h * dst_h)),
            np.clip(log_area_ratio, -3.0, 3.0) / 3.0,
            np.clip(log_w_ratio, -3.0, 3.0) / 3.0,
            np.clip(log_h_ratio, -3.0, 3.0) / 3.0,
        ]
    ).astype(np.float32)

    targets = np.asarray(
        [[node.target_obj, node.target_small, node.target_large, node.target_roi] for node in nodes],
        dtype=np.float32,
    )
    weights = np.asarray([node.weight for node in nodes], dtype=np.float32)
    return (
        torch.from_numpy(node_features),
        torch.from_numpy(np.asarray(edge_pairs, dtype=np.int64)).t().contiguous(),
        torch.from_numpy(edge_features),
        torch.from_numpy(targets),
        torch.from_numpy(weights),
    )


def build_size_aware_tensors(nodes, record, args):
    if not nodes:
        empty_x = torch.empty((0, SIZE_AWARE_NODE_DIM), dtype=torch.float32)
        empty_ei = torch.empty((2, 0), dtype=torch.long)
        empty_ea = torch.empty((0, SIZE_AWARE_EDGE_DIM), dtype=torch.float32)
        empty_y = torch.empty((0, SIZE_AWARE_OUTPUT_DIM), dtype=torch.float32)
        empty_w = torch.empty((0,), dtype=torch.float32)
        return empty_x, empty_ei, empty_ea, empty_y, empty_w
    return build_size_aware_tensors_vectorized(nodes, record, args)

    edge_pairs = build_size_aware_edge_pairs(nodes, args)
    density = defaultdict(int)
    for src, dst in edge_pairs:
        if src != dst:
            density[src] += 1
            density[dst] += 1

    x = torch.tensor([size_aware_node_feature(node, record, density[index]) for index, node in enumerate(nodes)], dtype=torch.float32)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([size_aware_edge_feature(nodes[src], nodes[dst], record, args) for src, dst in edge_pairs], dtype=torch.float32)
    targets = torch.tensor(
        [[node.target_obj, node.target_small, node.target_large, node.target_roi] for node in nodes],
        dtype=torch.float32,
    )
    weights = torch.tensor([node.weight for node in nodes], dtype=torch.float32)
    return x, edge_index, edge_attr, targets, weights


def _stage1_score_features(nodes):
    return torch.tensor(
        [
            [
                float(getattr(node, "stage1_obj", node.gnn_obj)),
                float(getattr(node, "stage1_small", node.gnn_small)),
                float(getattr(node, "stage1_large", node.gnn_large)),
                float(getattr(node, "stage1_roi", node.gnn_roi)),
            ]
            for node in nodes
        ],
        dtype=torch.float32,
    )


def _stage2_detection_flags(count):
    return torch.tensor([[1.0, 0.0, 0.0, 0.0] for _ in range(count)], dtype=torch.float32)


def _stage2_detection_component_type(count):
    return torch.zeros((count, STAGE2_COMPONENT_TYPE_DIM), dtype=torch.float32)


def _stage2_empty_tensors():
    return (
        torch.empty((0, STAGE2_NODE_DIM), dtype=torch.float32),
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, SIZE_AWARE_EDGE_DIM), dtype=torch.float32),
        torch.empty((0, STAGE2_ACTION_TARGET_DIM), dtype=torch.float32),
        torch.empty((0,), dtype=torch.float32),
    )


def _stage2_size_empty_tensors():
    return (
        torch.empty((0, STAGE2_NODE_DIM), dtype=torch.float32),
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, SIZE_AWARE_EDGE_DIM), dtype=torch.float32),
        torch.empty((0, STAGE2_SIZE_TARGET_DIM), dtype=torch.float32),
        torch.empty((0,), dtype=torch.float32),
    )


def _stage2_hyperedge_attr(kind, score=1.0):
    attr = torch.zeros((SIZE_AWARE_EDGE_DIM,), dtype=torch.float32)
    attr[0] = 1.0
    attr[1] = 1.0
    attr[2] = 1.0
    attr[10] = float(max(0.0, min(1.0, score)))
    if kind == "duplicate":
        attr[3] = 1.0
        attr[6] = 1.0
    elif kind in {"fragment", "fragment_full", "fragment_slice"}:
        attr[4] = 0.5
        attr[5] = 0.5
        attr[7] = 1.0
        if kind == "fragment_full":
            attr[9] = 1.0
            attr[13] = 0.5
        elif kind == "fragment_slice":
            attr[8] = 1.0
            attr[13] = -0.5
    elif kind == "roi":
        attr[8] = 0.5
        attr[13] = -0.5
    elif kind == "tile_overlap":
        attr[6] = 0.5
        attr[8] = 1.0
    elif kind == "semantic":
        attr[0] = 0.0
        attr[1] = 0.5
    elif kind == "cluster":
        attr[1] = 0.75
        attr[2] = 0.75
        attr[8] = 0.5
    return attr


def _stage2_self_edge_attr():
    attr = torch.zeros((SIZE_AWARE_EDGE_DIM,), dtype=torch.float32)
    attr[1] = 1.0
    attr[2] = 1.0
    return attr


def stage2_action_values(node):
    if node.stage2_target_keep > 0.5:
        action_target = STAGE2_KEEP
    elif node.stage2_target_fragment > 0.5:
        action_target = STAGE2_FRAGMENT
    elif node.stage2_target_duplicate > 0.5:
        action_target = STAGE2_DUPLICATE
    else:
        action_target = STAGE2_BACKGROUND
    return [
        float(action_target),
        float(node.stage2_target_roi_refine),
        float(node.stage2_target_large_preserve),
    ]


def _stage2_action_targets(nodes):
    return torch.tensor([stage2_action_values(node) for node in nodes], dtype=torch.float32)


def _stage2_action_weights(nodes):
    return torch.tensor([float(node.stage2_action_weight) for node in nodes], dtype=torch.float32)


def _stage2_size_targets(nodes):
    return torch.tensor([[float(node.target_small), float(node.target_large)] for node in nodes], dtype=torch.float32)


def _stage2_size_weights(nodes):
    values = []
    for node in nodes:
        weight = float(node.weight)
        if float(node.target_small) > 0.5:
            weight = max(weight, 3.0)
        elif float(node.target_large) > 0.5:
            weight = max(weight, 2.5)
        values.append(weight)
    return torch.tensor(values, dtype=torch.float32)


def _component_type_vector(kind):
    kinds = ["duplicate", "fragment", "roi", "tile_overlap", "semantic", "cluster"]
    vec = torch.zeros((STAGE2_COMPONENT_TYPE_DIM,), dtype=torch.float32)
    if kind in kinds:
        vec[kinds.index(kind)] = 1.0
    return vec


def _stage2_component_target(kind, member_targets):
    target = torch.zeros((STAGE2_ACTION_TARGET_DIM,), dtype=torch.float32)
    actions = member_targets[:, 0].long() if member_targets.numel() else torch.empty((0,), dtype=torch.long)
    if kind == "fragment":
        action_target = STAGE2_FRAGMENT
    elif kind in {"duplicate", "tile_overlap"}:
        action_target = STAGE2_DUPLICATE
    elif kind == "roi":
        action_target = STAGE2_KEEP if (actions == STAGE2_KEEP).any() else STAGE2_BACKGROUND
    elif (actions == STAGE2_KEEP).any():
        action_target = STAGE2_KEEP
    else:
        action_target = STAGE2_BACKGROUND
    target[0] = float(action_target)
    if member_targets.numel():
        target[1] = float(member_targets[:, 1].max().item())
        target[2] = float(member_targets[:, 2].max().item())
    if kind == "roi":
        target[1] = 1.0
    return target


def _stage2_size_component_target(member_targets):
    if member_targets.numel() == 0:
        return torch.zeros((STAGE2_SIZE_TARGET_DIM,), dtype=torch.float32)
    return member_targets.max(dim=0).values


def _stage2_component_feature(x, members, nodes, kind):
    member_x = x[members]
    if kind == "fragment":
        full_members = [index for index in members if nodes[index].source == SOURCE_FULL]
        slice_members = [index for index in members if nodes[index].source == SOURCE_SLICE]
        parts = []
        if full_members:
            parts.append(0.60 * x[full_members].max(dim=0).values)
        if slice_members:
            parts.append(0.25 * x[slice_members].mean(dim=0))
            parts.append(0.15 * x[slice_members].max(dim=0).values)
        return sum(parts) if parts else member_x.mean(dim=0)
    if kind in {"duplicate", "tile_overlap"}:
        best_member = max(members, key=lambda index: float(nodes[index].max_score))
        return 0.70 * x[best_member] + 0.30 * member_x.mean(dim=0)
    if kind == "roi":
        return 0.50 * member_x.mean(dim=0) + 0.50 * member_x.max(dim=0).values
    if kind == "cluster":
        best_member = max(members, key=lambda index: float(nodes[index].max_score))
        return 0.45 * member_x.mean(dim=0) + 0.35 * member_x.max(dim=0).values + 0.20 * x[best_member]
    if kind == "semantic":
        best_member = max(members, key=lambda index: float(nodes[index].max_score))
        return 0.50 * member_x.mean(dim=0) + 0.50 * x[best_member]
    return member_x.mean(dim=0)


def _node_pair_norm_dist(src, dst):
    sx, sy = src.center
    dx, dy = dst.center
    scale_x = math.sqrt(max(1.0, float(src.bbox[2])) * max(1.0, float(dst.bbox[2])))
    scale_y = math.sqrt(max(1.0, float(src.bbox[3])) * max(1.0, float(dst.bbox[3])))
    return math.sqrt(((dx - sx) / scale_x) ** 2 + ((dy - sy) / scale_y) ** 2)


def _connected_components(count, edges):
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in edges:
        union(left, right)
    grouped = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)
    return [members for members in grouped.values() if len(members) >= 2]


def _cap_component_members(members, nodes, max_members):
    max_members = int(max_members or 0)
    if max_members <= 0 or len(members) <= max_members:
        return members
    return sorted(members, key=lambda index: float(nodes[index].max_score), reverse=True)[:max_members]


def _stage2_conflict_components(nodes, record, args):
    components = []
    max_members = int(getattr(args, "hgnn_max_component_members", 32))
    duplicate_iou_thr = float(getattr(args, "hgnn_duplicate_iou_thr", 0.50))
    duplicate_containment_thr = float(getattr(args, "hgnn_duplicate_containment_thr", 0.60))
    duplicate_norm_dist_thr = float(getattr(args, "hgnn_duplicate_norm_dist_thr", 1.50))
    fragment_containment_thr = float(getattr(args, "stage2_fragment_containment_thr", 0.30))
    fragment_iou_thr = float(getattr(args, "stage2_fragment_iou_thr", 0.10))
    roi_thr = float(getattr(args, "gnn_roi_thr", 0.45))

    by_class = defaultdict(list)
    for index, node in enumerate(nodes):
        by_class[int(node.category_id)].append(index)

    for _, members in by_class.items():
        edges = []
        for left_pos, left in enumerate(members):
            for right in members[left_pos + 1:]:
                left_node = nodes[left]
                right_node = nodes[right]
                overlap = iou_xywh(left_node.bbox, right_node.bbox)
                contain_lr = containment(left_node.bbox, right_node.bbox)
                contain_rl = containment(right_node.bbox, left_node.bbox)
                norm_dist = _node_pair_norm_dist(left_node, right_node)
                if (
                    overlap >= duplicate_iou_thr
                    or max(contain_lr, contain_rl) >= duplicate_containment_thr
                    or norm_dist <= duplicate_norm_dist_thr
                ):
                    edges.append((left, right))
        for component in _connected_components(len(nodes), edges):
            capped = _cap_component_members(component, nodes, max_members)
            if len(capped) >= 2:
                score = max(float(nodes[index].best_match_iou) for index in capped)
                components.append({"type": "duplicate", "members": capped, "score": score})

    by_cluster = defaultdict(list)
    for index, node in enumerate(nodes):
        if int(node.dbscan_label) >= 0:
            by_cluster[(int(node.category_id), int(node.dbscan_label))].append(index)
    for _, members in by_cluster.items():
        capped = _cap_component_members(members, nodes, max_members)
        if len(capped) >= 2:
            score = max(
                max(float(nodes[index].max_score), float(getattr(nodes[index], "stage1_obj", nodes[index].gnn_obj)))
                for index in capped
            )
            components.append({"type": "cluster", "members": capped, "score": score})

    full_large = [
        (index, node)
        for index, node in enumerate(nodes)
        if node.source == SOURCE_FULL and node.union_area >= SMALL_MEDIUM_AREA_THR
    ]
    for full_index, full_node in full_large:
        members = [full_index]
        for index, node in enumerate(nodes):
            if node.source != SOURCE_SLICE or node.category_id != full_node.category_id:
                continue
            if (
                containment(node.bbox, full_node.bbox) >= fragment_containment_thr
                or iou_xywh(node.bbox, full_node.bbox) >= fragment_iou_thr
            ):
                members.append(index)
        members = _cap_component_members(sorted(set(members)), nodes, max_members)
        if len(members) >= 2:
            components.append({"type": "fragment", "members": members, "score": float(full_node.max_score)})

    roi_candidates = [
        index
        for index, node in enumerate(nodes)
        if node.source == SOURCE_SLICE
        and node.union_area <= SMALL_MEDIUM_AREA_THR * 2.5
        and float(getattr(node, "stage1_roi", node.gnn_roi)) >= roi_thr
    ]
    roi_edges = []
    for left_pos, left in enumerate(roi_candidates):
        for right in roi_candidates[left_pos + 1:]:
            if nodes[left].category_id != nodes[right].category_id:
                continue
            if _node_pair_norm_dist(nodes[left], nodes[right]) <= 2.0:
                roi_edges.append((left, right))
    for component in _connected_components(len(nodes), roi_edges):
        capped = _cap_component_members(component, nodes, max_members)
        if len(capped) >= 2:
            score = max(float(getattr(nodes[index], "stage1_roi", nodes[index].gnn_roi)) for index in capped)
            components.append({"type": "roi", "members": capped, "score": score})

    semantic_pairs = {(1, 3), (3, 1), (1, 10), (10, 1), (4, 6), (6, 4), (4, 9), (9, 4)}
    semantic_edges = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if (nodes[left].category_id, nodes[right].category_id) not in semantic_pairs:
                continue
            if _node_pair_norm_dist(nodes[left], nodes[right]) <= 2.5:
                semantic_edges.append((left, right))
    for component in _connected_components(len(nodes), semantic_edges):
        capped = _cap_component_members(component, nodes, max_members)
        if len(capped) >= 2:
            components.append({"type": "semantic", "members": capped, "score": 0.5})

    return components


def build_stage2_tensors(nodes, record, args, hypergraph=False):
    if not nodes:
        return _stage2_empty_tensors()

    base_x, edge_index, edge_attr, _, _ = build_size_aware_tensors(nodes, record, args)
    if base_x.numel() == 0:
        return _stage2_empty_tensors()

    stage1_features = _stage1_score_features(nodes)
    targets = _stage2_action_targets(nodes)
    weights = _stage2_action_weights(nodes)
    x = torch.cat(
        [
            base_x,
            stage1_features,
            _stage2_detection_flags(len(nodes)),
            _stage2_detection_component_type(len(nodes)),
        ],
        dim=-1,
    )
    if not hypergraph:
        return x, edge_index, edge_attr, targets, weights

    hyper_features = []
    hyper_edges = []
    hyper_edge_attrs = []
    component_targets = []
    component_weights = []
    components = _stage2_conflict_components(nodes, record, args)
    component_loss_weight = float(getattr(args, "hgnn_component_loss_weight", 0.35))
    for component in components:
        kind = component["type"]
        members = component["members"]
        hyper_index = x.shape[0] + len(hyper_features)
        feature = _stage2_component_feature(x, members, nodes, kind)
        flag_start = SIZE_AWARE_NODE_DIM + SIZE_AWARE_OUTPUT_DIM
        type_start = flag_start + 4
        feature[flag_start:type_start] = torch.tensor(
            [
                0.0,
                1.0,
                1.0 if kind in {"duplicate", "fragment", "tile_overlap"} else 0.0,
                1.0 if kind in {"roi", "semantic", "cluster"} else 0.0,
            ],
            dtype=feature.dtype,
        )
        feature[type_start : type_start + STAGE2_COMPONENT_TYPE_DIM] = _component_type_vector(kind)
        hyper_features.append(feature)
        score = float(component.get("score", stage1_features[members].mean().item()))
        attr = _stage2_hyperedge_attr(kind, score=score)
        for member in members:
            hyper_edges.append((int(member), hyper_index))
            member_kind = kind
            if kind == "fragment":
                member_kind = "fragment_full" if nodes[member].source == SOURCE_FULL else "fragment_slice"
            hyper_edge_attrs.append(_stage2_hyperedge_attr(member_kind, score=score))
            hyper_edges.append((hyper_index, int(member)))
            hyper_edge_attrs.append(_stage2_hyperedge_attr(member_kind, score=score))
        hyper_edges.append((hyper_index, hyper_index))
        hyper_edge_attrs.append(_stage2_self_edge_attr())
        member_targets = targets[members]
        component_targets.append(_stage2_component_target(kind, member_targets))
        component_weights.append(component_loss_weight)

    if not hyper_features:
        return x, edge_index, edge_attr, targets, weights

    x = torch.cat([x, torch.stack(hyper_features, dim=0)], dim=0)
    extra_edge_index = torch.tensor(hyper_edges, dtype=torch.long).t().contiguous()
    extra_edge_attr = torch.stack(hyper_edge_attrs, dim=0)
    edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, extra_edge_attr], dim=0)
    targets = torch.cat(
        [targets, torch.stack(component_targets, dim=0).to(dtype=targets.dtype)],
        dim=0,
    )
    weights = torch.cat([weights, torch.tensor(component_weights, dtype=weights.dtype)], dim=0)
    return x, edge_index, edge_attr, targets, weights


def build_stage2_size_tensors(nodes, record, args, hypergraph=False):
    if not nodes:
        return _stage2_size_empty_tensors()

    base_x, edge_index, edge_attr, _, _ = build_size_aware_tensors(nodes, record, args)
    if base_x.numel() == 0:
        return _stage2_size_empty_tensors()

    stage1_features = _stage1_score_features(nodes)
    targets = _stage2_size_targets(nodes)
    weights = _stage2_size_weights(nodes)
    x = torch.cat(
        [
            base_x,
            stage1_features,
            _stage2_detection_flags(len(nodes)),
            _stage2_detection_component_type(len(nodes)),
        ],
        dim=-1,
    )
    if not hypergraph:
        return x, edge_index, edge_attr, targets, weights

    hyper_features = []
    hyper_edges = []
    hyper_edge_attrs = []
    component_targets = []
    component_weights = []
    components = _stage2_conflict_components(nodes, record, args)
    component_loss_weight = float(getattr(args, "hgnn_component_loss_weight", 0.35))
    for component in components:
        kind = component["type"]
        members = component["members"]
        hyper_index = x.shape[0] + len(hyper_features)
        feature = _stage2_component_feature(x, members, nodes, kind)
        flag_start = SIZE_AWARE_NODE_DIM + SIZE_AWARE_OUTPUT_DIM
        type_start = flag_start + 4
        feature[flag_start:type_start] = torch.tensor(
            [
                0.0,
                1.0,
                1.0 if kind in {"duplicate", "fragment", "tile_overlap"} else 0.0,
                1.0 if kind in {"roi", "semantic", "cluster"} else 0.0,
            ],
            dtype=feature.dtype,
        )
        feature[type_start : type_start + STAGE2_COMPONENT_TYPE_DIM] = _component_type_vector(kind)
        hyper_features.append(feature)
        score = float(component.get("score", stage1_features[members].mean().item()))
        for member in members:
            member_kind = kind
            if kind == "fragment":
                member_kind = "fragment_full" if nodes[member].source == SOURCE_FULL else "fragment_slice"
            hyper_edges.append((int(member), hyper_index))
            hyper_edge_attrs.append(_stage2_hyperedge_attr(member_kind, score=score))
            hyper_edges.append((hyper_index, int(member)))
            hyper_edge_attrs.append(_stage2_hyperedge_attr(member_kind, score=score))
        hyper_edges.append((hyper_index, hyper_index))
        hyper_edge_attrs.append(_stage2_self_edge_attr())
        component_targets.append(_stage2_size_component_target(targets[members]))
        component_weights.append(component_loss_weight)

    if not hyper_features:
        return x, edge_index, edge_attr, targets, weights

    x = torch.cat([x, torch.stack(hyper_features, dim=0)], dim=0)
    extra_edge_index = torch.tensor(hyper_edges, dtype=torch.long).t().contiguous()
    extra_edge_attr = torch.stack(hyper_edge_attrs, dim=0)
    edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, extra_edge_attr], dim=0)
    targets = torch.cat(
        [targets, torch.stack(component_targets, dim=0).to(dtype=targets.dtype)],
        dim=0,
    )
    weights = torch.cat([weights, torch.tensor(component_weights, dtype=weights.dtype)], dim=0)
    return x, edge_index, edge_attr, targets, weights




class EdgeGatedLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return self.norm(F.relu(self.self_linear(h)) + h)

        src, dst = edge_index
        pair = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        messages = self.message(pair) * self.gate(pair)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
        agg = agg / deg.clamp_min(1.0)
        return self.norm(F.relu(self.self_linear(h) + agg) + h)



class SizeAwareGraphGNN(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers, output_dim=SIZE_AWARE_OUTPUT_DIM):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList([EdgeGatedLayer(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.input(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return self.head(torch.cat([h, x], dim=-1))



def weighted_size_aware_bce_loss(logits, targets, weights, pos_weight=None):
    if targets.numel() == 0:
        return logits.sum() * 0.0
    if targets.shape[1] == STAGE2_ACTION_OUTPUT_DIM:
        # keep, duplicate, fragment, roi_refine, large_preserve
        head_weight_values = [2.0, 1.25, 1.5, 2.0, 1.5]
    else:
        head_weight_values = [1.0, 2.0, 1.5, 2.5]
    head_weights = torch.tensor(head_weight_values, dtype=logits.dtype, device=logits.device)
    fixed_pos_weight = None
    if pos_weight is not None:
        fixed_pos_weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    losses = []
    for head_index in range(targets.shape[1]):
        target = targets[:, head_index]
        if fixed_pos_weight is None:
            positive = (target > 0.5).float()
            pos_count = positive.sum().clamp_min(1.0)
            neg_count = (1.0 - positive).sum().clamp_min(1.0)
            head_pos_weight = (neg_count / pos_count).clamp(1.0, 25.0)
        else:
            head_pos_weight = fixed_pos_weight[head_index].clamp(1.0, 25.0)
        raw = F.binary_cross_entropy_with_logits(
            logits[:, head_index],
            target,
            reduction="none",
            pos_weight=head_pos_weight,
        )
        losses.append(head_weights[head_index] * (raw * weights).sum() / weights.sum().clamp_min(1.0))
    return sum(losses)


def sigmoid_focal_loss_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    raw = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    return raw * torch.pow((1.0 - pt).clamp_min(1e-6), gamma)


def stage2_action_refiner_loss(logits, targets, weights, loss_weight=None):
    if targets.numel() == 0:
        return logits.sum() * 0.0
    action_targets = targets[:, 0].long().clamp(0, STAGE2_ACTION_DIM - 1)
    aux_targets = targets[:, 1:3].to(dtype=logits.dtype)
    class_weight = None
    aux_pos_weight = None
    if loss_weight is not None:
        loss_weight = torch.as_tensor(loss_weight, dtype=logits.dtype, device=logits.device)
        class_weight = loss_weight[:STAGE2_ACTION_DIM].clamp(0.1, 25.0)
        aux_pos_weight = loss_weight[STAGE2_ACTION_DIM : STAGE2_ACTION_DIM + STAGE2_AUX_DIM].clamp(1.0, 25.0)
    action_raw = F.cross_entropy(
        logits[:, :STAGE2_ACTION_DIM],
        action_targets,
        weight=class_weight,
        reduction="none",
    )
    denom = weights.sum().clamp_min(1.0)
    action_loss = (action_raw * weights).sum() / denom
    aux_raw = sigmoid_focal_loss_with_logits(
        logits[:, STAGE2_ACTION_DIM : STAGE2_ACTION_DIM + STAGE2_AUX_DIM],
        aux_targets,
        gamma=2.0,
        pos_weight=aux_pos_weight,
    )
    aux_head_weights = torch.tensor([0.5, 0.5], dtype=logits.dtype, device=logits.device)
    aux_loss = ((aux_raw * aux_head_weights) * weights[:, None]).sum() / denom
    return action_loss + aux_loss


def stage2_size_refiner_loss(logits, targets, weights, pos_weight=None):
    if targets.numel() == 0:
        return logits.sum() * 0.0
    fixed_pos_weight = None
    if pos_weight is not None:
        fixed_pos_weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device).clamp(1.0, 25.0)
    head_weights = torch.tensor([2.0, 1.5], dtype=logits.dtype, device=logits.device)
    losses = []
    for head_index in range(STAGE2_SIZE_TARGET_DIM):
        target = targets[:, head_index].to(dtype=logits.dtype)
        if fixed_pos_weight is None:
            positive = (target > 0.5).float()
            pos_count = positive.sum().clamp_min(1.0)
            neg_count = (1.0 - positive).sum().clamp_min(1.0)
            head_pos_weight = (neg_count / pos_count).clamp(1.0, 25.0)
        else:
            head_pos_weight = fixed_pos_weight[head_index]
        raw = F.binary_cross_entropy_with_logits(
            logits[:, head_index],
            target,
            reduction="none",
            pos_weight=head_pos_weight,
        )
        losses.append(head_weights[head_index] * (raw * weights).sum() / weights.sum().clamp_min(1.0))
    return sum(losses)




def make_size_aware_detection_samples(records, full_cache, coarse_cache, gt_by_image, args):
    samples = []
    for record in tqdm(records, desc="Build size-aware detection graph samples"):
        full_predictions = full_cache.get(str(record.image_id), [])
        slice_predictions = coarse_cache.get(str(record.image_id), [])
        gt_boxes = gt_by_image.get(record.image_id, [])
        nodes = build_size_aware_detection_nodes(record, full_predictions, slice_predictions, gt_boxes, args)
        samples.append(
            {
                "record": record,
                "full_predictions": full_predictions,
                "predictions": slice_predictions,
                "nodes": nodes,
            }
        )
    return samples




def merge_rois(scored_rois, iou_threshold=0.65):
    merged = []
    for roi, score in sorted(scored_rois, key=lambda item: item[1], reverse=True):
        consumed = False
        for index, (existing_roi, existing_score) in enumerate(merged):
            if iou_xywh(roi, existing_roi) >= iou_threshold:
                merged[index] = (union_xywh([roi, existing_roi]), max(score, existing_score))
                consumed = True
                break
        if not consumed:
            merged.append((roi, score))
    return merged




def attach_size_aware_scores(samples, model, args, device):
    model.eval()
    for sample in tqdm(samples, desc="score size-aware detection graphs"):
        nodes = sample["nodes"]
        if not nodes:
            continue
        x, edge_index, edge_attr, _, _ = build_size_aware_tensors(nodes, sample["record"], args)
        probs = torch.sigmoid(
            model(x.to(device), edge_index.to(device), edge_attr.to(device))
        ).detach().cpu().tolist()
        for node, prob in zip(nodes, probs):
            obj_prob, small_prob, large_prob, roi_prob = [float(value) for value in prob]
            node.gnn_obj = obj_prob
            node.gnn_small = small_prob
            node.gnn_large = large_prob
            node.gnn_roi = roi_prob
            size_prob = large_prob if node.union_area >= SMALL_MEDIUM_AREA_THR else small_prob
            node.stage2_roi_prob = roi_prob
            node.stage2_prob = float(max(obj_prob, size_prob, roi_prob))


def size_aware_detection_node_scores(nodes, prefix="gnn"):
    scores = {}
    for node in nodes:
        if node.source != SOURCE_SLICE:
            continue
        if bool(getattr(node, "stage2_action_scored", False)):
            keep = float(getattr(node, "stage2_keep", 0.0))
            background = float(getattr(node, "stage2_background", 0.0))
            score = keep * (1.0 - background)
            scores[node.det_index] = max(scores.get(node.det_index, 0.0), score)
            continue
        obj_prob = float(getattr(node, f"{prefix}_obj", 0.0))
        small_prob = float(getattr(node, f"{prefix}_small", 0.0))
        large_prob = float(getattr(node, f"{prefix}_large", 0.0))
        size_prob = large_prob if node.union_area >= SMALL_MEDIUM_AREA_THR else small_prob
        score = max(obj_prob, size_prob)
        scores[node.det_index] = max(scores.get(node.det_index, 0.0), score)
    return scores


def size_aware_full_node_scores(nodes, prefix="gnn"):
    scores = {}
    for node in nodes:
        if node.source != SOURCE_FULL:
            continue
        if bool(getattr(node, "stage2_action_scored", False)):
            score = max(float(getattr(node, "stage2_large_preserve", 0.0)), float(getattr(node, "stage2_keep", 0.0)))
            score *= 1.0 - float(getattr(node, "stage2_background", 0.0))
            scores[node.det_index] = max(scores.get(node.det_index, 0.0), score)
            continue
        obj_prob = float(getattr(node, f"{prefix}_obj", 0.0))
        large_prob = float(getattr(node, f"{prefix}_large", 0.0))
        score = max(obj_prob, large_prob)
        scores[node.det_index] = max(scores.get(node.det_index, 0.0), score)
    return scores


def build_size_aware_gnn_rois(nodes, record, args):
    scored_rois = []
    roi_threshold = float(getattr(args, "gnn_roi_thr", 0.45))
    for node in nodes:
        small_like = node.union_area <= SMALL_MEDIUM_AREA_THR * 2.5
        score = float(node.stage2_roi_refine) if bool(getattr(node, "stage2_action_scored", False)) else float(node.gnn_roi)
        if small_like and score >= roi_threshold:
            scored_rois.append((expand_xywh(node.bbox, record.width, record.height, args.roi_margin), score))
    return merge_rois(scored_rois)[: args.max_stage2_rois]


def add_annotation_ids(predictions):
    fixed = []
    for index, pred in enumerate(predictions, start=1):
        pred = dict(pred)
        pred["id"] = index
        pred["area"] = bbox_area(pred["bbox"])
        pred["iscrowd"] = 0
        fixed.append(pred)
    return fixed


def write_predictions(path, predictions):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(add_annotation_ids(predictions), f, indent=2)


def read_prediction_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    return data


def filter_predictions_to_images(predictions, allowed_image_ids):
    if allowed_image_ids is None:
        return predictions
    return [pred for pred in predictions if int(pred["image_id"]) in allowed_image_ids]



def run_full_baseline(model, records, args):
    all_predictions = []
    for record in tqdm(records, desc="00 full inference"):
        all_predictions.extend(
            predict_full_image(
                model,
                record,
                conf=args.full_conf,
                iou=args.model_iou,
                max_det=args.max_det,
                device=args.device,
            )
        )
    return all_predictions




def fuse_final_predictions(
    model,
    record,
    coarse_predictions,
    full_predictions,
    rois,
    args,
    node_scores=None,
    full_node_scores=None,
    fine_roi_predictions=None,
):
    final_predictions = []
    alpha = min(1.0, max(0.0, float(getattr(args, "gnn_score_alpha", 0.0))))
    use_large_gate = bool(getattr(args, "gnn_large_gate", True))
    cleanup = bool(getattr(args, "stage2_cleanup", False))
    cleanup_min_gnn = float(getattr(args, "stage2_cleanup_min_gnn", 0.25))
    cleanup_high_conf = float(getattr(args, "stage2_cleanup_high_conf", 0.65))
    cleanup_roi_thr = float(getattr(args, "stage2_cleanup_roi_thr", 0.55))
    cleanup_fine_conf = float(getattr(args, "stage2_cleanup_fine_conf", 0.20))

    for index, pred in enumerate(full_predictions):
        if bbox_area(pred["bbox"]) < SMALL_MEDIUM_AREA_THR or pred["score"] < args.large_keep_conf:
            continue
        if full_node_scores is None or not use_large_gate:
            final_predictions.append(pred)
            continue
        large_score = float(full_node_scores.get(index, 0.0))
        if large_score >= float(getattr(args, "gnn_large_thr", 0.45)):
            adjusted = dict(pred)
            if alpha > 0.0:
                fused_score = (1.0 - alpha) * float(pred["score"]) + alpha * large_score
                adjusted["score"] = float(fused_score if cleanup else max(pred["score"], fused_score))
            final_predictions.append(adjusted)

    for index, pred in enumerate(coarse_predictions):
        node_score = 0.0 if node_scores is None else node_scores.get(index, 0.0)
        detector_score = float(pred["score"])
        small_like = bbox_area(pred["bbox"]) <= SMALL_MEDIUM_AREA_THR * 2.5
        if cleanup:
            keep = (
                (small_like and node_score >= args.stage2_prob_threshold)
                or (detector_score >= cleanup_high_conf and node_score >= cleanup_min_gnn)
            )
        else:
            keep = detector_score >= args.stage1_keep_conf or (
                small_like and node_score >= args.stage2_prob_threshold
            )
        if not keep:
            continue
        adjusted = dict(pred)
        if node_score > 0.0 and alpha > 0.0:
            fused_score = (1.0 - alpha) * detector_score + alpha * float(node_score)
            adjusted["score"] = float(fused_score if cleanup else max(detector_score, fused_score))
        final_predictions.append(adjusted)

    if fine_roi_predictions is None:
        fine_roi_predictions = []
        for roi, _ in rois:
            fine_roi_predictions.extend(
                predict_roi_sliced(
                    model,
                    record,
                    roi,
                    conf=args.fine_conf,
                    iou=args.model_iou,
                    max_det=args.max_det,
                    device=args.device,
                    slice_size=args.fine_slice_size,
                    overlap=args.fine_overlap,
                )
            )
    for pred in fine_roi_predictions:
        if bbox_area(pred["bbox"]) > SMALL_MEDIUM_AREA_THR * 2.5:
            continue
        roi_score = float(pred.get("_roi_score", 0.0))
        detector_score = float(pred.get("score", 0.0))
        if cleanup and (roi_score < cleanup_roi_thr or detector_score < cleanup_fine_conf):
            continue
        adjusted = dict(pred)
        adjusted.pop("_roi_score", None)
        if cleanup and roi_score > 0.0 and alpha > 0.0:
            adjusted["score"] = float((1.0 - alpha) * detector_score + alpha * roi_score)
        final_predictions.append(adjusted)

    return classwise_nms(final_predictions, iou_threshold=args.final_nms_iou)



def run_size_aware_gnn_large_preserve_variant(model, eval_samples, full_by_image, args):
    all_predictions = []
    plans = []
    for sample in tqdm(eval_samples, desc="prepare GNN fusion plans"):
        record = sample["record"]
        nodes = sample["nodes"]
        plans.append(
            {
                "record": record,
                "coarse_predictions": sample["predictions"],
                "full_predictions": full_by_image.get(record.image_id, []),
                "rois": build_size_aware_gnn_rois(nodes, record, args),
                "node_scores": size_aware_detection_node_scores(nodes, prefix="gnn"),
                "full_node_scores": size_aware_full_node_scores(nodes, prefix="gnn"),
                "fine_roi_keys": [],
            }
        )
    prefetch_fine_roi_predictions(model, plans, args)
    for plan in tqdm(plans, desc="detection-node GNN fusion"):
        all_predictions.extend(
            fuse_final_predictions(
                model,
                plan["record"],
                plan["coarse_predictions"],
                plan["full_predictions"],
                plan["rois"],
                args,
                node_scores=plan["node_scores"],
                full_node_scores=plan["full_node_scores"],
                fine_roi_predictions=fine_roi_predictions_for_plan(plan, args),
            )
        )
    flush_fine_roi_cache(args, force=True)
    return all_predictions



def evaluate_predictions(gt_path, pred_path):
    coco_gt = COCO(gt_path)
    predictions = read_prediction_file(pred_path)
    if not predictions:
        return {key: 0.0 for key in metric_names()}
    tmp_path = str(Path(pred_path).with_suffix(".eval_list.json"))
    with open(tmp_path, "w") as f:
        json.dump(predictions, f)
    coco_dt = coco_gt.loadRes(tmp_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats.tolist()
    return dict(zip(metric_names(), stats))


def metric_names():
    return [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]


def write_evaluation_csv(output_path, rows):
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant"] + metric_names())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def group_predictions_by_image(predictions):
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[int(pred["image_id"])].append(pred)
    return grouped
