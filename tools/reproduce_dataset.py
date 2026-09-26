#!/usr/bin/env python3
"""Download/verify the shared Experiment 1/2/3 data; never run a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'final/scripts'))
from prepare_visdrone_10class_train import CLASS_TO_INDEX, IGNORED_LABELS, clipped_yolo_box

REPO = 'Voxel51/VisDrone2019-DET'
REVISION = '3c3b9e9bd44c91121c1fc19fce428f0e6b71bba2'
SAMPLES_SHA = '45777883b6b95fe504dfa59e963955190ad07c2714677bd3dd31b730036f4d82'
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_name(remote):
    p = PurePosixPath(remote)
    if len(p.parts) != 2 or p.parts[0] != 'data' or p.suffix.lower() != '.jpg' or '\\' in remote:
        raise ValueError(f'Unexpected image path: {remote}')
    return p.name


def fetch(remote, destination, expected_sha):
    """Reuse verified cache only; leave originals intact on failed downloads."""
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == expected_sha:
        return destination
    if destination.exists():
        raise ValueError(f'Checksum mismatch in cache: {destination}; move it aside before retrying')
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}'
    for attempt in range(3):
        fd, temp_name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.part', dir=destination.parent)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, 'wb') as out:
                request = urllib.request.Request(url, headers={'User-Agent': 'visdrone-reproduction/1'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    shutil.copyfileobj(response, out)
            if sha256(temp) != expected_sha:
                raise ValueError(f'Download checksum mismatch: {remote}')
            # Do not overwrite another concurrent download.
            try:
                os.link(temp, destination)
            except FileExistsError:
                if sha256(destination) != expected_sha:
                    raise ValueError(f'Conflicting cache file: {destination}')
            return destination
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            temp.unlink(missing_ok=True)


def convert(samples):
    """Preserve train 8-decimal/clipped and val 6-decimal/pixel-box conventions."""
    labels = {}
    images = []
    annotations = []
    selected = {}
    for split in ('train', 'val'):
        subset = sorted((s for s in samples if split in s.get('tags', [])), key=lambda s: s['filepath'])
        for sample in subset:
            name = safe_name(sample['filepath'])
            key = f'{split}/{name}'
            if key in selected:
                raise ValueError(f'Duplicate sample: {key}')
            selected[key] = sample
            w, h = sample['metadata']['width'], sample['metadata']['height']
            if w <= 0 or h <= 0:
                raise ValueError('Invalid image dimensions')
            if split == 'val':
                image_id = len(images) + 1
                images.append(dict(id=image_id, file_name=name, width=w, height=h))
            lines = []
            for d in sample['ground_truth']['detections']:
                if d['label'] in IGNORED_LABELS:
                    continue
                c = CLASS_TO_INDEX[d['label']]
                box = d['bounding_box']
                if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    raise ValueError('Invalid bounding box')
                if split == 'train':
                    yolo = clipped_yolo_box(box)
                    if yolo is None:
                        continue
                    lines.append(f'{c} ' + ' '.join(f'{v:.8f}' for v in yolo))
                else:
                    # The pinned mirror normalizes integer VisDrone pixel boxes.
                    pixel = [float(v) * size for v, size in zip(box, (w, h, w, h))]
                    if any(abs(v - round(v)) > 1e-6 for v in pixel):
                        raise ValueError('Validation coordinates are not integer source pixels')
                    x, y, bw, bh = [float(round(v)) for v in pixel]
                    if bw <= 0 or bh <= 0:
                        continue
                    yolo = ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)
                    lines.append(f'{c} ' + ' '.join(f'{v:.6f}' for v in yolo))
                    annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                        category_id=c + 1, bbox=[x, y, bw, bh], area=bw * bh,
                        iscrowd=0, truncation=d['truncation'], occlusion=d['occlusion']))
            # Historical val labels have no trailing newline; train labels do.
            labels[f'{split}/{Path(name).stem}.txt'] = '\n'.join(lines) + ('\n' if lines and split == 'train' else '')
    if {Path(k).name for k in selected if k.startswith('train/')} & {Path(k).name for k in selected if k.startswith('val/')}:
        raise ValueError('Train/val overlap')
    coco = dict(images=images, annotations=annotations,
                categories=[dict(id=i + 1, name=n, supercategory='object') for i, n in enumerate(NAMES)])
    return selected, labels, coco


def verify(dataset, lock):
    for group in ('images', 'labels'):
        actual = {str(p.relative_to(dataset / group)) for p in (dataset / group).glob('*/*') if p.is_file()}
        if actual != set(lock[group]):
            raise ValueError(f'{group} file inventory differs from lock')
        for relative, digest in lock[group].items():
            if sha256(dataset / group / relative) != digest:
                raise ValueError(f'{group} checksum mismatch: {relative}')
    coco = json.loads((dataset / 'annotations/val_coco_gt.json').read_text())
    if json_digest(coco) != lock['val_coco_semantic_sha256']:
        raise ValueError('COCO validation ground truth mismatch')
    print('VERIFIED: train=6471 val=548; all images, labels, and COCO GT match the locked dataset', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=ROOT / 'datasets/visdrone.lock.json')
    parser.add_argument('--verify-existing', type=Path)
    parser.add_argument('--download', action='store_true', help='Explicitly allow network downloads (~GBs)')
    parser.add_argument('--cache', type=Path, default=ROOT / 'downloads/visdrone')
    parser.add_argument('--output', type=Path, default=ROOT / 'Full/data/visdrone_det_yolo_10class')
    parser.add_argument('--samples-json', type=Path, help='Use local pinned metadata instead of downloading')
    parser.add_argument('--source-images', type=Path, action='append', help='Offline image directory; repeat as needed')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if lock['revision'] != REVISION or lock['samples_sha256'] != SAMPLES_SHA:
        raise ValueError('Lock does not match the source revision')
    if args.verify_existing:
        verify(args.verify_existing.resolve(), lock)
        return
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}; use --verify-existing')
    if not args.download and (args.samples_json is None or not args.source_images):
        parser.error('Use --download or supply both --samples-json and --source-images for offline conversion')
    samples_path = args.samples_json or fetch('samples.json', args.cache / 'samples.json', SAMPLES_SHA)
    if sha256(samples_path) != SAMPLES_SHA:
        raise ValueError('Metadata checksum mismatch')
    selected, labels, coco = convert(json.loads(samples_path.read_text())['samples'])
    if {s: sum(k.startswith(s + '/') for k in selected) for s in ('train', 'val')} != {'train': 6471, 'val': 548}:
        raise ValueError('Unexpected split counts')
    if {k: hashlib.sha256(v.encode()).hexdigest() for k, v in labels.items()} != lock['labels']:
        raise ValueError('Converted labels differ from current experiment data')
    if json_digest(coco) != lock['val_coco_semantic_sha256'] or set(selected) != set(lock['images']):
        raise ValueError('Converted dataset differs from lock')
    sources = {}
    for directory in args.source_images or []:
        for path in directory.glob('*.jpg'):
            if path.name in sources and path.resolve() != sources[path.name]:
                raise ValueError(f'Duplicate offline image: {path.name}')
            sources[path.name] = path.resolve()

    def get_image(item):
        relative, sample = item
        name = safe_name(sample['filepath'])
        source = sources.get(name)
        if source is None:
            if not args.download:
                raise FileNotFoundError(name)
            source = fetch(sample['filepath'], args.cache / 'data' / name, lock['images'][relative])
        if sha256(source) != lock['images'][relative]:
            raise ValueError(f'Image checksum mismatch: {name}')
        return relative, source.resolve()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        image_paths = dict(executor.map(get_image, selected.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.visdrone-build-', dir=output.parent))
    try:
        for relative, source in image_paths.items():
            target = staging / 'images' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.path.relpath(source, target.parent))
        for relative, content in labels.items():
            target = staging / 'labels' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
        (staging / 'annotations').mkdir()
        (staging / 'annotations/val_coco_gt.json').write_text(json.dumps(coco), encoding='utf-8')
        config = f'path: {json.dumps(str(output), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/val\nnames:\n'
        config += ''.join(f'  {i}: {name}\n' for i, name in enumerate(NAMES))
        (staging / 'dataset.yaml').write_text(config, encoding='utf-8')
        provenance = dict(source=REPO, revision=REVISION, samples_sha256=SAMPLES_SHA,
                          lock_sha256=sha256(args.lock), train=6471, val=548,
                          image_storage='symlinks; retain source/download cache', model_execution=False)
        (staging / 'annotations/reproduction.json').write_text(json.dumps(provenance, indent=2) + '\n')
        verify(staging, lock)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
        print(f'Dataset ready: {output}')
    except Exception:
        print(f'Incomplete build retained for inspection: {staging}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
