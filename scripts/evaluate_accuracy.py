"""Evaluate detection accuracy (mAP) of an ONNX detector on the fixed COCO subset.

Accuracy and latency are measured by different tools on purpose:

* this script produces **mAP** with a documented preprocessing and NMS protocol,
* ``scripts/benchmark_pi.py`` produces **latency and peak memory**.

A fair mAP needs letterboxing, class-aware NMS and COCO's matching rules; folding
that into a timing script is how silent accuracy bugs get in. The two are merged
by passing this script's output to ``benchmark_pi.py --accuracy-json``.

Output
------
``--out`` receives both

* ``accuracy_map50.json``  -- ``{"<model>|<resolution>|<precision>": mAP50}``,
  the format ``benchmark_pi.py`` expects, and
* ``accuracy_detail.json`` -- mAP50, mAP50-95 and the detection protocol.

Usage
-----
    python scripts/evaluate_accuracy.py \
        --models-dir models --data data/coco_subset \
        --models yolov5n --resolutions 320 416 --precision fp32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: COCO's 80 detector classes mapped to the 91-entry annotation space.
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
]


# ---------------------------------------------------------------------------
# Pre-processing: letterbox (aspect ratio preserved)
# ---------------------------------------------------------------------------
def letterbox(image: np.ndarray, size: int, color: int = 114) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize to fit ``size`` x ``size`` keeping aspect ratio, then pad."""
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    pad_w, pad_h = (size - new_w) // 2, (size - new_h) // 2

    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = pad_h, size - new_h - pad_h
    left, right = pad_w, size - new_w - pad_w
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(color, color, color))
    return image, ratio, (pad_w, pad_h)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------
def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Greedy non-maximum suppression on ``xyxy`` boxes."""
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_current + area_rest - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[np.where(iou <= iou_threshold)[0]]
    return keep


def postprocess(output: np.ndarray, ratio: float, pad: Tuple[int, int],
                conf_threshold: float, iou_threshold: float,
                max_detections: int) -> np.ndarray:
    """Convert raw YOLOv5 output into ``[x1, y1, x2, y2, score, class]`` in image pixels."""
    predictions = np.asarray(output, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T  # ensure (N, 5 + nc)

    objectness = predictions[:, 4]
    class_logits = predictions[:, 5:]
    class_scores = class_logits * objectness[:, None]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores > conf_threshold
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)

    boxes = predictions[mask, :4].copy()
    scores = scores[mask]
    class_ids = class_ids[mask]

    # cx, cy, w, h -> x1, y1, x2, y2 (input-image coordinates)
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # undo letterbox
    pad_w, pad_h = pad
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / ratio

    detections: List[List[float]] = []
    for class_id in np.unique(class_ids):
        index = np.where(class_ids == class_id)[0]
        keep = nms(xyxy[index], scores[index], iou_threshold)
        for k in keep:
            x1, y1, x2, y2 = xyxy[index][k]
            detections.append([x1, y1, x2, y2, float(scores[index][k]), float(class_id)])

    if not detections:
        return np.zeros((0, 6), dtype=np.float32)
    detections = np.asarray(detections, dtype=np.float32)
    order = detections[:, 4].argsort()[::-1][:max_detections]
    return detections[order]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate_configuration(model_path: Path, images: List[Path], image_ids: List[int],
                           size: int, conf: float, iou: float) -> Dict[str, object]:
    import cv2
    import onnxruntime as ort

    options = ort.SessionOptions()
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    results: List[Dict] = []
    started = time.perf_counter()

    for path, image_id in zip(images, image_ids):
        image = cv2.imread(str(path))
        if image is None:
            continue
        original_h, original_w = image.shape[:2]
        padded, ratio, pad = letterbox(image, size)
        tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None, ...]

        output = session.run(None, {input_name: tensor})[0]
        detections = postprocess(output, ratio, pad, conf, iou, max_detections=300)

        for x1, y1, x2, y2, score, class_id in detections:
            x1 = float(np.clip(x1, 0, original_w - 1))
            y1 = float(np.clip(y1, 0, original_h - 1))
            x2 = float(np.clip(x2, 0, original_w - 1))
            y2 = float(np.clip(y2, 0, original_h - 1))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            results.append({
                "image_id": int(image_id),
                "category_id": COCO80_TO_COCO91[int(class_id)],
                "bbox": [x1, y1, width, height],
                "score": float(score),
            })

    elapsed = time.perf_counter() - started
    return {"detections": results, "seconds": elapsed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "coco_subset"))
    parser.add_argument("--models", nargs="+", default=["yolov5n"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[320, 416, 640])
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="confidence threshold before NMS (YOLOv5 uses 0.001 for mAP)")
    parser.add_argument("--iou", type=float, default=0.65,
                        help="NMS IoU threshold (YOLOv5 default for validation)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images")
    parser.add_argument("--out", default=None,
                        help="output directory (default: results/accuracy)")
    args = parser.parse_args(argv)

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("ERROR: pycocotools is required: pip install -r requirements-benchmark.txt",
              file=sys.stderr)
        return 2

    data_dir = Path(args.data)
    annotations_path = data_dir / "annotations.json"
    images_dir = data_dir / "images"
    if not annotations_path.is_file() or not images_dir.is_dir():
        print(f"ERROR: subset not found under {data_dir}. "
              f"Run scripts/prepare_coco_subset.py first.", file=sys.stderr)
        return 2

    with annotations_path.open(encoding="utf-8") as handle:
        subset = json.load(handle)
    images_meta = sorted(subset["images"], key=lambda item: item["id"])
    if args.limit:
        images_meta = images_meta[: args.limit]

    image_paths = [images_dir / meta["file_name"] for meta in images_meta]
    image_ids = [meta["id"] for meta in images_meta]
    present = [(path, image_id) for path, image_id in zip(image_paths, image_ids) if path.is_file()]
    if not present:
        print(f"ERROR: no images present in {images_dir}", file=sys.stderr)
        return 2
    image_paths, image_ids = [p for p, _ in present], [i for _, i in present]

    print("=" * 76)
    print("Detection accuracy evaluation (COCO mAP)")
    print("=" * 76)
    print(f"subset     : {len(image_paths)} images from {data_dir.name}")
    print(f"protocol   : letterbox, conf>{args.conf}, NMS IoU={args.iou}, per-class NMS")
    print()

    coco_gt = COCO(str(annotations_path))

    accuracy_map: Dict[str, float] = {}
    details: Dict[str, Dict[str, object]] = {}

    for model in args.models:
        model_path = Path(args.models_dir) / f"{model}_{args.precision}.onnx"
        if not model_path.is_file():
            print(f"skip {model}: {model_path.name} not found")
            continue

        for resolution in args.resolutions:
            outcome = evaluate_configuration(model_path, image_paths, image_ids,
                                             resolution, args.conf, args.iou)
            detections = outcome["detections"]
            key = f"{model}|{resolution}|{args.precision}"

            if not detections:
                print(f"  {key:<28} no detections above threshold")
                continue

            coco_dt = coco_gt.loadRes(detections)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

            # stats[0] = mAP@[.5:.95], stats[1] = mAP@.5
            map_50_95 = float(evaluator.stats[0])
            map_50 = float(evaluator.stats[1])
            accuracy_map[key] = round(map_50, 4)
            details[key] = {
                "map50": round(map_50, 4),
                "map50_95": round(map_50_95, 4),
                "num_images": len(image_paths),
                "num_detections": len(detections),
                "conf_threshold": args.conf,
                "nms_iou": args.iou,
                "seconds": round(outcome["seconds"], 1),
            }
            print(f"  {key:<28} mAP50={map_50:.4f}  mAP50-95={map_50_95:.4f}  "
                  f"({outcome['seconds']:.1f}s)")

    if not accuracy_map:
        print("\nERROR: no configuration produced detections.", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else REPO_ROOT / "results" / "accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accuracy_map50.json").write_text(json.dumps(accuracy_map, indent=2),
                                                 encoding="utf-8")
    (out_dir / "accuracy_detail.json").write_text(json.dumps(details, indent=2),
                                                  encoding="utf-8")

    print()
    print(f"written to {out_dir / 'accuracy_map50.json'}")
    print("Feed that file to benchmark_pi.py via --accuracy-json to merge accuracy")
    print("with latency and memory into one benchmark table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
