"""Export the YOLOv5 variants to ONNX for the benchmark step.

This is an **optional, one-off preparation step**. The core experiment never
loads a model: it reads the archived benchmark table. You only need this script
if you are regenerating measurements on real hardware.

It requires ``ultralytics`` and ``torch``, which are deliberately *not* in
``requirements.txt`` — they are heavy, and the core results must stay
reproducible without them.

The exported files follow the naming convention expected by
``scripts/benchmark_pi.py``:

    models/yolov5n_fp32.onnx
    models/yolov5s_fp32.onnx
    models/yolov5m_fp32.onnx

Dynamic axes are enabled so that a single file can be benchmarked at 320, 416
and 640 pixels.

Usage
-----
    pip install ultralytics
    python models/download_models.py --dry-run
    python models/download_models.py --variants yolov5n yolov5s yolov5m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Classic YOLOv5 (v7.0) anchor-based checkpoints.
#
# IMPORTANT: these are deliberately the *anchor-based* v7.0 weights, not the
# newer ``yolov5nu/su/mu.pt`` files. YOLOv5u is a different model family: it is
# anchor-free, uses a decoupled detection head with a distribution-focal-loss
# regression branch, and therefore has different layer shapes and a different
# parameter count.
#
# ``src/model_spec.py`` describes the v7.0 topology, and the reference parameter
# counts used to validate it (1.9M / 7.2M / 21.2M) are v7.0 numbers. The whole
# capacity score is a function of the convolution shapes, so exporting a YOLOv5u
# checkpoint would mean scoring a network that is not the one being benchmarked.
#
# If you ever switch families, change ``model_spec.py`` too, and re-run
# ``python scripts/check_model_spec.py`` -- it compares the analytic parameter
# count against the ONNX graph and fails when they diverge.
WEIGHTS = {
    "yolov5n": "yolov5n.pt",
    "yolov5s": "yolov5s.pt",
    "yolov5m": "yolov5m.pt",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=list(WEIGHTS),
                        choices=list(WEIGHTS), help="which variants to export")
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    parser.add_argument("--imgsz", type=int, default=640,
                        help="reference export size (dynamic axes keep it flexible)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    print("=" * 74)
    print("YOLOv5 -> ONNX export")
    print("=" * 74)
    print(f"output dir : {out_dir}")
    print(f"reference  : {args.imgsz}px, dynamic axes")

    plan = [(v, WEIGHTS[v], out_dir / f"{v}_fp32.onnx") for v in args.variants]
    for variant, weights, target in plan:
        state = "exists" if target.is_file() else "to export"
        print(f"  {variant:<9} {weights:<14} -> {target.name}  ({state})")

    if args.dry_run:
        print("\ndry run: nothing downloaded or exported.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\nERROR: ultralytics is not installed.", file=sys.stderr)
        print("       pip install ultralytics", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    exported: List[str] = []

    for variant, weights, target in plan:
        if target.is_file():
            print(f"\n[{variant}] already exported, skipping")
            exported.append(target.name)
            continue
        print(f"\n[{variant}] exporting {weights} ...")
        model = YOLO(weights)
        produced = model.export(format="onnx", dynamic=True, simplify=True,
                                imgsz=args.imgsz)
        produced_path = Path(str(produced))
        produced_path.replace(target)
        exported.append(target.name)
        print(f"[{variant}] wrote {target}")

    print()
    print(f"exported {len(exported)} file(s) to {out_dir}")
    print("Next: python scripts/benchmark_pi.py --dry-run")
    print("Note: INT8 export additionally needs a calibration dataset; it is treated")
    print("      as an extension and is not part of the default pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
