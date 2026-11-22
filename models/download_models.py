"""Export the classic YOLOv5 (v7.0) variants to ONNX for the benchmark step.

This is an **optional, one-off preparation step**. The core experiment never loads
a model: it reads the archived benchmark table. You only need this script if you
are regenerating measurements on real hardware.

Why ``torch.hub`` and not ``ultralytics.YOLO("yolov5n.pt")``
------------------------------------------------------------
The ``ultralytics`` package resolves ``yolov5n.pt`` to the **YOLOv5u** weights
(anchor-free, decoupled head with DFL regression). YOLOv5u is a different
network -- the nano variant has 2.64M parameters against 1.87M for v7.0.

``src/model_spec.py`` describes the **classic v7.0** topology, and the capacity
score is a function of the convolution shapes, so benchmarking a YOLOv5u graph
while scoring a v7.0 specification would silently score the wrong network.
This script therefore builds the model from the ``ultralytics/yolov5`` hub
repository (v7.0 code) and loads the genuine v7.0 checkpoints, exporting with
dynamic axes so one file serves every input resolution.

Dependencies (not in requirements.txt, deliberately): ``torch``, ``onnx`` and
network access.

Usage
-----
    python models/download_models.py --dry-run
    python models/download_models.py --variants yolov5n yolov5s yolov5m
    python models/download_models.py --mirror https://ghproxy.net/
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Classic YOLOv5 v7.0 checkpoints on the official release page.
RELEASE = "https://github.com/ultralytics/yolov5/releases/download/v7.0"
WEIGHTS = {
    "yolov5n": "yolov5n.pt",
    "yolov5s": "yolov5s.pt",
    "yolov5m": "yolov5m.pt",
}


def fetch_weights(filename: str, target: Path, mirror: Optional[str]) -> None:
    """Download a checkpoint, optionally through a mirror prefix."""
    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"  {filename}: already present ({target.stat().st_size / 1e6:.1f} MB)")
        return

    url = f"{RELEASE}/{filename}"
    if mirror:
        url = f"{mirror.rstrip('/')}/{url}"

    print(f"  {filename}: downloading from {url.split('/')[2]} ...", end="", flush=True)
    urllib.request.urlretrieve(url, target)
    print(f"\r  {filename}: done ({target.stat().st_size / 1e6:.1f} MB)")


def build_and_load(variant: str, weights_path: Path):
    """Build the v7.0 architecture from the hub repo and load the checkpoint."""
    import torch

    model = torch.hub.load("ultralytics/yolov5", variant, pretrained=False,
                           trust_repo=True, autoshape=False)
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = state.float().state_dict() if hasattr(state, "state_dict") else state
    model.load_state_dict(state)
    return model.float().eval()


def export_variant(variant: str, weights_path: Path, target: Path, imgsz: int) -> None:
    import torch

    print(f"[{variant}] building v7.0 model from torch.hub ...")
    model = build_and_load(variant, weights_path)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"[{variant}] parameters: {parameters / 1e6:.3f}M")

    dummy = torch.zeros(1, 3, imgsz, imgsz)
    print(f"[{variant}] exporting to {target.name} (dynamic axes) ...")
    torch.onnx.export(
        model, dummy, str(target),
        opset_version=12, do_constant_folding=True, dynamo=False,
        input_names=["images"], output_names=["output"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 1: "anchors"},
        },
    )
    print(f"[{variant}] wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=list(WEIGHTS),
                        choices=list(WEIGHTS))
    parser.add_argument("--out", default=str(REPO_ROOT / "models"))
    parser.add_argument("--weights-dir", default=str(REPO_ROOT / "models" / "weights"),
                        help="where the .pt checkpoints are cached")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mirror", default=None,
                        help="prefix for the download URL, e.g. https://ghproxy.net/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    weights_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("YOLOv5 v7.0 -> ONNX export (classic anchor-based checkpoints)")
    print("=" * 74)
    print(f"output dir : {out_dir}")
    print(f"weights dir: {weights_dir}")
    print(f"reference  : {args.imgsz}px, dynamic axes")

    plan = [(v, weights_dir / WEIGHTS[v], out_dir / f"{v}_fp32.onnx") for v in args.variants]
    for variant, weights, target in plan:
        state = "exists" if target.is_file() else "to export"
        print(f"  {variant:<9} {weights.name:<14} -> {target.name}  ({state})")

    if args.dry_run:
        print("\ndry run: nothing downloaded or exported.")
        return 0

    try:
        import torch  # noqa: F401
    except ImportError:
        print("\nERROR: torch is not installed; export needs torch.", file=sys.stderr)
        return 2

    exported: List[str] = []
    try:
        for variant, weights, target in plan:
            fetch_weights(weights.name, weights, args.mirror)
            if target.is_file():
                print(f"[{variant}] already exported, skipping")
                exported.append(target.name)
                continue
            export_variant(variant, weights, target, args.imgsz)
            exported.append(target.name)
    except Exception as error:  # noqa: BLE001 - surface the real cause
        print(f"\nERROR during export: {error}", file=sys.stderr)
        print("If this is a network error, retry with --mirror https://ghproxy.net/",
              file=sys.stderr)
        return 1

    print()
    print(f"exported {len(exported)} file(s) to {out_dir}")
    print("Verify the export against the analytic specification:")
    print("    python scripts/check_model_spec.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
