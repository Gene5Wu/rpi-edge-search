"""Regenerate the benchmark table on a Raspberry Pi (optional hardware step).

This script is **not** required to reproduce the core results: those run from the
archived table. It exists so the measurements can be regenerated on real
hardware, and so the report can compare prediction against reality.

It measures, for every configuration in the search space:

* median and p95 wall-clock latency per image,
* peak process memory.

Detection accuracy is **not** measured here, because a fair mAP requires the full
COCO evaluation pipeline (scaling, NMS settings, per-class matching), which is
deliberately kept separate. Supply the accuracies with ``--accuracy-json``, a
mapping from ``"<model>|<resolution>|<precision>"`` to mAP@0.5. Without it the
script refuses to write a table unless ``--allow-missing-accuracy`` is given, in
which case the file is not a valid benchmark table.

Usage
-----
    python scripts/benchmark_pi.py --dry-run
    python scripts/benchmark_pi.py --images-dir data/coco_subset/val \
        --accuracy-json data/accuracy_map50.json \
        --out benchmarks/pi4b_2gb_reference.csv
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from search_space import enumerate_configs, load_space  # noqa: E402


# ---------------------------------------------------------------------------
# Peak memory (portable, no third-party dependency)
# ---------------------------------------------------------------------------
class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB."""
    try:  # Linux / Raspberry Pi OS
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kilobytes on Linux, bytes on macOS
        return usage / 1024.0 if platform.system() != "Darwin" else usage / 1e6
    except ImportError:
        pass

    try:  # Windows
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if ok:
            return counters.PeakWorkingSetSize / 1e6
    except Exception:  # pragma: no cover - best effort only
        pass

    return float("nan")


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------
def preprocess(image, size: int):
    """Resize to a square input, BGR->RGB, HWC->CHW, normalise to [0, 1]."""
    import cv2
    import numpy as np

    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"could not read image: {image}")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return tensor.transpose(2, 0, 1)[None, ...]


def list_images(images_dir: Path, limit: int) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(f"no images found under {images_dir}")
    return images[:limit]


def measure_config(session, images: List[Path], size: int,
                   warmup: int, repeats: int) -> Dict[str, float]:
    """Run warm-up passes, then time every image ``repeats`` times."""
    timings: List[float] = []
    input_name = session.get_inputs()[0].name

    for index, image in enumerate(images):
        tensor = preprocess(image, size)
        if index < warmup:
            session.run(None, {input_name: tensor})
            continue
        for _ in range(repeats):
            started = time.perf_counter()
            session.run(None, {input_name: tensor})
            timings.append((time.perf_counter() - started) * 1000.0)

    if not timings:
        raise RuntimeError("no timings collected; increase --num-images")

    timings.sort()
    p95_index = max(0, min(len(timings) - 1, int(round(0.95 * (len(timings) - 1)))))
    return {
        "latency_median_ms": statistics.median(timings),
        "latency_p95_ms": timings[p95_index],
        "num_timings": len(timings),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--images-dir", default=None,
                        help="directory of validation images (required to run)")
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "pi4b_2gb_reference.csv"))
    parser.add_argument("--accuracy-json", default=None,
                        help='JSON mapping "<model>|<resolution>|<precision>" to mAP@0.5')
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--resolutions", nargs="+", type=int, default=None)
    parser.add_argument("--precisions", nargs="+", default=None)
    parser.add_argument("--threads", nargs="+", type=int, default=None)
    parser.add_argument("--allow-missing-accuracy", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without loading a model")
    args = parser.parse_args(argv)

    space = load_space()
    configs = enumerate_configs(space, models=args.models, resolutions=args.resolutions,
                                precisions=args.precisions, threads=args.threads)

    print("=" * 78)
    print("Raspberry Pi benchmark")
    print("=" * 78)
    print(f"platform   : {platform.system()} {platform.machine()}")
    print(f"configs    : {len(configs)}")
    print(f"images     : {args.num_images} (warm-up {args.warmup}, repeats {args.repeats})")
    print(f"output     : {args.out}")

    if args.dry_run:
        models_dir = Path(args.models_dir)
        print(f"models dir : {models_dir}")
        print("\nplan:")
        for config in configs[:8]:
            path = models_dir / f"{config.model}_{config.precision}.onnx"
            print(f"  {config.config_id:<26} <- {path.name} "
                  f"({'found' if path.is_file() else 'MISSING'})")
        if len(configs) > 8:
            print(f"  ... ({len(configs) - 8} more)")
        print("\ndry run: nothing measured, nothing written.")
        return 0

    if not args.images_dir:
        parser.error("--images-dir is required (or use --dry-run)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.", file=sys.stderr)
        print("       pip install -r requirements-benchmark.txt", file=sys.stderr)
        return 2

    images = list_images(Path(args.images_dir), args.num_images)
    models_dir = Path(args.models_dir)

    accuracy: Dict[str, float] = {}
    if args.accuracy_json:
        accuracy = json.loads(Path(args.accuracy_json).read_text(encoding="utf-8"))
    elif not args.allow_missing_accuracy:
        print("ERROR: no --accuracy-json supplied. A table without accuracy is not a "
              "valid benchmark table.", file=sys.stderr)
        print("       Provide accuracies, or pass --allow-missing-accuracy to write "
              "an incomplete file.", file=sys.stderr)
        return 2

    rows = []
    for config in configs:
        model_path = models_dir / f"{config.model}_{config.precision}.onnx"
        if not model_path.is_file():
            print(f"skip {config.config_id}: {model_path.name} not found")
            continue

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        session = ort.InferenceSession(str(model_path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        metrics = measure_config(session, images, config.resolution,
                                 args.warmup, args.repeats)

        key = f"{config.model}|{config.resolution}|{config.precision}"
        value = accuracy.get(key)
        if value is None:
            print(f"  WARNING: no accuracy for {key}")
            value = float("nan")

        rows.append({
            "config_id": config.config_id,
            "model": config.model,
            "resolution": config.resolution,
            "precision": config.precision,
            "threads": config.threads,
            "accuracy_map50": value,
            "latency_median_ms": round(metrics["latency_median_ms"], 2),
            "latency_p95_ms": round(metrics["latency_p95_ms"], 2),
            "memory_mb": round(peak_rss_mb(), 1),
            "num_images": args.num_images,
            "source": "raspberry_pi_4b",
            "notes": f"{platform.machine()}, {metrics['num_timings']} timed runs",
        })
        print(f"  {config.config_id:<26} {metrics['latency_median_ms']:8.1f} ms  "
              f"mAP={value:.4f}")

    if not rows:
        print("ERROR: nothing measured (no model files found).", file=sys.stderr)
        return 1

    import pandas as pd

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
