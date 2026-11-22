"""Regenerate the benchmark table on real hardware (optional step).

This script is **not** required to reproduce the core results: those run from the
archived table. It exists so the measurements can be regenerated, and so the
report can compare prediction against reality.

Two measurement details matter and are handled explicitly:

* **Peak memory is per-process.** ``ru_maxrss`` reports the high-water mark since
  the process started, so measuring several configurations inside one process
  lets a large model pollute the reading of every configuration that follows it.
  Each configuration is therefore measured in its **own subprocess**, and the
  parent aggregates the results.
* **The source label reflects the real machine.** Writing ``raspberry_pi_4b``
  unconditionally would label laptop measurements as Raspberry Pi data -- exactly
  the kind of quiet falsification the report must avoid. The platform is detected
  and ``--source`` can override it explicitly.

For every configuration it measures median and p95 latency per image, and peak
process memory.

Detection accuracy is **not** measured here: a fair mAP needs the full COCO
evaluation pipeline (scaling, NMS thresholds, per-class matching), kept separate
on purpose. Supply accuracies with ``--accuracy-json``; without it the script
refuses to write a table unless ``--allow-missing-accuracy`` is given.

Usage
-----
    python scripts/benchmark_pi.py --dry-run
    python scripts/benchmark_pi.py --images-dir data/coco_subset/val \
        --accuracy-json data/accuracy_map50.json \
        --out benchmarks/pi4b_2gb_reference.csv
    python scripts/benchmark_pi.py --precisions fp32 --source laptop \
        --images-dir data/coco_subset/val --accuracy-json data/accuracy_map50.json
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from search_space import enumerate_configs, load_space  # noqa: E402


# ---------------------------------------------------------------------------
# Platform identification
# ---------------------------------------------------------------------------
def detect_platform_source() -> str:
    """Best-effort label for where this measurement was taken."""
    machine = platform.machine().lower()

    if platform.system() == "Linux":
        model_file = Path("/proc/device-tree/model")
        if model_file.exists():
            try:
                model = model_file.read_text(errors="ignore").lower()
            except OSError:
                model = ""
            if "raspberry pi" in model:
                if "4" in model and "model b" in model:
                    return "raspberry_pi_4b"
                return "raspberry_pi"

    if machine.startswith("arm") or machine.startswith("aarch64"):
        return "arm_device"
    return "laptop"


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
    """Peak resident set size of *this* process, in MB."""
    try:  # Linux / Raspberry Pi OS
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024.0 if platform.system() != "Darwin" else usage / 1e6
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)

            # GetCurrentProcess returns a HANDLE; without an explicit restype
            # ctypes assumes a 32-bit int and truncates the pseudo-handle, which
            # makes the memory query fail silently.
            kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
            process = kernel32.GetCurrentProcess()

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)

            query = psapi.GetProcessMemoryInfo
            query.argtypes = [ctypes.wintypes.HANDLE,
                              ctypes.POINTER(_ProcessMemoryCounters),
                              ctypes.wintypes.DWORD]
            query.restype = ctypes.wintypes.BOOL
            if query(process, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 1e6
        except OSError:  # pragma: no cover - defensive
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


# ---------------------------------------------------------------------------
# Child mode: measure exactly one configuration in a fresh process
# ---------------------------------------------------------------------------
def run_single_configuration(args) -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.", file=sys.stderr)
        return 2

    space = load_space()
    configs = enumerate_configs(space)
    match = [c for c in configs if c.config_id == args.single_config]
    if not match:
        print(f"ERROR: unknown configuration {args.single_config!r}", file=sys.stderr)
        return 2
    config = match[0]

    model_path = Path(args.models_dir) / f"{config.model}_{config.precision}.onnx"
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    images = list_images(Path(args.images_dir), args.num_images)

    options = ort.SessionOptions()
    options.intra_op_num_threads = config.threads
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    metrics = measure_config(session, images, config.resolution, args.warmup, args.repeats)

    # Read the peak *after* the run, in a process that loaded only this model.
    payload = {
        "config_id": config.config_id,
        "latency_median_ms": metrics["latency_median_ms"],
        "latency_p95_ms": metrics["latency_p95_ms"],
        "memory_mb": peak_rss_mb(),
        "num_timings": metrics["num_timings"],
        "platform": platform.machine(),
        "platform_system": platform.system(),
    }
    print("RESULT_JSON " + json.dumps(payload))
    return 0


# ---------------------------------------------------------------------------
# Parent mode: one subprocess per configuration
# ---------------------------------------------------------------------------
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
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="restrict to these precisions, e.g. fp32")
    parser.add_argument("--threads", nargs="+", type=int, default=None)
    parser.add_argument("--source", default=None,
                        help="override the detected platform label "
                             "(default: auto-detect, e.g. raspberry_pi_4b or laptop)")
    parser.add_argument("--allow-missing-accuracy", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without loading a model")

    # internal child-mode switch
    parser.add_argument("--single-config", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    if args.single_config:
        return run_single_configuration(args)

    space = load_space()
    configs = enumerate_configs(space, models=args.models, resolutions=args.resolutions,
                                precisions=args.precisions, threads=args.threads)
    source = args.source or detect_platform_source()

    print("=" * 78)
    print("Hardware benchmark")
    print("=" * 78)
    print(f"platform   : {platform.system()} {platform.machine()}")
    print(f"source tag : {source}" + ("  (overridden)" if args.source else "  (detected)"))
    print(f"configs    : {len(configs)}")
    print(f"images     : {args.num_images} (warm-up {args.warmup}, repeats {args.repeats})")
    print("isolation  : one subprocess per configuration (clean peak-memory reading)")
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
        import onnxruntime  # noqa: F401  (fail early and clearly)
    except ImportError:
        print("ERROR: onnxruntime is not installed.", file=sys.stderr)
        print("       pip install -r requirements-benchmark.txt", file=sys.stderr)
        return 2

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
        model_path = Path(args.models_dir) / f"{config.model}_{config.precision}.onnx"
        if not model_path.is_file():
            print(f"skip {config.config_id}: {model_path.name} not found")
            continue

        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--single-config", config.config_id,
            "--models-dir", args.models_dir,
            "--images-dir", args.images_dir,
            "--num-images", str(args.num_images),
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        payload = None
        for line in (completed.stdout or "").splitlines():
            if line.startswith("RESULT_JSON "):
                payload = json.loads(line[len("RESULT_JSON "):])
                break
        if payload is None:
            detail = (completed.stderr or completed.stdout or "").strip()[:160]
            print(f"  FAILED {config.config_id}: {detail}")
            continue

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
            "latency_median_ms": round(payload["latency_median_ms"], 2),
            "latency_p95_ms": round(payload["latency_p95_ms"], 2),
            "memory_mb": round(payload["memory_mb"], 1),
            "num_images": args.num_images,
            "source": source,
            "notes": f"{payload['platform_system']} {payload['platform']}, "
                     f"{payload['num_timings']} timed runs, isolated process",
        })
        print(f"  {config.config_id:<26} {payload['latency_median_ms']:8.1f} ms  "
              f"{payload['memory_mb']:7.1f} MB  mAP={value:.4f}")

    if not rows:
        print("ERROR: nothing measured (no model files found).", file=sys.stderr)
        return 1

    import pandas as pd

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {len(rows)} rows to {out} (source={source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
