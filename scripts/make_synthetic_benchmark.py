"""Generate a *synthetic* benchmark table to exercise the pipeline.

This is **not** a measurement and must never be reported as a result. It exists
so that the evaluator, the search algorithms, the metrics and the figures can be
developed and tested before the Raspberry Pi run produces the real table.

The synthetic ground truth is deliberately *not* identical to the analytic
prediction: it uses a different accuracy curve and multiplies the predicted
latency and memory by a model-specific bias and log-normal noise. If it matched
the predictor exactly, the "does the heuristic predict reality?" analysis would
be trivially perfect and would hide bugs.

Usage
-----
    python scripts/make_synthetic_benchmark.py --out benchmarks/synthetic_demo.csv
    python scripts/make_synthetic_benchmark.py --seed 7 --noise 0.08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from resource_model import ResourceModel  # noqa: E402
from search_space import enumerate_configs, load_space  # noqa: E402

#: Relative accuracy of each variant against the largest one.
SIZE_FACTOR = {"yolov5n": 0.82, "yolov5s": 0.93, "yolov5m": 1.00}
#: Per-variant multiplicative bias on the predicted latency (hardware reality).
LATENCY_BIAS = {"yolov5n": 0.95, "yolov5s": 1.05, "yolov5m": 1.15}
#: Accuracy retention after INT8 quantisation.
INT8_RETENTION = 0.985


def synthetic_accuracy(model: str, resolution: int, precision: str,
                       noise: float = 0.0, rng=None) -> float:
    """Plausible mAP@0.5 curve: rises with size and resolution, dips for INT8.

    ``noise`` adds multiplicative measurement noise. Without it the synthetic
    truth would be an exact function of the score's own factors, the heuristic
    would hit the oracle every time, and the comparison would test nothing.
    """
    ceiling = 0.62
    size = SIZE_FACTOR[model]
    res = 1.0 - np.exp(-resolution / 320.0)
    quant = INT8_RETENTION if precision == "int8" else 1.0
    value = ceiling * size * res * quant
    if noise > 0.0 and rng is not None:
        value *= float(np.exp(rng.normal(0.0, noise)))
    return float(min(value, 0.999))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "synthetic_demo.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.05,
                        help="log-normal noise on latency and memory")
    parser.add_argument("--acc-noise", type=float, default=0.03,
                        help="multiplicative measurement noise on accuracy")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="restrict the table to these precisions, e.g. fp32")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    space = load_space()
    configs = enumerate_configs(space, precisions=args.precisions)
    resource = ResourceModel()

    rows = []
    for config in configs:
        estimate = resource.estimate(config)
        latency = (estimate["latency_ms"]
                   * LATENCY_BIAS[config.model]
                   * float(np.exp(rng.normal(0.0, args.noise))))
        memory = estimate["memory_mb"] * float(np.exp(rng.normal(0.0, args.noise)))
        p95 = latency * float(1.05 + rng.uniform(0.0, 0.08))

        rows.append({
            "config_id": config.config_id,
            "model": config.model,
            "resolution": config.resolution,
            "precision": config.precision,
            "threads": config.threads,
            "accuracy_map50": round(synthetic_accuracy(config.model, config.resolution,
                                                       config.precision,
                                                       args.acc_noise, rng), 4),
            "latency_median_ms": round(latency, 2),
            "latency_p95_ms": round(p95, 2),
            "memory_mb": round(memory, 2),
            "num_images": args.num_images,
            "source": "synthetic",
            "notes": "placeholder for pipeline testing only",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)

    print(f"wrote {len(frame)} synthetic rows to {out}")
    print(frame[["config_id", "accuracy_map50", "latency_median_ms", "memory_mb"]]
          .head(6).to_string(index=False))
    print("...")
    print("REMINDER: source=synthetic; these are not measurements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
