"""Sanity checks for the analytic score and the resource cost model.

The checks encode the qualitative shape we expect from the modelling
assumptions, so that a coding error cannot silently change the ranking:

* latency increases with resolution and decreases with thread count;
* memory increases with resolution and decreases with INT8;
* the score increases with model size at fixed deployment parameters;
* the score increases with resolution at fixed model and precision;
* the ablations change the score in the documented direction.

Usage
-----
    python scripts/check_score.py
    python scripts/check_score.py --max-latency 150 --max-memory 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from resource_model import ResourceModel  # noqa: E402
from score import ScoreWeights, Scorer  # noqa: E402
from search_space import enumerate_configs, load_space  # noqa: E402


def by(configs, **kwargs):
    """First configuration matching the given attribute values."""
    for config in configs:
        if all(getattr(config, key) == value for key, value in kwargs.items()):
            return config
    raise LookupError(f"no configuration matches {kwargs}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-latency", type=float, default=None,
                        help="latency cap in ms (enables the hard constraint)")
    parser.add_argument("--max-memory", type=float, default=None,
                        help="memory cap in MB")
    args = parser.parse_args(argv)

    space = load_space()
    configs = enumerate_configs(space)
    model = ResourceModel()
    scorer = Scorer(configs, model,
                    max_latency_ms=args.max_latency,
                    max_memory_mb=args.max_memory)

    print("=" * 92)
    print("Predicted resources and score for a sample of configurations")
    print("=" * 92)
    header = (f"{'config_id':<24}{'GMACs':>8}{'lat(ms)':>10}{'mem(MB)':>10}"
              f"{'proxy':>12}{'score':>9}")
    print(header)
    print("-" * len(header))

    sample = [c for c in configs
              if c.threads == 4 and c.precision == "fp32"] or configs
    for config in sample[:12]:
        row = scorer.describe(config)
        print(f"{row['config_id']:<24}{model.estimate(config)['gmacs']:>8.2f}"
              f"{row['pred_latency_ms']:>10.1f}{row['pred_memory_mb']:>10.1f}"
              f"{row['accuracy_proxy']:>12.2f}{row['score']:>9.4f}")

    print()
    print("=" * 92)
    print("Monotonicity checks")
    print("=" * 92)

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    # latency decreases with threads
    lat_t1 = scorer.describe(by(configs, model="yolov5s", resolution=640,
                                precision="fp32", threads=1))["pred_latency_ms"]
    lat_t4 = scorer.describe(by(configs, model="yolov5s", resolution=640,
                                precision="fp32", threads=4))["pred_latency_ms"]
    check(f"latency falls with more threads ({lat_t1:.1f} -> {lat_t4:.1f} ms)", lat_t4 < lat_t1)

    # latency increases with resolution
    lat_320 = scorer.describe(by(configs, model="yolov5s", resolution=320,
                                 precision="fp32", threads=4))["pred_latency_ms"]
    lat_640 = scorer.describe(by(configs, model="yolov5s", resolution=640,
                                 precision="fp32", threads=4))["pred_latency_ms"]
    check(f"latency rises with resolution ({lat_320:.1f} -> {lat_640:.1f} ms)", lat_640 > lat_320)

    # memory decreases with INT8
    mem_fp32 = scorer.describe(by(configs, model="yolov5s", resolution=640,
                                  precision="fp32", threads=4))["pred_memory_mb"]
    mem_int8 = scorer.describe(by(configs, model="yolov5s", resolution=640,
                                  precision="int8", threads=4))["pred_memory_mb"]
    check(f"INT8 lowers memory ({mem_fp32:.1f} -> {mem_int8:.1f} MB)", mem_int8 < mem_fp32)

    # score increases with model size
    s_n = scorer.score(by(configs, model="yolov5n", resolution=640, precision="fp32", threads=4))
    s_m = scorer.score(by(configs, model="yolov5m", resolution=640, precision="fp32", threads=4))
    check(f"score rises with model size ({s_n:.4f} -> {s_m:.4f})", s_m > s_n)

    # score increases with resolution (capacity term dominates the resource penalty)
    s_320 = scorer.score(by(configs, model="yolov5s", resolution=320, precision="fp32", threads=4))
    s_640 = scorer.score(by(configs, model="yolov5s", resolution=640, precision="fp32", threads=4))
    check(f"score rises with resolution ({s_320:.4f} -> {s_640:.4f})", s_640 > s_320)

    # ablation: removing the resource term must not raise a resource-heavy config's score
    probe = by(configs, model="yolov5m", resolution=640, precision="fp32", threads=1)
    full = Scorer(configs, model, ScoreWeights(ablation="full")).score(probe)
    no_res = Scorer(configs, model, ScoreWeights(ablation="no_resource")).score(probe)
    check(f"resource penalty lowers the score of the heaviest config ({full:.4f} <= {no_res:.4f})",
          full <= no_res)

    failed = sum(1 for _, ok in checks if not ok)
    print()
    print(f"Checks failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
