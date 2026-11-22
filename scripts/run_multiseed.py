"""Multi-seed comparison of the search algorithms.

A single random seed cannot separate "this method is better" from "this method
got lucky". Random search and greedy local search are stochastic, so they are run
over a range of seeds and reported as a distribution; the heuristic is
deterministic and is reported once.

For every budget the script reports:

* mean and standard deviation of the best feasible accuracy,
* mean and standard deviation of the regret against the exhaustive oracle,
* the **feasible-success rate** -- the fraction of runs that found at least one
  feasible configuration. A run that finds nothing is a qualitatively different
  outcome from a run that finds something slightly worse, so it is counted
  separately rather than folded into a regret of zero.

Usage
-----
    python scripts/run_multiseed.py
    python scripts/run_multiseed.py --seeds 30 --budgets 4 8 12 --max-latency 400
    python scripts/run_multiseed.py --precisions fp32 --out results/multiseed_fp32.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_table import BenchmarkTable, split_by_coverage  # noqa: E402
from evaluator import Evaluator, is_feasible  # noqa: E402
from score import ScoreWeights, Scorer  # noqa: E402
from search import GreedyLocalSearch, HeuristicBestFirstSearch, RandomSearch  # noqa: E402
from search_space import enumerate_configs, load_space  # noqa: E402


def oracle_accuracy(configs, table, max_latency, max_memory) -> Optional[float]:
    best = None
    for config in configs:
        measurement = table.get(config.config_id)
        if measurement is None:
            continue
        if not is_feasible(measurement, max_latency, max_memory):
            continue
        best = measurement.accuracy if best is None else max(best, measurement.accuracy)
    return best


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "benchmarks" / "synthetic_demo.csv"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--seeds", type=int, default=30,
                        help="number of seeds (0 .. N-1) for stochastic algorithms")
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--max-memory", type=float, default=None)
    parser.add_argument("--precisions", nargs="+", default=None)
    parser.add_argument("--heuristic-ablations", nargs="+",
                        default=["full", "no_capacity"],
                        choices=["full", "no_resource", "no_capacity"])
    parser.add_argument("--out", default=None,
                        help="CSV output path (default: results/multiseed.csv)")
    args = parser.parse_args(argv)

    space = load_space()
    all_configs = enumerate_configs(space, precisions=args.precisions)
    table = BenchmarkTable(args.table, known_config_ids=[c.config_id for c in all_configs])

    configs, missing = split_by_coverage(all_configs, table)
    if not configs:
        raise SystemExit(
            f"{table.path.name} contains none of the {len(all_configs)} selected configurations"
        )
    if missing:
        print(f"NOTE: {len(missing)} of {len(all_configs)} configurations have no "
              f"measurement and are excluded (e.g. {missing[0].config_id}).")
    if table.is_synthetic():
        print("WARNING: synthetic table - these numbers test the pipeline only.\n")

    oracle = oracle_accuracy(configs, table, args.max_latency, args.max_memory)

    print("=" * 100)
    print(f"table  : {table.path.name}   pool: {len(configs)}   seeds: 0..{args.seeds - 1}")
    print(f"caps   : latency<={args.max_latency} ms, memory<={args.max_memory} MB")
    print(f"oracle : {'n/a' if oracle is None else f'{oracle:.4f}'}")
    print("=" * 100)

    header = (f"{'budget':>7}{'algorithm':>22}{'runs':>6}{'acc mean':>10}{'acc std':>9}"
              f"{'regret mean':>13}{'regret std':>12}{'feas. rate':>12}")
    print(header)
    print("-" * len(header))

    rows: List[Dict[str, object]] = []

    for budget in args.budgets:
        # -- stochastic algorithms over many seeds ------------------------
        for name in ("random", "greedy"):
            accuracies: List[float] = []
            regrets: List[float] = []
            successes = 0
            for seed in range(args.seeds):
                evaluator = Evaluator(table, budget=budget)
                if name == "random":
                    algorithm = RandomSearch(seed=seed, max_latency_ms=args.max_latency,
                                             max_memory_mb=args.max_memory)
                else:
                    algorithm = GreedyLocalSearch(seed=seed, max_latency_ms=args.max_latency,
                                                  max_memory_mb=args.max_memory)
                result = algorithm.run(evaluator, configs)

                if result.best_accuracy is None:
                    successes += 0
                    continue
                successes += 1
                accuracies.append(result.best_accuracy)
                if oracle is not None:
                    regrets.append(oracle - result.best_accuracy)

            acc = summarise(accuracies)
            reg = summarise(regrets)
            success_rate = successes / args.seeds
            print(f"{budget:>7}{name:>22}{args.seeds:>6}{acc['mean']:>10.4f}"
                  f"{acc['std']:>9.4f}{reg['mean']:>13.4f}{reg['std']:>12.4f}"
                  f"{success_rate:>12.2f}")
            rows.append({
                "budget": budget, "algorithm": name, "ablation": "",
                "runs": args.seeds,
                "acc_mean": round(acc["mean"], 4) if accuracies else None,
                "acc_std": round(acc["std"], 4) if accuracies else None,
                "acc_min": round(acc["min"], 4) if accuracies else None,
                "acc_max": round(acc["max"], 4) if accuracies else None,
                "regret_mean": round(reg["mean"], 4) if regrets else None,
                "regret_std": round(reg["std"], 4) if regrets else None,
                "feasible_success_rate": round(success_rate, 3),
                "deterministic": False,
            })

        # -- deterministic heuristic (+ ablations) ------------------------
        for ablation in args.heuristic_ablations:
            evaluator = Evaluator(table, budget=budget)
            scorer = Scorer(configs, weights=ScoreWeights(ablation=ablation),
                            max_latency_ms=args.max_latency,
                            max_memory_mb=args.max_memory)
            algorithm = HeuristicBestFirstSearch(scorer=scorer,
                                                 max_latency_ms=args.max_latency,
                                                 max_memory_mb=args.max_memory)
            result = algorithm.run(evaluator, configs)

            acc = result.best_accuracy
            reg = None if (oracle is None or acc is None) else oracle - acc
            print(f"{budget:>7}{'heuristic/' + ablation:>22}{'1':>6}"
                  f"{'n/a' if acc is None else f'{acc:.4f}':>10}{'-':>9}"
                  f"{'n/a' if reg is None else f'{reg:.4f}':>13}{'-':>12}"
                  f"{'0.00' if acc is None else '1.00':>12}")
            rows.append({
                "budget": budget, "algorithm": "heuristic", "ablation": ablation,
                "runs": 1,
                "acc_mean": None if acc is None else round(acc, 4),
                "acc_std": 0.0 if acc is not None else None,
                "acc_min": None if acc is None else round(acc, 4),
                "acc_max": None if acc is None else round(acc, 4),
                "regret_mean": None if reg is None else round(reg, 4),
                "regret_std": 0.0 if reg is not None else None,
                "feasible_success_rate": 0.0 if acc is None else 1.0,
                "deterministic": True,
            })

    out = Path(args.out) if args.out else REPO_ROOT / "results" / "multiseed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print()
    print("regret = oracle accuracy - accuracy found; a run that finds no feasible")
    print("configuration is counted in the feasible-success rate, not as zero regret.")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
