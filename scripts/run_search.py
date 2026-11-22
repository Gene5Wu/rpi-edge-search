"""Run the four search algorithms under identical budgets and report the outcome.

For each budget the script builds a fresh evaluator (so budgets never leak
between runs), runs every algorithm, and reports the best feasible accuracy
found plus the regret against the exhaustive oracle.

Usage
-----
    python scripts/run_search.py
    python scripts/run_search.py --budgets 4 8 12 --max-latency 400
    python scripts/run_search.py --table benchmarks/pi4b_2gb_reference.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_table import BenchmarkTable, split_by_coverage  # noqa: E402
from evaluator import Evaluator, is_feasible  # noqa: E402
from score import ScoreWeights, Scorer  # noqa: E402
from search import (  # noqa: E402
    ExhaustiveSearch,
    GreedyLocalSearch,
    HeuristicBestFirstSearch,
    RandomSearch,
)
from search_space import enumerate_configs, load_space  # noqa: E402


def oracle_result(configs, table, max_latency, max_memory):
    """Best feasible accuracy over the whole pool (the exhaustive oracle)."""
    best = None
    for config in configs:
        measurement = table.get(config.config_id)
        if measurement is None:
            continue
        if not is_feasible(measurement, max_latency, max_memory):
            continue
        if best is None or measurement.accuracy > best:
            best = measurement.accuracy
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "benchmarks" / "synthetic_demo.csv"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--max-latency", type=float, default=None,
                        help="latency cap in ms")
    parser.add_argument("--max-memory", type=float, default=None,
                        help="memory cap in MB")
    parser.add_argument("--ablation", default="full",
                        choices=["full", "no_resource", "no_capacity"],
                        help="ablation mode for the heuristic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="restrict to these precisions, e.g. --precisions fp32")
    args = parser.parse_args(argv)

    space = load_space()
    all_configs = enumerate_configs(space, precisions=args.precisions)
    known = [c.config_id for c in all_configs]
    table = BenchmarkTable(args.table, known_config_ids=known)

    configs, missing = split_by_coverage(all_configs, table)
    if not configs:
        raise SystemExit(
            f"{table.path.name} contains none of the {len(all_configs)} selected configurations"
        )
    if missing:
        print(f"NOTE: {len(missing)} of {len(all_configs)} configurations have no "
              f"measurement and are excluded (e.g. {missing[0].config_id}); "
              f"searching the {len(configs)} covered configurations.\n")

    if table.is_synthetic():
        print("WARNING: table is synthetic; results below test the pipeline only.\n")

    oracle = oracle_result(configs, table, args.max_latency, args.max_memory)
    print("=" * 88)
    print(f"table    : {table.path.name} ({len(table)} rows)")
    print(f"pool     : {len(configs)} configurations")
    print(f"caps     : latency<={args.max_latency} ms, memory<={args.max_memory} MB")
    print(f"oracle   : best feasible accuracy = "
          f"{'n/a' if oracle is None else f'{oracle:.4f}'}")
    print("=" * 88)

    header = (f"{'algorithm':<12}{'budget':>8}{'evaluated':>11}"
              f"{'best acc':>11}{'regret':>10}{'feasible':>10}")
    print(header)
    print("-" * len(header))

    rows: List[str] = []
    for budget in args.budgets:
        for builder in ("random", "greedy", "heuristic"):
            evaluator = Evaluator(table, budget=budget)
            scorer = Scorer(configs, weights=ScoreWeights(ablation=args.ablation),
                            max_latency_ms=args.max_latency,
                            max_memory_mb=args.max_memory)

            if builder == "random":
                algorithm = RandomSearch(seed=args.seed,
                                         max_latency_ms=args.max_latency,
                                         max_memory_mb=args.max_memory)
            elif builder == "greedy":
                algorithm = GreedyLocalSearch(seed=args.seed,
                                              max_latency_ms=args.max_latency,
                                              max_memory_mb=args.max_memory)
            else:
                algorithm = HeuristicBestFirstSearch(scorer=scorer,
                                                     max_latency_ms=args.max_latency,
                                                     max_memory_mb=args.max_memory)

            result = algorithm.run(evaluator, configs)
            best = result.best_accuracy
            regret = None if (oracle is None or best is None) else oracle - best

            print(f"{result.algorithm:<12}{budget:>8}{result.n_evaluated:>11}"
                  f"{'n/a' if best is None else f'{best:.4f}':>11}"
                  f"{'n/a' if regret is None else f'{regret:.4f}':>10}"
                  f"{result.feasible_count:>10}")

    # the oracle row, for reference
    print(f"{'exhaustive':<12}{'all':>8}{len(configs):>11}"
          f"{'n/a' if oracle is None else f'{oracle:.4f}':>11}{'0.0000':>10}"
          f"{sum(1 for c in configs if (m := table.get(c.config_id)) is not None and is_feasible(m, args.max_latency, args.max_memory)):>10}")

    print()
    print("regret = oracle accuracy - accuracy found by the algorithm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
