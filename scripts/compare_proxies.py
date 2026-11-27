"""Compare the accuracy proxies head to head.

The project's score needs an ingredient that stands in for accuracy. Three
candidates are available, in increasing order of machinery:

* ``params`` -- parameter count. Nearly free, but blind to resolution.
* ``macs``   -- predicted compute at the actual resolution. Also free, and it
  already captures the resolution effect that dominates accuracy.
* ``nsc``    -- the NSC capacity term scaled by resolution and precision. This is
  the project's proposal, and it costs real implementation effort.

The scientific question is not "does the heuristic beat random search" -- it is
**"does the NSC term earn its place over a compute proxy that costs nothing?"**
This script answers it on measured data, per constraint scenario, at several
budgets, and also reports the rank correlation of each proxy against measured
accuracy. If ``macs`` matches or beats ``nsc``, the report says so.

Usage
-----
    python scripts/compare_proxies.py --table benchmarks/laptop_fp32.csv --precisions fp32
    python scripts/compare_proxies.py --seeds 10 --budgets 4 8
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
from metrics import spearman  # noqa: E402
from score import ScoreWeights, Scorer  # noqa: E402
from search import GreedyLocalSearch, HeuristicBestFirstSearch, RandomSearch  # noqa: E402
from search_space import enumerate_configs, load_space  # noqa: E402

PROXIES = ("nsc", "macs", "params")


def feasible_oracle(configs, table, max_latency, max_memory) -> Optional[float]:
    best = None
    for config in configs:
        measurement = table.get(config.config_id)
        if measurement is None:
            continue
        if not is_feasible(measurement, max_latency, max_memory):
            continue
        best = measurement.accuracy if best is None else max(best, measurement.accuracy)
    return best


def run_seeds(builder, seeds: int, table, configs, budget, max_latency, max_memory) -> List[float]:
    """Best feasible accuracy over ``seeds`` runs of a stochastic algorithm."""
    results: List[float] = []
    for seed in range(seeds):
        evaluator = Evaluator(table, budget=budget)
        algorithm = builder(seed, max_latency, max_memory)
        outcome = algorithm.run(evaluator, configs)
        if outcome.best_accuracy is not None:
            results.append(outcome.best_accuracy)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "benchmarks" / "laptop_fp32.csv"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--precisions", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    space = load_space()
    all_configs = enumerate_configs(space, precisions=args.precisions)
    table = BenchmarkTable(args.table, known_config_ids=[c.config_id for c in all_configs])
    configs, missing = split_by_coverage(all_configs, table)
    if not configs:
        raise SystemExit(f"{table.path.name} covers none of the selected configurations")
    if missing:
        print(f"NOTE: {len(missing)} of {len(all_configs)} configurations are unmeasured "
              f"and excluded; analysing {len(configs)}.")

    measured = [table.get(c.config_id) for c in configs]
    accuracy = [m.accuracy for m in measured]

    scenarios = [
        ("unconstrained", None, None),
        ("latency 25ms", 25.0, None),
        ("latency 25ms + memory 140MB", 25.0, 140.0),
    ]

    rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []

    print("=" * 104)
    print(f"table {table.path.name} | pool {len(configs)} | seeds {args.seeds}")
    print("=" * 104)

    for label, max_latency, max_memory in scenarios:
        feasible = [is_feasible(m, max_latency, max_memory) for m in measured]
        n_feasible = sum(feasible)
        oracle = feasible_oracle(configs, table, max_latency, max_memory)
        oracle_text = "n/a" if oracle is None else f"{oracle:.4f}"

        print()
        print(f"--- {label}  |  feasible {n_feasible}/{len(configs)}  |  oracle {oracle_text} ---")

        # rank correlation of each proxy against measured accuracy
        for proxy in PROXIES:
            scorer = Scorer(configs, weights=ScoreWeights(proxy=proxy),
                            max_latency_ms=max_latency, max_memory_mb=max_memory)
            raw = {p: [scorer._accuracy_proxy(c) for c in configs] for p in PROXIES}
            values = raw[proxy]
            feasible_values = [v for v, keep in zip(values, feasible) if keep]
            feasible_accuracy = [a for a, keep in zip(accuracy, feasible) if keep]
            rho_all = spearman(values, accuracy)
            rho_feasible = (spearman(feasible_values, feasible_accuracy)
                            if len(feasible_values) >= 3 else float("nan"))
            rank_rows.append({"scenario": label, "proxy": proxy,
                              "spearman_all": round(rho_all, 4),
                              "spearman_feasible": round(rho_feasible, 4)})
        rho_text = ", ".join(
            f"{r['proxy']}={r['spearman_all']:.3f}"
            for r in rank_rows if r["scenario"] == label
        )
        print(f"    rank correlation vs measured accuracy: {rho_text}")

        header = (f"{'budget':>7}{'random':>10}{'greedy':>10}"
                  f"{'heur[nsc]':>12}{'heur[macs]':>12}{'heur[params]':>14}")
        print(header)
        print("-" * len(header))

        for budget in args.budgets:
            random_values = run_seeds(
                lambda s, lat, mem: RandomSearch(seed=s, max_latency_ms=lat, max_memory_mb=mem),
                args.seeds, table, configs, budget, max_latency, max_memory)
            greedy_values = run_seeds(
                lambda s, lat, mem: GreedyLocalSearch(seed=s, max_latency_ms=lat,
                                                      max_memory_mb=mem),
                args.seeds, table, configs, budget, max_latency, max_memory)

            random_mean = statistics.fmean(random_values) if random_values else float("nan")
            greedy_mean = statistics.fmean(greedy_values) if greedy_values else float("nan")

            heuristic_values: Dict[str, float] = {}
            for proxy in PROXIES:
                evaluator = Evaluator(table, budget=budget)
                scorer = Scorer(configs, weights=ScoreWeights(proxy=proxy),
                                max_latency_ms=max_latency, max_memory_mb=max_memory)
                outcome = HeuristicBestFirstSearch(
                    scorer=scorer, max_latency_ms=max_latency,
                    max_memory_mb=max_memory).run(evaluator, configs)
                value = outcome.best_accuracy if outcome.best_accuracy is not None else float("nan")
                heuristic_values[proxy] = value

            print(f"{budget:>7}{random_mean:>10.4f}{greedy_mean:>10.4f}"
                  f"{heuristic_values['nsc']:>12.4f}{heuristic_values['macs']:>12.4f}"
                  f"{heuristic_values['params']:>14.4f}")

            row = {
                "scenario": label, "budget": budget,
                "n_feasible": n_feasible, "oracle": oracle,
                "random_mean": round(random_mean, 4),
                "random_std": round(statistics.pstdev(random_values), 4) if len(random_values) > 1 else 0.0,
                "greedy_mean": round(greedy_mean, 4),
                "greedy_std": round(statistics.pstdev(greedy_values), 4) if len(greedy_values) > 1 else 0.0,
            }
            for proxy in PROXIES:
                row[f"heuristic_{proxy}"] = round(heuristic_values[proxy], 4)
                row[f"regret_{proxy}"] = (
                    round(oracle - heuristic_values[proxy], 4) if oracle is not None else None
                )
            rows.append(row)

    out = Path(args.out) if args.out else REPO_ROOT / "results" / "proxy_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rank_rows).to_csv(out.with_name("proxy_rank_correlation.csv"), index=False)

    print()
    print("Reading the table: if heur[macs] matches heur[nsc], the NSC term does not")
    print("earn its place on this benchmark and the report must say so.")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
