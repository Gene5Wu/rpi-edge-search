"""Produce the tables and figures the report needs.

Outputs (into ``results/`` by default):

    rank_correlation.csv        Spearman rho / Kendall tau-b of every predictor
    pareto_accuracy_latency.csv non-dominated accuracy/latency points
    search_regret.csv           regret vs evaluation budget for each algorithm
    fig_pareto.png              accuracy vs latency, with the Pareto front
    fig_score_vs_accuracy.png   analytic score vs measured accuracy
    fig_regret.png              regret vs budget

Usage
-----
    python scripts/make_figures.py
    python scripts/make_figures.py --table benchmarks/pi4b_2gb_reference.csv
    python scripts/make_figures.py --max-latency 400 --out results/pi_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_table import BenchmarkTable, split_by_coverage  # noqa: E402
from evaluator import Evaluator, is_feasible  # noqa: E402
from metrics import kendall_tau_b, pareto_front, regret, spearman  # noqa: E402
from score import ScoreWeights, Scorer  # noqa: E402
from search import (  # noqa: E402
    ExhaustiveSearch,
    GreedyLocalSearch,
    HeuristicBestFirstSearch,
    RandomSearch,
)
from search_space import enumerate_configs, load_space  # noqa: E402


def build_predictors(configs, scorer_full, scorer_no_resource, scorer_no_capacity,
                     resource_model) -> Dict[str, np.ndarray]:
    """Every cheap signal we can rank configurations with."""
    capacity_only = np.array([scorer_full._capacity[c.model] for c in configs])
    return {
        "score_full": np.array([scorer_full.score(c) for c in configs]),
        "score_no_resource": np.array([scorer_no_resource.score(c) for c in configs]),
        "score_no_capacity": np.array([scorer_no_capacity.score(c) for c in configs]),
        "capacity_only": capacity_only,
        "pred_latency_ms": np.array([resource_model.estimate(c)["latency_ms"] for c in configs]),
        "pred_memory_mb": np.array([resource_model.estimate(c)["memory_mb"] for c in configs]),
        "pred_macs": np.array([resource_model.estimate(c)["macs"] for c in configs]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "benchmarks" / "synthetic_demo.csv"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--max-memory", type=float, default=None)
    parser.add_argument("--max-budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="restrict to these precisions, e.g. --precisions fp32 "
                             "(the FP32-only campaign is a valid benchmark)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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
              f"measurement and are excluded (e.g. {missing[0].config_id}); "
              f"analysing the {len(configs)} covered configurations.")

    synthetic = table.is_synthetic()
    if synthetic:
        print("WARNING: synthetic table - the figures demonstrate the pipeline only.\n")
    data_kind = "simulated" if synthetic else "measured"

    from resource_model import ResourceModel
    resource_model = ResourceModel()

    scorer_full = Scorer(configs, resource_model, ScoreWeights(ablation="full"),
                         max_latency_ms=args.max_latency, max_memory_mb=args.max_memory)
    scorer_no_resource = Scorer(configs, resource_model, ScoreWeights(ablation="no_resource"),
                                max_latency_ms=args.max_latency, max_memory_mb=args.max_memory)
    scorer_no_capacity = Scorer(configs, resource_model, ScoreWeights(ablation="no_capacity"),
                                max_latency_ms=args.max_latency, max_memory_mb=args.max_memory)

    measurements = [table.get(c.config_id) for c in configs]
    accuracy = np.array([m.accuracy for m in measurements])
    latency = np.array([m.latency_median_ms for m in measurements])
    memory = np.array([m.memory_mb for m in measurements])

    feasible_mask = np.array([is_feasible(m, args.max_latency, args.max_memory)
                              for m in measurements])
    n_feasible = int(feasible_mask.sum())

    # -- 1. ranking ------------------------------------------------------
    predictors = build_predictors(configs, scorer_full, scorer_no_resource,
                                  scorer_no_capacity, resource_model)
    rows = []
    for name, values in predictors.items():
        row = {
            "predictor": name,
            "spearman_vs_accuracy": round(spearman(values, accuracy), 4),
            "kendall_vs_accuracy": round(kendall_tau_b(values, accuracy), 4),
            # The comparison that survives a hard constraint: rank only the
            # configurations that are actually deployable.
            "spearman_feasible_only": (
                round(spearman(values[feasible_mask], accuracy[feasible_mask]), 4)
                if n_feasible >= 3 else float("nan")
            ),
            "spearman_vs_latency": round(spearman(values, latency), 4),
            "spearman_vs_memory": round(spearman(values, memory), 4),
        }
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values("spearman_vs_accuracy", ascending=False)
    ranking.insert(1, "n_feasible", n_feasible)
    ranking.to_csv(out / "rank_correlation.csv", index=False)

    print(f"Rank correlation against {data_kind} accuracy")
    print("-" * 92)
    print(ranking.to_string(index=False))
    print()

    # -- 2. pareto front -------------------------------------------------
    pairs = list(zip(accuracy.tolist(), latency.tolist()))
    front = pareto_front(pairs)
    pareto_rows = [{"config_id": configs[i].config_id,
                    "accuracy": float(accuracy[i]),
                    "latency_median_ms": float(latency[i]),
                    "memory_mb": float(memory[i]),
                    "feasible": is_feasible(measurements[i], args.max_latency, args.max_memory)}
                   for i in sorted(front, key=lambda k: latency[k])]
    pd.DataFrame(pareto_rows).to_csv(out / "pareto_accuracy_latency.csv", index=False)
    print(f"Pareto front (accuracy vs latency): {len(pareto_rows)} points")
    print()

    # -- 3. regret vs budget ---------------------------------------------
    def oracle_accuracy() -> float:
        best = None
        for measurement in measurements:
            if not is_feasible(measurement, args.max_latency, args.max_memory):
                continue
            best = measurement.accuracy if best is None else max(best, measurement.accuracy)
        return float(best) if best is not None else float("nan")

    oracle = oracle_accuracy()
    budgets = list(range(1, args.max_budget + 1))
    regret_rows = []
    for budget in budgets:
        for kind in ("random", "greedy", "heuristic"):
            evaluator = Evaluator(table, budget=budget)
            if kind == "random":
                algorithm = RandomSearch(seed=args.seed, max_latency_ms=args.max_latency,
                                         max_memory_mb=args.max_memory)
            elif kind == "greedy":
                algorithm = GreedyLocalSearch(seed=args.seed, max_latency_ms=args.max_latency,
                                              max_memory_mb=args.max_memory)
            else:
                algorithm = HeuristicBestFirstSearch(scorer=scorer_full,
                                                     max_latency_ms=args.max_latency,
                                                     max_memory_mb=args.max_memory)
            result = algorithm.run(evaluator, configs)
            value = regret(oracle, result.best_accuracy)
            regret_rows.append({
                "budget": budget,
                "algorithm": kind,
                "best_accuracy": result.best_accuracy,
                "regret": value if value is not None else float("nan"),
            })
    regret_frame = pd.DataFrame(regret_rows)
    regret_frame.to_csv(out / "search_regret.csv", index=False)

    # -- 4. figures ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(latency[feasible_mask], accuracy[feasible_mask], s=26,
               label="feasible", color="#2a7ab9")
    ax.scatter(latency[~feasible_mask], accuracy[~feasible_mask], s=26,
               label="over budget", color="#c0c0c0", marker="x")
    front_sorted = sorted(((latency[i], accuracy[i]) for i in front))
    ax.plot([p[0] for p in front_sorted], [p[1] for p in front_sorted],
            color="#c0392b", linewidth=1.6, label="Pareto front")
    ax.set_xlabel(f"{data_kind} latency (ms)")
    ax.set_ylabel(f"{data_kind} mAP@0.5")
    ax.set_title(f"Accuracy vs latency ({data_kind} benchmark"
                 + (", synthetic placeholder)" if synthetic else ")"))
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_pareto.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(predictors["score_full"], accuracy, s=26, color="#2a7ab9")
    ax.set_xlabel("analytic score (full)")
    ax.set_ylabel(f"{data_kind} mAP@0.5")
    rho = spearman(predictors["score_full"], accuracy)
    ax.set_title(f"Analytic score vs {data_kind} accuracy (Spearman rho = {rho:.3f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_score_vs_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for kind, style in (("random", "o-"), ("greedy", "s-"), ("heuristic", "^-")):
        subset = regret_frame[regret_frame["algorithm"] == kind]
        ax.plot(subset["budget"], subset["regret"], style, markersize=3, label=kind)
    ax.set_xlabel("evaluation budget")
    ax.set_ylabel("regret vs oracle")
    ax.set_title(f"Regret against the exhaustive oracle ({data_kind} benchmark)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_regret.png", dpi=160)
    plt.close(fig)

    print(f"wrote tables and figures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
