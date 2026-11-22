"""Exercise the evaluator contract.

Checks that the black box behaves the way the experiment requires:

* a measured configuration returns its row;
* an unmeasured configuration raises ``NotMeasuredError`` (no peeking);
* the budget counts *distinct* evaluations, and re-querying a known
  configuration is free;
* exceeding the budget raises ``BudgetExhausted``.

Usage
-----
    python scripts/try_evaluator.py
    python scripts/try_evaluator.py --table benchmarks/synthetic_demo.csv --budget 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluator import BudgetExhausted, Evaluator, NotMeasuredError  # noqa: E402
from benchmark_table import BenchmarkTable, split_by_coverage  # noqa: E402
from search_space import enumerate_configs, load_space  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "benchmarks" / "synthetic_demo.csv"))
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="restrict to these precisions, e.g. --precisions fp32")
    args = parser.parse_args(argv)

    space = load_space()
    all_configs = enumerate_configs(space, precisions=args.precisions)
    known = [c.config_id for c in all_configs]

    table = BenchmarkTable(args.table, known_config_ids=known)
    evaluator = Evaluator(table, budget=args.budget)

    configs, missing = split_by_coverage(all_configs, table)
    if not configs:
        raise SystemExit(
            f"{table.path.name} contains none of the {len(all_configs)} selected configurations"
        )
    if missing:
        print(f"NOTE: {len(missing)} of {len(all_configs)} configurations have no "
              f"measurement and are excluded (e.g. {missing[0].config_id})")

    print("=" * 78)
    print(f"table      : {table.path.name}  ({len(table)} rows, sources={table.source_counts()})")
    print(f"budget     : {args.budget}")
    print("=" * 78)

    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")

    # 1. normal evaluation
    first = evaluator.evaluate(configs[0])
    print(f"evaluate({configs[0].config_id}) -> "
          f"acc={first.accuracy:.4f}, lat={first.latency_median_ms:.1f} ms")
    check("measured configuration returns a row", first.accuracy > 0)

    # 2. budget accounting
    for config in configs[1:args.budget]:
        evaluator.evaluate(config)
    check(f"budget spent after {args.budget} distinct queries",
          evaluator.n_queries == args.budget and evaluator.remaining == 0)

    # 3. re-query is free
    before = evaluator.n_queries
    evaluator.evaluate(configs[0])
    check("re-querying a known configuration is free", evaluator.n_queries == before)

    # 4. budget exhaustion
    exhausted = False
    try:
        evaluator.evaluate(configs[args.budget])
    except BudgetExhausted:
        exhausted = True
    check("exceeding the budget raises BudgetExhausted", exhausted)

    # 5. no peeking
    unmeasured = False
    try:
        evaluator.evaluate("yolov5x_r1024_fp32_t8")
    except NotMeasuredError:
        unmeasured = True
    check("unmeasured configuration raises NotMeasuredError", unmeasured)

    # 6. query log length matches distinct evaluations
    check("query log matches distinct evaluations",
          len(evaluator.query_log()) == evaluator.n_queries)

    failed = sum(1 for _, ok in checks if not ok)
    print()
    print(f"Checks failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
