"""One-command reproduction of every result in the repository.

Runs the whole pipeline in order and writes a summary so that a reviewer can see
what was executed, how long it took, and whether anything failed:

    self-test -> capacity check -> score check -> evaluator contract
              -> search comparison -> tables and figures

Usage
-----
    python scripts/reproduce.py
    python scripts/reproduce.py --table benchmarks/pi4b_2gb_reference.csv
    python scripts/reproduce.py --quick
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def run_step(name: str, command: List[str], cwd: Path) -> Tuple[str, bool, float, str]:
    """Run one pipeline step, returning (name, ok, seconds, tail-of-output)."""
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(cwd), capture_output=True, text=True, errors="replace"
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    tail = "\n".join(line for line in output.strip().splitlines()[-3:])
    return (name, completed.returncode == 0, elapsed, tail)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "benchmarks" / "synthetic_demo.csv"),
                        help="benchmark table used for the search and figures")
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    parser.add_argument("--quick", action="store_true",
                        help="use fewer Monte-Carlo samples (faster)")
    parser.add_argument("--max-latency", type=float, default=None)
    parser.add_argument("--max-memory", type=float, default=None)
    args = parser.parse_args(argv)

    python = sys.executable
    table = str(Path(args.table))
    samples = "25" if args.quick else "150"

    constraints = []
    if args.max_latency is not None:
        constraints += ["--max-latency", str(args.max_latency)]
    if args.max_memory is not None:
        constraints += ["--max-memory", str(args.max_memory)]

    steps = [
        ("search space self-test", [python, "src/search_space.py", "--self-test"]),
        ("capacity numerical check", [python, "scripts/check_capacity.py",
                                      "--samples", samples]),
        ("score monotonicity checks", [python, "scripts/check_score.py"] + constraints),
        ("evaluator contract checks", [python, "scripts/try_evaluator.py",
                                       "--table", table]),
        ("search comparison", [python, "scripts/run_search.py", "--table", table]
         + constraints),
        ("tables and figures", [python, "scripts/make_figures.py", "--table", table,
                                "--out", args.out] + constraints),
    ]

    print("=" * 78)
    print("Reproducing the CS5491 edge-vision configuration search project")
    print("=" * 78)
    print(f"started   : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"python    : {platform.python_version()} ({platform.system()})")
    print(f"table     : {Path(table).name}")
    print(f"results   : {args.out}")
    print()

    results = []
    for index, (name, command) in enumerate(steps, 1):
        print(f"[{index}/{len(steps)}] {name} ...", flush=True)
        outcome = run_step(name, command, REPO_ROOT)
        _, ok, elapsed, tail = outcome
        results.append(outcome)
        status = "PASS" if ok else "FAIL"
        print(f"        {status} in {elapsed:6.1f}s")
        if not ok and tail:
            print("        " + tail.replace("\n", "\n        "))
        if not ok:
            break

    total = sum(r[2] for r in results)
    failures = [r for r in results if not r[1]]

    # -- write the summary ------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "reproduce_summary.txt"

    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from benchmark_table import BenchmarkTable  # noqa: WPS433
        table_info = BenchmarkTable(table).source_counts()
        synthetic = BenchmarkTable(table).is_synthetic()
    except Exception:  # pragma: no cover - summary must never crash the run
        table_info, synthetic = {}, False

    lines = []
    lines.append("Reproduction summary")
    lines.append("====================")
    lines.append(f"finished  : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"python    : {platform.python_version()} ({platform.system()})")
    lines.append(f"table     : {Path(table).name}  sources={table_info}")
    lines.append(f"total time: {total:.1f}s")
    lines.append("")
    lines.append(f"{'step':<32}{'status':>8}{'seconds':>10}")
    for name, ok, elapsed, _ in results:
        lines.append(f"{name:<32}{'PASS' if ok else 'FAIL':>8}{elapsed:>10.1f}")
    lines.append("")
    if synthetic:
        lines.append("WARNING: the benchmark table is SYNTHETIC. Every number produced by")
        lines.append("         this run verifies the pipeline and must not be reported as a")
        lines.append("         measurement. Re-run with the Raspberry Pi table for results.")
    lines.append(f"overall   : {'PASS' if not failures else 'FAIL'}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 78)
    print(f"overall: {'PASS' if not failures else 'FAIL'}  "
          f"({len(results) - len(failures)}/{len(steps)} steps, {total:.1f}s)")
    print(f"summary written to {summary_path}")
    if synthetic:
        print("WARNING: synthetic table - results verify the pipeline only.")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
