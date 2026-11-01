"""Validate the analytic capacity score and report per-variant values.

Two checks are performed:

1. **Numerical check (Monte Carlo).** For a few matrix shapes, draw random
   ``N(0, s^2)`` weights, compute the exact ``ln det(I + W^T W)`` from their
   singular values, and compare the sample mean with the Marchenko--Pastur
   estimate :func:`src.nsc.psi_mp`. The two should agree within a few percent.

2. **Variant table.** Report the parameter count and the summed analytic
   capacity for YOLOv5n / s / m, which is the architecture term used by the
   deployment score.

Usage
-----
    python scripts/check_capacity.py
    python scripts/check_capacity.py --samples 400 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_spec import VARIANTS, build_variant  # noqa: E402
from nsc import DEFAULT_INIT_STD, capacity, exact_psi, psi_mp  # noqa: E402


def monte_carlo_check(shape, s, samples, rng, tolerance=0.10):
    """Compare the MP estimate with the exact average over random weights."""
    m, n = shape
    values = [exact_psi(rng.normal(0.0, s, size=(m, n))) for _ in range(samples)]
    empirical = float(np.mean(values))
    analytic = psi_mp(m, n, s)
    rel_error = abs(analytic - empirical) / max(abs(empirical), 1e-12)
    status = "OK  " if rel_error <= tolerance else "FAIL"
    return status, m, n, empirical, analytic, rel_error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=200,
                        help="Monte Carlo samples per shape")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", type=float, default=DEFAULT_INIT_STD)
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="relative tolerance for the numerical check")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    shapes = [(64, 576), (128, 1152), (256, 2304), (32, 32), (512, 256)]

    print("=" * 78)
    print(f"Numerical check: psi_MP vs exact Monte-Carlo estimate (s = {args.init_std})")
    print("=" * 78)
    print(f"{'shape (m x n)':>18} {'MC mean':>12} {'psi_MP':>12} {'rel. error':>11}  status")

    failures = 0
    for shape in shapes:
        status, m, n, empirical, analytic, rel_error = monte_carlo_check(
            shape, args.init_std, args.samples, rng, args.tolerance
        )
        failures += status.startswith("FAIL")
        print(f"{f'{m} x {n}':>18} {empirical:>12.4f} {analytic:>12.4f} "
              f"{rel_error:>10.2%}  {status}")

    print()
    print("=" * 78)
    print("Architecture capacity by variant")
    print("=" * 78)
    header = f"{'variant':<10}{'conv matrices':>15}{'params (M)':>13}{'ref. (M)':>11}{'capacity':>14}"
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        convs, params = build_variant(variant)
        value = capacity(convs, args.init_std)
        print(f"{variant:<10}{len(convs):>15}{params / 1e6:>13.4f}"
              f"{VARIANTS[variant]['params_m']:>11.1f}{value:>14.4f}")

    print()
    print(f"Numerical checks failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
