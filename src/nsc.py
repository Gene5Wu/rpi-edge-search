"""Neural Spectral Capacity (NSC) as an analytic capacity score.

For a weight matrix ``W`` with singular values ``sigma_i`` the reference
capacity is

    psi(W) = ln det(I + W^T W) = sum_i ln(1 + sigma_i^2).

For a matrix of shape ``(m, n)`` whose entries are i.i.d. ``N(0, s^2)``, this
quantity concentrates, and the Marchenko--Pastur (MP) law gives its expectation
from the dimensions and the initialisation scale ``s`` alone. That is what makes
NSC a *cheap, specification-only* score: no training, no forward pass.

The MP distribution for ratio ``gamma = m / n`` has support
``[(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2]`` and density

    rho(x) = sqrt((b - x)(x - a)) / (2 pi gamma x).

Because ``W = s Z`` and the eigenvalues of ``Z^T Z`` scale with ``n``, the
singular-value squares are ``s^2 n lambda`` with ``lambda ~ MP(gamma)``, so

    E[psi(W)] = min(m, n) * integral ln(1 + s^2 * max(m, n) * lambda) dMP_gamma.

This module implements that integral numerically and exposes a ``capacity``
function that sums it over a list of convolution weight matrices.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

#: Default initialisation standard deviation used by the reference convention.
DEFAULT_INIT_STD = 0.02

#: Number of quadrature points for the MP integral.
DEFAULT_GRID_POINTS = 4096


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """`np.trapezoid` on new NumPy, `np.trapz` on old NumPy."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def mp_support(ratio: float) -> Tuple[float, float]:
    """Support ``[a, b]`` of the Marchenko--Pastur law with the given ratio."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"MP ratio must be in (0, 1], got {ratio!r}")
    sqrt_ratio = np.sqrt(ratio)
    lower = (1.0 - sqrt_ratio) ** 2
    upper = (1.0 + sqrt_ratio) ** 2
    if upper - lower <= 0:
        raise ValueError(f"degenerate MP support for ratio {ratio!r}")
    return float(lower), float(upper)


def psi_mp(m: int, n: int, s: float = DEFAULT_INIT_STD,
           num_points: int = DEFAULT_GRID_POINTS) -> float:
    """Analytic estimate of ``E[ln det(I + W^T W)]`` for ``W`` of shape ``(m, n)``.

    ``s`` is the standard deviation of the initialisation (``N(0, s^2)``).

    The quadrature uses the substitution ``x = a + (b - a) t^2`` on ``t in
    [0, 1]``. This removes the inverse-square-root singularity of the MP density
    at the lower edge, which matters when the matrix is (nearly) square and
    ``a = 0``; a plain uniform grid in ``x`` silently underestimates the
    integral there.
    """
    if m <= 0 or n <= 0:
        return 0.0
    if s <= 0.0:
        return 0.0

    rows, cols = int(m), int(n)
    rank = min(rows, cols)
    span = max(rows, cols)
    ratio = rank / span

    lower, upper = mp_support(ratio)
    width = upper - lower

    t = np.linspace(1e-9, 1.0 - 1e-9, num_points)
    x = lower + width * t ** 2
    jacobian = 2.0 * width * t

    density = np.sqrt(np.maximum((upper - x) * (x - lower), 0.0)) / (2.0 * np.pi * ratio * x)

    weight = density * jacobian
    norm = _trapezoid(weight, t)
    if norm <= 0.0:
        raise ValueError(f"empty MP density for shape ({rows}, {cols})")

    integrand = np.log1p((s * s) * span * x) * weight
    return rank * _trapezoid(integrand, t) / norm


def exact_psi(weight: np.ndarray) -> float:
    """Exact ``ln det(I + W^T W)`` from the singular values (for validation)."""
    singular = np.linalg.svd(weight, compute_uv=False)
    return float(np.sum(np.log1p(singular ** 2)))


def capacity(matrices: Iterable, s: float = DEFAULT_INIT_STD) -> float:
    """Sum the analytic capacity over a sequence of convolution weight matrices.

    Each element must expose ``m`` and ``n`` attributes (see
    :class:`src.model_spec.ConvMatrix`).
    """
    return float(sum(psi_mp(matrix.m, matrix.n, s) for matrix in matrices))


def capacity_by_layer(matrices: Sequence, s: float = DEFAULT_INIT_STD) -> list:
    """Per-matrix capacity contributions, for inspection/ablation."""
    return [(matrix.name, psi_mp(matrix.m, matrix.n, s)) for matrix in matrices]
