"""Evaluation metrics for ranking and fixed-budget selection.

The course topic evaluates a cheap score by (i) how well it *ranks* trained
architectures and (ii) how well it *selects* under a budget. The same two
questions are asked here, so this module provides:

* ``spearman`` and ``kendall_tau_b`` — rank correlation with tie handling;
* ``regret`` — the gap to the best feasible configuration;
* ``pareto_front`` — non-dominated accuracy/latency (and accuracy/memory) points.

Everything is implemented with NumPy only: the core experiment must run on a
machine that has no SciPy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rankdata(values: Sequence[float]) -> np.ndarray:
    """Ranks with average ranks for ties (1-based, like SciPy's ``rankdata``)."""
    array = np.asarray(values, dtype=float)
    n = array.size
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    sorted_values = array[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            average = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rank correlation (``rho``). ``nan`` when undefined."""
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denominator == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denominator)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b, correcting for ties in either variable."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    n = xa.size
    concordant = discordant = ties_x = ties_y = 0

    for i in range(n - 1):
        dx = xa[i] - xa[i + 1:]
        dy = ya[i] - ya[i + 1:]
        both_tied = (dx == 0) & (dy == 0)
        only_x = (dx == 0) & ~both_tied
        only_y = (dy == 0) & ~both_tied
        productive = ~(dx == 0) & ~(dy == 0)

        concordant += int(np.sum(productive & (dx * dy > 0)))
        discordant += int(np.sum(productive & (dx * dy < 0)))
        ties_x += int(np.sum(both_tied | only_x))
        ties_y += int(np.sum(both_tied | only_y))

    n0 = n * (n - 1) / 2.0
    denominator = np.sqrt((n0 - ties_x) * (n0 - ties_y))
    if denominator == 0.0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def tie_counts(values: Sequence[float]) -> Dict[str, object]:
    """How many distinct values, and how large the largest tied block is."""
    array = np.asarray(values, dtype=float)
    _, counts = np.unique(array, return_counts=True)
    return {
        "n": int(array.size),
        "distinct": int(counts.size),
        "max_tie_block": int(counts.max()) if counts.size else 0,
        "tied_values": int(np.sum(counts > 1)),
    }


# ---------------------------------------------------------------------------
# Selection quality
# ---------------------------------------------------------------------------
def regret(oracle_best: Optional[float], found_best: Optional[float]) -> Optional[float]:
    """Absolute regret: how much accuracy the selection left on the table."""
    if oracle_best is None or found_best is None:
        return None
    return float(oracle_best - found_best)


def normalized_regret(oracle_best: Optional[float], found_best: Optional[float]) -> Optional[float]:
    """Regret divided by the oracle accuracy, so budgets are comparable."""
    if oracle_best is None or found_best is None or oracle_best == 0:
        return None
    return float((oracle_best - found_best) / oracle_best)


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------
def pareto_front(pairs: Sequence[Tuple[float, float]]) -> List[int]:
    """Indices of non-dominated points, maximising the first and minimising the second.

    ``pairs`` are ``(objective_to_maximize, objective_to_minimize)``.
    """
    points = [(float(a), float(b)) for a, b in pairs]
    keep: List[int] = []
    for i, (ai, bi) in enumerate(points):
        dominated = False
        for j, (aj, bj) in enumerate(points):
            if i == j:
                continue
            if aj >= ai and bj <= bi and (aj > ai or bj < bi):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def pareto_points(pairs: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """The non-dominated points themselves, sorted by the maximised objective."""
    indices = pareto_front(pairs)
    return sorted((tuple(pairs[i]) for i in indices), key=lambda p: p[0])


# ---------------------------------------------------------------------------
# Size-controlled comparison (Topic C asks for one)
# ---------------------------------------------------------------------------
def within_tolerance_groups(records: Sequence[Dict[str, object]],
                            key: str, tolerance: float) -> List[List[int]]:
    """Group records whose ``key`` differs by at most ``tolerance`` (relative).

    A coarse stand-in for the topic's "same width / parameters within 10%"
    control: it lets the report say how many comparisons survive the control
    instead of silently comparing a 2M model with a 21M model.
    """
    order = sorted(range(len(records)), key=lambda i: float(records[i][key]))
    groups: List[List[int]] = []
    current: List[int] = []

    def value(index: int) -> float:
        return float(records[index][key])

    for index in order:
        if not current:
            current = [index]
            continue
        reference = value(current[0])
        if reference == 0:
            continues = value(index) == 0
        else:
            continues = (value(index) - reference) / abs(reference) <= tolerance
        if continues:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    if current:
        groups.append(current)
    return groups
