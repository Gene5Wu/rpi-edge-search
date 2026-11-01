"""Search algorithms over the deployment configuration space.

The project is literally a search problem:

    state           = a deployment configuration
    successors      = configurations that differ in exactly one factor
    cost            = one evaluation from the budget
    goal            = a feasible configuration with high accuracy

Four algorithms are compared under an identical budget:

``random``      uniform sampling — the standard black-box baseline.
``greedy``      local search: evaluate the neighbours, move to the best observed.
``heuristic``   best-first search ordered by the analytic score (this project's
                method, and the direct analogue of the Topic C protocol).
``exhaustive``  evaluate everything; the oracle used to compute regret.

Only ``exhaustive`` is allowed to ignore the budget. Every algorithm asks the
:class:`~src.evaluator.Evaluator` for outcomes, so a configuration is never
charged twice and an unmeasured configuration can never be read.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark_table import Measurement
from evaluator import BudgetExhausted, Evaluator, NotMeasuredError, is_feasible
from score import Scorer

#: Factor names, in the order used by the canonical configuration id.
FACTORS: Tuple[str, ...] = ("model", "resolution", "precision", "threads")


@dataclass
class SearchResult:
    """Outcome of one algorithm run."""

    algorithm: str
    evaluated: List[str] = field(default_factory=list)
    best_config_id: Optional[str] = None
    best_accuracy: Optional[float] = None
    feasible_count: int = 0
    budget: Optional[int] = None

    @property
    def n_evaluated(self) -> int:
        return len(self.evaluated)

    def to_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "n_evaluated": self.n_evaluated,
            "best_config_id": self.best_config_id,
            "best_accuracy": self.best_accuracy,
            "feasible_count": self.feasible_count,
            "budget": self.budget,
        }


def build_neighbor_map(configs: Sequence) -> Dict[str, List[str]]:
    """Adjacency between configurations: differ in exactly one factor, one step.

    The pool is a full Cartesian product, so the ordered factor values can be
    recovered from the configurations themselves; a neighbour is the previous or
    next value of a single factor.
    """
    values: Dict[str, List[object]] = {factor: [] for factor in FACTORS}
    for config in configs:
        for factor in FACTORS:
            value = getattr(config, factor)
            if value not in values[factor]:
                values[factor].append(value)

    index = {factor: {v: i for i, v in enumerate(vals)} for factor, vals in values.items()}
    by_tuple = {
        tuple(getattr(c, f) for f in FACTORS): c.config_id for c in configs
    }

    neighbors: Dict[str, List[str]] = {}
    for config in configs:
        own = tuple(getattr(config, f) for f in FACTORS)
        found: List[str] = []
        for pos, factor in enumerate(FACTORS):
            values_factor = values[factor]
            i = index[factor][own[pos]]
            for j in (i - 1, i + 1):
                if 0 <= j < len(values_factor):
                    altered = list(own)
                    altered[pos] = values_factor[j]
                    candidate = by_tuple.get(tuple(altered))
                    if candidate is not None:
                        found.append(candidate)
        neighbors[config.config_id] = found
    return neighbors


def _bounds(configs: Sequence):
    """Ordered factor values, for documenting/validating the adjacency."""
    bounds: Dict[str, List[object]] = {factor: [] for factor in FACTORS}
    for config in configs:
        for factor in FACTORS:
            value = getattr(config, factor)
            if value not in bounds[factor]:
                bounds[factor].append(value)
    return bounds


class SearchAlgorithm:
    """Base class: subclasses implement :meth:`run`."""

    name = "base"

    def __init__(self, max_latency_ms: Optional[float] = None,
                 max_memory_mb: Optional[float] = None):
        self.max_latency_ms = max_latency_ms
        self.max_memory_mb = max_memory_mb

    # -- helpers -----------------------------------------------------------
    def _feasible(self, measurement: Measurement) -> bool:
        return is_feasible(measurement, self.max_latency_ms, self.max_memory_mb)

    def _evaluate(self, evaluator: Evaluator, config) -> Optional[Measurement]:
        """Evaluate unless the budget is spent or the row is missing."""
        if not evaluator.can_evaluate(config):
            return None
        try:
            return evaluator.evaluate(config)
        except (BudgetExhausted, NotMeasuredError):
            return None

    def _finish(self, algorithm: str, evaluator: Evaluator,
                configs_by_id: Dict[str, object]) -> SearchResult:
        measurements = evaluator.evaluated_measurements()
        feasible = [(m, configs_by_id[m.config_id]) for m in measurements
                    if self._feasible(m)]
        if feasible:
            best_measurement, _ = max(feasible, key=lambda pair: pair[0].accuracy)
            best_id = best_measurement.config_id
            best_accuracy = best_measurement.accuracy
        else:
            best_id, best_accuracy = None, None
        return SearchResult(
            algorithm=algorithm,
            evaluated=evaluator.evaluated_ids(),
            best_config_id=best_id,
            best_accuracy=best_accuracy,
            feasible_count=len(feasible),
            budget=evaluator.budget,
        )

    # -- interface ---------------------------------------------------------
    def run(self, evaluator: Evaluator, configs: Sequence) -> SearchResult:  # pragma: no cover
        raise NotImplementedError


class RandomSearch(SearchAlgorithm):
    """Uniform random sampling without replacement."""

    name = "random"

    def __init__(self, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(seed)

    def run(self, evaluator: Evaluator, configs: Sequence) -> SearchResult:
        pool = list(configs)
        self.rng.shuffle(pool)
        for config in pool:
            if self._evaluate(evaluator, config) is None:
                break
        return self._finish(self.name, evaluator, {c.config_id: c for c in configs})


class GreedyLocalSearch(SearchAlgorithm):
    """Evaluate the neighbours of the current point, move to the best observed.

    The move rule uses *measured* accuracy, which is what makes this the
    conventional heuristic baseline: it follows the data, not a model.
    """

    name = "greedy"

    def __init__(self, seed_config_id: Optional[str] = None, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.seed_config_id = seed_config_id
        self.rng = random.Random(seed)

    def _seed(self, configs: Sequence) -> object:
        if self.seed_config_id is not None:
            for config in configs:
                if config.config_id == self.seed_config_id:
                    return config
            raise KeyError(f"unknown seed configuration {self.seed_config_id!r}")
        return configs[len(configs) // 2]

    def run(self, evaluator: Evaluator, configs: Sequence) -> SearchResult:
        neighbors = build_neighbor_map(configs)
        by_id = {c.config_id: c for c in configs}
        current = self._seed(configs)

        if self._evaluate(evaluator, current) is None:
            return self._finish(self.name, evaluator, by_id)

        while evaluator.remaining is None or evaluator.remaining > 0:
            candidates = [cid for cid in neighbors[current.config_id]
                          if cid not in evaluator.evaluated_ids()]
            if not candidates:
                # local optimum reached: restart from a random unvisited point
                unvisited = [c for c in configs if c.config_id not in evaluator.evaluated_ids()]
                if not unvisited:
                    break
                current = self.rng.choice(unvisited)
                if self._evaluate(evaluator, current) is None:
                    break
                continue

            for cid in candidates:
                if self._evaluate(evaluator, by_id[cid]) is None:
                    break

            evaluated_here = [by_id[cid] for cid in candidates
                              if cid in evaluator.evaluated_ids()]
            if not evaluated_here:
                break
            best = max(evaluated_here, key=lambda c: evaluator.evaluate(c).accuracy)
            current_measurement = evaluator.evaluate(current)
            if (self._feasible(evaluator.evaluate(best))
                    and evaluator.evaluate(best).accuracy > current_measurement.accuracy):
                current = best
            else:
                # no improvement: jump somewhere new to avoid stalling
                unvisited = [c for c in configs if c.config_id not in evaluator.evaluated_ids()]
                if not unvisited:
                    break
                current = self.rng.choice(unvisited)
                if self._evaluate(evaluator, current) is None:
                    break

        return self._finish(self.name, evaluator, by_id)


class HeuristicBestFirstSearch(SearchAlgorithm):
    """Best-first search ordered by the analytic score.

    The frontier starts at the highest-scoring configuration and grows with the
    neighbours of everything evaluated so far. At each step the frontier node
    with the best score is evaluated next. The score is the *only* signal used to
    choose: measurements on other configurations are never inspected, which is
    the search analogue of an informed heuristic that must not see the future.
    """

    name = "heuristic"

    def __init__(self, scorer: Scorer, seed_config_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.scorer = scorer
        self.seed_config_id = seed_config_id

    def _seed(self, configs: Sequence) -> object:
        if self.seed_config_id is not None:
            for config in configs:
                if config.config_id == self.seed_config_id:
                    return config
            raise KeyError(f"unknown seed configuration {self.seed_config_id!r}")
        return max(configs, key=self.scorer.score)

    def run(self, evaluator: Evaluator, configs: Sequence) -> SearchResult:
        neighbors = build_neighbor_map(configs)
        by_id = {c.config_id: c for c in configs}

        start = self._seed(configs)
        frontier: List[object] = [start]
        seen_frontier = {start.config_id}

        while frontier:
            # highest analytic score first; tie-break by config id for determinism
            frontier.sort(key=lambda c: (-self.scorer.score(c), c.config_id))
            current = frontier.pop(0)
            if current.config_id in evaluator.evaluated_ids():
                continue
            if self._evaluate(evaluator, current) is None:
                break
            for cid in neighbors[current.config_id]:
                if cid not in seen_frontier and cid not in evaluator.evaluated_ids():
                    seen_frontier.add(cid)
                    frontier.append(by_id[cid])

        return self._finish(self.name, evaluator, by_id)


class ExhaustiveSearch(SearchAlgorithm):
    """Evaluate the whole pool. The oracle used to compute regret."""

    name = "exhaustive"

    def run(self, evaluator: Evaluator, configs: Sequence) -> SearchResult:
        for config in configs:
            if self._evaluate(evaluator, config) is None:
                break
        return self._finish(self.name, evaluator, {c.config_id: c for c in configs})
