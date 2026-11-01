"""Budgeted black-box evaluator.

Every configuration evaluation is treated as an expensive query: the search
algorithms may spend a limited number of them. The evaluator enforces the three
properties the experiment depends on:

1. **No peeking.** A configuration that has not been measured raises
   :class:`NotMeasuredError` — the search cannot accidentally read the answer.
2. **Honest budget accounting.** The first query for a configuration consumes one
   unit of budget; repeating it is free because the outcome is already known.
3. **Determinism.** The answer depends only on the table, never on call order.

The evaluator is the *only* place that touches ground truth; the search code
receives :class:`Measurement` objects and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from benchmark_table import BenchmarkTable, Measurement

ConfigLike = Union[str, object]


class NotMeasuredError(KeyError):
    """Raised when a configuration has no archived measurement."""


class BudgetExhausted(RuntimeError):
    """Raised when the evaluation budget has been spent."""


@dataclass(frozen=True)
class QueryRecord:
    """One entry of the query log (useful for plots and analysis)."""

    step: int
    config_id: str

    def to_dict(self) -> Dict[str, object]:
        return {"step": self.step, "config_id": self.config_id}


def _as_config_id(config: ConfigLike) -> str:
    if isinstance(config, str):
        return config
    config_id = getattr(config, "config_id", None)
    if config_id is None:
        raise TypeError(f"expected a config_id or an object with .config_id, got {config!r}")
    return str(config_id)


class Evaluator:
    """Black-box oracle backed by an archived benchmark table."""

    def __init__(self, table: BenchmarkTable, budget: Optional[int] = None):
        if budget is not None and budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget!r}")
        self.table = table
        self.budget = budget
        self._cache: Dict[str, Measurement] = {}
        self._log: List[QueryRecord] = []

    # -- budget ------------------------------------------------------------
    @property
    def n_queries(self) -> int:
        """Number of distinct configurations evaluated so far."""
        return len(self._cache)

    @property
    def remaining(self) -> Optional[int]:
        """Queries left, or ``None`` when the budget is unlimited."""
        if self.budget is None:
            return None
        return self.budget - self.n_queries

    def can_evaluate(self, config: ConfigLike) -> bool:
        """Whether ``config`` can be evaluated right now without breaking the budget."""
        config_id = _as_config_id(config)
        if config_id in self._cache:
            return True
        if not self.table.has(config_id):
            return False
        if self.budget is not None and self.n_queries >= self.budget:
            return False
        return True

    # -- querying ----------------------------------------------------------
    def evaluate(self, config: ConfigLike) -> Measurement:
        """Evaluate ``config`` and return its measurement.

        Raises
        ------
        NotMeasuredError
            The configuration has no archived row.
        BudgetExhausted
            The budget is spent and this is a *new* configuration.
        """
        config_id = _as_config_id(config)

        cached = self._cache.get(config_id)
        if cached is not None:
            return cached

        measurement = self.table.get(config_id)
        if measurement is None:
            raise NotMeasuredError(
                f"configuration {config_id!r} has not been measured "
                f"(table: {self.table.path.name})"
            )

        if self.budget is not None and self.n_queries >= self.budget:
            raise BudgetExhausted(
                f"budget of {self.budget} evaluations exhausted; "
                f"{config_id!r} was not evaluated"
            )

        self._cache[config_id] = measurement
        self._log.append(QueryRecord(step=len(self._log) + 1, config_id=config_id))
        return measurement

    # -- introspection -----------------------------------------------------
    def evaluated_ids(self) -> List[str]:
        return list(self._cache)

    def evaluated_measurements(self) -> List[Measurement]:
        return list(self._cache.values())

    def query_log(self) -> List[QueryRecord]:
        return list(self._log)

    def reset(self) -> None:
        """Forget every evaluation (the budget is restored)."""
        self._cache.clear()
        self._log.clear()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Evaluator(rows={len(self.table)}, evaluated={self.n_queries}, "
                f"budget={self.budget})")


def load_evaluator(path: Path | str, known_config_ids: Optional[Iterable[str]] = None,
                   budget: Optional[int] = None) -> Evaluator:
    """Convenience constructor: load a table and wrap it in an evaluator."""
    table = BenchmarkTable(path, known_config_ids=known_config_ids)
    return Evaluator(table, budget=budget)


def is_feasible(measurement: Measurement,
                max_latency_ms: Optional[float] = None,
                max_memory_mb: Optional[float] = None) -> bool:
    """Whether a measurement satisfies the deployment constraints."""
    if max_latency_ms is not None and measurement.latency_median_ms > max_latency_ms:
        return False
    if max_memory_mb is not None and measurement.memory_mb > max_memory_mb:
        return False
    return True
