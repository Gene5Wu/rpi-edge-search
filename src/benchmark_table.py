"""Loading and validation of archived benchmark tables.

The benchmark table is the interface between the *expensive* part of the
project (running a detector on real hardware) and the *cheap* part (searching
over the cached outcomes). Keeping the validation strict here means the search
code can trust its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

REQUIRED_COLUMNS: Sequence[str] = (
    "config_id",
    "model",
    "resolution",
    "precision",
    "threads",
    "accuracy_map50",
    "latency_median_ms",
    "latency_p95_ms",
    "memory_mb",
    "num_images",
    "source",
)

#: Sources that are placeholders and must never be reported as measurements.
SYNTHETIC_SOURCES = frozenset({"synthetic"})


@dataclass(frozen=True)
class Measurement:
    """One evaluated configuration."""

    config_id: str
    accuracy: float
    latency_median_ms: float
    latency_p95_ms: float
    memory_mb: float
    num_images: int
    source: str

    @property
    def is_synthetic(self) -> bool:
        return self.source in SYNTHETIC_SOURCES

    def to_dict(self) -> Dict[str, object]:
        return {
            "config_id": self.config_id,
            "accuracy": self.accuracy,
            "latency_median_ms": self.latency_median_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "memory_mb": self.memory_mb,
            "num_images": self.num_images,
            "source": self.source,
        }


class BenchmarkTable:
    """A validated, read-only view over one benchmark CSV."""

    def __init__(self, path: Path | str, known_config_ids: Optional[Iterable[str]] = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"benchmark table not found: {self.path}")

        frame = pd.read_csv(self.path)
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing_columns:
            raise ValueError(f"{self.path} is missing columns: {missing_columns}")

        duplicates = frame["config_id"][frame["config_id"].duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(f"{self.path} has duplicate config_id rows: {duplicates}")

        if known_config_ids is not None:
            known = set(known_config_ids)
            unknown = sorted(set(frame["config_id"]) - known)
            if unknown:
                raise ValueError(
                    f"{self.path} contains config_ids that are not in the search space: "
                    f"{unknown}"
                )

        numeric = ["accuracy_map50", "latency_median_ms", "latency_p95_ms", "memory_mb"]
        for column in numeric:
            if frame[column].isna().any():
                raise ValueError(f"{self.path} has missing values in '{column}'")
            if (frame[column] < 0).any():
                raise ValueError(f"{self.path} has negative values in '{column}'")

        self._frame = frame
        self._rows: Dict[str, Measurement] = {
            row.config_id: Measurement(
                config_id=row.config_id,
                accuracy=float(row.accuracy_map50),
                latency_median_ms=float(row.latency_median_ms),
                latency_p95_ms=float(row.latency_p95_ms),
                memory_mb=float(row.memory_mb),
                num_images=int(row.num_images),
                source=str(row.source),
            )
            for row in frame.itertuples(index=False)
        }

    # -- access ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, config_id: object) -> bool:
        return config_id in self._rows

    def has(self, config_id: str) -> bool:
        return config_id in self._rows

    def get(self, config_id: str) -> Optional[Measurement]:
        return self._rows.get(config_id)

    def config_ids(self) -> List[str]:
        return list(self._rows)

    def measurements(self) -> List[Measurement]:
        return list(self._rows.values())

    def missing(self, config_ids: Iterable[str]) -> List[str]:
        return [cid for cid in config_ids if cid not in self._rows]

    # -- diagnostics -------------------------------------------------------
    def source_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for measurement in self._rows.values():
            counts[measurement.source] = counts.get(measurement.source, 0) + 1
        return counts

    def is_synthetic(self) -> bool:
        return all(m.is_synthetic for m in self._rows.values())

    def to_frame(self) -> pd.DataFrame:
        return self._frame.copy()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"BenchmarkTable(path={self.path.name!r}, rows={len(self)}, "
                f"sources={self.source_counts()})")
