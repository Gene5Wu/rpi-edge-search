"""Deployment configuration space for the edge-vision search project.

The space is defined entirely by ``configs/search_space.yaml``; nothing is
hard-coded here so that the YAML file stays the single source of truth.

Usage
-----
    python src/search_space.py --stats
    python src/search_space.py --json
    python src/search_space.py --csv configs/configs.csv
    python src/search_space.py --precision fp32 --model yolov5n --stats
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_PATH = REPO_ROOT / "configs" / "search_space.yaml"

REQUIRED_KEYS = ("models", "resolution", "precision", "threads", "backend")


@dataclass(frozen=True)
class Config:
    """One deployable configuration (a point in the search space)."""

    config_id: str
    model: str
    resolution: int
    precision: str
    threads: int
    backend: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def make_config_id(model: str, resolution: int, precision: str, threads: int) -> str:
    """Deterministic, human-readable identifier for a configuration.

    Example: ``yolov5n_r320_fp32_t4``.
    """
    return f"{model}_r{resolution}_{precision}_t{threads}"


def load_space(path: Path | str = DEFAULT_SPACE_PATH) -> Dict:
    """Load and validate the search-space definition."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"search space file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        space = yaml.safe_load(handle)

    if not isinstance(space, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    missing = [key for key in REQUIRED_KEYS if key not in space]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    _validate_space(space)
    return space


def _validate_space(space: Dict) -> None:
    """Fail fast on an inconsistent definition."""
    model_ids = [m["id"] for m in space["models"]]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError(f"duplicate model ids: {model_ids}")

    for key in ("resolution", "precision", "threads"):
        values = space[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"'{key}' must be a non-empty list")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate values in '{key}': {values}")

    for value in space["resolution"]:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"resolution must be a positive int, got {value!r}")

    for value in space["threads"]:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"threads must be a positive int, got {value!r}")

    if "id" not in space["backend"]:
        raise ValueError("'backend' must contain an 'id'")


def enumerate_configs(
    space: Dict,
    models: Optional[Sequence[str]] = None,
    resolutions: Optional[Sequence[int]] = None,
    precisions: Optional[Sequence[str]] = None,
    threads: Optional[Sequence[int]] = None,
) -> List[Config]:
    """Return the (filtered) Cartesian product of the space, as Config objects.

    Ordering is deterministic: model, then resolution, then precision, then
    threads. This matters because tie-breaking in the search algorithms must be
    reproducible.
    """
    model_ids = [m["id"] for m in space["models"]]
    backend_id = space["backend"]["id"]

    def _filter(values: Iterable, requested: Optional[Sequence]) -> List:
        values = list(values)
        if requested is None:
            return values
        unknown = [v for v in requested if v not in values]
        if unknown:
            raise ValueError(
                f"requested value(s) {unknown} not in space; available: {values}"
            )
        # preserve the order defined in the YAML file, not the CLI order
        return [v for v in values if v in set(requested)]

    model_ids = _filter(model_ids, models)
    resolutions = _filter(space["resolution"], resolutions)
    precisions = _filter(space["precision"], precisions)
    thread_counts = _filter(space["threads"], threads)

    configs: List[Config] = []
    for model, resolution, precision, thread_count in itertools.product(
        model_ids, resolutions, precisions, thread_counts
    ):
        configs.append(
            Config(
                config_id=make_config_id(model, resolution, precision, thread_count),
                model=model,
                resolution=resolution,
                precision=precision,
                threads=thread_count,
                backend=backend_id,
            )
        )
    return configs


def check_configs(configs: Sequence[Config], space: Dict) -> None:
    """Sanity-check the generated pool (used by --self-test)."""
    ids = [c.config_id for c in configs]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate config ids generated")

    expected = (
        len(space["models"])
        * len(space["resolution"])
        * len(space["precision"])
        * len(space["threads"])
    )
    if len(configs) != expected:
        raise AssertionError(f"expected {expected} configs, got {len(configs)}")

    for config in configs:
        if config.precision not in space["precision"]:
            raise AssertionError(f"unknown precision {config.precision}")
        if config.resolution not in space["resolution"]:
            raise AssertionError(f"unknown resolution {config.resolution}")
        if config.threads not in space["threads"]:
            raise AssertionError(f"unknown threads {config.threads}")


def summarize(configs: Sequence[Config], space: Dict) -> str:
    """Human-readable summary of the generated pool."""
    lines: List[str] = []
    lines.append("Search space summary")
    lines.append("=" * 20)
    lines.append(f"models      : {[m['id'] for m in space['models']]}")
    lines.append(f"resolution  : {space['resolution']}")
    lines.append(f"precision   : {space['precision']}")
    lines.append(f"threads     : {space['threads']}")
    lines.append(f"backend     : {space['backend']['id']}")
    lines.append(f"total configs: {len(configs)}")

    by_model: Dict[str, int] = {}
    by_precision: Dict[str, int] = {}
    for config in configs:
        by_model[config.model] = by_model.get(config.model, 0) + 1
        by_precision[config.precision] = by_precision.get(config.precision, 0) + 1
    lines.append(f"per model   : {by_model}")
    lines.append(f"per precision: {by_precision}")
    lines.append("")
    lines.append("First / last configurations:")
    for config in configs[:3]:
        lines.append(f"  {config.config_id}")
    if len(configs) > 3:
        lines.append("  ...")
        lines.append(f"  {configs[-1].config_id}")
    return "\n".join(lines)


def write_csv(configs: Sequence[Config], path: Path | str) -> None:
    """Write the pool as CSV (config_id first, then parameters)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(configs[0].to_dict().keys()))
        writer.writeheader()
        for config in configs:
            writer.writerow(config.to_dict())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=str(DEFAULT_SPACE_PATH),
                        help="path to search_space.yaml")
    parser.add_argument("--model", action="append", default=None,
                        help="restrict to a model id (repeatable)")
    parser.add_argument("--resolution", action="append", type=int, default=None,
                        help="restrict to a resolution (repeatable)")
    parser.add_argument("--precision", action="append", default=None,
                        help="restrict to a precision (repeatable)")
    parser.add_argument("--threads", action="append", type=int, default=None,
                        help="restrict to a thread count (repeatable)")
    parser.add_argument("--stats", action="store_true", help="print a summary")
    parser.add_argument("--json", action="store_true", help="print configs as JSON")
    parser.add_argument("--csv", default=None, help="write configs to a CSV file")
    parser.add_argument("--self-test", action="store_true",
                        help="run internal consistency checks and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    space = load_space(args.space)
    configs = enumerate_configs(
        space,
        models=args.model,
        resolutions=args.resolution,
        precisions=args.precision,
        threads=args.threads,
    )

    if args.self_test:
        check_configs(configs, space)
        print(f"self-test passed: {len(configs)} unique configurations")
        return 0

    if args.stats:
        print(summarize(configs, space))

    if args.json:
        print(json.dumps([c.to_dict() for c in configs], indent=2))

    if args.csv:
        write_csv(configs, args.csv)
        print(f"wrote {len(configs)} configurations to {args.csv}")

    if not (args.stats or args.json or args.csv):
        for config in configs:
            print(config.config_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
