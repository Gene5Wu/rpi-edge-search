"""Combined analytic score used to rank and select deployment configurations.

This is the project's adaptation of the course's *Neural Spectral Capacity*
idea. Topic C scores a Transformer architecture from its specification alone;
here the candidate is a **deployment configuration**, so the score has two
ingredients:

1. **Capacity term** — the NSC sum over the detector's convolution matrices,
   scaled by the input resolution (a larger input feeds the network more spatial
   evidence, so the same architecture has more usable capacity) and by a small
   quantisation penalty for INT8.
2. **Resource term** — predicted latency and memory violations of the
   deployment constraints.

The score is therefore

    A_hat(x) = C(model) * (R / R_ref)^alpha * q(precision)
    S(x)     = A_hat(x)/max(A_hat) - lam * P_latency(x) - mu * P_memory(x)

where the penalties are zero when the prediction is inside the constraint and
grow linearly outside it. ``alpha``, ``q``, ``lam`` and ``mu`` are explicit
modelling assumptions, not fitted numbers; the ablation removes one ingredient
at a time so the report can say which ingredient carries the signal.

**Important.** The score never reads a measured accuracy, latency or memory for
the configuration being scored. It uses only the configuration's own parameters
and quantities derived from the architecture specification, exactly like the
heuristic in a search problem, which must not see the outcome of an unevaluated
state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from model_spec import VARIANTS, build_variant
from nsc import DEFAULT_INIT_STD, capacity
from resource_model import ResourceModel

#: Reference resolution used by the resolution scaling term.
REFERENCE_RESOLUTION = 640.0


@dataclass(frozen=True)
class ScoreWeights:
    """Modelling assumptions of the analytic score."""

    #: Exponent of the resolution scaling term (R / 640)^alpha.
    alpha: float = 1.0
    #: Relative capacity retained after INT8 quantisation.
    int8_retention: float = 0.97
    #: Weight of the latency-violation penalty.
    lam: float = 1.0
    #: Weight of the memory-violation penalty.
    mu: float = 1.0
    #: Weight of the *soft* resource pressure (tie-break between equally capable
    #: configurations when no hard cap is active).
    soft_weight: float = 0.05
    #: Upper bound applied to each violation penalty. Without it, a configuration
    #: that misses the cap by 10x contributes a penalty of 10 and swamps the
    #: accuracy term, so the score ranks by *violation size* instead of by
    #: accuracy. Clamping keeps the penalty a hint rather than a dictator.
    penalty_cap: float = 1.0
    #: Ablation switch: "full", "no_resource", or "no_capacity".
    ablation: str = "full"
    #: Which cheap quantity stands in for accuracy when ranking candidates:
    #:   "nsc"    -- the NSC capacity term scaled by resolution and precision
    #:               (this project's proposal),
    #:   "macs"   -- predicted multiply-accumulates, a free compute proxy,
    #:   "params" -- parameter count, the cheapest possible proxy.
    #: The first is only worth its extra machinery if it beats the other two on
    #: measured data; ``scripts/compare_proxies.py`` runs that comparison.
    proxy: str = "nsc"

    def __post_init__(self) -> None:
        if self.ablation not in ("full", "no_resource", "no_capacity"):
            raise ValueError(f"unknown ablation mode {self.ablation!r}")
        if self.proxy not in ("nsc", "macs", "params"):
            raise ValueError(f"unknown accuracy proxy {self.proxy!r}")


def architectural_capacity(model: str, init_std: float = DEFAULT_INIT_STD) -> float:
    """NSC capacity of one detector variant (architecture only)."""
    convs, _ = build_variant(model)
    return capacity(convs, init_std)


class Scorer:
    """Turns a configuration into an analytic score between roughly 0 and 1."""

    def __init__(
        self,
        configs: Sequence,
        resource_model: Optional[ResourceModel] = None,
        weights: Optional[ScoreWeights] = None,
        init_std: float = DEFAULT_INIT_STD,
        max_latency_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
    ) -> None:
        self.resource_model = resource_model or ResourceModel()
        self.weights = weights or ScoreWeights()
        self.max_latency_ms = max_latency_ms
        self.max_memory_mb = max_memory_mb

        self._capacity: Dict[str, float] = {
            model: architectural_capacity(model, init_std) for model in VARIANTS
        }
        self._params: Dict[str, float] = {
            model: float(sum(matrix.params for matrix in build_variant(model)[0]))
            for model in VARIANTS
        }
        self._resources: Dict[str, Dict[str, float]] = {
            config.config_id: self.resource_model.estimate(config) for config in configs
        }
        self._reference = self._compute_reference(configs)

    # -- internals ---------------------------------------------------------
    def _accuracy_proxy(self, config) -> float:
        """The cheap stand-in for accuracy; which one is set by ``weights.proxy``."""
        mode = self.weights.proxy

        if mode == "nsc":
            base = self._capacity[config.model]
            resolution_scale = (config.resolution / REFERENCE_RESOLUTION) ** self.weights.alpha
            quant_scale = 1.0 if config.precision == "fp32" else self.weights.int8_retention
            return base * resolution_scale * quant_scale

        if mode == "macs":
            # Predicted compute already scales with resolution, so no extra factor.
            return float(self._resources[config.config_id]["macs"])

        if mode == "params":
            # Architecture only: blind to resolution and precision on purpose.
            return self._params[config.model]

        raise ValueError(f"unknown accuracy proxy {mode!r}")

    def _compute_reference(self, configs: Sequence) -> Dict[str, float]:
        proxies = [self._accuracy_proxy(c) for c in configs]
        latencies = [self._resources[c.config_id]["latency_ms"] for c in configs]
        memories = [self._resources[c.config_id]["memory_mb"] for c in configs]
        return {
            "max_proxy": max(proxies) if proxies else 1.0,
            "max_latency": max(latencies) if latencies else 1.0,
            "max_memory": max(memories) if memories else 1.0,
        }

    def _penalties(self, config) -> tuple:
        resource = self._resources[config.config_id]
        latency = resource["latency_ms"]
        memory = resource["memory_mb"]

        if self.max_latency_ms:
            latency_penalty = max(0.0, latency - self.max_latency_ms) / self.max_latency_ms
        else:
            latency_penalty = 0.0
        if self.max_memory_mb:
            memory_penalty = max(0.0, memory - self.max_memory_mb) / self.max_memory_mb
        else:
            memory_penalty = 0.0

        cap = self.weights.penalty_cap
        if cap is not None:
            latency_penalty = min(latency_penalty, cap)
            memory_penalty = min(memory_penalty, cap)

        # Soft pressure toward cheaper configurations even without hard caps,
        # so that the score still ranks inside a model variant.
        soft_latency = latency / max(self._reference["max_latency"], 1e-9)
        soft_memory = memory / max(self._reference["max_memory"], 1e-9)
        return latency_penalty, memory_penalty, soft_latency, soft_memory

    # -- public API --------------------------------------------------------
    def score(self, config) -> float:
        proxy = self._accuracy_proxy(config) / max(self._reference["max_proxy"], 1e-9)
        lat_pen, mem_pen, soft_lat, soft_mem = self._penalties(config)

        mode = self.weights.ablation
        if mode == "no_capacity":
            accuracy_term = 0.0
        else:
            accuracy_term = proxy

        if mode == "no_resource":
            return accuracy_term

        return (accuracy_term
                - self.weights.lam * lat_pen
                - self.weights.mu * mem_pen
                - self.weights.soft_weight * (soft_lat + soft_mem))

    def describe(self, config) -> Dict[str, object]:
        resource = self._resources[config.config_id]
        lat_pen, mem_pen, soft_lat, soft_mem = self._penalties(config)
        return {
            "config_id": config.config_id,
            "model": config.model,
            "resolution": config.resolution,
            "precision": config.precision,
            "threads": config.threads,
            "accuracy_proxy": self._accuracy_proxy(config),
            "pred_latency_ms": resource["latency_ms"],
            "pred_memory_mb": resource["memory_mb"],
            "capacity": self._capacity[config.model],
            "latency_penalty": lat_pen,
            "memory_penalty": mem_pen,
            "score": self.score(config),
        }

    def rank(self, configs: Iterable) -> List[tuple]:
        """Configurations sorted by descending score."""
        return sorted(((self.score(c), c) for c in configs), key=lambda item: item[0],
                      reverse=True)
