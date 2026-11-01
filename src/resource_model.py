"""Analytic resource cost model.

The search must decide *which* configuration to evaluate next without having
evaluated it. That requires a cheap estimate of latency and memory derived from
the configuration alone — the "resource side" of the score.

Model
-----
    MACs(model, R)  = sum over convs of  H_out * W_out * C_out * C_in * k^2
    latency_ms      = MACs / (macs_per_second_per_thread * threads) * 1000 + overhead_ms
    memory_mb       = weights_bytes + peak_activation_bytes

where ``peak_activation_bytes`` uses the largest intermediate tensor, which is
what dominates the peak RSS for a small detector at a large resolution.

The throughput constant is a *calibration prior*, not a measured value. It is
fitted from the benchmark table once measurements exist (``ResourceModel.fit``);
until then every estimate is explicitly a prediction, and the report must say so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from model_spec import (
    ConvMatrix,
    build_variant,
    peak_activation_elements,
    total_macs,
)

#: Bytes stored per weight, by numerical precision.
BYTES_PER_PARAM: Dict[str, float] = {"fp32": 4.0, "int8": 1.0}


@lru_cache(maxsize=256)
def _spec(variant: str, resolution: int) -> Tuple[List[ConvMatrix], int]:
    """Cached (convolution list, parameter count) for a variant at a resolution."""
    return build_variant(variant, input_size=resolution)


@dataclass(frozen=True)
class ResourceModel:
    """Predicts latency and memory from the configuration specification."""

    #: Sustained MAC/s per CPU thread. Prior for a Cortex-A72 class core running
    #: ONNX Runtime; to be fitted from measurements.
    macs_per_second_per_thread: float = 4.0e9
    #: Fixed per-inference overhead (framework dispatch, pre/post-processing).
    fixed_overhead_ms: float = 8.0
    #: Multiplier on the activation buffer (temporary im2col / output buffers).
    activation_factor: float = 1.0
    #: Constant memory baseline of the inference process (interpreter, runtime
    #: arenas, pre/post-processing buffers). Calibrated from measured RSS.
    runtime_overhead_mb: float = 50.0

    # -- predictions -------------------------------------------------------
    def latency_ms(self, macs: int, threads: int) -> float:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads!r}")
        compute_ms = macs / (self.macs_per_second_per_thread * threads) * 1000.0
        return compute_ms + self.fixed_overhead_ms

    def memory_mb(self, params: int, peak_elements: int, precision: str) -> float:
        if precision not in BYTES_PER_PARAM:
            raise ValueError(f"unknown precision {precision!r}")
        nbytes = BYTES_PER_PARAM[precision]
        weights_mb = params * nbytes / 1e6
        activation_mb = peak_elements * nbytes * self.activation_factor / 1e6
        return weights_mb + activation_mb + self.runtime_overhead_mb

    def estimate(self, config) -> Dict[str, float]:
        """Predict the resource cost of one configuration."""
        convs, params = _spec(config.model, config.resolution)
        macs = total_macs(convs)
        peak = peak_activation_elements(convs)
        return {
            "macs": float(macs),
            "gmacs": macs / 1e9,
            "params": float(params),
            "peak_activation_elements": float(peak),
            "latency_ms": self.latency_ms(macs, config.threads),
            "memory_mb": self.memory_mb(params, peak, config.precision),
        }

    # -- calibration -------------------------------------------------------
    def fit(self, configs: Sequence, measured: Dict[str, Dict[str, float]]) -> "ResourceModel":
        """Refit the throughput prior from measured latencies.

        ``measured`` maps ``config_id`` to at least ``{"latency_ms": ...}``. Only
        the compute term is fitted (least squares through the origin on
        ``macs / threads`` versus ``latency - fixed_overhead``); the overhead and
        activation factor keep their prior values.
        """
        numerator = 0.0
        denominator = 0.0
        for config in configs:
            record = measured.get(config.config_id)
            if not record:
                continue
            convs, _ = _spec(config.model, config.resolution)
            macs = total_macs(convs)
            compute_ms = float(record["latency_ms"]) - self.fixed_overhead_ms
            if compute_ms <= 0:
                continue
            # latency_compute = macs / (k * threads) * 1000  =>  k = 1000*macs/(threads*latency)
            rhs = 1000.0 * macs / (config.threads * compute_ms)
            numerator += rhs
            denominator += 1.0
        if denominator == 0:
            raise ValueError("no usable measurements to fit the resource model")
        return replace(self, macs_per_second_per_thread=numerator / denominator)


def predicted_table(configs: Iterable, model: ResourceModel,
                    include_capacity: bool = False) -> List[Dict[str, object]]:
    """Convenience: predict every configuration and return plain rows."""
    rows: List[Dict[str, object]] = []
    for config in configs:
        row = {"config_id": config.config_id,
               "model": config.model,
               "resolution": config.resolution,
               "precision": config.precision,
               "threads": config.threads}
        row.update(model.estimate(config))
        rows.append(row)
    return rows
