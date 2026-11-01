"""Architecture specifications for the YOLOv5 variants in the search space.

The three variants (n / s / m) share one layer topology and differ only in the
``depth_multiple`` and ``width_multiple`` scaling factors, exactly as in the
official Ultralytics configuration files. Building the specification
analytically keeps the core experiment free of any PyTorch dependency: the
analytic capacity score only needs the *shape* of every weight matrix.

Reference topology: Ultralytics YOLOv5 (v7.0), base file ``yolov5s.yaml``,
blocks expanded as in the reference implementation:

    C3(c1, c2, n) = Conv(c1, c2/2, 1) + Conv(c1, c2/2, 1)
                    + n * Bottleneck(c2/2, c2/2) + Conv(c2, c2, 1)
    Bottleneck(c, c) = Conv(c, c, 1) + Conv(c, c, 3)
    SPPF(c1, c2)     = Conv(c1, c2/2, 1) + Conv(4*c2/2, c2, 1)

Scaled as in the reference implementation:
    channels: make_divisible(nominal * width_multiple, 8)
    depth:    max(round(nominal * depth_multiple), 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: (from, number, module, args) in the order used by the reference YAML.
BACKBONE: Sequence[Tuple] = (
    (-1, 1, "Conv", (64, 6, 2)),      # 0  P1/2
    (-1, 1, "Conv", (128, 3, 2)),     # 1  P2/4
    (-1, 3, "C3", (128,)),            # 2
    (-1, 1, "Conv", (256, 3, 2)),     # 3  P3/8
    (-1, 6, "C3", (256,)),            # 4
    (-1, 1, "Conv", (512, 3, 2)),     # 5  P4/16
    (-1, 9, "C3", (512,)),            # 6
    (-1, 1, "Conv", (1024, 3, 2)),    # 7  P5/32
    (-1, 3, "SPPF", (1024, 5)),       # 8
)

HEAD: Sequence[Tuple] = (
    (-1, 1, "Conv", (512, 1, 1)),         # 9
    (-1, 1, "Upsample", (2,)),            # 10
    ((-1, 6), 1, "Concat", ()),           # 11
    (-1, 3, "C3", (512,)),                # 12
    (-1, 1, "Conv", (256, 1, 1)),         # 13
    (-1, 1, "Upsample", (2,)),            # 14
    ((-1, 4), 1, "Concat", ()),           # 15
    (-1, 3, "C3", (256,)),                # 16
    (-1, 1, "Conv", (256, 3, 2)),         # 17
    ((-1, 14), 1, "Concat", ()),          # 18
    (-1, 3, "C3", (512,)),                # 19
    (-1, 1, "Conv", (512, 3, 2)),         # 20
    ((-1, 10), 1, "Concat", ()),          # 21
    (-1, 3, "C3", (1024,)),               # 22
    ((17, 20, 22), 1, "Detect", (80, 3)),  # 23
)

LAYERS: Sequence[Tuple] = BACKBONE + HEAD

#: depth_multiple, width_multiple, and the reference parameter count (millions)
#: reported by the official Ultralytics YOLOv5 model table.
VARIANTS: Dict[str, Dict[str, float]] = {
    "yolov5n": {"depth": 0.33, "width": 0.25, "params_m": 1.9},
    "yolov5s": {"depth": 0.33, "width": 0.50, "params_m": 7.2},
    "yolov5m": {"depth": 0.67, "width": 0.75, "params_m": 21.2},
}


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round ``x`` up to the nearest multiple of ``divisor`` (reference rule)."""
    return int(math.ceil(x / divisor) * divisor)


@dataclass(frozen=True)
class ConvMatrix:
    """One weight matrix of a convolution: ``(out_channels, in_channels*k*k)``."""

    name: str
    out_channels: int
    in_channels: int
    kernel: int

    @property
    def m(self) -> int:
        return self.out_channels

    @property
    def n(self) -> int:
        return self.in_channels * self.kernel * self.kernel

    @property
    def params(self) -> int:
        """Weights plus batch-norm affine parameters (Conv has no bias)."""
        return self.out_channels * self.n + 2 * self.out_channels


def _resolve_inputs(frm, outputs: List[int]) -> List[int]:
    if isinstance(frm, (list, tuple)):
        return [outputs[f] for f in frm]
    return [outputs[frm]]


def build_variant(variant: str) -> Tuple[List[ConvMatrix], int]:
    """Return every convolution weight matrix of ``variant`` and the total parameter count."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")

    gd = VARIANTS[variant]["depth"]
    gw = VARIANTS[variant]["width"]

    outputs: List[int] = [3]  # stem input = RGB
    convs: List[ConvMatrix] = []

    def add(out: int, inp: int, k: int, tag: str) -> None:
        convs.append(ConvMatrix(f"{tag}", out, inp, k))

    for index, (frm, repeat, module, args) in enumerate(LAYERS):
        inputs = _resolve_inputs(frm, outputs)
        c1 = inputs[0] if len(inputs) == 1 else sum(inputs)

        if module == "Conv":
            nominal, k = args[0], args[1]
            c2 = make_divisible(nominal * gw, 8)
            add(c2, c1, k, f"{index}.conv")
            outputs.append(c2)

        elif module == "C3":
            nominal = args[0]
            c2 = make_divisible(nominal * gw, 8)
            c_ = c2 // 2
            n = max(round(repeat * gd), 1)
            add(c_, c1, 1, f"{index}.cv1")
            add(c_, c1, 1, f"{index}.cv2")
            for b in range(n):
                add(c_, c_, 1, f"{index}.m.{b}.cv1")
                add(c_, c_, 3, f"{index}.m.{b}.cv2")
            add(c2, 2 * c_, 1, f"{index}.cv3")
            outputs.append(c2)

        elif module == "SPPF":
            nominal = args[0]
            c2 = make_divisible(nominal * gw, 8)
            c_ = c2 // 2
            add(c_, c1, 1, f"{index}.cv1")
            add(c2, 4 * c_, 1, f"{index}.cv2")
            outputs.append(c2)

        elif module == "Upsample":
            outputs.append(c1)

        elif module == "Concat":
            outputs.append(sum(inputs))

        elif module == "Detect":
            nc = int(args[0])
            na = int(args[1])
            for level, c_in in enumerate(inputs):
                # one 1x1 convolution per detection level producing
                # na * (nc + 5) outputs (box + objectness + class logits)
                c_out = na * (nc + 5)
                add(c_out, c_in, 1, f"{index}.cv{level}")
            outputs.append(na * (nc + 5))

        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported module {module!r}")

    total_params = sum(c.params for c in convs)
    return convs, total_params


def variant_summary(variant: str) -> Dict[str, object]:
    convs, params = build_variant(variant)
    return {
        "variant": variant,
        "num_conv_matrices": len(convs),
        "params": params,
        "params_m": params / 1e6,
        "reference_params_m": VARIANTS[variant]["params_m"],
    }
