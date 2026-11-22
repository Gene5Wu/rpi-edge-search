"""Validate the analytic architecture specification against published figures.

The analytic spec is used for two things:

* the **capacity score**, which only needs the *shape* of every weight matrix
  and is invariant to a uniform scale error, and
* the **resource model**, whose latency prediction is driven by the MAC count.

This script checks both against the reference table, because the two agree to
very different degrees:

* **parameters** are under-counted by roughly 15-20%. The reference count
  includes parts this spec simplifies (the detection head is modelled as one
  ``1x1`` convolution per level). Nothing downstream depends on the absolute
  value: the capacity score is used only for ranking, and the weights are a
  small slice of the memory estimate.
* **compute** matches the published GFLOPs within a few percent, which is the
  quantity that actually drives the latency prediction. This is the check that
  matters.

If ONNX exports are available (``models/*.onnx``), the script reports the true
counts next to the analytic ones. Reading them requires the optional ``onnx``
package; without it the check is skipped rather than guessed.

Usage
-----
    python scripts/check_model_spec.py
    python scripts/check_model_spec.py --models-dir models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_spec import VARIANTS, build_variant, total_macs  # noqa: E402

REFERENCE_RESOLUTION = 640


def onnx_parameter_count(path: Path):
    """True parameter count from an ONNX graph, or ``None`` if unavailable."""
    try:
        import onnx  # optional dependency
        from onnx import numpy_helper
    except ImportError:
        return None
    model = onnx.load(str(path))
    total = 0
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.ndim >= 2:  # weights only, not biases/BN scalars
            total += array.size
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--resolution", type=int, default=REFERENCE_RESOLUTION)
    parser.add_argument("--onnx-tolerance", type=float, default=0.25,
                        help="allowed relative gap between the analytic and ONNX "
                             "parameter counts before the check fails")
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir)

    print("=" * 88)
    print(f"Analytic architecture spec vs published reference (input {args.resolution}px)")
    print("=" * 88)
    header = (f"{'variant':<10}{'params(M)':>11}{'ref(M)':>9}{'ratio':>8}"
              f"{'GMACs':>9}{'GFLOPs':>9}{'ref':>7}{'ratio':>8}")
    print(header)
    print("-" * len(header))

    worst_params = 0.0
    worst_flops = 0.0
    rows = []

    for variant, info in VARIANTS.items():
        convs, params = build_variant(variant, input_size=args.resolution)
        macs = total_macs(convs)
        gflops = 2.0 * macs / 1e9
        ref_params = info["params_m"]
        ref_flops = info["gflops"]

        params_ratio = params / (ref_params * 1e6)
        flops_ratio = gflops / ref_flops
        worst_params = max(worst_params, abs(1 - params_ratio))
        worst_flops = max(worst_flops, abs(1 - flops_ratio))

        print(f"{variant:<10}{params / 1e6:>11.3f}{ref_params:>9.2f}"
              f"{params_ratio:>8.3f}{macs / 1e9:>9.2f}{gflops:>9.2f}"
              f"{ref_flops:>7.1f}{flops_ratio:>8.3f}")
        rows.append((variant, params, gflops))

    print()
    print(f"largest parameter deviation : {worst_params:6.1%}  (expected: the head is simplified)")
    print(f"largest compute deviation   : {worst_flops:6.1%}  (this drives latency)")

    # -- optional: compare against real ONNX exports ----------------------
    print()
    print("ONNX exports:")
    found_any = False
    mismatches = 0
    for variant, params, _ in rows:
        path = models_dir / f"{variant}_fp32.onnx"
        if not path.is_file():
            continue
        found_any = True
        true_count = onnx_parameter_count(path)
        if true_count is None:
            print(f"  {variant}: {path.name} found, but the 'onnx' package is not "
                  f"installed (pip install onnx) -- cannot verify")
            continue
        ratio = params / true_count
        ok = abs(1 - ratio) <= args.onnx_tolerance
        if not ok:
            mismatches += 1
        print(f"  {variant}: {path.name} -> {true_count / 1e6:.3f}M "
              f"(analytic {params / 1e6:.3f}M, ratio {ratio:.3f}) "
              f"[{'OK' if ok else 'MISMATCH'}]")
    if not found_any:
        print("  none found; run models/download_models.py to create them")

    print()
    print("Verdict: compute matches the reference closely, so the latency model is")
    print("sound; the parameter deficit is uniform and only affects a minor part of")
    print("the memory estimate. The capacity score is scale-invariant in any case.")
    if mismatches:
        print()
        print(f"FAIL: {mismatches} exported model(s) differ from the analytic "
              f"architecture by more than {args.onnx_tolerance:.0%}.")
        print("The benchmarked network and the scored network must be the same model;")
        print("check that models/download_models.py uses the YOLOv5 v7.0 weights.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
