# Benchmark tables

A benchmark table is the **cached outcome of every configuration evaluation**.
It is what makes the core experiment reproducible on a laptop: the search
algorithms query the evaluator, the evaluator answers from this file, and no
detector has to be run.

## Files

| File | Content |
|---|---|
| `synthetic_demo.csv` | Generated placeholder, **not** measurements. Used only to exercise the pipeline end to end. |
| `pi4b_2gb_reference.csv` | Real Raspberry Pi 4B / 2 GB measurements (added once the hardware run is done). |

Rows from different sources may be concatenated **only** if they never share a
`config_id`; the loader rejects duplicates.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `config_id` | string | Stable identifier, e.g. `yolov5n_r320_fp32_t4`. Must match the search space. |
| `model` | string | `yolov5n` / `yolov5s` / `yolov5m` |
| `resolution` | int | Square input resolution |
| `precision` | string | `fp32` / `int8` |
| `threads` | int | CPU thread count used at inference |
| `accuracy_map50` | float | Detection accuracy (mAP@0.5) on the fixed COCO subset |
| `latency_median_ms` | float | Median wall-clock latency per image |
| `latency_p95_ms` | float | 95th percentile latency |
| `memory_mb` | float | Peak process memory during inference |
| `num_images` | int | Number of images the accuracy was measured on |
| `source` | string | `synthetic`, `raspberry_pi_4b`, or `laptop` |
| `notes` | string | Free text (optional) |

## Rules

1. **One row per configuration.** Duplicate `config_id` values are a hard error.
2. **All configurations share the evaluation pipeline**: same image subset, same
   preprocessing, same post-processing thresholds.
3. **`source` is not cosmetic.** Any result computed from rows whose source is
   `synthetic` may only be used to test the code, never reported as a finding.
4. **Missing rows are meaningful.** A configuration with no row simply has not
   been evaluated yet; the evaluator returns an error rather than guessing.

## Producing a table

```bash
# Placeholder, for testing the pipeline:
python scripts/make_synthetic_benchmark.py --out benchmarks/synthetic_demo.csv

# Real measurements, on the Raspberry Pi:
python scripts/benchmark_pi.py --out benchmarks/pi4b_2gb_reference.csv
```

The second command is added in the hardware step and is **optional** for anyone
reproducing the core results.
