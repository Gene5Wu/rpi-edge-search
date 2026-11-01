# Resource-Aware Heuristic Search for Edge Vision Deployment Configuration Selection

CS5491 Artificial Intelligence — Course Project

**Research question.** Given a limited number of configuration evaluations, can a
resource-aware heuristic identify a high-accuracy edge-vision deployment
configuration that satisfies latency and memory constraints — without
exhaustively benchmarking every candidate?

---

## Overview

Deploying a vision model on an edge device is a trade-off between detection
accuracy, inference latency and memory consumption. Which trade-off point you get
depends on the deployment configuration: the detector variant, the input
resolution, the numerical precision and the number of CPU threads.

This project treats **deployment configuration selection as a resource-constrained
search problem**. Each configuration is an expensive black-box query that returns
an accuracy, a latency and a memory measurement. Under a fixed evaluation budget
only a subset of configurations can be queried, and the goal is to find a good
feasible configuration within that budget.

### Two-layer design (why this is reproducible without a Raspberry Pi)

| Layer | What it does | Requirement |
|---|---|---|
| **Core — laptop** | Search algorithms, baselines and evaluation run against an **archived benchmark table** (`benchmarks/*.csv`). Each row is a cached black-box evaluation. | Python + 4 packages. No GPU, no Pi. |
| **Hardware validation — Raspberry Pi 4B / 2 GB** | Regenerates the benchmark table with real measurements and checks that the configurations selected offline behave as predicted on real hardware. | A Raspberry Pi (optional). |

A reviewer can therefore reproduce **every table and figure** in the report on a
fresh laptop, and the Raspberry Pi only serves to validate the measurements.

### Methodological reference

The experimental protocol follows the course topic *Neural Spectral Capacity as a
Heuristic for Architecture Design*: an analytic, specification-only score is used
to rank candidates, and is then evaluated against measured outcomes with
rank correlation and fixed-budget selection.

| Topic C | This project |
|---|---|
| Analytic capacity score from the architecture spec | Analytic score from the configuration spec (architecture capacity + resource cost) |
| Spearman ρ / Kendall τb vs. trained perplexity | Spearman ρ / Kendall τb vs. measured accuracy, latency, memory |
| Fixed-budget selection under a parameter cap; regret | Fixed-budget selection under latency/memory caps; regret vs. exhaustive oracle |
| Ablation removing the new mechanism | Ablations removing the capacity term / the resource term |

---

## Repository structure

```text
rpi-edge-search/
├── README.md
├── PROGRESS.md                  # decision log + step-by-step progress
├── requirements.txt             # core deps (laptop-only reproduction)
├── requirements-benchmark.txt   # optional deps to regenerate measurements
├── configs/
│   └── search_space.yaml        # single source of truth for the candidate pool
├── benchmarks/                  # archived benchmark tables (CSV)
├── models/                      # model download / export instructions
├── results/                     # generated tables and figures
├── scripts/                     # runnable entry points
└── src/                         # library code
    └── search_space.py          # candidate pool enumeration
```

---

## Quick start

### 1. Environment

```bash
git clone https://github.com/Gene5Wu/rpi-edge-search.git
cd rpi-edge-search
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` deliberately contains only `numpy`, `pandas`, `PyYAML` and
`matplotlib`, so the core experiment runs on a fresh machine with no GPU and no
model downloads.

### 2. Enumerate the configuration pool

```bash
python src/search_space.py --stats
```

Expected output:

```text
Search space summary
====================
models      : ['yolov5n', 'yolov5s', 'yolov5m']
resolution  : [320, 416, 640]
precision   : ['fp32', 'int8']
threads     : [1, 2, 4]
backend     : onnxruntime
total configs: 54
per model   : {'yolov5n': 18, 'yolov5s': 18, 'yolov5m': 18}
per precision: {'fp32': 27, 'int8': 27}
```

Other useful invocations:

```bash
python src/search_space.py --self-test          # internal consistency checks
python src/search_space.py --precision fp32 --stats
python src/search_space.py --csv configs/configs.csv
python src/search_space.py --json
```

### 3. Configuration space

A configuration is the Cartesian product of the factors defined in
`configs/search_space.yaml`:

| Factor | Values |
|---|---|
| Model | `yolov5n`, `yolov5s`, `yolov5m` |
| Input resolution | 320, 416, 640 |
| Numerical precision | `fp32`, `int8` |
| CPU threads | 1, 2, 4 |
| Backend | ONNX Runtime (CPU) |

That yields **54** configurations (27 if restricted to FP32). Each configuration
has a stable identifier such as `yolov5n_r320_fp32_t4`, which is the join key
between the search space, the benchmark table and the results.

### 4. Try the benchmark evaluator

A placeholder benchmark table is included so the pipeline can be exercised
before real measurements exist. **It is synthetic — never report results derived
from it.**

```bash
python scripts/make_synthetic_benchmark.py   # regenerate the placeholder
python scripts/try_evaluator.py              # check the evaluator contract
```

Expected: six `[PASS]` lines, ending with `Checks failed: 0`.

### 5. Run the search comparison

```bash
python scripts/run_search.py --budgets 4 8 12
python scripts/run_search.py --budgets 8 --max-latency 400
```

Prints, for each algorithm and budget: evaluated count, best feasible accuracy
and regret against the exhaustive oracle.

### 6. Generate the tables and figures

```bash
python scripts/make_figures.py
```

Writes to `results/`:

| File | Content |
|---|---|
| `rank_correlation.csv` | Spearman ρ / Kendall τb of every predictor vs measured accuracy, latency, memory |
| `pareto_accuracy_latency.csv` | Non-dominated accuracy/latency configurations |
| `search_regret.csv` | Regret vs budget for random / greedy / heuristic |
| `fig_pareto.png` | Accuracy vs latency with the Pareto front |
| `fig_score_vs_accuracy.png` | Analytic score vs measured accuracy |
| `fig_regret.png` | Regret curves |

---

## Status

| Component | State |
|---|---|
| Repository skeleton | done |
| Configuration space + generator | **done** |
| Analytic capacity score (NSC / Marchenko–Pastur) | **done** |
| Resource cost term + combined score + ablation | **done** |
| Benchmark table + black-box evaluator | **done** |
| Search algorithms (random / greedy / resource-aware / exhaustive) | **done** |
| Metrics and figures (regret, rank correlation, Pareto) | **done** |
| One-command reproduction (`scripts/reproduce.py`) | planned |
| Raspberry Pi benchmark script | planned |

See [`PROGRESS.md`](PROGRESS.md) for the decision log and the detailed,
step-by-step record of what has been built and how.

---

## Reproducibility contract

The repository is organised so that a fresh machine can regenerate the core
results with two commands:

```bash
pip install -r requirements.txt
python scripts/reproduce.py      # (added in a later step)
```

Guarantees we maintain:

- **No hard-coded parameters.** Every searchable factor lives in `configs/`.
- **Deterministic.** Fixed random seeds and a deterministic enumeration order.
- **Archived measurements are committed.** The benchmark table is part of the
  repository, so search results do not require re-running the detector.
- **Hardware is optional.** Raspberry Pi measurements are regenerated by a
  separate, explicitly optional script.

---

## Raspberry Pi (optional hardware validation)

```bash
# On the Raspberry Pi, after installing requirements-benchmark.txt
python scripts/benchmark_pi.py      # (added in a later step)
```

This regenerates the benchmark table with real ARM measurements. It is **not**
required to reproduce the core results.
