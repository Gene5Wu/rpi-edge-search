# Resource-Aware Heuristic Search for Edge Vision Deployment Configuration Selection

CS5491 Artificial Intelligence — Course Project

**Research question.** Given a limited number of configuration evaluations, can an
analytic resource-aware score select better edge-vision deployment
configurations than random or purely data-driven local search?

---

## Overview

Deploying a vision model on an edge device is a trade-off between detection
accuracy, inference latency and memory. The trade-off point is set by the
**deployment configuration**: detector variant, input resolution, numerical
precision and CPU thread count.

This project treats configuration selection as a **resource-constrained search
problem**. Each configuration is an expensive black-box query; only a fixed
number can be afforded; the goal is a feasible configuration with high accuracy.

### Two-layer design

| Layer | What it does | Requirement |
|---|---|---|
| **Core — laptop** | Analytic score, search algorithms, metrics and figures run against an **archived benchmark table**. | Python + 4 packages. No GPU, no Pi, no model download. |
| **Hardware validation — Raspberry Pi 4B / 2 GB** | Regenerates the benchmark table with real measurements. | A Raspberry Pi (optional). |

A reviewer reproduces **every table and figure** on a fresh machine; the Pi only
regenerates the measurements.

### Methodological reference

The protocol follows the course topic *Neural Spectral Capacity*: a cheap,
specification-only score ranks candidates, and is then evaluated against measured
outcomes.

| Topic C | This project |
|---|---|
| Analytic capacity score from the architecture spec | Analytic score from the configuration spec (capacity + resource cost) |
| Spearman ρ / Kendall τb vs. trained perplexity | Spearman ρ / Kendall τb vs. measured accuracy |
| Fixed-budget selection under a parameter cap; regret | Fixed-budget selection under latency/memory caps; regret vs. exhaustive oracle |
| Ablation removing the new mechanism | Two ablations: no capacity term, no resource term |

---

## Repository structure

```text
rpi-edge-search/
├── README.md                     # this file
├── PROGRESS.md                   # decision log and step-by-step build record
├── requirements.txt              # core deps (laptop reproduction)
├── requirements-lock.txt         # exact tested versions
├── requirements-benchmark.txt    # optional deps (regenerate measurements)
├── configs/
│   └── search_space.yaml         # single source of truth for the pool
├── benchmarks/
│   ├── README.md                 # table schema and rules
│   ├── synthetic_demo.csv        # placeholder, pipeline testing only
│   ├── laptop_fp32.csv           # MEASURED: 18 configs, COCO mAP + latency
│   └── pi4b_2gb_reference.csv    # Raspberry Pi measurements (later)
├── models/
│   ├── README.md
│   └── download_models.py        # optional ONNX export
├── report/
│   ├── final_report.tex
│   └── final_report.pdf
├── results/                      # generated tables and figures
├── scripts/
│   ├── reproduce.py              # one command: everything
│   ├── check_capacity.py
│   ├── check_score.py
│   ├── check_model_spec.py       # spec vs published params / GFLOPs
│   ├── try_evaluator.py
│   ├── run_search.py
│   ├── run_multiseed.py          # mean/std/feasible-success over 30 seeds
│   ├── compare_proxies.py        # NSC vs MACs vs params as accuracy proxy
│   ├── make_figures.py
│   ├── make_synthetic_benchmark.py
│   ├── prepare_coco_subset.py     # fixed COCO val subset
│   ├── evaluate_accuracy.py       # mAP via pycocotools
│   └── benchmark_pi.py           # optional hardware step
└── src/
    ├── search_space.py           # pool enumeration
    ├── model_spec.py             # analytic YOLOv5n/s/m architecture spec
    ├── nsc.py                    # Marchenko–Pastur capacity score
    ├── resource_model.py         # latency / memory prediction
    ├── score.py                  # combined analytic score + ablation
    ├── benchmark_table.py        # table loading and validation
    ├── evaluator.py              # budgeted black-box evaluator
    ├── search.py                 # random / greedy / heuristic / exhaustive
    └── metrics.py                # spearman, kendall, regret, pareto
```

---

## Environment setup

**Requires Python 3.11+** (tested on 3.11.7; the pinned NumPy needs 3.11).

Use a virtual environment. It is not strictly necessary — if your global Python
already has the four packages the scripts will run — but it is what makes the
results reproducible: it proves the code does not silently depend on a package
that `requirements.txt` forgets to declare.

### 1. Create and activate

```bash
git clone https://github.com/Gene5Wu/rpi-edge-search.git
cd rpi-edge-search

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows Git Bash
source .venv/Scripts/activate
# Linux / macOS
source .venv/bin/activate
```

If PowerShell refuses to run the activation script, either use
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current
shell only, or skip activation entirely and call the interpreter directly:
`.\.venv\Scripts\python.exe scripts\reproduce.py`.

### 2. Install dependencies

```bash
pip install -r requirements.txt        # loose lower bounds, quick start
# or exact tested versions:
pip install -r requirements-lock.txt
```

### 3. Verify

```bash
python -c "import numpy, pandas, yaml, matplotlib; print('ok')"
python scripts/reproduce.py
```

The second command runs the whole pipeline and ends with `overall: PASS`.

`requirements-benchmark.txt` (ONNX Runtime, OpenCV, pycocotools) is **separate on
purpose** and only needed to regenerate measurements on hardware. Put it in a
second environment so the core one stays minimal:

```bash
python -m venv .venv-bench
# activate it, then:
pip install -r requirements.txt -r requirements-benchmark.txt
```

Both `.venv/` and `.venv-bench/` are git-ignored.

---

## Reproducing everything

With the environment from the previous section active, a single command runs the
whole pipeline:

```bash
python scripts/reproduce.py
```

This runs, in order:

| Step | What it checks |
|---|---|
| 1 | configuration-space self-test (54 unique configurations) |
| 2 | Marchenko–Pastur capacity against a Monte-Carlo estimate |
| 3 | score and resource-model monotonicity |
| 4 | architecture spec vs published params / GFLOPs |
| 5 | evaluator contract (budget, no peeking) |
| 6 | four-algorithm search comparison |
| 7 | rank correlation, Pareto front, regret curve |
| 8 | multi-seed comparison (30 seeds) |

It writes `results/reproduce_summary.txt` with the status and timing of every
step. Use `--quick` for a faster capacity check, and
`--table benchmarks/pi4b_2gb_reference.csv` to run against real measurements.

---

## Running the steps individually

```bash
python src/search_space.py --stats               # list the pool
python scripts/check_capacity.py                 # capacity + Monte-Carlo check
python scripts/check_score.py                    # score shape checks
python scripts/check_model_spec.py               # spec vs published params / GFLOPs
python scripts/try_evaluator.py                  # evaluator contract
python scripts/run_search.py --budgets 4 8 12    # algorithm comparison
python scripts/make_figures.py                   # tables and figures -> results/
python scripts/run_multiseed.py --seeds 30    # multi-seed distribution
```

Add constraints to any of them, e.g.

```bash
python scripts/run_search.py --budgets 8 --max-latency 400 --max-memory 120
```

---

## Configuration space

Defined in `configs/search_space.yaml`; nothing is hard-coded in Python.

| Factor | Values |
|---|---|
| Model | `yolov5n`, `yolov5s`, `yolov5m` |
| Input resolution | 320, 416, 640 |
| Numerical precision | `fp32`, `int8` |
| CPU threads | 1, 2, 4 |
| Backend | ONNX Runtime (CPU) |

**54** configurations (27 when restricted to FP32). Each has a stable id such as
`yolov5n_r320_fp32_t4`, the join key between the space, the benchmark table and
the results.

Three model variants are included on purpose: the capacity score is an
*architecture* quantity, so a single architecture would make it constant and
unable to rank anything.

---

## Outputs

`scripts/make_figures.py` writes to `results/`:

| File | Content |
|---|---|
| `rank_correlation.csv` | Spearman ρ / Kendall τb of each predictor vs accuracy, latency, memory — both over the full pool and **restricted to feasible configurations** |
| `pareto_accuracy_latency.csv` | Non-dominated configurations |
| `search_regret.csv` | Regret vs budget for every algorithm |
| `fig_pareto.png` | Accuracy vs latency with the Pareto front |
| `fig_score_vs_accuracy.png` | Analytic score vs measured accuracy |
| `fig_regret.png` | Regret curves |
| `reproduce_summary.txt` | Status and timing of every pipeline step |

`results/` is regenerated and not committed.

---

## Raspberry Pi (optional)

Only needed to regenerate the measurements; the core results never touch a model.

```bash
# 1. export ONNX (heavy, optional dependencies)
pip install ultralytics
python models/download_models.py --variants yolov5n yolov5s yolov5m

# 2. measure on the Pi
pip install -r requirements-benchmark.txt
python scripts/benchmark_pi.py --dry-run          # check the plan
python scripts/benchmark_pi.py \
    --images-dir data/coco_subset/val \
    --accuracy-json data/accuracy_map50.json \
    --out benchmarks/pi4b_2gb_reference.csv

# 3. re-run the analysis on the real table
python scripts/reproduce.py --table benchmarks/pi4b_2gb_reference.csv
```

`benchmark_pi.py` measures latency and peak memory. Detection accuracy needs the
full COCO evaluation pipeline, so it is supplied through `--accuracy-json`
mapping `"<model>|<resolution>|<precision>"` to mAP@0.5.

Two details are deliberate:

- **One subprocess per configuration.** `ru_maxrss` is a process-lifetime peak, so
  measuring several configurations in one process would let a large model inflate
  every reading after it. Each configuration gets a fresh process.
- **The `source` column is detected, not assumed.** A laptop run is tagged
  `laptop`; only a real Pi run is tagged `raspberry_pi_4b`. Override with
  `--source` if the detection is wrong on your board.

The weights are the **classic YOLOv5 v7.0** checkpoints (`yolov5n.pt`,
`yolov5s.pt`, `yolov5m.pt`), which is the topology `src/model_spec.py` scores.
The newer `yolov5nu/su/mu.pt` files are the anchor-free YOLOv5u family and are
**not** interchangeable here: scoring one model and benchmarking another would
invalidate the capacity term. `scripts/check_model_spec.py` fails if an
exported ONNX graph disagrees with the analytic parameter count.

An FP32-only table is a valid benchmark:

```bash
python scripts/benchmark_pi.py --precisions fp32 --source laptop \
    --images-dir data/coco_subset/val --accuracy-json data/accuracy_map50.json \
    --out benchmarks/laptop_fp32.csv
python scripts/reproduce.py --table benchmarks/laptop_fp32.csv --precisions fp32
```

---

## Reports

| File | Purpose |
|---|---|
| `report/milestone_report.pdf` | Three-page milestone report (Times New Roman 12 pt, 2.5 cm margins, single spacing) |
| `report/final_report.pdf` | Four-page final report draft |

Both are built from the matching `.tex` with `xelatex`.

**The numerical tables in both documents come from the synthetic placeholder
table** and are labelled as such. Replace them with measured results before
submission.

---

## Status

| Component | State |
|---|---|
| Configuration space + generator | done |
| Analytic capacity score (NSC / Marchenko–Pastur) | done |
| Resource cost model + combined score + ablation | done |
| Benchmark table contract + black-box evaluator | done |
| Search algorithms (random / greedy / heuristic / exhaustive) | done |
| Metrics and figures (regret, rank correlation, Pareto) | done |
| One-command reproduction (8 steps) | done |
| Multi-seed comparison (30 seeds) | done |
| Hardware benchmark script (per-config subprocess, detected source) | done (runs when hardware is attached) |
| Milestone report | done (needs measured numbers) |
| Final report | draft (needs the hardware table) |

See [`PROGRESS.md`](PROGRESS.md) for the decision log and the detailed record of
how each piece was built and verified.

---

## Reproducibility contract

- **No hard-coded parameters.** Every searchable factor lives in `configs/`.
- **Deterministic.** Fixed seeds and a deterministic enumeration order.
- **Archived measurements are committed** (once measured), so the search does not
  need the detector.
- **Hardware is optional.** Pi measurements come from a separate script.
- **Synthetic data is labelled.** Any row with `source=synthetic` is a pipeline
  test, never a result.
