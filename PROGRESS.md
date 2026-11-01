# Project Progress Log

**Project:** Resource-Aware Heuristic Search for Edge Vision Deployment Configuration Selection
**Course:** CS5491 Artificial Intelligence — Course Project
**Repository:** https://github.com/Gene5Wu/rpi-edge-search

This file records *what* was done, *why*, and *how* (methods used), step by step.
It is updated after every completed step so the project can be resumed at any time.

---

## 1. Project Definition

**Research question.** Given a limited number of configuration evaluations, can a
resource-aware heuristic identify a high-accuracy edge-vision deployment
configuration that satisfies latency and memory constraints, without exhaustively
benchmarking every candidate?

**Two-layer design.**

| Layer | What it does | Who can reproduce it |
|---|---|---|
| **Core (laptop)** | Search + baselines + evaluation run against an archived **benchmark table** (`benchmarks/pi4b_2gb_reference.csv`). Each row of the table is a black-box evaluation result. | Anyone, no Raspberry Pi, no GPU |
| **Hardware validation (Raspberry Pi 4B, 2 GB)** | Regenerates the benchmark table with real measurements and validates that the selected configurations behave as predicted. | Only with a Pi (optional) |

**Methodology borrowed from the course topic _Neural Spectral Capacity as a
Heuristic for Architecture Design_ (Topic C).** The topic studies a cheap,
*analytic* score computed from a network specification alone, then asks whether
that score ranks and selects trained architectures well under a fixed budget.
We transfer that experimental protocol to deployment configurations:

| Topic C | This project |
|---|---|
| Analytic capacity score from the architecture spec | Analytic score from the configuration spec (architecture capacity **+** resource cost) |
| Rank correlation vs. trained perplexity (Spearman ρ, Kendall τb) | Rank correlation vs. measured accuracy / latency / memory |
| Fixed-budget selection under a parameter cap; regret | Fixed-budget selection under latency/memory caps; regret vs. exhaustive oracle |
| Ablation removing the new mechanism | Ablations removing the capacity term / the resource term |
| Size-controlled comparisons | Size-controlled comparisons (same resolution, similar parameter count) |

> **Why model variants are part of the space.** NSC is an *architecture* score.
> If every candidate shared one architecture, the score would be constant and
> could not rank anything. The space therefore spans three YOLOv5 variants
> (n / s / m), so the capacity term has real signal, while resolution,
> precision and thread count drive the resource side.

---

## 2. Decision Log

| # | Decision | Value | Rationale |
|---|---|---|---|
| D1 | Detector family | YOLOv5 (**n, s, m**) | Matches the submitted proposal; cheap; three variants give the analytic capacity score real ranking signal |
| D2 | Inference backend | ONNX Runtime (CPU) | Mature, reproducible on laptop and Pi, no custom toolchain |
| D3 | Search-space factors | model × resolution (320/416/640) × precision (fp32/int8) × threads (1/2/4) | Small enough for an exhaustive oracle, large enough that budgeted search is non-trivial |
| D4 | Pool size | **54** configurations (27 FP32) | 3 × 3 × 2 × 3 |
| D5 | Dataset | Fixed COCO 2017 val subset, 500 images, same images for every config | Identical evaluation pipeline for every candidate |
| D6 | Metrics | detection accuracy, median+p95 latency, peak memory | Core trade-off axes |
| D7 | Baselines | Random search, greedy/local search, proposed resource-aware heuristic; exhaustive = oracle | Matches the proposal and Topic C's protocol |
| D8 | Topic C integration | Methodology + analytic scoring (option B/1) | Directly comparable ranking/selection protocol |
| D9 | Reproducibility | Archived benchmark CSV is committed; Pi is optional | A reviewer must be able to rerun everything on a laptop |
| D10 | Language / stack | English docs & code, Python 3.11 | Course requirement |

---

## 3. Step-by-step Log

### Step 0 — Requirements consolidation (this session)

**What.** Read the two ChatGPT design conversations, the one-page proposal
(`proposal/proposal.tex`) and the course topic sheet
(`topic-neural-spectral-capacity.pdf`); extracted the frozen experimental design.

**Method.** Manual review of 18.5k lines of design conversation, plus PDF text
extraction (`pdftotext -layout`) for the topic sheet.

**Outcome.** Confirmed scope: YOLOv5 family + ONNX Runtime, 54-configuration
space, benchmark-table-driven core experiment, Pi as optional validation.
Resolved the open design question about how Topic C's scoring method applies
(see D8 and the table in §1).

---

### Step 1 — Repository bootstrap (completed previously)

`README.md` plus the empty directory skeleton (`configs/`, `benchmarks/`, `src/`,
`scripts/`, `models/`, `results/`) and the initial commits.

---

### Step 2 — Configuration search-space generator (completed this step)

**Goal.** Make the candidate pool a *single source of truth*, so that the
evaluator and every search algorithm read the same definition and no parameter
is hard-coded in Python.

**Files created**

| File | Purpose |
|---|---|
| `configs/search_space.yaml` | Declarative definition of models, resolutions, precisions, threads, backend, evaluation defaults |
| `src/search_space.py` | Loads the YAML, validates it, enumerates the Cartesian product, filters, emits JSON/CSV |
| `src/__init__.py` | Makes `src` a package |
| `requirements.txt` | Core deps only (numpy, pandas, PyYAML, matplotlib) |
| `requirements-benchmark.txt` | Optional deps for regenerating hardware measurements |
| `.gitignore` | Keeps data/models/results out of Git; benchmark CSV stays tracked |

**Method.**

1. Declarative YAML spec instead of Python constants — the space is an object of
   study, so it must be versioned and diffable.
2. Deterministic enumeration order (model → resolution → precision → threads)
   so that tie-breaking in search is reproducible across runs.
3. Stable, human-readable `config_id`, e.g. `yolov5n_r320_fp32_t4`, used as the
   join key between the search space, the benchmark table and the results.
4. Fail-fast validation: duplicate ids, empty lists, non-positive resolutions
   and threads are rejected at load time.
5. Self-test mode (`--self-test`) asserts pool size equals
   `|models| × |res| × |precision| × |threads|` and that all ids are unique.

**Commands run**

```bash
python src/search_space.py --self-test
python src/search_space.py --stats
python src/search_space.py --precision fp32 --stats
```

**Results**

```
self-test passed: 54 unique configurations

models       : ['yolov5n', 'yolov5s', 'yolov5m']
resolution   : [320, 416, 640]
precision    : ['fp32', 'int8']
threads      : [1, 2, 4]
backend      : onnxruntime
total configs: 54
per model    : {'yolov5n': 18, 'yolov5s': 18, 'yolov5m': 18}
per precision: {'fp32': 27, 'int8': 27}
```

**Acceptance criteria met.** The generator runs on a clean Python install with
only PyYAML, produces 54 unique configurations, and is deterministic.

---

### Step 3 — NSC-style analytic capacity score (completed this step)

**Goal.** Implement the *Topic C* core: a cheap, **specification-only** score
that estimates how much "capacity" a candidate architecture has, so that
candidates can be ranked **without running them**.

**Files created**

| File | Purpose |
|---|---|
| `src/model_spec.py` | Builds the YOLOv5n/s/m architecture **analytically** from the shared topology plus the official `depth_multiple` / `width_multiple`; yields every convolution weight matrix as `(m, n) = (out_channels, in_channels·k²)` |
| `src/nsc.py` | `psi_mp(m, n, s)` — the Marchenko–Pastur expectation of `ln det(I + WᵀW)` — plus `capacity()` which sums it over the matrices |
| `scripts/check_capacity.py` | Monte-Carlo validation of `psi_mp`, and the per-variant capacity table |

**Method.**

1. **Architecture spec without PyTorch.** The three variants share one topology
   and differ only by width/depth multipliers, so the specification is generated
   with the reference scaling rules (`make_divisible(c·w, 8)` for channels,
   `max(round(n·d), 1)` for repeats). This keeps the *core* experiment free of
   any deep-learning dependency.
2. **MP-law capacity.** For `W ∈ R^{m×n}` with i.i.d. `N(0, s²)` entries,
   `W = sZ` and the eigenvalues of `ZᵀZ` scale with `max(m,n)`, giving

   ```
   E[ψ(W)] = min(m,n) · ∫ ln(1 + s²·max(m,n)·λ) dMP_γ(λ),   γ = min/max.
   ```

3. **Singularity-safe quadrature.** The MP density has an inverse-square-root
   singularity at its lower edge, which is at zero for (near-)square matrices. A
   uniform grid in `λ` silently underestimates the integral there (the 32×32
   check was off by 99%). The substitution `λ = a + (b−a)t²` removes the
   singularity; the quadrature is then accurate everywhere.

**Commands run**

```bash
python scripts/check_capacity.py --samples 150
```

**Results**

```text
Numerical check: psi_MP vs exact Monte-Carlo estimate (s = 0.02)
     shape (m x n)      MC mean       psi_MP  rel. error  status
          64 x 576      13.1386      13.1462      0.06%  OK
        128 x 1152      47.7979      47.8110      0.03%  OK
        256 x 2304     163.9741     163.9702      0.00%  OK
           32 x 32       0.4077       0.4045      0.80%  OK
         512 x 256      45.9192      45.9222      0.01%  OK

Architecture capacity by variant
variant     conv matrices   params (M)   ref. (M)      capacity
yolov5n                55       1.5765        1.9      547.4796
yolov5s                55       6.0534        7.2     1901.6585
yolov5m                75      17.0564       21.2     4925.3482
```

**Acceptance criteria met.** The analytic estimate matches brute-force
Monte-Carlo within 0.1% for the layer shapes that occur in the models, and the
capacity ordering follows the model sizes (n < s < m), so the score carries real
ranking signal across the three variants.

**Known limitation (tracked).** The analytic spec reproduces ~82% of the
published parameter counts, because it models the convolution stack and treats
the detection head as a single 1×1 convolution per level; the rest of the
reference count comes from the head's extra convolutions and the
non-convolution parameters. This is a *constant* offset in the same direction
for all three variants, so it does not disturb their ordering. It will be
reconciled exactly once the ONNX exports are available in the benchmark stage,
where the true parameter count can be read from the exported graph.

---

## 4. Repository Map (current)

```text
rpi-edge-search/
├── README.md
├── PROGRESS.md                     <- this file
├── requirements.txt
├── requirements-benchmark.txt
├── .gitignore
├── configs/
│   └── search_space.yaml           <- single source of truth for the pool
├── benchmarks/                     <- archived benchmark tables (CSV)
├── models/                         <- model download/export instructions
├── results/                        <- generated tables and figures
├── scripts/                        <- runnable entry points
│   └── check_capacity.py           <- capacity validation + variant table
└── src/
    ├── __init__.py
    ├── search_space.py             <- pool enumeration
    ├── model_spec.py               <- analytic YOLOv5n/s/m architecture spec
    └── nsc.py                      <- Marchenko–Pastur capacity score
```

---

## 5. Next Steps

| Step | Task | Status |
|---|---|---|
| S2 | Configuration search space + generator | **done** |
| S3 | Analytic scoring model — NSC capacity term | **done** |
| S3b | Resource cost term (latency/memory prediction from resolution/precision/threads) and the combined score | next |
| S4 | Benchmark table schema + black-box evaluator (refuses to reveal unmeasured rows) | planned |
| S5 | Search algorithms: random, greedy, resource-aware heuristic, exhaustive oracle | planned |
| S6 | Metrics & figures: regret, Spearman/Kendall ranking, Pareto fronts, ablation | planned |
| S7 | One-command reproduction (`scripts/reproduce.py`) | planned |
| S8 | Raspberry Pi benchmark script (`scripts/benchmark_pi.py`) | planned |
| S9 | Four-page final report + ablation study | planned |

## 6. Open Questions

1. **INT8 support.** ONNX Runtime INT8 requires a quantisation step and a
   calibration set. Plan: implement FP32 first (27 configs) and treat INT8 as an
   extension, so the core result does not depend on it.
2. **Benchmark cost on the Pi.** 54 configurations × 500 images is heavy. The
   plan is to measure latency on a smaller fixed subset and accuracy on the full
   fixed subset, documenting the split.
3. **Parameter-count reconciliation for the capacity term.** The analytic
   architecture spec gives ~82% of the published YOLOv5 parameter counts (the
   detection head is simplified). Reconcile exactly against the ONNX graphs in
   the benchmark stage and, if the offset turns out to be non-uniform across
   variants, recalibrate the width/depth mapping.
4. **Pending push.** The commits for Step 2 and Step 3 are local only; the
   network to `github.com` was unreachable during this session. Push when the
   connection is back (no further code change is required).
