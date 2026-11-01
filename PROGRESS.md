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

### Step 3b — Resource cost model and combined analytic score (completed this step)

**Goal.** Complete the "cheap score" so that a configuration can be ranked
*without being run*: the capacity term answers "how capable is this model",
the resource term answers "will it fit the deployment constraints".

**Files created**

| File | Purpose |
|---|---|
| `src/resource_model.py` | Predicts latency and memory from the configuration: MACs from the analytic spec at resolution `R`, a throughput prior per thread, and a memory baseline |
| `src/score.py` | Combines capacity × resolution scaling × quantisation retention, minus resource penalties; supports the ablation modes |
| `scripts/check_score.py` | Enforces the qualitative shape of the model (monotonicity checks) |

**Method.**

1. **Compute cost from the graph.** Every convolution carries its output spatial
   size once an input resolution is given, so

   ```
   MACs(model, R) = Σ H_out·W_out·C_out·C_in·k²
   ```

   Cross-check: `yolov5s @ 640` gives **7.74 GMACs**, and the published figure is
   16.5 GFLOPs = 8.25 GMACs, so the analytic compute is within ~6% of the
   reference — an independent confirmation that the architecture spec is right.

2. **Latency / memory prediction.**

   ```
   latency_ms = MACs / (throughput_per_thread · threads) · 1000 + fixed_overhead
   memory_mb  = weights + peak_activation + runtime_baseline
   ```

   The three constants are **priors**, not measurements: they are documented in
   `ResourceModel` and are refitted by `ResourceModel.fit(...)` once the
   benchmark table exists. The report must state that the analytic numbers are
   predictions until calibrated.

3. **Combined score** (the project's adaptation of the Topic C score):

   ```
   A_hat(x) = C(model) · (R / 640)^alpha · q(precision)
   S(x)     = A_hat(x)/max A_hat − lam·P_latency(x) − mu·P_memory(x) − soft·(…) 
   ```

   * `alpha = 1.0` — a larger input resolution feeds the same architecture more
     spatial evidence, so the usable capacity grows with `R`.
   * `q(int8) = 0.97` — a small retention penalty for quantisation.
   * `P_*` are zero inside a hard constraint and grow linearly outside it.
   * a small `soft_weight` breaks ties toward cheaper configurations when no hard
     cap is active.

4. **Ablation switch.** `ScoreWeights.ablation ∈ {full, no_resource,
   no_capacity}` — the same scorer can be run with the resource ingredient
   removed or the capacity ingredient removed, which is the ablation the topic
   requires.

**Commands run**

```bash
python scripts/check_score.py
```

**Results** (4 threads, FP32)

```text
config_id                  GMACs   lat(ms)   mem(MB)       proxy    score
yolov5n_r320_fp32_t4        0.53      41.1      57.9      273.74   0.0342
yolov5n_r640_fp32_t4        2.12     140.3      62.9      547.48   0.0871
yolov5s_r320_fp32_t4        1.94     129.0      77.5      950.83   0.1638
yolov5s_r640_fp32_t4        7.74     492.1      87.3     1901.66   0.3501
yolov5m_r320_fp32_t4        5.70     364.0     123.1     2462.67   0.4522
yolov5m_r640_fp32_t4       22.78    1432.0     137.9     4925.35   0.9374

[PASS] latency falls with more threads (1944.2 -> 492.1 ms)
[PASS] latency rises with resolution (129.0 -> 492.1 ms)
[PASS] INT8 lowers memory (87.3 -> 59.3 MB)
[PASS] score rises with model size (0.0871 -> 0.9374)
[PASS] score rises with resolution (0.1638 -> 0.3501)
[PASS] resource penalty lowers the score of the heaviest config
```

**Acceptance criteria met.** The score is computable for all 54 configurations
from specification alone, is monotone in the directions the modelling assumes,
and exposes an ablation switch. The predicted latency for `yolov5s @ 640` with 4
threads (≈490 ms) is in the right order of magnitude for a Raspberry Pi 4, which
makes the priors usable for the experiments before calibration.

---

### Step 4 — Benchmark table contract and black-box evaluator (completed this step)

**Goal.** Turn the archived measurements into a strict interface with three
guarantees, so that the search algorithms cannot accidentally read an answer
they have not paid for.

**Files created**

| File | Purpose |
|---|---|
| `benchmarks/README.md` | The table schema and the rules that make it trustworthy |
| `src/benchmark_table.py` | Loads and validates a table (required columns, duplicate ids, negative values, ids outside the search space) |
| `src/evaluator.py` | Budgeted black box: `evaluate()`, `can_evaluate()`, query log, `NotMeasuredError`, `BudgetExhausted`, `is_feasible()` |
| `scripts/make_synthetic_benchmark.py` | Generates a clearly-labelled placeholder table so the pipeline can be developed before the hardware run |
| `scripts/try_evaluator.py` | Encodes the evaluator contract as executable checks |

**Method — the three guarantees.**

1. **No peeking.** `evaluate()` on a configuration with no archived row raises
   `NotMeasuredError` instead of interpolating or guessing.
2. **Honest budget.** The first query for a configuration consumes one unit; a
   repeat is free because the outcome is already known. The budget therefore
   counts *distinct* evaluations, which is exactly the quantity the report
   compares across algorithms.
3. **Determinism.** The answer depends only on the table, never on call order.

**Synthetic ground truth is deliberately not the predictor.** The accuracy curve
uses a different functional form from the analytic score, and the predicted
latency and memory are multiplied by a per-variant bias and log-normal noise. If
the placeholder matched the predictor exactly, the "does the heuristic predict
reality?" analysis would be trivially perfect and would hide bugs.

**Commands run**

```bash
python scripts/make_synthetic_benchmark.py
python scripts/try_evaluator.py
```

**Results**

```text
wrote 54 synthetic rows to benchmarks/synthetic_demo.csv
           config_id  accuracy_map50  latency_median_ms  memory_mb
yolov5n_r320_fp32_t1          0.3214             134.08      57.56
yolov5n_r320_fp32_t4          0.3214              41.64      60.75
yolov5n_r320_int8_t4          0.3165              37.61      50.59

table      : synthetic_demo.csv  (54 rows, sources={'synthetic': 54})
budget     : 5
[PASS] measured configuration returns a row
[PASS] budget spent after 5 distinct queries
[PASS] re-querying a known configuration is free
[PASS] exceeding the budget raises BudgetExhausted
[PASS] unmeasured configuration raises NotMeasuredError
[PASS] query log matches distinct evaluations
Checks failed: 0
```

**Acceptance criteria met.** The evaluator enforces the budget and the "no
peeking" rule, the table loader rejects malformed or foreign rows, and a full
54-row placeholder table exists so the search and metrics stages can be built
and tested immediately. `synthetic_demo.csv` is committed for convenience but is
labelled `source=synthetic`; every result derived from it is a pipeline test, not
a finding.

---

### Step 5 — Search algorithms (completed this step)

**Goal.** Implement the four methods compared in the report, all sharing one
budget accounting so the comparison is fair.

**Files created**

| File | Purpose |
|---|---|
| `src/search.py` | `SearchResult`, neighbour graph, and the four algorithms |
| `scripts/run_search.py` | Runs every algorithm at several budgets against a table and prints best accuracy + regret |

**Method — the problem stated as a search problem.**

```
state        = a deployment configuration
successors   = configurations differing in exactly one factor (one step in the grid)
cost         = one evaluation from the budget
goal         = a feasible configuration with high accuracy
```

The adjacency is derived from the pool itself (a full Cartesian product), so the
neighbour graph cannot drift away from `configs/search_space.yaml`.

| Algorithm | Rule for choosing the next configuration |
|---|---|
| `random` | uniform sample without replacement (black-box baseline) |
| `greedy` | evaluate the neighbours of the current point, move to the best **measured** one; restart when stuck in a local optimum |
| `heuristic` | best-first over the frontier, ordered by the **analytic score** (this project's method, and the direct analogue of the Topic C protocol) |
| `exhaustive` | evaluate everything — the oracle |

**Why the heuristic cannot cheat.** Its ordering comes only from
`Scorer.score`, which reads the configuration specification. It never inspects a
measurement for a configuration it has not evaluated; the evaluator would raise
`NotMeasuredError` if it tried.

**Budget accounting.** Each run gets a *fresh* `Evaluator`, so budgets never leak
between runs, and `n_evaluated` is reported next to the accuracy so a method can
never look better by simply spending more.

**Commands run**

```bash
python scripts/make_synthetic_benchmark.py --seed 0
python scripts/run_search.py --budgets 4 8 12
python scripts/run_search.py --budgets 8 --max-latency 400
```

**Results** (synthetic table — pipeline test only)

```text
oracle: best feasible accuracy = 0.5503

algorithm     budget  evaluated   best acc    regret  feasible
random             4          4     0.4561    0.0942         4
greedy             4          4     0.4306    0.1197         4
heuristic          4          4     0.5503    0.0000         4
random             8          8     0.5110    0.0393         8
greedy             8          8     0.5110    0.0393         8
heuristic          8          8     0.5503    0.0000         8
random            12         12     0.5206    0.0297        12
greedy            12         12     0.5353    0.0150        12
heuristic         12         12     0.5503    0.0000        12
exhaustive       all         54     0.5503    0.0000        54
```

With a 400 ms latency cap the picture changes, because feasibility starts to
bite: at budget 8 the heuristic reaches 0.4223 with 4 feasible configurations
found, against 3 for random and 2 for greedy.

**Note on the synthetic numbers.** They are *not* findings. The placeholder truth
is a smooth function of the same factors the score uses, so the heuristic looks
strong here by construction. Adding multiplicative measurement noise (default
3%) keeps the test honest about noise but cannot reproduce the real
rank-disagreements that make the question interesting — only the hardware table
can.

**Acceptance criteria met.** All four algorithms run under identical budgets,
produce comparable `SearchResult` objects, and report evaluated count, best
feasible accuracy, regret and feasibility count.

---

### Step 6 — Metrics, tables and figures (completed this step)

**Goal.** Answer the two questions the course topic asks of a cheap score:
does it **rank** candidates well, and does it **select** well under a budget?
Plus the ablation the topic requires.

**Files created**

| File | Purpose |
|---|---|
| `src/metrics.py` | `spearman`, `kendall_tau_b` (with tie handling), `regret`, `normalized_regret`, `pareto_front`, `within_tolerance_groups` — NumPy only, no SciPy |
| `scripts/make_figures.py` | Builds every table and figure from a benchmark table |

**Method.**

1. **Rank correlation.** Every cheap signal — the full score, the two ablations,
   the capacity term alone, the predicted MACs, latency and memory — is
   correlated with the **measured** accuracy, latency and memory using
   Spearman's ρ and Kendall's τb. Ties are handled with average ranks (ρ) and
   the tau-b tie correction, which matters because configurations share the same
   accuracy whenever only the thread count changes.
2. **Pareto front.** Non-dominated points under (maximise accuracy, minimise
   latency), computed by pairwise domination — 54 configurations make an O(n²)
   scan cheaper and clearer than a library call.
3. **Regret curve.** Each algorithm is re-run for budgets 1…12 on a fresh
   evaluator, and the gap to the exhaustive oracle is plotted.
4. **Ablation.** `full` vs `no_resource` vs `no_capacity` are correlated side by
   side, so the report can attribute the signal to an ingredient rather than to
   the score as a whole.

**Commands run**

```bash
python scripts/make_figures.py
python scripts/make_figures.py --max-latency 400
```

**Results** (synthetic table — pipeline test only)

```text
        predictor  spearman_vs_accuracy  kendall_vs_accuracy  spearman_vs_latency
        pred_macs                0.8799               0.6917               0.8877
  pred_latency_ms                0.7651               0.5796               0.9967
score_no_resource                0.7465               0.5720               0.8752
       score_full                0.7457               0.5556               0.8300
   pred_memory_mb                0.5507               0.3968               0.6730
    capacity_only                0.5122               0.4019               0.7873
score_no_capacity               -0.6838              -0.5010              -0.8958

Pareto front (accuracy vs latency): 12 points
```

**Reading the table.** The negative `score_no_capacity` is the expected
direction: with the capacity ingredient removed the score only prefers cheap
configurations, which are exactly the least accurate ones. The capacity term
therefore carries the accuracy signal, and the resource term carries the
feasibility signal — which is the claim the ablation is supposed to test.

**An honest observation to carry into the real run.** Under a tight latency cap
(400 ms) on the synthetic table, `score_full`'s correlation with accuracy turns
**negative** (−0.42): once nearly every configuration violates the cap, the
penalty dominates and the score ranks by *violation size* rather than by
accuracy. That is a property of the penalty form, not of the data, and the
report should either (a) shrink the penalty scale near the constraint boundary,
or (b) evaluate the score only on the feasible subset. This is recorded as an
open issue rather than hidden.

**Outputs written** (regenerated, not committed — `results/` is ignored)

```text
results/rank_correlation.csv
results/pareto_accuracy_latency.csv
results/search_regret.csv
results/fig_pareto.png
results/fig_score_vs_accuracy.png
results/fig_regret.png
```

**Acceptance criteria met.** All four evaluation artefacts the topic asks for
exist — ranking correlation with ties, fixed-budget selection with regret,
Pareto front, and an ablation table — and any of them can be regenerated from a
benchmark table with one command.

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
│   ├── README.md                   <- table schema and rules
│   └── synthetic_demo.csv          <- placeholder for pipeline testing only
├── models/                         <- model download/export instructions
├── results/                        <- generated tables and figures
├── scripts/                        <- runnable entry points
│   ├── check_capacity.py           <- capacity validation + variant table
│   ├── check_score.py              <- score/resource-model monotonicity checks
│   ├── make_synthetic_benchmark.py <- placeholder table generator
│   ├── try_evaluator.py            <- evaluator contract checks
│   ├── run_search.py               <- four-algorithm comparison
│   └── make_figures.py             <- ranking / Pareto / regret tables and plots
└── src/
    ├── __init__.py
    ├── search_space.py             <- pool enumeration
    ├── model_spec.py               <- analytic YOLOv5n/s/m architecture spec
    ├── nsc.py                      <- Marchenko–Pastur capacity score
    ├── resource_model.py           <- latency / memory prediction
    ├── score.py                    <- combined analytic score + ablation
    ├── benchmark_table.py          <- table loading and validation
    ├── evaluator.py                <- budgeted black-box evaluator
    ├── search.py                   <- random / greedy / heuristic / exhaustive
    └── metrics.py                  <- spearman, kendall, regret, pareto
```

---

## 5. Next Steps

| Step | Task | Status |
|---|---|---|
| S2 | Configuration search space + generator | **done** |
| S3 | Analytic scoring model — NSC capacity term | **done** |
| S3b | Resource cost term + combined score + ablation | **done** |
| S4 | Benchmark table schema + black-box evaluator | **done** |
| S5 | Search algorithms: random, greedy, resource-aware heuristic, exhaustive oracle | **done** |
| S6 | Metrics & figures: regret, Spearman/Kendall ranking, Pareto fronts, ablation | **done** |
| S7 | One-command reproduction (`scripts/reproduce.py`) | next |
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
