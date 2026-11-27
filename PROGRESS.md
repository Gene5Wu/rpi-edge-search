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

### Step 7 — One-command reproduction (completed this step)

**Goal.** Make the reviewer's job a single command that either passes or fails
loudly.

**Files created**

| File | Purpose |
|---|---|
| `scripts/reproduce.py` | Runs the seven pipeline steps in order, times each, and writes `results/reproduce_summary.txt` |

**Method.** Each step runs as a subprocess with its own exit code, so a failure
stops the run instead of being swallowed; the summary records the Python version,
the table name and its `source=` counts, per-step status and duration, and an
explicit warning when the table is synthetic. `--quick` reduces the Monte-Carlo
sample count, and `--table` switches to the hardware table without touching the
code.

**Command run**

```bash
python scripts/reproduce.py
```

**Results**

```text
step                              status   seconds
search space self-test              PASS       0.1
capacity numerical check            PASS      27.9
score monotonicity checks           PASS       0.4
architecture spec validation        PASS       0.1
evaluator contract checks           PASS       0.9
search comparison                   PASS       1.0
tables and figures                  PASS       1.7
overall   : PASS      total 32.2s
```

**Acceptance criteria met.** One command reproduces every artefact on a fresh
machine in about half a minute, and the summary states plainly that the numbers
are pipeline verification rather than measurements.

---

### Step 8 — Raspberry Pi benchmark script and model export (completed this step)

**Goal.** Provide the optional path that produces real measurements, without
making a Raspberry Pi necessary for anyone else.

**Files created**

| File | Purpose |
|---|---|
| `scripts/benchmark_pi.py` | Measures median/p95 latency and peak process memory per configuration on ARM; portable peak-RSS reader (Linux `getrusage`, Windows `psapi`), `--dry-run` planning mode |
| `models/download_models.py` | Optional YOLOv5 → ONNX export with dynamic axes, so one file serves every resolution |
| `models/README.md` | Explains that models are never committed and what needs them |

**Design decisions.**

* **Accuracy is not measured here.** A fair mAP needs the full COCO evaluation
  pipeline; mixing it into a timing script invites silently wrong numbers. It is
  supplied through `--accuracy-json`, and the script **refuses to write a table**
  without it (unless `--allow-missing-accuracy` is passed, and then the file is
  not a valid benchmark table).
* **Thresholds are explicit.** `--num-images`, `--warmup`, `--repeats` are
  arguments, and the number of timed runs is recorded in the `notes` column.
* **Naming is fixed.** Files must be `models/{model}_{precision}.onnx`, the same
  convention the dry run prints.

**Commands run** (no hardware attached, so only the planning paths were executed)

```bash
python scripts/benchmark_pi.py --dry-run
python models/download_models.py --dry-run
```

**Results**

```text
plan:
  yolov5n_r320_fp32_t1       <- yolov5n_fp32.onnx (MISSING)
  yolov5n_r320_int8_t1       <- yolov5n_int8.onnx (MISSING)
  ... (46 more)
dry run: nothing measured, nothing written.
```

**Acceptance criteria met.** The hardware path exists, is documented, fails
clearly when dependencies or models are missing, and cannot contaminate the
committed table by accident.

---

### Step 9 — Final report (completed as a draft this step)

**Files created**

| File | Purpose |
|---|---|
| `report/final_report.tex` | Four-page report: problem, method, protocol, results, ablation, limitations, reproducibility |
| `report/final_report.pdf` | Compiled, 4 pages |

**Method.** The report opens with a boxed statement that every numerical table
comes from the synthetic placeholder and must be replaced before submission, so
the draft cannot be mistaken for finished results. The results tables are the
pipeline-verification numbers; the method, protocol, ablation design, limitation
analysis and reproducibility statement are final.

**Command run**

```bash
cd report && xelatex -interaction=nonstopmode -halt-on-error final_report.tex
```

**Results.** `final_report.pdf`, 4 pages, no warnings that affect content.

**Acceptance criteria met.** A complete four-page report exists whose structure
and claims match the code; only the hardware numbers are missing.

---

### Step 10 — Specification validation and penalty calibration (completed this step)

Two loose ends from earlier steps, both closed.

**10a. Validating the analytic architecture spec.**

`scripts/check_model_spec.py` compares the spec against the published YOLOv5
reference figures. The result separates two quantities that were previously
conflated:

```text
variant     params(M)   ref(M)   ratio    GMACs   GFLOPs    ref   ratio
yolov5n         1.576     1.90   0.830     2.12     4.23    4.5   0.940
yolov5s         6.053     7.20   0.841     7.74    15.49   16.5   0.939
yolov5m        17.056    21.20   0.805    22.78    45.57   49.0   0.930
```

* **Compute is accurate to within 7%**, and the ratio is nearly identical across
  the three variants (0.930–0.940). Since the latency model is driven by MACs,
  this is the check that matters, and it passes.
* **Parameters are under-counted by ~19%**, uniformly. The detection head is
  modelled as one `1×1` convolution per level. Nothing downstream depends on the
  absolute count: the capacity score is used only for ranking (and is invariant
  to a uniform scale error), and the weights are a small slice of the memory
  estimate, which is dominated by the runtime baseline.

The script also reads true counts from `models/*.onnx` when the optional `onnx`
package and the exports are present, and skips the comparison rather than
guessing when they are not.

**10b. Penalty calibration under a hard constraint.**

Earlier the score ranked *by violation size* once almost every configuration
missed the cap, giving a negative correlation with accuracy. Two changes:

1. **Clamped penalties.** `ScoreWeights.penalty_cap` bounds each violation term
   at 1.0, so missing the cap by 10× no longer contributes a penalty of 10 that
   swamps the accuracy term.
2. **The right statistic.** `make_figures.py` now also reports the correlation
   **restricted to the feasible subset**, which is the population the score is
   actually asked to rank.

**Results at a 400 ms cap** (22 feasible of 54):

```text
        predictor  spearman_all  spearman_feasible_only
        pred_macs          0.880                  0.853
  pred_latency_ms          0.765                  0.590
score_no_resource          0.747                  0.641
       score_full         -0.285                  0.656
score_no_capacity         -0.731                 -0.373
```

**Interpretation.** On the full pool the score's correlation with raw accuracy
is *negative by design*: it ranks infeasible configurations last, while accuracy
does not care about the cap. Restricted to deployable configurations the score
recovers ρ = 0.66, and it beats the resource-free variant (0.641), which is the
small piece of evidence the method needs. The honest reporting rule is therefore:

> Report rank correlation on the feasible subset, and compare selection quality
> with regret — the global correlation is not a meaningful statistic once a hard
> constraint binds.

The clamping moved the full-pool correlation from −0.42 to −0.29 **without
changing `penalty_cap`'s default of 1.0 into a tuned parameter**; it is a
documented modelling choice, not a fitted one.

**Commands run**

```bash
python scripts/check_model_spec.py
python scripts/make_figures.py
python scripts/make_figures.py --max-latency 400 --out results/cap400
```

---

### Step 11 — Environment isolation and dependency pinning (completed this step)

**Goal.** Make sure "it runs on my machine" cannot hide a missing dependency, and
make the environment reproducible rather than "whatever pip resolves today".

**Why a virtual environment.** Not because the code fails without one — the
developer machine's Anaconda base already had all four packages — but precisely
*because* it already had them: a global environment can mask a package that
`requirements.txt` forgets to declare, and the course requires the repository to
run on a fresh machine.

**What was done.**

1. Created `.venv` from the existing Python 3.11.7 and installed
   `requirements.txt` into it.
2. The install resolved to **newer major versions** than the ones used during
   development:

   | package | development | fresh install |
   |---|---|---|
   | numpy | 1.26.4 | **2.4.6** |
   | pandas | 2.1.4 | **3.1.1** |
   | matplotlib | 3.8.0 | **3.11.2** |
   | PyYAML | 6.0.1 | 6.0.4 |

3. Ran the full pipeline inside the new environment. It passes **7/7**, which is
   the useful part of this step: it proves the code survives NumPy 2.x and
   pandas 3.x, and that the `np.trapz` → `np.trapezoid` compatibility shim in
   `src/nsc.py` was necessary rather than decorative.
4. `requirements.txt` keeps loose lower bounds (quick start, more portable), so a
   **`requirements-lock.txt`** was added with the exact resolved versions and the
   transitive dependencies, for bit-for-bit reproduction.
5. `README.md` gained an **Environment setup** section: activation commands for
   PowerShell / Git Bash / Linux, the `ExecutionPolicy` workaround for PowerShell,
   how to verify the install, and why `requirements-benchmark.txt` belongs in a
   *second* environment (`.venv-bench`) so the core one stays minimal.

**Commands run**

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/reproduce.py --quick
.\.venv\Scripts\python.exe -m pip freeze    # became requirements-lock.txt
```

**Result**

```text
overall: PASS  (7/7 steps, 11.2s)
```

**Notes.** `.venv/` and `.venv-bench/` are already covered by `.gitignore`. The
interpreter floor is now stated explicitly as **Python 3.11+**, because the
pinned NumPy does not build on older versions — worth knowing before the grader
tries it on 3.9.

---

### Step 12 — Pre-milestone correctness pass (completed this step)

An external review of the repository identified five problems worth fixing before
the milestone. All five are fixed; no research question, architecture or search
algorithm was changed.

**12a. FP32-only campaigns could not be analysed.**

`make_figures.py` raised *"the benchmark table does not cover the whole search
space"* whenever INT8 rows were absent — directly contradicting the plan to
measure FP32 first. Fixed by adding `split_by_coverage()` to
`src/benchmark_table.py` and a `--precisions` flag to `reproduce.py`,
`run_search.py`, `make_figures.py`, `try_evaluator.py` and
`make_synthetic_benchmark.py`. Tools now analyse the **covered subset** and say so:

```text
NOTE: 27 of 54 configurations have no measurement and are excluded
      (e.g. yolov5n_r320_int8_t1); analysing the 27 covered configurations.
```

Verified with a 27-row FP32 table: `reproduce.py --precisions fp32` reports
**8/8 PASS**.

**12b. Memory measurement was polluted across configurations; the platform label
was hard-coded.**

`resource.getrusage(RUSAGE_SELF).ru_maxrss` is a process-lifetime high-water
mark, so measuring every configuration in one process let a large model inflate
the reading of everything after it. The script also wrote
`"source": "raspberry_pi_4b"` unconditionally, which would label laptop
measurements as Raspberry Pi data.

Fixed: each configuration is measured in **its own subprocess** (`--single-config`
child mode, results returned as one JSON line), and the source tag is now
**detected** — `/proc/device-tree/model` on Linux, ARM machine names otherwise —
with `--source` as an explicit override. On this Windows machine it correctly
reports `laptop`.

**12c. The analytic architecture and the exported weights were different models.**

`src/model_spec.py` describes the **classic YOLOv5 v7.0** topology (anchor-based),
while `models/download_models.py` pointed at `yolov5nu/su/mu.pt` — the
**YOLOv5u** family, which is anchor-free with a decoupled, DFL-based detection
head. Those are not the same network, and since the capacity score is a function
of the convolution shapes, the score would have described a model that was never
benchmarked.

Fixed: the exporter now uses the v7.0 checkpoints (`yolov5n.pt`, `yolov5s.pt`,
`yolov5m.pt`), with the difference documented at the point of use.
`scripts/check_model_spec.py` additionally now **fails** (`--onnx-tolerance`,
default 0.25) when an exported ONNX graph disagrees with the analytic parameter
count, so the mismatch cannot recur silently.

**12d. No multi-seed experiment.**

Random and greedy were reported from a single seed. Added
`scripts/run_multiseed.py`: seeds 0–29, mean and standard deviation of the best
feasible accuracy, regret statistics, and the **feasible-success rate** (a run
that finds nothing is counted separately rather than as zero regret). The
heuristic is deterministic and reported once, with ablations.

Result under a 400 ms cap (22 of 54 feasible, oracle 0.4505):

```text
 budget             algorithm  runs  acc mean  acc std  regret mean  feas. rate
      4                random    30    0.3944   0.0394       0.0561        0.90
      4                greedy    30    0.3810   0.0000       0.0695        1.00
      4        heuristic/full     1    0.4223        -       0.0282        1.00
      8                random    30    0.4088   0.0328       0.0417        1.00
      8                greedy    30    0.3952   0.0244       0.0554        1.00
      8        heuristic/full     1    0.4223        -       0.0282        1.00
      8 heuristic/no_capacity     1    0.4420        -       0.0085        1.00
     12                random    30    0.4171   0.0303       0.0334        1.00
     12                greedy    30    0.4143   0.0307       0.0362        1.00
     12        heuristic/full     1    0.4223        -       0.0282        1.00
     12 heuristic/no_capacity     1    0.4505        -       0.0000        1.00
```

**This is the finding that matters.** The heuristic is never worse than either
baseline, but its margin over random search at budget 12 is 0.4223 versus 0.4171
— far smaller than the unconstrained result suggested. More importantly,
**removing the capacity term reaches the oracle (0.4505)**. On this synthetic
table the capacity term is therefore not what drives the constrained
improvement. That is a result about the synthetic benchmark, not about NSC, and
the milestone reports it as such.

**12e. Figures said "measured" while showing simulated data.**

`make_figures.py` derived a `data_kind` label from the table's `source` column, so
synthetic runs are now labelled **simulated** in every axis, title and printout,
and the Pareto figure says *"simulated benchmark, synthetic placeholder"*. A
reader cannot mistake a pipeline check for a measurement.

**Regression.** `python scripts/reproduce.py` now runs **8 steps** (the multi-seed
comparison was added to the pipeline) and reports **8/8 PASS** in about 9 s on the
synthetic table.

**Milestone report.** `report/milestone_report.tex` / `.pdf` — three pages,
Times New Roman 12 pt, 2.5 cm margins, single spacing. One figure in the original
draft was wrong (the 400 ms / budget 8 greedy value, 0.3810, is actually 0.4223);
the results section was rebuilt from the current code and now reports the
constrained comparison over 30 seeds instead of a single run.

---

### Step 13 — Real COCO benchmark and first measured results (completed this step)

The repository previously contained only a synthetic table; the milestone requires
evaluation on the chosen benchmark. This step builds that path and runs it.

**Files created**

| File | Purpose |
|---|---|
| `scripts/prepare_coco_subset.py` | Downloads a **fixed** COCO val2017 subset (images sorted by id, first N) plus its filtered annotations; resume-capable downloader |
| `scripts/evaluate_accuracy.py` | Letterbox → ONNX inference → per-class NMS → pycocotools mAP |
| `benchmarks/laptop_fp32.csv` | **Measured** benchmark table (18 configurations) |

**Problems found and fixed while doing this.**

1. `ultralytics.YOLO("yolov5n.pt")` silently resolves to **YOLOv5u** (anchor-free,
   2.64M parameters) rather than v7.0 (1.87M). The exported graph therefore did
   not match `src/model_spec.py`. `check_model_spec.py` caught it
   (`ratio 0.596 [MISMATCH]`) and the exporter was rewritten to build the v7.0
   model from the `ultralytics/yolov5` hub repository, which yields
   **1,872,157 parameters and 4.6 GFLOPs** — the published figures.
2. The COCO annotation archive aborted at 193/253 MB twice; the downloader now
   resumes with `Range` requests and retries.
3. `import onnxruntime` **segfaults** with version 1.32 on this Anaconda/Windows
   build. Pinned to 1.18.1 in `requirements-benchmark.txt`, which also pins
   `numpy<2` because that ORT build is compiled against the 1.x ABI — the reason
   the benchmark environment is separate from the core one.
4. `peak_rss_mb()` returned `nan` on Windows: `GetCurrentProcess` defaults to a
   32-bit return type, truncating the handle, so the memory query silently
   failed. Fixed with explicit `argtypes`/`restype`.

**Measured protocol.** 100 COCO val2017 images (smallest ids, 714 instances);
letterbox; per-class NMS, IoU 0.65, confidence 0.001; pycocotools; ONNX Runtime
CPU, one process per configuration; 60 timed images, 2 warm-up, 2 repeats.

**Results.**

```text
              mAP@0.5            latency@640   peak mem
model    320     416     640      (4 threads)   (MB)
yolov5n  0.4607  0.4918  0.5193     22.0 ms     146.8
yolov5s  0.5977  0.6331  0.6789     49.4 ms     178.9
```

Under a 25 ms latency cap (10 of 18 feasible, oracle 0.6331), 30 seeds:

```text
 budget             algorithm  acc mean  acc std  regret mean
      4                random    0.5737   0.0565       0.0594
      4                greedy    0.5977   0.0000       0.0354
      4        heuristic/full    0.6331        -       0.0000
      4 heuristic/no_capacity    0.4918        -       0.1413
      8                random    0.5993   0.0441       0.0338
      8                greedy    0.6083   0.0162       0.0248
      8        heuristic/full    0.6331        -       0.0000
      8 heuristic/no_capacity    0.5193        -       0.1138
     12                random    0.6178   0.0284       0.0153
     12                greedy    0.6213   0.0167       0.0118
     12        heuristic/full    0.6331        -       0.0000
     12 heuristic/no_capacity    0.6331        -       0.0000
```

**What this changes.** On *measured* data the story is the opposite of the
synthetic table: the capacity term now matters (ablating it costs 0.14 regret at
budget 4), and the heuristic reaches the feasible optimum with four evaluations
while random search is still 0.015 short after twelve. Rank correlation agrees —
the full score reaches Spearman ρ = 0.9875 against measured accuracy, the
no-capacity ablation collapses to ρ = −0.9499, and predicted MACs reach 0.9429
(so a MAC baseline is now clearly required in the final comparison).

`reproduce.py --table benchmarks/laptop_fp32.csv --precisions fp32` runs
**8/8 PASS** on the measured table, with no synthetic warning.

**Not yet measured.** YOLOv5m (weight download failed repeatedly) and INT8. The
measured pool is 18 of 54 configurations; the coverage-aware tooling handles that
without changes.

---

### Step 14 — Where does the ranking signal come from? (completed this step)

The milestone report left one question open, and it is the question the final
report has to answer: **predicted MACs reach a rank correlation of 0.943 on their
own -- so what does the NSC capacity term add?**

**What was built.**

| File | Purpose |
|---|---|
| `src/score.py` | `ScoreWeights.proxy` selects the accuracy proxy: `nsc` (this project's proposal), `macs` (free compute proxy), `params` (cheapest possible proxy) |
| `scripts/compare_proxies.py` | Runs the heuristic with each proxy, plus random and greedy, over three constraint scenarios and three budgets; also reports each proxy's rank correlation |

**Results** (measured table, 18 configurations, 30 seeds):

```text
--- unconstrained           feasible 18/18   oracle 0.6789 ---
    rank correlation: nsc=1.000  macs=0.943  params=0.878
 budget    random    greedy   heur[nsc]  heur[macs]  heur[params]
      4    0.6532    0.6331      0.6789      0.6789        0.6331
      8    0.6774    0.6789      0.6789      0.6789        0.6789
     12    0.6789    0.6789      0.6789      0.6789        0.6789

--- latency 25ms            feasible 10/18   oracle 0.6331 ---
      4    0.5737    0.5977      0.6331      0.6331        0.6331
      8    0.5993    0.6083      0.6331      0.6331        0.6331
     12    0.6178    0.6213      0.6331      0.6331        0.6331

--- latency 25ms + memory 140MB   feasible 9/18   oracle 0.6331 ---
      4    0.5716    0.5977      0.6331      0.6331        0.6331
```

**The honest finding.** `heur[macs]` ties `heur[nsc]` at every budget under every
constraint. The NSC term does rank strictly better (ρ = 1.000 against 0.943), but
on an 18-configuration pool the search reaches the same optimum either way, so
selection cannot separate them. The only proxy that is clearly worse is
`params` (0.6331 instead of 0.6789 at budget 4), because it is blind to
resolution.

This is recorded as a limitation, not smoothed over: on this benchmark the
project's central ingredient has **not** demonstrated an advantage over a free
compute proxy. What it *has* demonstrated is that the resource-aware search
framework turns any of these proxies into a large improvement over random
search.

**Why the pool is only 18.** YOLOv5m belongs in the space and its checkpoint was
obtained (42.8 MB), but exporting it to ONNX is blocked by an environment
conflict: `torch` is built against the NumPy 2 C-API while `onnxruntime` 1.18
(needed because 1.32 segfaults here) is built against NumPy 1. Swapping NumPy
between export and measurement is defeated by `pandas 2.1`'s own NumPy pin. The
clean fix is a third environment used only for export; that is tracked, not done.

**Next step arising from this.** A larger pool, not a better score, is what
would let the comparison separate the proxies: more detectors or more
resolutions, and INT8. That is now the highest-value remaining experiment.

---

## 4. Repository Map (current)

```text
rpi-edge-search/
├── README.md
├── PROGRESS.md                     <- this file
├── requirements.txt
├── requirements-lock.txt           <- exact tested versions
├── requirements-benchmark.txt
├── .gitignore
├── configs/
│   └── search_space.yaml           <- single source of truth for the pool
├── benchmarks/                     <- archived benchmark tables (CSV)
│   ├── README.md                   <- table schema and rules
│   └── synthetic_demo.csv          <- placeholder for pipeline testing only
├── models/                         <- model download/export instructions
│   ├── README.md
│   └── download_models.py          <- optional ONNX export
├── report/
│   ├── final_report.tex            <- four-page report source
│   └── final_report.pdf            <- compiled report
├── results/                        <- generated tables and figures
├── scripts/                        <- runnable entry points
│   ├── reproduce.py                <- one command: the whole pipeline
│   ├── check_capacity.py           <- capacity validation + variant table
│   ├── check_score.py              <- score/resource-model monotonicity checks
│   ├── make_synthetic_benchmark.py <- placeholder table generator
│   ├── try_evaluator.py            <- evaluator contract checks
│   ├── run_search.py               <- four-algorithm comparison
│   ├── make_figures.py             <- ranking / Pareto / regret tables and plots
│   └── benchmark_pi.py             <- optional Raspberry Pi measurement script
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
| S7 | One-command reproduction (`scripts/reproduce.py`) | **done** |
| S8 | Raspberry Pi benchmark script + ONNX export | **done** (runs when hardware is attached) |
| S9 | Four-page final report | **draft** (needs the hardware table) |
| S11 | Environment isolation + dependency lock | **done** |

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
4. **Pending push.** All commits are local only; the network to `github.com` was
   unreachable during these sessions. Push when the connection is back — no code
   change is required.

---

## 7. What remains before submission

Only work that needs the physical device or a decision, not more code.

| # | Task | Notes |
|---|---|---|
| 1 | Run `scripts/benchmark_pi.py` on the Raspberry Pi | Fill `benchmarks/pi4b_2gb_reference.csv` (54 rows, or the FP32 subset). |
| 2 | Supply accuracies via `--accuracy-json` | One mAP@0.5 per (model, resolution, precision); reuse across thread counts. |
| 3 | Re-run `python scripts/reproduce.py --table benchmarks/pi4b_2gb_reference.csv` | Regenerates every table and figure from real measurements. |
| 4 | Replace the numbers in `report/final_report.tex` | The structure, method and limitations sections do not change. |
| 5 | ~~Reconcile the parameter counts~~ | **Done in Step 10a.** The compute match (within 7%) validates the spec; the uniform ~19% parameter deficit is documented as immaterial to ranking and to the memory estimate. |
| 6 | ~~Decide the penalty fix~~ | **Done in Step 10b.** Penalties are clamped and the correlation is reported on the feasible subset. |
| 7 | `git push` | Commits are waiting when the network allows. |

### What is already finished

Everything that does not need the hardware: the configuration space, the analytic
capacity score with its numerical validation, the resource model, the composite
score with two ablations, the benchmark-table contract, the budgeted evaluator,
the four search algorithms, the ranking/regret/Pareto metrics, the figures, the
one-command reproduction, the hardware measurement script, and the four-page
report skeleton. `python scripts/reproduce.py` currently reports **7/7 steps
PASS** on a synthetic table.
