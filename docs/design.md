# Genetic algorithms library — approved design

- Date: 2026-11-20
- Status: Initial implementation complete; version 0.1.0 published to PyPI and installation verified on 2026-11-20. Remaining scope boundaries are listed in Section 11.
- Owner: `galevy88`
- Repository and package name: `labga` (confirmed by the user); public, MIT licensed.
- Repository: https://github.com/galevy88/labga
- Package: https://pypi.org/project/labga/
- Python support target: 3.11+; initial local verification on 3.12, CI matrix covers 3.11–3.13.

This document records the approved design and the initial implementation contract. The implementation belongs in a separate GitHub repository. The examples describe the approved interface implemented in the initial library.

## 1. Goal and agreed requirements

Provide a small Python library that lab colleagues can install from PyPI, import, and use to optimize an input vector against a desired prediction. A user supplies the search space and an `Objective` that bundles a fixed predictor, a target, a loss, and optional preprocessing. The library manages evolution and evaluation.

Agreed scope:

- A new repository under `galevy88`; Git_HH provides experience and inspiration.
- Binary and continuous numeric genomes, with configurable initialization.
- Shared numeric bounds plus optional bounds for individual coordinates.
- Standard GA and an island GA selected with a flag.
- Automatic CPU worker budgeting based on available CPUs, with a manual override.
- A user-supplied predictor/wrapper, including a possible DKO integration.
- Built-in MSE and MAE plus a custom loss function.
- A shared GPU predictor, with CPU work separated from inference.
- Straightforward examples, results, and documentation suitable for colleagues.

Initial operator defaults, numeric initialization details, and migration settings are implemented as described below.

## 2. Optimization contract

```text
candidate genome → optional preprocessing → fixed predictor → prediction
                                                           ↓
target ─────────────────────────────────────────────────── loss
                                                           ↓
                            selection → crossover → mutation
                                                           ↓
                                                   new candidates
```

One genome is one complete candidate solution. A gene is one coordinate. Genome length, prediction length, and population size are independent quantities. The objective produces one scalar score per candidate using the same target throughout the run, including across islands.

The predictor is not trained during optimization. The GA changes its input, evaluates the resulting prediction, and minimizes the loss. The library does not require gradients or a differentiable predictor. It cannot guarantee a global optimum or establish biological validity beyond the predictor's validated domain.

For DKO, the proposed use is a binary intervention vector mapped to an expression vector. The meaning of 0/1, supported simultaneous knockouts, required baseline/context, and output gene ordering must come from the actual DKO interface. The library must not invent or assume those capabilities. A demonstration wrapper can be provided before that integration is verified, clearly labelled synthetic.

## 3. Main API

The common entry point is `optimize(...)`. Users do not need to subclass an optimizer.

### Smallest CPU example

```python
from labga import BinaryGenome, Objective, optimize

objective = Objective(
    predictor=my_predictor,
    target=target_expression,
    loss="mse",
)

result = optimize(
    genome=BinaryGenome(size=48, probability_of_one=0.1),
    objective=objective,
    population_size=1000,
    generations=100,
    workers="auto",
    seed=42,
)

print(result.best_solution)
print(result.best_loss)
```

`my_predictor` is a callable `x → prediction` or an object exposing `predict(x)`. `workers=1` runs CPU evaluation in the caller and supports ordinary in-memory objects. Multiprocessing requires importable/serializable callables and initialization data. Heavy predictors should use the factory wrapper below, loading once per CPU worker rather than once per candidate.

### GPU and island example

```python
from labga import BinaryGenome, Migration, Objective, Predictor, optimize
from my_lab.dko_adapter import build_dko

objective = Objective(
    predictor=Predictor(
        factory=build_dko,
        device="cuda:0",
        batch_size="auto",
    ),
    target=target_expression,
    loss="mse",
)

result = optimize(
    genome=BinaryGenome(size=48, probability_of_one=0.1),
    objective=objective,
    population_size=1000,
    generations=100,
    distributed=True,
    n_islands=4,
    migration=Migration(interval=10, count=5, topology="ring"),
    workers="auto",
    seed=42,
)
```

`Predictor` is an optional factory wrapper supplied inside `Objective`. Here `build_dko` is a user-provided, importable factory that accepts the keyword `device` and returns a predictor adapter. It loads the model inside its owner process. The caller supplies checkpoint paths and fixed context through serializable factory configuration such as `functools.partial`. Model weights and credentials stay with the caller.

The same GPU example works with a single population by omitting `distributed`, `n_islands`, and `migration`. GPU usage and the search algorithm are independent settings.

| `optimize` argument | Meaning / default |
| --- | --- |
| `genome` | Required search-space specification |
| `objective` | Required `Objective` with predictor, target, loss, and optional preprocessing |
| `population_size` | Total number of candidates; proposed default 100 |
| `generations` | Number of evolutionary updates after initial evaluation; proposed default 100 |
| `distributed` | Default `False`; `True` enables the island model |
| `n_islands` | Default 4 in island mode; otherwise unused and omitted |
| `migration` | Island configuration; default ring, every 10 generations, 1 migrant per island |
| `evaluation_timeout` | Positive finite seconds per spawned request; default 300.0 |
| `workers` | `"auto"` or positive integer; a global CPU budget, never per island |
| `seed` | Integer or `None`; controls library randomness |
| `initial_population` | Optional valid array of shape `(population_size, genome.size)` |
| `operators` | Optional typed settings for selection, crossover, mutation, and elitism |

Reject conflicting settings rather than silently ignoring them, such as migration options in single-population mode. Advanced configuration stays optional so the basic call remains short.

## 4. Genome definitions

### Binary

```python
BinaryGenome(size=48, probability_of_one=0.1)
```

Each coordinate is initialized independently as 1 with probability 0.1. This controls an expected fraction, not an exact number of ones. Mutation flips bits; crossover preserves binary values. The adapter defines whether 1 means knockout, retained gene, or another action.

### Numeric

```python
from labga import Normal, NumericGenome

# One range for all 1,000 coordinates; uniform initialization.
genome = NumericGenome(size=1000, bounds=(0.0, 1.0))

# Same default range, a few coordinate-specific overrides.
genome = NumericGenome(
    size=1000,
    bounds=(0.0, 1.0),
    custom_bounds={0: (-2.0, 2.0), 17: (0.0, 100.0), 999: (-10.0, 10.0)},
)

# Concentrate initialization while retaining the same valid range.
genome = NumericGenome(
    size=1000,
    bounds=(0.0, 1.0),
    initialization=Normal(mean=0.5, std=0.1),
)
```

`custom_bounds` also accepts a full `(size, 2)` numeric array, for example loaded from laboratory metadata by the caller. No hand-written thousand-element list is required. Indexing is zero-based. Missing dictionary entries inherit `bounds`.

Proposed normal initialization: sample a normal distribution and clip to the coordinate bounds, explicitly documented as **clipped normal**, not a truncated-normal distribution. Boundary mass is therefore possible. Scalar mean/std apply to all coordinates; arrays of length `size` allow individual initialization settings. Advanced users can generate a custom valid `initial_population` externally.

Bounds are permanent constraints, not merely initialization settings. Numeric mutation is clipped to the same bounds; supplied initial populations are validated, not silently repaired. Bounds must be finite and satisfy lower < upper. Continuous floating-point genomes are the numeric scope of v1. Integers, categories, permutations, exact knockout counts, and sum-to-one vectors require separate operators/constraints and are outside this release.

## 5. Objective, predictor, preprocessing, and loss contracts

| `Objective` argument | Meaning / default |
| --- | --- |
| `predictor` | Required callable, object exposing `predict(x)`, or optional `Predictor` factory wrapper |
| `target` | Required finite numeric vector in the predictor's output order |
| `loss` | `"mse"`, `"mae"`, or callable. Defaults to `"mse"` and is always minimized. |
| `preprocess` | Optional CPU-only candidate transform. Defaults to identity. |

The default model input is the raw genome vector, not the `BinaryGenome`/`NumericGenome` configuration object.

| Operation | Input → output |
| --- | --- |
| `predict(x)` / callable | `(D,) → (M,)` |
| Optional `predict_batch(X)` | `(B, D) → (B, M)` |
| Optional CPU `preprocess(x)` | `(D,) → (K,)`, fixed numeric length; predictor then accepts K |
| Custom `loss(prediction, target)` | Two `(M,)` arrays → one finite scalar |

Single-candidate prediction is the minimum contract. Batch prediction must preserve candidate order and give independent results equivalent to individual inference, within numeric tolerance. Inference must be deterministic and side-effect-free so results remain valid when a batch is split or retried. Without `predict_batch`, both CPU and GPU execution evaluate candidates independently through `predict(x)` or the callable. Internal DKO tensor/dictionary conversion remains encapsulated in its adapter.

For output length M:

```text
MSE(candidate) = mean((prediction - target)²)
MAE(candidate) = mean(abs(prediction - target))
```

The library computes **one scalar loss per candidate**, yielding B losses for B candidates. Predictions and the run's shared target must have identical output dimensions and meaning. The caller is responsible for output labels and units. A custom loss can apply weighting or scaling.

CPU adapters return NumPy-compatible vectors. The optional PyTorch adapter can retain tensors on CUDA and calculate built-in losses there. It puts the model in evaluation mode and uses inference mode; adapters must not mutate model state between candidate evaluations. Custom losses use NumPy inputs on CPU in v1, transferring predictions as necessary. This trades some performance for a simple, framework-independent custom-loss contract.

No separate scalar-objective API is required for v1. General simulator/parameter fitting is supported through the same predictor-target-loss contract. A direct `objective(genome)` shortcut can be considered later.

## 6. Standard GA and island GA

### Standard GA

1. Initialize and evaluate the population (history generation 0).
2. Select parents, construct children, mutate, and enforce genome validity.
3. Evaluate children using the run's objective.
4. Form the next population, keeping configured elites.
5. Record history; repeat until `generations` updates complete.

Proposed operator defaults: tournament selection of up to 3 candidates; uniform crossover with probability 0.9 per pair; independent per-coordinate mutation probability `1 / genome.size`; binary bit flips or numeric normal perturbations with std 0.1 of each coordinate's range; 1 elite per population. All remain configurable through typed `operators` settings. Mutation probability and numeric mutation magnitude are separate controls.

The remaining population slots are filled by evaluated offspring. Candidate scores are comparable because the objective is fixed. The initial implementation assumes deterministic evaluation: stochastic simulators must provide a wrapper with a fixed evaluation protocol/seeds. It must not silently compare elites measured on different random datasets.

### Island GA

For `population_size=1000, n_islands=4`, maintain four populations of 250. Each has its own parents, offspring, and RNG stream. Between migration events, all four evolve independently and share the evaluation infrastructure.

Proposed migration: every 10 generations, copy the configured number of best candidates from island i to island `(i+1) mod n_islands`, replacing the receiver's worst non-protected candidates. Snapshot all donors before replacement to avoid passing a new migrant through several islands in one event. Copying retains the donor and preserves all population sizes. Validate migrant count against the smallest island and its elite count. Scores can accompany migrants because the objective is common and fixed.

For a non-divisible total, distribute the remainder deterministically so island sizes differ by at most one. Require at least two candidates per island and compatible operator/migration settings. Randomly partition any supplied initial population with the run seed to avoid assigning sorted input blocks unintentionally.

An island is a separate evolving population, not a physical basin and not necessarily a separate process or computer. Starting populations may overlap in search space. In v1, the coordinator advances islands in synchronized rounds and schedules their evaluations concurrently; genetic operators remain cheap CPU work. Network-distributed execution is outside scope.

## 7. CPU and GPU execution

```text
                   CPU coordinator
         single GA OR multiple island populations
                         │
          candidate / island / generation IDs
                         │
         optional CPU preprocessing worker pool
                         │
                bounded request queue
                         │
            GPU owner: one model instance
              batch → predict → score
                         │
              scores returned by ID
                         │
                   CPU coordinator
```

| Work | CPU predictor mode | GPU predictor mode |
| --- | --- | --- |
| Selection, crossover, mutation, migration | CPU coordinator | CPU coordinator |
| Expensive preprocessing | Evaluation workers | Optional shared CPU worker pool |
| Predictor inference | CPU workers; one initialized model per worker if needed | One owner process and one model instance on the selected GPU |
| MSE/MAE | CPU | GPU for PyTorch tensors; otherwise CPU |
| Custom loss | CPU evaluation path | CPU after prediction transfer |
| CPU parallelism | Independent candidate or batch evaluations | Preparation and applicable postprocessing |

`workers="auto"` resolves a budget from CPUs available to the current process. Prefer process-aware CPU counts where supported, then affinity where available, then system count, with a minimum of one. Account for discoverable container quotas. Record detection and actual worker counts; allow an explicit cap. The count is logical CPU availability, not a guarantee of distinct physical cores or linear speedup. See the [Python CPU-count API](https://docs.python.org/3/library/os.html#os.process_cpu_count).

CPU mode caps workers by runnable tasks and uses one thread per worker for supported BLAS/PyTorch pools to avoid nested oversubscription. GPU mode does not create a pool simply because cores exist: without substantial CPU preprocessing, the coordinator can submit whole candidate batches directly. This is still `workers="auto"`; the value is a budget, not a requirement to occupy every core.

With `predict_batch`, the coordinator can combine compatible candidates from all islands while preserving their IDs. The default `batch_size="auto"` starts at 64 in both CPU and GPU modes and caps each batch at the number of available candidates. The final partial batch is submitted immediately. `Predictor(batch_size=...)` accepts an explicit positive integer to cap each batch when batch prediction is available. An adapter without batch prediction uses independent single-candidate calls.

Adaptive batch retry is limited to automatic batching when CUDA batch inference raises `torch.cuda.OutOfMemoryError`. Halve the failed batch size using integer division, with a minimum of one, and retry the pending candidate IDs in smaller chunks. Use the reduced size for subsequent batches in the run. If a batch of size one still raises CUDA OOM, propagate the failure. With an explicit integer size, surface a CUDA OOM without downsizing or retrying. All other batch errors follow the evaluation error policy in Section 8.

All populations share one CUDA owner process created with `spawn`. It initializes CUDA and one model instance inside the process, owns device tensors, and handles inference for the entire run. This follows [PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html). Pools and queues have bounded capacity, shutdown paths, and error propagation. Candidate IDs preserve deterministic association even when completion order changes.

Initial scope: NumPy-compatible CPU predictors plus an optional PyTorch/CUDA adapter on one GPU. CPU mode must work without installing PyTorch. Other accelerators, multiple GPUs, remote execution, and serving infrastructure are extensions rather than implicit promises.

For scripts, document the multiprocessing main guard. For notebooks, demonstrate an importable adapter module/factory; support arbitrary cell-local models in serial CPU mode. Do not promise arbitrary notebook closures or already-loaded CUDA models will work under spawn. GPU initialization failure is reported explicitly, not silently replaced by CPU inference.

## 8. Results, validation, and reproducibility

```python
result.best_solution  # Raw genome in its input encoding
result.best_prediction  # Prediction corresponding to that exact solution
result.best_loss  # Scalar objective value
result.history  # Initial state + per-generation best/mean and per-island summaries
result.population  # Final genomes, losses, and island membership
result.metadata  # Seed, resolved settings, evaluation count, workers/device, timings
```

Return owned arrays; avoid retaining every prediction of every generation. Best-so-far and per-generation statistics are labelled distinctly. History records each migration event. Users can plot history without a mandatory visualization dependency. No automatic file writes or telemetry.

Validate shapes, finite values, genome encoding/bounds, probabilities, target dimensions, population sizes, worker counts, and migration settings before or at the earliest relevant evaluation boundary. Errors identify the candidate, island, and generation when applicable.

Evaluation collects successful submitted tasks and failures. An individual call failure identifies that candidate. A batch exception marks the submitted batch as failed unless the automatic CUDA OOM retry in Section 7 applies. A returned batch with non-finite predictions can record valid rows and invalid row IDs separately. Preserve already-completed results from other tasks and batches. The default run policy raises a contextual evaluation error carrying the partial results and diagnostics before any further evolution. Penalties for infeasible candidates require a future explicit exception contract. Model-level failures, unrecovered OOM, worker exit, and cancellation shut down resources cleanly.

Derive stable RNG streams from the run seed for each island. Variation occurs in deterministic order in the coordinator; asynchronous evaluation completion must not change selection order or tie-breaking. Library seeding does not guarantee bit-for-bit reproducibility for arbitrary stochastic predictors or nondeterministic GPU kernels; the adapter's evaluation protocol and environment must be recorded.

## 9. New repository and release shape

```text
new-repository/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── src/labga/
│   ├── __init__.py
│   ├── optimize.py
│   ├── genomes.py
│   ├── operators.py
│   ├── islands.py
│   ├── objective.py
│   ├── predictors.py
│   ├── evaluation.py
│   ├── losses.py
│   └── results.py
├── tests/
├── examples/
│   ├── binary_search.py
│   ├── parameter_fitting.py
│   ├── islands.py
│   └── gpu_predictor.py
├── docs/
└── .github/workflows/
```

Dependency boundary: NumPy and threadpoolctl in core; PyTorch optional for the GPU adapter; plotting/notebook dependencies optional. DKO itself and its weights remain user supplied. The package uses `src/` layout, `pyproject.toml`, a deliberately selected build backend (proposed Hatchling), `uv` for development, Ruff, strict typing, and pytest. Build wheel/sdist and verify installation independently of the source checkout, following the [Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

README should start with installation and a short complete optimization example, followed by binary/numeric genomes, islands, and GPU execution. Clearly distinguish population parallelism from evaluation parallelism. Include model factory and notebook examples, expected input/output shapes, and troubleshooting.

The initial implementation includes the core and CPU examples, islands, shared GPU evaluation, and compatibility checks. Publication of version 0.1.0 was explicitly authorized and completed on 2026-11-20 using Trusted Publishing. A clean installation from production PyPI passed standard-GA and island/multiprocessing smoke checks; TestPyPI remains an optional rehearsal for later versions. Actual CUDA and workload-specific performance checks remain separate scope boundaries.

## 10. Verification and scope boundaries

Before release, verify:

- Binary validity, initialization probability behavior, shared/custom numeric bounds, and mutation constraints.
- Objective preprocessing and MSE/MAE and custom-loss semantics, including one loss per batch row against the run's shared target.
- Single/multi-worker score association, seeded reproducibility, and predictor construction once per owner.
- Automatic batching up to 64 available candidates, independent calls for adapters without batch support, and explicit positive batch sizes.
- Automatic CUDA batch reduction only for the typed PyTorch OOM, with failure at size one. Verify immediate error propagation for an explicit size or any other batch exception.
- Correct population sizes, simultaneous ring migration, elite protection, and independent island RNGs.
- Clear failure reporting, partial evaluation evidence, no orphan workers, cancellation, and GPU-owner shutdown.
- A real optional CUDA smoke test: one loaded model, batch/single consistency, correct CPU/GPU loss within tolerance. CPU CI alone does not establish CUDA correctness.
- A clean installed wheel running the examples. CPU compatibility on Linux, macOS, and Windows; explicit tested Python range selected before packaging.
- Benchmarks of representative CPU and GPU workloads, including preparation time, batch size, and peak memory. Report measured performance without assuming all-core evaluation is fastest.

Not in v1: multi-machine scheduling, multiple GPUs, multi-objective/Pareto search, arbitrary mixed/structured genomes, domain-specific feasibility constraints, checkpoint/resume, predictor training, or a bundled biological model. Add these only for a demonstrated need.

## 11. Relationship to Git_HH and outstanding decisions

Read-only inspection of Git_HH at commit `b6e1ff20c2c31ef8636e669dab5ac9bc4398a08d` found a useful GA structure and domain-specific preprocessing. In the inspected runner paths, multiprocessing launches independent complete GA runs; the inspected inner GA evaluates candidates serially. Therefore per-candidate multiprocessing and island migration should be designed explicitly, not described as already extracted working features.

Evidence: [GA loop](https://github.com/galevy88/Git_HH/blob/b6e1ff20c2c31ef8636e669dab5ac9bc4398a08d/UltraProduction/Modules/Algo/ga.py), [parallel runner](https://github.com/galevy88/Git_HH/blob/b6e1ff20c2c31ef8636e669dab5ac9bc4398a08d/UltraProduction/Modules/Algo/MP_RUNNER.py). This observation is scoped to inspected files, not a claim about every historical version.

Reuse the problem-solving approach and preserve credit/provenance where code is reused. Do not carry HH-specific imports, preprocessing scripts, datasets, shared-output-file assumptions, or inconsistent bound/fitness handling into the general core.

Decisions to settle before the relevant milestone:

1. **Confirmed:** `galevy88/labga`, public repository, MIT license, Python 3.11+. Version 0.1.0 was published to PyPI on 2026-11-20 and independently installed from the public index. See `docs/verification.md` for release evidence.
2. **Before real DKO integration:** inspect its actual wrapper, supported perturbations, batch semantics, model construction, framework, and device support. This does not block implementing generic CPU examples.
3. **Implemented initial defaults:** tournament selection, uniform crossover, clipped numeric normal initialization/mutation and configurable ring migration. The synchronized island model runs on one computer. Wider deployment and performance claims require workload-specific measurement.
