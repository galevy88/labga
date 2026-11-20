# Genetic algorithms library implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development with task ownership and independent review. The user has approved implementation; do not ask for another design approval.

**Goal:** Deliver a tested, installable Python library in a new repository owned by galevy88, with Objective-based evaluation, binary/numeric search spaces, ordinary/island GA, CPU multiprocessing, and one shared GPU predictor.

**Architecture:** A synchronous CPU coordinator owns genetic operations and seeded population states. An evaluation session owns workers and the predictor lifecycle. Objective encapsulates target/loss/preprocessing; GPU mode uses a single spawned owner and automatic batches, with stable candidate ordering.

**Tech stack:** Python 3.11+, NumPy, threadpoolctl for CPU thread limits, optional PyTorch, Hatchling, uv, pytest, Ruff, mypy strict.

**Spec:** ../specs/2026-11-20-genetic-algorithms-library-design.md

## Global constraints

- New standalone repository, never modify presentation code or research analyses.
- User confirmed repository/package name labga and public visibility with MIT license; no PyPI upload in this task.
- Confirmed namespace `labga`.
- All technical documentation/code in English; user communication in Hebrew.
- `optimize` takes `objective=Objective(...)`, not separate target/predictor/loss.
- Target is a fixed finite nonempty 1D vector; each candidate gets one finite scalar loss.
- Population size is total across islands. Ring migration copies simultaneous donor snapshots.
- `workers` is one global CPU budget, not multiplied by islands. Spawn multiprocessing; no GPU model in CPU workers.
- GPU device support is optional PyTorch/CUDA. Actual hardware verification must be reported separately from CPU and synthetic tests.
- Factory functions must be importable for spawn; CPU serial mode accepts ordinary callables/objects.
- Fully typed public API; validation and contextual errors; no swallowed failures or random penalty scores.

## Review focus

- Wrong target/output shape or non-finite rows must identify failed candidates without inventing scores (Task 2).
- CPU worker failure or GPU factory failure must not leave orphan processes or hang (Task 2).
- Migration must not cascade arrivals through all islands in one event or overwrite elites (Task 3).
- Batch and scalar evaluation must preserve candidate identity and per-row reduction (Task 2).
- Installed-wheel examples must work outside the source checkout, with optional torch absent (Task 4).

## Shared interfaces

Use `FloatArray = numpy.typing.NDArray[numpy.float64]` for internal population/prediction matrices. Binary genome vectors contain exact 0/1 values and are delivered as int8 to predictor/preprocess; numeric vectors are float64. Internal engine population can remain float64.

- `Genome` protocol: `size: int`, `sample(count: int, rng: Generator) -> FloatArray`, `validate(population: ArrayLike) -> FloatArray`, `mutate(population: FloatArray, rng: Generator, probability: float, scale: float) -> FloatArray`, `encode(candidate: FloatArray) -> NDArray[Any]`. Internal numpy dtype interop is the only justified Any here.
- `Objective(predictor, target, loss='mse', preprocess=None)`: validation, single/batch scoring; fixed immutable copied target. Predictors may be scalar callable, object with `predict`, optional `predict_batch`, or a `Predictor` factory config.
- `Predictor(factory, device='cpu', batch_size='auto')`: frozen validated configuration; factory accepts keyword `device` and returns callable/object; automatic initial batch cap 64.
- `EvaluationSession(objective: Objective, *, workers: int | Literal['auto'], genome: Genome)` context manager; `evaluate(population: FloatArray) -> EvaluationBatch`; batch owns `predictions: FloatArray`, `losses: FloatArray`. Session exposes serializable `metadata`, `evaluation_count`. Root owns these evaluation types.
- `Operators(tournament_size=3, crossover_probability=0.9, mutation_probability=None, mutation_scale=0.1, elite_count=1)`.
- `Migration(interval=10, count=1, topology='ring')`.
- `optimize(*, genome, objective, population_size=100, generations=100, distributed=False, n_islands=None, migration=None, workers='auto', seed=None, initial_population=None, operators=None) -> OptimizationResult`.
- `OptimizationResult`: `best_solution`, `best_prediction`, `best_loss`, `history`, `population`, `metadata`. `Population` contains solutions/losses/island_ids. `GenerationRecord` contains generation,best_loss,mean_loss,best_so_far,island_best,island_mean,migrated. Avoid retaining whole historical populations.

## Task 1: Search spaces and installable core

**Ownership:** Implementer owns `src/labga/genomes.py`, `src/labga/operators.py`, `tests/test_genomes.py`, `tests/test_operators.py`. Coordinator owns project metadata, shared protocols/types and exports.

- [ ] Scaffold src layout, Hatchling metadata, typed marker, development groups, CI command configuration and isolated uv environment.
- [ ] Write and observe failing tests for shared/custom bounds, probability endpoints, normal initialization, explicit invalid values, binary bit mutation and numeric boundary preservation.
- [ ] Implement `BinaryGenome(size, probability_of_one=0.5)`, `NumericGenome(size, bounds=(0,1), custom_bounds=None, initialization=None)` and `Normal(mean=0.5,std=0.1)` with finite validation and nonempty dimensions; reject booleans as integer sizes. Clipped normal is explicit. Copy user data.
- [ ] Implement configurable validation and offspring production in `operators.py`, preserving elites in engine rather than in the offspring function. `make_offspring(population, losses, count, genome, operators, rng)` returns exact count and does not mutate parents.
- [ ] Run targeted tests plus suite; report exact commands/results. Review implementation before depending on it.

## Task 2: Objective and evaluation infrastructure

**Ownership:** Coordinator owns `src/labga/objective.py`, `predictors.py`, `evaluation.py`, internal evaluation helpers as needed, and `tests/test_objective.py`, `test_evaluation.py`, `test_gpu.py`.

- [ ] Test Objective MSE/MAE/custom scalar losses, preprocessing, output dimensions, read-only target, batch rows and callable/predict routing.
- [ ] Implement plain CPU Objective evaluation and contextual `EvaluationError` carrying completed data and failed candidate IDs. Returned invalid rows identifiable; a raised batch call fails the whole submitted chunk.
- [ ] Test persistent spawned CPU worker factories, deterministic score ordering, auto worker detection/caps, initialization failure, and shutdown. Use importable test helper models; no cell closures under spawn.
- [ ] Implement a session with one worker pool per run and stable ordered evaluations; threadpoolctl limits native CPU threading inside workers. Retain completed chunks on failure.
- [ ] Test factory configuration and shared owner batching through real spawned CPU-owned factory tests; optional torch CPU tests cover eval/inference mode and typed CUDA OOM adaptation policy at the adapter boundary.
- [ ] Implement one GPU-owner process, initialized once; batch size auto 64 (or fewer remaining candidates), optional batch fallback. In auto mode only, typed PyTorch CUDA OOM halves batch size until 1; fixed-size mode surfaces the error. Record resolved size/fallback. Owner communicates through managed IPC with bounded outstanding requests; shutdown waits are bounded and failure checked.
- [ ] Run real optional CUDA test if hardware exists; otherwise mark explicitly skipped and document hardware verification pending. Never claim CUDA verified from mocks or CPU execution.

## Task 3: Ordinary and island GA

**Ownership:** Implementer owns `src/labga/optimize.py`, `islands.py`, `results.py`, `tests/test_optimize.py`, `test_islands.py`. Uses Task 1/2 interfaces verbatim.

- [ ] Write failing seeded CPU tests for numeric improvement, binary optimum, best prediction/loss consistency, generation zero, unchanged initial input and repeatability.
- [ ] Implement generational GA with fixed deterministic objective, elites and evaluated offspring. Count exactly initialization plus new evaluations; hold current predictions for migrated/surviving candidates.
- [ ] Write failing tests for 1000/4 population allocation, non-divisible remainder, invalid configuration, migration count constraints and simultaneous non-cascading ring replacement.
- [ ] Implement synchronized islands, independent seeded RNG streams, generation histories and migration metadata using one global EvaluationSession.
- [ ] Confirm engine outputs identical results across valid worker/batch schedules for deterministic predictors, including stable tie-breaking.
- [ ] Run suite and obtain task review.

## Task 4: Documentation, distribution and repository delivery

**Ownership:** Coordinator owns README, examples, docs, metadata, CI and release artifacts. Final reviewer reads all implementation.

- [ ] Write short complete runnable examples: binary fitting, numeric parameter fitting, islands, GPU adapter, importable notebook factory guidance. Mark synthetic biology illustrations honestly; no bundled DKO weights.
- [ ] Include updated design document in new repository and give the user its direct local and GitHub links.
- [ ] Run pytest with coverage, Ruff format/check and mypy strict. Build wheel/sdist; install wheel in a fresh environment and run examples outside source tree.
- [ ] Review complete diff for correctness, lifecycle/serialization, encoding, RNG and packaging. Fix confirmed issues with regression tests.
- [ ] Create the new repository using verified galevy88 authentication and user-selected visibility/license, commit only this new library, and push. Verify remote commit and file access. Do not publish to PyPI.
- [ ] Create a durable checkout in the user's projects directory if authorized filesystem access permits, preserving existing paths. Report actual achieved tests and limitations without claiming a production-validated biological/GPU integration.

## Preflight and progress ledger

| Pair/task | Interface / self-consistency | Decision |
| --- | --- | --- |
| 1→2 | Genome encode versus internal float matrices | Binary inference uses int8; engine uses float64 |
| 1→3 | Genome/operators methods | Shared interfaces above bind both implementers |
| 2→3 | EvaluationBatch/session/results | Evaluator returns input-order rows; engine handles generation/island context |
| 3→4 | Public result/examples | README uses only exported implemented API |
| 1 | Bounds and initialization remain distinct | User initial data validated, mutation clipped |
| 2 | Auto batching vs batch failure | Retry only typed CUDA OOM in auto; other failures preserve partial evidence |
| 3 | Workers versus islands | One session for all islands; no nested pools |
| 4 | GitHub account and publishing | Verified personal CLI account; connector uses different account, so use scoped CLI auth |

Ruling: include threadpoolctl as a small core dependency to enforce per-worker native thread limits; explicit control is safer than setting environment variables after NumPy import.

Progress will be recorded here with verification evidence as tasks complete. Initial implementation completed; final review fixes and release verification are tracked below.


### Verification progress — 2026-11-20

- Repository created: https://github.com/galevy88/labga (public, user-approved MIT).
- Initial implementation commit: f30c668; a separate hardening commit includes the review fixes.
- Initial full local verification: 172 tests passed, 2 platform-dependent skips; coverage 91.45%. Strict mypy and Ruff passed.
- All four runnable examples passed: binary exact match, numeric time-series fitting, four islands/two CPU workers, and PyTorch inference on CPU.
- Wheel and source distribution built successfully. Fresh CPU-only environment imports the wheel without PyTorch.
- Independent review accepted six fixes: real-valued objective validation, complete IPC timeout supervision, CUDA model ownership guard, bfloat16 conversion, exact bounds conversion, and overflow-safe builtin loss reduction. All six findings were fixed with regressions and approved in a scoped independent re-review.
- Real CUDA hardware is unavailable on the development host; hardware smoke test is explicitly skipped.
- Final local verification: 201 passed, 2 skipped, 92.31% coverage; minimum-dependency installed-wheel suite: 184 passed, 10 optional/platform skips. Delivery includes a public repository and a durable local checkout. GitHub CI results are tracked in workflow runs.

Ruling: document caller-side startup serialization/OS process creation as outside interruptible evaluation timeouts. Spawned request transfers and replies are supervised; serial callbacks remain synchronous. Cost: users with expensive custom serialization should use the documented lightweight factory interface.
