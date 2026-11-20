# labga

**Define a solution, predict its outcome, and evolve it toward your target.**

labga provides ordinary genetic algorithms and island populations through one Python API. Bring a fixed predictor or simulator, a target vector, and a loss. Use binary interventions or continuous parameters. Evaluate on CPU workers or through one shared, batched GPU predictor.

This is an initial research-library release. The included examples are synthetic. A real DKO model and its validation are not bundled.

## Install

The package has not been published to PyPI yet. Install directly from this repository:

```bash
pip install "git+https://github.com/galevy88/labga.git"
```

Optional PyTorch support:

```bash
pip install "labga[torch] @ git+https://github.com/galevy88/labga.git"
```

Python 3.11 or later is required. Core use requires NumPy and threadpoolctl, not PyTorch. A suitable PyTorch/CUDA installation and GPU are needed for CUDA inference.

## A complete first example

```python
import numpy as np
from labga import BinaryGenome, Objective, optimize


def predict(genome):
    return genome.astype(float)


if __name__ == "__main__":
    objective = Objective(
        predictor=predict,
        target=np.array([1, 0, 1, 1, 0, 1]),
        loss="mse",
    )

    result = optimize(
        genome=BinaryGenome(size=6, probability_of_one=0.3),
        objective=objective,
        population_size=80,
        generations=30,
        workers=1,
        seed=42,
    )

    print(result.best_solution)
    print(result.best_prediction)
    print(result.best_loss)
```

`Objective` groups everything that defines a good solution. The GA changes the genome; the predictor remains fixed. MSE/MAE compare each predicted output to the same target. Every candidate gets its own score, even when inference is batched. For a scalar prediction, return a length-one vector.

The identity predictor above makes the example fully transparent. Replace it with an experimental simulator or an adapter around your own model.

## Search spaces

```python
from labga import BinaryGenome, Normal, NumericGenome

binary = BinaryGenome(size=48, probability_of_one=0.1)
numeric = NumericGenome(size=1000, bounds=(0.0, 1.0))

custom = NumericGenome(
    size=1000,
    bounds=(0.0, 1.0),
    custom_bounds={0: (-2.0, 2.0), 17: (0.0, 100.0)},
)

concentrated = NumericGenome(
    size=1000,
    bounds=(0.0, 1.0),
    initialization=Normal(mean=0.5, std=0.1),
)
```

- Binary probability controls independent initialization, not an exact count of ones. Your adapter defines whether 1 means knockout, retention, or something else.
- Numeric initialization is uniform by default. Normal initialization is clipped to bounds and may accumulate values at the boundaries. Scalar or per-coordinate mean/std are accepted.
- Custom bounds can be a zero-based override dictionary or a complete `(size, 2)` array loaded from your metadata.
- Bounds remain enforced during evolution. `initial_population=` can supply a validated complete population instead of sampling one.
- Binary predictors receive int8 0/1 vectors; numeric predictors receive float64 vectors. Unsupported lossy numeric conversions are rejected.
- Genome length and prediction length may differ. Continuous numeric genomes are not automatically constrained to sum to one.

## Four populations, one evaluation budget

```python
from labga import Migration

result = optimize(
    genome=binary,
    objective=Objective(predict, target=np.ones(48)),
    population_size=1000,
    generations=100,
    distributed=True,
    n_islands=4,
    migration=Migration(interval=10, count=5),
    workers="auto",
    seed=42,
)
```

This means **four populations of 250**, not four populations of 1,000. Islands have independent evolutionary random streams. Every ten generations, each copies five of its best solutions to its next neighbor in a ring, replacing poor non-protected solutions. Donors are selected from simultaneous snapshots; migration never cascades a new arrival through the entire ring in one event.

The island model runs on one computer in this release. Independent islands do not imply separate search basins or separate GPUs. All islands share a single evaluation budget. Standard GA also supports CPU multiprocessing.

Defaults are tournament selection, uniform crossover, binary bit-flip or bounded numeric mutation, and one elite per population. Use `Operators(...)` to adjust them.

## Your predictor

Supported interfaces:

```text
callable(x) or predict(x):  (D,) → (M,)
optional predict_batch(X): (B, D) → (B, M)
```

Each batch row must be independent and in the same order as the input. An optional `Objective(preprocess=...)` transforms each input vector before prediction. A custom `loss(prediction, target)` receives two NumPy vectors and returns one finite scalar; lower is better. The target is copied and must be finite and one-dimensional.

Use fixed, comparable evaluation conditions across candidates. Stochastic predictors should implement a reproducible protocol in the adapter. A run seed controls labga's evolution; it cannot make arbitrary model code or GPU kernels deterministic.

## Automatic batching and GPU inference

```python
from labga import Objective, Predictor
from my_lab.dko_adapter import build_dko  # Your importable adapter factory.

objective = Objective(
    predictor=Predictor(factory=build_dko, device="cuda:0"),
    target=desired_expression,
    loss="mse",
)
```

`build_dko(*, device)` loads and returns your adapter. Serializable fixed configuration can be bound with `functools.partial`. A returned PyTorch `nn.Module` is automatically adapted to evaluation/inference mode. An existing wrapper can expose `predict` and optionally `predict_batch` instead.

One spawned process owns the CUDA model. CPU work prepares candidates; the GPU owner groups compatible requests into batches and returns predictions/scores in candidate order. Models are not copied into every CPU worker.

`batch_size="auto"` starts conservatively at 64 (or fewer available candidates). A typed CUDA out-of-memory error during automatic inference reduces the cap by half, down to one. Other errors are surfaced. An explicit `batch_size=128` is respected and is not silently reduced. Without batch support, the model is called per candidate. Actual batch size and fallback are reported in result metadata. This is a memory-conservative policy, not a benchmark-based throughput tuner.

Built-in MSE/MAE can be reduced on the GPU when predictions are tensors; custom losses use NumPy on CPU. Predictions needed for results are transferred back to CPU. No model training occurs.

CUDA code must be tested on a CUDA-equipped machine. The initial development host is macOS/Apple Silicon; CPU/PyTorch tests do not substitute for a successful CUDA hardware test.

## Multiprocessing and notebooks

`workers="auto"` detects available CPU capacity and caps workers by runnable work. It does not create one pool per island. With GPU inference and no expensive preprocessing, additional CPU workers are unnecessary. More CPU workers do not multiply the throughput of one GPU.

Use a main guard in scripts. Put predictor factories and preprocessing functions in an importable module for spawn-based multiprocessing. In notebooks, import that module; use `workers=1` for cell-local CPU callables. Heavy CPU models can also use `Predictor(factory=..., device="cpu")`, loading once per evaluation owner.

`evaluation_timeout=300.0` sets a bounded multiprocessing request wait in seconds; adjust it for legitimately slow simulations. Invalid inputs, model exceptions, worker failure, and unavailable CUDA produce explicit errors. Evaluation failures preserve available completed-result diagnostics rather than silently assigning random penalties.

## Inspect the result

```python
result.best_solution
result.best_prediction
result.best_loss
result.population.solutions
result.population.losses
result.population.island_ids

for generation in result.history:
    print(generation.generation, generation.best_loss, generation.mean_loss)

print(result.metadata)
```

History includes initialization as generation zero, best-so-far scores, per-island summaries, and migration markers. Results own their arrays; labga does not write output files automatically. Elitism protects the best evaluated score, but no global-optimum guarantee is implied.

## Examples and development

- `examples/binary_search.py`: complete binary optimization.
- `examples/parameter_fitting.py`: fit two parameters to a synthetic time series.
- `examples/islands.py`: ring migration with CPU multiprocessing.
- `examples/gpu_predictor.py`: a synthetic PyTorch model; use `--device cpu` or `--device cuda:0`.

```bash
uv sync --extra torch
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv build
```

See the [design](docs/design.md) and [implementation plan](docs/implementation-plan.md). The original [Git_HH project](https://github.com/galevy88/Git_HH) inspired this library's workflow; this is a separate implementation with a general prediction interface.

MIT license. Model weights, experimental data, and third-party predictors retain their own licenses and validation requirements.
