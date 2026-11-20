import numpy as np
import pytest

from labga.evaluation import EvaluationSession
from labga.genomes import NumericGenome
from labga.objective import Objective
from labga.predictors import Predictor


def cuda_factory(*, device):
    import torch

    return torch.nn.Identity()


@pytest.mark.cuda
def test_real_cuda_single_owner_and_loss():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A real CUDA device is required")
    objective = Objective(Predictor(cuda_factory, device="cuda", batch_size=2), [0, 1])
    x = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=np.float64)
    with EvaluationSession(objective, workers=4, genome=NumericGenome(2)) as runner:
        outcome = runner.evaluate(x)
        np.testing.assert_allclose(outcome.predictions, x)
        np.testing.assert_allclose(outcome.losses, [0, 0.25, 1])
        assert runner.metadata["gpu_owners"] == 1
        assert runner.metadata["workers_actual"] == 0


class CPUBackedConfig(Predictor):
    """Exercise the spawned owner protocol using a CPU model; no device emulation."""

    def build(self):
        return self.factory(device="cpu")


class CapacityModel:
    def __init__(self, path, limit, error):
        self.path = path
        self.limit = limit
        self.error = error

    def predict_batch(self, x):
        import torch

        with self.path.open("a") as out:
            out.write(f"{len(x)}\n")
        if len(x) > self.limit:
            if self.error == "typed":
                raise torch.cuda.OutOfMemoryError("synthetic test capacity")
            raise RuntimeError("CUDA out of memory but untyped")
        return x


def capacity_factory(path, limit, error, *, device):
    return CapacityModel(path, limit, error)


def double(row):
    return row * 2


@pytest.mark.torch
def test_auto_shrink_persists_across_spawned_requests(tmp_path):
    from functools import partial

    pytest.importorskip("torch")
    log = tmp_path / "sizes"
    config = CPUBackedConfig(partial(capacity_factory, log, 2, "typed"), device="cuda")
    objective = Objective(config, [0], preprocess=double)
    x = np.arange(8, dtype=float).reshape(-1, 1) / 8
    with EvaluationSession(objective, workers=2, genome=NumericGenome(1)) as runner:
        for _ in range(2):
            outcome = runner.evaluate(x)
            np.testing.assert_equal(outcome.predictions, x * 2)
            np.testing.assert_equal(outcome.losses, (x.ravel() * 2) ** 2)
            assert runner.metadata["batch_size_resolved"] == 2
        assert runner.metadata["workers_actual"] == 2
        assert runner.metadata["gpu_owners"] == 1
    assert log.read_text().splitlines() == ["8", "4", *(["2"] * 8)]


@pytest.mark.torch
@pytest.mark.parametrize(
    ("size", "limit", "error", "calls"),
    [(4, 1, "typed", [4]), ("auto", 1, "untyped", [4]), ("auto", 0, "typed", [4, 2, 1])],
)
def test_gpu_error_policy(tmp_path, size, limit, error, calls):
    from functools import partial

    from labga.evaluation import EvaluationError

    pytest.importorskip("torch")
    log = tmp_path / "calls"
    config = CPUBackedConfig(
        partial(capacity_factory, log, limit, error), device="cuda", batch_size=size
    )
    with EvaluationSession(Objective(config, [0]), workers=1, genome=NumericGenome(1)) as runner:
        with pytest.raises(EvaluationError) as caught:
            runner.evaluate(np.zeros((4, 1)))
    assert len(caught.value.failed_indices) == calls[-1]
    assert list(map(int, log.read_text().splitlines())) == calls


def gpu_factory_error(*, device):
    raise RuntimeError("device owner initialization failed")


def test_gpu_owner_initialization_failure_cleans_up():
    import multiprocessing as mp

    from labga.evaluation import EvaluationError

    before = {p.pid for p in mp.active_children()}
    objective = Objective(CPUBackedConfig(gpu_factory_error, device="cuda"), [0])
    with EvaluationSession(objective, workers=3, genome=NumericGenome(1)) as runner:
        with pytest.raises(EvaluationError, match="device owner initialization failed"):
            runner.evaluate(np.zeros((5, 1)))
    assert {p.pid for p in mp.active_children()} == before
