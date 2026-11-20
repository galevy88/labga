import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from labga.evaluation import EvaluationError, EvaluationSession
from labga.genomes import BinaryGenome, NumericGenome
from labga.objective import Objective
from labga.predictors import Predictor


class Scalar:
    def predict(self, x):
        assert x.dtype == np.int8
        return np.array([2 * x[0] + x[1]])


@dataclass
class Factory:
    folder: str

    def __call__(self, *, device):
        assert device == "cpu"
        with (Path(self.folder) / str(os.getpid())).open("a") as out:
            out.write("built\n")
        return Scalar()


def factory_failure(*, device):
    raise RuntimeError("factory broke")


def delayed(row):
    time.sleep(4)
    return row


def exit_process(row):
    if row[0] == 1:
        time.sleep(0.15)
        os._exit(7)
    return row


def selective(row):
    if row[0] == 1:
        raise ValueError("individual failure")
    return row


class PartialBatch:
    def predict(self, x):
        raise AssertionError("unexpected scalar inference")

    def predict_batch(self, x):
        values = x[:, :1].astype(float)
        values[values == 1] = np.nan
        return values


def session(model, workers, *, genome=None, **kwargs):
    return EvaluationSession(
        Objective(model, [0]), workers=workers, genome=genome or NumericGenome(1), **kwargs
    )


def children():
    return {child.pid for child in mp.active_children()}


def test_spawn_lifecycle_order_and_binary_encoding(tmp_path):
    population = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=float)
    before = children()
    with session(Predictor(Factory(str(tmp_path))), 2, genome=BinaryGenome(2)) as runner:
        for _ in range(2):
            outcome = runner.evaluate(population)
            np.testing.assert_array_equal(outcome.losses, [4, 0, 9, 1])
            np.testing.assert_array_equal(outcome.predictions.ravel(), [2, 0, 3, 1])
        assert runner.evaluation_count == 8
        assert runner.metadata["workers_actual"] == 2
        assert runner.metadata["batch_fallback"] is True
    assert len(list(tmp_path.iterdir())) == 2
    assert all(p.read_text() == "built\n" for p in tmp_path.iterdir())
    assert children() == before


def test_serial_closure_and_preprocess():
    offset = 2
    objective = Objective(lambda row: [row[0] + offset], [1], preprocess=lambda x: x * 2)
    with EvaluationSession(objective, workers=1, genome=NumericGenome(1)) as runner:
        np.testing.assert_equal(runner.evaluate(np.array([[0.5], [1]])).losses, [4, 9])


@pytest.mark.parametrize("model", [selective, PartialBatch()])
def test_completed_results_and_candidate_errors(model):
    with session(model, 1) as runner, pytest.raises(EvaluationError) as caught:
        runner.evaluate(np.array([[0.5], [1], [0.25]]))
    err = caught.value
    assert err.failed_indices == (1,)
    assert set(err.completed) >= {0}
    assert err.completed[0][1] == 0.25
    err.with_context(generation=3, island_ids=[0, 2, 1])
    assert "generation 3" in str(err) and "island 2" in str(err)


def test_worker_exit_preserves_completed_and_closes_children():
    before = children()
    with session(exit_process, 2) as runner, pytest.raises(EvaluationError) as caught:
        runner.evaluate(np.array([[0.25], [1.0]]))
    assert set(caught.value.completed) == {0}
    assert caught.value.failed_indices == (1,)
    assert children() == before


def test_initialization_error_cleans_up():
    before = children()
    with (
        pytest.raises(EvaluationError, match="factory broke"),
        session(Predictor(factory_failure), 2) as runner,
    ):
        runner.evaluate(np.array([[0.0], [1.0]]))
    assert children() == before


def test_worker_timeout_cleans_up():
    before = children()
    with (
        pytest.raises(EvaluationError, match="timed out"),
        session(delayed, 2, timeout=1.5) as runner,
    ):
        runner.evaluate(np.array([[0.0], [1.0]]))
    assert children() == before


def identity(row):
    return row


def batch_factory(*, device):
    return PartialBatch()


def test_auto_budget_detection_and_cap(monkeypatch):
    from labga import _resources, evaluation

    monkeypatch.setattr(_resources.os, "process_cpu_count", lambda: 8, raising=False)
    monkeypatch.setattr(_resources.os, "sched_getaffinity", lambda _: set(range(6)), raising=False)
    monkeypatch.setattr(_resources, "_cgroup_limit", lambda: 2)
    budget, details = _resources.available_cpus()
    assert budget == 2
    assert (details["process"], details["affinity"], details["quota"]) == (8, 6, 2)
    monkeypatch.setattr(evaluation, "available_cpus", lambda: (20, {}))
    with session(Predictor(batch_factory, batch_size=3), "auto") as runner:
        assert runner.evaluate(np.zeros((7, 1))).losses.shape == (7,)
        assert runner.metadata["workers_actual"] == 3


def test_spawn_batch_and_scalar_equal():
    x = np.arange(10).reshape(-1, 1) / 10
    with (
        session(Predictor(batch_factory, batch_size=2), 3) as batched,
        session(identity, 2) as scalar,
    ):
        batch = batched.evaluate(x)
        single = scalar.evaluate(x)
        np.testing.assert_equal(batch.predictions, single.predictions)
        np.testing.assert_equal(batch.losses, single.losses)


def test_batch_exception_fails_submitted_chunk_only():
    class BatchError:
        def __call__(self, x):
            raise AssertionError("fallback forbidden")

        def predict_batch(self, x):
            if np.any(x):
                raise LookupError("batch failed")
            return x

    with session(Predictor(lambda *, device: BatchError(), batch_size=2), 1) as runner:
        with pytest.raises(EvaluationError) as caught:
            runner.evaluate(np.array([[0.0], [0.0], [0.0], [1.0], [0.0]]))
    assert caught.value.failed_indices == (2, 3)
    assert caught.value.skipped_indices == (4,)
    assert set(caught.value.completed) == {0, 1}


def test_quota_parsing(tmp_path):
    from labga._resources import _quota

    (tmp_path / "cpu.max").write_text("150000 100000")
    assert _quota(tmp_path) == 2
    (tmp_path / "cpu.max").write_text("max 100000")
    assert _quota(tmp_path) is None
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu.cfs_quota_us").write_text("50000")
    (tmp_path / "cpu.cfs_period_us").write_text("100000")
    assert _quota(tmp_path) == 1


@pytest.mark.parametrize("budget", [False, 0, 2.5, -1, "3"])
def test_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="workers"):
        session(identity, budget)


def test_custom_loss_errors_preserve_peers_and_call_each_loss_once():
    seen = []

    def loss(x, target):
        seen.append(x[0])
        if x[0] == 0.25:
            raise LookupError("bad loss")
        return float(x[0])

    objective = Objective(PartialBatch(), [0], loss=loss)
    with EvaluationSession(objective, workers=1, genome=NumericGenome(1)) as runner:
        with pytest.raises(EvaluationError) as caught:
            runner.evaluate(np.array([[0.0], [0.25], [0.5]]))
    assert caught.value.failed_indices == (1,)
    assert set(caught.value.completed) == {0, 2}
    assert seen == [0.0, 0.25, 0.5]


def test_ragged_return_localizes_the_invalid_row():
    class Ragged:
        def predict(self, row):
            return row

        def predict_batch(self, x):
            return [[0.25], [0.25, 0.5], [0.5]]

    with session(Ragged(), 1) as runner, pytest.raises(EvaluationError) as caught:
        runner.evaluate(np.zeros((3, 1)))
    assert caught.value.failed_indices == (1,)
    assert set(caught.value.completed) == {0, 2}


def test_coordinator_interrupt_cleans_up(monkeypatch):
    from labga import _worker_pool

    before = children()
    with session(identity, 2) as runner:
        runner.evaluate(np.zeros((2, 1)))

        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(_worker_pool, "wait", interrupt)
        with pytest.raises(KeyboardInterrupt):
            runner.evaluate(np.zeros((2, 1)))
        assert children() == before
