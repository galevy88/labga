"""Ordered evaluation with session-owned CPU workers or one spawned CUDA owner."""

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from types import TracebackType
from typing import Literal, Self

import numpy as np

from labga._evaluation_worker import Array, Evaluator, Job, Reply, WorkerSpec, describe, prepare
from labga._resources import available_cpus
from labga._types import FloatArray, Genome
from labga._worker_pool import SpawnPool
from labga.objective import EvaluationBatch, Objective
from labga.predictors import Predictor


class EvaluationError(RuntimeError):
    """Candidate failures with original input positions and completed row evidence."""

    def __init__(
        self,
        diagnostics: dict[int, str],
        completed: dict[int, tuple[FloatArray, float]],
        *,
        count: int,
    ) -> None:
        self.diagnostics = dict(sorted(diagnostics.items()))
        self.failed_indices = tuple(self.diagnostics)
        self.completed = dict(sorted(completed.items()))
        self.skipped_indices = tuple(
            sorted(set(range(count)) - completed.keys() - diagnostics.keys())
        )
        self.generation: int | None = None
        self.island_ids: dict[int, int] = {}
        super().__init__(self._message())

    def _message(self) -> str:
        entries = []
        for index in self.failed_indices[:4]:
            island = f" (island {self.island_ids[index]})" if index in self.island_ids else ""
            entries.append(f"candidate {index}{island}: {self.diagnostics[index]}")
        where = "" if self.generation is None else f" in generation {self.generation}"
        return f"Evaluation failed{where}: " + "; ".join(entries)

    def with_context(self, *, generation: int, island_ids: Sequence[int]) -> Self:
        """Attach the engine's generation and per-input island assignments."""
        self.generation = generation
        self.island_ids = {i: int(island_ids[i]) for i in self.failed_indices}
        self.args = (self._message(),)
        return self


def _jobs(ids: tuple[int, ...], rows: Array, size: int) -> Iterator[Job]:
    for start in range(0, len(ids), size):
        yield Job(ids[start : start + size], rows[start : start + size])


class EvaluationSession:
    """Manage persistent workers, preserve input order, and retain partial evidence.

    Explicit serial CPU evaluation runs in the caller. Spawned initialization and
    each outstanding request have independent timeouts in seconds.
    """

    def __init__(
        self,
        objective: Objective,
        *,
        workers: int | Literal["auto"] = "auto",
        genome: Genome,
        timeout: float = 300.0,
    ) -> None:
        available, detection = available_cpus()
        if workers == "auto":
            self._budget = available
        elif type(workers) is int and workers > 0:
            self._budget = workers
        else:
            raise ValueError("workers must be 'auto' or a positive integer")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("evaluation timeout must be a positive finite number")
        self._timeout = float(timeout)
        self._objective = objective
        self._genome = genome
        config = objective.predictor
        device = config.device if isinstance(config, Predictor) else "cpu"
        size = config.batch_size if isinstance(config, Predictor) else "auto"
        self._cap = 64 if size == "auto" else size
        self._gpu = device.startswith("cuda")
        self._active = False
        self._closed = False
        self._owner: SpawnPool | None = None
        self._prep: SpawnPool | None = None
        self._local: Evaluator | None = None
        self._width: int | None = None
        self.evaluation_count = 0
        self.metadata: dict[str, object] = {
            "workers_requested": workers,
            "worker_budget": self._budget,
            "cpu_detection": detection,
            "workers_actual": 0,
            "device": device,
            "gpu_owners": 0,
            "batch_size_requested": size,
            "batch_size_resolved": self._cap,
            "batch_fallback": None,
            "evaluation_timeout": timeout,
        }

    def __enter__(self) -> Self:
        if self._active or self._closed:
            raise RuntimeError("evaluation session has already been entered")
        self._active = True
        return self

    def _initialize(self, count: int) -> None:
        if not self._gpu and self._budget == 1:
            if self._local is None:
                self._local = Evaluator(self._objective, self._cap)
            batched = self._local.batched
            self.metadata["workers_actual"] = 1
        else:
            if self._owner is None:
                objective = (
                    replace(self._objective, preprocess=None) if self._gpu else self._objective
                )
                self._owner = SpawnPool(
                    WorkerSpec(objective=objective, cap=self._cap), self._timeout
                )
                self._owner.ensure(1)
            batched = bool(self._owner.batched)
            if self._gpu:
                self.metadata["gpu_owners"] = 1
            else:
                runnable = math.ceil(count / self._cap) if batched else count
                self._owner.ensure(min(runnable, self._budget))
                self.metadata["workers_actual"] = len(self._owner.children)
        if not batched:
            self._cap = 1
        self.metadata["batch_size_resolved"] = self._cap
        self.metadata["batch_fallback"] = not batched

    def _gpu_inputs(
        self, ids: tuple[int, ...], rows: Array, errors: dict[int, str]
    ) -> tuple[tuple[int, ...], Array]:
        transform = self._objective.preprocess
        if transform is None:
            return ids, rows
        replies: Iterable[Reply]
        if self._budget == 1:
            replies = (prepare(Job(ids, rows), transform),)
            self.metadata["workers_actual"] = 1
        else:
            workers = min(self._budget, len(ids))
            if self._prep is None:
                self._prep = SpawnPool(WorkerSpec(preprocess=transform), self._timeout)
            self._prep.ensure(workers)
            size = min(64, math.ceil(len(ids) / workers))
            replies = self._prep.run(_jobs(ids, rows, size))
            self.metadata["workers_actual"] = len(self._prep.children)
        values: dict[int, Array] = {}
        widths: dict[int, int] = {}
        for reply in replies:
            errors.update(reply.errors)
            values.update(reply.prepared)
            widths.update(reply.widths)
        self._check_widths(widths, errors)
        valid = tuple(i for i in ids if i in values and i not in errors)
        return valid, np.stack([values[i] for i in valid]) if valid else rows[:0]

    def _check_widths(self, widths: dict[int, int], errors: dict[int, str]) -> None:
        if widths and self._width is None:
            self._width = widths[min(widths)]
        for index, width in widths.items():
            if width != self._width:
                errors[index] = f"preprocess returned length {width}, expected {self._width}"

    def _replies(self, ids: tuple[int, ...], rows: Array) -> Iterator[Reply]:
        if self._gpu:
            assert self._owner is not None
            # The CUDA owner can shrink the batch cap on each reply.
            start = 0
            while start < len(ids):
                chunk = ids[start : start + self._cap]
                job = Job(chunk, rows[start : start + len(chunk)])
                reply = next(self._owner.run([job]))
                self._cap = reply.cap
                self.metadata["batch_size_resolved"] = self._cap
                yield reply
                if reply.errors:
                    break
                start += len(chunk)
        elif self._owner is not None:
            yield from self._owner.run(_jobs(ids, rows, self._cap))
        else:
            assert self._local is not None
            for job in _jobs(ids, rows, self._cap):
                reply = self._local(job)
                yield reply
                if reply.errors:
                    break

    def evaluate(self, population: FloatArray) -> EvaluationBatch:
        """Evaluate rows ordered by input position. Close all workers on failure."""
        if not self._active:
            raise RuntimeError("evaluate requires an open EvaluationSession context")
        try:
            rows = self._genome.validate(population)
            if not len(rows):
                raise ValueError("population must contain at least one candidate")
            encoded = np.stack([self._genome.encode(row) for row in rows])
            return self._collect(encoded)
        except BaseException:
            self._close(abort=True)
            raise

    def _collect(self, encoded: Array) -> EvaluationBatch:
        count = len(encoded)
        ids = tuple(range(count))
        completed: dict[int, tuple[FloatArray, float]] = {}
        errors: dict[int, str] = {}
        widths: dict[int, int] = {}
        try:
            self._initialize(count)
            if self._gpu:
                ids, encoded = self._gpu_inputs(ids, encoded, errors)
            for reply in self._replies(ids, encoded):
                completed.update(reply.completed)
                errors.update(reply.errors)
                widths.update(reply.widths)
        except Exception as exc:
            errors.update({i: describe(exc) for i in ids if i not in completed and i not in errors})
            if not errors:
                raise
        self._check_widths(widths, errors)
        for index in errors:
            completed.pop(index, None)
        self.evaluation_count += len(completed)
        if errors:
            raise EvaluationError(errors, completed, count=count)
        return EvaluationBatch(
            predictions=np.stack([completed[i][0] for i in range(count)]),
            losses=np.asarray([completed[i][1] for i in range(count)]),
        )

    def _close(self, *, abort: bool) -> None:
        self._active = False
        self._closed = True
        for pool in (self._prep, self._owner):
            if pool is not None:
                pool.close(abort=abort)
        self._local = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close(abort=exc_type is not None)
