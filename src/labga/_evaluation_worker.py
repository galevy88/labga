"""Model evaluation and preprocessing executed exclusively by their owners."""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from labga._types import FloatArray
from labga.objective import Objective, Preprocess, _real_array
from labga.predictors import Predictor

Array = NDArray[Any]


@dataclass
class Job:
    """Candidate IDs and their encoded inputs for a bounded submission."""

    ids: tuple[int, ...]
    rows: Array


@dataclass
class Reply:
    """Row-level evidence and current adaptive capacity from one submission."""

    completed: dict[int, tuple[FloatArray, float]] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    prepared: dict[int, Array] = field(default_factory=dict)
    widths: dict[int, int] = field(default_factory=dict)
    cap: int = 1


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def prepare(job: Job, preprocess: Preprocess) -> Reply:
    reply = Reply()
    for index, row in zip(job.ids, job.rows, strict=True):
        try:
            value = _real_array(preprocess(row.copy()), "preprocess")
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise ValueError("preprocess must return a nonempty finite numeric vector")
            reply.prepared[index] = value.copy()
            reply.widths[index] = len(value)
        except Exception as exc:
            reply.errors[index] = describe(exc)
    return reply


@dataclass(frozen=True)
class WorkerSpec:
    """Only preprocessing configuration is sent to preprocessing children."""

    objective: Objective | None = None
    preprocess: Preprocess | None = None
    cap: int = 64


class Evaluator:
    """Persistent predictor state shared by all requests served by one owner."""

    def __init__(self, objective: Objective, cap: int) -> None:
        self.objective = objective
        config = objective.predictor
        self.model = config.build() if isinstance(config, Predictor) else config
        self.adaptive = (
            isinstance(config, Predictor)
            and config.device.startswith("cuda")
            and config.batch_size == "auto"
        )
        self.predict_batch = getattr(self.model, "predict_batch", None)
        self.batched = callable(self.predict_batch)
        self.cap = cap if self.batched else 1

    def _score(self, ids: tuple[int, ...], raw: Any, reply: Reply) -> None:
        expected = (len(ids), len(np.asarray(self.objective.target)))
        if not hasattr(raw, "shape"):
            if isinstance(raw, (list, tuple)):
                # Infer a common array only when this preserves real-valued peer rows.
                # A ragged, complex, or textual row must be diagnosed independently.
                try:
                    array = np.asarray(raw)
                except (TypeError, ValueError):
                    pass
                else:
                    if array.dtype.kind in "biuf":
                        raw = array
            else:
                raw = np.asarray(raw)
        shape = getattr(raw, "shape", None)
        row_count = len(raw) if shape is None or len(shape) else None
        if row_count != len(ids):
            raise ValueError(f"prediction shape must be {expected}; got {shape}")
        if shape == expected and isinstance(self.objective.loss, str):
            try:
                batch = self.objective.score(raw, count=len(ids))
            except ValueError:
                # Localize invalid predictions or overflowing built-in losses.
                # Neither model calls nor user-defined losses are repeated.
                pass
            else:
                for index, pred, loss in zip(ids, batch.predictions, batch.losses, strict=True):
                    reply.completed[index] = (pred, float(loss))
                return

        for pos, index in enumerate(ids):
            try:
                row = raw[pos]
                if not hasattr(row, "shape"):
                    row = np.asarray(row)
                single = self.objective.score(row[None, ...], count=1)
                reply.completed[index] = (single.predictions[0], float(single.losses[0]))
            except Exception as exc:
                reply.errors[index] = describe(exc)

    def _inference(self, job: Job, reply: Reply) -> None:
        if not self.batched:
            single = cast(Callable[[Array], Any], getattr(self.model, "predict", self.model))
            for index, row in zip(job.ids, job.rows, strict=True):
                try:
                    raw = single(row.copy())
                    if not hasattr(raw, "shape"):
                        raw = np.asarray(raw)
                    self._score((index,), raw[None, ...], reply)
                except Exception as exc:
                    reply.errors[index] = describe(exc)
            return

        predict_batch = cast(Callable[[Array], Any], self.predict_batch)
        start = 0
        while start < len(job.ids):
            ids = job.ids[start : start + self.cap]
            try:
                raw = predict_batch(job.rows[start : start + len(ids)].copy())
            except Exception as exc:
                torch = sys.modules.get("torch")
                if (
                    self.adaptive
                    and len(ids) > 1
                    and torch is not None
                    and isinstance(exc, torch.cuda.OutOfMemoryError)
                ):
                    self.cap = max(1, len(ids) // 2)
                    continue
                reply.errors.update(dict.fromkeys(ids, describe(exc)))
                break
            try:
                self._score(ids, raw, reply)
            except Exception as exc:
                reply.errors.update(dict.fromkeys(ids, describe(exc)))
            del raw  # Release device output before allocating the next batch.
            if reply.errors:
                break
            start += len(ids)

    def __call__(self, job: Job) -> Reply:
        reply = Reply(cap=self.cap)
        if self.objective.preprocess is not None:
            prepared = prepare(job, self.objective.preprocess)
            reply.errors.update(prepared.errors)
            reply.widths.update(prepared.widths)
            if not prepared.prepared:
                return reply
            width = next(iter(prepared.widths.values()))
            ids = []
            rows = []
            for index, row in prepared.prepared.items():
                if len(row) != width:
                    reply.errors[index] = "preprocess must return fixed-length vectors"
                else:
                    ids.append(index)
                    rows.append(row)
            job = Job(tuple(ids), np.stack(rows))
        self._inference(job, reply)
        reply.cap = self.cap
        return reply
