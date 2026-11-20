"""A spawn pool with one outstanding request per child and bounded waits."""

import multiprocessing as mp
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import cast

from threadpoolctl import threadpool_limits

from labga._evaluation_worker import Evaluator, Job, Reply, WorkerSpec, describe, prepare


def _serve(wire: Connection, spec: WorkerSpec) -> None:
    try:
        with threadpool_limits(limits=1):
            run: Callable[[Job], Reply]
            if spec.objective is None:
                assert spec.preprocess is not None
                transform = spec.preprocess

                def run(job: Job) -> Reply:
                    return prepare(job, transform)

                batched = True
            else:
                owner = Evaluator(spec.objective, spec.cap)
                run = owner
                batched = owner.batched
            torch = sys.modules.get("torch")
            if torch is not None:
                torch.set_num_threads(1)
            # Apply the limit again to native libraries imported by the factory.
            with threadpool_limits(limits=1):
                wire.send(batched)
                while (job := wire.recv()) is not None:
                    wire.send(run(job))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            wire.send(describe(exc))
        except (OSError, EOFError):
            pass
    finally:
        wire.close()


@dataclass
class Child:
    process: BaseProcess
    wire: Connection


class SpawnPool:
    """Own every process and its transport, including partially initialized children."""

    def __init__(self, spec: WorkerSpec, timeout: float) -> None:
        self.spec = spec
        self.timeout = timeout
        self.children: list[Child] = []
        self.batched: bool | None = None

    def ensure(self, count: int) -> None:
        ctx = mp.get_context("spawn")
        starting = {}
        try:
            while len(self.children) < count:
                parent, child = ctx.Pipe()
                process = ctx.Process(target=_serve, args=(child, self.spec))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.children.append(Child(process, parent))
                starting[parent] = time.monotonic() + self.timeout

            while starting:
                delay = max(0.0, min(starting.values()) - time.monotonic())
                ready = cast(list[Connection], wait(list(starting), delay))
                if not ready:
                    raise RuntimeError("worker initialization timed out")
                for wire in ready:
                    try:
                        status = wire.recv()
                    except EOFError as exc:
                        raise RuntimeError("worker exited during initialization") from exc
                    if not isinstance(status, bool):
                        raise RuntimeError(f"worker initialization failed: {status}")
                    if self.batched is not None and self.batched != status:
                        raise RuntimeError("factory returned inconsistent batch capabilities")
                    self.batched = status
                    del starting[wire]
        except BaseException:
            self.close(abort=True)
            raise

    def run(self, jobs: Iterable[Job]) -> Iterator[Reply]:
        remaining = iter(jobs)
        idle = [child.wire for child in reversed(self.children)]
        pending: dict[Connection, tuple[Job, float]] = {}
        submitting = True
        while pending or (idle and submitting):
            while idle and submitting:
                job = next(remaining, None)
                if job is None:
                    submitting = False
                    break
                wire = idle.pop()
                try:
                    wire.send(job)
                except (OSError, EOFError) as exc:
                    submitting = False
                    yield Reply(errors=dict.fromkeys(job.ids, f"worker exited: {exc}"))
                    break
                pending[wire] = (job, time.monotonic() + self.timeout)
            if not pending:
                break
            delay = max(0.0, min(deadline for _, deadline in pending.values()) - time.monotonic())
            ready = cast(list[Connection], wait(list(pending), delay))
            for wire in ready:
                job, _ = pending.pop(wire)
                try:
                    result = wire.recv()
                    if not isinstance(result, Reply):
                        raise RuntimeError(f"worker failed: {result}")
                except (OSError, EOFError, RuntimeError) as exc:
                    result = Reply(errors=dict.fromkeys(job.ids, f"worker exited: {exc}"))
                if result.errors:
                    submitting = False
                else:
                    idle.append(wire)
                yield result
            expired = [wire for wire, (_, end) in pending.items() if end <= time.monotonic()]
            for wire in expired:
                job, _ = pending.pop(wire)
                submitting = False
                yield Reply(errors=dict.fromkeys(job.ids, "worker request timed out"))

    def close(self, *, abort: bool = False) -> None:
        for child in self.children:
            if abort:
                if child.process.is_alive():
                    child.process.terminate()
            else:
                try:
                    child.wire.send(None)
                except (OSError, EOFError):
                    pass
            child.wire.close()
        end = time.monotonic() + 2
        for child in self.children:
            child.process.join(max(0.0, end - time.monotonic()))
        for child in self.children:
            if child.process.is_alive():
                child.process.kill()
        for child in self.children:
            child.process.join(2)
            if not child.process.is_alive():
                child.process.close()
        self.children.clear()
