"""A spawn pool with one outstanding request per child and bounded waits."""

import multiprocessing as mp
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from queue import SimpleQueue
from threading import Thread

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


# The timestamp is captured after reading and unpickling the entire message.
_Response = Future[tuple[float, object]]


class Child:
    """Exchange messages on one dedicated thread so partial frames never block the scheduler."""

    def __init__(self, process: BaseProcess, wire: Connection) -> None:
        self.process = process
        self.wire = wire
        self.ready: _Response = Future()
        self.requests: SimpleQueue[tuple[Job, _Response] | None] = SimpleQueue()
        self.thread = Thread(target=self._exchange, name=f"labga-ipc-{process.pid}", daemon=True)

    def _exchange(self) -> None:
        response = self.ready
        try:
            status = self.wire.recv()
            response.set_result((time.monotonic(), status))
            while (request := self.requests.get()) is not None:
                job, response = request
                self.wire.send(job)
                reply = self.wire.recv()
                response.set_result((time.monotonic(), reply))
        except Exception as exc:
            response.set_result((time.monotonic(), exc))
        finally:
            # Closing the last parent endpoint also tells an idle child to exit.
            self.wire.close()

    def submit(self, job: Job) -> _Response:
        response: _Response = Future()
        self.requests.put((job, response))
        return response


def _responses(deadlines: dict[_Response, float]) -> Iterator[tuple[_Response, object]]:
    """Yield completed or timed-out exchanges, including ones completed after their deadline."""
    delay = max(0.0, min(deadlines.values()) - time.monotonic())
    wait(deadlines, delay, FIRST_COMPLETED)
    now = time.monotonic()
    for future, deadline in list(deadlines.items()):
        result: object = TimeoutError("timed out")
        if future.done():
            received, payload = future.result()
            if received <= deadline:
                result = payload
        elif deadline > now:
            continue
        del deadlines[future]
        yield future, result


class SpawnPool:
    """Own every process and its transport, including partially initialized children."""

    def __init__(self, spec: WorkerSpec, timeout: float) -> None:
        self.spec = spec
        self.timeout = timeout
        self.children: list[Child] = []
        self.batched: bool | None = None

    def ensure(self, count: int) -> None:
        ctx = mp.get_context("spawn")
        starting: dict[_Response, float] = {}
        try:
            while len(self.children) < count:
                parent, child_wire = ctx.Pipe()
                process = ctx.Process(target=_serve, args=(child_wire, self.spec))
                deadline = time.monotonic() + self.timeout
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child_wire.close()
                child = Child(process, parent)
                self.children.append(child)
                starting[child.ready] = deadline
                child.thread.start()

            while starting:
                for _, status in _responses(starting):
                    if isinstance(status, TimeoutError):
                        raise RuntimeError("worker initialization timed out")
                    if isinstance(status, EOFError):
                        raise RuntimeError("worker exited during initialization") from status
                    if not isinstance(status, bool):
                        raise RuntimeError(f"worker initialization failed: {status}")
                    if self.batched is not None and self.batched != status:
                        raise RuntimeError("factory returned inconsistent batch capabilities")
                    self.batched = status
        except BaseException:
            self.close(abort=True)
            raise

    def run(self, jobs: Iterable[Job]) -> Iterator[Reply]:
        remaining = iter(jobs)
        idle = list(reversed(self.children))
        pending: dict[_Response, tuple[Child, Job]] = {}
        deadlines: dict[_Response, float] = {}
        submitting = True
        while pending or (idle and submitting):
            while idle and submitting:
                job = next(remaining, None)
                if job is None:
                    submitting = False
                    break
                child = idle.pop()
                deadline = time.monotonic() + self.timeout
                future = child.submit(job)
                pending[future] = (child, job)
                deadlines[future] = deadline
            if not pending:
                break
            for future, result in _responses(deadlines):
                child, job = pending.pop(future)
                if not isinstance(result, Reply):
                    message = (
                        "worker request timed out"
                        if isinstance(result, TimeoutError)
                        else f"worker exited: {result}"
                    )
                    result = Reply(errors=dict.fromkeys(job.ids, message))
                if result.errors:
                    submitting = False
                else:
                    idle.append(child)
                yield result

    def close(self, *, abort: bool = False) -> None:
        for child in self.children:
            child.requests.put(None)
            if abort and child.process.is_alive():
                child.process.terminate()
        end = time.monotonic() + 2
        for child in self.children:
            child.process.join(max(0.0, end - time.monotonic()))
        for child in self.children:
            if child.process.is_alive():
                child.process.kill()
        end = time.monotonic() + 2
        for child in self.children:
            child.process.join(max(0.0, end - time.monotonic()))
            if not child.process.is_alive():
                child.process.close()
            if child.thread.ident is not None:
                child.thread.join(max(0.0, end - time.monotonic()))
            else:
                child.wire.close()
        self.children.clear()
