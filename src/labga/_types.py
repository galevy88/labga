"""Internal array and search-space contracts."""

from typing import Any, Protocol

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class Genome(Protocol):
    """Minimal search-space behavior consumed by the optimizer."""

    @property
    def size(self) -> int: ...

    def sample(self, count: int, rng: Generator) -> FloatArray: ...

    def validate(self, population: ArrayLike) -> FloatArray: ...

    def mutate(
        self, population: FloatArray, rng: Generator, probability: float, scale: float
    ) -> FloatArray: ...

    def encode(self, candidate: FloatArray) -> NDArray[Any]: ...
