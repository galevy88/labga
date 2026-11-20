"""Population allocation and synchronous ring migration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from labga._types import FloatArray
from labga.genomes import _checked_integer


@dataclass(frozen=True, slots=True)
class Migration:
    """Configure synchronous migration between neighboring islands.

    Args:
        interval: Apply migration after every `interval` generations.
        count: Number of candidates copied from each island to its successor.
        topology: Migration topology. The ring sends island i to (i + 1) modulo
            the number of islands.
    """

    interval: int = 10
    count: int = 1
    topology: Literal["ring"] = "ring"

    def __post_init__(self) -> None:
        object.__setattr__(self, "interval", _checked_integer(self.interval, "interval", 1))
        object.__setattr__(self, "count", _checked_integer(self.count, "count", 1))
        if not isinstance(self.topology, str) or self.topology != "ring":
            raise ValueError("topology must be 'ring'")


@dataclass(frozen=True, slots=True)
class _ScoredPopulation:
    solutions: FloatArray
    predictions: FloatArray
    losses: FloatArray


def _allocate_islands(population_size: int, n_islands: int) -> list[int]:
    size, remainder = divmod(population_size, n_islands)
    if size < 2:
        raise ValueError("population_size must provide at least two candidates per island")
    return [size + (island < remainder) for island in range(n_islands)]


def _migrate_ring(islands: list[_ScoredPopulation], *, count: int, elite_count: int) -> None:
    # Integer-array indexing snapshots the three corresponding arrays before any arrivals.
    donors: list[_ScoredPopulation] = []
    for island in islands:
        best = np.argsort(island.losses, kind="stable")[:count]
        donors.append(
            _ScoredPopulation(island.solutions[best], island.predictions[best], island.losses[best])
        )

    for destination, island in enumerate(islands):
        # Elites occupy the first slots. Among tied losses, keep the earlier rows.
        ranked = np.argsort(island.losses[elite_count:], kind="stable")
        replaced = ranked[-count:][::-1] + elite_count
        migrant = donors[destination - 1]
        island.solutions[replaced] = migrant.solutions
        island.predictions[replaced] = migrant.predictions
        island.losses[replaced] = migrant.losses
