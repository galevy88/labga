"""Return values for a completed optimization."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from labga._types import FloatArray


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Summarize an evaluated population after any scheduled migration.

    Attributes:
        generation: Zero for the initial population, then one per update.
        best_loss: Lowest loss in this generation.
        mean_loss: Mean across all candidates in this generation.
        best_so_far: Lowest loss from initialization through this generation.
        island_best: Lowest loss per island, in island order.
        island_mean: Mean loss per island, in island order.
        migrated: Whether this generation ended with ring migration.
    """

    generation: int
    best_loss: float
    mean_loss: float
    best_so_far: float
    island_best: tuple[float, ...]
    island_mean: tuple[float, ...]
    migrated: bool


@dataclass(frozen=True, slots=True)
class Population:
    """Store final candidates and their associated scores and island IDs.

    Attributes:
        solutions: Candidate matrix in genome encoding. Each row is one solution.
        losses: Finite scalar loss for each solution, in the same row order.
        island_ids: Zero-based island ID for each solution. A standard GA uses ID zero.
    """

    solutions: NDArray[Any]
    losses: FloatArray
    island_ids: NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Store the best evaluated solution, compact history, and final population.

    Attributes:
        best_solution: Owned vector in genome encoding for the best candidate observed.
        best_prediction: Owned vector produced by evaluating that exact candidate.
        best_loss: Scalar loss associated with the best solution and prediction.
        history: Generation summaries ordered from initialization onward.
        population: Final candidates with their losses and island membership.
        metadata: JSON-serializable settings, evaluation counts, and run timing.
    """

    best_solution: NDArray[Any]
    best_prediction: FloatArray
    best_loss: float
    history: tuple[GenerationRecord, ...]
    population: Population
    metadata: dict[str, object]
