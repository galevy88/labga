"""Settings and offspring generation for generational genetic algorithms."""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from labga._types import FloatArray, Genome
from labga.genomes import (
    _as_real_array,
    _checked_integer,
    _checked_nonnegative,
    _checked_probability,
)


@dataclass(frozen=True, slots=True)
class Operators:
    """Configure tournament selection, uniform crossover, mutation, and elitism.

    Args:
        tournament_size: Maximum number of distinct candidates per tournament.
        crossover_probability: Probability of uniform crossover per parent pair.
        mutation_probability: Per-coordinate mutation probability. The default
            is the reciprocal of the genome size.
        mutation_scale: Numeric mutation standard deviation as a fraction of
            the coordinate's range.
        elite_count: Number of unchanged candidates to keep per population.
    """

    tournament_size: int = 3
    crossover_probability: float = 0.9
    mutation_probability: float | None = None
    mutation_scale: float = 0.1
    elite_count: int = 1

    def __post_init__(self) -> None:
        tournament_size = _checked_integer(self.tournament_size, "tournament_size", 1)
        crossover = _checked_probability(self.crossover_probability, "crossover_probability")
        mutation = self.mutation_probability
        if mutation is not None:
            mutation = _checked_probability(mutation, "mutation_probability")
        scale = _checked_nonnegative(self.mutation_scale, "mutation_scale")
        elite_count = _checked_integer(self.elite_count, "elite_count", 0)

        object.__setattr__(self, "tournament_size", tournament_size)
        object.__setattr__(self, "crossover_probability", crossover)
        object.__setattr__(self, "mutation_probability", mutation)
        object.__setattr__(self, "mutation_scale", scale)
        object.__setattr__(self, "elite_count", elite_count)


def make_offspring(
    population: FloatArray,
    losses: FloatArray,
    count: int,
    genome: Genome,
    operators: Operators,
    rng: Generator,
) -> FloatArray:
    """Select parents, apply uniform crossover, and mutate exactly `count` children.

    Args:
        population: Current candidate matrix.
        losses: One finite scalar loss per parent candidate. Lower is better.
        count: Nonnegative number of children to generate.
        genome: Search space implementing validation and mutation.
        operators: Operator configuration. The caller handles elite retention.
        rng: Random generator used for all variation.

    Returns:
        An owned float64 matrix of shape `(count, genome.size)`.
    """
    count = _checked_integer(count, "count", 0)
    parents = genome.validate(population)
    scores = _as_real_array(losses, "losses")
    if scores.shape != (len(parents),) or not np.isfinite(scores).all():
        raise ValueError("losses must contain one finite scalar per parent")
    if count == 0:
        return np.empty((0, genome.size), dtype=np.float64)

    # Tournament draws are independent. A tournament uses each entrant at most once.
    tournament_size = min(operators.tournament_size, len(parents))
    pair_count = (count + 1) // 2
    parent_indices = np.empty((pair_count, 2), dtype=np.intp)
    for pair in parent_indices:
        for role in range(2):
            entrants = rng.choice(len(parents), size=tournament_size, replace=False)
            pair[role] = entrants[np.argmin(scores[entrants])]

    pairs = parents[parent_indices]
    crossover = rng.random(pair_count) < operators.crossover_probability
    selected = pairs[crossover]
    swap = rng.random((len(selected), genome.size)) < 0.5
    pairs[crossover] = np.where(swap[:, None, :], selected[:, ::-1, :], selected)
    children = pairs.reshape(-1, genome.size)[:count]

    probability = operators.mutation_probability
    if probability is None:
        probability = 1 / genome.size
    return genome.mutate(children, rng, probability, operators.mutation_scale)
