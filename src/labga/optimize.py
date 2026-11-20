"""Coordinator for standard and synchronous island genetic algorithms."""

from dataclasses import asdict
from time import perf_counter
from typing import Literal, cast

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import ArrayLike

from labga._types import FloatArray, Genome
from labga.evaluation import EvaluationError, EvaluationSession
from labga.genomes import _checked_integer, _checked_real
from labga.islands import Migration, _allocate_islands, _migrate_ring, _ScoredPopulation
from labga.objective import Objective
from labga.operators import Operators, make_offspring
from labga.results import GenerationRecord, OptimizationResult, Population


def optimize(
    *,
    genome: Genome,
    objective: Objective,
    population_size: int = 100,
    generations: int = 100,
    distributed: bool = False,
    n_islands: int | None = None,
    migration: Migration | None = None,
    workers: int | Literal["auto"] = "auto",
    seed: int | None = None,
    initial_population: ArrayLike | None = None,
    operators: Operators | None = None,
    evaluation_timeout: float = 300.0,
) -> OptimizationResult:
    """Minimize a fixed objective with a standard or synchronous island GA.

    Args:
        genome: Search space used to generate, validate, and encode candidates.
        objective: Deterministic prediction pipeline, fixed target, and scalar loss.
        population_size: Total number of candidates across the run.
        generations: Number of evolutionary updates after evaluating generation zero.
        distributed: Enable separate synchronized island populations.
        n_islands: Number of islands in distributed mode, defaulting to four.
        migration: Island ring settings, defaulting to `Migration()` in island mode.
        workers: Positive CPU evaluation worker count or automatic budget.
        seed: Nonnegative random seed. If omitted, the resolved seed is returned in metadata.
        initial_population: Optional complete candidate matrix of shape
            `(population_size, genome.size)`. Island mode shuffles the rows before partitioning.
        operators: Selection, crossover, mutation, and elitism settings.
        evaluation_timeout: Positive finite timeout in seconds for multiprocessing responses.

    Returns:
        The best candidate observed, its corresponding prediction and loss,
        generation summaries, and the final population.

    Raises:
        ValueError: A configuration value or supplied initial population is invalid.
        EvaluationError: A candidate batch could not be evaluated.
    """
    started = perf_counter()
    population_size = _checked_integer(population_size, "population_size", 1)
    generations = _checked_integer(generations, "generations", 0)
    if not isinstance(distributed, bool):
        raise ValueError("distributed must be a boolean")
    if seed is not None:
        seed = _checked_integer(seed, "seed", 0)
    if isinstance(workers, str):
        if workers != "auto":
            raise ValueError("workers must be 'auto' or a positive integer")
    else:
        workers = _checked_integer(workers, "workers", 1)
    evaluation_timeout = _checked_real(evaluation_timeout, "evaluation_timeout")
    if evaluation_timeout <= 0:
        raise ValueError("evaluation_timeout must be positive")
    if operators is None:
        operators = Operators()
    if not isinstance(operators, Operators):
        raise ValueError("operators must be an Operators instance")
    island_sizes, migration = _resolve_islands(
        population_size, distributed, n_islands, migration, operators.elite_count
    )

    seed_sequence = SeedSequence(seed)
    streams = seed_sequence.spawn(len(island_sizes) + 1)
    generators = [default_rng(stream) for stream in streams[:-1]]
    if initial_population is None:
        populations = [
            genome.sample(size, rng) for size, rng in zip(island_sizes, generators, strict=True)
        ]
    else:
        try:
            provided = genome.validate(initial_population)
        except ValueError as error:
            raise ValueError(f"invalid initial_population: {error}") from error
        if len(provided) != population_size:
            raise ValueError(f"initial_population must contain exactly {population_size} rows")
        if distributed:
            provided = default_rng(streams[-1]).permutation(provided, axis=0)
        populations = list(np.split(provided, np.cumsum(island_sizes)[:-1]))

    history: list[GenerationRecord] = []
    migration_events: list[int] = []
    best_solution: FloatArray | None = None
    best_prediction: FloatArray | None = None
    best_loss = float("inf")
    with EvaluationSession(
        objective, workers=workers, genome=genome, timeout=evaluation_timeout
    ) as session:
        islands = _evaluate(session, populations, generation=0)
        for generation in range(generations + 1):
            if generation > 0:
                islands = _advance(islands, generators, genome, operators, session, generation)
            migrated = False
            if migration is not None and generation > 0 and generation % migration.interval == 0:
                _migrate_ring(islands, count=migration.count, elite_count=operators.elite_count)
                migration_events.append(generation)
                migrated = True

            best_rows = [int(np.argmin(island.losses)) for island in islands]
            island_best = tuple(
                float(island.losses[row]) for island, row in zip(islands, best_rows, strict=True)
            )
            leading = min(range(len(islands)), key=island_best.__getitem__)
            generation_best = island_best[leading]
            if generation_best < best_loss:
                best_loss = generation_best
                best_solution = islands[leading].solutions[best_rows[leading]].copy()
                best_prediction = islands[leading].predictions[best_rows[leading]].copy()

            history.append(
                GenerationRecord(
                    generation=generation,
                    best_loss=generation_best,
                    mean_loss=_mean_loss(np.concatenate([island.losses for island in islands])),
                    best_so_far=best_loss,
                    island_best=island_best,
                    island_mean=tuple(_mean_loss(island.losses) for island in islands),
                    migrated=migrated,
                )
            )

    assert best_solution is not None and best_prediction is not None
    solutions = np.concatenate([island.solutions for island in islands])
    return OptimizationResult(
        best_solution=genome.encode(best_solution),
        best_prediction=best_prediction,
        best_loss=best_loss,
        history=tuple(history),
        population=Population(
            solutions=np.stack([genome.encode(candidate) for candidate in solutions]),
            losses=np.concatenate([island.losses for island in islands]),
            island_ids=np.repeat(np.arange(len(islands), dtype=np.intp), island_sizes),
        ),
        metadata={
            **session.metadata,
            # Supplying an integer or None to SeedSequence produces integer entropy.
            "seed": cast(int, seed_sequence.entropy),
            "population_size": population_size,
            "generations": generations,
            "distributed": distributed,
            "n_islands": len(islands),
            "island_sizes": island_sizes,
            "migration": None if migration is None else asdict(migration),
            "migration_events": migration_events,
            "operators": asdict(operators),
            "evaluation_count": session.evaluation_count,
            "evaluation_timeout": evaluation_timeout,
            "elapsed_seconds": perf_counter() - started,
        },
    )


def _resolve_islands(
    population_size: int,
    distributed: bool,
    n_islands: int | None,
    migration: Migration | None,
    elite_count: int,
) -> tuple[list[int], Migration | None]:
    if distributed:
        n_islands = 4 if n_islands is None else _checked_integer(n_islands, "n_islands", 2)
        island_sizes = _allocate_islands(population_size, n_islands)
        if migration is None:
            migration = Migration()
        if not isinstance(migration, Migration):
            raise ValueError("migration must be a Migration instance")
    else:
        if n_islands is not None or migration is not None:
            raise ValueError("n_islands and migration require distributed=True")
        island_sizes = [population_size]

    smallest = min(island_sizes)
    if elite_count > smallest:
        raise ValueError("elite_count must not exceed the smallest population size")
    if migration is not None and migration.count > smallest - elite_count:
        raise ValueError("migration count must fit the non-elite slots on the smallest island")
    return island_sizes, migration


def _evaluate(
    session: EvaluationSession, populations: list[FloatArray], *, generation: int
) -> list[_ScoredPopulation]:
    try:
        batch = session.evaluate(np.concatenate(populations))
    except EvaluationError as error:
        island_ids = [island for island, population in enumerate(populations) for _ in population]
        error.with_context(generation=generation, island_ids=island_ids)
        raise

    evaluated: list[_ScoredPopulation] = []
    offset = 0
    for population in populations:
        end = offset + len(population)
        evaluated.append(
            _ScoredPopulation(population, batch.predictions[offset:end], batch.losses[offset:end])
        )
        offset = end
    return evaluated


def _advance(
    islands: list[_ScoredPopulation],
    generators: list[Generator],
    genome: Genome,
    operators: Operators,
    session: EvaluationSession,
    generation: int,
) -> list[_ScoredPopulation]:
    elite_count = operators.elite_count
    if len(islands) == 1 and len(islands[0].solutions) == elite_count:
        return islands

    offspring = [
        make_offspring(
            island.solutions,
            island.losses,
            len(island.solutions) - elite_count,
            genome,
            operators,
            rng,
        )
        for island, rng in zip(islands, generators, strict=True)
    ]
    children = _evaluate(session, offspring, generation=generation)
    next_islands: list[_ScoredPopulation] = []
    for parents, child in zip(islands, children, strict=True):
        elites = np.argsort(parents.losses, kind="stable")[:elite_count]
        next_islands.append(
            _ScoredPopulation(
                solutions=np.concatenate((parents.solutions[elites], child.solutions)),
                predictions=np.concatenate((parents.predictions[elites], child.predictions)),
                losses=np.concatenate((parents.losses[elites], child.losses)),
            )
        )
    return next_islands


def _mean_loss(losses: FloatArray) -> float:
    # Scaling avoids overflow even when the sum of finite losses cannot fit in float64.
    scale = float(np.max(np.abs(losses)))
    return 0.0 if scale == 0 else float(np.mean(losses / scale) * scale)
