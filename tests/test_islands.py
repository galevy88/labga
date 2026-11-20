import numpy as np
import pytest

from labga.genomes import NumericGenome
from labga.islands import Migration, _migrate_ring, _ScoredPopulation
from labga.objective import Objective
from labga.operators import Operators
from labga.optimize import optimize


def identity(candidate):
    return candidate


@pytest.mark.parametrize(("total", "sizes"), [(1000, [250, 250, 250, 250]), (11, [3, 3, 3, 2])])
def test_island_allocation_uses_total_population_budget(total, sizes):
    result = optimize(
        genome=NumericGenome(1),
        objective=Objective(identity, [0.3]),
        population_size=total,
        generations=0,
        distributed=True,
        workers=1,
        seed=6,
    )

    assert len(result.population.solutions) == total
    assert result.metadata["island_sizes"] == sizes
    assert result.metadata["evaluation_count"] == total
    np.testing.assert_array_equal(np.bincount(result.population.island_ids), sizes)


def test_initial_partition_uses_seed_and_keeps_each_provided_candidate_once():
    initial = np.arange(12).reshape(-1, 1)
    options = {
        "genome": NumericGenome(1, bounds=(0, 12)),
        "objective": Objective(identity, [0]),
        "population_size": len(initial),
        "generations": 0,
        "distributed": True,
        "workers": 1,
        "initial_population": initial,
    }

    first = optimize(**options, seed=22)
    repeated = optimize(**options, seed=22)
    changed = optimize(**options, seed=7)

    assert not np.array_equal(first.population.solutions, initial)
    assert not np.array_equal(first.population.solutions, changed.population.solutions)
    np.testing.assert_array_equal(first.population.solutions, repeated.population.solutions)
    np.testing.assert_array_equal(np.sort(first.population.solutions[:, 0]), np.arange(12))
    np.testing.assert_array_equal(initial[:, 0], np.arange(12))


def test_islands_start_from_independent_rng_streams():
    result = optimize(
        genome=NumericGenome(2),
        objective=Objective(identity, [0, 0]),
        population_size=12,
        generations=0,
        distributed=True,
        n_islands=3,
        workers=1,
        seed=89,
    )

    first, second, third = result.population.solutions.reshape(3, 4, 2)
    assert not np.array_equal(first, second)
    assert not np.array_equal(second, third)
    assert not np.array_equal(first, third)


def test_ring_migration_is_simultaneous_and_preserves_carried_elites():
    def scored(positions):
        # The first row is the carried elite. Deliberately give it the worst score.
        # The best offspring then expose accidental replacement or cascading copies.
        solutions = np.asarray(positions, dtype=float).reshape(-1, 1)
        return _ScoredPopulation(solutions, solutions * 2, solutions[:, 0].copy())

    islands = [scored([90, 1, 4]), scored([80, 10, 14]), scored([70, 20, 24])]
    _migrate_ring(islands, count=1, elite_count=1)

    for island, expected in zip(islands, [[90, 1, 20], [80, 10, 1], [70, 20, 10]], strict=True):
        np.testing.assert_array_equal(island.solutions[:, 0], expected)
        np.testing.assert_array_equal(island.predictions[:, 0], np.asarray(expected) * 2)
        np.testing.assert_array_equal(island.losses, expected)


def test_migration_history_and_evaluation_count():
    result = optimize(
        genome=NumericGenome(2),
        objective=Objective(identity, [0, 0]),
        population_size=11,
        generations=5,
        distributed=True,
        n_islands=3,
        migration=Migration(interval=2, count=2),
        workers=1,
        seed=8,
    )

    assert [row.generation for row in result.history] == [0, 1, 2, 3, 4, 5]
    assert [row.generation for row in result.history if row.migrated] == [2, 4]
    assert result.metadata["migration_events"] == [2, 4]
    assert result.metadata["evaluation_count"] == 11 + 5 * (11 - 3)
    assert all(len(row.island_best) == 3 for row in result.history)
    assert all(len(row.island_mean) == 3 for row in result.history)
    np.testing.assert_allclose(result.population.losses, (result.population.solutions**2).mean(1))


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"distributed": False, "n_islands": 3}, "n_islands"),
        ({"distributed": False, "migration": Migration()}, "migration"),
        ({"n_islands": 1}, "n_islands"),
        ({"n_islands": True}, "n_islands"),
        ({"n_islands": 5}, "at least two candidates"),
        ({"n_islands": 4, "migration": Migration(count=2)}, "count"),
        ({"n_islands": 4, "operators": Operators(elite_count=3)}, "elite_count"),
    ],
)
def test_invalid_island_configuration(override, field):
    def predictor(candidate):
        pytest.fail("Invalid configurations must never reach the predictor")

    options = {
        "genome": NumericGenome(1),
        "objective": Objective(predictor, [0.3]),
        "population_size": 8,
        "generations": 0,
        "distributed": True,
        "workers": 1,
        **override,
    }
    with pytest.raises(ValueError, match=field):
        optimize(**options)


@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"interval": 2.5}, "interval"),
        ({"interval": False}, "interval"),
        ({"topology": "full"}, "topology"),
    ],
)
def test_invalid_migration_settings(options, field):
    with pytest.raises(ValueError, match=field):
        Migration(**options)
