import numpy as np
import pytest

from labga.genomes import BinaryGenome, NumericGenome
from labga.operators import Operators, make_offspring


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tournament_size": 0},
        {"tournament_size": True},
        {"crossover_probability": 1.01},
        {"crossover_probability": [0.5]},
        {"mutation_probability": -0.1},
        {"mutation_probability": np.nan},
        {"mutation_scale": -1.0},
        {"mutation_scale": np.inf},
        {"elite_count": True},
        {"elite_count": -1},
    ],
)
def test_invalid_operator_options(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        Operators(**kwargs)


def test_tournament_selects_lowest_loss_when_it_covers_population() -> None:
    population = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    losses = np.array([10.0, 3.0, 9.0])
    settings = Operators(tournament_size=12, crossover_probability=0, mutation_probability=0)

    children = make_offspring(
        population, losses, 7, BinaryGenome(2), settings, np.random.default_rng(5)
    )

    np.testing.assert_array_equal(children, np.zeros((7, 2)))


def test_uniform_crossover_recombines_parent_coordinates() -> None:
    population = np.array([[0.0] * 12, [1.0] * 12])
    settings = Operators(tournament_size=1, crossover_probability=1, mutation_probability=0)

    children = make_offspring(
        population, np.array([0.0, 1.0]), 75, BinaryGenome(12), settings, np.random.default_rng(6)
    )

    assert children.shape == (75, 12)
    assert np.all((children == 0) | (children == 1))
    assert np.any((children.sum(axis=1) > 0) & (children.sum(axis=1) < 12))


def test_offspring_mutation_leaves_parent_storage_untouched() -> None:
    population = np.array([[0.0, 1.0], [1.0, 1.0]])
    losses = np.array([1.0, 2.0])
    settings = Operators(tournament_size=2, crossover_probability=0, mutation_probability=1)

    children = make_offspring(
        population, losses, 3, BinaryGenome(2), settings, np.random.default_rng(4)
    )

    np.testing.assert_array_equal(children, [[1.0, 0.0]] * 3)
    np.testing.assert_array_equal(population, [[0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(losses, [1.0, 2.0])
    assert not np.shares_memory(children, population)


def test_default_mutation_probability_is_inverse_genome_size() -> None:
    population = np.zeros((2, 20))

    children = make_offspring(
        population,
        np.zeros(2),
        1000,
        BinaryGenome(20),
        Operators(),
        np.random.default_rng(5),
    )

    assert 0.04 < children.mean() < 0.06


def test_offspring_accepts_zero_count() -> None:
    children = make_offspring(
        np.array([[0.5], [0.75]]),
        np.array([1.0, 2.0]),
        0,
        NumericGenome(1),
        Operators(),
        np.random.default_rng(10),
    )

    assert children.shape == (0, 1)
    assert children.dtype == np.float64


@pytest.mark.parametrize(
    ("losses", "count", "label"),
    [
        (np.array([1.0]), 1, "losses"),
        (np.array([[1.0], [2.0]]), 1, "losses"),
        (np.array([1.0, np.inf]), 1, "losses"),
        (np.array([1.0, 2.0]), -1, "count"),
        (np.array([1.0, 2.0]), True, "count"),
    ],
)
def test_offspring_rejects_invalid_inputs(losses: np.ndarray, count: int, label: str) -> None:
    with pytest.raises(ValueError, match=label):
        make_offspring(
            np.array([[0.0], [1.0]]),
            losses,
            count,
            BinaryGenome(1),
            Operators(),
            np.random.default_rng(5),
        )
