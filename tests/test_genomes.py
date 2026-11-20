import numpy as np
import pytest

from labga.genomes import BinaryGenome, Normal, NumericGenome


@pytest.mark.parametrize(("probability", "bit"), [(0.0, 0), (1.0, 1)])
def test_binary_sampling_at_probability_endpoints(probability: float, bit: int) -> None:
    samples = BinaryGenome(4, probability).sample(3, np.random.default_rng(1))

    assert samples.dtype == np.float64
    np.testing.assert_array_equal(samples, np.full((3, 4), bit))


def test_binary_mutation_flips_bits_and_owns_result() -> None:
    genome = BinaryGenome(3)
    parents = np.array([[0.0, 1.0, 1.0]])
    rng = np.random.default_rng(12)

    flipped = genome.mutate(parents, rng, probability=1.0, scale=0.1)
    unchanged = genome.mutate(parents, rng, probability=0.0, scale=0.1)

    np.testing.assert_array_equal(flipped, [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(unchanged, parents)
    assert not np.shares_memory(unchanged, parents)
    np.testing.assert_array_equal(parents, [[0.0, 1.0, 1.0]])


def test_binary_validation_and_encoding_copy_input() -> None:
    genome = BinaryGenome(2)
    population = np.array([[1.0, 0.0]])

    validated = genome.validate(population)
    encoded = genome.encode(population[0])
    population[0, 0] = 0

    np.testing.assert_array_equal(validated, [[1.0, 0.0]])
    np.testing.assert_array_equal(encoded, [1, 0])
    assert encoded.dtype == np.int8


@pytest.mark.parametrize("size", [True, np.bool_(False), 0, -1, 2.0])
def test_genomes_require_positive_integer_dimensions(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        BinaryGenome(size)
    with pytest.raises(ValueError, match="size"):
        NumericGenome(size)


@pytest.mark.parametrize("probability", [-0.01, 1.01, np.nan, np.inf, [0.5], True])
def test_binary_probability_rejects_invalid_scalar(probability: float) -> None:
    with pytest.raises(ValueError, match="probability_of_one"):
        BinaryGenome(1, probability)


@pytest.mark.parametrize("population", [[[0, 0.7]], [[0, np.nan]], [0, 1], [[]], [[0, 1j]]])
def test_binary_validation_rejects_invalid_population(population: object) -> None:
    with pytest.raises(ValueError, match="population"):
        BinaryGenome(2).validate(population)


def test_numeric_shared_bounds_and_indexed_overrides() -> None:
    genome = NumericGenome(3, bounds=(-1, 1), custom_bounds={1: (10, 20)})

    sampled = genome.sample(60, np.random.default_rng(2))

    assert sampled.dtype == np.float64
    assert np.all((-1 <= sampled[:, [0, 2]]) & (sampled[:, [0, 2]] <= 1))
    assert np.all((10 <= sampled[:, 1]) & (sampled[:, 1] <= 20))
    np.testing.assert_array_equal(genome.validate([[-1, 20, 1]]), [[-1, 20, 1]])


def test_numeric_full_bounds_are_copied() -> None:
    bounds = np.array([[1.0, 2.0], [10.0, 20.0]])
    genome = NumericGenome(2, custom_bounds=bounds)
    bounds[:] = 100

    np.testing.assert_array_equal(genome.validate([[1.5, 15]]), [[1.5, 15]])
    assert np.all(genome.sample(20, np.random.default_rng(4)) < 20)


def test_normal_initialization_clips_individual_coordinates_and_copies_parameters() -> None:
    mean = np.array([-100.0, 100.0, 0.5])
    std = np.array([0.01, 0.01, 0.0])
    initialization = Normal(mean=mean, std=std)
    mean[:] = 0
    std[:] = 5
    genome = NumericGenome(3, initialization=initialization)

    samples = genome.sample(5, np.random.default_rng(23))

    np.testing.assert_array_equal(samples, np.tile([0.0, 1.0, 0.5], (5, 1)))


def test_normal_initialization_broadcasts_scalars() -> None:
    genome = NumericGenome(2, initialization=Normal(mean=0.5, std=0))

    np.testing.assert_array_equal(genome.sample(3, np.random.default_rng(15)), 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": (1, 1)},
        {"bounds": (np.nan, 1)},
        {"bounds": (2, 1)},
        {"custom_bounds": {2: (0, 1)}},
        {"custom_bounds": {True: (0, 1)}},
        {"custom_bounds": {-1: (0, 1)}},
        {"custom_bounds": {0: (2, 1)}},
        {"custom_bounds": [[0, 1]]},
        {"custom_bounds": [[0, 1], [2, np.inf]]},
    ],
)
def test_numeric_rejects_invalid_bounds(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="bounds"):
        NumericGenome(2, **kwargs)


@pytest.mark.parametrize("kwargs", [{"mean": np.nan}, {"std": -1}, {"std": [[1]]}, {"mean": []}])
def test_normal_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="mean|std"):
        Normal(**kwargs)


def test_numeric_rejects_wrong_length_normal_parameters() -> None:
    with pytest.raises(ValueError, match="mean"):
        NumericGenome(2, initialization=Normal(mean=[1, 2, 3]))


@pytest.mark.parametrize("population", [[[0.1, 11]], [[0.1, np.inf]], [[1]], [0.1, 0.2]])
def test_numeric_validation_rejects_invalid_population(population: object) -> None:
    with pytest.raises(ValueError, match="population"):
        NumericGenome(2).validate(population)


def test_numeric_encoding_owns_float64_result() -> None:
    candidate = np.array([0.25, 0.75])
    result = NumericGenome(2).encode(candidate)
    candidate[:] = 0

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [0.25, 0.75])


def test_numeric_mutation_uses_coordinate_width_as_scale() -> None:
    genome = NumericGenome(2, custom_bounds=[[0, 1], [0, 100]])
    parents = np.tile([0.5, 50.0], (30, 1))

    children = genome.mutate(parents, np.random.default_rng(13), probability=1, scale=0.1)

    assert np.std(children[:, 0]) > 0
    assert np.std(children[:, 1]) > 50 * np.std(children[:, 0])
    np.testing.assert_array_equal(parents, np.tile([0.5, 50.0], (30, 1)))


def test_numeric_mutation_clips_to_bounds() -> None:
    genome = NumericGenome(2, custom_bounds=[[-1, 1], [90, 100]])

    children = genome.mutate(
        np.tile([0.0, 95.0], (100, 1)),
        np.random.default_rng(145),
        probability=1,
        scale=10,
    )

    assert np.all((-1 <= children[:, 0]) & (children[:, 0] <= 1))
    assert np.all((90 <= children[:, 1]) & (children[:, 1] <= 100))
    assert {-1.0, 1.0}.issubset(set(children[:, 0]))
    assert {90.0, 100.0}.issubset(set(children[:, 1]))


@pytest.mark.parametrize("probability,scale", [(-0.1, 1.0), (1.0, np.nan), (1.0, -1.0)])
def test_mutation_rejects_invalid_settings(probability: float, scale: float) -> None:
    population = np.array([[0.0]])
    rng = np.random.default_rng(8)

    with pytest.raises(ValueError, match="probability|scale"):
        BinaryGenome(1).mutate(population, rng, probability, scale)
    with pytest.raises(ValueError, match="probability|scale"):
        NumericGenome(1).mutate(population, rng, probability, scale)


def test_numeric_sampling_and_mutation_accept_large_finite_bounds() -> None:
    limit = np.finfo(np.float64).max
    genome = NumericGenome(1, bounds=(-limit, limit))

    population = genome.sample(100, np.random.default_rng(9))
    children = genome.mutate(population, np.random.default_rng(10), probability=1, scale=0.1)

    assert np.all(np.isfinite(children))
    assert np.any(children != population)
    assert np.any(np.abs(children / limit - population / limit) < 0.25)
    assert np.any(population < 0)
    assert np.any(population > 0)


def test_numeric_validation_rejects_integer_rounded_into_bounds() -> None:
    genome = NumericGenome(1, bounds=(0, 2**63))

    with pytest.raises(ValueError, match="population"):
        genome.validate(np.array([[2**63 + 1]], dtype=np.uint64))

    np.testing.assert_array_equal(genome.validate([[2**63]]), [[float(2**63)]])


def test_numeric_validation_rejects_mixed_inputs_that_round_out_of_bounds() -> None:
    genome = NumericGenome(2, bounds=(0, 2**63))

    with pytest.raises(ValueError, match="population"):
        genome.validate([[2**63 + 1, 0.25]])


def test_numeric_encoding_rejects_integer_rounded_into_bounds() -> None:
    genome = NumericGenome(1, bounds=(0, 2**63))

    with pytest.raises(ValueError, match="candidate"):
        genome.encode(np.array([2**63 + 1], dtype=np.uint64))


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps == np.finfo(np.float64).eps,
    reason="longdouble has the same precision as float64 on this platform",
)
def test_binary_validation_and_encoding_reject_high_precision_nonbit() -> None:
    genome = BinaryGenome(1)
    nonbit = np.nextafter(np.longdouble(1), np.longdouble(2))
    population = np.array([[nonbit]], dtype=np.longdouble)

    with pytest.raises(ValueError, match="population"):
        genome.validate(population)
    with pytest.raises(ValueError, match="candidate"):
        genome.encode(population[0])


def test_numeric_mutation_preserves_small_additive_step_in_wide_range() -> None:
    genome = NumericGenome(1, bounds=(-1e16, 1e16))

    mutated = genome.mutate(np.array([[1.0]]), np.random.default_rng(1), 1, 1e-18)

    assert mutated[0, 0] == pytest.approx(1.0164323628700231)


def test_numeric_mutation_recovers_from_overflowing_step_before_clipping() -> None:
    limit = np.finfo(np.float64).max
    genome = NumericGenome(1, bounds=(-limit, limit))
    parent = np.array([[-0.75 * limit]])

    mutated = genome.mutate(parent, np.random.default_rng(1), 1, 0.8)

    assert mutated[0, 0] / limit == pytest.approx(0.5645890296018534)


def test_numeric_mutation_preserves_additive_step_when_range_width_overflows() -> None:
    limit = np.finfo(np.float64).max
    genome = NumericGenome(1, bounds=(-limit, limit))

    mutated = genome.mutate(np.array([[1.0]]), np.random.default_rng(1), 1, 1e-308)

    assert mutated[0, 0] == pytest.approx(3.9540345921007063)
