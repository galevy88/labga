"""Binary and bounded continuous search spaces."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from labga._types import FloatArray


def _checked_integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


def _checked_real(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _checked_probability(value: float, name: str) -> float:
    number = _checked_real(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")
    return number


def _checked_nonnegative(value: float, name: str) -> float:
    number = _checked_real(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _as_real_array(value: ArrayLike, name: str, *, exact: bool = False) -> FloatArray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", under="ignore"):
        converted = array.astype(np.float64, copy=False)
    if exact:
        _require_exact_float64(value, array, converted, name)
    return converted


def _require_exact_float64(
    value: ArrayLike, array: NDArray[np.generic], converted: FloatArray, name: str
) -> None:
    constraint = "be exactly representable as float64"
    if array.dtype.kind in "iu" and array.dtype.itemsize >= 8:
        # Check the cast limit explicitly because an out-of-range round trip is undefined.
        value_bits = 8 * array.dtype.itemsize - (array.dtype.kind == "i")
        upper_limit = float(1 << value_bits)
        with np.errstate(invalid="ignore"):
            valid = (converted < upper_limit) & (converted.astype(array.dtype) == array)
        _require_all(valid, name, constraint)
    elif array.dtype.kind == "f" and array.dtype.itemsize > 8:
        _require_all(~np.isfinite(array) | (converted == array), name, constraint)

    if array.dtype.kind == "f" and not isinstance(value, np.ndarray):
        # Mixed sequences can lose integer precision before astype through dtype inference.
        originals = np.asarray(value, dtype=object).flat
        for index, (original, rounded) in enumerate(zip(originals, converted.flat, strict=True)):
            if isinstance(original, Integral) and int(original) != float(rounded):
                coordinates = np.unravel_index(index, converted.shape)
                location = tuple(int(coordinate) for coordinate in coordinates)
                raise ValueError(
                    f"{name} values must {constraint}. Invalid value at index {location}"
                )


def _population_array(population: ArrayLike, size: int) -> FloatArray:
    array = _as_real_array(population, "population", exact=True)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != size:
        raise ValueError(f"population must have shape (count, {size}) with count at least 1")
    return array


def _candidate_array(candidate: FloatArray, size: int) -> FloatArray:
    array = _as_real_array(candidate, "candidate", exact=True)
    if array.shape != (size,):
        raise ValueError(f"candidate must have shape ({size},)")
    return array


def _require_all(valid: NDArray[np.bool_], name: str, constraint: str) -> None:
    if not np.all(valid):
        first_invalid = np.unravel_index(np.argmin(valid), valid.shape)
        index = tuple(int(coordinate) for coordinate in first_invalid)
        raise ValueError(f"{name} values must {constraint}. Invalid value at index {index}")


def _normal_parameter(value: ArrayLike, name: str) -> float | tuple[float, ...]:
    array = _as_real_array(value, name)
    if array.ndim > 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or a nonempty 1D vector")
    if name == "std" and np.any(array < 0):
        raise ValueError("std must be nonnegative")
    return float(array) if array.ndim == 0 else tuple(float(number) for number in array)


@dataclass(frozen=True, slots=True, init=False)
class Normal:
    """Parameters for a numeric genome's clipped normal initialization.

    Args:
        mean: Scalar or one mean per coordinate.
        std: Scalar or one nonnegative standard deviation per coordinate.
            Zero makes the corresponding coordinate deterministic.
    """

    mean: float | tuple[float, ...]
    std: float | tuple[float, ...]

    def __init__(self, mean: ArrayLike = 0.5, std: ArrayLike = 0.1) -> None:
        object.__setattr__(self, "mean", _normal_parameter(mean, "mean"))
        object.__setattr__(self, "std", _normal_parameter(std, "std"))


@dataclass(frozen=True, slots=True)
class BinaryGenome:
    """A fixed-length vector of independent binary coordinates.

    Args:
        size: Positive number of coordinates in each candidate.
        probability_of_one: Per-coordinate probability of sampling a one in
            the initial population.
    """

    size: int
    probability_of_one: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _checked_integer(self.size, "size", 1))
        probability = _checked_probability(self.probability_of_one, "probability_of_one")
        object.__setattr__(self, "probability_of_one", probability)

    def sample(self, count: int, rng: Generator) -> FloatArray:
        """Generate a population of binary vectors.

        Args:
            count: Positive number of candidates to sample.
            rng: Random generator used for initialization.

        Returns:
            An owned float64 matrix of shape `(count, size)`.
        """
        count = _checked_integer(count, "count", 1)
        return (rng.random((count, self.size)) < self.probability_of_one).astype(np.float64)

    def validate(self, population: ArrayLike) -> FloatArray:
        """Check a nonempty candidate matrix for exact zero and one values.

        Args:
            population: Candidate matrix of shape `(count, size)`.

        Returns:
            An owned float64 matrix with the validated candidates.
        """
        array = _population_array(population, self.size)
        _require_all((array == 0) | (array == 1), "population", "be either 0 or 1")
        return array.copy()

    def mutate(
        self, population: FloatArray, rng: Generator, probability: float, scale: float
    ) -> FloatArray:
        """Flip each bit independently with the given probability.

        Args:
            population: Binary candidate matrix.
            rng: Random generator used to select bits to flip.
            probability: Per-coordinate mutation probability.
            scale: Nonnegative mutation magnitude used for numeric genomes.

        Returns:
            A new float64 matrix of binary candidates.
        """
        probability = _checked_probability(probability, "probability")
        _checked_nonnegative(scale, "scale")
        offspring = self.validate(population)
        selected = rng.random(offspring.shape) < probability
        offspring[selected] = 1 - offspring[selected]
        return offspring

    def encode(self, candidate: FloatArray) -> NDArray[np.int8]:
        """Copy a binary candidate into the predictor's int8 representation.

        Args:
            candidate: Single vector of shape `(size,)`.

        Returns:
            An owned int8 vector of zeros and ones.
        """
        array = _candidate_array(candidate, self.size)
        _require_all((array == 0) | (array == 1), "candidate", "be either 0 or 1")
        return array.astype(np.int8)


def _check_bounds(bounds: FloatArray, name: str) -> None:
    if not np.isfinite(bounds).all() or np.any(bounds[..., 0] >= bounds[..., 1]):
        raise ValueError(f"{name} must contain finite (lower, upper) pairs with lower < upper")


def _bounds_pair(values: ArrayLike, name: str) -> FloatArray:
    pair = _as_real_array(values, name, exact=True)
    if pair.shape != (2,):
        raise ValueError(f"{name} must be a (lower, upper) pair")
    _check_bounds(pair, name)
    return pair


def _coordinate_bounds(
    size: int, bounds: ArrayLike, custom_bounds: Mapping[int, ArrayLike] | ArrayLike | None
) -> FloatArray:
    default = _bounds_pair(bounds, "bounds")
    if custom_bounds is None:
        return np.tile(default, (size, 1))
    if isinstance(custom_bounds, Mapping):
        resolved = np.tile(default, (size, 1))
        for index, values in custom_bounds.items():
            coordinate = _checked_integer(index, "custom_bounds index", 0)
            if coordinate >= size:
                raise ValueError(f"custom_bounds index {coordinate} must be less than {size}")
            resolved[coordinate] = _bounds_pair(values, f"custom_bounds[{coordinate}]")
        return resolved

    resolved = _as_real_array(custom_bounds, "custom_bounds", exact=True)
    if resolved.shape != (size, 2):
        raise ValueError(f"custom_bounds must have shape ({size}, 2)")
    _check_bounds(resolved, "custom_bounds")
    return resolved.copy()


def _interpolate(lower: FloatArray, upper: FloatArray, fraction: FloatArray) -> FloatArray:
    # Convex interpolation keeps the arithmetic valid even if upper - lower overflows.
    with np.errstate(over="ignore"):
        return np.clip((1 - fraction) * lower + fraction * upper, lower, upper)


def _extreme_normal_step(
    candidates: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    width: FloatArray,
    draws: FloatArray,
    scale: float,
) -> FloatArray:
    # Combine mantissas and exponents before multiplying extreme magnitudes.
    wide = ~np.isfinite(width)
    width_fraction, width_exponent = np.frexp(np.where(wide, upper / 2 - lower / 2, width))
    draw_fraction, draw_exponent = np.frexp(draws)
    scale_fraction, scale_exponent = np.frexp(scale)
    fraction = width_fraction * draw_fraction * scale_fraction
    exponent = width_exponent + draw_exponent + scale_exponent + wide
    steps = np.ldexp(fraction, exponent)
    mutated: FloatArray = candidates + steps
    overflow = ~np.isfinite(steps)
    # An overflowing step and an opposite-sign candidate can have a finite sum.
    half_steps = np.ldexp(fraction[overflow], exponent[overflow] - 1)
    mutated[overflow] = 2 * (candidates[overflow] / 2 + half_steps)
    return mutated


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NumericGenome:
    """A continuous vector constrained by finite per-coordinate bounds.

    Args:
        size: Positive number of coordinates in each candidate.
        bounds: Default (lower, upper) pair shared by all coordinates.
        custom_bounds: Zero-based mapping of coordinate overrides or a
            complete array of shape `(size, 2)`.
        initialization: Clipped normal parameters. The default is uniform
            sampling within each coordinate's range.
    """

    size: int
    initialization: Normal | None
    _bounds: FloatArray = field(repr=False)

    def __init__(
        self,
        size: int,
        bounds: ArrayLike = (0, 1),
        custom_bounds: Mapping[int, ArrayLike] | ArrayLike | None = None,
        initialization: Normal | None = None,
    ) -> None:
        size = _checked_integer(size, "size", 1)
        resolved = _coordinate_bounds(size, bounds, custom_bounds)
        if initialization is not None:
            if not isinstance(initialization, Normal):
                raise ValueError("initialization must be Normal or None")
            for name, parameter in (("mean", initialization.mean), ("std", initialization.std)):
                if isinstance(parameter, tuple) and len(parameter) != size:
                    raise ValueError(f"initialization {name} must be scalar or have length {size}")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "initialization", initialization)
        object.__setattr__(self, "_bounds", resolved)

    def sample(self, count: int, rng: Generator) -> FloatArray:
        """Generate a population within the coordinate bounds.

        Args:
            count: Positive number of candidates to sample.
            rng: Random generator used for initialization.

        Returns:
            An owned float64 matrix of shape `(count, size)`.
        """
        count = _checked_integer(count, "count", 1)
        lower, upper = self._bounds.T
        shape = (count, self.size)
        if self.initialization is None:
            return _interpolate(lower, upper, rng.random(shape))
        with np.errstate(over="ignore"):
            samples = rng.normal(self.initialization.mean, self.initialization.std, size=shape)
        return np.clip(samples, lower, upper)

    def validate(self, population: ArrayLike) -> FloatArray:
        """Check a nonempty candidate matrix against all coordinate bounds.

        Args:
            population: Candidate matrix of shape `(count, size)`.

        Returns:
            An owned float64 matrix with the validated candidates.

        Raises:
            ValueError: The shape or bounds are invalid, or values cannot
                be represented exactly as float64.
        """
        array = _population_array(population, self.size)
        lower, upper = self._bounds.T
        _require_all((array >= lower) & (array <= upper), "population", "be within bounds")
        return array.copy()

    def mutate(
        self, population: FloatArray, rng: Generator, probability: float, scale: float
    ) -> FloatArray:
        """Apply independent normal perturbations and clip to the bounds.

        Args:
            population: Valid numeric candidate matrix.
            rng: Random generator used for mutation.
            probability: Per-coordinate mutation probability.
            scale: Perturbation standard deviation as a fraction of each
                coordinate's range.

        Returns:
            A new float64 matrix of bounded candidates.
        """
        probability = _checked_probability(probability, "probability")
        scale = _checked_nonnegative(scale, "scale")
        offspring = self.validate(population)
        if probability == 0 or scale == 0:
            return offspring

        rows, columns = np.nonzero(rng.random(offspring.shape) < probability)
        lower, upper = self._bounds[columns].T
        selected = offspring[rows, columns]
        draws = rng.standard_normal(len(rows))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            width = upper - lower
            deviation = width * scale
            steps = deviation * draws
            mutated = selected + steps
            extreme = ~np.isfinite(steps) | (deviation < np.finfo(np.float64).tiny)
            if np.any(extreme):
                mutated[extreme] = _extreme_normal_step(
                    selected[extreme],
                    lower[extreme],
                    upper[extreme],
                    width[extreme],
                    draws[extreme],
                    scale,
                )
        offspring[rows, columns] = np.clip(mutated, lower, upper)
        return offspring

    def encode(self, candidate: FloatArray) -> FloatArray:
        """Copy a numeric candidate into the predictor's float64 representation.

        Args:
            candidate: Single vector of shape `(size,)`.

        Returns:
            An owned float64 vector within the coordinate bounds.

        Raises:
            ValueError: The shape or bounds are invalid, or values cannot
                be represented exactly as float64.
        """
        array = _candidate_array(candidate, self.size)
        lower, upper = self._bounds.T
        _require_all((array >= lower) & (array <= upper), "candidate", "be within bounds")
        return array.copy()
