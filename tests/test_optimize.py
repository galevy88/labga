import json

import numpy as np
import pytest

from labga.evaluation import EvaluationError
from labga.genomes import BinaryGenome, NumericGenome
from labga.objective import Objective
from labga.operators import Operators
from labga.optimize import optimize
from labga.predictors import Predictor


def identity(candidate):
    return candidate


def to_predictions(candidate):
    return np.array([candidate.sum(), candidate[0] - candidate[1]])


class BatchPredictor:
    def predict(self, candidate):
        return to_predictions(candidate)

    def predict_batch(self, candidates):
        return np.stack([self.predict(candidate) for candidate in candidates])


def build_batch_predictor(*, device):
    return BatchPredictor()


def test_numeric_ga_improves_and_returns_matching_prediction():
    objective = Objective(to_predictions, [0.6, -0.2])
    result = optimize(
        genome=NumericGenome(2),
        objective=objective,
        population_size=25,
        generations=25,
        workers=1,
        seed=23,
    )

    assert result.best_loss < result.history[0].best_loss
    assert result.best_loss < 0.0001
    np.testing.assert_array_equal(result.best_prediction, to_predictions(result.best_solution))
    assert result.best_loss == np.mean((result.best_prediction - objective.target) ** 2)


def test_binary_mutation_finds_exact_solution():
    result = optimize(
        genome=BinaryGenome(3, probability_of_one=0),
        objective=Objective(identity, [1, 1, 1]),
        population_size=5,
        generations=1,
        workers=1,
        seed=12,
        operators=Operators(mutation_probability=1),
    )

    np.testing.assert_array_equal(result.best_solution, [1, 1, 1])
    assert result.best_solution.dtype == np.int8
    assert result.best_loss == 0
    assert result.metadata["evaluation_count"] == 9
    assert [record.best_so_far for record in result.history] == [1, 0]


def test_generation_zero_uses_copied_input_and_keeps_first_tied_best():
    initial = np.array([[0.75], [0.25], [0.0]], dtype=np.float64)
    before = initial.copy()
    result = optimize(
        genome=NumericGenome(1),
        objective=Objective(identity, [0.5]),
        population_size=3,
        generations=0,
        initial_population=initial,
        workers=1,
        seed=2,
    )

    assert [record.generation for record in result.history] == [0]
    assert result.history[0].migrated is False
    assert result.history[0].island_best == (0.0625,)
    assert result.metadata["evaluation_count"] == 3
    np.testing.assert_array_equal(result.best_solution, [0.75])
    np.testing.assert_array_equal(result.best_prediction, [0.75])
    np.testing.assert_array_equal(result.population.solutions, before)
    np.testing.assert_array_equal(initial, before)
    assert not np.shares_memory(result.population.solutions, initial)
    assert not np.shares_memory(result.population.solutions, result.best_solution)
    json.dumps(result.metadata)


def test_best_so_far_survives_loss_of_solution_without_elites():
    result = optimize(
        genome=BinaryGenome(1, probability_of_one=1),
        objective=Objective(identity, [1]),
        population_size=3,
        generations=1,
        workers=1,
        seed=42,
        operators=Operators(mutation_probability=1, elite_count=0),
    )

    np.testing.assert_array_equal(result.population.solutions, [[0], [0], [0]])
    np.testing.assert_array_equal(result.best_solution, [1])
    np.testing.assert_array_equal(result.best_prediction, [1])
    assert result.best_loss == 0
    assert result.history[-1].best_loss == 1
    assert result.history[-1].best_so_far == 0


def test_all_elite_population_needs_no_offspring_evaluations():
    result = optimize(
        genome=NumericGenome(1),
        objective=Objective(identity, [0.5]),
        population_size=1,
        generations=2,
        initial_population=[[0.25]],
        workers=1,
    )

    assert result.metadata["evaluation_count"] == 1
    assert [row.best_loss for row in result.history] == [0.0625, 0.0625, 0.0625]
    np.testing.assert_array_equal(result.population.solutions, [[0.25]])


def test_evaluation_error_maps_offspring_to_generation_and_island():
    class RejectMutated:
        def predict(self, candidate):
            return self.predict_batch(candidate[None, :])[0]

        def predict_batch(self, candidates):
            return np.where(candidates == 1, np.nan, candidates)

    with pytest.raises(EvaluationError) as caught:
        optimize(
            genome=BinaryGenome(1, probability_of_one=0),
            objective=Objective(RejectMutated(), [0]),
            population_size=4,
            generations=1,
            distributed=True,
            n_islands=2,
            workers=1,
            seed=18,
            operators=Operators(mutation_probability=1),
        )

    assert caught.value.failed_indices == (0, 1)
    assert "generation 1" in str(caught.value)
    assert "island 0" in str(caught.value)
    assert "island 1" in str(caught.value)


def test_omitted_seed_is_recorded_for_replay():
    options = {
        "genome": NumericGenome(2),
        "objective": Objective(identity, [0.2, 0.3]),
        "population_size": 6,
        "generations": 2,
        "workers": 1,
    }
    original = optimize(**options)
    replay = optimize(**options, seed=original.metadata["seed"])

    np.testing.assert_array_equal(replay.population.solutions, original.population.solutions)
    assert replay.history == original.history


def test_generation_mean_remains_finite_for_large_finite_losses():
    result = optimize(
        genome=NumericGenome(1, bounds=(0, 1e308)),
        objective=Objective(identity, [0], loss="mae"),
        population_size=2,
        generations=0,
        initial_population=[[1e308], [1e308]],
        workers=1,
    )

    assert result.history[0].mean_loss == 1e308


def test_seeded_results_agree_across_worker_and_batch_sizes():
    options = {
        "genome": NumericGenome(2),
        "population_size": 11,
        "generations": 4,
        "distributed": True,
        "n_islands": 3,
        "seed": 180,
    }
    serial_objective = Objective(Predictor(build_batch_predictor, batch_size=2), [0.6, -0.2])
    parallel_objective = Objective(Predictor(build_batch_predictor, batch_size=7), [0.6, -0.2])
    serial = optimize(**options, workers=1, objective=serial_objective)
    parallel = optimize(**options, workers=2, objective=parallel_objective)

    assert parallel.history == serial.history
    np.testing.assert_array_equal(parallel.best_solution, serial.best_solution)
    np.testing.assert_array_equal(parallel.best_prediction, serial.best_prediction)
    np.testing.assert_array_equal(parallel.population.solutions, serial.population.solutions)
    np.testing.assert_array_equal(parallel.population.losses, serial.population.losses)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"population_size": True}, "population_size"),
        ({"population_size": 0}, "population_size"),
        ({"generations": np.bool_(False)}, "generations"),
        ({"generations": -1}, "generations"),
        ({"generations": 1.5}, "generations"),
        ({"distributed": "yes"}, "distributed"),
        ({"seed": -1}, "seed"),
        ({"seed": False}, "seed"),
        ({"workers": True}, "workers"),
        ({"workers": 0}, "workers"),
        ({"workers": "all"}, "workers"),
        ({"evaluation_timeout": 0}, "evaluation_timeout"),
        ({"evaluation_timeout": float("nan")}, "evaluation_timeout"),
        ({"evaluation_timeout": float("inf")}, "evaluation_timeout"),
        ({"evaluation_timeout": True}, "evaluation_timeout"),
        ({"operators": Operators(elite_count=4)}, "elite_count"),
        ({"initial_population": [[0, 0], [0, 0]]}, "initial_population"),
        ({"initial_population": [[0, 0], [0, 2], [0, 0]]}, "initial_population.*index"),
    ],
)
def test_invalid_settings_are_rejected_before_predicting(override, field):
    def predictor(candidate):
        pytest.fail("Invalid configurations must never reach the predictor")

    options = {
        "genome": BinaryGenome(2),
        "objective": Objective(predictor, [0, 0]),
        "population_size": 3,
        "generations": 0,
        "workers": 1,
        **override,
    }
    with pytest.raises(ValueError, match=field):
        optimize(**options)
