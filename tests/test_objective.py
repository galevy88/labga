import numpy as np
import pytest

from labga.objective import Objective
from labga.predictors import Predictor


class Doubler:
    def predict(self, x):
        return 2 * x

    def predict_batch(self, x):
        return 2 * x


def test_mse_keeps_one_loss_per_candidate():
    objective = Objective(Doubler(), target=[1, 1])
    result = objective.evaluate(np.array([[0.5, 0.5], [1, 0]], dtype=float))
    np.testing.assert_allclose(result.predictions, [[1, 1], [2, 0]])
    np.testing.assert_allclose(result.losses, [0, 1])


def test_scalar_callable_and_mae():
    objective = Objective(lambda x: x, target=[0, 2], loss="mae")
    result = objective.evaluate(np.array([[2, 2]], dtype=float))
    assert result.losses.tolist() == [1]


def test_custom_loss_and_preprocess():
    objective = Objective(
        lambda x: x,
        target=[2, 4],
        loss=lambda p, t: np.max(np.abs(p - t)),
        preprocess=lambda x: x * 2,
    )
    assert objective.evaluate(np.array([[1, 2]], dtype=float)).losses.tolist() == [0]


def test_target_is_copied_and_read_only():
    target = np.array([0.0, 1.0])
    objective = Objective(lambda x: x, target)
    target[:] = 5
    np.testing.assert_array_equal(objective.target, [0, 1])
    with pytest.raises(ValueError):
        objective.target[0] = 9


@pytest.mark.parametrize("target", [[], [[1, 2]], [np.nan], [np.inf]])
def test_rejects_invalid_target(target):
    with pytest.raises(ValueError, match="target"):
        Objective(lambda x: x, target)


def test_rejects_unknown_loss():
    with pytest.raises(ValueError, match="loss"):
        Objective(lambda x: x, [1], loss="wrong")


def test_rejects_invalid_predictor():
    with pytest.raises(TypeError, match="predictor"):
        Objective(5, [1])


def test_rejects_prediction_shape():
    with pytest.raises(ValueError, match="shape"):
        Objective(lambda x: [1], [1, 2]).evaluate(np.array([[1, 2]], dtype=float))


def test_rejects_nonfinite_prediction():
    with pytest.raises(ValueError, match="finite"):
        Objective(lambda x: [np.nan], [1]).evaluate(np.array([[1]], dtype=float))


@pytest.mark.parametrize("loss", [lambda p, t: np.nan, lambda p, t: [1, 2]])
def test_rejects_nonscalar_or_nonfinite_custom_loss(loss):
    with pytest.raises(ValueError, match="loss"):
        Objective(lambda x: x, [1], loss=loss).evaluate(np.array([[1]], dtype=float))


@pytest.mark.parametrize("size", [0, -1, True, 1.5, "wrong"])
def test_predictor_rejects_invalid_batch_size(size):
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        Predictor(factory=lambda **kwargs: Doubler(), batch_size=size)


def test_predictor_defaults_to_auto_batching():
    config = Predictor(factory=lambda **kwargs: Doubler())
    assert config.batch_size == "auto"
    assert config.device == "cpu"


@pytest.mark.parametrize("device", ["mps", "cuda:-1", "", 3])
def test_predictor_rejects_unsupported_device(device):
    with pytest.raises((TypeError, ValueError), match="device"):
        Predictor(factory=lambda **kwargs: Doubler(), device=device)
