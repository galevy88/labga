import numpy as np
import pytest

from labga.objective import Objective
from labga.predictors import Predictor, TorchPredictor

torch = pytest.importorskip("torch")


def build_linear(*, device):
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    return model


def test_torch_adapter_uses_eval_and_inference_mode():
    model = build_linear(device="cpu")
    model.train()
    adapter = TorchPredictor(model)
    predictions = adapter.predict_batch([[1, 2], [3, 4]])
    assert not model.training
    assert not predictions.requires_grad
    np.testing.assert_allclose(predictions.numpy(), [[1, 2], [3, 4]])
    np.testing.assert_allclose(adapter.predict([1, 2]).numpy(), [1, 2])


def test_factory_wraps_torch_module_and_preserves_batch_loss():
    model = Predictor(build_linear).build()
    objective = Objective(model, [1, 2])
    result = objective.evaluate(np.array([[1, 2], [3, 4]], dtype=float))
    np.testing.assert_allclose(result.losses, [0, 4])


def test_torch_adapter_rejects_nonnumeric_output_shape():
    model = TorchPredictor(torch.nn.Linear(2, 1))
    with pytest.raises(ValueError, match="shape"):
        Objective(model, [1, 2]).evaluate(np.zeros((3, 2)))


def test_torch_adapter_custom_loss_receives_numpy():
    def loss(prediction, target):
        assert isinstance(prediction, np.ndarray)
        return np.mean(np.abs(prediction - target))

    model = Predictor(build_linear).build()
    result = Objective(model, [1, 2], loss=loss).evaluate(np.array([[3, 4]], dtype=float))
    assert result.losses[0] == 2
