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


@pytest.mark.parametrize("loss", ["mse", "mae", lambda p, t: np.sum(abs(p - t))])
def test_bfloat16_output_and_target(loss):
    target = torch.tensor([1, 0], dtype=torch.bfloat16)
    prediction = torch.tensor([[1, 1]], dtype=torch.bfloat16)
    result = Objective(lambda row: prediction[0], target, loss=loss).score(prediction, count=1)
    np.testing.assert_equal(result.predictions, [[1, 1]])
    np.testing.assert_equal(result.losses, [0.5 if isinstance(loss, str) else 1.0])


def test_rejects_complex_torch_before_loss_or_conversion():
    pred = torch.tensor([[3 + 14j]])
    with pytest.raises(ValueError, match="prediction.*real"):
        Objective(lambda x: x, [3]).score(pred, count=1)


@pytest.mark.parametrize("workers", [1, 2])
def test_session_rejects_preloaded_cuda_adapter_without_initializing_device(workers):
    from labga.evaluation import EvaluationSession
    from labga.genomes import NumericGenome

    # Identity has no parameters or buffers, so this marks its intended device
    # without allocating a CUDA tensor. The guard must run before first inference.
    adapter = TorchPredictor(torch.nn.Identity(), device="cuda:0")
    with (
        pytest.raises(ValueError, match="Predictor.*factory"),
        EvaluationSession(Objective(adapter, [0]), workers=workers, genome=NumericGenome(1)),
    ):
        pass
