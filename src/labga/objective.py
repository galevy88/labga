"""A fixed prediction target and its per-candidate evaluation rule."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from labga._types import FloatArray
from labga.predictors import Predictor

Loss = str | Callable[[FloatArray, FloatArray], float]
Preprocess = Callable[[NDArray[Any]], ArrayLike]


@dataclass(frozen=True)
class EvaluationBatch:
    """Predictions and scalar losses in exactly the input candidate order."""

    predictions: FloatArray
    losses: FloatArray


def as_array(value: object) -> FloatArray:
    """Convert NumPy-compatible data or a detached torch tensor to float64."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class Objective:
    """Encapsulate a fixed predictor, desired output and loss.

    Args:
        predictor: Callable, object with ``predict``, or a factory configuration.
        target: Desired finite, nonempty one-dimensional prediction vector.
        loss: ``mse``, ``mae``, or a callable returning one finite scalar.
        preprocess: Optional CPU transform applied independently to candidates.
    """

    predictor: object
    target: ArrayLike
    loss: Loss = "mse"
    preprocess: Preprocess | None = None

    def __post_init__(self) -> None:
        target = np.array(self.target, dtype=np.float64, copy=True)
        if target.ndim != 1 or not target.size or not np.isfinite(target).all():
            raise ValueError("target must be a finite, nonempty 1D vector")
        target.setflags(write=False)
        object.__setattr__(self, "target", target)
        if isinstance(self.loss, str):
            if self.loss not in {"mse", "mae"}:
                raise ValueError("loss must be 'mse', 'mae', or callable")
        elif not callable(self.loss):
            raise TypeError("loss must be 'mse', 'mae', or callable")
        if self.preprocess is not None and not callable(self.preprocess):
            raise TypeError("preprocess must be callable")
        if not isinstance(self.predictor, Predictor):
            if not callable(self.predictor) and not callable(
                getattr(self.predictor, "predict", None)
            ):
                raise TypeError("predictor must be callable or expose predict(x)")

    def evaluate(
        self, candidates: NDArray[Any], *, predictor: object | None = None
    ) -> EvaluationBatch:
        """Evaluate candidate rows, using an already-built model when provided."""
        if candidates.ndim != 2 or not len(candidates) or not np.isfinite(candidates).all():
            raise ValueError("candidates must be a finite nonempty 2D array")
        model = self.predictor if predictor is None else predictor
        if isinstance(model, Predictor):
            raise TypeError("Use optimize/EvaluationSession to manage a Predictor factory")
        inputs = candidates.copy()
        if self.preprocess is not None:
            inputs = np.asarray([self.preprocess(row.copy()) for row in inputs])
            if inputs.ndim != 2 or not np.isfinite(inputs).all():
                raise ValueError("preprocess must return a fixed-length finite numeric vector")
        batch_method = getattr(model, "predict_batch", None)
        if callable(batch_method):
            raw = batch_method(inputs)
        else:
            single = getattr(model, "predict", model)
            call = cast(Callable[[NDArray[Any]], object], single)
            raw = np.stack([as_array(call(row.copy())) for row in inputs])
        return self.score(raw, count=len(inputs))

    def score(self, predictions: object, *, count: int) -> EvaluationBatch:
        """Validate output shape and reduce each prediction row independently."""
        target = np.asarray(self.target)
        expected = (count, len(target))
        torch = sys.modules.get("torch")
        tensor_losses = None
        if torch is not None and isinstance(predictions, torch.Tensor):
            if tuple(predictions.shape) != expected:
                raise ValueError(
                    f"prediction shape must be {expected}; got {tuple(predictions.shape)}"
                )
            if isinstance(self.loss, str):
                with torch.inference_mode():
                    values = predictions.to(dtype=torch.float64)
                    desired = torch.tensor(target.copy(), dtype=values.dtype, device=values.device)
                    residual = values - desired
                    tensor_losses = (
                        residual.square().mean(dim=1)
                        if self.loss == "mse"
                        else residual.abs().mean(dim=1)
                    )
        values_np = as_array(predictions)
        if values_np.shape != expected:
            raise ValueError(f"prediction shape must be {expected}; got {values_np.shape}")
        if not np.isfinite(values_np).all():
            raise ValueError("prediction must contain only finite values")
        if tensor_losses is not None:
            losses = as_array(tensor_losses)
        elif isinstance(self.loss, str):
            residual_np = values_np - target
            with np.errstate(over="ignore", invalid="ignore"):
                losses = np.mean(
                    residual_np**2 if self.loss == "mse" else np.abs(residual_np), axis=1
                )
        else:
            computed = []
            for row in values_np:
                value = np.asarray(self.loss(row.copy(), target.copy()))
                if value.ndim != 0:
                    raise ValueError("custom loss must return one scalar per candidate")
                computed.append(float(value))
            losses = np.asarray(computed, dtype=np.float64)
        if not np.isfinite(losses).all():
            raise ValueError("loss must contain only finite scalars")
        return EvaluationBatch(values_np.copy(), losses.copy())
