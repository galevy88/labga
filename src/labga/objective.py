"""A fixed prediction target and its per-candidate evaluation rule."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
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


def _real_array(value: object, name: str) -> NDArray[Any]:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        if value.is_complex():
            raise ValueError(f"{name} must contain real numeric values")
        tensor = value.detach().cpu()
        # NumPy has no bfloat16 or float8 dtype. Both convert exactly to float32.
        if tensor.dtype == torch.bfloat16 or (
            tensor.is_floating_point() and tensor.element_size() == 1
        ):
            tensor = tensor.float()
        value = tensor.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    return array


def as_array(value: object, *, name: str = "prediction") -> FloatArray:
    """Convert real NumPy-compatible data or a detached tensor to float64.

    Args:
        value: Numeric scalar or array.
        name: Field name used in validation errors.

    Returns:
        An array with the input shape and float64 dtype.
    """
    with np.errstate(over="ignore"):
        return _real_array(value, name).astype(np.float64, copy=False)


def _overflow_loss(row: FloatArray, target: FloatArray, loss: str) -> float:
    # Rare fallback when subtraction, squaring, or the sum overflows float64.
    # Exact binary fractions avoid intermediate overflow even on systems where
    # longdouble has the same range as float64.
    total = Fraction()
    for value, desired in zip(row, target, strict=True):
        delta = Fraction.from_float(float(value)) - Fraction.from_float(float(desired))
        total += delta * delta if loss == "mse" else abs(delta)
    mean = total / len(target)
    if mean > sys.float_info.max:
        raise ValueError("loss must contain only finite scalars")
    return float(mean)


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
        target = as_array(self.target, name="target").copy()
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
        candidates = _real_array(candidates, "candidates")
        if candidates.ndim != 2 or not len(candidates) or not np.isfinite(candidates).all():
            raise ValueError("candidates must be a finite nonempty 2D array")
        model = self.predictor if predictor is None else predictor
        if isinstance(model, Predictor):
            raise TypeError("Use optimize/EvaluationSession to manage a Predictor factory")
        inputs = candidates.copy()
        if self.preprocess is not None:
            inputs = np.asarray(
                [_real_array(self.preprocess(row.copy()), "preprocess") for row in inputs]
            )
            if inputs.ndim != 2 or not inputs.shape[1] or not np.isfinite(inputs).all():
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
        values_np = as_array(predictions)
        if values_np.shape != expected:
            raise ValueError(f"prediction shape must be {expected}; got {values_np.shape}")
        if not np.isfinite(values_np).all():
            raise ValueError("prediction must contain only finite values")
        if isinstance(self.loss, str):
            torch = sys.modules.get("torch")
            if torch is not None and isinstance(predictions, torch.Tensor):
                with torch.inference_mode():
                    values = predictions.to(dtype=torch.float64)
                    desired = torch.tensor(target, dtype=values.dtype, device=values.device)
                    residual = values - desired
                    terms = residual.square() if self.loss == "mse" else residual.abs()
                    losses = as_array(terms.mean(dim=1), name="loss")
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_np = values_np - target
                    terms_np = residual_np**2 if self.loss == "mse" else np.abs(residual_np)
                    losses = np.mean(terms_np, axis=1)
            for index in np.flatnonzero(~np.isfinite(losses)):
                losses[index] = _overflow_loss(values_np[index], target, self.loss)
        else:
            computed = []
            for row in values_np:
                value = _real_array(self.loss(row.copy(), target.copy()), "loss")
                if value.ndim != 0:
                    raise ValueError("custom loss must return one scalar per candidate")
                computed.append(float(value))
            losses = np.asarray(computed, dtype=np.float64)
        if not np.isfinite(losses).all():
            raise ValueError("loss must contain only finite scalars")
        return EvaluationBatch(values_np.copy(), losses.copy())
