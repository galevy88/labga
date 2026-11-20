"""Predictor construction and optional PyTorch inference adapters."""

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Predictor:
    """Configure a model built once inside each evaluation owner.

    Args:
        factory: Importable callable accepting the keyword ``device``.
        device: ``cpu``, ``cuda``, or a numbered CUDA device.
        batch_size: Automatic conservative batching or an explicit positive size.
    """

    factory: Callable[..., object]
    device: str = "cpu"
    batch_size: int | Literal["auto"] = "auto"

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("factory must be callable")
        if not isinstance(self.device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:N'")
        if self.batch_size != "auto" and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be 'auto' or a positive integer")

    def build(self) -> object:
        """Build the predictor in its owner process, with CUDA validation."""
        if self.device.startswith("cuda"):
            try:
                import torch
            except ImportError as error:
                raise ImportError("CUDA evaluation requires the labga[torch] extra") from error
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but no CUDA device is available")
        model = self.factory(device=self.device)
        torch_module = sys.modules.get("torch")
        if torch_module is not None and isinstance(model, torch_module.nn.Module):
            return TorchPredictor(model, self.device)
        if not callable(model) and not callable(getattr(model, "predict", None)):
            raise TypeError("factory must return a callable or an object with predict(x)")
        return model


class TorchPredictor:
    """Adapt a PyTorch module to independent single/batch inference.

    Args:
        model: A PyTorch ``nn.Module`` returning a numeric tensor.
        device: Device on which to keep the model and perform inference.
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:
        import torch

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model: Any = model.to(device).eval()
        self.device: str = device
        parameter = next(self.model.parameters(), None)
        self.dtype: Any = (
            parameter.dtype
            if parameter is not None and parameter.is_floating_point()
            else torch.float32
        )

    def predict_batch(self, inputs: ArrayLike) -> Any:
        """Return a tensor with one prediction row per input row."""
        import torch

        with torch.inference_mode():
            tensor = torch.as_tensor(np.asarray(inputs), dtype=self.dtype, device=self.device)
            return self.model(tensor)

    def predict(self, inputs: ArrayLike) -> Any:
        """Predict one candidate using a singleton batch."""
        return self.predict_batch(np.asarray(inputs)[None, :])[0]
