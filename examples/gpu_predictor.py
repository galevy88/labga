"""Synthetic PyTorch inference example; this is not a DKO implementation.

Run with --device cpu for a CPU smoke test or --device cuda:0 on a CUDA host.
"""

import argparse

import numpy as np

from labga import BinaryGenome, Objective, Predictor, optimize


def build_model(*, device: str):
    import torch

    torch.manual_seed(21)
    return torch.nn.Sequential(torch.nn.Linear(24, 4), torch.nn.Sigmoid())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    objective = Objective(
        predictor=Predictor(factory=build_model, device=args.device),
        target=np.array([0.2, 0.7, 0.4, 0.8]),
    )
    result = optimize(
        genome=BinaryGenome(size=24),
        objective=objective,
        population_size=128,
        generations=20,
        distributed=True,
        n_islands=4,
        workers=1,
        seed=42,
    )
    print("Best prediction:", result.best_prediction)
    print("Best MSE:", result.best_loss)
    print("Execution:", result.metadata)


if __name__ == "__main__":
    main()
