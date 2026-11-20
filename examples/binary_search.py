"""Find a binary input matching a desired output; no biological model required."""

import numpy as np

from labga import BinaryGenome, Objective, optimize


def predict(genome: np.ndarray) -> np.ndarray:
    """Simple transparent predictor for the getting-started example."""
    return genome.astype(float)


def main() -> None:
    target = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=float)
    result = optimize(
        genome=BinaryGenome(size=len(target), probability_of_one=0.3),
        objective=Objective(predictor=predict, target=target, loss="mse"),
        population_size=80,
        generations=30,
        workers=1,
        seed=42,
    )
    print("Best input:", result.best_solution)
    print("Best MSE:", result.best_loss)


if __name__ == "__main__":
    main()
