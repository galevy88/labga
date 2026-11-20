"""Fit amplitude and decay rate to a small synthetic measured time series."""

import numpy as np

from labga import NumericGenome, Objective, optimize

TIMES = np.linspace(0, 5, 20)


def predict(parameters: np.ndarray) -> np.ndarray:
    amplitude, decay_rate = parameters
    return amplitude * np.exp(-decay_rate * TIMES)


def main() -> None:
    measured = predict(np.array([1.8, 0.65]))
    result = optimize(
        genome=NumericGenome(size=2, bounds=(0.01, 3.0)),
        objective=Objective(predict, target=measured),
        population_size=100,
        generations=60,
        workers=1,
        seed=7,
    )
    print("Amplitude and decay rate:", result.best_solution)
    print("Initial / final MSE:", result.history[0].best_loss, result.best_loss)


if __name__ == "__main__":
    main()
