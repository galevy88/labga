"""Four populations with ring migration, sharing CPU evaluation workers."""

import numpy as np

from labga import Migration, NumericGenome, Objective, optimize


def predict(parameters: np.ndarray) -> np.ndarray:
    return np.sin(parameters) + 0.1 * parameters


def main() -> None:
    result = optimize(
        genome=NumericGenome(size=6, bounds=(-8.0, 8.0)),
        objective=Objective(predict, target=np.full(6, 0.5)),
        population_size=200,
        generations=40,
        distributed=True,
        n_islands=4,
        migration=Migration(interval=5, count=2),
        workers=2,
        seed=11,
    )
    print("Best MSE:", result.best_loss)
    print("Island sizes:", result.metadata["island_sizes"])
    print("Evaluation workers:", result.metadata["workers_actual"])


if __name__ == "__main__":
    main()
