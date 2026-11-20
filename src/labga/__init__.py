"""Genetic algorithms for fixed-predictor optimization."""

from labga.evaluation import EvaluationError
from labga.genomes import BinaryGenome, Normal, NumericGenome
from labga.islands import Migration
from labga.objective import Objective
from labga.operators import Operators
from labga.optimize import optimize
from labga.predictors import Predictor, TorchPredictor
from labga.results import GenerationRecord, OptimizationResult, Population

__version__ = "0.1.0"

__all__ = [
    "BinaryGenome",
    "EvaluationError",
    "GenerationRecord",
    "Migration",
    "Normal",
    "NumericGenome",
    "Objective",
    "Operators",
    "OptimizationResult",
    "Population",
    "Predictor",
    "TorchPredictor",
    "optimize",
]
