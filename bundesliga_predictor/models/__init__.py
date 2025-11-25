"""
Prediction models for Bundesliga match forecasting.
"""

from .base import BaseModel
from .multi_output import MultiOutputRegressionModel
from .classification import MultiClassClassificationModel
from .poisson import PoissonRegressionModel
from .naive_odds import NaiveOddsModel

__all__ = [
    "BaseModel",
    "MultiOutputRegressionModel",
    "MultiClassClassificationModel",
    "PoissonRegressionModel",
    "NaiveOddsModel",
]
