"""
Experimental models for Bundesliga prediction.

These models are experimental and being tested for potential integration
into the main prediction pipeline.
"""

from .gradient_boosting import GradientBoostingModel
from .bivariate_poisson import BivariatePoissonModel
from .smart_odds import SmartOddsModel
from .tendency_first import TendencyFirstModel
from .probability_max import ProbabilityMaxModel

__all__ = [
    "GradientBoostingModel",
    "BivariatePoissonModel",
    "SmartOddsModel",
    "TendencyFirstModel",
    "ProbabilityMaxModel",
]
